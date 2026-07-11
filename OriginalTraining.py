#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import math
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from LossCombiner import LossCombiner
import Edges
import Saliency
from TrainingReport import log_metrics_to_excel, collect_optimization_budget
import random, numpy as np, torch, os
import torch
import torch.nn.functional as F
from pytorch_msssim import  ms_ssim
import lpips


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

# Initialize once (outside function, global)
lpips_fn = lpips.LPIPS(net='vgg').cuda()  # perceptual similarity


def _sample_gaussian_image_metrics(viewpoint_cam, gaussians, visibility_filter, importance_map, error_map):
    """Sample image-space SEGS importance/error at each visible Gaussian projection."""
    visible_ids = visibility_filter.reshape(-1)
    if visible_ids.numel() == 0:
        return None, None

    points = gaussians.get_xyz[visible_ids]
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    hom_points = torch.cat((points, ones), dim=1)
    projected = hom_points @ viewpoint_cam.full_proj_transform
    ndc = projected[:, :2] / projected[:, 3:4].clamp_min(1e-7)

    # grid_sample expects normalized coordinates in [-1, 1], matching projected NDC.
    grid = ndc.view(1, -1, 1, 2)
    importance = F.grid_sample(importance_map, grid, align_corners=True, padding_mode="border").view(-1)
    error = F.grid_sample(error_map, grid, align_corners=True, padding_mode="border").view(-1)
    return importance, error


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def curriculum_lambda_weights(iteration, total_iterations, alpha_max, beta_max, args):
    """Return scheduled edge/saliency weights for the adaptive SEGS curriculum.

    The schedule keeps early optimization close to standard photometric 3DGS,
    ramps edge guidance once coarse geometry starts forming, introduces saliency
    later with a sigmoid ramp, and optionally decays both terms near the end to
    stabilize global image quality.
    """
    if not getattr(args, "adaptive_curriculum", False):
        return alpha_max, beta_max

    total_iterations = max(int(total_iterations), 1)
    t = max(float(iteration), 0.0)

    edge_delay = float(args.curriculum_edge_delay) * total_iterations
    edge_tau = max(float(args.curriculum_edge_tau) * total_iterations, 1.0)
    edge_t = max(t - edge_delay, 0.0)
    alpha = alpha_max * (1.0 - math.exp(-edge_t / edge_tau))

    saliency_midpoint = float(args.curriculum_saliency_start) * total_iterations
    saliency_width = max(float(args.curriculum_saliency_width) * total_iterations, 1.0)
    beta = beta_max * _sigmoid((t - saliency_midpoint) / saliency_width)

    decay_start = float(args.curriculum_final_decay_start) * total_iterations
    if t > decay_start and decay_start < total_iterations:
        progress = (t - decay_start) / max(total_iterations - decay_start, 1.0)
        floor = float(args.curriculum_final_weight_floor)
        decay = 1.0 - (1.0 - floor) * min(max(progress, 0.0), 1.0)
        alpha *= decay
        beta *= decay

    return alpha, beta

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, edge_name, saliency_name, args):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    cls = Edges.get_edge_processor(edge_name)
    saliency_cls = Saliency.get_saliency_processor(saliency_name)
    combiner = LossCombiner(
        cls,
        saliency_cls,
        use_edge=args.use_edge,
        use_saliency=args.use_saliency,
        lambda_edge=args.lambda_edge,
        lambda_saliency=args.lambda_saliency,
        normalize=True,
        constant_scaling_control=args.weighting_control,
    )


    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, shuffle= False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    train_cameras = scene.getTrainCameras().copy()
    validation_cameras = []
    if getattr(args, "enable_early_stopping", False):
        if not 0.0 < args.validation_fraction < 1.0:
            raise ValueError("--validation_fraction must be between 0 and 1 for early stopping.")
        validation_count = max(1, int(len(train_cameras) * args.validation_fraction)) if len(train_cameras) > 1 else 0
        validation_cameras = train_cameras[-validation_count:] if validation_count else []
        train_cameras = train_cameras[:-validation_count] if validation_count else train_cameras
        if not train_cameras:
            raise ValueError("Validation split consumed all training cameras; reduce --validation_fraction.")
        print(f"[Validation] Holding out {len(validation_cameras)} training cameras for early stopping; test cameras are reserved for final evaluation only.")
    viewpoint_stack = train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    iteration = first_iter + 1
    best_validation_psnr = float("-inf")
    patience = args.early_stopping_patience
    stale_counter = 0
    stop_training = False
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()

    progress_bar = tqdm(desc="Training progress (deterministic)")

    # # === Deterministic camera cycling setup ===
    # train_cameras = scene.getTrainCameras()
    # num_cameras = len(train_cameras)

    while iteration <= opt.iterations:

        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background,
                                    scaling_modifier=scaling_modifer,
                                    use_trained_exp=dataset.train_test_exp,
                                    separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255)
                                                .byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

         # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # # === Deterministic camera selection ===
        # camera_idx = (iteration - 1) % num_cameras   ### CHANGED
        # viewpoint_cam = train_cameras[camera_idx]    ### CHANGED

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = background   ### CHANGED (no random background)

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE)

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"]
        )

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        current_lambda_edge, current_lambda_saliency = curriculum_lambda_weights(
            iteration, opt.iterations, args.lambda_edge, args.lambda_saliency, args
        )
        combiner.lambda_edge = current_lambda_edge
        combiner.lambda_saliency = current_lambda_saliency
        Ll1 = combiner.compute_loss(image, gt_image)

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if tb_writer and getattr(args, "adaptive_curriculum", False):
                tb_writer.add_scalar('train_loss_patches/lambda_edge_curriculum', current_lambda_edge, iteration)
                tb_writer.add_scalar('train_loss_patches/lambda_saliency_curriculum', current_lambda_saliency, iteration)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "λe": f"{current_lambda_edge:.3f}", "λs": f"{current_lambda_saliency:.3f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations) or (iteration == opt.iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                importance_samples, error_samples = None, None
                if args.segs_densification:
                    importance_map = combiner.compute_importance_map(gt_image).detach()
                    error_map = torch.abs(image.detach() - gt_image).mean(dim=0, keepdim=True).unsqueeze(0)
                    importance_samples, error_samples = _sample_gaussian_image_metrics(
                        viewpoint_cam, gaussians, visibility_filter, importance_map, error_map
                    )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, importance_samples, error_samples)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii,
                        use_segs_score=args.segs_densification,
                        importance_power=args.segs_importance_power,
                        error_power=args.segs_error_power,
                        confidence_power=args.segs_confidence_power,
                        prune_score_threshold=args.segs_prune_score_threshold,
                    )
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        if getattr(args, "enable_early_stopping", False) and validation_cameras and iteration % args.early_stopping_interval == 0:
            validation_psnr = DetermineCamerasPSNR(
                tb_writer, iteration, validation_cameras, "validation", scene,
                render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp
            )
            if validation_psnr > best_validation_psnr:
                best_validation_psnr = validation_psnr
                stale_counter = 0
                print(f"\n✨ New best validation PSNR: {float(best_validation_psnr):.4f}")
            else:
                stale_counter += 1
                print(f"Validation PSNR did not improve. Patience counter = {stale_counter}/{patience}")

            if stale_counter >= patience:
                print(f"\n⏹️ Stopping training: no validation PSNR improvement after {patience} evaluations.")
                stop_training = True

        if (iteration == opt.iterations) or stop_training:
            if iteration not in saving_iterations:
                print("\n[ITER {}] Saving final Gaussians".format(iteration))
                scene.save(iteration)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_training_seconds = time.perf_counter() - training_start
            metrics = DetermineTestMetrics(
                tb_writer, iteration, testing_iterations, scene,
                render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp, combiner
            )
            if metrics is not None:
                metrics = {k: (v.detach().item() if torch.is_tensor(v) else float(v))
                        for k, v in metrics.items()}
                measured_render_fps = metrics.pop("render_fps", None)
                budget = collect_optimization_budget(
                    model_path=args.model_path,
                    gaussians=gaussians,
                    render_fps=measured_render_fps,
                )
                log_metrics_to_excel(
                    iteration,
                    metrics,
                    args.model_path,
                    lambda_edge=args.lambda_edge,
                    lambda_saliency=args.lambda_saliency,
                    use_edge=args.use_edge,
                    use_saliency=args.use_saliency,
                    use_method=args.saliency_name,
                    total_Time=total_training_seconds,
                    budget=budget,
                )
            break

        iteration = iteration + 1

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set

    # if iteration in testing_iterations:

    #     DeterminePSNR(tb_writer, iteration, l1_loss, testing_iterations, scene, renderFunc, renderArgs, train_test_exp)
        
    #     if tb_writer:
    #         tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
    #         tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

def DeterminePSNR(tb_writer, iteration, l1_loss, testing_iterations, scene, renderFunc, renderArgs, train_test_exp):
    torch.cuda.empty_cache()
    validation_configs = (
        {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
                            {'name': 'test', 'cameras' : scene.getTestCameras()})
    for config in validation_configs:
         with torch.no_grad():
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                #print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
    torch.cuda.empty_cache()
    return psnr_test

def DetermineTestPSNR(tb_writer, iteration, scene, renderFunc, renderArgs, train_test_exp):
    torch.cuda.empty_cache()

    # Only use test cameras
    test_cameras = scene.getTestCameras()
    psnr_test = 0.0

    if test_cameras and len(test_cameras) > 0:
        with torch.no_grad():
            for idx, viewpoint in enumerate(test_cameras):
                # Render and clamp
                image = torch.clamp(
                    renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 
                    0.0, 1.0
                )
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                # Optional: crop half if train/test experiment
                if train_test_exp:
                    image = image[..., image.shape[-1] // 2:]
                    gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                # Log some test renders
                if tb_writer and (idx < 5):
                    tb_writer.add_images(
                        f"test_view_{viewpoint.image_name}/render", 
                        image[None], 
                        global_step=iteration
                    )
                    # if iteration == testing_iterations[0]:
                    #     tb_writer.add_images(
                    #         f"test_view_{viewpoint.image_name}/ground_truth", 
                    #         gt_image[None], 
                    #         global_step=iteration
                    #     )

                # Accumulate PSNR
                psnr_test += psnr(image, gt_image).mean().double()

            psnr_test /= len(test_cameras)

            # Log scalar PSNR
            if tb_writer:
                tb_writer.add_scalar("test/psnr", psnr_test, iteration)

    torch.cuda.empty_cache()
    return psnr_test

def DetermineCamerasPSNR(tb_writer, iteration, cameras, name, scene, renderFunc, renderArgs, train_test_exp):
    """Evaluate PSNR on an explicit non-test camera split for model selection."""
    torch.cuda.empty_cache()
    psnr_value = 0.0
    if cameras and len(cameras) > 0:
        with torch.no_grad():
            for idx, viewpoint in enumerate(cameras):
                image = torch.clamp(
                    renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                    0.0, 1.0
                )
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if train_test_exp:
                    image = image[..., image.shape[-1] // 2:]
                    gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                if tb_writer and idx < 5:
                    tb_writer.add_images(f"{name}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                psnr_value += psnr(image, gt_image).mean().double()
            psnr_value /= len(cameras)
            if tb_writer:
                tb_writer.add_scalar(f"{name}/psnr", psnr_value, iteration)
    torch.cuda.empty_cache()
    return psnr_value

def DetermineTestMetrics(tb_writer, iteration, testing_iterations, scene, renderFunc, renderArgs, train_test_exp, combiner):
    torch.cuda.empty_cache()

    test_cameras = scene.getTestCameras()
    psnr_test, ssim_test, ms_ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
    render_time_seconds = 0.0
    edge_sim, saliency_sim = 0.0, 0.0

    if test_cameras and len(test_cameras) > 0:
        with torch.no_grad():
            for idx, viewpoint in enumerate(test_cameras):
                # Render and clamp
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                render_start = time.perf_counter()
                image = torch.clamp(
                    renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                    0.0, 1.0
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                render_time_seconds += time.perf_counter() - render_start
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                # Optional crop
                if train_test_exp:
                    image = image[..., image.shape[-1] // 2:]
                    gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                # TensorBoard logging
                if tb_writer and (idx < 5):
                    tb_writer.add_images(
                        f"test_view_{viewpoint.image_name}/render",
                        image[None],
                        global_step=iteration
                    )

                # PSNR
                psnr_test += psnr(image, gt_image).mean().double()

                # SSIM / MS-SSIM
                ssim_test += ssim(image, gt_image).mean().double()
                ms_ssim_test += ms_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), data_range=1.0, size_average=True)

                # LPIPS
                lpips_test += lpips_fn(image*2-1, gt_image*2-1).mean()  # scale to [-1,1]

                # Edge similarity (cosine)
                edge_Similaity, saliency_Similarity = combiner.Get_Edge_saliency_Similarity(image, gt_image)
                edge_sim += edge_Similaity

                # Saliency similarity (cosine)
                saliency_sim += saliency_Similarity

            # Average
            n = len(test_cameras)
            psnr_test /= n
            ssim_test /= n
            ms_ssim_test /= n
            lpips_test /= n
            edge_sim /= n
            saliency_sim /= n

            # Log scalars
            render_fps = n / render_time_seconds if render_time_seconds > 0 else None

            if tb_writer:
                tb_writer.add_scalar("test/psnr", psnr_test, iteration)
                tb_writer.add_scalar("test/ssim", ssim_test, iteration)
                tb_writer.add_scalar("test/ms_ssim", ms_ssim_test, iteration)
                tb_writer.add_scalar("test/lpips", lpips_test, iteration)
                tb_writer.add_scalar("test/edge_similarity", edge_sim, iteration)
                tb_writer.add_scalar("test/saliency_similarity", saliency_sim, iteration)
                if render_fps is not None:
                    tb_writer.add_scalar("test/render_fps", render_fps, iteration)

    torch.cuda.empty_cache()

    def scalar_or_none(value):
        if value is None:
            return None
        return value.item() if hasattr(value, "item") else value

    return {
        "psnr": scalar_or_none(psnr_test),
        "ssim": scalar_or_none(ssim_test),
        "ms_ssim": scalar_or_none(ms_ssim_test),
        "lpips": scalar_or_none(lpips_test),
        "edge_sim": scalar_or_none(edge_sim),
        "saliency_sim": scalar_or_none(saliency_sim),
        "render_fps": render_fps if 'render_fps' in locals() else None
    }

if __name__ == "__main__":

    def set_deterministic(seed: int = 42):
        # Python & NumPy
        random.seed(seed)
        np.random.seed(seed)

        # Torch seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # (Optional) make hash functions stable across runs
        os.environ["PYTHONHASHSEED"] = str(seed)

        print(f"[Reproducibility] Using seed {seed} and deterministic settings.")

    # === Call at startup ===
    set_deterministic(42)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 15000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 15000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--use_edge", action="store_true", default=False)
    parser.add_argument("--use_saliency", action="store_true", default=False)
    parser.add_argument("--edge_name", type=str, default="sobel",
                        choices=["sobel"])
    parser.add_argument("--saliency_name", type=str, default="Boolean",
                        choices=["Boolean", "itti"])

    parser.add_argument("--lambda_edge", type=float, default=0.2,
                        help="Weight for edge contribution")
    parser.add_argument("--lambda_saliency", type=float, default=0.1,
                        help="Weight for saliency contribution")
    parser.add_argument("--adaptive_curriculum", action="store_true", default=False,
                        help="Ramp edge/saliency loss weights over training instead of using fixed lambdas")
    parser.add_argument("--curriculum_edge_delay", type=float, default=0.05,
                        help="Fraction of training before edge ramp starts")
    parser.add_argument("--curriculum_edge_tau", type=float, default=0.18,
                        help="Exponential time constant, as a fraction of training, for edge ramp")
    parser.add_argument("--curriculum_saliency_start", type=float, default=0.55,
                        help="Training fraction at the midpoint of the saliency sigmoid ramp")
    parser.add_argument("--curriculum_saliency_width", type=float, default=0.08,
                        help="Sigmoid width, as a fraction of training, for saliency ramp")
    parser.add_argument("--curriculum_final_decay_start", type=float, default=0.90,
                        help="Training fraction where final stabilizing decay begins")
    parser.add_argument("--curriculum_final_weight_floor", type=float, default=0.50,
                        help="Fraction of ramped weights retained at the final iteration")
    parser.add_argument("--weighting_control", action="store_true", default=False,
                        help="Use L_control = c * L_3DGS, where c is the mean of the unnormalized edge/saliency weight map")
    parser.add_argument("--segs_densification", action="store_true", default=False,
                        help="Use one SEGS score to guide loss, densification, and low-score pruning")
    parser.add_argument("--segs_importance_power", type=float, default=1.0,
                        help="Exponent eta for perceptual importance in the SEGS densification score")
    parser.add_argument("--segs_error_power", type=float, default=1.0,
                        help="Exponent rho for reconstruction error in the SEGS densification score")
    parser.add_argument("--segs_confidence_power", type=float, default=0.5,
                        help="Exponent for multi-view visibility confidence in the SEGS densification score")
    parser.add_argument("--segs_prune_score_threshold", type=float, default=0.0,
                        help="Prune low-error Gaussians whose SEGS score is below this fraction of densify_grad_threshold")
    parser.add_argument("--enable_early_stopping", action="store_true", default=False,
                        help="Use a held-out validation split for early stopping; test views remain final-only.")
    parser.add_argument("--validation_fraction", type=float, default=0.1,
                        help="Fraction of training cameras held out for early stopping validation.")
    parser.add_argument("--early_stopping_interval", type=int, default=2000,
                        help="Validation interval for early stopping experiments.")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Number of validation checks without improvement before stopping.")


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.edge_name, args.saliency_name, args)

    # All done
    print("\nTraining complete.")