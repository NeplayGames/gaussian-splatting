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
from TrainingReport import log_metrics_to_excel
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

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    iteration = first_iter + 1
    best_metrics = {
        "psnr": float("-inf"),
        "ssim": float("-inf"),
        "ms_ssim": float("-inf"),
        "lpips": float("inf"),        # lower is better
        "edge_sim": float("-inf"),
        "saliency_sim": float("-inf"),
    }
    best_psnr = 0
    patience = 10
    stale_counter = 0

    progress_bar = tqdm(desc="Training progress (deterministic)")

    # # === Deterministic camera cycling setup ===
    # train_cameras = scene.getTrainCameras()
    # num_cameras = len(train_cameras)

    while True:

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
            viewpoint_stack = scene.getTrainCameras().copy()
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

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

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
        if iteration % 2000 == 0:
            # Run evaluation (returns dict of metrics)    

            psnr_val = DetermineTestPSNR(
                    tb_writer, iteration,  scene,
                    render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                    dataset.train_test_exp
                )
            #def DetermineTestPSNR(tb_writer, iteration, scene, renderFunc, renderArgs, train_test_exp):
            # --- Check for improvement (PSNR as criterion) ---
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                

                stale_counter = 0
                print(f"\n✨ New best PSNR: {best_psnr:.4f}")
                # Optionally: save best model here
                # scene.save(iteration)
            else:
                stale_counter += 1
                print(f"PSNR did not improve. Patience counter = {stale_counter}/{patience}")

            # --- Early stopping ---
            #if iteration % 30000 == 0:
            if stale_counter >= patience:
            #if psnr_val >= 25:
                scene.save(iteration)
            #
                metrics = DetermineTestMetrics(
                    tb_writer, iteration, testing_iterations, scene,
                    render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                    dataset.train_test_exp, combiner
                )
                if metrics is not None:
                    # Convert tensors → floats
                    metrics = {k: (v.detach().item() if torch.is_tensor(v) else float(v)) 
                            for k, v in metrics.items()}
                    best_metrics["psnr"] = best_psnr
                    best_metrics["ssim"] = metrics["ssim"]
                    best_metrics["ms_ssim"] = metrics["ms_ssim"]
                    best_metrics["lpips"] = metrics["lpips"]
                    best_metrics["edge_sim"] = metrics["edge_sim"]
                    best_metrics["saliency_sim"] = metrics["saliency_sim"]
                    log_metrics_to_excel(
                        iteration,
                        best_metrics,  # pass the full dict
                        args.model_path,
                        lambda_edge=args.lambda_edge,
                        lambda_saliency=args.lambda_saliency,                        
                        use_edge=args.use_edge,
                        use_saliency=args.use_saliency,
                        use_method = args.saliency_name,
                        total_Time= iter_start.elapsed_time(iter_end),
                    )
                print(f"\n⏹️ Stopping training: no PSNR improvement after {patience} evaluations.")
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

def DetermineTestMetrics(tb_writer, iteration, testing_iterations, scene, renderFunc, renderArgs, train_test_exp, combiner):
    torch.cuda.empty_cache()

    test_cameras = scene.getTestCameras()
    psnr_test, ssim_test, ms_ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
    edge_sim, saliency_sim = 0.0, 0.0

    if test_cameras and len(test_cameras) > 0:
        with torch.no_grad():
            for idx, viewpoint in enumerate(test_cameras):
                # Render and clamp
                image = torch.clamp(
                    renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                    0.0, 1.0
                )
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
            if tb_writer:
                tb_writer.add_scalar("test/psnr", psnr_test, iteration)
                tb_writer.add_scalar("test/ssim", ssim_test, iteration)
                tb_writer.add_scalar("test/ms_ssim", ms_ssim_test, iteration)
                tb_writer.add_scalar("test/lpips", lpips_test, iteration)
                tb_writer.add_scalar("test/edge_similarity", edge_sim, iteration)
                tb_writer.add_scalar("test/saliency_similarity", saliency_sim, iteration)

    torch.cuda.empty_cache()
    return {
        "psnr": psnr_test.item(),
        "ssim": ssim_test,
        "ms_ssim": ms_ssim_test,
        "lpips": lpips_test,
        "edge_sim": edge_sim,
        "saliency_sim": saliency_sim
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
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