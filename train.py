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
import json
import math
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
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
from TrainingReport import collect_optimization_budget
import Edges
import Saliency

def _set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def curriculum_lambda_weights(iteration, total_iterations, alpha_max, beta_max, args):
    if not getattr(args, "adaptive_curriculum", False):
        return alpha_max, beta_max

    total_iterations = max(int(total_iterations), 1)
    t = max(float(iteration), 0.0)
    edge_delay = float(args.curriculum_edge_delay) * total_iterations
    edge_tau = max(float(args.curriculum_edge_tau) * total_iterations, 1.0)
    alpha = alpha_max * (1.0 - math.exp(-max(t - edge_delay, 0.0) / edge_tau))

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


def _sample_gaussian_image_metrics(viewpoint_cam, gaussians, visibility_filter, importance_map, error_map):
    visible_ids = visibility_filter.reshape(-1)
    if visible_ids.numel() == 0:
        return None, None

    points = gaussians.get_xyz[visible_ids]
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    hom_points = torch.cat((points, ones), dim=1)
    projected = hom_points @ viewpoint_cam.full_proj_transform
    ndc = projected[:, :2] / projected[:, 3:4].clamp_min(1e-7)
    grid = ndc.view(1, -1, 1, 2)
    importance = F.grid_sample(importance_map, grid, align_corners=True, padding_mode="border").view(-1)
    error = F.grid_sample(error_map, grid, align_corners=True, padding_mode="border").view(-1)
    return importance, error


SEGS_METHODS = [
    "baseline",
    "segs_edge_only",
    "segs_saliency_only",
    "segs_loss",
    "segs_densification_only",
    "segs_loss_and_densification",
    "segs_curriculum",
    "segs_full",
    "constant_scale_control",
    "shuffled_map_control",
]


def _configure_segs_method(args):
    args.use_edge = False
    args.use_saliency = False
    args.adaptive_curriculum = False
    args.segs_densification = False
    requested_weighting_control = getattr(args, "weighting_control", False)
    requested_shuffle_map_control = getattr(args, "shuffle_map_control", False)
    args.weighting_control = False
    args.shuffle_map_control = False

    if args.method == "baseline":
        return
    if args.method == "segs_edge_only":
        args.use_edge = True
    elif args.method == "segs_saliency_only":
        args.use_saliency = True
    elif args.method in ("segs_loss", "constant_scale_control", "shuffled_map_control"):
        args.use_edge = True
        args.use_saliency = True
        args.weighting_control = args.method == "constant_scale_control" or requested_weighting_control
        args.shuffle_map_control = args.method == "shuffled_map_control" or requested_shuffle_map_control
    elif args.method == "segs_densification_only":
        args.use_edge = True
        args.use_saliency = True
        args.segs_densification = True
        args.weighting_control = requested_weighting_control
        args.shuffle_map_control = requested_shuffle_map_control
    elif args.method == "segs_loss_and_densification":
        args.use_edge = True
        args.use_saliency = True
        args.segs_densification = True
        args.weighting_control = requested_weighting_control
        args.shuffle_map_control = requested_shuffle_map_control
    elif args.method == "segs_curriculum":
        args.use_edge = True
        args.use_saliency = True
        args.adaptive_curriculum = True
        args.weighting_control = requested_weighting_control
        args.shuffle_map_control = requested_shuffle_map_control
    elif args.method == "segs_full":
        args.use_edge = True
        args.use_saliency = True
        args.adaptive_curriculum = True
        args.segs_densification = True
        args.weighting_control = requested_weighting_control
        args.shuffle_map_control = requested_shuffle_map_control
    else:
        raise ValueError(f"Unknown method: {args.method}")



def _assert_demo_method_configuration(args):
    if args.method == "baseline":
        assert args.method == "baseline"
        assert not args.use_edge, "baseline must disable edge weighting"
        assert not args.use_saliency, "baseline must disable saliency weighting"
        assert not args.segs_densification, "baseline must disable SEGS densification"
        assert not args.adaptive_curriculum, "baseline must disable adaptive curriculum"
        assert not args.weighting_control, "baseline must disable constant-scale controls"
        assert not args.shuffle_map_control, "baseline must disable shuffled-map controls"
    elif args.method == "segs_full":
        assert args.use_edge and args.use_saliency, "segs_full must enable edge and saliency weighting"
        assert args.adaptive_curriculum, "segs_full must enable adaptive curriculum"
        assert args.segs_densification, "segs_full must enable importance-aware densification"
        assert not args.weighting_control, "segs_full must not use constant-scale controls"
        assert not args.shuffle_map_control, "segs_full must not use shuffled-map controls"

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):

    training_start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    combiner = None
    if args.method != "baseline" or args.segs_densification:
        combiner = LossCombiner(
            Edges.get_edge_processor(args.edge_name),
            Saliency.get_saliency_processor(args.saliency_name),
            use_edge=args.use_edge,
            use_saliency=args.use_saliency,
            lambda_edge=args.lambda_edge,
            lambda_saliency=args.lambda_saliency,
            normalize=True,
            constant_scaling_control=args.weighting_control,
            shuffle_map_control=getattr(args, "shuffle_map_control", False),
        )

    tb_writer = prepare_output_and_logger(dataset, args)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
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

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
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

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if combiner is not None and (args.use_edge or args.use_saliency) and args.method != "segs_densification_only":
            current_lambda_edge, current_lambda_saliency = curriculum_lambda_weights(
                iteration, opt.iterations, args.lambda_edge, args.lambda_saliency, args
            )
            combiner.lambda_edge = current_lambda_edge
            combiner.lambda_saliency = current_lambda_saliency
            Ll1 = combiner.compute_weighted_l1(image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)

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
            if (iteration in saving_iterations):
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

    total_training_time = time.perf_counter() - training_start_time
    render_fps = measure_render_fps(scene, gaussians, pipe, background, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE)
    budget = collect_optimization_budget(scene.model_path, gaussians=gaussians, render_fps=render_fps)
    budget["total_training_time_seconds"] = total_training_time
    write_budget_report(scene.model_path, opt.iterations, budget)


def measure_render_fps(scene, gaussians, pipe, background, train_test_exp, separate_sh, max_frames=30, warmup_frames=3):
    cameras = scene.getTestCameras() or scene.getTrainCameras()
    if not cameras:
        return None
    cameras = cameras[:max_frames]
    torch.cuda.empty_cache()
    with torch.no_grad():
        for viewpoint in cameras[:warmup_frames]:
            render(viewpoint, gaussians, pipe, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for viewpoint in cameras:
            render(viewpoint, gaussians, pipe, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        return None
    return len(cameras) / elapsed


def write_budget_report(model_path, iteration, budget):
    os.makedirs(model_path, exist_ok=True)
    report = {"iteration": iteration, **budget}
    report_path = os.path.join(model_path, "optimization_budget.json")
    with open(report_path, "w") as budget_f:
        json.dump(report, budget_f, indent=2, sort_keys=True)
    print(f"[BUDGET] Saved optimization budget to {report_path}: {report}")

def _runtime_metadata():
    metadata = {}
    try:
        import subprocess
        metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        metadata["git_commit"] = None
    metadata["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    metadata["cuda_version"] = torch.version.cuda
    metadata["pytorch_version"] = torch.__version__
    return metadata


def prepare_output_and_logger(args, run_args=None):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    metadata_args = run_args if run_args is not None else args
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(metadata_args))))
    with open(os.path.join(args.model_path, "cfg_args.json"), 'w') as cfg_json_f:
        json.dump(vars(metadata_args), cfg_json_f, indent=2, sort_keys=True)
    with open(os.path.join(args.model_path, "runtime_metadata.json"), 'w') as metadata_f:
        json.dump(_runtime_metadata(), metadata_f, indent=2, sort_keys=True)
    with open(os.path.join(args.model_path, "seed.txt"), 'w') as seed_f:
        seed_f.write(str(getattr(metadata_args, "seed", 0)))

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
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
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
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1,7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1,7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, default="baseline")
    parser.add_argument("--use_edge", action="store_true", default=False)
    parser.add_argument("--use_saliency", action="store_true", default=False)
    parser.add_argument("--edge_name", type=str, default="sobel", choices=["sobel"])
    parser.add_argument("--saliency_name", type=str, default="BooleanMapApprox", choices=["BooleanMapApprox", "IntensityCenterSurround", "Boolean", "itti"])
    parser.add_argument("--lambda_edge", type=float, default=0.2)
    parser.add_argument("--lambda_saliency", type=float, default=0.1)
    parser.add_argument("--adaptive_curriculum", action="store_true", default=False)
    parser.add_argument("--curriculum_edge_delay", type=float, default=0.05)
    parser.add_argument("--curriculum_edge_tau", type=float, default=0.18)
    parser.add_argument("--curriculum_saliency_start", type=float, default=0.55)
    parser.add_argument("--curriculum_saliency_width", type=float, default=0.08)
    parser.add_argument("--curriculum_final_decay_start", type=float, default=0.90)
    parser.add_argument("--curriculum_final_weight_floor", type=float, default=0.50)
    parser.add_argument("--weighting_control", action="store_true", default=False)
    parser.add_argument("--shuffle_map_control", action="store_true", default=False)
    parser.add_argument("--segs_densification", action="store_true", default=False)
    parser.add_argument("--segs_importance_power", type=float, default=1.0)
    parser.add_argument("--segs_error_power", type=float, default=1.0)
    parser.add_argument("--segs_confidence_power", type=float, default=0.5)
    parser.add_argument("--segs_prune_score_threshold", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    if args.method == "eggs":
        parser.error("Unsupported method 'eggs'. EGGS is not implemented; use 'segs_edge_only' for the edge-only method.")
    if args.method not in SEGS_METHODS:
        parser.error(f"Unsupported method {args.method!r}. Supported methods: {', '.join(SEGS_METHODS)}")
    _configure_segs_method(args)
    _assert_demo_method_configuration(args)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)
    _set_deterministic(args.seed)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")