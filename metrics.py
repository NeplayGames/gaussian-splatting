#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

import lpips
import Edges
import Saliency
from utils.image_utils import psnr
from utils.loss_utils import ssim

_lpips_model = None

def lpips_distance(render, gt):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="vgg").cuda().eval()
    return _lpips_model(render * 2.0 - 1.0, gt * 2.0 - 1.0)


def read_images(render_dir, gt_dir):
    renders = sorted(Path(render_dir).glob("*.png"))
    gts = sorted(Path(gt_dir).glob("*.png"))
    if len(renders) != len(gts):
        raise RuntimeError(f"Mismatched render/GT counts: {render_dir} has {len(renders)}, {gt_dir} has {len(gts)}")
    if not renders:
        raise RuntimeError(f"No render images found in {render_dir}")

    for render_path, gt_path in zip(renders, gts):
        render = torchvision.io.read_image(str(render_path)).float()[:3, :, :] / 255.0
        gt = torchvision.io.read_image(str(gt_path)).float()[:3, :, :] / 255.0
        yield render.cuda(), gt.cuda()


def _top_fraction_mask(score_map, fraction):
    fraction = min(max(float(fraction), 1e-6), 1.0)
    flat = score_map.flatten(start_dim=1)
    threshold = torch.quantile(flat, 1.0 - fraction, dim=1, keepdim=True)
    return score_map >= threshold.view(score_map.shape[0], 1, 1, 1)


def _masked_mse(render, gt, mask):
    mask = mask.to(device=render.device, dtype=render.dtype)
    while mask.dim() < render.dim():
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and render.shape[1] != 1:
        mask = mask.expand(-1, render.shape[1], -1, -1)
    denom = mask.sum().clamp_min(1.0)
    return (((render - gt) ** 2) * mask).sum() / denom


def _masked_mae(render, gt, mask):
    mask = mask.to(device=render.device, dtype=render.dtype)
    while mask.dim() < render.dim():
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and render.shape[1] != 1:
        mask = mask.expand(-1, render.shape[1], -1, -1)
    denom = mask.sum().clamp_min(1.0)
    return ((render - gt).abs() * mask).sum() / denom


def _psnr_from_mse(mse):
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def region_metrics(render, gt, edge_processor, saliency_processor, edge_fraction, saliency_fraction):
    edge_score = edge_processor.sobel_filter(edge_processor.rgb_to_grayscale(gt))
    saliency_score = saliency_processor.get_saliency_map(gt)
    edge_mask = _top_fraction_mask(edge_score, edge_fraction)
    saliency_mask = _top_fraction_mask(saliency_score, saliency_fraction)

    edge_mse = _masked_mse(render, gt, edge_mask)
    saliency_mse = _masked_mse(render, gt, saliency_mask)
    return {
        "edge_region_PSNR": _psnr_from_mse(edge_mse).double(),
        "edge_region_MAE": _masked_mae(render, gt, edge_mask).double(),
        "edge_region_fraction": edge_mask.float().mean().double(),
        "saliency_region_PSNR": _psnr_from_mse(saliency_mse).double(),
        "saliency_region_MAE": _masked_mae(render, gt, saliency_mask).double(),
        "saliency_region_fraction": saliency_mask.float().mean().double(),
    }


def evaluate_split(model_path, split_dir, compute_regions=False, saliency_name="BooleanMapApprox", edge_fraction=0.10, saliency_fraction=0.10):
    render_dir = split_dir / "renders"
    gt_dir = split_dir / "gt"
    ssims = []
    psnrs = []
    lpipss = []
    region_values = {}
    edge_processor = Edges.get_edge_processor("sobel") if compute_regions else None
    saliency_processor = Saliency.get_saliency_processor(saliency_name) if compute_regions else None

    for render, gt in tqdm(read_images(render_dir, gt_dir), desc=f"Metrics {model_path.name}/{split_dir.parent.name}/{split_dir.name}"):
        render = render.unsqueeze(0)
        gt = gt.unsqueeze(0)
        ssims.append(ssim(render, gt).mean().double())
        psnrs.append(psnr(render, gt).mean().double())
        lpipss.append(lpips_distance(render, gt).mean().double())
        if compute_regions:
            values = region_metrics(render, gt, edge_processor, saliency_processor, edge_fraction, saliency_fraction)
            for key, value in values.items():
                region_values.setdefault(key, []).append(value)

    result = {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
    }
    for key, values in region_values.items():
        result[key] = torch.tensor(values).mean().item()
    return result


def find_render_sets(model_path):
    for split_name in ("test", "train"):
        split_root = model_path / split_name
        if not split_root.exists():
            continue
        for iteration_dir in sorted(split_root.glob("ours_*")):
            if (iteration_dir / "renders").is_dir() and (iteration_dir / "gt").is_dir():
                yield split_name, iteration_dir


def evaluate_model(model_path, compute_regions=False, saliency_name="BooleanMapApprox", edge_fraction=0.10, saliency_fraction=0.10):
    model_path = Path(model_path)
    results = {}
    for split_name, iteration_dir in find_render_sets(model_path):
        results[f"{split_name}/{iteration_dir.name}"] = evaluate_split(
            model_path,
            iteration_dir,
            compute_regions=compute_regions,
            saliency_name=saliency_name,
            edge_fraction=edge_fraction,
            saliency_fraction=saliency_fraction,
        )
    if not results:
        raise RuntimeError(f"No rendered evaluation sets found under {model_path}. Run render.py first.")
    metrics_path = model_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Scene: {model_path}")
    for key, values in results.items():
        region_text = ""
        if "edge_region_PSNR" in values:
            region_text = f" edgePSNR {values['edge_region_PSNR']:.7f} salPSNR {values['saliency_region_PSNR']:.7f}"
        print(f"  {key}: SSIM {values['SSIM']:.7f} PSNR {values['PSNR']:.7f} LPIPS {values['LPIPS']:.7f}{region_text}")
    print(f"[METRICS] Saved {metrics_path}")
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute PSNR, SSIM and LPIPS for rendered Gaussian Splatting outputs.")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", help="One or more trained model directories containing render.py outputs.")
    parser.add_argument("--region_metrics", action="store_true", help="Also compute edge-region and saliency-region PSNR/MAE from GT-derived masks.")
    parser.add_argument("--saliency_name", default="BooleanMapApprox", choices=["BooleanMapApprox", "IntensityCenterSurround", "Boolean", "itti"])
    parser.add_argument("--edge_region_fraction", type=float, default=0.10)
    parser.add_argument("--saliency_region_fraction", type=float, default=0.10)
    args = parser.parse_args()

    with torch.no_grad():
        for model_path in args.model_paths:
            evaluate_model(
                model_path,
                compute_regions=args.region_metrics,
                saliency_name=args.saliency_name,
                edge_fraction=args.edge_region_fraction,
                saliency_fraction=args.saliency_region_fraction,
            )
