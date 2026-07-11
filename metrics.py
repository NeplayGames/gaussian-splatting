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


def evaluate_split(model_path, split_dir):
    render_dir = split_dir / "renders"
    gt_dir = split_dir / "gt"
    ssims = []
    psnrs = []
    lpipss = []

    for render, gt in tqdm(read_images(render_dir, gt_dir), desc=f"Metrics {model_path.name}/{split_dir.parent.name}/{split_dir.name}"):
        render = render.unsqueeze(0)
        gt = gt.unsqueeze(0)
        ssims.append(ssim(render, gt).mean().double())
        psnrs.append(psnr(render, gt).mean().double())
        lpipss.append(lpips_distance(render, gt).mean().double())

    return {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
    }


def find_render_sets(model_path):
    for split_name in ("test", "train"):
        split_root = model_path / split_name
        if not split_root.exists():
            continue
        for iteration_dir in sorted(split_root.glob("ours_*")):
            if (iteration_dir / "renders").is_dir() and (iteration_dir / "gt").is_dir():
                yield split_name, iteration_dir


def evaluate_model(model_path):
    model_path = Path(model_path)
    results = {}
    for split_name, iteration_dir in find_render_sets(model_path):
        results[f"{split_name}/{iteration_dir.name}"] = evaluate_split(model_path, iteration_dir)
    if not results:
        raise RuntimeError(f"No rendered evaluation sets found under {model_path}. Run render.py first.")
    metrics_path = model_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Scene: {model_path}")
    for key, values in results.items():
        print(f"  {key}: SSIM {values['SSIM']:.7f} PSNR {values['PSNR']:.7f} LPIPS {values['LPIPS']:.7f}")
    print(f"[METRICS] Saved {metrics_path}")
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute PSNR, SSIM and LPIPS for rendered Gaussian Splatting outputs.")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", help="One or more trained model directories containing render.py outputs.")
    args = parser.parse_args()

    with torch.no_grad():
        for model_path in args.model_paths:
            evaluate_model(model_path)
