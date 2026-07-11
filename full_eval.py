#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

METHODS = [
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

DATASETS = {
    "m360": {
        "root_arg": "mipnerf360",
        "scenes": [("bicycle", "images_4"), ("flowers", "images_4"), ("garden", "images_4"), ("stump", "images_4"), ("treehill", "images_4"),
                   ("room", "images_2"), ("counter", "images_2"), ("kitchen", "images_2"), ("bonsai", "images_2")],
    },
    "tandt": {"root_arg": "tanksandtemples", "scenes": [("truck", None), ("train", None)]},
    "db": {"root_arg": "deepblending", "scenes": [("drjohnson", None), ("playroom", None)]},
}


def parse_csv(value, cast=str):
    return [cast(item) for item in value.split(",") if item]


from experiments.subprocess_runner import run_command as run_step
from experiments.command_builder import validate_method


def main():
    parser = ArgumentParser(description="Full evaluation script parameters")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_rendering", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip steps whose per-run status file is already successful.")
    parser.add_argument("--output_path", default="./eval")
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--use_expcomp", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--aa", action="store_true")
    parser.add_argument("--method", type=str, default="baseline", choices=METHODS)
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method list; overrides --method.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list; overrides --seed.")
    parser.add_argument("--weighting_control", action="store_true")
    parser.add_argument("--shuffle_map_control", action="store_true")
    parser.add_argument("--saliency_name", type=str, default=None)
    parser.add_argument("--mipnerf360", "-m360", type=str)
    parser.add_argument("--tanksandtemples", "-tat", type=str)
    parser.add_argument("--deepblending", "-db", type=str)
    args = parser.parse_args()

    methods = parse_csv(args.methods) if args.methods else [args.method]
    seeds = parse_csv(args.seeds, int) if args.seeds else [args.seed]
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "result_index.jsonl"

    need_sources = not args.skip_training or not args.skip_rendering
    if need_sources:
        missing = [spec["root_arg"] for spec in DATASETS.values() if getattr(args, spec["root_arg"]) is None]
        if missing:
            parser.error("Required dataset roots missing: " + ", ".join(missing))

    results = []
    for method in methods:
        if method not in METHODS:
            validate_method(method, METHODS)
        for seed in seeds:
            model_root = output_root / f"{method}_seed{seed}"
            scenes_for_metrics = []
            for dataset_name, spec in DATASETS.items():
                dataset_root = getattr(args, spec["root_arg"])
                for scene, image_dir in spec["scenes"]:
                    run_dir = model_root / scene
                    scenes_for_metrics.append(str(run_dir))
                    source = str(Path(dataset_root) / scene) if dataset_root else None
                    if not args.skip_training:
                        cmd = [sys.executable, "train.py", "-s", source, "-m", str(run_dir), "--disable_viewer", "--quiet", "--eval", "--test_iterations", "-1", "--method", method, "--seed", str(seed)]
                        if image_dir:
                            cmd.extend(["-i", image_dir])
                        if args.aa:
                            cmd.append("--antialiasing")
                        if args.use_depth:
                            cmd.extend(["-d", "depths2/"])
                        if args.use_expcomp:
                            cmd.extend(["--exposure_lr_init", "0.001", "--exposure_lr_final", "0.0001", "--exposure_lr_delay_steps", "5000", "--exposure_lr_delay_mult", "0.001", "--train_test_exp"])
                        if args.fast:
                            cmd.extend(["--optimizer_type", "sparse_adam"])
                        if args.weighting_control:
                            cmd.append("--weighting_control")
                        if args.shuffle_map_control:
                            cmd.append("--shuffle_map_control")
                        if args.saliency_name:
                            cmd.extend(["--saliency_name", args.saliency_name])
                        status = run_step(cmd, run_dir, "train", args.resume)
                        results.append({"method": method, "seed": seed, "dataset": dataset_name, "scene": scene, "step": "train", "status": status})
                    if not args.skip_rendering:
                        for iteration in (7000, 30000):
                            cmd = [sys.executable, "render.py", "--iteration", str(iteration), "-s", source, "-m", str(run_dir), "--quiet", "--eval", "--skip_train"]
                            if args.aa:
                                cmd.append("--antialiasing")
                            if args.use_expcomp:
                                cmd.append("--train_test_exp")
                            status = run_step(cmd, run_dir, f"render_{iteration}", args.resume)
                            results.append({"method": method, "seed": seed, "dataset": dataset_name, "scene": scene, "step": f"render_{iteration}", "status": status})
            if not args.skip_metrics:
                metrics_dir = model_root / "_metrics"
                status = run_step([sys.executable, "metrics.py", "-m", *scenes_for_metrics], metrics_dir, "metrics", args.resume)
                results.append({"method": method, "seed": seed, "step": "metrics", "status": status})

    with index_path.open("a") as index_f:
        for result in results:
            result["timestamp"] = time.time()
            index_f.write(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
