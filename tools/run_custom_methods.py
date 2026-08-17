import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.command_builder import render_command, train_command, validate_method


def parse_methods(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_logged(command: list[str], log_dir: Path, step: str, resume: bool, validator) -> dict:
    status_path = log_dir / f"{step}_status.json"
    if resume and validator():
        status = {"status": "skipped", "returncode": 0, "command": command}
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    stdout_path = log_dir / f"{step}.stdout.log"
    stderr_path = log_dir / f"{step}.stderr.log"
    start = time.time()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], stdout=stdout, stderr=stderr)
    status = {
        "status": "success" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "command": command,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "elapsed_seconds": time.time() - start,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"{step} failed for {log_dir.parent.name}/{log_dir.name}; see {stderr_path}")
    return status


def count_ply_vertices(path: Path):
    try:
        with path.open("r", errors="ignore") as handle:
            for line in handle:
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line.strip() == "end_header":
                    break
    except OSError:
        return None
    return None


def result_key(record: dict) -> tuple:
    return (
        record.get("dataset", ""),
        record.get("scene", ""),
        record.get("method", ""),
        record.get("run_name", ""),
        record.get("saliency_name", ""),
        record.get("normalization", ""),
        str(record.get("seed", "")),
        str(record.get("iteration", "")),
        record.get("split", ""),
        str(record.get("lambda_edge", "")),
        str(record.get("lambda_saliency", "")),
        str(record.get("eggs_beta", "")),
        str(record.get("edge_p", "")),
    )


def merge_existing_results(results_path: Path, records: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    if results_path.exists():
        with results_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                merged[result_key(row)] = row

    for record in records:
        merged[result_key(record)] = record
    return list(merged.values())


def load_existing_results(results_path: Path) -> dict[tuple, dict]:
    results: dict[tuple, dict] = {}
    if not results_path.exists():
        return results
    with results_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            results[result_key(row)] = row
    return results


def expected_record_key(args, dataset: str, scene_name: str, method: str, run_name: str, saliency_name: str | None) -> tuple:
    record = {
        "dataset": dataset,
        "scene": scene_name,
        "method": method,
        "run_name": run_name,
        "saliency_name": saliency_name or "",
        "normalization": "mean_one" if method.endswith("_norm") else "raw",
        "lambda_edge": args.lambda_edge,
        "lambda_saliency": args.lambda_saliency,
        "eggs_beta": args.eggs_beta if args.eggs_beta is not None else args.lambda_edge,
        "edge_p": args.edge_p,
        "seed": args.seed,
        "iteration": args.iterations,
        "split": args.split,
    }
    return result_key(record)


def csv_record_complete(record: dict | None) -> bool:
    if not record:
        return False
    for key in ("psnr", "ssim", "lpips"):
        try:
            value = float(record.get(key, ""))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def prune_after_metrics(model_dir: Path, output_root: Path, split: str) -> list[str]:
    model_root = (output_root / "models").resolve()
    resolved_model_dir = model_dir.resolve()
    if model_root not in resolved_model_dir.parents:
        raise ValueError(f"Refusing to prune outside output model root: {resolved_model_dir}")

    removed = []
    for path in (model_dir / "point_cloud", model_dir / split):
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))

    for path in model_dir.glob("chkpnt*.pth"):
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    return removed


def keep_render_samples(model_dir: Path, iteration: int, split: str, count: int, seed_label: str) -> list[str]:
    if count <= 0:
        return []

    render_root = model_dir / split / f"ours_{iteration}"
    render_dir = render_root / "renders"
    gt_dir = render_root / "gt"
    if not render_dir.is_dir():
        return []

    renders = sorted(render_dir.glob("*.png"))
    if not renders:
        return []

    rng = random.Random(seed_label)
    selected = sorted(rng.sample(renders, min(count, len(renders))))
    sample_root = model_dir / "render_samples" / f"{split}_ours_{iteration}"
    if sample_root.exists():
        shutil.rmtree(sample_root)
    (sample_root / "renders").mkdir(parents=True, exist_ok=True)
    (sample_root / "gt").mkdir(parents=True, exist_ok=True)

    copied = []
    for render_path in selected:
        render_target = sample_root / "renders" / render_path.name
        shutil.copy2(render_path, render_target)
        copied.append(str(render_target))

        gt_path = gt_dir / render_path.name
        if gt_path.exists():
            gt_target = sample_root / "gt" / gt_path.name
            shutil.copy2(gt_path, gt_target)
            copied.append(str(gt_target))
    return copied


def valid_training(model_dir: Path, iteration: int) -> bool:
    ply = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    return ply.exists() and ply.stat().st_size > 0 and (count_ply_vertices(ply) or 0) > 0


def valid_render(model_dir: Path, iteration: int, split: str) -> bool:
    root = model_dir / split / f"ours_{iteration}"
    renders = list((root / "renders").glob("*.png")) if (root / "renders").is_dir() else []
    gt = list((root / "gt").glob("*.png")) if (root / "gt").is_dir() else []
    return len(renders) > 0 and len(renders) == len(gt)


def valid_metrics(model_dir: Path, iteration: int, split: str) -> bool:
    path = model_dir / "metrics.json"
    if not path.exists():
        return False
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        values = metrics[f"{split}/ours_{iteration}"]
    except Exception:
        return False
    return all(isinstance(values.get(key), (int, float)) and math.isfinite(values[key]) for key in ("PSNR", "SSIM", "LPIPS"))


def load_metrics(model_dir: Path, iteration: int, split: str) -> dict:
    values = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))[f"{split}/ours_{iteration}"]
    metrics = {"psnr": values["PSNR"], "ssim": values["SSIM"], "lpips": values["LPIPS"]}
    for key, value in values.items():
        if key not in ("PSNR", "SSIM", "LPIPS") and isinstance(value, (int, float)):
            metrics[key.lower()] = value
    return metrics


def saliency_method_required(method: str) -> bool:
    return "saliency" in method


def method_run_name(method: str, saliency_name: str | None) -> str:
    if saliency_method_required(method) and saliency_name:
        return f"{method}_{saliency_name}"
    return method


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_jobs(args) -> list[dict]:
    if args.matrix:
        matrix = load_matrix(args.matrix)
        return matrix.get("scenes", [])
    return [
        {
            "scene_name": args.scene_name,
            "source": str(args.source),
            "dataset": args.dataset,
            "image_dir": args.image_dir,
            "white_background": args.white_background,
        }
    ]


def matrix_methods(args) -> list[str]:
    if args.matrix:
        matrix = load_matrix(args.matrix)
        return matrix.get("methods") or parse_methods(args.methods)
    return parse_methods(args.methods)


def matrix_saliency_methods(args) -> list[str]:
    if args.matrix:
        matrix = load_matrix(args.matrix)
        return matrix.get("saliency_methods") or parse_methods(args.saliency_methods)
    return parse_methods(args.saliency_methods)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run guidance ablations on one or more prepared Gaussian Splatting datasets.")
    parser.add_argument("--matrix", type=Path, help="JSON protocol containing scenes, methods, and saliency_methods.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--scene-name")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output-root", default=Path("demo_output"), type=Path)
    parser.add_argument(
        "--results-csv",
        default="guidance_ablation_results.csv",
        help="CSV filename under output-root, or an explicit path, for merged run results.",
    )
    parser.add_argument("--iterations", default=30000, type=int)
    parser.add_argument("--methods", default="baseline,eggs_paper,eggs,saliency,eggs_saliency,eggs_norm,saliency_norm,eggs_saliency_norm")
    parser.add_argument("--saliency-methods", default="BooleanMapApprox,IntensityCenterSurround")
    parser.add_argument("--lambda-edge", default=0.2, type=float)
    parser.add_argument("--lambda-saliency", default=0.1, type=float)
    parser.add_argument("--eggs-beta", default=None, type=float)
    parser.add_argument("--edge-p", default=1, type=int, choices=[1, 2])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--image-dir", help="Image directory name to pass to train.py/render.py, for example images_2 or images_4.")
    parser.add_argument("--white-background", action="store_true", help="Use white background for synthetic Blender/NeRF scenes.")
    parser.add_argument("--region-metrics", dest="region_metrics", action="store_true", default=True)
    parser.add_argument("--no-region-metrics", dest="region_metrics", action="store_false")
    parser.add_argument("--edge-region-fraction", default=0.10, type=float)
    parser.add_argument("--saliency-region-fraction", default=0.10, type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--csv-complete-resume",
        action="store_true",
        help="With --resume, skip a run when guidance_ablation_results.csv already has complete metrics for it.",
    )
    parser.add_argument(
        "--prune-after-metrics",
        action="store_true",
        help="After metrics are recorded, remove bulky point_cloud and render artifacts for the run.",
    )
    parser.add_argument(
        "--keep-render-samples",
        default=0,
        type=int,
        help="With --prune-after-metrics, keep this many random render/GT image pairs per run.",
    )
    parser.add_argument(
        "--render-sample-seed",
        default=0,
        type=int,
        help="Seed used to choose render samples reproducibly.",
    )
    args = parser.parse_args()
    if not args.matrix and (args.source is None or args.scene_name is None):
        parser.error("--source and --scene-name are required unless --matrix is provided")

    results_path = Path(args.results_csv)
    if not results_path.is_absolute():
        results_path = args.output_root / results_path
    existing_results = load_existing_results(results_path)
    records = []
    methods = matrix_methods(args)
    saliency_methods = matrix_saliency_methods(args)
    for method in methods:
        validate_method(method)
    for scene in scene_jobs(args):
        scene_name = scene["scene_name"]
        source = Path(scene["source"])
        dataset = scene.get("dataset", "")
        image_dir = scene.get("image_dir") or args.image_dir
        white_background = bool(scene.get("white_background", args.white_background))
        for method in methods:
            saliency_names = saliency_methods if saliency_method_required(method) else [None]
            for saliency_name in saliency_names:
                run_name = method_run_name(method, saliency_name)
                expected_key = expected_record_key(args, dataset, scene_name, method, run_name, saliency_name)
                if args.resume and args.csv_complete_resume and csv_record_complete(existing_results.get(expected_key)):
                    print(f"CSV resume: skipping completed {scene_name}/{run_name}")
                    continue

                model_dir = args.output_root / "models" / scene_name / run_name
                log_dir = args.output_root / "logs" / scene_name / run_name
                model_dir.mkdir(parents=True, exist_ok=True)
                log_dir.mkdir(parents=True, exist_ok=True)

                extra_flags = []
                if image_dir:
                    extra_flags.extend(["-i", image_dir])
                if white_background:
                    extra_flags.append("--white_background")
                train = train_command(
                    source,
                    model_dir,
                    method,
                    args.seed,
                    args.iterations,
                    saliency_name=saliency_name,
                    lambda_edge=args.lambda_edge,
                    lambda_saliency=args.lambda_saliency,
                    eggs_beta=args.eggs_beta,
                    edge_p=args.edge_p,
                    extra_flags=extra_flags,
                )
                render = render_command(source, model_dir, args.iterations, args.split, extra_flags=extra_flags)
                metrics = [sys.executable, "metrics.py", "-m", str(model_dir)]
                if args.region_metrics:
                    metrics.extend(
                        [
                            "--region_metrics",
                            "--saliency_name",
                            saliency_name or saliency_methods[0],
                            "--edge_region_fraction",
                            str(args.edge_region_fraction),
                            "--saliency_region_fraction",
                            str(args.saliency_region_fraction),
                        ]
                    )
                (log_dir / "commands.json").write_text(
                    json.dumps({"train": train, "render": render, "metrics": metrics}, indent=2),
                    encoding="utf-8",
                )

                train_status = run_logged(train, log_dir, "train", args.resume, lambda: valid_training(model_dir, args.iterations))
                run_logged(render, log_dir, "render", args.resume, lambda: valid_render(model_dir, args.iterations, args.split))
                run_logged(metrics, log_dir, "metrics", args.resume, lambda: valid_metrics(model_dir, args.iterations, args.split))

                values = load_metrics(model_dir, args.iterations, args.split)
                ply = model_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
                record = {
                    "dataset": dataset,
                    "scene": scene_name,
                    "method": method,
                    "run_name": run_name,
                    "saliency_name": saliency_name or "",
                    "normalization": "mean_one" if method.endswith("_norm") else "raw",
                    "lambda_edge": args.lambda_edge,
                    "lambda_saliency": args.lambda_saliency,
                    "eggs_beta": args.eggs_beta if args.eggs_beta is not None else args.lambda_edge,
                    "edge_p": args.edge_p,
                    "seed": args.seed,
                    "iteration": args.iterations,
                    "split": args.split,
                    "gaussian_count": count_ply_vertices(ply),
                    "model_size_bytes": ply.stat().st_size if ply.exists() else "",
                    "training_time_seconds": round(train_status.get("elapsed_seconds", 0.0), 3),
                    "model_path": str(model_dir.resolve()),
                    "log_path": str(log_dir.resolve()),
                }
                record.update({key: round(value, 6) if isinstance(value, float) else value for key, value in values.items()})
                if args.prune_after_metrics:
                    sample_seed = f"{args.render_sample_seed}:{scene_name}:{run_name}:{args.iterations}:{args.split}"
                    samples = keep_render_samples(model_dir, args.iterations, args.split, args.keep_render_samples, sample_seed)
                    removed = prune_after_metrics(model_dir, args.output_root, args.split)
                    record["kept_render_samples"] = json.dumps(samples)
                    record["pruned_artifacts"] = json.dumps(removed)
                records.append(record)

    if not records:
        print(f"No new results to write; kept {results_path}")
        return 0

    records = merge_existing_results(results_path, records)
    fieldnames = sorted({key for record in records for key in record.keys()})
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
