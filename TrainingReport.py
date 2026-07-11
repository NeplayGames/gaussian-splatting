import os

import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - budget logging still works without torch
    torch = None


def fmt(v, precision=4):
    if v is None:
        return "NA"
    if hasattr(v, "item"):  # torch tensor -> scalar
        v = v.item()
    return f"{v:.{precision}f}"


def collect_optimization_budget(model_path, gaussians=None, render_fps=None):
    """Collect cost metadata for fixed-budget SEGS++ comparisons."""
    point_cloud_path = os.path.join(model_path, "point_cloud", "iteration_*", "point_cloud.ply")
    model_file_size_bytes = None
    try:
        import glob
        candidates = glob.glob(point_cloud_path)
        if candidates:
            latest = max(candidates, key=os.path.getmtime)
            model_file_size_bytes = os.path.getsize(latest)
    except OSError:
        model_file_size_bytes = None

    gaussian_count = None
    if gaussians is not None and hasattr(gaussians, "get_xyz"):
        gaussian_count = int(gaussians.get_xyz.shape[0])

    peak_gpu_memory_bytes = None
    if torch is not None and torch.cuda.is_available():
        peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())

    return {
        "gaussian_count": gaussian_count,
        "model_file_size_bytes": model_file_size_bytes,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_gpu_memory_allocated_bytes": peak_gpu_memory_bytes,
        "point_cloud_file_size_bytes": model_file_size_bytes,
        "render_fps": render_fps,
        "render_fps_recorded_during_training": render_fps,
    }


def log_metrics_to_excel(iteration, metrics, model_path,
                         lambda_edge=0.0, lambda_saliency=0.0,
                            use_edge=False, use_saliency=False,
                         excel_path="results.xlsx", use_method = "Saliency",total_Time = 0,
                         budget=None):
    """
    Append metrics + config info to Excel file, create if not exists.

    Args:
        iteration (int): iteration number
        metrics (dict): {"psnr":..., "ssim":..., "ms_ssim":..., 
                         "lpips":..., "edge_sim":..., "saliency_sim":...}
        model_path (str): path to model checkpoint (for run name)
        lambda_edge (float): lambda for edge
        lambda_saliency (float): lambda for saliency
        mode (str): "add" or "mul"
        use_edge (bool): True if edge was used
        use_saliency (bool): True if saliency was used
        excel_path (str): Excel file path
    """
    run_name = os.path.basename(model_path)

    # Fill missing metrics with None if not provided
    record = {
        "run": run_name,
        "iteration": iteration,
        "psnr": metrics.get("psnr", None),
        "ssim": metrics.get("ssim", None),
        "ms_ssim": metrics.get("ms_ssim", None),
        "lpips": metrics.get("lpips", None),
        "edge_sim": metrics.get("edge_sim", None),
        "saliency_sim": metrics.get("saliency_sim", None),
        "lambda_edge": lambda_edge,
        "lambda_saliency": lambda_saliency,
        "use_edge": use_edge,
        "use_saliency": use_saliency,
        "Saliency_name":use_method,
        "Total_Time": total_Time,
        "gaussian_count": None,
        "model_file_size_bytes": None,
        "peak_gpu_memory_bytes": None,
        "render_fps": None
    }
    if budget:
        record.update({key: budget.get(key) for key in (
            "gaussian_count",
            "model_file_size_bytes",
            "peak_gpu_memory_bytes",
            "render_fps",
        )})

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_excel(excel_path, index=False)

    print(
        f"[LOG] Saved metrics to {excel_path}: run={run_name}, iter={iteration}, "
        f"PSNR={fmt(record['psnr'])}, "
        f"SSIM={fmt(record['ssim'])}, "
        f"MS-SSIM={fmt(record['ms_ssim'])}, "
        f"LPIPS={fmt(record['lpips'])}, "
        f"EdgeSim={fmt(record['edge_sim'])}, "
        f"SalSim={fmt(record['saliency_sim'])}, "
        f"λe={lambda_edge}, λs={lambda_saliency}, edge={use_edge}, saliency={use_saliency}, "
        f"gaussians={record['gaussian_count']}, "
        f"model_bytes={record['model_file_size_bytes']}, "
        f"peak_gpu_bytes={record['peak_gpu_memory_bytes']}, "
        f"render_fps={record['render_fps']}"
    )

