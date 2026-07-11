import os
import pandas as pd
def fmt(v, precision=4):
    if v is None:
        return "NA"
    if hasattr(v, "item"):  # torch tensor -> scalar
        v = v.item()
    return f"{v:.{precision}f}"

def log_metrics_to_excel(iteration, metrics, model_path,
                         lambda_edge=0.0, lambda_saliency=0.0,
                            use_edge=False, use_saliency=False,
                         excel_path="results.xlsx", use_method = "Saliency",total_Time = 0):
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
        "Total_Time": total_Time
    }

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
        f"λe={lambda_edge}, λs={lambda_saliency}, edge={use_edge}, saliency={use_saliency}"
    )

