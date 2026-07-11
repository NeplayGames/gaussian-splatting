import subprocess
import os

base_cmd = ["python", "OriginalTraining.py", "-s", "train", "--eval"]

# Define sweeps
lambda_edges = [0.6]      # try small → large
lambda_saliencies = [0.4]

configs = []

#None (baseline)
# configs.append({"args": [], "suffix": "none"})

# #Edge only sweeps
# for le in lambda_edges:
#     configs.append({
#         "args": ["--use_edge",
#                  "--lambda_edge", str(le), "--lambda_saliency", "0.0"],
#         "suffix": f"edge_le{le}"
#     })

# # Saliency only sweeps
# for ls in lambda_saliencies:
#     configs.append({
#         "args": ["--use_saliency",
#                  "--lambda_edge", "0.0", "--lambda_saliency", str(ls)],
#         "suffix": f"saliency_ls{ls}"
#     })

# Edge + saliency sweeps
for le in lambda_edges:
    for ls in lambda_saliencies:
        configs.append({
            "args": ["--use_edge", "--use_saliency",
                     "--lambda_edge", str(le), "--lambda_saliency", str(ls)],
            "suffix": f"edge_saliency_le{le}_ls{ls}"
        })

#Saliency methods to test
saliency_names = ["itti"]

# Run all configs with both saliency methods
for saliency_name in saliency_names:
    for cfg in configs:
        run_name = f"{cfg['suffix']}_salname_{saliency_name}"
        out_dir = os.path.join("output", run_name)

        print(f"\n=== Running: {run_name} ===\n")
        cmd = base_cmd + cfg["args"] + [
            "--model_path", out_dir,
            "--saliency_name", saliency_name
        ]
        subprocess.run(cmd)
