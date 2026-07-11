import pandas as pd
import re

def parse_tensor(x):
    """Extract float from tensor(...) string, safely."""
    if isinstance(x, str):
        import re
        m = re.search(r"tensor\(\s*([0-9.+-eE]+)", x)
        if m:
            val = m.group(1).replace(",", "").strip()
            return float(val)
    try:
        return float(str(x).replace(",", "").strip())
    except:
        return None


# Read Excel instead of CSV
df = pd.read_excel("results.xlsx")  # no sep needed

# Clean numeric columns
df["psnr"] = df["psnr"].apply(parse_tensor)
numeric_cols = ["ssim", "ms_ssim", "lpips", "edge_sim", "saliency_sim"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Define maximize/minimize metrics
maximize = ["psnr", "ssim", "ms_ssim", "edge_sim", "saliency_sim"]
minimize = ["lpips"]

# Find best runs for each metric
best_results = []
for metric in maximize:
    best_row = df.loc[df[metric].idxmax()]
    best_results.append({"metric": metric, "best_value": best_row[metric], "run": best_row["run"]})
for metric in minimize:
    best_row = df.loc[df[metric].idxmin()]
    best_results.append({"metric": metric, "best_value": best_row[metric], "run": best_row["run"]})

best_df = pd.DataFrame(best_results)
print(best_df)
