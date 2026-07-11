import cv2
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from arguments import OptimizationParams
import pandas as pd
import os
# Assuming argparse is being used for parsing
import argparse
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo

# Function to calculate PSNR, SSIM, and MSE for pairs of images
def calculate_metrics(original_image, generated_image):
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

    psnr_value = psnr(original_image, generated_image)
    ssim_value = ssim(original_gray, generated_gray)
    mse_value = mse(original_image, generated_image)
    
    return psnr_value, ssim_value, mse_value

# Create an argument parser instance
parser = argparse.ArgumentParser()

# Instantiate the OptimizationParams class
opt_params = OptimizationParams(parser)

# Now, you can access its attributes
print(opt_params.iterations)  # Output: 7000
# Folders containing images
original_folder = "C:/Thesis/output/main/test/ours_"+ str(opt_params.iterations) + "/gt/"  # Folder with original images
generated1_folder = "C:/Thesis/output/main/test/ours_"+ str(opt_params.iterations) + "/renders/"  # Folder with first set of generated images
generated2_folder = "C:/Thesis/output/Edge_Added/test/ours_"+ str(opt_params.compare_iter) + "/renders/"
# Initialize variables to accumulate metrics
psnr_generated1 = []
ssim_generated1 = []
mse_generated1 = []

psnr_generated2 = []
ssim_generated2 = []
mse_generated2 = []
# Loop through all image files in the folders
for filename in os.listdir(original_folder):
    if filename.endswith(".png"):  # Ensure we're only processing PNG images
        original_path = os.path.join(original_folder, filename)
        generated1_path = os.path.join(generated1_folder, filename)
        generated2_path = os.path.join(generated2_folder, filename)

        # Load images
        original_image = cv2.imread(original_path)
        generated1_image = cv2.imread(generated1_path)
        generated2_image = cv2.imread(generated2_path)

        # Calculate metrics for both generated images
        psnr1, ssim1, mse1 = calculate_metrics(original_image, generated1_image)
        psnr2, ssim2, mse2 = calculate_metrics(original_image, generated2_image)

         # Accumulate metrics for overall comparison
        psnr_generated1.append(psnr1)
        ssim_generated1.append(ssim1)
        mse_generated1.append(mse1)

        psnr_generated2.append(psnr2)
        ssim_generated2.append(ssim2)
        mse_generated2.append(mse2)
        # Print results for this image pair
        # print(f"Results for {filename}:")
        # print(f"  Generated Image 1 -> PSNR: {psnr1:.2f}, SSIM: {ssim1:.4f}, MSE: {mse1:.2f}")
        # print(f"  Generated Image 2 -> PSNR: {psnr2:.2f}, SSIM: {ssim2:.4f}, MSE: {mse2:.2f}")
        # print("-" * 50)

avg_psnr1 = np.mean(psnr_generated1)
avg_ssim1 = np.mean(ssim_generated1)
avg_mse1 = np.mean(mse_generated1)

avg_psnr2 = np.mean(psnr_generated2)
avg_ssim2 = np.mean(ssim_generated2)
avg_mse2 = np.mean(mse_generated2)



# Define the results
data = {
    "Without Edge Iteration, With Edge Iteration" : [str(opt_params.iterations) + ", " + str(opt_params.compare_iter)],
    "Avg PSNR Image 1": [avg_psnr1],
    "Avg SSIM Image 1": [avg_ssim1],
    "Avg MSE Image 1": [avg_mse1],
    "Avg PSNR Image 2": [avg_psnr2],
    "Avg SSIM Image 2": [avg_ssim2],
    "Avg MSE Image 2": [avg_mse2],
    "Better Image": ["Image 1" if avg_psnr1 > avg_psnr2 and avg_ssim1 > avg_ssim2 and avg_mse1 < avg_mse2 
                     else "Image 2" if avg_psnr2 > avg_psnr1 and avg_ssim2 > avg_ssim1 and avg_mse2 < avg_mse1 
                     else "Similar"]
}

df_new = pd.DataFrame(data)

# File name
file_name = "image_comparison.xlsx"

# Check if file exists
if os.path.exists(file_name):
    # Load existing data
    with pd.ExcelFile(file_name) as xls:
        df_existing = pd.read_excel(xls, sheet_name="Metrics")

    # Append new data
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Save updated data
    with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
        # Write intro text in a separate sheet
        # Write metrics data
        df_combined.to_excel(writer, sheet_name="Metrics", index=False)
else:
    # Create a new Excel file with the intro text
    with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
        # Write intro text in a separate sheet
        # Write metrics data
        df_new.to_excel(writer, sheet_name="Metrics", index=False)

print(f"Results saved to {file_name}")


