import os
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from EdgesMethods.Sobel import Sobel
from SaliencyMethods.BooleanMap import BooleanMap

# Folder path
folder_path = r"truck\images"

sobel = Sobel()
bms = BooleanMap()

# Pick 5 random images
all_images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
chosen_images = random.sample(all_images, 5)

# Target size for all images
TARGET_SIZE = (128, 128)

transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor()
])

plt.figure(figsize=(15, 10))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i, fname in enumerate(chosen_images):
    img_path = os.path.join(folder_path, fname)

    # Load + resize
    img = Image.open(img_path).convert("RGB")
    img_resized = transform(img)  # [3,128,128]
    img_tensor = img_resized.unsqueeze(0).to(device)  # [1,3,128,128]

    # === Sobel edges ===
    phi_map = sobel.phi_map(img_tensor)  # may be different size

    # Resize Sobel output to match 128x128
    phi_map_resized = F.interpolate(phi_map, size=TARGET_SIZE, mode="bilinear", align_corners=False)

    # === Boolean Map Saliency ===
    sal_map = bms.compute_bms(img_tensor)

    # Resize BMS map too
    sal_map_resized = F.interpolate(sal_map, size=TARGET_SIZE, mode="bilinear", align_corners=False)

    # --- Plot ---
    # Original
    plt.subplot(3, 5, i + 1)
    plt.imshow(img_resized.permute(1, 2, 0).cpu())  # tensor → image
    plt.title(f"Original {i+1}")
    plt.axis("off")

    # Sobel φ map
    plt.subplot(3, 5, i + 6)
    plt.imshow(phi_map_resized[0, 0].detach().cpu().numpy(), cmap="gray")
    plt.title(f"Sobel Edges {i+1}")
    plt.axis("off")

    # Boolean Map Saliency
    plt.subplot(3, 5, i + 11)
    plt.imshow(sal_map_resized[0, 0].detach().cpu().numpy(), cmap="hot")
    plt.title(f"BMS Saliency {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
