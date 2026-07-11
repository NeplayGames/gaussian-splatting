import torch
import torch.nn.functional as F
from .SaliencyAbstract import SaliencyAbs


class Itti(SaliencyAbs):
    def __init__(self, num_scales=6, device="cuda"):
        self.num_scales = num_scales
        self.device = device

    def gaussian_pyramid(self, img):
        """Build Gaussian pyramid more efficiently."""
        pyramid = [img]
        curr = img
        for _ in range(1, self.num_scales):
            curr = F.interpolate(curr, scale_factor=0.5, mode="area")  # efficient downsample
            up = F.interpolate(curr, size=img.shape[-2:], mode="bilinear", align_corners=False)
            pyramid.append(up)
        return torch.stack(pyramid, dim=1)  # [B, num_scales, 1, H, W]

    def compute_saliency(self, img):
        """Compute intensity-based saliency map (vectorized)."""
        B, C, H, W = img.shape
        gray = img.mean(dim=1, keepdim=True)  # [B,1,H,W]
        pyr = self.gaussian_pyramid(gray)     # [B,S,1,H,W]
        S = pyr.shape[1]

        # Center-surround differences (c in [2,3], s in [3,4])
        # You can vectorize instead of nested loops
        centers = pyr[:, 2:4]  # [B,2,1,H,W]
        surrounds = pyr[:, 3:5]  # [B,2,1,H,W]
        diff_maps = torch.abs(centers.unsqueeze(2) - surrounds.unsqueeze(1))  # [B,2,2,1,H,W]
        diff_maps = diff_maps.flatten(1, 3)  # [B,4,1,H,W]

        # Normalize all diff maps together
        min_v = diff_maps.amin(dim=(-2, -1), keepdim=True)
        max_v = diff_maps.amax(dim=(-2, -1), keepdim=True)
        diff_maps = (diff_maps - min_v) / (max_v - min_v + 1e-8)

        # Combine all differences
        saliency = diff_maps.mean(dim=1)  # [B,1,H,W]

        # Final normalization
        min_v, max_v = saliency.amin(dim=(-2, -1), keepdim=True), saliency.amax(dim=(-2, -1), keepdim=True)
        saliency = (saliency - min_v) / (max_v - min_v + 1e-8)

        return saliency

    def get_saliency_map(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.compute_saliency(image)

    def saliency_similarity(self, image, gt_image):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)

        sal_pred = self.compute_saliency(image)
        sal_gt = self.compute_saliency(gt_image)

        pred_flat = F.normalize(sal_pred.flatten(1), dim=1)
        gt_flat = F.normalize(sal_gt.flatten(1), dim=1)
        cos_sim = (pred_flat * gt_flat).sum(dim=1).mean()
        return cos_sim

    def saliency_loss(self, image, gt_image):
        cos_sim = self.saliency_similarity(image, gt_image)
        cos_sim_01 = 0.5 * (cos_sim + 1.0)
        return 1.0 - cos_sim_01
