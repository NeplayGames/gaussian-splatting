import torch
import torch.nn as nn
import torch.nn.functional as F
from .SaliencyAbstract import SaliencyAbs

class BooleanMapApprox(SaliencyAbs, nn.Module):
    def __init__(self, num_thresholds=8, device="cuda"):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.register_buffer("thresholds", torch.linspace(0, 1, num_thresholds + 2, device=device)[1:-1])

    def normalize_map(self, fmap):
        min_v = fmap.amin(dim=(-2, -1), keepdim=True)
        max_v = fmap.amax(dim=(-2, -1), keepdim=True)
        return (fmap - min_v) / (max_v - min_v + 1e-8)

    def compute_bms(self, img):
        B, C, H, W = img.shape
        gray = img.mean(dim=1, keepdim=True)  # [B,1,H,W]

        saliency_sum = torch.zeros((B, 1, H, W), device=img.device)
        for t in self.thresholds:
            bool_map = (gray > t).float()  # [B,1,H,W]

            # extract borders
            top    = bool_map[:, :, 0, :]     # [B,1,W]
            bottom = bool_map[:, :, -1, :]    # [B,1,W]
            left   = bool_map[:, :, :, 0]     # [B,1,H]
            right  = bool_map[:, :, :, -1]    # [B,1,H]

            # compute border mean intensity
            border_mean = torch.cat([
                top.flatten(2),
                bottom.flatten(2),
                left.flatten(2),
                right.flatten(2)
            ], dim=-1).mean(dim=-1, keepdim=True)  # [B,1,1]

            # broadcast to [B,1,H,W]
            border_mean = border_mean.view(B, 1, 1, 1)

            # accumulate difference
            saliency_sum += torch.abs(bool_map - border_mean)

        saliency = saliency_sum / len(self.thresholds)
        return self.normalize_map(saliency)

    def get_saliency_map(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.compute_bms(image)

    def saliency_similarity(self, image, gt_image):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)

        sal_pred = self.compute_bms(image)
        sal_gt = self.compute_bms(gt_image)

        dot = (sal_pred * sal_gt).sum(dim=[1,2,3])
        norm_pred = sal_pred.norm(p=2, dim=[1,2,3])
        norm_gt = sal_gt.norm(p=2, dim=[1,2,3])
        cos_sim = dot / (norm_pred * norm_gt + 1e-8)
        return cos_sim.mean()

    def saliency_loss(self, image, gt_image):
        return (1.0 - self.saliency_similarity(image, gt_image)) * 0.5


# Backward-compatible alias for existing checkpoints/scripts.
BooleanMap = BooleanMapApprox
