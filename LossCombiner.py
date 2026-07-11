# LossCombiner.py
import torch
from utils.loss_utils import ssim, l1_loss

def weighted_l1_loss(pred, gt, phi=None):
    """
    pred, gt: (B, C, H, W)
    phi: optional per-pixel weighting map (B, 1, H, W) or (B, H, W)
    """
    diff = torch.abs(pred - gt)
    if phi is not None:
        if phi.ndim == 3:
            phi = phi.unsqueeze(1)  # (B, 1, H, W)
        diff = diff * phi
    return diff.mean()


class LossCombiner:
    def __init__(
        self,
        edge_cls=None,
        saliency_cls=None,
        use_edge=False,
        use_saliency=False,
        lambda_edge=0.2,
        lambda_saliency=0.1,
        normalize=True,
        lam_dssim=0.2  # λ from original GS loss
    ):
        self.edge_cls = edge_cls
        self.saliency_cls = saliency_cls
        self.use_edge = use_edge
        self.use_saliency = use_saliency
        self.lambda_edge = lambda_edge
        self.lambda_saliency = lambda_saliency
        self.normalize = normalize
        self.lam_dssim = lam_dssim  # weight for DSSIM

    def _normalize_map(self, x):
        if x is None:
            return None
        if self.normalize:
            return x / (x.mean().detach() + 1e-8)
        return x

    def compute_phi(self, gt_image):
        """
        Build φ(u,v) = 1 + β1 * edge + β2 * saliency
        Output shape: (B, 1, H, W)
        """
        device = gt_image.device
        batch_size, _, H, W = gt_image.shape

        phi = torch.ones((batch_size, 1, H, W), device=device)

        if self.use_edge and self.edge_cls is not None:
            edge_map = self.edge_cls.get_edge_map(gt_image)  # (B, 1, H, W)
            edge_map = self._normalize_map(edge_map)
            phi += self.lambda_edge * edge_map

        if self.use_saliency and self.saliency_cls is not None:
            sal_map = self.saliency_cls.get_saliency_map(gt_image)  # (B, 1, H, W)
            sal_map = self._normalize_map(sal_map)
            phi += self.lambda_saliency * sal_map

        return phi

    def compute_loss(self, pred, gt):
        """
        Loss(c, im) = (1 - λ) * || φ(u,v)(c - im)
        """
        if not self.use_edge and not self.use_edge:
            return l1_loss(pred, gt)
        # Ensure pred and gt are 4D [B,C,H,W]
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)  # [1,C,H,W]
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)
            # Compute φ map
            phi = self.compute_phi(gt)

        # Weighted L1 part
        l1_part = weighted_l1_loss(pred, gt, phi)
        
        total = (1 - self.lam_dssim) * l1_part 
        return total

    def Get_Edge_saliency_Similarity(self, pred, gt_image):
        return self.edge_cls.edge_similarity(pred, gt_image), self.saliency_cls.saliency_similarity(pred, gt_image)
