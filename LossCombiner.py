# LossCombiner.py
import torch
from utils.loss_utils import ssim


def _as_batched_image(image):
    """Return image as [B, C, H, W]."""
    if image.dim() == 3:
        return image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"Expected image with 3 or 4 dimensions, got {image.dim()}.")
    return image


def weighted_l1_loss(pred, gt, weight=None):
    """
    Compute mean(weight * e), where e is the per-pixel mean absolute RGB error.

    pred, gt: [B, C, H, W]
    weight: optional [B, 1, H, W] spatial weighting map
    """
    per_pixel_error = torch.abs(pred - gt).mean(dim=1, keepdim=True)
    if weight is not None:
        per_pixel_error = per_pixel_error * weight
    return per_pixel_error.mean()


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
        lam_dssim=0.2,
        constant_scaling_control=False,
    ):
        self.edge_cls = edge_cls
        self.saliency_cls = saliency_cls
        self.use_edge = use_edge
        self.use_saliency = use_saliency
        self.lambda_edge = lambda_edge
        self.lambda_saliency = lambda_saliency
        self.normalize = normalize
        self.lam_dssim = lam_dssim
        self.constant_scaling_control = constant_scaling_control

    def _weight_mean(self, weight):
        """Return the per-image spatial mean of a completed weight map."""
        reduce_dims = tuple(range(1, weight.dim()))
        return weight.mean(dim=reduce_dims, keepdim=True).detach()

    def _normalize_weight(self, weight):
        """Normalize a completed spatial weight map to mean one."""
        if not self.normalize:
            return weight
        mean = self._weight_mean(weight)
        return weight / (mean + 1e-8)

    def _edge_indicator(self, gt_image):
        """Return raw E(u,v), not a completed Sobel phi/weight map."""
        if not self.use_edge or self.edge_cls is None:
            return None

        # The Sobel implementation exposes primitives for the raw gradient map;
        # prefer them over get_edge_map(), which returns 1 + beta * edge.
        if hasattr(self.edge_cls, "rgb_to_grayscale") and hasattr(self.edge_cls, "sobel_filter"):
            return self.edge_cls.sobel_filter(self.edge_cls.rgb_to_grayscale(gt_image))

        return self.edge_cls.get_edge_map(gt_image)

    def _saliency_indicator(self, gt_image):
        """Return raw S(u,v)."""
        if not self.use_saliency or self.saliency_cls is None:
            return None
        return self.saliency_cls.get_saliency_map(gt_image)

    def compute_raw_weight(self, gt_image):
        """
        Build the unnormalized SEGS spatial weight map:
            w(u,v) = 1 + alpha E(u,v) + beta S(u,v)

        Output shape: [B, 1, H, W].
        """
        gt_image = _as_batched_image(gt_image)
        batch_size, _, height, width = gt_image.shape
        weight = torch.ones((batch_size, 1, height, width), device=gt_image.device, dtype=gt_image.dtype)

        edge_map = self._edge_indicator(gt_image)
        if edge_map is not None:
            weight = weight + self.lambda_edge * edge_map.to(device=gt_image.device, dtype=gt_image.dtype)

        saliency_map = self._saliency_indicator(gt_image)
        if saliency_map is not None:
            weight = weight + self.lambda_saliency * saliency_map.to(device=gt_image.device, dtype=gt_image.dtype)

        return weight

    def compute_phi(self, gt_image):
        """
        Build the normalized SEGS spatial weight map:
            w(u,v) = 1 + alpha E(u,v) + beta S(u,v)
            w_hat(u,v) = w(u,v) / (mean(w) + eps)

        Output shape: [B, 1, H, W] with per-image spatial mean equal to one.
        """
        return self._normalize_weight(self.compute_raw_weight(gt_image))

    def compute_loss(self, pred, gt):
        """
        Compute the SEGS loss:
            e(u,v) = (1 / C) * sum_c |pred_c(u,v) - gt_c(u,v)|
            L_SEGS = (1 - lambda) * mean(w_hat * e) + lambda * L_DSSIM
        """
        pred = _as_batched_image(pred)
        gt = _as_batched_image(gt)

        if self.use_edge or self.use_saliency:
            raw_weight = self.compute_raw_weight(gt)
            if self.constant_scaling_control:
                # Control: L_control = c * L_3DGS, where c is the mean of the
                # unnormalized map. This matches the gradient scale increase
                # without changing where image-space gradients are directed.
                phi = self._weight_mean(raw_weight)
            else:
                # Main experiment: normalize the map so weights redistribute
                # optimization effort without changing the mean gradient scale.
                phi = self._normalize_weight(raw_weight)
        else:
            phi = None

        l1_part = weighted_l1_loss(pred, gt, phi)

        if self.lam_dssim == 0:
            return l1_part

        dssim_part = 1.0 - ssim(pred, gt)
        return (1.0 - self.lam_dssim) * l1_part + self.lam_dssim * dssim_part

    def Get_Edge_saliency_Similarity(self, pred, gt_image):
        return self.edge_cls.edge_similarity(pred, gt_image), self.saliency_cls.saliency_similarity(pred, gt_image)
