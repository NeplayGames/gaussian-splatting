import torch
from .EdgeAbstract import EdgeAbs
import torch.nn.functional as F

class Sobel(EdgeAbs):
    def __init__(self, beta=2.0, p=1, normalize=True, clamp_min=0.5, clamp_max=5.0):
        super().__init__()
        self.beta = beta
        self.p = p
        self.normalize = normalize
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    def get_edge_map(self, target):
        """
        Wrapper to match LossCombiner expected interface.
        Returns (B,1,H,W) edge intensity map.
        """
        return self.phi_map(target)  # Or grad_mag only if you don't want +1+beta

    def rgb_to_grayscale(self, img):
        """Convert [B,3,H,W] or [3,H,W] to grayscale [B,1,H,W]."""
        if img.ndim == 3:  # [3, H, W]
            img = img.unsqueeze(0)  # → [1,3,H,W]
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def sobel_filter(self, gray):
        """Apply Sobel to grayscale [B,1,H,W] and return gradient magnitude [B,1,H,W]."""
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32, device=gray.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32, device=gray.device).unsqueeze(0).unsqueeze(0)

        grad_x = F.conv2d(F.pad(gray, (1,1,1,1), mode="reflect"), sobel_x)
        grad_y = F.conv2d(F.pad(gray, (1,1,1,1), mode="reflect"), sobel_y)

        if self.p == 1:
            grad_mag = grad_x.abs() + grad_y.abs()
        elif self.p == 2:
            grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        else:
            grad_mag = (grad_x.abs().pow(self.p) + grad_y.abs().pow(self.p)).pow(1.0 / self.p)

        return grad_mag  # [B,1,H,W]

    def phi_map(self, target):
        """
        Compute edge indicator phi from target image.
        target: [B,3,H,W] or [3,H,W]
        returns: phi [B,1,H,W]
        """
        gray = self.rgb_to_grayscale(target)  # [B,1,H,W]
        grad_mag = self.sobel_filter(gray)

        if self.normalize:
            mean_val = grad_mag.mean().detach()
            grad_mag = grad_mag / (mean_val + 1e-6)

        phi = 1.0 + self.beta * grad_mag
        phi = phi.clamp(min=self.clamp_min, max=self.clamp_max)
        return phi  # [B,1,H,W]

    def edge_loss(self, pred, target):
        """
        Apply phi weighting to L1 loss.
        pred, target: [B,3,H,W]
        returns: scalar weighted L1 loss
        """
        phi = self.phi_map(target)  # [B,1,H,W]
        abs_diff = (pred - target).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        weighted = phi * abs_diff
        return weighted.mean()

    def edge_similarity(self, pred, target):
        """
        Compute edge similarity (cosine) between pred and target.
        pred, target: [B,3,H,W]
        returns: scalar similarity in [-1, 1]
        """
        # Get edge maps using your phi_map
        pred_edges = self.phi_map(pred)   # [B,1,H,W]
        target_edges = self.phi_map(target)  # [B,1,H,W]

        # Flatten to vectors
        pred_vec = pred_edges.view(pred_edges.size(0), -1)   # [B, H*W]
        target_vec = target_edges.view(target_edges.size(0), -1)

        # Cosine similarity (batchwise)
        cos_sim = F.cosine_similarity(pred_vec, target_vec, dim=1)  # [B]

        return 0.5 * (cos_sim.mean() + 1.0) # scalar in [0, 1]