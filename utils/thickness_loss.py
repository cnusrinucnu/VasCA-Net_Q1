import torch
import torch.nn as nn
import torch.nn.functional as F


class ThicknessAwareBCELoss(nn.Module):
    """
    Thickness-aware BCE loss.

    Thin vessel pixels receive larger weights than thick vessel pixels.
    Background pixels retain weight 1.

    Input:
        logits: [B, 1, H, W]
        target: [B, 1, H, W], values in {0, 1}
    """

    def __init__(
        self,
        alpha: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ):

        # --------------------------------------------------
        # Estimate vessel thickness using differentiable
        # max-pooling approximation.
        #
        # For this first experiment we use local vessel
        # density as a stable approximation of thickness.
        # --------------------------------------------------

        vessel = target.float()

        local_density = F.avg_pool2d(
            vessel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # Normalize local vessel density.
        d_min = local_density.amin(
            dim=(-2, -1),
            keepdim=True,
        )

        d_max = local_density.amax(
            dim=(-2, -1),
            keepdim=True,
        )

        d_norm = (
            local_density - d_min
        ) / (
            d_max - d_min + self.eps
        )

        # Thin/sparse vessel regions receive larger weight.
        vessel_weight = (
            1.0
            + self.alpha * (1.0 - d_norm)
        )

        # Background remains weight 1.
        weight = torch.ones_like(target)

        weight = torch.where(
            target > 0.5,
            vessel_weight,
            weight,
        )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        weighted_bce = (
            bce * weight
        ).mean()

        return weighted_bce
