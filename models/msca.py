"""
Multi-Scale Channel Attention (MSCA) module.

Implements Fig. 3 / Eqs. (1)-(6) of:
  "VasCA-Net: A vascular channel attention network for retinal vessel
   segmentation", Ma et al., Expert Systems With Applications, 2026.

Given F in R^{H x W x C}:
  1. Two pooled descriptors are computed: AvgPool(F) and MaxPool(F), each 1x1xC.
  2. Each pooled descriptor is passed through THREE parallel convolutions with
     kernel sizes 1x1, 3x3, 5x5 (channels reduced by `ratio`), producing
     F_conv1, F_conv3, F_conv5 for each pooling branch.
  3. The three kernel-scale outputs are summed for each branch, and the two
     branch sums are summed together -> F_comb.
  4. F_comb -> 1x1 conv (back to C channels) -> sigmoid -> attention map M.
  5. F_att = F * M (element-wise).
  6. F_drop = Dropout(F_att, p=0.5).
  7. F' = F_drop + F  (residual).
"""
import torch
import torch.nn as nn


class _MultiKernelBranch(nn.Module):
    """Applies 1x1, 3x3 and 5x5 convs to a pooled (1x1xC) descriptor and sums them."""

    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        reduced = max(channels // ratio, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        # Project the reduced representation back up so it can be summed
        # with the other branch and eventually mapped back to C channels.
        self.expand = nn.Conv2d(reduced, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x) + self.conv3(x) + self.conv5(x)
        return self.expand(out)


class MSCA(nn.Module):
    """Multi-Scale Channel Attention block used at every skip connection."""

    def __init__(self, channels: int, ratio: int = 8, dropout: float = 0.5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.avg_branch = _MultiKernelBranch(channels, ratio)
        self.max_branch = _MultiKernelBranch(channels, ratio)

        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_desc = self.avg_pool(x)
        max_desc = self.max_pool(x)

        f_comb = self.avg_branch(avg_desc) + self.max_branch(max_desc)
        attn_map = self.sigmoid(self.fuse(f_comb))  # (B, C, 1, 1)

        f_att = x * attn_map
        f_drop = self.dropout(f_att)
        return f_drop + x
