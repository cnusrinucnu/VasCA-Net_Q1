"""
EConv Block (encoder module).

Implements Fig. 2(B) of VasCA-Net:
  Given input X (H x W x C):
    - branch_1x1 = Conv1x1(X)                     -> H x W x (Cout/ratio)
    - branch_3x3 = Conv3x3(Conv3x3(X))             -> H x W x (Cout/ratio)
    - branch_gap = Conv1x1(GAP(X))                 -> 1 x 1 x (Cout/ratio)
    - fused      = branch_3x3 * branch_gap (broadcast mult.)
    - out        = Concat(branch_1x1, fused)       -> H x W x (2*Cout/ratio)

A final 1x1 conv (`proj`) maps the concatenated features to the desired
`out_channels`, and a strided/pooled downsample is applied afterwards by
the caller (see VasCA-Net encoder stage) so this block can also be reused
at the bottleneck without downsampling.
"""
import torch
import torch.nn as nn


class EConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio: int = 2):
        super().__init__()
        reduced = max(out_channels // ratio, 1)

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
        )

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, reduced, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.branch_gap = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(reduced * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_1x1(x)
        b3 = self.branch_3x3(x)
        bg = self.branch_gap(self.gap(x))  # (B, reduced, 1, 1)

        fused = b3 * bg  # broadcast multiply
        out = torch.cat([b1, fused], dim=1)
        return self.proj(out)
