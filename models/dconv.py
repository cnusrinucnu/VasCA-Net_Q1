"""
DConv Block (decoder module).

Implements Fig. 2(C) / Eqs. (7)-(11):
  A1' = UpSample(A1)                 # low-level feature map, upsampled
  F_concat = Concat(A1', A2)         # A2: high-level feature map (same res.)
  F_conv1  = Conv3x3(F_concat)
  F_conv2  = Conv3x3(F_concat)       # a second, independent 3x3 conv
  F_add    = F_conv1 + F_conv2
  F_out    = ReLU(F_add)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DConvBlock(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        concat_channels = low_channels + high_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        """
        low_feat:  lower-resolution, higher-level semantic feature map (A1)
        high_feat: higher-resolution, lower-level detail feature map (A2),
                   typically arriving via a skip connection (optionally
                   refined by MSCA before being passed in here).
        """
        low_up = self.up(low_feat)

        # Guard against off-by-one size mismatches from odd input dims.
        if low_up.shape[-2:] != high_feat.shape[-2:]:
            low_up = F.interpolate(low_up, size=high_feat.shape[-2:],
                                    mode="bilinear", align_corners=True)

        concat = torch.cat([low_up, high_feat], dim=1)
        out = self.conv1(concat) + self.conv2(concat)
        return self.relu(out)
