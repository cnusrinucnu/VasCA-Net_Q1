"""
VasCA-Net: A vascular channel attention network for retinal vessel segmentation.

Reference:
  Ma, Z., Li, X., Zhao, Y., & Wang, H. (2026). VasCA-Net: A vascular channel
  attention network for retinal vessel segmentation. Expert Systems With
  Applications, 303, 130591. https://doi.org/10.1016/j.eswa.2025.130591

Architecture (Fig. 2A):
  Encoder: 4 stages, each EConv Block -> downsample (maxpool), channels
           C1 -> C2 -> C3 -> C4, spatial res H -> H/2 -> H/4 -> H/8.
  Bottleneck: plain conv block at H/16 resolution.
  Skip connections: the (pre-downsample) output of every encoder stage is
           passed through an MSCA block before being fused in the decoder.
  Decoder: 4 stages of DConv Blocks that upsample and fuse with the
           MSCA-refined skip connections, mirroring the encoder back up to
           the input resolution.
  Head: 1x1 conv to `num_classes` (1 for binary vessel/background).
"""
import torch
import torch.nn as nn

from .econv import EConvBlock
from .msca import MSCA
from .dconv import DConvBlock


class VasCANet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        msca_ratio: int = 8,
        econv_ratio: int = 2,
    ):
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        c5 = base_channels * 16  # bottleneck

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------- Encoder ----------------
        self.enc1 = EConvBlock(in_channels, c1, ratio=econv_ratio)
        self.enc2 = EConvBlock(c1, c2, ratio=econv_ratio)
        self.enc3 = EConvBlock(c2, c3, ratio=econv_ratio)
        self.enc4 = EConvBlock(c3, c4, ratio=econv_ratio)

        # ---------------- Bottleneck ----------------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5, c5, kernel_size=3, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        # ---------------- Skip-connection attention ----------------
        self.msca1 = MSCA(c1, ratio=msca_ratio)
        self.msca2 = MSCA(c2, ratio=msca_ratio)
        self.msca3 = MSCA(c3, ratio=msca_ratio)
        self.msca4 = MSCA(c4, ratio=msca_ratio)

        # ---------------- Decoder ----------------
        self.dec4 = DConvBlock(low_channels=c5, high_channels=c4, out_channels=c4)
        self.dec3 = DConvBlock(low_channels=c4, high_channels=c3, out_channels=c3)
        self.dec2 = DConvBlock(low_channels=c3, high_channels=c2, out_channels=c2)
        self.dec1 = DConvBlock(low_channels=c2, high_channels=c1, out_channels=c1)

        # ---------------- Output head ----------------
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)          # H   x W
        p1 = self.pool(e1)         # H/2 x W/2

        e2 = self.enc2(p1)         # H/2 x W/2
        p2 = self.pool(e2)         # H/4 x W/4

        e3 = self.enc3(p2)         # H/4 x W/4
        p3 = self.pool(e3)         # H/8 x W/8

        e4 = self.enc4(p3)         # H/8 x W/8
        p4 = self.pool(e4)         # H/16 x W/16

        b = self.bottleneck(p4)    # H/16 x W/16

        # Skip connections refined with multi-scale channel attention
        s1 = self.msca1(e1)
        s2 = self.msca2(e2)
        s3 = self.msca3(e3)
        s4 = self.msca4(e4)

        # Decoder
        d4 = self.dec4(b, s4)      # H/8 x W/8
        d3 = self.dec3(d4, s3)     # H/4 x W/4
        d2 = self.dec2(d3, s2)     # H/2 x W/2
        d1 = self.dec1(d2, s1)     # H   x W

        out = self.head(d1)
        return out


if __name__ == "__main__":
    model = VasCANet(in_channels=1, num_classes=1, base_channels=32)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Parameters: {n_params / 1e6:.2f}M")
