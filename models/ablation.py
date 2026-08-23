"""
Ablation-friendly variant of VasCA-Net, letting you toggle each of the three
proposed modules independently -- mirrors Table 1 in the paper:

    Base            : plain U-Net-style conv blocks, no EConv/DConv/MSCA
    Base + A        : + EConv encoder blocks
    Base + B        : + DConv decoder blocks
    Base + C        : + MSCA attention on skip connections
    Base + A + B    : EConv + DConv
    Base + A + B + C: full VasCA-Net

Use `use_econv`, `use_dconv`, `use_msca` flags to reproduce any row of
Table 1 for your own ablation runs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .econv import EConvBlock
from .dconv import DConvBlock
from .msca import MSCA


class _PlainConvBlock(nn.Module):
    """Two 3x3 convs -- the 'Base' (vanilla U-Net) encoder building block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _PlainDecoderBlock(nn.Module):
    """Upsample + concat + single 3x3 conv -- the 'Base' decoder building block."""

    def __init__(self, low_channels: int, high_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(low_channels + high_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, low_feat, high_feat):
        low_up = self.up(low_feat)
        if low_up.shape[-2:] != high_feat.shape[-2:]:
            low_up = F.interpolate(low_up, size=high_feat.shape[-2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([low_up, high_feat], dim=1))


class _Identity(nn.Module):
    def forward(self, x):
        return x


class VasCANetAblation(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        use_econv: bool = True,
        use_dconv: bool = True,
        use_msca: bool = True,
        msca_ratio: int = 8,
        econv_ratio: int = 2,
    ):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        c5 = base_channels * 16

        self.pool = nn.MaxPool2d(2, 2)

        enc = (lambda cin, cout: EConvBlock(cin, cout, ratio=econv_ratio)) if use_econv else _PlainConvBlock
        self.enc1 = enc(in_channels, c1)
        self.enc2 = enc(c1, c2)
        self.enc3 = enc(c2, c3)
        self.enc4 = enc(c3, c4)

        self.bottleneck = _PlainConvBlock(c4, c5)

        if use_msca:
            self.msca1, self.msca2, self.msca3, self.msca4 = (
                MSCA(c1, ratio=msca_ratio), MSCA(c2, ratio=msca_ratio),
                MSCA(c3, ratio=msca_ratio), MSCA(c4, ratio=msca_ratio),
            )
        else:
            self.msca1 = self.msca2 = self.msca3 = self.msca4 = _Identity()

        dec = DConvBlock if use_dconv else _PlainDecoderBlock
        self.dec4 = dec(c5, c4, c4)
        self.dec3 = dec(c4, c3, c3)
        self.dec2 = dec(c3, c2, c2)
        self.dec1 = dec(c2, c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        e4 = self.enc4(p3); p4 = self.pool(e4)

        b = self.bottleneck(p4)

        s1, s2, s3, s4 = self.msca1(e1), self.msca2(e2), self.msca3(e3), self.msca4(e4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.head(d1)


CONFIGS = {
    "base": dict(use_econv=False, use_dconv=False, use_msca=False),
    "base+A": dict(use_econv=True, use_dconv=False, use_msca=False),
    "base+B": dict(use_econv=False, use_dconv=True, use_msca=False),
    "base+C": dict(use_econv=False, use_dconv=False, use_msca=True),
    "base+A+B": dict(use_econv=True, use_dconv=True, use_msca=False),
    "base+B+C": dict(use_econv=False, use_dconv=True, use_msca=True),
    "base+A+C": dict(use_econv=True, use_dconv=False, use_msca=True),
    "base+A+B+C": dict(use_econv=True, use_dconv=True, use_msca=True),
}
