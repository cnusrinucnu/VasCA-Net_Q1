"""
VasCA-Net-DS: VasCA-Net augmented with deep supervision.

Adds an auxiliary 1x1-conv segmentation head after each DConv decoder
stage (except the final, full-resolution one, which is the main output).
Each auxiliary head's output is bilinearly upsampled to the full input
resolution so it can be supervised against the same ground-truth mask as
the main output.

Motivation (see paper Section 4.5, "Discussion"): the authors note
VasCA-Net's sensitivity (Se) on thin/peripheral vessels is its weakest
metric relative to competing methods, even though specificity/AUC are
strong. Deep supervision is a well-established way (UNet++, PSPNet,
Inception, etc.) to push earlier/coarser decoder stages to already
produce vessel-shaped features rather than relying entirely on the final
1x1 head to recover fine structure from a single upsampling step -- this
tends to specifically help recall on thin, easily-smoothed-away structures.

Use with `utils.losses.DeepSupervisionLoss` wrapping a base per-pixel loss
(FocalTverskyLoss is recommended, since it also directly targets the
recall gap via its alpha/beta weighting).

At eval/inference time, only the main output is used -- call
`model(x, return_aux=False)` or simply take `output[0]` if you keep
`return_aux=True` for consistency with the training loss signature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .econv import EConvBlock
from .msca import MSCA
from .dconv import DConvBlock


class VasCANetDS(nn.Module):
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
        c5 = base_channels * 16

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = EConvBlock(in_channels, c1, ratio=econv_ratio)
        self.enc2 = EConvBlock(c1, c2, ratio=econv_ratio)
        self.enc3 = EConvBlock(c2, c3, ratio=econv_ratio)
        self.enc4 = EConvBlock(c3, c4, ratio=econv_ratio)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5, c5, kernel_size=3, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        self.msca1 = MSCA(c1, ratio=msca_ratio)
        self.msca2 = MSCA(c2, ratio=msca_ratio)
        self.msca3 = MSCA(c3, ratio=msca_ratio)
        self.msca4 = MSCA(c4, ratio=msca_ratio)

        self.dec4 = DConvBlock(low_channels=c5, high_channels=c4, out_channels=c4)
        self.dec3 = DConvBlock(low_channels=c4, high_channels=c3, out_channels=c3)
        self.dec2 = DConvBlock(low_channels=c3, high_channels=c2, out_channels=c2)
        self.dec1 = DConvBlock(low_channels=c2, high_channels=c1, out_channels=c1)

        # Main (full-resolution) output head
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

        # Auxiliary deep-supervision heads at H/8, H/4, H/2 resolutions
        # (outputs of dec4, dec3, dec2 respectively). Each is upsampled to
        # full resolution at forward time before being returned.
        self.aux_head4 = nn.Conv2d(c4, num_classes, kernel_size=1)
        self.aux_head3 = nn.Conv2d(c3, num_classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(c2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_aux: bool = True):
        input_size = x.shape[-2:]

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        b = self.bottleneck(p4)

        s1 = self.msca1(e1)
        s2 = self.msca2(e2)
        s3 = self.msca3(e3)
        s4 = self.msca4(e4)

        d4 = self.dec4(b, s4)   # H/8
        d3 = self.dec3(d4, s3)  # H/4
        d2 = self.dec2(d3, s2)  # H/2
        d1 = self.dec1(d2, s1)  # H

        main_out = self.head(d1)

        if not return_aux:
            return main_out

        aux4 = F.interpolate(self.aux_head4(d4), size=input_size, mode="bilinear", align_corners=True)
        aux3 = F.interpolate(self.aux_head3(d3), size=input_size, mode="bilinear", align_corners=True)
        aux2 = F.interpolate(self.aux_head2(d2), size=input_size, mode="bilinear", align_corners=True)

        return main_out, [aux2, aux3, aux4]
