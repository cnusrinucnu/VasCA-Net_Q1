"""
MS-Geo-VasCA-Net
----------------

Multi-Scale Geometry-Aware VasCA-Net.

Extension of Geo-VasCA-Net for the journal study.

The original VasCA-Net encoder, MSCA skip refinement,
and decoder are retained.

Two outputs are produced:

    1. Vessel segmentation logits
    2. Multi-scale vessel radius / geometry map

Unlike Geo-VasCA-Net, which predicts geometry only
from the final decoder feature d1, this model combines
multi-scale decoder features:

    d1 : fine vessel detail
    d2 : intermediate vessel structure
    d3 : contextual vessel structure
    d4 : coarse vessel context

All features are projected to a common channel dimension,
upsampled to d1 resolution, concatenated, and processed
by a lightweight geometry head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .econv import EConvBlock
from .msca import MSCA
from .dconv import DConvBlock


class MultiScaleGeometryFusion(nn.Module):
    """
    Fuse decoder features from multiple spatial scales.

    Inputs:
        d1 : H x W
        d2 : H/2 x W/2
        d3 : H/4 x W/4
        d4 : H/8 x W/8

    All features are projected to `projection_channels`
    and resized to d1 resolution.
    """

    def __init__(
        self,
        c1,
        c2,
        c3,
        c4,
        projection_channels=16,
    ):
        super().__init__()

        self.proj1 = nn.Sequential(
            nn.Conv2d(
                c1,
                projection_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                projection_channels
            ),
            nn.ReLU(inplace=True),
        )

        self.proj2 = nn.Sequential(
            nn.Conv2d(
                c2,
                projection_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                projection_channels
            ),
            nn.ReLU(inplace=True),
        )

        self.proj3 = nn.Sequential(
            nn.Conv2d(
                c3,
                projection_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                projection_channels
            ),
            nn.ReLU(inplace=True),
        )

        self.proj4 = nn.Sequential(
            nn.Conv2d(
                c4,
                projection_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                projection_channels
            ),
            nn.ReLU(inplace=True),
        )

        fused_channels = (
            projection_channels * 4
        )

        self.fusion = nn.Sequential(

            nn.Conv2d(
                fused_channels,
                c1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(c1),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                c1,
                c1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(c1),

            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        d1,
        d2,
        d3,
        d4,
    ):

        target_size = d1.shape[-2:]

        f1 = self.proj1(d1)

        f2 = self.proj2(d2)

        f3 = self.proj3(d3)

        f4 = self.proj4(d4)

        f2 = F.interpolate(
            f2,
            size=target_size,
            mode="bilinear",
            align_corners=True,
        )

        f3 = F.interpolate(
            f3,
            size=target_size,
            mode="bilinear",
            align_corners=True,
        )

        f4 = F.interpolate(
            f4,
            size=target_size,
            mode="bilinear",
            align_corners=True,
        )

        fused = torch.cat(
            [
                f1,
                f2,
                f3,
                f4,
            ],
            dim=1,
        )

        return self.fusion(fused)


class VasCANetMSGeo(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        msca_ratio: int = 8,
        econv_ratio: int = 2,
        geometry_projection: int = 16,
    ):
        super().__init__()

        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        c5 = base_channels * 16

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------

        self.enc1 = EConvBlock(
            in_channels,
            c1,
            ratio=econv_ratio,
        )

        self.enc2 = EConvBlock(
            c1,
            c2,
            ratio=econv_ratio,
        )

        self.enc3 = EConvBlock(
            c2,
            c3,
            ratio=econv_ratio,
        )

        self.enc4 = EConvBlock(
            c3,
            c4,
            ratio=econv_ratio,
        )

        # --------------------------------------------------
        # Bottleneck
        # --------------------------------------------------

        self.bottleneck = nn.Sequential(

            nn.Conv2d(
                c4,
                c5,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(c5),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                c5,
                c5,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(c5),

            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------
        # MSCA skip refinement
        # --------------------------------------------------

        self.msca1 = MSCA(
            c1,
            ratio=msca_ratio,
        )

        self.msca2 = MSCA(
            c2,
            ratio=msca_ratio,
        )

        self.msca3 = MSCA(
            c3,
            ratio=msca_ratio,
        )

        self.msca4 = MSCA(
            c4,
            ratio=msca_ratio,
        )

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------

        self.dec4 = DConvBlock(
            low_channels=c5,
            high_channels=c4,
            out_channels=c4,
        )

        self.dec3 = DConvBlock(
            low_channels=c4,
            high_channels=c3,
            out_channels=c3,
        )

        self.dec2 = DConvBlock(
            low_channels=c3,
            high_channels=c2,
            out_channels=c2,
        )

        self.dec1 = DConvBlock(
            low_channels=c2,
            high_channels=c1,
            out_channels=c1,
        )

        # --------------------------------------------------
        # Segmentation head
        # --------------------------------------------------

        self.vessel_head = nn.Conv2d(
            c1,
            num_classes,
            kernel_size=1,
        )

        # --------------------------------------------------
        # Multi-scale geometry fusion
        # --------------------------------------------------

        self.geometry_fusion = (
            MultiScaleGeometryFusion(
                c1=c1,
                c2=c2,
                c3=c3,
                c4=c4,
                projection_channels=
                    geometry_projection,
            )
        )

        # --------------------------------------------------
        # Geometry prediction head
        # --------------------------------------------------

        geometry_hidden = max(
            c1 // 2,
            8,
        )

        self.geometry_head = nn.Sequential(

            nn.Conv2d(
                c1,
                geometry_hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                geometry_hidden
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                geometry_hidden,
                geometry_hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                geometry_hidden
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                geometry_hidden,
                1,
                kernel_size=1,
            ),

            nn.Softplus(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # --------------------------------------------------
        # Bottleneck
        # --------------------------------------------------

        b = self.bottleneck(p4)

        # --------------------------------------------------
        # MSCA skip refinement
        # --------------------------------------------------

        s1 = self.msca1(e1)
        s2 = self.msca2(e2)
        s3 = self.msca3(e3)
        s4 = self.msca4(e4)

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------

        d4 = self.dec4(
            b,
            s4,
        )

        d3 = self.dec3(
            d4,
            s3,
        )

        d2 = self.dec2(
            d3,
            s2,
        )

        d1 = self.dec1(
            d2,
            s1,
        )

        # --------------------------------------------------
        # Segmentation
        # --------------------------------------------------

        vessel_logits = self.vessel_head(
            d1
        )

        # --------------------------------------------------
        # Multi-scale geometry
        # --------------------------------------------------

        geometry_features = (
            self.geometry_fusion(
                d1,
                d2,
                d3,
                d4,
            )
        )

        radius_pred = self.geometry_head(
            geometry_features
        )

        return (
            vessel_logits,
            radius_pred,
        )


if __name__ == "__main__":

    model = VasCANetMSGeo(
        in_channels=1,
        num_classes=1,
        base_channels=32,
    )

    x = torch.randn(
        2,
        1,
        64,
        64,
    )

    vessel, radius = model(x)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("=" * 70)
    print("MS-Geo-VasCA-Net")
    print("=" * 70)

    print(
        "Input:",
        tuple(x.shape),
    )

    print(
        "Vessel:",
        tuple(vessel.shape),
    )

    print(
        "Radius:",
        tuple(radius.shape),
    )

    print(
        f"Parameters: "
        f"{n_params / 1e6:.2f}M"
    )
