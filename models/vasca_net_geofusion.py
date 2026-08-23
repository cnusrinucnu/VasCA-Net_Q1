"""
GeoFusion-VasCA-Net
-------------------

VasCA-Net with explicit vessel-radius prediction and
geometry-guided feature fusion.

The predicted geometry map is not only supervised as an
auxiliary output. It is converted into an attention map and
used to refine decoder features before final vessel prediction.

Outputs:
    vessel_logits : [B, 1, H, W]
    radius_pred   : [B, 1, H, W]
"""

import torch
import torch.nn as nn

from .econv import EConvBlock
from .msca import MSCA
from .dconv import DConvBlock


class GeometryGuidedFusion(nn.Module):

    def __init__(
        self,
        feature_channels,
    ):

        super().__init__()

        hidden = max(
            feature_channels // 2,
            8,
        )

        self.geometry_projection = nn.Sequential(

            nn.Conv2d(
                1,
                hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                hidden
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                feature_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.Tanh(),
        )

        self.refine = nn.Sequential(

            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                feature_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        # Small learnable geometry contribution.
        # Starts close to the original decoder features.
        self.gamma = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(
        self,
        features,
        radius,
    ):

        geometry_features = (
            self.geometry_projection(
                radius
            )
        )

        refined_geometry = (
            self.refine(
                geometry_features
            )
        )

        fused = (
            features
            +
            self.gamma
            *
            refined_geometry
        )

        return fused


class VasCANetGeoFusion(nn.Module):

    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ):

        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # ==================================================
        # Encoder
        # ==================================================

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

        # ==================================================
        # Bottleneck
        # ==================================================

        self.bottleneck = nn.Sequential(

            nn.Conv2d(
                c4,
                c5,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(c5),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                c5,
                c5,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(c5),

            nn.ReLU(
                inplace=True
            ),
        )

        # ==================================================
        # MSCA skip refinement
        # ==================================================

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

        # ==================================================
        # Decoder
        # ==================================================

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

        # ==================================================
        # Radius prediction branch
        # ==================================================

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

            nn.ReLU(
                inplace=True
            ),

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

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                geometry_hidden,
                1,
                kernel_size=1,
            ),

            nn.Softplus(),
        )

        # ==================================================
        # Geometry-guided feature fusion
        # ==================================================

        self.geometry_fusion = (
            GeometryGuidedFusion(
                feature_channels=c1,
            )
        )

        # ==================================================
        # Final vessel segmentation
        # ==================================================

        self.vessel_head = nn.Conv2d(
            c1,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        x,
    ):

        # Encoder

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Bottleneck

        b = self.bottleneck(p4)

        # Refined skips

        s1 = self.msca1(e1)
        s2 = self.msca2(e2)
        s3 = self.msca3(e3)
        s4 = self.msca4(e4)

        # Decoder

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # --------------------------------------------------
        # Explicit geometry prediction
        # --------------------------------------------------

        radius_pred = self.geometry_head(
            d1
        )

        # --------------------------------------------------
        # Geometry-guided feature fusion
        # --------------------------------------------------

        geometry_features = (
            self.geometry_fusion(
                d1,
                radius_pred,
            )
        )

        # Final vessel prediction

        vessel_logits = self.vessel_head(
            geometry_features
        )

        return (
            vessel_logits,
            radius_pred,
        )


if __name__ == "__main__":

    model = VasCANetGeoFusion(
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

    parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print("=" * 70)
    print("GeoFusion-VasCA-Net")
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
        f"{parameters / 1e6:.2f}M"
    )
