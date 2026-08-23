"""
Geometry-aware losses for the journal extension of VasCA-Net.

The loss jointly optimizes:
    1. retinal vessel segmentation
    2. explicit vessel-radius prediction

The geometry target is a normalized distance-transform map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import DiceLoss


class GeometryRegressionLoss(nn.Module):
    """
    Smooth-L1 regression loss for vessel radius prediction.

    Background pixels are ignored so that the geometry branch
    learns vessel caliber rather than predicting zeros over the
    large background region.
    """

    def __init__(
        self,
        beta=0.1,
    ):
        super().__init__()

        self.beta = beta

    def forward(
        self,
        radius_pred,
        radius_target,
        vessel_mask,
    ):

        # Only vessel pixels contribute to geometry supervision.
        valid = vessel_mask > 0.5

        if valid.sum() == 0:
            return radius_pred.sum() * 0.0

        pred = radius_pred[valid]
        target = radius_target[valid]

        return F.smooth_l1_loss(
            pred,
            target,
            beta=self.beta,
        )


class GeoVasCALoss(nn.Module):
    """
    Combined segmentation + geometry loss.

    L_total =
        L_seg + lambda_geo * L_geometry

    Segmentation:
        BCE + Dice

    Geometry:
        vessel-radius Smooth-L1
    """

    def __init__(
        self,
        bce_weight=0.5,
        lambda_geo=0.10,
        smooth=1.0,
    ):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss(
            smooth=smooth
        )

        self.geometry = GeometryRegressionLoss()

        self.bce_weight = bce_weight
        self.lambda_geo = lambda_geo

    def forward(
        self,
        vessel_logits,
        vessel_target,
        radius_pred,
        radius_target,
    ):

        # --------------------------------------------------
        # Segmentation loss
        # --------------------------------------------------

        bce_loss = self.bce(
            vessel_logits,
            vessel_target,
        )

        dice_loss = self.dice(
            vessel_logits,
            vessel_target,
        )

        segmentation_loss = (
            self.bce_weight * bce_loss
            +
            (1.0 - self.bce_weight) * dice_loss
        )

        # --------------------------------------------------
        # Geometry loss
        # --------------------------------------------------

        geometry_loss = self.geometry(
            radius_pred,
            radius_target,
            vessel_target,
        )

        # --------------------------------------------------
        # Total
        # --------------------------------------------------

        total_loss = (
            segmentation_loss
            +
            self.lambda_geo * geometry_loss
        )

        return {
            "total": total_loss,
            "segmentation": segmentation_loss.detach(),
            "bce": bce_loss.detach(),
            "dice": dice_loss.detach(),
            "geometry": geometry_loss.detach(),
        }
