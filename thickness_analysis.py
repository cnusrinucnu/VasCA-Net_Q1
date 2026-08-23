import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models.vasca_net_ds import VasCANetDS
from datasets import RetinalVesselDataset


def load_model(cfg, checkpoint, device):

    model = VasCANetDS(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        msca_ratio=cfg["model"]["msca_ratio"],
        econv_ratio=cfg["model"]["econv_ratio"],
    ).to(device)

    ckpt = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(ckpt["model"])
    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{ckpt.get('epoch', 'unknown')}"
    )

    return model


def classify_vessel_thickness(gt):
    """
    Distance-transform-based evaluation only.

    Thin:
        radius <= 1.5 pixels

    Medium:
        1.5 < radius <= 3.0 pixels

    Thick:
        radius > 3.0 pixels
    """

    vessel = (
        gt.astype(np.uint8)
        > 0
    ).astype(np.uint8)

    if vessel.sum() == 0:

        empty = np.zeros_like(
            vessel,
            dtype=bool,
        )

        return empty, empty, empty

    distance = cv2.distanceTransform(
        vessel,
        cv2.DIST_L2,
        5,
    )

    thin = (
        vessel == 1
    ) & (
        distance <= 1.5
    )

    medium = (
        vessel == 1
    ) & (
        distance > 1.5
    ) & (
        distance <= 3.0
    )

    thick = (
        vessel == 1
    ) & (
        distance > 3.0
    )

    return thin, medium, thick


def calculate_sensitivity(
    prediction,
    target,
):

    tp = np.logical_and(
        prediction,
        target,
    ).sum()

    fn = np.logical_and(
        ~prediction,
        target,
    ).sum()

    if tp + fn == 0:
        return np.nan, 0, 0

    sensitivity = (
        tp / (tp + fn)
    )

    return sensitivity, tp, fn


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
    ) as f:

        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["train"]["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = RetinalVesselDataset(
        root=cfg["data"]["root"],
        split="test",
        patch_size=cfg["data"]["patch_size"],
        images_subdir=cfg["data"]["images_subdir"],
        masks_subdir=cfg["data"]["masks_subdir"],
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    model = load_model(
        cfg,
        args.checkpoint,
        device,
    )

    total_tp = {
        "thin": 0,
        "medium": 0,
        "thick": 0,
    }

    total_fn = {
        "thin": 0,
        "medium": 0,
        "thick": 0,
    }

    total_pixels = {
        "thin": 0,
        "medium": 0,
        "thick": 0,
    }

    for img, mask in loader:

        img = img.to(device)

        logits = model(
            img,
            return_aux=False,
        )

        probability = torch.sigmoid(
            logits
        )

        prediction = (
            probability[0, 0]
            .cpu()
            .numpy()
            >= args.threshold
        )

        ground_truth = (
            mask[0, 0]
            .cpu()
            .numpy()
            >= 0.5
        )

        thin, medium, thick = (
            classify_vessel_thickness(
                ground_truth
            )
        )

        classes = {
            "thin": thin,
            "medium": medium,
            "thick": thick,
        }

        for name, cls in classes.items():

            se, tp, fn = (
                calculate_sensitivity(
                    prediction,
                    cls,
                )
            )

            if not np.isnan(se):

                total_tp[name] += tp
                total_fn[name] += fn
                total_pixels[name] += (
                    tp + fn
                )

    results = {}

    for name in [
        "thin",
        "medium",
        "thick",
    ]:

        tp = total_tp[name]
        fn = total_fn[name]

        sensitivity = (
            tp / (tp + fn + 1e-8)
        )

        results[name] = sensitivity

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        args.output,
        "w",
    ) as f:

        f.write(
            "category,sensitivity,tp,fn,pixels\n"
        )

        for name in [
            "thin",
            "medium",
            "thick",
        ]:

            f.write(
                f"{name},"
                f"{results[name]:.6f},"
                f"{total_tp[name]},"
                f"{total_fn[name]},"
                f"{total_pixels[name]}\n"
            )

    print()
    print(
        "Thickness-stratified sensitivity"
    )
    print(
        "================================="
    )

    for name in [
        "thin",
        "medium",
        "thick",
    ]:

        print(
            f"{name.capitalize():8s} "
            f"Se={results[name]:.4f} "
            f"TP={total_tp[name]} "
            f"FN={total_fn[name]} "
            f"Pixels={total_pixels[name]}"
        )

    print()
    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()
