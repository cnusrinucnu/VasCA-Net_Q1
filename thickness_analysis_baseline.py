import argparse
import os
import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models import VasCANet
from datasets import RetinalVesselDataset


def classify_vessel_thickness(gt):
    vessel = (gt > 0).astype(np.uint8)

    if vessel.sum() == 0:
        z = np.zeros_like(vessel, dtype=bool)
        return z, z, z

    distance = cv2.distanceTransform(
        vessel,
        cv2.DIST_L2,
        5
    )

    thin = (vessel == 1) & (distance <= 1.5)

    medium = (
        (vessel == 1)
        & (distance > 1.5)
        & (distance <= 3.0)
    )

    thick = (
        (vessel == 1)
        & (distance > 3.0)
    )

    return thin, medium, thick


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.config, "r") as f:
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

    model = VasCANet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        msca_ratio=cfg["model"]["msca_ratio"],
        econv_ratio=cfg["model"]["econv_ratio"],
    ).to(device)

    ckpt = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(ckpt["model"])
    model.eval()

    print(
        f"Loaded baseline checkpoint from epoch "
        f"{ckpt.get('epoch', 'unknown')}"
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

    with torch.no_grad():

        for img, mask in loader:

            img = img.to(device)

            logits = model(img)

            probs = torch.sigmoid(logits)

            pred = (
                probs[0, 0]
                .cpu()
                .numpy()
                >= args.threshold
            )

            gt = (
                mask[0, 0]
                .cpu()
                .numpy()
                >= 0.5
            )

            thin, medium, thick = (
                classify_vessel_thickness(gt)
            )

            for name, cls in [
                ("thin", thin),
                ("medium", medium),
                ("thick", thick),
            ]:

                tp = np.logical_and(
                    pred,
                    cls
                ).sum()

                fn = np.logical_and(
                    ~pred,
                    cls
                ).sum()

                total_tp[name] += tp
                total_fn[name] += fn
                total_pixels[name] += tp + fn

    os.makedirs(
        os.path.dirname(args.output)
        if os.path.dirname(args.output)
        else ".",
        exist_ok=True
    )

    with open(args.output, "w") as f:

        f.write(
            "category,sensitivity,tp,fn,pixels\n"
        )

        for name in [
            "thin",
            "medium",
            "thick"
        ]:

            tp = total_tp[name]
            fn = total_fn[name]

            se = tp / (tp + fn + 1e-8)

            f.write(
                f"{name},{se:.6f},"
                f"{tp},{fn},"
                f"{total_pixels[name]}\n"
            )

            print(
                f"{name.capitalize():8s} "
                f"Se={se:.4f} "
                f"TP={tp} "
                f"FN={fn} "
                f"Pixels={total_pixels[name]}"
            )

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
