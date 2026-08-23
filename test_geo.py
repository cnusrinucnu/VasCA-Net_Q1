import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.retina_dataset_geo import RetinaPatchDatasetGeo
from models.vasca_net_geo import VasCANetGeo
from utils.geometry_loss import GeoVasCALoss


def calculate_metrics(
    logits,
    target,
    threshold=0.5,
    eps=1e-7,
):

    probs = torch.sigmoid(logits)

    pred = (
        probs >= threshold
    ).float()

    target = target.float()

    tp = (
        pred * target
    ).sum().item()

    tn = (
        (1.0 - pred)
        *
        (1.0 - target)
    ).sum().item()

    fp = (
        pred
        *
        (1.0 - target)
    ).sum().item()

    fn = (
        (1.0 - pred)
        *
        target
    ).sum().item()

    dice = (
        2.0 * tp
        /
        (2.0 * tp + fp + fn + eps)
    )

    iou = (
        tp
        /
        (tp + fp + fn + eps)
    )

    sensitivity = (
        tp
        /
        (tp + fn + eps)
    )

    specificity = (
        tn
        /
        (tn + fp + eps)
    )

    precision = (
        tp
        /
        (tp + fp + eps)
    )

    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn + eps)
    )

    return {
        "Dice": dice,
        "IoU": iou,
        "Se": sensitivity,
        "Sp": specificity,
        "Precision": precision,
        "ACC": accuracy,
    }


@torch.no_grad()
def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset_name = "DRIVE"

    root = (
        "./data_final/"
        + dataset_name
    )

    checkpoint_path = (
        "./checkpoints_q1/"
        "drive/geo/"
        "geo_vasca_best.pth"
    )

    print("=" * 70)
    print("Geo-VasCA-Net — FINAL DRIVE TEST")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    test_dataset = RetinaPatchDatasetGeo(
        root=root,
        split="test",
        patch_size=64,
        patches_per_image=1,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(
        "Test images:",
        len(test_dataset.image_files)
    )

    print(
        "Test samples:",
        len(test_dataset)
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = VasCANetGeo(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        ),
    )

    print(
        "Best validation Dice:",
        checkpoint.get(
            "best_val_dice",
            "unknown"
        ),
    )

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    all_metrics = []

    for index, batch in enumerate(
        test_loader
    ):

        images, masks, radius = batch

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        radius = radius.to(
            device,
            non_blocking=True,
        )

        vessel_logits, radius_pred = model(
            images
        )

        metrics = calculate_metrics(
            vessel_logits,
            masks,
            threshold=0.5,
        )

        all_metrics.append(
            metrics
        )

        print(
            f"Test image {index + 1}: "
            f"Dice={metrics['Dice']:.4f} "
            f"IoU={metrics['IoU']:.4f} "
            f"Se={metrics['Se']:.4f} "
            f"Sp={metrics['Sp']:.4f} "
            f"Precision={metrics['Precision']:.4f} "
            f"ACC={metrics['ACC']:.4f}"
        )

    # --------------------------------------------------
    # Aggregate
    # --------------------------------------------------

    keys = [
        "Dice",
        "IoU",
        "Se",
        "Sp",
        "Precision",
        "ACC",
    ]

    mean_metrics = {}

    for key in keys:

        mean_metrics[key] = float(
            np.mean(
                [
                    x[key]
                    for x in all_metrics
                ]
            )
        )

    print()
    print("=" * 70)
    print("FINAL DRIVE TEST RESULTS")
    print("=" * 70)

    for key in keys:

        print(
            f"{key:12s}: "
            f"{mean_metrics[key]:.4f}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_dir = (
        "./results_q1/drive/geo"
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_csv = os.path.join(
        output_dir,
        "test_metrics.csv",
    )

    with open(
        output_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()

        for metrics in all_metrics:
            writer.writerow(metrics)

        writer.writerow(
            mean_metrics
        )

    print()
    print(
        "Saved:",
        output_csv
    )


if __name__ == "__main__":
    main()
