import os
import csv
from glob import glob

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from models.vasca_net import VasCANet
from models.vasca_net_geo import VasCANetGeo


# ==========================================================
# THICKNESS CLASSES
# ==========================================================

def get_thickness_classes(mask):

    radius = distance_transform_edt(
        mask.astype(np.uint8)
    )

    return {
        "thin": (
            (radius > 0.0) &
            (radius <= 1.5)
        ),

        "medium": (
            (radius > 1.5) &
            (radius <= 3.0)
        ),

        "thick": (
            radius > 3.0
        ),
    }


# ==========================================================
# CLASS-SPECIFIC DETECTION METRICS
# ==========================================================
#
# TP:
# predicted vessel inside target thickness class
#
# FN:
# ground-truth vessel pixels in the target class
# that were missed
#
# FP:
# predicted vessel pixels outside the complete
# ground-truth vessel mask
#
# This avoids artificially removing false positives.
#

def class_metrics(
    pred,
    gt_mask,
    class_mask,
    eps=1e-7,
):

    pred = pred.astype(bool)
    gt_mask = gt_mask.astype(bool)
    class_mask = class_mask.astype(bool)

    # True positives for this thickness class
    tp = np.logical_and(
        pred,
        class_mask,
    ).sum()

    # False negatives in this thickness class
    fn = np.logical_and(
        np.logical_not(pred),
        class_mask,
    ).sum()

    # Global false positives
    fp = np.logical_and(
        pred,
        np.logical_not(gt_mask),
    ).sum()

    dice = (
        2.0 * tp
        /
        (2.0 * tp + fp + fn + eps)
    )

    sensitivity = (
        tp
        /
        (tp + fn + eps)
    )

    precision = (
        tp
        /
        (tp + fp + eps)
    )

    return {
        "Dice": float(dice),
        "Se": float(sensitivity),
        "Precision": float(precision),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
    }


# ==========================================================
# SLIDING WINDOW
# ==========================================================

def get_positions(
    length,
    patch_size,
    stride,
):

    if length <= patch_size:
        return [0]

    positions = list(
        range(
            0,
            length - patch_size + 1,
            stride,
        )
    )

    last_position = (
        length - patch_size
    )

    if positions[-1] != last_position:
        positions.append(
            last_position
        )

    return positions


@torch.no_grad()
def sliding_window_inference(
    model,
    image,
    device,
    patch_size=64,
    stride=32,
    is_geo=False,
):

    height, width = image.shape

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    count_map = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    y_positions = get_positions(
        height,
        patch_size,
        stride,
    )

    x_positions = get_positions(
        width,
        patch_size,
        stride,
    )

    for top in y_positions:

        for left in x_positions:

            patch = image[
                top:top + patch_size,
                left:left + patch_size,
            ]

            patch_tensor = (
                torch.from_numpy(
                    patch.copy()
                )
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            output = model(
                patch_tensor
            )

            if is_geo:

                logits, _ = output

            else:

                logits = output

            probs = torch.sigmoid(
                logits
            )

            probs = (
                probs
                .squeeze()
                .cpu()
                .numpy()
            )

            probability_map[
                top:top + patch_size,
                left:left + patch_size,
            ] += probs

            count_map[
                top:top + patch_size,
                left:left + patch_size,
            ] += 1.0

    probability_map = (
        probability_map
        /
        np.maximum(
            count_map,
            1e-7,
        )
    )

    return probability_map


# ==========================================================
# MASK FINDER
# ==========================================================

def find_mask(
    masks_dir,
    image_path,
):

    stem = os.path.splitext(
        os.path.basename(
            image_path
        )
    )[0]

    extensions = [
        ".gif",
        ".png",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".bmp",
    ]

    for ext in extensions:

        candidate = os.path.join(
            masks_dir,
            stem + ext,
        )

        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Mask not found for {image_path}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root = "./data_final/DRIVE"

    images_dir = os.path.join(
        root,
        "test",
        "images",
    )

    masks_dir = os.path.join(
        root,
        "test",
        "masks",
    )

    baseline_ckpt = (
        "./checkpoints_q1/drive/baseline/"
        "vasca_baseline_best.pth"
    )

    geo_ckpt = (
        "./checkpoints_q1/drive/geo/"
        "geo_vasca_best.pth"
    )

    print("=" * 80)
    print("CORRECTED THICKNESS-AWARE SLIDING-WINDOW EVALUATION")
    print("=" * 80)

    print("Device:", device)

    # ------------------------------------------------------
    # Baseline
    # ------------------------------------------------------

    print("\nLoading baseline...")

    baseline = VasCANet(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(device)

    checkpoint = torch.load(
        baseline_ckpt,
        map_location=device,
    )

    baseline.load_state_dict(
        checkpoint["model"]
    )

    baseline.eval()

    # ------------------------------------------------------
    # Geo-VasCA
    # ------------------------------------------------------

    print("Loading Geo-VasCA-Net...")

    geo = VasCANetGeo(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(device)

    checkpoint = torch.load(
        geo_ckpt,
        map_location=device,
    )

    geo.load_state_dict(
        checkpoint["model"]
    )

    geo.eval()

    categories = [
        "thin",
        "medium",
        "thick",
    ]

    results = {
        "baseline": {
            category: []
            for category in categories
        },
        "geo": {
            category: []
            for category in categories
        },
    }

    image_paths = sorted(
        glob(
            os.path.join(
                images_dir,
                "*"
            )
        )
    )

    print(
        "\nTest images:",
        len(image_paths),
    )

    rows = []

    # ------------------------------------------------------
    # Image evaluation
    # ------------------------------------------------------

    for image_path in image_paths:

        filename = os.path.basename(
            image_path
        )

        print(
            "\nEvaluating:",
            filename,
        )

        mask_path = find_mask(
            masks_dir,
            image_path,
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE,
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None or mask is None:

            raise RuntimeError(
                f"Unable to load {filename}"
            )

        image = (
            image.astype(np.float32)
            / 255.0
        )

        mask = (
            mask > 127
        ).astype(np.uint8)

        # Full-image inference

        baseline_probs = sliding_window_inference(
            baseline,
            image,
            device,
            patch_size=64,
            stride=32,
            is_geo=False,
        )

        geo_probs = sliding_window_inference(
            geo,
            image,
            device,
            patch_size=64,
            stride=32,
            is_geo=True,
        )

        baseline_pred = (
            baseline_probs >= 0.5
        )

        geo_pred = (
            geo_probs >= 0.5
        )

        thickness_classes = (
            get_thickness_classes(
                mask
            )
        )

        # --------------------------------------------------
        # Thickness metrics
        # --------------------------------------------------

        for category in categories:

            class_mask = (
                thickness_classes[
                    category
                ]
            )

            baseline_metrics = (
                class_metrics(
                    baseline_pred,
                    mask,
                    class_mask,
                )
            )

            geo_metrics = (
                class_metrics(
                    geo_pred,
                    mask,
                    class_mask,
                )
            )

            results[
                "baseline"
            ][category].append(
                baseline_metrics
            )

            results[
                "geo"
            ][category].append(
                geo_metrics
            )

            rows.append(
                {
                    "Image": filename,
                    "Class": category,
                    "Baseline_Dice":
                        baseline_metrics["Dice"],
                    "Geo_Dice":
                        geo_metrics["Dice"],
                    "Baseline_Se":
                        baseline_metrics["Se"],
                    "Geo_Se":
                        geo_metrics["Se"],
                    "Baseline_Precision":
                        baseline_metrics["Precision"],
                    "Geo_Precision":
                        geo_metrics["Precision"],
                    "Baseline_FN":
                        baseline_metrics["FN"],
                    "Geo_FN":
                        geo_metrics["FN"],
                }
            )

    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    print()
    print("=" * 80)
    print("CORRECTED THICKNESS-AWARE RESULTS")
    print("=" * 80)

    print(
        f"{'Class':12s}"
        f"{'Base Dice':>12s}"
        f"{'Geo Dice':>12s}"
        f"{'Delta':>12s}"
        f"{'Base Se':>12s}"
        f"{'Geo Se':>12s}"
        f"{'Se Delta':>12s}"
        f"{'Base Prec':>12s}"
        f"{'Geo Prec':>12s}"
    )

    print("-" * 116)

    for category in categories:

        base_dice = np.mean([
            x["Dice"]
            for x in results[
                "baseline"
            ][category]
        ])

        geo_dice = np.mean([
            x["Dice"]
            for x in results[
                "geo"
            ][category]
        ])

        base_se = np.mean([
            x["Se"]
            for x in results[
                "baseline"
            ][category]
        ])

        geo_se = np.mean([
            x["Se"]
            for x in results[
                "geo"
            ][category]
        ])

        base_precision = np.mean([
            x["Precision"]
            for x in results[
                "baseline"
            ][category]
        ])

        geo_precision = np.mean([
            x["Precision"]
            for x in results[
                "geo"
            ][category]
        ])

        print(
            f"{category:12s}"
            f"{base_dice:12.4f}"
            f"{geo_dice:12.4f}"
            f"{geo_dice - base_dice:+12.4f}"
            f"{base_se:12.4f}"
            f"{geo_se:12.4f}"
            f"{geo_se - base_se:+12.4f}"
            f"{base_precision:12.4f}"
            f"{geo_precision:12.4f}"
        )

    # ======================================================
    # SAVE CSV
    # ======================================================

    output_dir = (
        "./results_q1/drive/"
        "corrected_thickness_sliding"
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_csv = os.path.join(
        output_dir,
        "thickness_comparison.csv",
    )

    with open(
        output_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Image",
                "Class",
                "Baseline_Dice",
                "Geo_Dice",
                "Baseline_Se",
                "Geo_Se",
                "Baseline_Precision",
                "Geo_Precision",
                "Baseline_FN",
                "Geo_FN",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        "Saved:",
        output_csv,
    )


if __name__ == "__main__":
    main()
