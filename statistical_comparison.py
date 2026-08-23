import os
import csv
from glob import glob

import cv2
import numpy as np
import torch

from models.vasca_net import VasCANet
from models.vasca_net_geo import VasCANetGeo


# ==========================================================
# CONFIGURATION
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

ROOT = "./data_final/DRIVE"

BASELINE_CHECKPOINT = (
    "./checkpoints_q1/"
    "drive/baseline/"
    "vasca_baseline_best.pth"
)

GEO_CHECKPOINT = (
    "./checkpoints_q1/"
    "drive/geo/"
    "geo_vasca_best.pth"
)

PATCH_SIZE = 64
STRIDE = 32
THRESHOLD = 0.5


# ==========================================================
# IMAGE LOADING
# ==========================================================

def load_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read image: {path}"
        )

    return (
        image.astype(np.float32)
        / 255.0
    )


def load_mask(path):

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Unable to read mask: {path}"
        )

    return (
        mask > 127
    ).astype(np.float32)


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
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
    ]

    for extension in extensions:

        candidate = os.path.join(
            masks_dir,
            stem + extension,
        )

        if os.path.exists(candidate):

            return candidate

    raise FileNotFoundError(
        f"Mask not found for: {image_path}"
    )


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
        PATCH_SIZE,
        STRIDE,
    )

    x_positions = get_positions(
        width,
        PATCH_SIZE,
        STRIDE,
    )

    for top in y_positions:

        for left in x_positions:

            patch = image[
                top:top + PATCH_SIZE,
                left:left + PATCH_SIZE,
            ]

            patch_tensor = torch.from_numpy(
                patch.copy()
            ).float()

            patch_tensor = (
                patch_tensor
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            output = model(
                patch_tensor
            )

            # Geo model returns:
            # vessel_logits, radius_prediction

            if isinstance(
                output,
                tuple,
            ):

                logits = output[0]

            else:

                logits = output

            probabilities = torch.sigmoid(
                logits
            )

            probabilities = (
                probabilities
                .squeeze()
                .cpu()
                .numpy()
            )

            probability_map[
                top:top + PATCH_SIZE,
                left:left + PATCH_SIZE,
            ] += probabilities

            count_map[
                top:top + PATCH_SIZE,
                left:left + PATCH_SIZE,
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
# METRICS
# ==========================================================

def calculate_metrics(
    probabilities,
    target,
):

    prediction = (
        probabilities >= THRESHOLD
    ).astype(np.float32)

    target = target.astype(
        np.float32
    )

    tp = (
        prediction * target
    ).sum()

    tn = (
        (1.0 - prediction)
        *
        (1.0 - target)
    ).sum()

    fp = (
        prediction
        *
        (1.0 - target)
    ).sum()

    fn = (
        (1.0 - prediction)
        *
        target
    ).sum()

    eps = 1e-7

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

    precision = (
        tp
        /
        (tp + fp + eps)
    )

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Se": float(sensitivity),
        "Precision": float(precision),
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 80)
    print("BASELINE vs GEO-VASCA-NET")
    print("PER-IMAGE PAIRED STATISTICAL COMPARISON")
    print("=" * 80)

    print("Device:", DEVICE)

    # ------------------------------------------------------
    # Load baseline
    # ------------------------------------------------------

    baseline = VasCANet(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(DEVICE)

    baseline_checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location=DEVICE,
    )

    baseline.load_state_dict(
        baseline_checkpoint["model"]
    )

    baseline.eval()

    # ------------------------------------------------------
    # Load Geo model
    # ------------------------------------------------------

    geo = VasCANetGeo(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(DEVICE)

    geo_checkpoint = torch.load(
        GEO_CHECKPOINT,
        map_location=DEVICE,
    )

    geo.load_state_dict(
        geo_checkpoint["model"]
    )

    geo.eval()

    print()
    print(
        "Baseline checkpoint:",
        baseline_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "Geo checkpoint:",
        geo_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    # ------------------------------------------------------
    # Test images
    # ------------------------------------------------------

    images_dir = os.path.join(
        ROOT,
        "test",
        "images",
    )

    masks_dir = os.path.join(
        ROOT,
        "test",
        "masks",
    )

    image_patterns = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.tif",
        "*.tiff",
        "*.bmp",
    ]

    image_paths = []

    for pattern in image_patterns:

        image_paths.extend(
            glob(
                os.path.join(
                    images_dir,
                    pattern,
                )
            )
        )

    image_paths = sorted(
        image_paths
    )

    print()
    print(
        "Official test images:",
        len(image_paths)
    )

    # ------------------------------------------------------
    # Per-image evaluation
    # ------------------------------------------------------

    results = []

    for image_path in image_paths:

        image_name = os.path.basename(
            image_path
        )

        print()
        print(
            "Evaluating:",
            image_name
        )

        mask_path = find_mask(
            masks_dir,
            image_path,
        )

        image = load_image(
            image_path
        )

        target = load_mask(
            mask_path
        )

        baseline_probs = (
            sliding_window_inference(
                baseline,
                image,
            )
        )

        geo_probs = (
            sliding_window_inference(
                geo,
                image,
            )
        )

        baseline_metrics = calculate_metrics(
            baseline_probs,
            target,
        )

        geo_metrics = calculate_metrics(
            geo_probs,
            target,
        )

        row = {
            "Image": image_name,
        }

        for key in baseline_metrics:

            row[
                "Base_" + key
            ] = baseline_metrics[key]

            row[
                "Geo_" + key
            ] = geo_metrics[key]

            row[
                "Delta_" + key
            ] = (
                geo_metrics[key]
                -
                baseline_metrics[key]
            )

        results.append(
            row
        )

        print(
            f"Baseline Dice="
            f"{baseline_metrics['Dice']:.4f}"
        )

        print(
            f"Geo Dice="
            f"{geo_metrics['Dice']:.4f}"
        )

        print(
            f"Delta="
            f"{row['Delta_Dice']:+.4f}"
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("PER-IMAGE RESULTS")
    print("=" * 80)

    for row in results:

        print(
            f"{row['Image']:12s} "
            f"Base={row['Base_Dice']:.4f} "
            f"Geo={row['Geo_Dice']:.4f} "
            f"Delta={row['Delta_Dice']:+.4f}"
        )

    # ------------------------------------------------------
    # Mean and standard deviation
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("MEAN ± STANDARD DEVIATION")
    print("=" * 80)

    for metric in [
        "Dice",
        "IoU",
        "Se",
        "Precision",
    ]:

        base_values = np.array(
            [
                row[
                    "Base_" + metric
                ]
                for row in results
            ]
        )

        geo_values = np.array(
            [
                row[
                    "Geo_" + metric
                ]
                for row in results
            ]
        )

        delta_values = (
            geo_values
            -
            base_values
        )

        print(
            f"{metric:10s} "
            f"Base="
            f"{base_values.mean():.4f}"
            f"±{base_values.std(ddof=1):.4f}   "
            f"Geo="
            f"{geo_values.mean():.4f}"
            f"±{geo_values.std(ddof=1):.4f}   "
            f"Delta="
            f"{delta_values.mean():+.4f}"
        )

    # ------------------------------------------------------
    # Exact sign test interpretation
    # ------------------------------------------------------

    dice_deltas = np.array(
        [
            row["Delta_Dice"]
            for row in results
        ]
    )

    wins = int(
        (dice_deltas > 0).sum()
    )

    losses = int(
        (dice_deltas < 0).sum()
    )

    ties = int(
        (dice_deltas == 0).sum()
    )

    print()
    print("=" * 80)
    print("PAIRED DICE COMPARISON")
    print("=" * 80)

    print(
        "Geo wins:",
        wins,
    )

    print(
        "Baseline wins:",
        losses,
    )

    print(
        "Ties:",
        ties,
    )

    print(
        "Mean Dice difference:",
        f"{dice_deltas.mean():+.6f}",
    )

    # ------------------------------------------------------
    # Wilcoxon test
    # ------------------------------------------------------

    try:

        from scipy.stats import wilcoxon

        base_dice = np.array(
            [
                row["Base_Dice"]
                for row in results
            ]
        )

        geo_dice = np.array(
            [
                row["Geo_Dice"]
                for row in results
            ]
        )

        statistic, p_value = wilcoxon(
            geo_dice,
            base_dice,
            zero_method="wilcox",
            alternative="two-sided",
        )

        print()
        print(
            "Wilcoxon signed-rank test"
        )

        print(
            "Statistic:",
            f"{statistic:.4f}",
        )

        print(
            "p-value:",
            f"{p_value:.6f}",
        )

        if p_value < 0.05:

            print(
                "Result: "
                "Statistically significant"
            )

        else:

            print(
                "Result: "
                "Not statistically significant"
            )

    except Exception as error:

        print()
        print(
            "Wilcoxon test unavailable:"
        )

        print(
            str(error)
        )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_dir = (
        "./results_q1/drive/"
        "statistical_comparison"
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_csv = os.path.join(
        output_dir,
        "per_image_comparison.csv",
    )

    fieldnames = list(
        results[0].keys()
    )

    with open(
        output_csv,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:

            writer.writerow(
                row
            )

    print()
    print(
        "Saved:",
        output_csv,
    )


if __name__ == "__main__":
    main()
