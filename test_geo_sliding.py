import os
import csv
from glob import glob

import cv2
import numpy as np
import torch

from models.vasca_net_geo import VasCANetGeo


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(
    probs,
    target,
    threshold=0.5,
    eps=1e-7,
):

    pred = (
        probs >= threshold
    ).astype(np.float32)

    target = target.astype(
        np.float32
    )

    tp = (
        pred * target
    ).sum()

    tn = (
        (1.0 - pred)
        *
        (1.0 - target)
    ).sum()

    fp = (
        pred
        *
        (1.0 - target)
    ).sum()

    fn = (
        (1.0 - pred)
        *
        target
    ).sum()

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
        "Dice": float(dice),
        "IoU": float(iou),
        "Se": float(sensitivity),
        "Sp": float(specificity),
        "Precision": float(precision),
        "ACC": float(accuracy),
    }


# ==========================================================
# SLIDING WINDOW POSITION GENERATION
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


# ==========================================================
# FULL IMAGE SLIDING WINDOW INFERENCE
# ==========================================================

@torch.no_grad()
def sliding_window_inference(
    model,
    image,
    device,
    patch_size=64,
    stride=32,
):

    height, width = image.shape

    # ------------------------------------------------------
    # Pad image if smaller than patch
    # ------------------------------------------------------

    pad_height = max(
        0,
        patch_size - height,
    )

    pad_width = max(
        0,
        patch_size - width,
    )

    if pad_height > 0 or pad_width > 0:

        image = np.pad(
            image,
            (
                (0, pad_height),
                (0, pad_width),
            ),
            mode="reflect",
        )

    padded_height, padded_width = (
        image.shape
    )

    # ------------------------------------------------------
    # Create reconstruction maps
    # ------------------------------------------------------

    probability_map = np.zeros(
        (
            padded_height,
            padded_width,
        ),
        dtype=np.float32,
    )

    count_map = np.zeros(
        (
            padded_height,
            padded_width,
        ),
        dtype=np.float32,
    )

    y_positions = get_positions(
        padded_height,
        patch_size,
        stride,
    )

    x_positions = get_positions(
        padded_width,
        patch_size,
        stride,
    )

    # ------------------------------------------------------
    # Patch inference
    # ------------------------------------------------------

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

            vessel_logits, radius_pred = model(
                patch_tensor
            )

            logits = vessel_logits

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

    # ------------------------------------------------------
    # Average overlapping predictions
    # ------------------------------------------------------

    probability_map = (
        probability_map
        /
        np.maximum(
            count_map,
            1e-7,
        )
    )

    # Remove padding

    probability_map = probability_map[
        :height,
        :width,
    ]

    return probability_map


# ==========================================================
# LOAD DRIVE IMAGE
# ==========================================================

def load_image(
    path,
):

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise RuntimeError(
            f"Unable to read image: {path}"
        )

    image = (
        image.astype(np.float32)
        / 255.0
    )

    return image


# ==========================================================
# LOAD DRIVE MASK
# ==========================================================

def load_mask(
    path,
):

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:

        raise RuntimeError(
            f"Unable to read mask: {path}"
        )

    mask = (
        mask > 127
    ).astype(np.float32)

    return mask


# ==========================================================
# FIND MATCHING MASK
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

        if os.path.exists(
            candidate
        ):

            return candidate

    raise FileNotFoundError(
        f"Mask not found for "
        f"{image_path}"
    )


# ==========================================================
# MAIN
# ==========================================================

@torch.no_grad()
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

    checkpoint_path = (
        "./checkpoints_q1/"
        "drive/geo/"
        "geo_vasca_best.pth"
    )

    patch_size = 64
    stride = 32
    threshold = 0.5

    print("=" * 70)
    print(
        "VasCA-Net GEO"
        " — SLIDING-WINDOW DRIVE TEST"
    )
    print("=" * 70)

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    print(
        "Patch size:",
        patch_size
    )

    print(
        "Stride:",
        stride
    )

    print(
        "Overlap:",
        patch_size - stride
    )

    # ------------------------------------------------------
    # Find images
    # ------------------------------------------------------

    image_files = []

    extensions = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.tif",
        "*.tiff",
        "*.bmp",
    ]

    for extension in extensions:

        image_files.extend(
            glob(
                os.path.join(
                    images_dir,
                    extension,
                )
            )
        )

    image_files = sorted(
        image_files
    )

    if len(image_files) == 0:

        raise RuntimeError(
            f"No images found in "
            f"{images_dir}"
        )

    print(
        "Test images:",
        len(image_files)
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

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
            "unknown",
        )
    )

    print(
        "Best validation Dice:",
        checkpoint.get(
            "best_val_dice",
            "unknown",
        )
    )

    # ------------------------------------------------------
    # Test
    # ------------------------------------------------------

    all_metrics = []

    for index, image_path in enumerate(
        image_files
    ):

        mask_path = find_mask(
            masks_dir,
            image_path,
        )

        image = load_image(
            image_path
        )

        mask = load_mask(
            mask_path
        )

        probabilities = (
            sliding_window_inference(
                model=model,
                image=image,
                device=device,
                patch_size=patch_size,
                stride=stride,
            )
        )

        metrics = calculate_metrics(
            probabilities,
            mask,
            threshold=threshold,
        )

        all_metrics.append(
            metrics
        )

        image_name = os.path.basename(
            image_path
        )

        print(
            f"Test image {index + 1} "
            f"({image_name}): "
            f"Dice={metrics['Dice']:.4f} "
            f"IoU={metrics['IoU']:.4f} "
            f"Se={metrics['Se']:.4f} "
            f"Sp={metrics['Sp']:.4f} "
            f"Precision={metrics['Precision']:.4f} "
            f"ACC={metrics['ACC']:.4f}"
        )

    # ------------------------------------------------------
    # Mean metrics
    # ------------------------------------------------------

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
                    item[key]
                    for item in all_metrics
                ]
            )
        )

    print()
    print("=" * 70)
    print(
        "FINAL DRIVE GEO"
        " SLIDING-WINDOW RESULTS"
    )
    print("=" * 70)

    for key in keys:

        print(
            f"{key:12s}: "
            f"{mean_metrics[key]:.4f}"
        )

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    output_dir = (
        "./results_q1/drive/"
        "geo_sliding"
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

            writer.writerow(
                metrics
            )

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
