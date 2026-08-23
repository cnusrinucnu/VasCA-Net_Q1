import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.retina_dataset_geo import RetinaPatchDatasetGeo
from models.vasca_net import VasCANet
from models.vasca_net_geo import VasCANetGeo


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT = "./data_final/DRIVE"

BASELINE_CKPT = (
    "./checkpoints_q1/drive/baseline/"
    "vasca_baseline_best.pth"
)

GEO_CKPT = (
    "./checkpoints_q1/drive/geo/"
    "geo_vasca_best.pth"
)


def get_thickness_masks(gt):

    vessel = (
        gt > 0.5
    ).astype(np.uint8)

    distance = cv2.distanceTransform(
        vessel,
        cv2.DIST_L2,
        5
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

    return {
        "thin": thin,
        "medium": medium,
        "thick": thick,
    }


def evaluate_class(pred, gt_class):

    # Prediction is considered correct for this
    # thickness category only when it overlaps
    # the corresponding GT category.

    tp = np.logical_and(
        pred,
        gt_class
    ).sum()

    fn = np.logical_and(
        ~pred,
        gt_class
    ).sum()

    # FP is prediction outside this GT category.
    fp = np.logical_and(
        pred,
        ~gt_class
    ).sum()

    dice = (
        2.0 * tp /
        (
            2.0 * tp
            + fp
            + fn
            + 1e-8
        )
    )

    sensitivity = (
        tp /
        (
            tp
            + fn
            + 1e-8
        )
    )

    precision = (
        tp /
        (
            tp
            + fp
            + 1e-8
        )
    )

    return {
        "dice": float(dice),
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "gt_pixels": int(gt_class.sum()),
    }


def load_models():

    baseline = VasCANet(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(DEVICE)

    geo = VasCANetGeo(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(DEVICE)

    bckpt = torch.load(
        BASELINE_CKPT,
        map_location=DEVICE
    )

    gckpt = torch.load(
        GEO_CKPT,
        map_location=DEVICE
    )

    baseline.load_state_dict(
        bckpt["model"]
    )

    geo.load_state_dict(
        gckpt["model"]
    )

    baseline.eval()
    geo.eval()

    return baseline, geo


@torch.no_grad()
def evaluate_model(
    model,
    dataset
):

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    categories = [
        "thin",
        "medium",
        "thick"
    ]

    results = {
        c: []
        for c in categories
    }

    for image, mask, radius in loader:

        image = image.to(
            DEVICE
        )

        output = model(
            image
        )

        if isinstance(
            output,
            tuple
        ):
            logits = output[0]
        else:
            logits = output

        pred = (
            torch.sigmoid(
                logits
            )[0, 0]
            .cpu()
            .numpy()
            >= 0.5
        )

        gt = (
            mask[0, 0]
            .cpu()
            .numpy()
            >= 0.5
        )

        classes = get_thickness_masks(
            gt
        )

        for category in categories:

            results[category].append(
                evaluate_class(
                    pred,
                    classes[category]
                )
            )

    return results


def aggregate(results):

    output = {}

    for category, values in results.items():

        output[category] = {}

        for metric in [
            "dice",
            "sensitivity",
            "precision",
        ]:

            output[category][metric] = float(
                np.mean(
                    [
                        v[metric]
                        for v in values
                    ]
                )
            )

        for metric in [
            "tp",
            "fn",
            "fp",
            "gt_pixels",
        ]:

            output[category][metric] = int(
                sum(
                    v[metric]
                    for v in values
                )
            )

    return output


def print_table(
    name,
    results
):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"{'Class':<12}"
        f"{'Dice':>12}"
        f"{'Sensitivity':>15}"
        f"{'Precision':>15}"
        f"{'FN':>12}"
    )

    print("-" * 80)

    for category in [
        "thin",
        "medium",
        "thick",
    ]:

        r = results[category]

        print(
            f"{category:<12}"
            f"{r['dice']:>12.4f}"
            f"{r['sensitivity']:>15.4f}"
            f"{r['precision']:>15.4f}"
            f"{r['fn']:>12d}"
        )


def main():

    print("=" * 80)
    print(
        "CORRECTED THICKNESS-STRATIFIED "
        "EVALUATION"
    )
    print("=" * 80)

    print(
        "Device:",
        DEVICE
    )

    dataset = RetinaPatchDatasetGeo(
        root=ROOT,
        split="test",
        patch_size=64,
        patches_per_image=1,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    print(
        "Test images:",
        len(dataset.image_files)
    )

    baseline, geo = load_models()

    print(
        "\nEvaluating VasCA-Net..."
    )

    baseline_raw = evaluate_model(
        baseline,
        dataset
    )

    print(
        "Evaluating Geo-VasCA-Net..."
    )

    geo_raw = evaluate_model(
        geo,
        dataset
    )

    baseline = aggregate(
        baseline_raw
    )

    geo = aggregate(
        geo_raw
    )

    print_table(
        "BASELINE VasCA-Net",
        baseline
    )

    print_table(
        "Geo-VasCA-Net",
        geo
    )

    print()
    print("=" * 80)
    print("GEOMETRY IMPROVEMENT")
    print("=" * 80)

    for category in [
        "thin",
        "medium",
        "thick",
    ]:

        bd = baseline[category]["dice"]
        gd = geo[category]["dice"]

        bs = baseline[category]["sensitivity"]
        gs = geo[category]["sensitivity"]

        bp = baseline[category]["precision"]
        gp = geo[category]["precision"]

        print(
            f"{category.capitalize():<10}"
            f"Dice Δ={gd-bd:+.4f}   "
            f"Se Δ={gs-bs:+.4f}   "
            f"Precision Δ={gp-bp:+.4f}"
        )

    output_dir = (
        "./results_q1/drive"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        "corrected_thickness_comparison.csv"
    )

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "category",
            "baseline_dice",
            "geo_dice",
            "delta_dice",
            "baseline_sensitivity",
            "geo_sensitivity",
            "delta_sensitivity",
            "baseline_precision",
            "geo_precision",
            "delta_precision",
        ])

        for category in [
            "thin",
            "medium",
            "thick",
        ]:

            b = baseline[category]
            g = geo[category]

            writer.writerow([
                category,
                b["dice"],
                g["dice"],
                g["dice"] - b["dice"],
                b["sensitivity"],
                g["sensitivity"],
                g["sensitivity"] - b["sensitivity"],
                b["precision"],
                g["precision"],
                g["precision"] - b["precision"],
            ])

    print()
    print(
        "Saved:",
        output_file
    )


if __name__ == "__main__":
    main()
