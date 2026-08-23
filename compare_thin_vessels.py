import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import RetinalVesselDataset
from models import VasCANet
from models.vasca_net_geo import VasCANetGeo


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = "./data_final/DRIVE"

BASELINE_CKPT = (
    "./checkpoints_q1/drive/baseline/"
    "vasca_baseline_best.pth"
)

GEO_CKPT = (
    "./checkpoints_q1/drive/geo/"
    "geo_vasca_best.pth"
)


def thickness_classes(gt):

    vessel = (gt > 0.5).astype(np.uint8)

    distance = cv2.distanceTransform(
        vessel,
        cv2.DIST_L2,
        5
    )

    thin = (
        (vessel == 1) &
        (distance <= 1.5)
    )

    medium = (
        (vessel == 1) &
        (distance > 1.5) &
        (distance <= 3.0)
    )

    thick = (
        (vessel == 1) &
        (distance > 3.0)
    )

    return {
        "thin": thin,
        "medium": medium,
        "thick": thick,
    }


def metrics_for_class(pred, cls):

    tp = np.logical_and(pred, cls).sum()
    fn = np.logical_and(~pred, cls).sum()
    fp = np.logical_and(pred, cls == False).sum()

    sensitivity = (
        tp / (tp + fn + 1e-8)
    )

    precision = (
        tp / (tp + fp + 1e-8)
    )

    dice = (
        2.0 * tp /
        (2.0 * tp + fp + fn + 1e-8)
    )

    return {
        "sensitivity": sensitivity,
        "precision": precision,
        "dice": dice,
        "tp": tp,
        "fn": fn,
        "pixels": tp + fn,
    }


def load_baseline():

    model = VasCANet(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(DEVICE)

    ckpt = torch.load(
        BASELINE_CKPT,
        map_location=DEVICE
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    return model


def load_geo():

    model = VasCANetGeo(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(DEVICE)

    ckpt = torch.load(
        GEO_CKPT,
        map_location=DEVICE
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    return model


@torch.no_grad()
def evaluate(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    results = {
        "thin": [],
        "medium": [],
        "thick": [],
    }

    for image, mask in loader:

        image = image.to(DEVICE)

        output = model(image)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        prediction = (
            torch.sigmoid(logits)[0, 0]
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

        classes = thickness_classes(gt)

        for name, cls in classes.items():

            results[name].append(
                metrics_for_class(
                    prediction,
                    cls
                )
            )

    return results


def aggregate(results):

    output = {}

    for category, values in results.items():

        output[category] = {}

        for metric in [
            "sensitivity",
            "precision",
            "dice",
        ]:

            output[category][metric] = np.mean(
                [
                    x[metric]
                    for x in values
                ]
            )

        output[category]["tp"] = sum(
            x["tp"] for x in values
        )

        output[category]["fn"] = sum(
            x["fn"] for x in values
        )

        output[category]["pixels"] = sum(
            x["pixels"] for x in values
        )

    return output


def print_results(name, results):

    print()
    print("=" * 75)
    print(name)
    print("=" * 75)

    print(
        f"{'Category':<12}"
        f"{'Dice':>12}"
        f"{'Sensitivity':>15}"
        f"{'Precision':>15}"
        f"{'FN':>12}"
    )

    print("-" * 75)

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


def save_csv(
    baseline,
    geo,
    output_path
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(
        output_path,
        "w"
    ) as f:

        f.write(
            "category,"
            "baseline_dice,"
            "geo_dice,"
            "delta_dice,"
            "baseline_sensitivity,"
            "geo_sensitivity,"
            "delta_sensitivity,"
            "baseline_precision,"
            "geo_precision,"
            "delta_precision\n"
        )

        for category in [
            "thin",
            "medium",
            "thick",
        ]:

            b = baseline[category]
            g = geo[category]

            f.write(
                f"{category},"
                f"{b['dice']:.6f},"
                f"{g['dice']:.6f},"
                f"{g['dice']-b['dice']:.6f},"
                f"{b['sensitivity']:.6f},"
                f"{g['sensitivity']:.6f},"
                f"{g['sensitivity']-b['sensitivity']:.6f},"
                f"{b['precision']:.6f},"
                f"{g['precision']:.6f},"
                f"{g['precision']-b['precision']:.6f}\n"
            )


def main():

    print("=" * 75)
    print("VasCA-Net vs Geo-VasCA-Net")
    print("THICKNESS-STRATIFIED EVALUATION")
    print("=" * 75)

    print("Device:", DEVICE)

    dataset = RetinalVesselDataset(
        root=ROOT,
        split="test",
        patch_size=64,
        images_subdir="images",
        masks_subdir="masks",
    )

    print(
        "Test images:",
        len(dataset)
    )

    print()
    print("Loading baseline...")

    baseline_model = load_baseline()

    print("Loading Geo-VasCA...")

    geo_model = load_geo()

    print()
    print("Evaluating baseline...")

    baseline_raw = evaluate(
        baseline_model,
        dataset
    )

    print("Evaluating Geo-VasCA...")

    geo_raw = evaluate(
        geo_model,
        dataset
    )

    baseline = aggregate(
        baseline_raw
    )

    geo = aggregate(
        geo_raw
    )

    print_results(
        "BASELINE VasCA-Net",
        baseline
    )

    print_results(
        "Geo-VasCA-Net",
        geo
    )

    print()
    print("=" * 75)
    print("DIRECT IMPROVEMENT")
    print("=" * 75)

    for category in [
        "thin",
        "medium",
        "thick",
    ]:

        bd = baseline[category]["dice"]
        gd = geo[category]["dice"]

        bs = baseline[category]["sensitivity"]
        gs = geo[category]["sensitivity"]

        print(
            f"{category.capitalize():<10} "
            f"Dice Δ={gd-bd:+.4f}   "
            f"Sensitivity Δ={gs-bs:+.4f}"
        )

    output = (
        "./results_q1/drive/"
        "thickness_comparison.csv"
    )

    save_csv(
        baseline,
        geo,
        output
    )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
