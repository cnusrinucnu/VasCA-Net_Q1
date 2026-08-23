import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.retina_dataset_geo import RetinaPatchDatasetGeo
from models.vasca_net_geo import VasCANetGeo


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT = "./data_final/DRIVE"

CHECKPOINT = (
    "./checkpoints_q1/drive/geo/"
    "geo_vasca_best.pth"
)


def radius_metrics(pred, target, mask):

    valid = mask > 0.5

    pred = pred[valid]
    target = target[valid]

    if len(pred) == 0:
        return {
            "MAE": 0.0,
            "RMSE": 0.0,
            "Correlation": 0.0,
            "N": 0,
        }

    mae = np.mean(
        np.abs(pred - target)
    )

    rmse = np.sqrt(
        np.mean(
            (pred - target) ** 2
        )
    )

    if (
        np.std(pred) < 1e-8
        or np.std(target) < 1e-8
    ):
        correlation = 0.0
    else:
        correlation = np.corrcoef(
            pred,
            target
        )[0, 1]

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Correlation": float(correlation),
        "N": int(len(pred)),
    }


def thickness_groups(mask):

    vessel = (
        mask > 0.5
    ).astype(np.uint8)

    distance = cv2.distanceTransform(
        vessel,
        cv2.DIST_L2,
        5
    )

    return {
        "thin":
            (vessel == 1) &
            (distance <= 1.5),

        "medium":
            (vessel == 1) &
            (distance > 1.5) &
            (distance <= 3.0),

        "thick":
            (vessel == 1) &
            (distance > 3.0),
    }


def evaluate():

    print("=" * 75)
    print("Geo-VasCA-Net — RADIUS QUALITY EVALUATION")
    print("=" * 75)

    print("Device:", DEVICE)

    dataset = RetinaPatchDatasetGeo(
        root=ROOT,
        split="test",
        patch_size=64,
        patches_per_image=1,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = VasCANetGeo(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
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
        )
    )

    # --------------------------------------------------
    # Store metrics
    # --------------------------------------------------

    global_results = []

    grouped_results = {
        "thin": [],
        "medium": [],
        "thick": [],
    }

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    with torch.no_grad():

        for idx, batch in enumerate(loader):

            images, masks, radius_target = batch

            images = images.to(DEVICE)

            vessel_logits, radius_pred = model(
                images
            )

            pred_radius = (
                radius_pred[0, 0]
                .cpu()
                .numpy()
            )

            target_radius = (
                radius_target[0, 0]
                .numpy()
            )

            gt_mask = (
                masks[0, 0]
                .numpy()
            )

            metrics = radius_metrics(
                pred_radius,
                target_radius,
                gt_mask,
            )

            global_results.append(
                metrics
            )

            groups = thickness_groups(
                gt_mask
            )

            for name, group_mask in groups.items():

                group_result = radius_metrics(
                    pred_radius,
                    target_radius,
                    group_mask.astype(
                        np.float32
                    ),
                )

                grouped_results[
                    name
                ].append(
                    group_result
                )

            print(
                f"Image {idx + 1}: "
                f"MAE={metrics['MAE']:.5f} "
                f"RMSE={metrics['RMSE']:.5f} "
                f"Corr={metrics['Correlation']:.4f}"
            )

    # --------------------------------------------------
    # Aggregate
    # --------------------------------------------------

    def mean_metric(results, key):

        return float(
            np.mean(
                [
                    r[key]
                    for r in results
                ]
            )
        )

    print()
    print("=" * 75)
    print("OVERALL RADIUS QUALITY")
    print("=" * 75)

    overall_mae = mean_metric(
        global_results,
        "MAE"
    )

    overall_rmse = mean_metric(
        global_results,
        "RMSE"
    )

    overall_corr = mean_metric(
        global_results,
        "Correlation"
    )

    print(
        f"MAE         : {overall_mae:.6f}"
    )

    print(
        f"RMSE        : {overall_rmse:.6f}"
    )

    print(
        f"Correlation : {overall_corr:.4f}"
    )

    print()
    print("=" * 75)
    print("RADIUS QUALITY BY VESSEL THICKNESS")
    print("=" * 75)

    print(
        f"{'Class':<12}"
        f"{'MAE':>14}"
        f"{'RMSE':>14}"
        f"{'Correlation':>16}"
    )

    print("-" * 75)

    rows = []

    for name in [
        "thin",
        "medium",
        "thick",
    ]:

        results = grouped_results[name]

        mae = mean_metric(
            results,
            "MAE"
        )

        rmse = mean_metric(
            results,
            "RMSE"
        )

        corr = mean_metric(
            results,
            "Correlation"
        )

        print(
            f"{name:<12}"
            f"{mae:>14.6f}"
            f"{rmse:>14.6f}"
            f"{corr:>16.4f}"
        )

        rows.append({
            "category": name,
            "MAE": mae,
            "RMSE": rmse,
            "Correlation": corr,
        })

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    output_dir = (
        "./results_q1/drive/geo"
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_file = os.path.join(
        output_dir,
        "radius_quality.csv"
    )

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "MAE",
                "RMSE",
                "Correlation",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        "Saved:",
        output_file
    )


if __name__ == "__main__":
    evaluate()
