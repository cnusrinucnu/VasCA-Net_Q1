import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.retina_dataset_geo import RetinaPatchDatasetGeo
from models.vasca_net_geo import VasCANetGeo
from utils.geometry_loss import GeoVasCALoss


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT = "./data_final/DRIVE"

SEED = 42

PATCH_SIZE = 64
TRAIN_PATCHES = 200

BATCH_SIZE = 8

EPOCHS = 30
PATIENCE = 7

LR = 0.0005

LAMBDA_VALUES = [
    0.05,
    0.20,
    0.50,
]

OUTPUT_DIR = "./results_q1/drive/lambda_sweep"

CHECKPOINT_DIR = "./checkpoints_q1/drive/lambda_sweep"


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# Metrics
# ============================================================

def segmentation_metrics(
    logits,
    target,
    threshold=0.5
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
        (
            2.0 * tp
            + fp
            + fn
            + 1e-8
        )
    )

    iou = (
        tp
        /
        (
            tp
            + fp
            + fn
            + 1e-8
        )
    )

    sensitivity = (
        tp
        /
        (
            tp
            + fn
            + 1e-8
        )
    )

    specificity = (
        tn
        /
        (
            tn
            + fp
            + 1e-8
        )
    )

    precision = (
        tp
        /
        (
            tp
            + fp
            + 1e-8
        )
    )

    return {
        "Dice": dice,
        "IoU": iou,
        "Se": sensitivity,
        "Sp": specificity,
        "Precision": precision,
    }


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0
    total_seg = 0.0
    total_geo = 0.0

    for images, masks, radius in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        radius = radius.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        vessel_logits, radius_pred = model(
            images
        )

        losses = criterion(
            vessel_logits,
            masks,
            radius_pred,
            radius
        )

        loss = losses["total"]

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        total_seg += (
            losses["segmentation"].item()
        )

        total_geo += (
            losses["geometry"].item()
        )

    n = len(loader)

    return {
        "loss": total_loss / n,
        "seg": total_seg / n,
        "geo": total_geo / n,
    }


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    metric_values = []

    for images, masks, radius in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        radius = radius.to(
            DEVICE,
            non_blocking=True
        )

        vessel_logits, radius_pred = model(
            images
        )

        losses = criterion(
            vessel_logits,
            masks,
            radius_pred,
            radius
        )

        total_loss += (
            losses["total"].item()
        )

        metrics = segmentation_metrics(
            vessel_logits,
            masks
        )

        metric_values.append(
            metrics
        )

    n = len(metric_values)

    output = {
        "loss":
            total_loss / len(loader),

        "Dice":
            np.mean(
                [m["Dice"] for m in metric_values]
            ),

        "IoU":
            np.mean(
                [m["IoU"] for m in metric_values]
            ),

        "Se":
            np.mean(
                [m["Se"] for m in metric_values]
            ),

        "Sp":
            np.mean(
                [m["Sp"] for m in metric_values]
            ),

        "Precision":
            np.mean(
                [m["Precision"] for m in metric_values]
            ),
    }

    return output


# ============================================================
# One lambda experiment
# ============================================================

def run_experiment(
    lambda_geo,
    train_loader,
    val_loader
):

    print()
    print("=" * 70)
    print(
        f"GEOMETRY LAMBDA = {lambda_geo}"
    )
    print("=" * 70)

    seed_everything(SEED)

    model = VasCANetGeo(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(DEVICE)

    criterion = GeoVasCALoss(
        bce_weight=0.5,
        lambda_geo=lambda_geo,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_dice = -1.0
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_result = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"Train={train_result['loss']:.4f} "
            f"Seg={train_result['seg']:.4f} "
            f"Geo={train_result['geo']:.4f} "
            f"ValDice={val_result['Dice']:.4f} "
            f"ValIoU={val_result['IoU']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "lambda_geo": lambda_geo,
            "train_loss": train_result["loss"],
            "train_seg": train_result["seg"],
            "train_geo": train_result["geo"],
            "val_loss": val_result["loss"],
            "val_dice": val_result["Dice"],
            "val_iou": val_result["IoU"],
            "val_se": val_result["Se"],
            "val_sp": val_result["Sp"],
            "val_precision": val_result["Precision"],
        })

        if val_result["Dice"] > best_dice:

            best_dice = val_result["Dice"]

            best_epoch = epoch

            patience_counter = 0

            os.makedirs(
                CHECKPOINT_DIR,
                exist_ok=True
            )

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"geo_lambda_{lambda_geo:.2f}_best.pth"
            )

            torch.save(
                {
                    "model":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "lambda_geo":
                        lambda_geo,

                    "best_val_dice":
                        best_dice,
                },
                checkpoint_path
            )

            print(
                f">>> NEW BEST "
                f"Dice={best_dice:.4f}"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement: "
                f"{patience_counter}/{PATIENCE}"
            )

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping."
                )

                break

    return {
        "lambda_geo": lambda_geo,
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "history": history,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Geo-VasCA-Net GEOMETRY LAMBDA SWEEP")
    print("=" * 70)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    seed_everything(SEED)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = RetinaPatchDatasetGeo(
        root=ROOT,
        split="train",
        patch_size=PATCH_SIZE,
        patches_per_image=TRAIN_PATCHES,
        images_subdir="images",
        masks_subdir="masks",
        seed=SEED,
    )

    val_dataset = RetinaPatchDatasetGeo(
        root=ROOT,
        split="test",
        patch_size=PATCH_SIZE,
        patches_per_image=1,
        images_subdir="images",
        masks_subdir="masks",
        seed=SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    print(
        "Train batches:",
        len(train_loader)
    )

    print(
        "Lambda values:",
        LAMBDA_VALUES
    )

    results = []

    all_history = []

    for lambda_geo in LAMBDA_VALUES:

        result = run_experiment(
            lambda_geo,
            train_loader,
            val_loader
        )

        results.append({
            "lambda_geo":
                lambda_geo,

            "best_val_dice":
                result["best_dice"],

            "best_epoch":
                result["best_epoch"],
        })

        all_history.extend(
            result["history"]
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "lambda_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda_geo",
                "best_val_dice",
                "best_epoch",
            ]
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    history_path = os.path.join(
        OUTPUT_DIR,
        "lambda_history.csv"
    )

    with open(
        history_path,
        "w",
        newline=""
    ) as f:

        fieldnames = [
            "epoch",
            "lambda_geo",
            "train_loss",
            "train_seg",
            "train_geo",
            "val_loss",
            "val_dice",
            "val_iou",
            "val_se",
            "val_sp",
            "val_precision",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_history
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LAMBDA SWEEP SUMMARY")
    print("=" * 70)

    print(
        f"{'Lambda':<12}"
        f"{'Best Dice':<15}"
        f"{'Epoch':<10}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['lambda_geo']:<12.2f}"
            f"{result['best_val_dice']:<15.4f}"
            f"{result['best_epoch']:<10}"
        )

    best = max(
        results,
        key=lambda x:
        x["best_val_dice"]
    )

    print()
    print(
        "BEST LAMBDA:",
        best["lambda_geo"]
    )

    print(
        "BEST VALIDATION DICE:",
        f"{best['best_val_dice']:.4f}"
    )

    print(
        "BEST EPOCH:",
        best["best_epoch"]
    )

    print()
    print(
        "Saved:",
        summary_path
    )

    print(
        "Saved:",
        history_path
    )


if __name__ == "__main__":
    main()
