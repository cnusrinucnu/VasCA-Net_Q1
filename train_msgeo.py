import os
import csv
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.retina_dataset_geo import RetinaPatchDatasetGeo
from models.vasca_net_msgeo import VasCANetMSGeo
from utils.geometry_loss import GeoVasCALoss


# ============================================================
# Configuration
# ============================================================

ROOT = "./data_final/DRIVE"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

PATCH_SIZE = 64
PATCHES_PER_IMAGE = 200

BATCH_SIZE = 8

EPOCHS = 50
PATIENCE = 10

LR = 0.0005

LAMBDA_GEO = 0.10

NUM_WORKERS = 4

CHECKPOINT_DIR = (
    "./checkpoints_q1/drive/msgeo"
)

RESULT_DIR = (
    "./results_q1/drive/msgeo"
)


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

def calculate_metrics(
    logits,
    target,
    threshold=0.5,
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
        (1 - pred)
        *
        (1 - target)
    ).sum().item()

    fp = (
        pred
        *
        (1 - target)
    ).sum().item()

    fn = (
        (1 - pred)
        *
        target
    ).sum().item()

    eps = 1e-8

    dice = (
        2 * tp
        /
        (2 * tp + fp + fn + eps)
    )

    iou = (
        tp
        /
        (tp + fp + fn + eps)
    )

    se = (
        tp
        /
        (tp + fn + eps)
    )

    sp = (
        tn
        /
        (tn + fp + eps)
    )

    precision = (
        tp
        /
        (tp + fp + eps)
    )

    acc = (
        (tp + tn)
        /
        (tp + tn + fp + fn + eps)
    )

    return {
        "Dice": dice,
        "IoU": iou,
        "Se": se,
        "Sp": sp,
        "Precision": precision,
        "ACC": acc,
    }


# ============================================================
# Train
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0
    total_seg = 0.0
    total_geo = 0.0

    for step, batch in enumerate(
        loader,
        start=1
    ):

        images, masks, radius = batch

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
            radius,
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
    criterion,
):

    model.eval()

    total_loss = 0.0

    metrics_list = []

    for batch in loader:

        images, masks, radius = batch

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
            radius,
        )

        total_loss += (
            losses["total"].item()
        )

        metrics = calculate_metrics(
            vessel_logits,
            masks,
        )

        metrics_list.append(
            metrics
        )

    return {
        "loss":
            total_loss / len(loader),

        "Dice":
            np.mean(
                [x["Dice"] for x in metrics_list]
            ),

        "IoU":
            np.mean(
                [x["IoU"] for x in metrics_list]
            ),

        "Se":
            np.mean(
                [x["Se"] for x in metrics_list]
            ),

        "Sp":
            np.mean(
                [x["Sp"] for x in metrics_list]
            ),

        "Precision":
            np.mean(
                [x["Precision"] for x in metrics_list]
            ),

        "ACC":
            np.mean(
                [x["ACC"] for x in metrics_list]
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    seed_everything(SEED)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("MS-Geo-VasCA-Net — REAL TRAINING")
    print("=" * 70)

    print("Dataset: DRIVE")
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Patch size:",
        PATCH_SIZE
    )

    print(
        "Patches/image:",
        PATCHES_PER_IMAGE
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Learning rate:",
        LR
    )

    print(
        "Geometry weight:",
        LAMBDA_GEO
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = RetinaPatchDatasetGeo(
        root=ROOT,
        split="train",
        patch_size=PATCH_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
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
        num_workers=NUM_WORKERS,
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

    print()
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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = VasCANetMSGeo(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
        geometry_projection=16,
    ).to(DEVICE)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Trainable parameters:",
        f"{n_params / 1e6:.2f}M"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = GeoVasCALoss(
        bce_weight=0.5,
        lambda_geo=LAMBDA_GEO,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    # Same LR reduction strategy used in previous
    # experiments.

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_dice = -1.0

    best_epoch = 0

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        t0 = time.time()

        print()
        print(
            f"Epoch {epoch:03d}/{EPOCHS}"
        )

        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_result = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_result["Dice"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        elapsed = (
            time.time() - t0
        )

        print(
            f"Train Loss="
            f"{train_result['loss']:.4f} "
            f"Seg="
            f"{train_result['seg']:.4f} "
            f"Geo="
            f"{train_result['geo']:.4f}"
        )

        print(
            f"Val   Loss="
            f"{val_result['loss']:.4f} "
            f"Dice="
            f"{val_result['Dice']:.4f} "
            f"IoU="
            f"{val_result['IoU']:.4f} "
            f"Se="
            f"{val_result['Se']:.4f} "
            f"Sp="
            f"{val_result['Sp']:.4f} "
            f"Precision="
            f"{val_result['Precision']:.4f} "
            f"ACC="
            f"{val_result['ACC']:.4f}"
        )

        print(
            f"LR={current_lr:.7f} "
            f"Time={elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_seg": train_result["seg"],
            "train_geo": train_result["geo"],
            "val_loss": val_result["loss"],
            "val_dice": val_result["Dice"],
            "val_iou": val_result["IoU"],
            "val_se": val_result["Se"],
            "val_sp": val_result["Sp"],
            "val_precision": val_result["Precision"],
            "val_acc": val_result["ACC"],
            "lr": current_lr,
        })

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_result["Dice"] > best_dice:

            best_dice = (
                val_result["Dice"]
            )

            best_epoch = epoch

            patience_counter = 0

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "msgeo_vasca_best.pth",
            )

            torch.save(
                {
                    "model":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "best_val_dice":
                        best_dice,

                    "metrics":
                        val_result,

                    "lambda_geo":
                        LAMBDA_GEO,
                },
                checkpoint_path,
            )

            print(
                ">>> NEW BEST CHECKPOINT"
            )

            print(
                f">>> Val Dice: "
                f"{best_dice:.4f}"
            )

            print(
                ">>> Saved:",
                checkpoint_path
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

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = os.path.join(
        RESULT_DIR,
        "training_history.csv"
    )

    with open(
        history_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys()
        )

        writer.writeheader()

        writer.writerows(history)

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MS-GEO TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Dice: "
        f"{best_dice:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        "Checkpoint:",
        os.path.join(
            CHECKPOINT_DIR,
            "msgeo_vasca_best.pth"
        )
    )

    print(
        "History:",
        history_path
    )


if __name__ == "__main__":
    main()
