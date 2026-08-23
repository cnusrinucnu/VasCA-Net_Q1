import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.retina_dataset_geo import RetinaPatchDatasetGeo
from models.vasca_net_geo import VasCANetGeo
from utils.geometry_loss import GeoVasCALoss


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================

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
        (1.0 - pred) * (1.0 - target)
    ).sum().item()

    fp = (
        pred * (1.0 - target)
    ).sum().item()

    fn = (
        (1.0 - pred) * target
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


# ============================================================
# One training epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_seg = 0.0
    total_geo = 0.0

    for images, masks, radius in loader:

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

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Non-finite training loss detected."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += loss.item()
        total_seg += losses[
            "segmentation"
        ].item()

        total_geo += losses[
            "geometry"
        ].item()

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
    device,
    threshold=0.5,
):

    model.eval()

    total_loss = 0.0
    total_seg = 0.0
    total_geo = 0.0

    metric_sum = {
        "Dice": 0.0,
        "IoU": 0.0,
        "Se": 0.0,
        "Sp": 0.0,
        "Precision": 0.0,
        "ACC": 0.0,
    }

    n_batches = 0

    for images, masks, radius in loader:

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

        losses = criterion(
            vessel_logits,
            masks,
            radius_pred,
            radius,
        )

        total_loss += losses[
            "total"
        ].item()

        total_seg += losses[
            "segmentation"
        ].item()

        total_geo += losses[
            "geometry"
        ].item()

        metrics = calculate_metrics(
            vessel_logits,
            masks,
            threshold=threshold,
        )

        for key in metric_sum:
            metric_sum[key] += metrics[key]

        n_batches += 1

    results = {
        "loss": total_loss / n_batches,
        "seg": total_seg / n_batches,
        "geo": total_geo / n_batches,
    }

    for key in metric_sum:
        results[key] = (
            metric_sum[key]
            /
            n_batches
        )

    return results


# ============================================================
# Main
# ============================================================

def main():

    set_seed(42)

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

    patch_size = 64

    patches_per_image = 200

    batch_size = 8

    epochs = 50

    lr = 5e-4

    weight_decay = 1e-4

    lambda_geo = 0.10

    threshold = 0.5

    patience = 10

    num_workers = 4

    checkpoint_dir = (
        "./checkpoints_q1/"
        + dataset_name.lower()
        + "/geo"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    print("=" * 72)
    print("Geo-VasCA-Net — REAL TRAINING")
    print("=" * 72)

    print("Dataset:", dataset_name)
    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Patch size:", patch_size)
    print(
        "Patches/image:",
        patches_per_image,
    )

    print("Batch size:", batch_size)
    print("Epochs:", epochs)
    print("Learning rate:", lr)
    print(
        "Geometry weight:",
        lambda_geo,
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = RetinaPatchDatasetGeo(
        root=root,
        split="train",
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    val_dataset = RetinaPatchDatasetGeo(
        root=root,
        split="val",
        patch_size=patch_size,
        patches_per_image=1,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print()
    print(
        "Train samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    print(
        "Train batches:",
        len(train_loader),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = VasCANetGeo(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{n_params / 1e6:.2f}M"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = GeoVasCALoss(
        bce_weight=0.5,
        lambda_geo=lambda_geo,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_f1 = -1.0

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        start_time = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            threshold=threshold,
        )

        scheduler.step(
            val_metrics["Dice"]
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        print()
        print(
            f"Epoch {epoch:03d}/{epochs}"
        )

        print(
            f"Train "
            f"Loss={train_metrics['loss']:.4f} "
            f"Seg={train_metrics['seg']:.4f} "
            f"Geo={train_metrics['geo']:.4f}"
        )

        print(
            f"Val   "
            f"Loss={val_metrics['loss']:.4f} "
            f"Dice={val_metrics['Dice']:.4f} "
            f"IoU={val_metrics['IoU']:.4f} "
            f"Se={val_metrics['Se']:.4f} "
            f"Sp={val_metrics['Sp']:.4f}"
        )

        print(
            f"Precision={val_metrics['Precision']:.4f} "
            f"ACC={val_metrics['ACC']:.4f} "
            f"LR={current_lr:.7f} "
            f"Time={elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics[
                    "loss"
                ],
                "train_seg": train_metrics[
                    "seg"
                ],
                "train_geo": train_metrics[
                    "geo"
                ],
                "val_loss": val_metrics[
                    "loss"
                ],
                "val_dice": val_metrics[
                    "Dice"
                ],
                "val_iou": val_metrics[
                    "IoU"
                ],
                "val_se": val_metrics[
                    "Se"
                ],
                "val_sp": val_metrics[
                    "Sp"
                ],
                "val_precision": val_metrics[
                    "Precision"
                ],
                "val_acc": val_metrics[
                    "ACC"
                ],
                "lr": current_lr,
            }
        )

        # ----------------------------------------------------
        # Save best model using VALIDATION Dice
        # ----------------------------------------------------

        if val_metrics["Dice"] > best_f1:

            best_f1 = val_metrics["Dice"]

            patience_counter = 0

            checkpoint_path = os.path.join(
                checkpoint_dir,
                "geo_vasca_best.pth",
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_dice": best_f1,
                    "val_metrics": val_metrics,
                    "lambda_geo": lambda_geo,
                    "patch_size": patch_size,
                    "patches_per_image": patches_per_image,
                },
                checkpoint_path,
            )

            print(
                ">>> NEW BEST CHECKPOINT"
            )

            print(
                ">>> Val Dice:",
                f"{best_f1:.4f}",
            )

            print(
                ">>> Saved:",
                checkpoint_path,
            )

        else:

            patience_counter += 1

            print(
                f"No improvement: "
                f"{patience_counter}/{patience}"
            )

            if patience_counter >= patience:

                print(
                    "Early stopping."
                )

                break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    import csv

    history_path = os.path.join(
        checkpoint_dir,
        "training_history.csv",
    )

    if history:

        keys = list(
            history[0].keys()
        )

        with open(
            history_path,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=keys,
            )

            writer.writeheader()

            writer.writerows(
                history
            )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"Best validation Dice: "
        f"{best_f1:.4f}"
    )

    print(
        "Checkpoint:",
        os.path.join(
            checkpoint_dir,
            "geo_vasca_best.pth",
        ),
    )

    print(
        "History:",
        history_path,
    )


if __name__ == "__main__":
    main()
