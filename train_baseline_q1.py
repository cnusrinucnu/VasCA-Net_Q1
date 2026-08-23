import os
import random
import time
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.retina_dataset_geo import RetinaPatchDatasetGeo
from models.vasca_net import VasCANet
from utils.losses import BCEDiceLoss


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    tp = (pred * target).sum().item()

    tn = (
        (1 - pred) *
        (1 - target)
    ).sum().item()

    fp = (
        pred *
        (1 - target)
    ).sum().item()

    fn = (
        (1 - pred) *
        target
    ).sum().item()

    dice = (
        2 * tp /
        (2 * tp + fp + fn + eps)
    )

    iou = (
        tp /
        (tp + fp + fn + eps)
    )

    se = (
        tp /
        (tp + fn + eps)
    )

    sp = (
        tn /
        (tn + fp + eps)
    )

    precision = (
        tp /
        (tp + fp + eps)
    )

    acc = (
        (tp + tn) /
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


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    for images, masks, radius in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Non-finite baseline loss."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        running_loss += loss.item()

    return (
        running_loss /
        max(len(loader), 1)
    )


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
):

    model.eval()

    running_loss = 0.0

    metrics_sum = {
        "Dice": 0.0,
        "IoU": 0.0,
        "Se": 0.0,
        "Sp": 0.0,
        "Precision": 0.0,
        "ACC": 0.0,
    }

    n = 0

    for images, masks, radius in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        running_loss += loss.item()

        metrics = calculate_metrics(
            logits,
            masks,
            threshold,
        )

        for key in metrics:

            metrics_sum[key] += (
                metrics[key]
            )

        n += 1

    results = {
        "loss":
            running_loss / max(n, 1)
    }

    for key in metrics_sum:

        results[key] = (
            metrics_sum[key] /
            max(n, 1)
        )

    return results


def main():

    set_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root = "./data_final/DRIVE"

    patch_size = 64

    patches_per_image = 200

    batch_size = 8

    epochs = 50

    lr = 5e-4

    weight_decay = 1e-4

    threshold = 0.5

    patience = 10

    checkpoint_dir = (
        "./checkpoints_q1/"
        "drive/baseline"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    print("=" * 70)
    print("CONTROLLED VasCA-Net BASELINE")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_set = RetinaPatchDatasetGeo(
        root=root,
        split="train",
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    val_set = RetinaPatchDatasetGeo(
        root=root,
        split="val",
        patch_size=patch_size,
        patches_per_image=1,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(
        "Train samples:",
        len(train_set)
    )

    print(
        "Validation samples:",
        len(val_set)
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = VasCANet(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{params / 1e6:.2f}M"
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = BCEDiceLoss(
        bce_weight=0.5
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

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

    best_dice = -1.0

    patience_counter = 0

    history = []

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(
        1,
        epochs + 1
    ):

        t0 = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val = validate(
            model,
            val_loader,
            criterion,
            device,
            threshold,
        )

        scheduler.step(
            val["Dice"]
        )

        lr_now = optimizer.param_groups[
            0
        ]["lr"]

        elapsed = (
            time.time() - t0
        )

        print()
        print(
            f"Epoch {epoch:03d}/{epochs}"
        )

        print(
            f"Train Loss={train_loss:.4f}"
        )

        print(
            f"Val Loss={val['loss']:.4f} "
            f"Dice={val['Dice']:.4f} "
            f"IoU={val['IoU']:.4f} "
            f"Se={val['Se']:.4f} "
            f"Sp={val['Sp']:.4f} "
            f"Precision={val['Precision']:.4f} "
            f"ACC={val['ACC']:.4f}"
        )

        print(
            f"LR={lr_now:.7f} "
            f"Time={elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_dice": val["Dice"],
                "val_iou": val["IoU"],
                "val_se": val["Se"],
                "val_sp": val["Sp"],
                "val_precision": val["Precision"],
                "val_acc": val["ACC"],
                "lr": lr_now,
            }
        )

        if val["Dice"] > best_dice:

            best_dice = val["Dice"]

            patience_counter = 0

            path = os.path.join(
                checkpoint_dir,
                "vasca_baseline_best.pth",
            )

            torch.save(
                {
                    "model":
                        model.state_dict(),
                    "epoch":
                        epoch,
                    "best_val_dice":
                        best_dice,
                    "val_metrics":
                        val,
                },
                path,
            )

            print(
                f">>> NEW BEST "
                f"Dice={best_dice:.4f}"
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

    # --------------------------------------------------
    # Save history
    # --------------------------------------------------

    history_path = os.path.join(
        checkpoint_dir,
        "training_history.csv",
    )

    with open(
        history_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    print()
    print("=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Dice: "
        f"{best_dice:.4f}"
    )

    print(
        "Checkpoint:",
        os.path.join(
            checkpoint_dir,
            "vasca_baseline_best.pth",
        )
    )


if __name__ == "__main__":
    main()
