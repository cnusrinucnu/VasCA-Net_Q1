import os
import csv
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.retina_dataset_geo import RetinaPatchDatasetGeo
from models.vasca_net_geofusion import VasCANetGeoFusion
from utils.geometry_loss import GeoVasCALoss


# ==========================================================
# SEED
# ==========================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(
    logits,
    target,
    threshold=0.5,
    eps=1e-7,
):

    probs = torch.sigmoid(logits)

    pred = (probs >= threshold).float()

    target = target.float()

    tp = (pred * target).sum().item()

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


# ==========================================================
# MAIN
# ==========================================================

def main():

    set_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root = "./data_final/DRIVE"

    checkpoint_dir = (
        "./checkpoints_q1/"
        "drive/geofusion"
    )

    results_dir = (
        "./results_q1/"
        "drive/geofusion"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    print("=" * 70)
    print("GEOFUSION-VASCA-NET TRAINING")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ------------------------------------------------------
    # DATASETS
    # ------------------------------------------------------

    train_dataset = RetinaPatchDatasetGeo(
        root=root,
        split="train_q1",
        patch_size=64,
        patches_per_image=200,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    val_dataset = RetinaPatchDatasetGeo(
        root=root,
        split="val_q1",
        patch_size=64,
        patches_per_image=1,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(
        "Training images:",
        len(train_dataset.image_files)
    )

    print(
        "Training patches:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset.image_files)
    )

    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------

    model = VasCANetGeoFusion(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{total_params / 1e6:.2f}M"
    )

    # ------------------------------------------------------
    # LOSS
    # ------------------------------------------------------

    criterion = GeoVasCALoss(
        lambda_geo=0.2
    )

    # ------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # ------------------------------------------------------
    # TRAINING SETTINGS
    # ------------------------------------------------------

    epochs = 50

    patience = 10

    best_dice = -1.0

    best_epoch = 0

    no_improve = 0

    history = []

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "geofusion_vasca_best.pth",
    )

    # ======================================================
    # EPOCH LOOP
    # ======================================================

    for epoch in range(1, epochs + 1):

        model.train()

        train_losses = []

        train_seg_losses = []

        train_geo_losses = []

        for batch in train_loader:

            images, masks, radius = batch

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

            optimizer.zero_grad()

            vessel_logits, radius_pred = model(
                images
            )

            loss_output = criterion(
                vessel_logits,
                masks,
                radius_pred,
                radius,
            )

            loss = loss_output["total"]
            seg_loss = loss_output["segmentation"]
            geo_loss = loss_output["geometry"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

            train_seg_losses.append(
                seg_loss.item()
            )

            train_geo_losses.append(
                geo_loss.item()
            )

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        model.eval()

        val_losses = []

        all_metrics = []

        with torch.no_grad():

            for batch in val_loader:

                images, masks, radius = batch

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

                loss_output = criterion(
                    vessel_logits,
                    masks,
                    radius_pred,
                    radius,
                )

                loss = loss_output["total"]

                val_losses.append(
                    loss.item()
                )

                metrics = calculate_metrics(
                    vessel_logits,
                    masks,
                    threshold=0.5,
                )

                all_metrics.append(
                    metrics
                )

        # --------------------------------------------------
        # AGGREGATE
        # --------------------------------------------------

        train_loss = float(
            np.mean(train_losses)
        )

        train_seg = float(
            np.mean(train_seg_losses)
        )

        train_geo = float(
            np.mean(train_geo_losses)
        )

        val_loss = float(
            np.mean(val_losses)
        )

        mean_metrics = {}

        for key in all_metrics[0].keys():

            mean_metrics[key] = float(
                np.mean(
                    [
                        x[key]
                        for x in all_metrics
                    ]
                )
            )

        val_dice = mean_metrics["Dice"]

        scheduler.step(
            val_dice
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print()

        print(
            f"Epoch {epoch:03d}/{epochs}"
        )

        print(
            f"Train Loss={train_loss:.4f} "
            f"Seg={train_seg:.4f} "
            f"Geo={train_geo:.4f}"
        )

        print(
            f"Val Loss={val_loss:.4f} "
            f"Dice={mean_metrics['Dice']:.4f} "
            f"IoU={mean_metrics['IoU']:.4f} "
            f"Se={mean_metrics['Se']:.4f} "
            f"Sp={mean_metrics['Sp']:.4f} "
            f"Precision={mean_metrics['Precision']:.4f} "
            f"ACC={mean_metrics['ACC']:.4f}"
        )

        print(
            f"LR={current_lr:.7f}"
        )

        # --------------------------------------------------
        # HISTORY
        # --------------------------------------------------

        row = {
            "Epoch": epoch,
            "Train_Loss": train_loss,
            "Train_Seg": train_seg,
            "Train_Geo": train_geo,
            "Val_Loss": val_loss,
            "Dice": mean_metrics["Dice"],
            "IoU": mean_metrics["IoU"],
            "Se": mean_metrics["Se"],
            "Sp": mean_metrics["Sp"],
            "Precision": mean_metrics["Precision"],
            "ACC": mean_metrics["ACC"],
            "LR": current_lr,
        }

        history.append(
            row
        )

        # --------------------------------------------------
        # CHECKPOINT
        # --------------------------------------------------

        if val_dice > best_dice:

            best_dice = val_dice

            best_epoch = epoch

            no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_dice": best_dice,
                },
                checkpoint_path,
            )

            print()

            print(
                f">> New best Dice: "
                f"{best_dice:.4f}"
            )

            print(
                f">> Saved: "
                f"{checkpoint_path}"
            )

        else:

            no_improve += 1

            print(
                f"No improvement: "
                f"{no_improve}/{patience}"
            )

        if no_improve >= patience:

            print(
                "Early stopping."
            )

            break

    # ======================================================
    # SAVE HISTORY
    # ======================================================

    history_path = os.path.join(
        results_dir,
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

        for row in history:

            writer.writerow(
                row
            )

    print()

    print("=" * 70)
    print("GEOFUSION-VASCA-NET TRAINING COMPLETE")
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
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"History: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()
