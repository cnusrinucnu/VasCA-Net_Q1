import os
import time
import csv
import torch
import yaml

from torch.utils.data import DataLoader

from datasets.retina_dataset_geo import RetinaPatchDatasetGeo
from models.vasca_net import VasCANet
from utils.losses import BCEDiceLoss


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT = "./data_final/DRIVE"

BATCH_SIZE = 8
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 200

EPOCHS = 50
LR = 5e-4
PATIENCE = 10

CHECKPOINT_DIR = (
    "./checkpoints_q1/drive/"
    "corrected_baseline"
)

RESULT_DIR = (
    "./results_q1/drive/"
    "corrected_baseline"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def calculate_metrics(logits, target):

    probs = torch.sigmoid(logits)

    pred = (
        probs >= 0.5
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

    eps = 1e-7

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


def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    all_metrics = []

    with torch.no_grad():

        for batch in loader:

            images = batch[0].to(DEVICE)
            masks = batch[1].to(DEVICE)

            logits = model(images)

            loss = criterion(
                logits,
                masks
            )

            total_loss += loss.item()

            all_metrics.append(
                calculate_metrics(
                    logits,
                    masks
                )
            )

    metrics = {}

    for key in all_metrics[0]:

        metrics[key] = sum(
            x[key]
            for x in all_metrics
        ) / len(all_metrics)

    metrics["loss"] = (
        total_loss /
        len(loader)
    )

    return metrics


def main():

    print("=" * 70)
    print("CORRECTED DRIVE BASELINE")
    print("=" * 70)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # DATASETS
    # --------------------------------------------------

    train_dataset = RetinaPatchDatasetGeo(
        root=ROOT,
        split="train_q1",
        patch_size=PATCH_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    val_dataset = RetinaPatchDatasetGeo(
        root=ROOT,
        split="val_q1",
        patch_size=PATCH_SIZE,
        patches_per_image=1,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    print(
        "Train images:",
        len(train_dataset.image_files)
    )

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset.image_files)
    )

    # --------------------------------------------------
    # LOADERS
    # --------------------------------------------------

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
        num_workers=2,
        pin_memory=True,
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = VasCANet(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        msca_ratio=8,
        econv_ratio=2,
    ).to(DEVICE)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Parameters: {params / 1e6:.2f}M"
    )

    # --------------------------------------------------
    # LOSS
    # --------------------------------------------------

    criterion = BCEDiceLoss(
        bce_weight=0.5
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5,
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
    # TRAIN
    # --------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        start = time.time()

        model.train()

        running_loss = 0.0

        for images, masks, _ in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                masks
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss /
            len(train_loader)
        )

        val = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val["Dice"]
        )

        elapsed = time.time() - start

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:03d}/{EPOCHS}"
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
            f"LR={lr:.7f} "
            f"Time={elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **val,
            "lr": lr,
        })

        if val["Dice"] > best_dice:

            best_dice = val["Dice"]

            patience_counter = 0

            path = os.path.join(
                CHECKPOINT_DIR,
                "vasca_corrected_best.pth"
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
                        val,
                },
                path,
            )

            print(
                ">>> NEW BEST"
            )

            print(
                ">>> Saved:",
                path
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

    # --------------------------------------------------
    # HISTORY
    # --------------------------------------------------

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

        writer.writerows(
            history
        )

    print()
    print("=" * 70)
    print("CORRECTED BASELINE COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Dice: "
        f"{best_dice:.4f}"
    )

    print(
        "Checkpoint:",
        os.path.join(
            CHECKPOINT_DIR,
            "vasca_corrected_best.pth"
        )
    )


if __name__ == "__main__":
    main()
