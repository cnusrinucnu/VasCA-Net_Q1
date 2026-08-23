"""
Final training protocol for VasCA-Net-DS + Thickness-Aware Loss.

Protocol:
    train -> model optimization
    val   -> checkpoint selection / early stopping
    test  -> NEVER used during training

Usage:
    python train_final.py --config configs/stare_final.yaml
"""

import argparse
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from models import VasCANetDS
from datasets import RetinalVesselDataset
from utils.metrics import MetricAccumulator
from utils.thickness_loss import ThicknessAwareBCELoss


def deep_supervision_loss(
    main_out,
    aux_outputs,
    masks,
    criterion,
):
    aux2, aux3, aux4 = aux_outputs

    loss_main = criterion(main_out, masks)
    loss_aux2 = criterion(aux2, masks)
    loss_aux3 = criterion(aux3, masks)
    loss_aux4 = criterion(aux4, masks)

    return (
        0.4 * loss_main
        + 0.3 * loss_aux2
        + 0.2 * loss_aux3
        + 0.1 * loss_aux4
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    log_every=20,
):
    model.train()

    running_loss = 0.0

    for step, (imgs, masks) in enumerate(loader, 1):

        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        main_out, aux_outputs = model(
            imgs,
            return_aux=True,
        )

        loss = deep_supervision_loss(
            main_out,
            aux_outputs,
            masks,
            criterion,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % log_every == 0:
            print(
                f"  step {step}/{len(loader)} "
                f"- loss: "
                f"{running_loss / step:.4f}"
            )

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
):
    model.eval()

    running_loss = 0.0
    acc = MetricAccumulator()

    for imgs, masks in loader:

        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(
            imgs,
            return_aux=False,
        )

        loss = criterion(
            logits,
            masks,
        )

        running_loss += loss.item()

        probs = torch.sigmoid(logits)

        acc.update(
            probs,
            masks,
            threshold=threshold,
        )

    metrics = acc.compute()

    metrics["loss"] = (
        running_loss
        / max(len(loader), 1)
    )

    return metrics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["train"]["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    os.makedirs(
        cfg["train"]["checkpoint_dir"],
        exist_ok=True,
    )

    # ==================================================
    # TRAIN DATA
    # ==================================================

    train_set = RetinalVesselDataset(
        root=cfg["data"]["root"],
        split="train",
        patch_size=cfg["data"]["patch_size"],
        patches_per_image=cfg["data"]["patches_per_image"],
        images_subdir=cfg["data"]["images_subdir"],
        masks_subdir=cfg["data"]["masks_subdir"],
    )

    # ==================================================
    # VALIDATION DATA
    # ==================================================

    val_set = RetinalVesselDataset(
        root=cfg["data"]["root"],
        split="val",
        patch_size=cfg["data"]["patch_size"],
        images_subdir=cfg["data"]["images_subdir"],
        masks_subdir=cfg["data"]["masks_subdir"],
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    # ==================================================
    # MODEL
    # ==================================================

    model = VasCANetDS(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        msca_ratio=cfg["model"]["msca_ratio"],
        econv_ratio=cfg["model"]["econv_ratio"],
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"VasCA-Net-DS-TA parameters: "
        f"{n_params / 1e6:.2f}M"
    )

    print(
        f"Train images: {len(train_set.images)}"
    )

    print(
        f"Validation images: "
        f"{len(val_set.images)}"
    )

    print(
        "Official test set is NOT used during training."
    )

    # ==================================================
    # LOSS
    # ==================================================

    alpha = cfg["train"].get(
        "thickness_alpha",
        0.25,
    )

    criterion = ThicknessAwareBCELoss(
        alpha=alpha
    )

    print(
        f"Thickness-aware alpha: {alpha}"
    )

    print(
        "Deep supervision weights: "
        "0.4 / 0.3 / 0.2 / 0.1"
    )

    # ==================================================
    # OPTIMIZER
    # ==================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
    )

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    patience = cfg["train"][
        "early_stopping_patience"
    ]

    # ==================================================
    # TRAINING
    # ==================================================

    for epoch in range(
        1,
        cfg["train"]["epochs"] + 1,
    ):

        t0 = time.time()

        print(
            f"Epoch {epoch}/"
            f"{cfg['train']['epochs']}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            log_every=cfg["train"]["log_every"],
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=cfg["train"]["threshold"],
        )

        dt = time.time() - t0

        print(
            f"  train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"Se={val_metrics['Se']:.4f} "
            f"Sp={val_metrics['Sp']:.4f} "
            f"Precision={val_metrics['Precision']:.4f} "
            f"F1={val_metrics['F1']:.4f} "
            f"ACC={val_metrics['ACC']:.4f} "
            f"AUC={val_metrics.get('AUC', float('nan')):.4f} "
            f"({dt:.1f}s)"
        )

        # ==================================================
        # CHECKPOINT SELECTION USING VALIDATION ONLY
        # ==================================================

        if val_metrics["F1"] > best_f1:

            best_f1 = val_metrics["F1"]
            best_epoch = epoch
            patience_counter = 0

            ckpt_path = os.path.join(
                cfg["train"]["checkpoint_dir"],
                "vasca_net_final_best.pth",
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "alpha": alpha,
                },
                ckpt_path,
            )

            print(
                f"  New best validation F1="
                f"{best_f1:.4f}"
            )

            print(
                f"  Saved: {ckpt_path}"
            )

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(
                    f"Early stopping at epoch "
                    f"{epoch}"
                )

                break

    print()
    print(
        "Training complete."
    )

    print(
        f"Best validation F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Best validation epoch: "
        f"{best_epoch}"
    )


if __name__ == "__main__":
    main()
