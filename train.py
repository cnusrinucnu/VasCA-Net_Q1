"""
Training script for VasCA-Net.

Usage:
    python train.py --config configs/default.yaml
"""
import argparse
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from models import VasCANet
from datasets import RetinalVesselDataset
from utils.losses import BCELoss, DiceLoss, BCEDiceLoss
from utils.metrics import MetricAccumulator


def build_loss(name: str):
    return {"bce": BCELoss, "dice": DiceLoss, "bce_dice": BCEDiceLoss}[name]()


def train_one_epoch(model, loader, optimizer, criterion, device, log_every=20):
    model.train()
    running_loss = 0.0
    for step, (imgs, masks) in enumerate(loader, 1):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % log_every == 0:
            print(f"  step {step}/{len(loader)} - loss: {running_loss / step:.4f}")

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    acc = MetricAccumulator()

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        acc.update(probs, masks, threshold=threshold)

    metrics = acc.compute()
    metrics["loss"] = running_loss / max(len(loader), 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)

    train_set = RetinalVesselDataset(
        root=cfg["data"]["root"],
        split="train",
        patch_size=cfg["data"]["patch_size"],
        patches_per_image=cfg["data"]["patches_per_image"],
        images_subdir=cfg["data"]["images_subdir"],
        masks_subdir=cfg["data"]["masks_subdir"],
    )
    val_set = RetinalVesselDataset(
        root=cfg["data"]["root"],
        split="test",
        patch_size=cfg["data"]["patch_size"],
        images_subdir=cfg["data"]["images_subdir"],
        masks_subdir=cfg["data"]["masks_subdir"],
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=cfg["train"]["num_workers"],
    )

    model = VasCANet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        msca_ratio=cfg["model"]["msca_ratio"],
        econv_ratio=cfg["model"]["econv_ratio"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"VasCA-Net parameters: {n_params / 1e6:.2f}M")

    criterion = build_loss(cfg["train"]["loss"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    best_f1 = -1.0
    patience_counter = 0
    patience = cfg["train"]["early_stopping_patience"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{cfg['train']['epochs']}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            log_every=cfg["train"]["log_every"],
        )
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=cfg["train"]["threshold"])

        dt = time.time() - t0
        print(
            f"  train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"Se={val_metrics['Se']:.4f}  Sp={val_metrics['Sp']:.4f}  "
            f"F1={val_metrics['F1']:.4f}  ACC={val_metrics['ACC']:.4f}  "
            f"AUC={val_metrics.get('AUC', float('nan')):.4f}  ({dt:.1f}s)"
        )

        if val_metrics["F1"] > best_f1:
            best_f1 = val_metrics["F1"]
            patience_counter = 0
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "vasca_net_best.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": val_metrics}, ckpt_path)
            print(f"  New best F1={best_f1:.4f} -> saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no F1 improvement for {patience} epochs).")
                break

    print(f"Training complete. Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
