"""
Runs all 8 module configurations from Table 1 (Base, Base+A, Base+B, Base+C,
Base+A+B, Base+B+C, Base+A+C, Base+A+B+C) on a single dataset and reports
Sp/ACC/AUC/F1, so you can reproduce (and extend) the paper's ablation study.

Usage:
    python scripts/run_ablation.py --config configs/default.yaml --epochs 20
"""
import argparse
import copy
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VasCANetAblation, CONFIGS
from datasets import RetinalVesselDataset
from utils.losses import BCELoss
from utils.metrics import MetricAccumulator


def train_and_eval(cfg, variant_name, variant_kwargs, epochs, device):
    train_set = RetinalVesselDataset(
        root=cfg["data"]["root"], split="train",
        patch_size=cfg["data"]["patch_size"],
        patches_per_image=cfg["data"]["patches_per_image"],
    )
    val_set = RetinalVesselDataset(
        root=cfg["data"]["root"], split="test",
        patch_size=cfg["data"]["patch_size"],
    )
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = VasCANetAblation(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        msca_ratio=cfg["model"]["msca_ratio"],
        econv_ratio=cfg["model"]["econv_ratio"],
        **variant_kwargs,
    ).to(device)

    criterion = BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    for epoch in range(epochs):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()

    model.eval()
    acc = MetricAccumulator()
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            probs = torch.sigmoid(model(imgs))
            acc.update(probs, masks, threshold=cfg["eval"]["threshold"])
    metrics = acc.compute()
    print(f"[{variant_name}] Sp={metrics['Sp']:.4f} ACC={metrics['ACC']:.4f} "
          f"AUC={metrics.get('AUC', float('nan')):.4f} F1={metrics['F1']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    results = {}
    for name, kwargs in CONFIGS.items():
        print(f"\n=== Training variant: {name} ===")
        results[name] = train_and_eval(cfg, name, kwargs, args.epochs, device)

    print("\n\nMethod        Sp      ACC     AUC     F1")
    for name, m in results.items():
        print(f"{name:12s} {m['Sp']:.4f}  {m['ACC']:.4f}  {m.get('AUC', float('nan')):.4f}  {m['F1']:.4f}")


if __name__ == "__main__":
    main()
