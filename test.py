"""
Evaluate a trained VasCA-Net checkpoint on a held-out test split, and
optionally dump predicted vessel maps to disk.

Usage:
    python test.py --config configs/default.yaml --checkpoint checkpoints/vasca_net_best.pth --save_dir preds/
"""
import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models import VasCANet
from datasets import RetinalVesselDataset
from utils.metrics import MetricAccumulator


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    threshold = args.threshold if args.threshold is not None else cfg["eval"]["threshold"]
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    test_set = RetinalVesselDataset(
        root=cfg["data"]["root"],
        split="test",
        patch_size=cfg["data"]["patch_size"],
        images_subdir=cfg["data"]["images_subdir"],
        masks_subdir=cfg["data"]["masks_subdir"],
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    model = VasCANet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        msca_ratio=cfg["model"]["msca_ratio"],
        econv_ratio=cfg["model"]["econv_ratio"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    acc = MetricAccumulator()
    for i, (img, mask) in enumerate(test_loader):
        img, mask = img.to(device), mask.to(device)
        probs = torch.sigmoid(model(img))
        acc.update(probs, mask, threshold=threshold)

        if args.save_dir:
            pred = (probs[0, 0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_dir, f"pred_{i:03d}.png"), pred)

    metrics = acc.compute()
    print("Test set results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
