import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.retina_dataset_geo import RetinaPatchDatasetGeo
from models.vasca_net_geo import VasCANetGeo
from utils.geometry_loss import GeoVasCALoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("Geo-VasCA-Net SMOKE TEST")
    print("=" * 70)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # Smoke-test configuration
    # --------------------------------------------------

    root = "./data_final/DRIVE"

    patch_size = 64

    # Only 2 patches/image for smoke testing.
    patches_per_image = 2

    batch_size = 2

    epochs = 3

    lr = 5e-4

    lambda_geo = 0.10

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dataset = RetinaPatchDatasetGeo(
        root=root,
        split="train",
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("Training samples:", len(train_dataset))
    print("Batches/epoch:", len(train_loader))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = GeoVasCALoss(
        bce_weight=0.5,
        lambda_geo=lambda_geo,
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    model.train()

    for epoch in range(1, epochs + 1):

        running_total = 0.0
        running_seg = 0.0
        running_geo = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

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

            total_loss = losses["total"]

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite loss at "
                    f"epoch={epoch}, step={step}"
                )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            running_total += (
                total_loss.item()
            )

            running_seg += (
                losses["segmentation"].item()
            )

            running_geo += (
                losses["geometry"].item()
            )

            if step == 1 or step % 5 == 0:

                print(
                    f"Epoch {epoch}/{epochs} "
                    f"Step {step}/{len(train_loader)} "
                    f"Total={total_loss.item():.4f} "
                    f"Seg={losses['segmentation'].item():.4f} "
                    f"Geo={losses['geometry'].item():.4f}"
                )

        n = len(train_loader)

        print(
            f"\nEpoch {epoch} summary: "
            f"Total={running_total / n:.4f} "
            f"Seg={running_seg / n:.4f} "
            f"Geo={running_geo / n:.4f}\n"
        )

    # --------------------------------------------------
    # Save smoke-test checkpoint
    # --------------------------------------------------

    output_dir = "./checkpoints_q1/smoke"

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        output_dir,
        "geo_vasca_smoke.pth",
    )

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epochs,
            "lambda_geo": lambda_geo,
        },
        checkpoint_path,
    )

    print("=" * 70)
    print("SMOKE TEST COMPLETED SUCCESSFULLY")
    print("Checkpoint:", checkpoint_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
