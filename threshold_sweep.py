import argparse
import csv
import os

import torch
import yaml
from torch.utils.data import DataLoader

from models import VasCANetDS
from datasets import RetinalVesselDataset
from utils.metrics import MetricAccumulator


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["train"]["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    test_set = RetinalVesselDataset(
        root=cfg["data"]["root"],
        split="test",
        patch_size=cfg["data"]["patch_size"],
        images_subdir=cfg["data"]["images_subdir"],
        masks_subdir=cfg["data"]["masks_subdir"],
    )

    loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    model = VasCANetDS(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        msca_ratio=cfg["model"]["msca_ratio"],
        econv_ratio=cfg["model"]["econv_ratio"],
    ).to(device)

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(ckpt["model"])
    model.eval()

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ]

    results = []

    # Store predictions once so threshold changes
    # do not require another forward pass.
    predictions = []

    with torch.no_grad():

        for img, mask in loader:

            img = img.to(device)
            mask = mask.to(device)

            logits = model(
                img,
                return_aux=False,
            )

            probs = torch.sigmoid(logits)

            predictions.append(
                (
                    probs.cpu(),
                    mask.cpu(),
                )
            )

    for threshold in thresholds:

        acc = MetricAccumulator()

        for probs, mask in predictions:

            acc.update(
                probs,
                mask,
                threshold=threshold,
            )

        metrics = acc.compute()

        row = {
            "threshold": threshold,
            "Se": metrics["Se"],
            "Sp": metrics["Sp"],
            "Precision": metrics["Precision"],
            "F1": metrics["F1"],
            "ACC": metrics["ACC"],
            "FPR": metrics["FPR"],
            "AUC": metrics.get("AUC", float("nan")),
            "PR_AUC": metrics.get("PR_AUC", float("nan")),
        }

        results.append(row)

        print(
            f"Threshold={threshold:.2f} "
            f"Se={row['Se']:.4f} "
            f"Sp={row['Sp']:.4f} "
            f"Precision={row['Precision']:.4f} "
            f"F1={row['F1']:.4f} "
            f"AUC={row['AUC']:.4f} "
            f"PR_AUC={row['PR_AUC']:.4f}"
        )

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True,
    )

    with open(
        args.output,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    best = max(
        results,
        key=lambda x: x["F1"],
    )

    print()
    print(
        "Best F1 threshold:",
        best["threshold"],
    )
    print(
        "Best F1:",
        best["F1"],
    )

    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()
