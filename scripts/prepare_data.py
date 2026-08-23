"""
Helper to lay out a downloaded DRIVE / STARE / CHASE_DB1 dataset into the
folder structure expected by `datasets.RetinalVesselDataset`:

    <out_root>/train/images/*.png
    <out_root>/train/masks/*.png
    <out_root>/test/images/*.png
    <out_root>/test/masks/*.png

This script does NOT download the datasets (they require registration /
manual download from their original hosts):
    DRIVE:     https://drive.grand-challenge.org/
    STARE:     https://cecas.clemson.edu/~ahoover/stare/
    CHASEDB1:  https://blogs.kingston.ac.uk/retinal/chasedb1/

Usage example (DRIVE, which already ships train/test folders):
    python scripts/prepare_data.py \
        --dataset drive \
        --raw_dir /path/to/DRIVE \
        --out_dir ./data/DRIVE

Usage example (STARE / CHASEDB1, flat folders needing a manual split):
    python scripts/prepare_data.py \
        --dataset stare \
        --raw_images_dir /path/to/stare-images \
        --raw_masks_dir /path/to/stare-labels \
        --out_dir ./data/STARE \
        --n_train 10
"""
import argparse
import glob
import os
import random
import shutil


def _copy_pairs(image_paths, mask_paths, out_images_dir, out_masks_dir):
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)
    for img_p, mask_p in zip(image_paths, mask_paths):
        shutil.copy(img_p, out_images_dir)
        shutil.copy(mask_p, out_masks_dir)


def prepare_drive(raw_dir: str, out_dir: str):
    # Standard DRIVE layout: training/images, training/1st_manual, test/images, test/1st_manual
    for split_src, split_dst in [("training", "train"), ("test", "test")]:
        img_dir = os.path.join(raw_dir, split_src, "images")
        mask_dir = os.path.join(raw_dir, split_src, "1st_manual")
        images = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.gif")))
        _copy_pairs(images, masks,
                    os.path.join(out_dir, split_dst, "images"),
                    os.path.join(out_dir, split_dst, "masks"))
        print(f"[DRIVE] {split_dst}: {len(images)} pairs")


def prepare_flat_split(raw_images_dir: str, raw_masks_dir: str, out_dir: str, n_train: int, seed: int = 42):
    """Generic splitter for STARE / CHASEDB1, which ship as a flat pool of images."""
    images = sorted(
        glob.glob(os.path.join(raw_images_dir, "*.png"))
        + glob.glob(os.path.join(raw_images_dir, "*.ppm"))
        + glob.glob(os.path.join(raw_images_dir, "*.jpg"))
    )
    masks = sorted(
        glob.glob(os.path.join(raw_masks_dir, "*.png"))
        + glob.glob(os.path.join(raw_masks_dir, "*.ppm"))
        + glob.glob(os.path.join(raw_masks_dir, "*.gif"))
    )
    assert len(images) == len(masks), f"Mismatched counts: {len(images)} images vs {len(masks)} masks"

    rng = random.Random(seed)
    idx = list(range(len(images)))
    rng.shuffle(idx)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    for name, subset in [("train", train_idx), ("test", test_idx)]:
        imgs = [images[i] for i in subset]
        msks = [masks[i] for i in subset]
        _copy_pairs(imgs, msks,
                    os.path.join(out_dir, name, "images"),
                    os.path.join(out_dir, name, "masks"))
        print(f"{name}: {len(imgs)} pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["drive", "stare", "chasedb1"], required=True)
    parser.add_argument("--raw_dir", type=str, default=None, help="Used for 'drive' (contains training/ and test/)")
    parser.add_argument("--raw_images_dir", type=str, default=None, help="Used for 'stare'/'chasedb1'")
    parser.add_argument("--raw_masks_dir", type=str, default=None, help="Used for 'stare'/'chasedb1'")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=None,
                         help="Number of images to use for training (STARE: 10, CHASEDB1: 20, per paper).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "drive":
        assert args.raw_dir, "--raw_dir is required for DRIVE"
        prepare_drive(args.raw_dir, args.out_dir)
    else:
        assert args.raw_images_dir and args.raw_masks_dir, "--raw_images_dir/--raw_masks_dir required"
        n_train = args.n_train or (10 if args.dataset == "stare" else 20)
        prepare_flat_split(args.raw_images_dir, args.raw_masks_dir, args.out_dir, n_train, args.seed)


if __name__ == "__main__":
    main()
