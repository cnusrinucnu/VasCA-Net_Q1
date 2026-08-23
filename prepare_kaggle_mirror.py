"""
Prep script tailored to the specific Kaggle mirror:
  kaggle datasets download -d umairinayat/retinal-vessel-segmentation-datasets

Handles that mirror's actual on-disk layout/naming, which does NOT match
generic assumptions:
  - DRIVE:  training/images/*_training.tif + training/1st_manual/*_manual1.gif
            test/images/*_test.tif (+ test/1st_manual/*_manual1.gif if present)
  - STARE:  images/imXXXX.ppm(+.png)  <->  masks/imXXXX.ah.ppm(+.png)
            (image stem != mask stem; matched by the numeric imXXXX id)
  - CHASE:  images/test_XX_test.tif  <->  masks/test_XX_manual1.tif
            (matched by shared 'test_XX' prefix, not identical stem)

Usage:
    python scripts/prepare_kaggle_mirror.py \
        --raw_root ~/srinivas/raw_data \
        --out_root ~/srinivas/VasCA-Net/data
"""
import argparse
import glob
import os
import re
import shutil


def _copy_pair(img_path, mask_path, out_images_dir, out_masks_dir, common_id=None):
    """Copy an (image, mask) pair.

    `RetinalVesselDataset._match_files` pairs files by IDENTICAL FILENAME
    STEM. This mirror's image/mask filenames often don't share a stem
    (e.g. '26_training.tif' <-> '26_manual1.gif'), so when `common_id` is
    given we rename on copy (keeping each file's original extension) so
    the dataset loader can find the pair without any changes on its end.
    """
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    if common_id is None:
        shutil.copy(img_path, out_images_dir)
        shutil.copy(mask_path, out_masks_dir)
        return

    img_ext = os.path.splitext(img_path)[1]
    mask_ext = os.path.splitext(mask_path)[1]
    shutil.copy(img_path, os.path.join(out_images_dir, f"{common_id}{img_ext}"))
    shutil.copy(mask_path, os.path.join(out_masks_dir, f"{common_id}{mask_ext}"))


def prep_drive(raw_root: str, out_root: str):
    raw_drive = os.path.join(raw_root, "DRIVE")
    out_drive = os.path.join(out_root, "DRIVE")

    # NOTE: DRIVE ships TWO different kinds of "mask":
    #   - vessel ground truth (what we want): 'training/1st_manual/*_manual1.gif'
    #   - field-of-view (FOV) mask, i.e. the circular sensor boundary,
    #     NOT a segmentation label: 'training/mask/*_training_mask.gif' and
    #     'test/mask/*_test_mask.gif'.
    # We must only ever treat the vessel-annotation folder as ground truth.
    # A folder literally named 'mask'/'masks' is deliberately EXCLUDED from
    # this candidate list, since on this mirror it is the FOV mask, not a
    # vessel label -- using it would silently corrupt training/eval.
    GT_FOLDER_CANDIDATES = ["1st_manual", "manual1", "manual", "vessel_gt", "labels"]

    # ---- train split ----
    train_imgs = sorted(glob.glob(os.path.join(raw_drive, "training", "images", "*_training.tif")))
    train_masks_dir = None
    for cand in GT_FOLDER_CANDIDATES:
        p = os.path.join(raw_drive, "training", cand)
        if os.path.isdir(p):
            train_masks_dir = p
            break
    if train_masks_dir is None:
        raise FileNotFoundError(
            f"Could not find a vessel ground-truth folder under {raw_drive}/training "
            f"(looked for: {GT_FOLDER_CANDIDATES}). Do NOT fall back to a folder named "
            f"'mask' -- on DRIVE that is the field-of-view mask, not vessel labels."
        )

    n_train = 0
    for img_p in train_imgs:
        idx = re.search(r"(\d+)_training", os.path.basename(img_p)).group(1)
        candidates = glob.glob(os.path.join(train_masks_dir, f"{idx}_manual1.*"))
        if not candidates:
            print(f"  [WARN] no training mask for index {idx}, skipping")
            continue
        _copy_pair(img_p, candidates[0],
                   os.path.join(out_drive, "train", "images"),
                   os.path.join(out_drive, "train", "masks"),
                   common_id=idx)
        n_train += 1
    print(f"DRIVE train: {n_train} pairs (ground truth from '{os.path.basename(train_masks_dir)}/')")

    # ---- test split ----
    test_imgs = sorted(glob.glob(os.path.join(raw_drive, "test", "images", "*_test.tif")))
    test_masks_dir = None
    for cand in GT_FOLDER_CANDIDATES:
        p = os.path.join(raw_drive, "test", cand)
        if os.path.isdir(p):
            test_masks_dir = p
            break

    n_test = 0
    if test_masks_dir is None:
        print("  [WARN] No vessel ground-truth folder found under DRIVE/test (only a "
              "FOV 'mask/' folder exists here, which is NOT a segmentation label and "
              "is correctly being ignored). This mirror does not ship labeled test "
              "images. Splitting off part of the training set for evaluation instead.")
    else:
        for img_p in test_imgs:
            idx = re.search(r"(\d+)_test", os.path.basename(img_p)).group(1)
            candidates = glob.glob(os.path.join(test_masks_dir, f"{idx}_manual1.*"))
            if not candidates:
                print(f"  [WARN] no test mask for index {idx}, skipping")
                continue
            _copy_pair(img_p, candidates[0],
                       os.path.join(out_drive, "test", "images"),
                       os.path.join(out_drive, "test", "masks"),
                       common_id=idx)
            n_test += 1
        print(f"DRIVE test: {n_test} pairs (ground truth from '{os.path.basename(test_masks_dir)}/')")

    if n_test == 0:
        print("  -> Re-carving DRIVE/train into train/val (17/3) since no "
              "labeled test images are available in this mirror.")
        _reserve_val_split(out_drive, n_val=3)


def _reserve_val_split(out_dataset_dir: str, n_val: int, seed: int = 42):
    import random
    train_img_dir = os.path.join(out_dataset_dir, "train", "images")
    train_mask_dir = os.path.join(out_dataset_dir, "train", "masks")
    imgs = sorted(glob.glob(os.path.join(train_img_dir, "*")))
    rng = random.Random(seed)
    rng.shuffle(imgs)
    val_imgs = imgs[:n_val]

    test_img_dir = os.path.join(out_dataset_dir, "test", "images")
    test_mask_dir = os.path.join(out_dataset_dir, "test", "masks")
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)

    for img_p in val_imgs:
        # After renaming, train images are already just "{idx}.tif" -- the
        # matching mask is "{idx}.<ext>" in train_mask_dir.
        stem = os.path.splitext(os.path.basename(img_p))[0]
        mask_candidates = glob.glob(os.path.join(train_mask_dir, f"{stem}.*"))
        if not mask_candidates:
            continue
        shutil.move(img_p, test_img_dir)
        shutil.move(mask_candidates[0], test_mask_dir)
    print(f"  -> moved {len(val_imgs)} pairs from train to test/val split")


def prep_stare(raw_root: str, out_root: str, n_train: int = 10, seed: int = 42):
    import random
    raw_stare = os.path.join(raw_root, "STARE")
    out_stare = os.path.join(out_root, "STARE")

    # Use the .ppm.png versions (already-decoded, easy to read with cv2)
    images = sorted(glob.glob(os.path.join(raw_stare, "images", "im*.ppm.png")))
    pairs = []
    for img_p in images:
        m = re.search(r"(im\d+)\.ppm\.png$", os.path.basename(img_p))
        if not m:
            continue
        stem = m.group(1)  # e.g. "im0001"
        mask_p = os.path.join(raw_stare, "masks", f"{stem}.ah.ppm.png")
        if os.path.exists(mask_p):
            pairs.append((img_p, mask_p))
        else:
            print(f"  [WARN] no mask found for {stem}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    train_pairs, test_pairs = pairs[:n_train], pairs[n_train:]

    for split, subset in [("train", train_pairs), ("test", test_pairs)]:
        for img_p, mask_p in subset:
            stem = re.search(r"(im\d+)\.ppm\.png$", os.path.basename(img_p)).group(1)
            _copy_pair(img_p, mask_p,
                       os.path.join(out_stare, split, "images"),
                       os.path.join(out_stare, split, "masks"),
                       common_id=stem)
        print(f"STARE {split}: {len(subset)} pairs")


def prep_chase(raw_root: str, out_root: str, n_train: int = 20, seed: int = 42):
    import random
    raw_chase = os.path.join(raw_root, "CHASE")
    out_chase = os.path.join(out_root, "CHASEDB1")

    images = sorted(glob.glob(os.path.join(raw_chase, "images", "*.tif")))
    pairs = []
    for img_p in images:
        m = re.match(r"(test_\d+)_test\.tif$", os.path.basename(img_p))
        if not m:
            print(f"  [WARN] unexpected filename pattern: {img_p}")
            continue
        prefix = m.group(1)  # e.g. "test_01"
        mask_p = os.path.join(raw_chase, "masks", f"{prefix}_manual1.tif")
        if os.path.exists(mask_p):
            pairs.append((img_p, mask_p))
        else:
            print(f"  [WARN] no mask found for {prefix}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    train_pairs, test_pairs = pairs[:n_train], pairs[n_train:]

    for split, subset in [("train", train_pairs), ("test", test_pairs)]:
        for img_p, mask_p in subset:
            prefix = re.match(r"(test_\d+)_test\.tif$", os.path.basename(img_p)).group(1)
            _copy_pair(img_p, mask_p,
                       os.path.join(out_chase, split, "images"),
                       os.path.join(out_chase, split, "masks"),
                       common_id=prefix)
        print(f"CHASEDB1 {split}: {len(subset)} pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, required=True,
                         help="Path to the unzipped Kaggle mirror root (contains DRIVE/, STARE/, CHASE/, ...)")
    parser.add_argument("--out_root", type=str, required=True,
                         help="Path to write the repo-standard data/ layout into")
    parser.add_argument("--datasets", nargs="+", default=["drive", "stare", "chase"],
                         choices=["drive", "stare", "chase"])
    args = parser.parse_args()

    raw_root = os.path.expanduser(args.raw_root)
    out_root = os.path.expanduser(args.out_root)

    if "drive" in args.datasets:
        print("=== DRIVE ===")
        prep_drive(raw_root, out_root)
    if "stare" in args.datasets:
        print("=== STARE ===")
        prep_stare(raw_root, out_root)
    if "chase" in args.datasets:
        print("=== CHASE_DB1 ===")
        prep_chase(raw_root, out_root)


if __name__ == "__main__":
    main()
