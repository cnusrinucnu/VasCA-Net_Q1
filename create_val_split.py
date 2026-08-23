import os
import shutil
import random
from pathlib import Path

DATASETS = {
    "STARE": 0.20,
    "DRIVE": 0.20,
    "CHASEDB1": 0.20,
}

SEED = 42


def get_files(directory):
    exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".ppm", ".gif",
    }

    return sorted(
        [
            p for p in Path(directory).iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
    )


def find_mask(mask_dir, image_path):

    stem = image_path.stem

    candidates = [
        p for p in get_files(mask_dir)
        if p.stem == stem
    ]

    if not candidates:
        candidates = [
            p for p in get_files(mask_dir)
            if p.stem.startswith(stem)
        ]

    if not candidates:
        raise FileNotFoundError(
            f"No mask found for {image_path} in {mask_dir}"
        )

    return sorted(candidates)[0]


def copy_pair(
    image_path,
    mask_path,
    output_root,
    split,
):

    image_out = (
        output_root
        / split
        / "images"
    )

    mask_out = (
        output_root
        / split
        / "masks"
    )

    image_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        image_path,
        image_out / image_path.name,
    )

    shutil.copy2(
        mask_path,
        mask_out / mask_path.name,
    )


def process_dataset(
    name,
    val_fraction,
):

    source = Path("data") / name
    destination = Path("data_final") / name

    train_images_dir = (
        source / "train" / "images"
    )

    train_masks_dir = (
        source / "train" / "masks"
    )

    test_images_dir = (
        source / "test" / "images"
    )

    test_masks_dir = (
        source / "test" / "masks"
    )

    images = get_files(
        train_images_dir
    )

    if len(images) < 2:

        raise RuntimeError(
            f"Too few training images for {name}"
        )

    random.seed(SEED)

    shuffled = images.copy()
    random.shuffle(shuffled)

    n_val = max(
        1,
        round(len(shuffled) * val_fraction),
    )

    val_images = shuffled[:n_val]
    train_images = shuffled[n_val:]

    print()
    print(name)
    print(
        f"Original train: {len(images)}"
    )
    print(
        f"New train:      {len(train_images)}"
    )
    print(
        f"Validation:     {len(val_images)}"
    )

    # ----------------------------------------
    # Training
    # ----------------------------------------

    for image_path in train_images:

        mask_path = find_mask(
            train_masks_dir,
            image_path,
        )

        copy_pair(
            image_path,
            mask_path,
            destination,
            "train",
        )

    # ----------------------------------------
    # Validation
    # ----------------------------------------

    for image_path in val_images:

        mask_path = find_mask(
            train_masks_dir,
            image_path,
        )

        copy_pair(
            image_path,
            mask_path,
            destination,
            "val",
        )

    # ----------------------------------------
    # Official test set
    # ----------------------------------------

    test_images = get_files(
        test_images_dir
    )

    print(
        f"Official test:  {len(test_images)}"
    )

    for image_path in test_images:

        mask_path = find_mask(
            test_masks_dir,
            image_path,
        )

        copy_pair(
            image_path,
            mask_path,
            destination,
            "test",
        )


def main():

    for name, fraction in DATASETS.items():

        process_dataset(
            name,
            fraction,
        )

    print()
    print(
        "Created clean datasets under:"
    )
    print(
        "    data_final/"
    )


if __name__ == "__main__":
    main()
