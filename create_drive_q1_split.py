import os
import shutil
import random
from glob import glob


ROOT = "./data_final/DRIVE"

SEED = 42

VAL_COUNT = 3

IMAGE_DIR = os.path.join(
    ROOT,
    "train",
    "images"
)

MASK_DIR = os.path.join(
    ROOT,
    "train",
    "masks"
)


def find_file(directory, stem):

    extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".gif",
        ".bmp",
    ]

    for ext in extensions:

        path = os.path.join(
            directory,
            stem + ext
        )

        if os.path.exists(path):
            return path

    return None


def main():

    images = sorted(
        glob(
            os.path.join(
                IMAGE_DIR,
                "*"
            )
        )
    )

    images = [
        x for x in images
        if os.path.isfile(x)
    ]

    print(
        "Original training images:",
        len(images)
    )

    if len(images) != 14:

        raise RuntimeError(
            f"Expected 14 DRIVE training images, "
            f"found {len(images)}"
        )

    pairs = []

    for image_path in images:

        filename = os.path.basename(
            image_path
        )

        stem = os.path.splitext(
            filename
        )[0]

        mask_path = find_file(
            MASK_DIR,
            stem
        )

        if mask_path is None:

            raise FileNotFoundError(
                f"No mask found for {image_path}"
            )

        pairs.append(
            (
                image_path,
                mask_path,
                stem
            )
        )

    rng = random.Random(
        SEED
    )

    shuffled = pairs.copy()

    rng.shuffle(
        shuffled
    )

    val_pairs = shuffled[
        :VAL_COUNT
    ]

    train_pairs = shuffled[
        VAL_COUNT:
    ]

    train_image_dir = os.path.join(
        ROOT,
        "train_q1",
        "images"
    )

    train_mask_dir = os.path.join(
        ROOT,
        "train_q1",
        "masks"
    )

    val_image_dir = os.path.join(
        ROOT,
        "val_q1",
        "images"
    )

    val_mask_dir = os.path.join(
        ROOT,
        "val_q1",
        "masks"
    )

    for directory in [
        train_image_dir,
        train_mask_dir,
        val_image_dir,
        val_mask_dir,
    ]:

        os.makedirs(
            directory,
            exist_ok=True
        )

    # --------------------------------------------------
    # Copy training images
    # --------------------------------------------------

    for image_path, mask_path, stem in train_pairs:

        shutil.copy2(
            image_path,
            os.path.join(
                train_image_dir,
                os.path.basename(
                    image_path
                )
            )
        )

        shutil.copy2(
            mask_path,
            os.path.join(
                train_mask_dir,
                os.path.basename(
                    mask_path
                )
            )
        )

    # --------------------------------------------------
    # Copy validation images
    # --------------------------------------------------

    for image_path, mask_path, stem in val_pairs:

        shutil.copy2(
            image_path,
            os.path.join(
                val_image_dir,
                os.path.basename(
                    image_path
                )
            )
        )

        shutil.copy2(
            mask_path,
            os.path.join(
                val_mask_dir,
                os.path.basename(
                    mask_path
                )
            )
        )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("DRIVE Q1 SPLIT")
    print("=" * 70)

    print(
        "Training images:",
        len(train_pairs)
    )

    print(
        "Validation images:",
        len(val_pairs)
    )

    print(
        "Official test images:",
        3
    )

    print()
    print("TRAIN:")

    for _, _, stem in sorted(
        train_pairs,
        key=lambda x: x[2]
    ):

        print(
            " ",
            stem
        )

    print()
    print("VALIDATION:")

    for _, _, stem in sorted(
        val_pairs,
        key=lambda x: x[2]
    ):

        print(
            " ",
            stem
        )

    print()
    print(
        "Official test directory remains:",
        os.path.join(
            ROOT,
            "test"
        )
    )


if __name__ == "__main__":
    main()
