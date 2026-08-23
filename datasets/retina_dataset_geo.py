import os
import random
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.geometry_target import mask_to_radius


class RetinaPatchDatasetGeo(Dataset):
    """
    Geometry-aware retinal vessel patch dataset.

    Returns:
        image  : [1, H, W]
        mask   : [1, H, W]
        radius : [1, H, W]

    The radius map is generated from the vessel mask using
    a Euclidean distance transform.

    Image and mask filenames are matched by filename stem,
    allowing different image/mask extensions such as:

        21.tif -> 21.gif
        22.tif -> 22.gif
    """

    def __init__(
        self,
        root,
        split="train",
        patch_size=64,
        patches_per_image=200,
        images_subdir="images",
        masks_subdir="masks",
        seed=42,
    ):
        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

        self.images_dir = os.path.join(
            root,
            split,
            images_subdir,
        )

        self.masks_dir = os.path.join(
            root,
            split,
            masks_subdir,
        )

        # --------------------------------------------------
        # Find images
        # --------------------------------------------------

        image_patterns = [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.tif",
            "*.tiff",
            "*.bmp",
        ]

        image_files = []

        for pattern in image_patterns:
            image_files.extend(
                glob(
                    os.path.join(
                        self.images_dir,
                        pattern,
                    )
                )
            )

        self.image_files = sorted(
            image_files
        )

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in: "
                f"{self.images_dir}"
            )

        # --------------------------------------------------
        # Match masks by filename stem
        # --------------------------------------------------

        self.mask_files = []

        mask_extensions = [
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".gif",
            ".bmp",
        ]

        for image_path in self.image_files:

            image_name = os.path.basename(
                image_path
            )

            stem = os.path.splitext(
                image_name
            )[0]

            mask_path = None

            for ext in mask_extensions:

                candidate = os.path.join(
                    self.masks_dir,
                    stem + ext,
                )

                if os.path.exists(candidate):

                    mask_path = candidate
                    break

            if mask_path is None:

                raise FileNotFoundError(
                    "Mask not found for image:\n"
                    f"  Image: {image_path}\n"
                    f"  Expected stem: {stem}\n"
                    f"  Mask directory: {self.masks_dir}"
                )

            self.mask_files.append(
                mask_path
            )

        # --------------------------------------------------
        # Generate patch index
        # --------------------------------------------------

        self.samples = []

        rng = random.Random(seed)

        for image_idx in range(
            len(self.image_files)
        ):

            for _ in range(
                self.patches_per_image
            ):

                self.samples.append(
                    (
                        image_idx,
                        rng.random(),
                    )
                )

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------
    # Image loading
    # ------------------------------------------------------

    def _load_image(self, path):

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:

            raise RuntimeError(
                f"Unable to read image: {path}"
            )

        image = (
            image.astype(np.float32)
            / 255.0
        )

        return image

    # ------------------------------------------------------
    # Mask loading
    # ------------------------------------------------------

    def _load_mask(self, path):

        mask = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:

            raise RuntimeError(
                f"Unable to read mask: {path}"
            )

        mask = (
            mask > 127
        ).astype(np.float32)

        return mask

    # ------------------------------------------------------
    # Random crop
    # ------------------------------------------------------

    def _random_crop(
        self,
        image,
        mask,
        radius,
    ):

        h, w = image.shape

        ps = self.patch_size

        # --------------------------------------------------
        # If image is smaller than patch size
        # --------------------------------------------------

        if h < ps or w < ps:

            image = cv2.resize(
                image,
                (ps, ps),
                interpolation=cv2.INTER_LINEAR,
            )

            mask = cv2.resize(
                mask,
                (ps, ps),
                interpolation=cv2.INTER_NEAREST,
            )

            radius = cv2.resize(
                radius,
                (ps, ps),
                interpolation=cv2.INTER_LINEAR,
            )

            return (
                image,
                mask,
                radius,
            )

        # --------------------------------------------------
        # Random crop
        # --------------------------------------------------

        top = random.randint(
            0,
            h - ps,
        )

        left = random.randint(
            0,
            w - ps,
        )

        image = image[
            top:top + ps,
            left:left + ps,
        ]

        mask = mask[
            top:top + ps,
            left:left + ps,
        ]

        radius = radius[
            top:top + ps,
            left:left + ps,
        ]

        return (
            image,
            mask,
            radius,
        )

    # ------------------------------------------------------
    # Dataset item
    # ------------------------------------------------------

    def __getitem__(self, index):

        image_idx, _ = self.samples[
            index
        ]

        image_path = self.image_files[
            image_idx
        ]

        mask_path = self.mask_files[
            image_idx
        ]

        # --------------------------------------------------
        # Load
        # --------------------------------------------------

        image = self._load_image(
            image_path
        )

        mask = self._load_mask(
            mask_path
        )

        # --------------------------------------------------
        # Generate explicit geometry target
        # --------------------------------------------------

        radius = mask_to_radius(
            mask
        )

        # --------------------------------------------------
        # Apply identical crop to image,
        # mask and radius map.
        # --------------------------------------------------

        if self.split == "train":

            image, mask, radius = (
                self._random_crop(
                    image,
                    mask,
                    radius,
                )
            )

        else:

            ps = self.patch_size

            h, w = image.shape

            if h < ps or w < ps:

                image = cv2.resize(
                    image,
                    (ps, ps),
                    interpolation=cv2.INTER_LINEAR,
                )

                mask = cv2.resize(
                    mask,
                    (ps, ps),
                    interpolation=cv2.INTER_NEAREST,
                )

                radius = cv2.resize(
                    radius,
                    (ps, ps),
                    interpolation=cv2.INTER_LINEAR,
                )

        # --------------------------------------------------
        # Convert to tensors
        # --------------------------------------------------

        image = torch.from_numpy(
            image.copy()
        ).float().unsqueeze(0)

        mask = torch.from_numpy(
            mask.copy()
        ).float().unsqueeze(0)

        radius = torch.from_numpy(
            radius.copy()
        ).float().unsqueeze(0)

        return (
            image,
            mask,
            radius,
        )
