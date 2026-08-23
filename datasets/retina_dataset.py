"""
Generic retinal vessel segmentation dataset loader.

Expects a directory layout of the form:

    root/
      images/   *.png|*.tif|*.jpg   (fundus images, RGB)
      masks/    *.png                (ground-truth vessel maps, binary)
      fov/      *.png                (optional field-of-view masks)

Implements the preprocessing pipeline described in Section 4.1 / Fig. 5:
  1. RGB -> grayscale
  2. Normalisation
  3. CLAHE (contrast-limited adaptive histogram equalisation)
  4. Gamma correction
  5. (training only) random rotation / horizontal & vertical / diagonal flips
  6. Patch extraction (random crop to `patch_size`)
"""
import os
import glob
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _match_files(images_dir: str, masks_dir: str):
    exts = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.gif", "*.ppm")
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(images_dir, e)))
    image_paths = sorted(image_paths)

    mask_paths = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        candidates = [p for e in exts for p in glob.glob(os.path.join(masks_dir, stem + e[1:]))]
        # fallback: any mask file that starts with the same stem
        if not candidates:
            candidates = [p for e in exts for p in glob.glob(os.path.join(masks_dir, stem + "*" + e[1:]))]
        if not candidates:
            raise FileNotFoundError(f"No mask found for image '{img_path}' in '{masks_dir}'.")
        mask_paths.append(sorted(candidates)[0])
    return image_paths, mask_paths


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """RGB -> grayscale -> normalise -> CLAHE -> gamma correction (Fig. 5)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Normalisation to [0, 255]
    norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(norm.astype(np.uint8))

    # Gamma correction
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv_gamma * 255
    table = table.astype(np.uint8)
    gamma_corrected = cv2.LUT(clahe_img, table)

    return gamma_corrected


class RetinalVesselDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: int = 64,
        patches_per_image: int = 200,
        augment: Optional[bool] = None,
        images_subdir: str = "images",
        masks_subdir: str = "masks",
    ):
        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment if augment is not None else (split == "train")

        images_dir = os.path.join(root, split, images_subdir)
        masks_dir = os.path.join(root, split, masks_subdir)
        self.image_paths, self.mask_paths = _match_files(images_dir, masks_dir)

        self.images = []
        self.masks = []
        for img_p, mask_p in zip(self.image_paths, self.mask_paths):
            img_bgr = cv2.imread(img_p, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or mask is None:
                raise IOError(f"Failed to read pair: {img_p}, {mask_p}")
            proc = preprocess_image(img_bgr)
            mask_bin = (mask > 127).astype(np.uint8) * 255
            self.images.append(proc)
            self.masks.append(mask_bin)

        self._len = (
            len(self.images) * self.patches_per_image
            if split == "train"
            else len(self.images)
        )

    def __len__(self) -> int:
        return self._len

    def _random_patch(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape
        ps = self.patch_size
        if h < ps or w < ps:
            pad_h, pad_w = max(0, ps - h), max(0, ps - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            h, w = img.shape
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)
        return img[y:y + ps, x:x + ps], mask[y:y + ps, x:x + ps]

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
        if random.random() < 0.5:
            img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
        if random.random() < 0.5:
            img, mask = img.T.copy(), mask.T.copy()  # diagonal flip
        k = random.choice([0, 1, 2, 3])
        if k:
            img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()
        return img, mask

    def __getitem__(self, idx: int):
        if self.split == "train":
            img_idx = idx % len(self.images)
        else:
            img_idx = idx

        img, mask = self.images[img_idx], self.masks[img_idx]

        if self.split == "train":
            img, mask = self._random_patch(img, mask)
            if self.augment:
                img, mask = self._augment(img, mask)
        else:
            ps = self.patch_size
            h, w = img.shape
            if h < ps or w < ps:
                pad_h, pad_w = max(0, ps - h), max(0, ps - w)
                img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        return img_t, mask_t
