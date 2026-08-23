import cv2
import numpy as np
import torch


def mask_to_radius(mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary retinal vessel mask into a normalized
    vessel-radius map using the Euclidean distance transform.

    Vessel pixels contain their distance to the nearest background.
    Background pixels are zero.
    """

    mask_bin = (mask > 0).astype(np.uint8)

    if mask_bin.max() == 0:
        return np.zeros_like(mask_bin, dtype=np.float32)

    distance = cv2.distanceTransform(
        mask_bin,
        cv2.DIST_L2,
        5,
    )

    max_distance = distance.max()

    if max_distance > 0:
        distance = distance / max_distance

    distance[mask_bin == 0] = 0.0

    return distance.astype(np.float32)


def radius_tensor_from_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of masks [B,1,H,W] into radius maps [B,1,H,W].
    """

    masks = mask_tensor.detach().cpu().numpy()

    radius_maps = []

    for mask in masks:
        radius = mask_to_radius(mask[0])
        radius_maps.append(radius)

    radius_maps = np.stack(radius_maps, axis=0)

    radius_maps = torch.from_numpy(
        radius_maps
    ).unsqueeze(1)

    return radius_maps.to(
        device=mask_tensor.device,
        dtype=mask_tensor.dtype,
    )
