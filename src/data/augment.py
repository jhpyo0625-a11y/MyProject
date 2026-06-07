"""
T2.3 -- Augmentation pipeline for minority-class crops.

Applies mild, physically-plausible transforms to a uint8 (H, W, 3) numpy
array and returns a uint8 (H, W, 3) array.

Transforms (in order):
  1. Brightness / contrast jitter  -- mimics lighting variation between shots
  2. Horizontal + vertical flip    -- defects are position-agnostic
  3. Small rotation (<=5 deg)      -- sensor tilt tolerance
  4. Small translation             -- horizontal: +-50 px (measured jitter);
                                      vertical:   +-10 px (spec: only 8 px drift)

NOT applied: heavy zoom, elastic distortion, or aggressive crop.
See PROJECT_SPEC.md §6 for rationale.
"""

import random
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance


def augment(
    arr: np.ndarray,
    brightness: float = 0.3,
    contrast: float = 0.3,
    max_rotation_deg: float = 5.0,
    max_translate_x: int = 50,
    max_translate_y: int = 10,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """
    Apply random augmentation to a uint8 (H, W, 3) numpy array.

    Args:
        arr: input crop, shape (H, W, 3), dtype uint8
        brightness: max brightness jitter factor (symmetric around 1.0)
        contrast: max contrast jitter factor
        max_rotation_deg: maximum rotation in either direction
        max_translate_x: max horizontal translation in pixels
        max_translate_y: max vertical translation in pixels
        rng: optional seeded random.Random for reproducibility

    Returns:
        Augmented array, same shape and dtype as input.
    """
    if rng is None:
        rng = random

    img = Image.fromarray(arr)

    # --- brightness jitter ---------------------------------------------------
    factor = rng.uniform(1.0 - brightness, 1.0 + brightness)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # --- contrast jitter -----------------------------------------------------
    factor = rng.uniform(1.0 - contrast, 1.0 + contrast)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # --- horizontal flip (p=0.5) ---------------------------------------------
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # --- vertical flip (p=0.5) -----------------------------------------------
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # --- rotation + translation (affine) -------------------------------------
    angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
    tx    = rng.randint(-max_translate_x, max_translate_x)
    ty    = rng.randint(-max_translate_y, max_translate_y)
    img   = img.rotate(
        angle,
        translate=(tx, ty),
        resample=Image.BILINEAR,
        fillcolor=(0, 0, 0),
    )

    return np.asarray(img, dtype=np.uint8)


def augment_from_config(arr: np.ndarray, cfg: dict, rng=None) -> np.ndarray:
    """Convenience wrapper that reads parameters from config.yaml augmentation block."""
    aug = cfg["augmentation"]
    return augment(
        arr,
        brightness=aug["brightness_factor"],
        contrast=aug["contrast_factor"],
        max_rotation_deg=aug["max_rotation_deg"],
        max_translate_x=aug["max_translate_px"],
        max_translate_y=10,
        rng=rng,
    )
