"""Image helpers shared by the app tabs: load raw BMPs, draw the crop box,
and convert to Qt pixmaps. Kept binding-agnostic of Pillow's ImageQt by going
through a numpy buffer (robust across PyQt/Pillow versions)."""

import numpy as np
from PIL import Image, ImageDraw
from PyQt5.QtGui import QImage, QPixmap


def np_to_pixmap(arr: np.ndarray) -> QPixmap:
    """uint8 (H, W, 3) RGB array -> QPixmap (owns its buffer via copy())."""
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    qim = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qim.copy())


def load_full_with_box(path, crop_box, max_side=900) -> QPixmap:
    """Load a raw image, draw the crop rectangle, downscale for display."""
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = crop_box
    # line width scaled to image size so it stays visible after downscaling
    lw = max(3, img.width // 400)
    draw.rectangle([x0, y0, x1, y1], outline=(0, 200, 255), width=lw)
    img.thumbnail((max_side, max_side), Image.BILINEAR)
    return np_to_pixmap(np.asarray(img, dtype=np.uint8))


def load_crop(path, crop_box) -> np.ndarray:
    """Return the cropped coil as a uint8 RGB array."""
    img = Image.open(path).convert("RGB").crop(crop_box)
    return np.asarray(img, dtype=np.uint8)


def load_crop_pixmap(path, crop_box, max_side=820) -> QPixmap:
    arr = load_crop(path, crop_box)
    img = Image.fromarray(arr)
    img.thumbnail((max_side, max_side), Image.BILINEAR)
    return np_to_pixmap(np.asarray(img, dtype=np.uint8))
