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


def _anomaly_colormap(norm: np.ndarray) -> np.ndarray:
    """norm (H,W) in [0,1] -> (H,W,3) uint8 cool(blue)->hot(red) ramp.
    Dependency-free (no matplotlib) so it never slows app startup."""
    stops = np.array([[0.00,   0,   0, 140],
                      [0.30,   0, 140, 255],
                      [0.55,   0, 200,  60],
                      [0.78, 255, 210,   0],
                      [1.00, 210,   0,   0]], dtype=np.float32)
    chans = [np.interp(norm, stops[:, 0], stops[:, i]) for i in (1, 2, 3)]
    return np.stack(chans, axis=-1).astype(np.uint8)


def load_full_with_heatmap(path, crop_box, amap, peak=None, t_flag=None,
                           max_side=900, alpha=0.6) -> QPixmap:
    """Full frame with the PaDiM anomaly heatmap composited into the crop region,
    the crop box drawn, and the peak patch boxed -- i.e. WHERE the coil was
    flagged. `amap` is the (h, w) per-patch map from predictor.localize()."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    x0, y0, x1, y1 = crop_box
    cw, ch = x1 - x0, y1 - y0

    # Normalize by t_flag for ABSOLUTE meaning (a clean coil stays cool, no false
    # hot spot); fall back to per-image max if no threshold is available.
    denom = float(t_flag) if (t_flag and t_flag > 0) else float(amap.max() or 1.0)
    norm01 = np.clip(amap.astype(np.float32) / denom, 0.0, 1.0)
    up = Image.fromarray((norm01 * 255).astype(np.uint8)).resize(
        (cw, ch), Image.BILINEAR)
    norm_up = np.asarray(up, dtype=np.float32) / 255.0          # (ch, cw)
    heat = _anomaly_colormap(norm_up).astype(np.float32)        # (ch, cw, 3)

    a = (alpha * norm_up)[..., None]                            # blend weighted by heat
    region = arr[y0:y1, x0:x1]
    arr[y0:y1, x0:x1] = region * (1.0 - a) + heat * a
    out = Image.fromarray(arr.astype(np.uint8))

    draw = ImageDraw.Draw(out)
    lw = max(3, out.width // 400)
    draw.rectangle([x0, y0, x1, y1], outline=(0, 200, 255), width=lw)
    if peak is not None:
        h, w = amap.shape
        prow, pcol = peak
        px = x0 + (pcol + 0.5) / w * cw
        py = y0 + (prow + 0.5) / h * ch
        bw, bh = cw / w, ch / h
        draw.rectangle([px - bw, py - bh, px + bw, py + bh],
                       outline=(255, 255, 255), width=lw)
    out.thumbnail((max_side, max_side), Image.BILINEAR)
    return np_to_pixmap(np.asarray(out, dtype=np.uint8))
