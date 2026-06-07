"""
T1.1 -- Crop box validation.

Strategy: check that each edge of the configured crop window falls in the
dark inspection-table background, NOT in the PCB/coil content.
  - Extract a 30-px strip from each crop edge.
  - Compute mean luminance of that strip.
  - If luminance < EDGE_THRESHOLD -> edge is safely in background (PASS).
  - If luminance >= EDGE_THRESHOLD -> crop may cut into PCB content (WARN).

Also generates a 3x3 crop montage for visual inspection.

Outputs:
  reports/crop_validation.png  -- 3x3 montage of applied crops
  reports/crop_validation.txt  -- per-image result + summary
"""

import sys
import random
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.config_loader import load_config

cfg = load_config()

EDGE_STRIP_PX  = 30    # pixels inside crop edge to sample
EDGE_THRESHOLD = 72    # mean luminance above this -> crop may cut content
N_PER_CLASS    = 15
WARN_FAIL_FRAC = 0.5   # exit non-zero if > this fraction of images warn (real gate)
REPORTS        = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

crop_cfg = cfg["preprocessing"]["crop"]
CROP     = (crop_cfg["x_min"], crop_cfg["y_min"],
            crop_cfg["x_max"], crop_cfg["y_max"])
MANIFEST = ROOT / cfg["paths"]["manifest"]

CLASSES = ["Pass", "Dent", "Loose"]


# ---------------------------------------------------------------------------

def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def apply_crop(img: Image.Image) -> Image.Image:
    x0, y0, x1, y1 = CROP
    return img.crop((x0, y0, x1, y1))


def luminance(arr: np.ndarray) -> float:
    """Mean luminance (ITU-R BT.601) of an RGB array."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float(0.299 * r.mean() + 0.587 * g.mean() + 0.114 * b.mean())


def check_crop_edges(img: Image.Image) -> dict:
    """
    Apply the crop, then sample 30-px strips from each interior edge.
    Returns dict with edge name -> mean luminance, and a 'warnings' list.
    """
    x0, y0, x1, y1 = CROP
    arr = np.asarray(img)

    s = EDGE_STRIP_PX
    strips = {
        "top":    arr[y0:y0+s, x0:x1],
        "bottom": arr[y1-s:y1, x0:x1],
        "left":   arr[y0:y1, x0:x0+s],
        "right":  arr[y0:y1, x1-s:x1],
    }

    results  = {k: luminance(v) for k, v in strips.items()}
    warnings = [f"{k}={lum:.1f}" for k, lum in results.items()
                if lum >= EDGE_THRESHOLD]
    results["warnings"] = warnings
    return results


# ---------------------------------------------------------------------------

def main():
    random.seed(42)
    rows   = load_manifest()
    by_cls = {c: [r for r in rows if r["label"] == c] for c in CLASSES}

    log_lines    = []
    warn_images  = []
    montage_imgs = {c: [] for c in CLASSES}

    header = (f"Crop box: x=[{CROP[0]},{CROP[2]}]  "
              f"y=[{CROP[1]},{CROP[3]}]  "
              f"edge_threshold={EDGE_THRESHOLD}  strip={EDGE_STRIP_PX}px")
    print(header)
    log_lines.append(header)

    for cls in CLASSES:
        sample = random.sample(by_cls[cls], min(N_PER_CLASS, len(by_cls[cls])))
        for row in sample:
            path = Path(row["filepath"])
            img  = Image.open(path).convert("RGB")
            res  = check_crop_edges(img)
            warns = res["warnings"]
            status = "WARN" if warns else "PASS"
            edge_str = "  ".join(f"{k}={res[k]:.1f}" for k in ["top","bottom","left","right"])
            msg = f"{status}  {cls}/{path.name[:45]}  [{edge_str}]"
            if warns:
                msg += f"  -> high edge(s): {', '.join(warns)}"
                warn_images.append((cls, path.name, warns))
            log_lines.append(msg)
            print(msg)

            if len(montage_imgs[cls]) < 3:
                montage_imgs[cls].append(apply_crop(img))

    # summary ----------------------------------------------------------------
    total_tested = sum(min(N_PER_CLASS, len(by_cls[c])) for c in CLASSES)
    verdict = "ALL PASS" if not warn_images else f"{len(warn_images)} WARNINGS"
    summary = (f"\n{'='*60}\n"
               f"Tested {total_tested} images ({N_PER_CLASS}/class).  {verdict}")
    print(summary)
    log_lines.append(summary)

    if warn_images:
        note = ("NOTE: WARN means a crop edge falls in a brighter area.\n"
                "  Inspect the montage -- if the coil is fully visible, the crop is fine.\n"
                "  If the coil is cut off, expand the crop box in config.yaml.")
        print(note)
        log_lines.append(note)

    # save log ---------------------------------------------------------------
    log_path = REPORTS / "crop_validation.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    # montage: 3 per class = 9 crops, 3 cols x 3 rows -----------------------
    thumb_w, thumb_h = 448, 192
    canvas = Image.new("RGB", (3 * thumb_w, 3 * thumb_h), color=(30, 30, 30))
    for row_i, cls in enumerate(CLASSES):
        for col_i, crop_img in enumerate(montage_imgs[cls][:3]):
            thumb = crop_img.resize((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(thumb, (col_i * thumb_w, row_i * thumb_h))
    montage_path = REPORTS / "crop_validation.png"
    canvas.save(montage_path)

    print(f"\nMontage -> {montage_path}")
    print(f"Log     -> {log_path}")

    # Real gate: fail if too many sampled images warn (crop box likely cuts
    # content). A handful of bright-edge warnings is fine; a majority is not.
    warn_frac = len(warn_images) / max(total_tested, 1)
    if warn_frac > WARN_FAIL_FRAC:
        print(f"\nFAIL: {warn_frac:.0%} of images warn (> {WARN_FAIL_FRAC:.0%}). "
              f"Inspect the montage and fix the crop box in config.yaml.")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
