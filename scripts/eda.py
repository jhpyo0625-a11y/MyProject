"""
T1.2 — Exploratory Data Analysis.

Outputs:
  reports/eda_report.md           -- text report (counts, stats, sessions, flags)
  reports/eda_samples.png         -- 5 thumbnail crops per class (15 total)
  reports/eda_brightness.png      -- per-class brightness distributions
"""

import sys
import random
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.config_loader import load_config

cfg      = load_config()
MANIFEST = ROOT / cfg["paths"]["manifest"]
REPORTS  = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

CLASSES     = ["Pass", "Dent", "Loose"]
N_THUMB     = 5
N_BRIGHTNESS = 40   # images sampled per class for brightness distribution

crop_cfg = cfg["preprocessing"]["crop"]
CROP     = (crop_cfg["x_min"], crop_cfg["y_min"],
            crop_cfg["x_max"], crop_cfg["y_max"])


# ---------------------------------------------------------------------------
def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def crop_image(img):
    x0, y0, x1, y1 = CROP
    return img.crop((x0, y0, x1, y1))


def image_brightness(path):
    img  = Image.open(path).convert("L")
    arr  = np.asarray(img, dtype=np.float32)
    return float(arr.mean())


# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    rows    = load_manifest()
    by_cls  = {c: [r for r in rows if r["label"] == c] for c in CLASSES}

    report = []
    report.append("# EDA Report — Coil Defect Dataset\n")

    # --- class counts -------------------------------------------------------
    report.append("## Class Distribution\n")
    report.append("| Class | Count | % |\n|---|---|---|")
    total = len(rows)
    for cls in CLASSES:
        n = len(by_cls[cls])
        report.append(f"| {cls} | {n} | {100*n/total:.1f}% |")
    report.append(f"| **Total** | **{total}** | 100% |\n")

    # --- image dimensions ---------------------------------------------------
    sample_path = Path(rows[0]["filepath"])
    sample_img  = Image.open(sample_path)
    report.append(f"## Image Properties\n")
    report.append(f"- Size: {sample_img.width} x {sample_img.height} px")
    report.append(f"- Mode: {sample_img.mode}")
    report.append(f"- Crop box: x=[{CROP[0]},{CROP[2]}]  y=[{CROP[1]},{CROP[3]}]")
    report.append(f"- Crop size: {CROP[2]-CROP[0]} x {CROP[3]-CROP[1]} px\n")

    # --- brightness analysis ------------------------------------------------
    report.append("## Per-Class Brightness (mean gray value, full image)\n")
    brightness = {}
    flagged    = []
    for cls in CLASSES:
        sample = random.sample(by_cls[cls], min(N_BRIGHTNESS, len(by_cls[cls])))
        vals   = []
        for row in sample:
            b = image_brightness(Path(row["filepath"]))
            vals.append(b)
            if b < 30:
                flagged.append((cls, row["filepath"], f"very dark ({b:.1f})"))
            elif b > 210:
                flagged.append((cls, row["filepath"], f"very bright ({b:.1f})"))
        brightness[cls] = np.array(vals)
        report.append(f"- **{cls}**: mean={np.mean(vals):.1f}  std={np.std(vals):.1f}"
                       f"  min={np.min(vals):.1f}  max={np.max(vals):.1f}"
                       f"  (n={len(vals)})")

    # brightness plot
    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = {"Pass": "#4caf50", "Dent": "#f44336", "Loose": "#ff9800"}
    for cls in CLASSES:
        ax.hist(brightness[cls], bins=20, alpha=0.6, label=cls, color=colors[cls])
    ax.set_xlabel("Mean image brightness (0-255)")
    ax.set_ylabel("Count")
    ax.set_title("Per-class brightness distribution")
    ax.legend()
    plt.tight_layout()
    brightness_path = REPORTS / "eda_brightness.png"
    fig.savefig(brightness_path, dpi=100)
    plt.close(fig)
    report.append(f"\n![Brightness distribution](eda_brightness.png)\n")

    # --- flagged images -----------------------------------------------------
    report.append("## Flagged Images\n")
    if flagged:
        report.append("| Class | File | Reason |")
        report.append("|---|---|---|")
        for cls, fp, reason in flagged:
            report.append(f"| {cls} | {Path(fp).name[:50]} | {reason} |")
    else:
        report.append("None — all sampled images are within normal brightness range.\n")

    # --- session distribution -----------------------------------------------
    report.append("\n## Session Distribution (top 10 sessions by image count)\n")
    session_count = Counter(r["session_id"] for r in rows)
    session_cls   = defaultdict(Counter)
    for r in rows:
        session_cls[r["session_id"]][r["label"]] += 1

    report.append("| Session | Total | Pass | Dent | Loose |")
    report.append("|---|---|---|---|---|")
    for sess, cnt in session_count.most_common(10):
        sc = session_cls[sess]
        report.append(f"| {sess} | {cnt} | {sc['Pass']} | {sc['Dent']} | {sc['Loose']} |")

    unique_sessions = len(session_count)
    report.append(f"\nTotal unique sessions: **{unique_sessions}**\n")

    # --- thumbnail montage --------------------------------------------------
    # 5 thumbs per class, arranged as 3 rows x 5 cols
    thumb_w, thumb_h = 280, 120
    canvas = Image.new("RGB",
                       (N_THUMB * thumb_w, len(CLASSES) * thumb_h),
                       color=(20, 20, 20))
    for row_i, cls in enumerate(CLASSES):
        sample = random.sample(by_cls[cls], min(N_THUMB, len(by_cls[cls])))
        for col_i, row in enumerate(sample):
            img   = Image.open(Path(row["filepath"]))
            crop  = crop_image(img)
            thumb = crop.resize((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(thumb, (col_i * thumb_w, row_i * thumb_h))
    samples_path = REPORTS / "eda_samples.png"
    canvas.save(samples_path)
    report.append(f"## Sample Crops (5 per class)\n\n![Sample crops](eda_samples.png)\n")

    # --- write report -------------------------------------------------------
    report_path = REPORTS / "eda_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"EDA complete.")
    print(f"  Report:     {report_path}")
    print(f"  Samples:    {samples_path}")
    print(f"  Brightness: {brightness_path}")
    if flagged:
        print(f"\n  WARNING: {len(flagged)} suspicious image(s) flagged — see eda_report.md")


if __name__ == "__main__":
    main()
