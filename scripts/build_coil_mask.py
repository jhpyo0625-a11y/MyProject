"""Build the committed coil template (mask + alignment reference) for the
clean-coil display toggle. Source: Real-coil-Part/Pass.bmp (the coil isolated on
black). Re-run if the crop box in config.yaml changes.

    python scripts/build_coil_mask.py
"""
import sys
import numpy as np
import yaml
from pathlib import Path
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
c = cfg["preprocessing"]["crop"]
box = (c["x_min"], c["y_min"], c["x_max"], c["y_max"])

src = ROOT / "Real-coil-Part" / "Pass.bmp"
if not src.exists():
    sys.exit(f"need {src} (the isolated-coil reference) to build the template")

W = 850
H = round(W * (box[3] - box[1]) / (box[2] - box[0]))
ref = Image.open(src).convert("RGB").crop(box).resize((W, H), Image.BILINEAR)
g = np.asarray(ref.convert("L"), dtype=np.uint8)

m = (g > 28).astype(np.uint8) * 255                       # non-black = coil annulus
k = (max(3, W // 170) | 1)                                # odd morphology kernel
m = np.asarray(Image.fromarray(m).filter(ImageFilter.MaxFilter(k))
                                 .filter(ImageFilter.MinFilter(k)))
mask = m > 127

out = ROOT / "src" / "app" / "assets"
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "coil_template.npz",
                    mask=mask, ref_gray=g, crop=np.array(box))
print(f"saved {out/'coil_template.npz'}  mask={mask.shape}  "
      f"coverage={100*mask.mean():.1f}%")
