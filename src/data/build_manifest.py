"""
Build data/manifest.csv from the raw dataset folder structure.

Columns:
  filepath     - absolute path to the BMP
  label        - Pass | Dent | Loose  (derived from folder)
  session_id   - YYMMDD_HHMMSS prefix from filename
  sensor       - sensor ID (e.g. A35W)
  position     - coil position number
  layer        - layer number
  param        - bracketed parameter value in filename (preserved as-is)
  fold         - assigned in a later step (blank until split.py runs)
"""

import re
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DATASET_ROOT = ROOT / cfg["paths"]["dataset_root"]
OUT_CSV = ROOT / cfg["paths"]["manifest"]

# Map folder path → label
CLASS_DIRS = {
    DATASET_ROOT / "Pass":       "Pass",
    DATASET_ROOT / "Fail" / "Dent":  "Dent",
    DATASET_ROOT / "Fail" / "Loose": "Loose",
}

# filename: 250825_152739_A35W_10-2 [16384].bmp
FNAME_RE = re.compile(
    r"^(?P<session>\d{6}_\d{6})_(?P<sensor>[^_]+)_(?P<position>\d+)-(?P<layer>\d+)"
    r"(?:\s+\[(?P<param>\d+)\])?\.bmp$",
    re.IGNORECASE,
)


def parse_filename(name: str) -> dict:
    m = FNAME_RE.match(name)
    if not m:
        return {}
    return m.groupdict()


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []

    for folder, label in CLASS_DIRS.items():
        if not folder.exists():
            print(f"  WARNING: folder not found: {folder}")
            continue
        bmps = sorted(folder.glob("*.bmp"))
        for bmp in bmps:
            parsed = parse_filename(bmp.name)
            if not parsed:
                skipped.append(str(bmp))
                continue
            rows.append({
                "filepath":   str(bmp),
                "label":      label,
                "session_id": parsed["session"],
                "sensor":     parsed["sensor"],
                "position":   parsed["position"],
                "layer":      parsed["layer"],
                "param":      parsed.get("param") or "",
                "fold":       "",
            })

    fieldnames = ["filepath", "label", "session_id", "sensor", "position", "layer", "param", "fold"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    print(f"\nManifest written to: {OUT_CSV}")
    print(f"  Total images : {len(rows)}")
    for cls in ["Pass", "Dent", "Loose"]:
        print(f"  {cls:<8}: {counts[cls]}")
    if skipped:
        print(f"\n  WARNING: {len(skipped)} files could not be parsed and were skipped:")
        for s in skipped:
            print(f"    {s}")
    else:
        print("\n  No parsing errors.")


if __name__ == "__main__":
    main()
