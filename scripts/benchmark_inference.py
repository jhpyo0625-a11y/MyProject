"""
T4.2 -- Inference latency benchmark.

Runs predictor.predict() on 20 randomly sampled raw BMP images and reports:
  - Mean / min / p95 / max end-to-end latency in ms (load + crop + forward)
  - Decision breakdown (Pass / Fail / Review)
  - Pass/fail verdict against the 1 s/image spec limit

Saves a text report to reports/latency_benchmark.txt.

Run:
    python scripts/benchmark_inference.py
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.inference.predictor import Predictor

cfg      = load_config()
MANIFEST = ROOT / cfg["paths"]["manifest"]
REPORTS  = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

N_IMAGES = 20
SPEC_LIMIT_MS = 1000.0


def main():
    random.seed(42)

    with open(MANIFEST, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("is_augmented", "False") == "False"
                or "is_augmented" not in r]

    # Sample: at least 5 from each class if available
    by_cls = {"Pass": [], "Dent": [], "Loose": []}
    for r in rows:
        if r["label"] in by_cls:
            by_cls[r["label"]].append(r["filepath"])

    sample = []
    for cls, paths in by_cls.items():
        n = min(5, len(paths))
        sample += random.sample(paths, n)
    # Pad to N_IMAGES with random unused paths, but never loop forever if the
    # dataset has fewer than N_IMAGES unique images.
    sample_set = set(sample)
    remaining  = [r["filepath"] for r in rows if r["filepath"] not in sample_set]
    random.shuffle(remaining)
    sample += remaining[:max(0, N_IMAGES - len(sample))]
    if len(sample) < N_IMAGES:
        print(f"  NOTE: only {len(sample)} unique images available "
              f"(< {N_IMAGES} requested).")
    random.shuffle(sample)
    sample = sample[:N_IMAGES]

    print(f"Loading predictor...")
    predictor = Predictor()
    print(f"Running benchmark on {len(sample)} images...\n")

    latencies = []
    decisions = {"Pass": 0, "Fail": 0, "Review": 0}

    for i, path in enumerate(sample, 1):
        result = predictor.predict(path)
        lat = result["latency_ms"]
        latencies.append(lat)
        decisions[result["decision"]] += 1
        print(f"  [{i:2d}/{N_IMAGES}]  {Path(path).name[:50]:<52} "
              f"{result['decision']:<8}  {lat:.0f}ms")

    lats = np.array(latencies)
    mean_ms = float(lats.mean())
    p95_ms  = float(np.percentile(lats, 95))
    max_ms  = float(lats.max())
    min_ms  = float(lats.min())

    spec_ok = max_ms <= SPEC_LIMIT_MS
    verdict = "PASS" if spec_ok else "FAIL"

    report_lines = [
        "Inference Latency Benchmark",
        "=" * 40,
        f"Model type    : {predictor._type}",
        f"Images tested : {N_IMAGES}",
        f"Spec limit    : {SPEC_LIMIT_MS:.0f} ms/image",
        "",
        f"Mean          : {mean_ms:.1f} ms",
        f"Min           : {min_ms:.1f} ms",
        f"p95           : {p95_ms:.1f} ms",
        f"Max           : {max_ms:.1f} ms",
        "",
        f"Decisions     : Pass={decisions['Pass']}  Fail={decisions['Fail']}  "
        f"Review={decisions['Review']}",
        "",
        f"Verdict       : {verdict}  (max {max_ms:.1f}ms vs limit {SPEC_LIMIT_MS:.0f}ms)",
    ]

    print()
    for line in report_lines:
        print(line)

    report_path = REPORTS / "latency_benchmark.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved to {report_path}")

    sys.exit(0 if spec_ok else 1)


if __name__ == "__main__":
    main()
