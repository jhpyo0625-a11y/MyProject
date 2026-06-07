"""
CLI wrapper for single-image and batch inference.

Usage:
    # Single image
    python scripts/run_inference.py path/to/image.bmp

    # Batch (folder of BMPs)
    python scripts/run_inference.py path/to/folder/ --output results.csv

    # Show probabilities
    python scripts/run_inference.py path/to/image.bmp --verbose
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predictor import Predictor


def fmt_result(path: Path, r: dict, verbose: bool) -> str:
    if r.get("backend") == "padim":
        line = (f"{path.name:<55} "
                f"{r['band']:<10} "
                f"score={r['anomaly_score']:.1f}  "
                f"{r['latency_ms']:.0f}ms")
        if verbose:
            t = r["thresholds"]
            line += (f"\n  norm={r['score_norm']:.3f}  "
                     f"t_low={t['t_low']:.1f}  t_flag={t['t_flag']:.1f}  "
                     f"-> {r['decision']}")
        return line
    line = (f"{path.name:<55} "
            f"{r['decision']:<8} "
            f"p_fail={r['p_fail']:.3f}  "
            f"{r['latency_ms']:.0f}ms")
    if verbose:
        probs = r["probabilities"]
        line += (f"\n  Pass={probs['Pass']:.3f}  "
                 f"Dent={probs['Dent']:.3f}  "
                 f"Loose={probs['Loose']:.3f}")
    return line


def main():
    parser = argparse.ArgumentParser(description="Coil defect inference")
    parser.add_argument("path", help="BMP image file or folder of BMP files")
    parser.add_argument("--output", "-o", default=None,
                        help="Save batch results to CSV (batch mode only)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-class probabilities")
    args = parser.parse_args()

    target = Path(args.path)
    predictor = Predictor()
    print()

    if target.is_file():
        r = predictor.predict(target)
        print(fmt_result(target, r, args.verbose))
        sys.exit(0 if r["pass_fail"] == "Pass" else 1)

    # Batch mode
    bmps = sorted(target.glob("*.bmp"))
    if not bmps:
        print(f"No .bmp files found in {target}")
        sys.exit(2)

    results = []
    counts = {"Pass": 0, "Fail": 0, "Review": 0}
    for bmp in bmps:
        r = predictor.predict(bmp)
        results.append(r)
        counts[r["decision"]] += 1
        print(fmt_result(bmp, r, args.verbose))

    print(f"\nSummary: {len(bmps)} images  "
          f"Pass={counts['Pass']}  "
          f"Fail={counts['Fail']}  "
          f"Review={counts['Review']}")

    if args.output:
        out = Path(args.output)
        is_padim = results and results[0].get("backend") == "padim"
        if is_padim:
            fieldnames = ["filepath", "band", "decision", "pass_fail",
                          "anomaly_score", "score_norm", "latency_ms"]
        else:
            fieldnames = ["filepath", "label", "pass_fail", "decision",
                          "p_Pass", "p_Dent", "p_Loose", "p_fail", "latency_ms"]
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for bmp, r in zip(bmps, results):
                if is_padim:
                    w.writerow({
                        "filepath":      str(bmp),
                        "band":          r["band"],
                        "decision":      r["decision"],
                        "pass_fail":     r["pass_fail"],
                        "anomaly_score": r["anomaly_score"],
                        "score_norm":    r["score_norm"],
                        "latency_ms":    r["latency_ms"],
                    })
                else:
                    w.writerow({
                        "filepath":   str(bmp),
                        "label":      r["label"],
                        "pass_fail":  r["pass_fail"],
                        "decision":   r["decision"],
                        "p_Pass":     r["probabilities"]["Pass"],
                        "p_Dent":     r["probabilities"]["Dent"],
                        "p_Loose":    r["probabilities"]["Loose"],
                        "p_fail":     r["p_fail"],
                        "latency_ms": r["latency_ms"],
                    })
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
