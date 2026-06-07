"""
T7.3 -- Model comparison report.

Scores the current production model and the candidate model on the locked
comparison fold (fold 4) and produces a side-by-side report: recall at matched
false-reject budgets, plus the human-assist band workload (auto-pass / review /
auto-flag) and the critical metric -- defects that would be silently
auto-passed. Saved to reports/model_comparison_{timestamp}.md and returned as a
dict for the Retrain tab.

Both models score the SAME fold-4 images (fold 4 never receives new labels), so
the comparison is apples-to-apples. Both share the seeded channel subset, so
their scores live in the same feature space.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config
from src.training.anomaly import (
    PaDiM, load_features, recall_at_fr, _band_split, COMP_FOLD,
)

cfg      = load_config()
REPORTS  = ROOT / cfg["paths"]["reports"]
MODELS   = (ROOT / cfg["paths"]["production_dir"]).parent
FR_BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30)


def _load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _score_fold4(artifact, X, y, folds, chan_idx):
    if not np.array_equal(np.asarray(artifact["chan_idx"]), chan_idx):
        raise ValueError("channel subset mismatch -- model uses a different "
                         "feature config; cannot compare on shared features.")
    det = PaDiM()
    det.mean = artifact["mean"]
    det.inv  = artifact["inv"]
    f4 = folds == COMP_FOLD
    scores = det.score(X[f4])
    ybin   = (y[f4] != 0).astype(int)
    table  = {b: recall_at_fr(scores, ybin, b)[0] for b in FR_BUDGETS}
    split  = _band_split(scores, ybin, artifact["t_low"], artifact["t_flag"])
    return {"recall_at_fr": table, "band_split": split,
            "t_low": artifact["t_low"], "t_flag": artifact["t_flag"],
            "n_fold4": int(f4.sum()), "n_defects": int(ybin.sum())}


def compare(timestamp, prod_dir=None, cand_dir=None) -> dict:
    prod_dir = Path(prod_dir or MODELS / "production")
    cand_dir = Path(cand_dir or MODELS / "candidate")
    prod_pkl = prod_dir / "padim.pkl"
    cand_pkl = cand_dir / "padim.pkl"
    if not prod_pkl.exists():
        raise FileNotFoundError(f"no production model at {prod_pkl}")
    if not cand_pkl.exists():
        raise FileNotFoundError(f"no candidate model at {cand_pkl} "
                                "-- run retrain first")

    X, y, folds, chan_idx = load_features(verbose=False)
    prod = _score_fold4(_load_artifact(prod_pkl), X, y, folds, chan_idx)
    cand = _score_fold4(_load_artifact(cand_pkl), X, y, folds, chan_idx)

    md = _render_md(timestamp, prod, cand)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"model_comparison_{timestamp}.md"
    out.write_text(md, encoding="utf-8")

    # headline deltas (positive = candidate better)
    rec20_delta = cand["recall_at_fr"][0.20] - prod["recall_at_fr"][0.20]
    miss_delta  = (cand["band_split"]["autopass_defect_miss"]
                   - prod["band_split"]["autopass_defect_miss"])
    return {
        "report_path": str(out), "markdown": md,
        "prod": prod, "cand": cand,
        "recall20_delta": rec20_delta,
        "autopass_miss_delta": miss_delta,
        # a safe default recommendation: better/equal recall AND no more leakage
        "candidate_better": rec20_delta >= 0 and miss_delta <= 0,
    }


def _render_md(ts, prod, cand) -> str:
    def pct(x):
        return f"{x:.1%}"

    lines = [
        f"# Model comparison -- {ts}",
        "",
        f"Both models scored on the locked comparison fold "
        f"(fold {COMP_FOLD}: {prod['n_fold4']} images, "
        f"{prod['n_defects']} defects). Higher recall at a given false-reject "
        f"budget is better; auto-passed defects must stay ~0.",
        "",
        "## Recall @ false-reject budget (fold 4)",
        "",
        "| FR budget | Production | Candidate | delta |",
        "|---|---|---|---|",
    ]
    for b in FR_BUDGETS:
        p, c = prod["recall_at_fr"][b], cand["recall_at_fr"][b]
        lines.append(f"| <= {b:.0%} | {pct(p)} | {pct(c)} | {c-p:+.1%} |")

    ps, cs = prod["band_split"], cand["band_split"]
    lines += [
        "",
        "## Human-assist workload + safety (fold 4, each at its own thresholds)",
        "",
        "| Metric | Production | Candidate |",
        "|---|---|---|",
        f"| AUTO-PASS share | {pct(ps['auto_pass'])} | {pct(cs['auto_pass'])} |",
        f"| REVIEW share | {pct(ps['review'])} | {pct(cs['review'])} |",
        f"| AUTO-FLAG share | {pct(ps['flag'])} | {pct(cs['flag'])} |",
        f"| **Defects auto-passed (critical)** | **{pct(ps['autopass_defect_miss'])}** "
        f"| **{pct(cs['autopass_defect_miss'])}** |",
        f"| Defects auto-flagged | {pct(ps['flag_defect_recall'])} "
        f"| {pct(cs['flag_defect_recall'])} |",
        "",
        f"_Recall @20% FR change: {cand['recall_at_fr'][0.20]-prod['recall_at_fr'][0.20]:+.1%}; "
        f"auto-passed-defect change: "
        f"{cs['autopass_defect_miss']-ps['autopass_defect_miss']:+.1%}._",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    res = compare(ts)
    print(res["markdown"])
    print(f"\nCandidate better: {res['candidate_better']}")
