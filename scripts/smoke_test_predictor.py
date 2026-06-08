"""Quick smoke-test: one prediction per class."""
import csv, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.predictor import Predictor

random.seed(42)
pred = Predictor()
print()
with open(ROOT / "data/manifest.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for cls in ["Pass", "Dent", "Loose"]:
    row = random.choice([r for r in rows if r["label"] == cls])
    r = pred.predict(row["filepath"])
    if r.get("backend") == "padim":
        print(f"{cls:6} -> decision={r['decision']:8}  band={r['band']:10}  "
              f"score={r['anomaly_score']:.2f}  {r['latency_ms']:.0f}ms")
    else:
        print(f"{cls:6} -> label={r['label']:6}  decision={r['decision']:8}  "
              f"p_fail={r['p_fail']:.3f}  {r['latency_ms']:.0f}ms")
        print(f"         {r['probabilities']}")

# localize() exposes the spatial heatmap (defect localization) -- verify it is
# consistent: the map's max must equal the reported scalar score, and the peak
# patch must lie inside the grid.
loc = pred.localize(random.choice(rows)["filepath"])
if loc.get("amap") is not None:
    amap = loc["amap"]
    h, w = loc["amap_hw"]
    pr, pc = loc["peak"]
    assert amap.shape == (h, w), amap.shape
    assert 0 <= pr < h and 0 <= pc < w, loc["peak"]
    assert abs(float(amap.max()) - loc["anomaly_score"]) < 0.01
    print(f"localize ok: heatmap {amap.shape}  peak=(r{pr},c{pc})  "
          f"max==score ({loc['anomaly_score']:.2f})")
else:
    print("localize ok: non-PaDiM backend (no heatmap)")
