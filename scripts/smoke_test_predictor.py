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
    print(f"{cls:6} -> label={r['label']:6}  decision={r['decision']:8}  "
          f"p_fail={r['p_fail']:.3f}  {r['latency_ms']:.0f}ms")
    print(f"         {r['probabilities']}")
