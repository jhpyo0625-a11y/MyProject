"""
Embedding dataset loader for training.

Reads data/embeddings/index.csv and loads all cached .npy embedding
files into numpy arrays ready for scikit-learn.
"""

import csv
from pathlib import Path

import numpy as np

LABEL_MAP   = {"Pass": 0, "Dent": 1, "Loose": 2}
LABEL_NAMES = ["Pass", "Dent", "Loose"]
LABEL_INV   = {v: k for k, v in LABEL_MAP.items()}


def load_embeddings(cfg: dict, root: Path):
    """
    Load all cached embeddings.

    Returns
    -------
    X          : (N, D) float32  embedding matrix
    y3         : (N,)   int32    3-class labels (0=Pass, 1=Dent, 2=Loose)
    y_bin      : (N,)   int32    binary labels  (0=Pass, 1=Fail)
    folds      : (N,)   int32    fold index 0-4
    is_aug     : (N,)   bool     True for augmented rows
    labels_str : list[str]       original label string per row
    """
    emb_dir   = root / cfg["paths"]["embeddings_dir"]
    index_csv = emb_dir / "index.csv"

    with open(index_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    X_list, y3_list, yb_list = [], [], []
    fold_list, aug_list, lbl_list = [], [], []
    missing = []

    for row in rows:
        path = root / row["embedding_path"]
        if not path.exists():
            missing.append(row["stem"])
            continue

        X_list.append(np.load(path))
        lbl = row["label"]
        y3_list.append(LABEL_MAP[lbl])
        yb_list.append(0 if lbl == "Pass" else 1)
        fold_list.append(int(row["fold"]))
        aug_list.append(row["is_augmented"] == "True")
        lbl_list.append(lbl)

    if missing:
        print(f"  WARNING: {len(missing)} embedding file(s) not found -- skipped.")

    return (
        np.array(X_list,    dtype=np.float32),
        np.array(y3_list,   dtype=np.int32),
        np.array(yb_list,   dtype=np.int32),
        np.array(fold_list, dtype=np.int32),
        np.array(aug_list,  dtype=bool),
        lbl_list,
    )
