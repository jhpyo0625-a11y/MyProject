"""
T4.1 -- Inference wrapper. Single entry point for all coil defect prediction.

Handles two model backends (auto-detected from models/production/):
  - Fine-tuned PyTorch model  (model.pt)      -- preferred
  - Frozen backbone + sklearn (classifier.pkl) -- fallback

Usage:
    predictor = Predictor()
    result = predictor.predict("path/to/image.bmp")
    # {
    #   "label":         "Dent",
    #   "pass_fail":     "Fail",      # binary gate (Review -> Fail: never auto-pass)
    #   "decision":      "Fail",      # Pass | Fail | Review
    #   "probabilities": {"Pass": 0.08, "Dent": 0.76, "Loose": 0.16},
    #   "p_fail":        0.92,
    #   "latency_ms":    87.3,
    # }
"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Union

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.config_loader import load_config

LABEL_NAMES = ["Pass", "Dent", "Loose"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Thin inference model (mirrors CoilNet in finetune.py, no training logic)
# ---------------------------------------------------------------------------

class _InferenceNet(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """
    Load a production model and run inference on raw BMP images.

    The predictor applies the full pipeline:
        raw 2448x2048 BMP -> fixed crop -> backbone -> classifier -> decision

    Decision logic (spec §7):
        P(Fail) >= threshold + delta  -> Fail
        P(Fail) <= threshold - delta  -> Pass
        otherwise                     -> Review   (mapped to Fail in binary gate)
    """

    def __init__(self, config_path=None):
        cfg      = load_config(config_path)
        prod_dir = ROOT / cfg["paths"]["production_dir"]

        crop_c         = cfg["preprocessing"]["crop"]
        self.crop      = (crop_c["x_min"], crop_c["y_min"],
                          crop_c["x_max"], crop_c["y_max"])
        self.threshold = float(cfg["decision"]["pass_threshold"])
        self.delta     = float(cfg["decision"]["review_band_delta"])

        if (prod_dir / "model.pt").exists():
            self._load_pytorch(prod_dir)
            self._type = "pytorch"
            print(f"Predictor: fine-tuned PyTorch model ({prod_dir / 'model.pt'})")
        elif (prod_dir / "classifier.pkl").exists():
            self._load_sklearn(prod_dir, cfg)
            self._type = "sklearn"
            print(f"Predictor: sklearn pipeline ({prod_dir / 'classifier.pkl'})")
        else:
            raise FileNotFoundError(
                f"No model found in {prod_dir}. "
                "Run src/training/finetune.py or src/training/train.py first."
            )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_pytorch(self, prod_dir: Path) -> None:
        ckpt = torch.load(prod_dir / "model.pt", map_location="cpu",
                          weights_only=False)
        backbone = timm.create_model(
            ckpt["backbone"], pretrained=False,
            num_classes=0, global_pool="avg",
        )
        head = nn.Sequential(
            nn.Dropout(ckpt["dropout"]),
            nn.Linear(backbone.num_features, ckpt["n_classes"]),
        )
        net = _InferenceNet(backbone, head)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

        self._net      = net
        self._input_w  = ckpt["input_w"]
        self._input_h  = ckpt["input_h"]
        self.threshold = float(ckpt.get("threshold", self.threshold))

        self._val_tf = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _load_sklearn(self, prod_dir: Path, cfg: dict) -> None:
        with open(prod_dir / "classifier.pkl", "rb") as f:
            self._clf = pickle.load(f)

        emb_info_path = ROOT / cfg["paths"]["embeddings_dir"] / "backbone_info.json"
        with open(emb_info_path, encoding="utf-8") as f:
            info = json.load(f)

        self._backbone = timm.create_model(
            info["backbone"], pretrained=True,
            num_classes=0, global_pool="",   # return spatial map
        )
        self._backbone.eval()

        self._sk_resize_w = cfg["preprocessing"]["resize"]["width"]
        self._sk_resize_h = cfg["preprocessing"]["resize"]["height"]

        # Pull threshold from saved metadata if available
        meta_path = prod_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self.threshold = float(meta.get("pass_threshold", self.threshold))

    # ------------------------------------------------------------------
    # Probability extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _probs_pytorch(self, crop_arr: np.ndarray) -> np.ndarray:
        img = Image.fromarray(crop_arr).resize(
            (self._input_w, self._input_h), Image.BILINEAR,
        )
        tensor = self._val_tf(img).unsqueeze(0)          # (1, C, H, W)
        logits = self._net(tensor)
        return torch.softmax(logits, dim=-1).numpy()[0]  # (3,)

    @torch.no_grad()
    def _probs_sklearn(self, crop_arr: np.ndarray) -> np.ndarray:
        # Resize to backbone's expected input, normalize
        img_resized = Image.fromarray(crop_arr).resize(
            (self._sk_resize_w, self._sk_resize_h), Image.LANCZOS,
        )
        t = torch.from_numpy(np.asarray(img_resized, dtype=np.float32) / 255.0)
        t = t.permute(2, 0, 1)                               # (C, H, W)
        t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        feats = self._backbone(t.unsqueeze(0))               # (1, D, H', W')
        emb   = feats.amax(dim=(2, 3)).numpy()               # (1, D) global max pool
        return self._clf.predict_proba(emb)[0]               # (3,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_path: Union[str, Path]) -> dict:
        """
        Full-pipeline inference on one raw coil image.

        Parameters
        ----------
        image_path : path to a 2448x2048 BMP file

        Returns
        -------
        dict with keys:
            label        : 3-class argmax prediction ("Pass", "Dent", or "Loose")
            pass_fail    : binary gate ("Pass" or "Fail")
            decision     : 3-state decision ("Pass", "Fail", or "Review")
            probabilities: {"Pass": float, "Dent": float, "Loose": float}
            p_fail       : P(Dent) + P(Loose)  (the threshold input)
            latency_ms   : wall-clock time in ms (includes image load + crop)
        """
        t0 = time.perf_counter()

        img  = Image.open(image_path).convert("RGB")
        crop = np.asarray(img.crop(self.crop), dtype=np.uint8)

        probs = (self._probs_pytorch(crop) if self._type == "pytorch"
                 else self._probs_sklearn(crop))

        label  = LABEL_NAMES[int(probs.argmax())]
        p_fail = float(1.0 - probs[0])   # P(Fail) = 1 - P(Pass)

        if p_fail >= self.threshold + self.delta:
            decision = "Fail"
        elif p_fail <= self.threshold - self.delta:
            decision = "Pass"
        else:
            decision = "Review"

        # Conservative binary gate: when uncertain, never auto-pass
        pass_fail = "Pass" if decision == "Pass" else "Fail"

        return {
            "label":         label,
            "pass_fail":     pass_fail,
            "decision":      decision,
            "probabilities": {n: round(float(p), 4)
                              for n, p in zip(LABEL_NAMES, probs)},
            "p_fail":        round(p_fail, 4),
            "latency_ms":    round((time.perf_counter() - t0) * 1000, 1),
        }

    def predict_batch(self, image_paths) -> list[dict]:
        """Convenience wrapper: run predict() on each path in order."""
        return [self.predict(p) for p in image_paths]
