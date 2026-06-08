"""
T4.1 -- Inference wrapper. Single entry point for all coil defect prediction.

Handles three model backends (auto-detected from models/production/):
  - PaDiM anomaly detector    (padim.pkl)     -- preferred (human-assist deploy)
  - Fine-tuned PyTorch model  (model.pt)      -- supervised 3-class
  - Frozen backbone + sklearn (classifier.pkl) -- legacy fallback

The PaDiM backend is the deployed one: supervised and anomaly approaches both
capped at ~55% Fail-recall @20% false-reject, so the system runs as a
human-assist triage (3-band: AUTO-PASS / REVIEW / AUTO-FLAG), never a silent
>=95% auto-pass gate. See models/production/padim_card.md.

Usage:
    predictor = Predictor()
    result = predictor.predict("path/to/image.bmp")
    # PaDiM backend:
    # {
    #   "backend":       "padim",
    #   "decision":      "Review",        # Pass | Review | Fail
    #   "band":          "REVIEW",        # AUTO-PASS | REVIEW | AUTO-FLAG
    #   "pass_fail":     "Fail",          # Pass only if AUTO-PASS (never auto-pass when unsure)
    #   "anomaly_score": 31.7,            # max patch Mahalanobis
    #   "score_norm":    0.43,            # 0..1 (score / t_flag), for display
    #   "latency_ms":    47.3,
    # }
    # Supervised backend additionally returns label / probabilities / p_fail.
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
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, UnidentifiedImageError

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

        if (prod_dir / "padim.pkl").exists():
            self._load_padim(prod_dir)
            self._type = "padim"
            print(f"Predictor: PaDiM anomaly detector ({prod_dir / 'padim.pkl'})")
        elif (prod_dir / "model.pt").exists():
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
                "Run src/training/anomaly.py (PaDiM) or src/training/finetune.py first."
            )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_padim(self, prod_dir: Path) -> None:
        with open(prod_dir / "padim.pkl", "rb") as f:
            a = pickle.load(f)
        self._pd_model = timm.create_model(
            a["backbone"], pretrained=True, features_only=True,
            out_indices=tuple(a["out_indices"]),
        )
        self._pd_model.eval()
        self._pd_mean    = a["mean"]                       # (L, R)
        self._pd_inv     = a["inv"]                         # (L, R, R)
        self._pd_chan    = np.asarray(a["chan_idx"])
        self._input_w    = a["input_w"]
        self._input_h    = a["input_h"]
        self._pd_t_low   = float(a["t_low"])
        self._pd_t_flag  = float(a["t_flag"])
        self._pd_norm_mean = torch.tensor(a["imagenet_mean"]).view(1, 3, 1, 1)
        self._pd_norm_std  = torch.tensor(a["imagenet_std"]).view(1, 3, 1, 1)

    def _load_pytorch(self, prod_dir: Path) -> None:
        # weights_only=True: the checkpoint holds only tensors + plain
        # str/int/float/dict, so the safe loader suffices and won't execute a
        # pickle payload from a tampered model file.
        ckpt = torch.load(prod_dir / "model.pt", map_location="cpu",
                          weights_only=True)
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
    def _padim_score_map(self, crop_arr: np.ndarray):
        """uint8 crop -> (image score, per-patch anomaly map of shape (h, w)).

        The image score is the MAX patch Mahalanobis distance; the map is the
        same per-patch distances reshaped to the stride-8 grid -- i.e. WHERE the
        anomaly is. _score_padim() keeps only the scalar; localize() keeps both.
        """
        t = (torch.from_numpy(crop_arr.copy()).float().div(255.0)
             .permute(2, 0, 1)[None])
        t = F.interpolate(t, size=(self._input_h, self._input_w),
                          mode="bilinear", align_corners=False)
        t = (t - self._pd_norm_mean) / self._pd_norm_std
        maps = self._pd_model(t)
        ref_hw = maps[-1].shape[-2:]
        aligned = [F.interpolate(m, size=ref_hw, mode="bilinear",
                                 align_corners=False) for m in maps]
        feat = torch.cat(aligned, dim=1)[:, self._pd_chan]   # (1, R, h, w)
        C = feat.shape[1]
        X = feat.squeeze(0).reshape(C, -1).T.numpy()         # (L, R)
        d = X - self._pd_mean                                # (L, R)
        m = np.einsum("lr,lrs,ls->l", d, self._pd_inv, d)    # (L,)
        dist = np.sqrt(np.maximum(m, 0.0))                   # (L,)
        h, w = int(ref_hw[0]), int(ref_hw[1])
        return float(dist.max()), dist.reshape(h, w)         # scalar, (h, w)

    def _score_padim(self, crop_arr: np.ndarray) -> float:
        """uint8 crop -> image anomaly score (max patch Mahalanobis distance)."""
        return self._padim_score_map(crop_arr)[0]

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
    # Decision logic (pure functions -- unit-testable without a model)
    # ------------------------------------------------------------------

    @staticmethod
    def _decide_supervised(p_fail: float, threshold: float, delta: float):
        """Map P(Fail) to a (decision, pass_fail) pair. Review never auto-passes."""
        if p_fail >= threshold + delta:
            decision = "Fail"
        elif p_fail <= threshold - delta:
            decision = "Pass"
        else:
            decision = "Review"
        pass_fail = "Pass" if decision == "Pass" else "Fail"
        return decision, pass_fail

    @staticmethod
    def _decide_padim(score: float, t_low: float, t_flag: float):
        """Map an anomaly score to a (band, decision) pair (3-band human-assist)."""
        if score >= t_flag:
            return "AUTO-FLAG", "Fail"
        if score < t_low:
            return "AUTO-PASS", "Pass"
        return "REVIEW", "Review"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _load_crop(self, image_path: Union[str, Path]) -> np.ndarray:
        """Read a raw frame and apply the fixed crop, validating the input size."""
        try:
            img = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Cannot read image {image_path}: {e}") from e

        x_max, y_max = self.crop[2], self.crop[3]
        if img.width < x_max or img.height < y_max:
            raise ValueError(
                f"Image {Path(image_path).name} is {img.width}x{img.height}, "
                f"smaller than the crop box (needs >= {x_max}x{y_max}). Pass the "
                "full raw frame, not a pre-cropped image -- cropping out of "
                "bounds would silently produce a black-padded, wrong prediction.")
        return np.asarray(img.crop(self.crop), dtype=np.uint8)

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

        crop = self._load_crop(image_path)

        if self._type == "padim":
            return self._predict_padim(crop, t0)

        probs = (self._probs_pytorch(crop) if self._type == "pytorch"
                 else self._probs_sklearn(crop))

        label  = LABEL_NAMES[int(probs.argmax())]
        p_fail = float(1.0 - probs[0])   # P(Fail) = 1 - P(Pass)

        decision, pass_fail = self._decide_supervised(
            p_fail, self.threshold, self.delta)

        return {
            "label":         label,
            "pass_fail":     pass_fail,
            "decision":      decision,
            "probabilities": {n: round(float(p), 4)
                              for n, p in zip(LABEL_NAMES, probs)},
            "p_fail":        round(p_fail, 4),
            "latency_ms":    round((time.perf_counter() - t0) * 1000, 1),
        }

    def _padim_result(self, score: float, t0: float) -> dict:
        """Build the human-assist result dict from a PaDiM image score."""
        band, decision = self._decide_padim(
            score, self._pd_t_low, self._pd_t_flag)
        return {
            "backend":       "padim",
            "decision":      decision,
            "band":          band,
            # Conservative gate: only an AUTO-PASS band clears; review/flag -> Fail
            "pass_fail":     "Pass" if band == "AUTO-PASS" else "Fail",
            "anomaly_score": round(score, 3),
            "score_norm":    round(min(score / self._pd_t_flag, 1.0), 4),
            "thresholds":    {"t_low": round(self._pd_t_low, 3),
                              "t_flag": round(self._pd_t_flag, 3)},
            "latency_ms":    round((time.perf_counter() - t0) * 1000, 1),
        }

    def _predict_padim(self, crop: np.ndarray, t0: float) -> dict:
        """Human-assist 3-band decision from the PaDiM anomaly score."""
        return self._padim_result(self._score_padim(crop), t0)

    def localize(self, image_path: Union[str, Path]) -> dict:
        """Like predict(), but for the PaDiM backend also returns the spatial
        anomaly heatmap so the UI can show WHERE the coil was flagged.

        Extra keys (PaDiM only):
            amap    : (h, w) float ndarray of per-patch Mahalanobis distances
            amap_hw : (h, w) grid shape
            peak    : (row, col) of the hottest patch (the one that set the score)
        For non-PaDiM backends `amap` is None. One forward pass, on-demand only --
        predict() and the batch path stay scalar so CSV export is unaffected."""
        if self._type != "padim":
            return {**self.predict(image_path), "amap": None}
        t0 = time.perf_counter()
        crop = self._load_crop(image_path)
        score, amap = self._padim_score_map(crop)
        res = self._padim_result(score, t0)
        res["amap"]    = amap
        res["amap_hw"] = (int(amap.shape[0]), int(amap.shape[1]))
        res["peak"]    = tuple(int(v) for v in
                               np.unravel_index(int(amap.argmax()), amap.shape))
        return res

    def predict_batch(self, image_paths) -> list[dict]:
        """Convenience wrapper: run predict() on each path in order."""
        return [self.predict(p) for p in image_paths]
