"""
T2.1 -- Frozen backbone feature extractor.

Uses a pretrained CNN from timm with global_pool='' to return the raw
spatial feature map, then applies global MAX pooling to produce a 1-D
embedding.

Why max (not avg): each frame shows 2-3 coil assemblies. Only one may be
defective. Max pooling preserves the strongest local defect signal rather
than averaging it away across the frame.

Usage:
    embedder = CoilEmbedder.from_config()
    arr = np.load("data/crops/Dent/some_image.npy")  # uint8 (H, W, 3)
    embedding = embedder.embed(arr)                   # float32 (D,)
"""

import torch
import torchvision.transforms.functional as TF
import timm
import numpy as np

from src.data.config_loader import load_config, get_root


class CoilEmbedder:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, backbone_name: str):
        self.backbone_name = backbone_name
        self.model = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="",   # return spatial (B, C, H', W'), not pooled
        )
        self.model.eval()
        self.embedding_dim: int = self.model.num_features

    @classmethod
    def from_config(cls, config_path=None) -> "CoilEmbedder":
        cfg = load_config(config_path)
        return cls(cfg["model"]["backbone"])

    def preprocess(self, arr: np.ndarray) -> torch.Tensor:
        """uint8 (H, W, 3) numpy → normalised float32 (1, C, H, W) tensor."""
        t = torch.from_numpy(arr.copy()).float() / 255.0  # (H, W, 3)
        t = t.permute(2, 0, 1)                            # (C, H, W)
        t = TF.normalize(t, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        return t.unsqueeze(0)                             # (1, C, H, W)

    @torch.no_grad()
    def embed(self, arr: np.ndarray) -> np.ndarray:
        """uint8 (H, W, 3) → float32 (embedding_dim,) embedding via global max pool."""
        tensor   = self.preprocess(arr)
        features = self.model(tensor)              # (1, C, H', W')
        pooled   = features.amax(dim=(2, 3))       # (1, C)  -- global max
        return pooled.squeeze(0).numpy()           # (C,)

    def benchmark(self, arr: np.ndarray, n: int = 10) -> float:
        """Return mean forward-pass latency in ms over n runs."""
        import time
        self.embed(arr)  # warmup
        t0 = time.perf_counter()
        for _ in range(n):
            self.embed(arr)
        return (time.perf_counter() - t0) / n * 1000


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(get_root()))
    embedder = CoilEmbedder.from_config()
    print(f"Backbone : {embedder.backbone_name}")
    print(f"Emb dim  : {embedder.embedding_dim}")

    # smoke test with a random crop-shaped array
    dummy = np.random.randint(0, 255, (192, 448, 3), dtype=np.uint8)
    emb   = embedder.embed(dummy)
    print(f"Output   : shape={emb.shape}  dtype={emb.dtype}")
    lat   = embedder.benchmark(dummy, n=5)
    print(f"Latency  : {lat:.1f} ms/image (CPU)")
