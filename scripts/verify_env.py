"""Run this after install to confirm all dependencies are importable and sane."""
import sys
import importlib

REQUIRED = [
    ("torch",         "torch"),
    ("torchvision",   "torchvision"),
    ("timm",          "timm"),
    ("sklearn",       "scikit-learn"),
    ("cv2",           "opencv-python"),
    ("PIL",           "Pillow"),
    ("numpy",         "numpy"),
    ("pandas",        "pandas"),
    ("mlflow",        "mlflow"),
    ("yaml",          "PyYAML"),
    ("PyQt5",         "PyQt5"),
]

MIN_PYTHON = (3, 11)
WARN_PYTHON = (3, 13)  # PyTorch wheel availability uncertain above 3.12

print(f"Python {sys.version}")

if sys.version_info < MIN_PYTHON:
    print(f"  ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required.")
    sys.exit(1)
if sys.version_info >= WARN_PYTHON:
    print(f"  WARNING: Python {sys.version_info.major}.{sys.version_info.minor} "
          "may not have official PyTorch CPU wheels yet.\n"
          "  If 'import torch' fails, install from nightly:\n"
          "    pip install --pre torch torchvision "
          "--index-url https://download.pytorch.org/whl/nightly/cpu")

ok = True
for import_name, pip_name in REQUIRED:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  OK  {pip_name:<20} {ver}")
    except ImportError:
        print(f"  MISSING  {pip_name}  →  pip install {pip_name}")
        ok = False

if ok:
    # Quick torch sanity check
    import torch
    t = torch.zeros(3)
    print(f"\nTorch tensor smoke test: {t}  device={t.device}")
    print("\nAll dependencies OK.")
else:
    print("\nSome packages are missing. Run scripts/install.bat to install them.")
    sys.exit(1)
