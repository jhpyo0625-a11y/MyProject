@echo off
echo === Coil Defect Inspection — Environment Setup ===

python -m venv .venv
call .venv\Scripts\activate.bat

echo Installing PyTorch (CPU-only)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo Installing remaining dependencies...
pip install -r requirements.txt

echo Verifying environment...
python scripts\verify_env.py

echo.
echo Done. To activate: .venv\Scripts\activate.bat
