@echo off
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
python src\app\main.py %*
