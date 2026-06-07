"""Shared helper: load config.yaml from the project root."""
import yaml
from pathlib import Path


def get_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = get_root() / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
