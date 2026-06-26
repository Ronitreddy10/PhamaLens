"""Stable project paths, independent of the process working directory."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT / "config"
DATA_DIR = PACKAGE_ROOT / "data"


def resolve_data_path(value: str) -> Path:
    """Resolve config data paths against the package root."""
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return PACKAGE_ROOT / path
    return PACKAGE_ROOT / path
