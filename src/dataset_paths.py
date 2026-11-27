"""Locate the MVTec AD data directory without loading ML dependencies."""

from pathlib import Path
from typing import Union

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "dataset"


def resolve_data_root(data_root: Union[str, Path]) -> Path:
    """Accept the dataset itself or a checkout containing its dataset directory."""
    root = Path(data_root).resolve()
    nested = root / "dataset"
    if (nested / "train_split.csv").is_file():
        return nested
    return root
