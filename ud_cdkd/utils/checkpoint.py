from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, **state: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dictionary: {path}")
    return checkpoint
