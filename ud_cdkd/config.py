from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigNode(dict):
    """Dictionary with attribute-style access used for YAML configuration."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_node(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigNode({k: _to_node(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_node(v) for v in value]
    return value


def load_config(path: str | Path) -> ConfigNode:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    cfg = _to_node(data)
    cfg.config_path = str(path.resolve())
    validate_config(cfg)
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = deepcopy(dict(cfg))
    serialisable.pop("config_path", None)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(serialisable, f, sort_keys=False)


def validate_config(cfg: ConfigNode) -> None:
    required = ["experiment", "data", "models", "training", "distillation", "ducl"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")

    ratio = float(cfg.data.label_ratio)
    if not 0 < ratio <= 1:
        raise ValueError("data.label_ratio must be in (0, 1].")

    strategy = str(cfg.distillation.strategy)
    valid = {
        "no_distillation",
        "fixed_alternation",
        "one_way",
        "simultaneous_bidirectional",
        "stability_guided",
    }
    if strategy not in valid:
        raise ValueError(f"Unknown distillation strategy '{strategy}'. Valid choices: {sorted(valid)}")
