#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from ud_cdkd.config import load_config
from ud_cdkd.data.datasets import build_synthetic_datasets
from ud_cdkd.engine.evaluator import evaluate_3d
from ud_cdkd.models.factory import build_models
from ud_cdkd.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained 3-D branch on the synthetic validation set.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, validation = build_synthetic_datasets(cfg, write_manifest=False)
    loader = DataLoader(
        validation,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
    )
    _, model_3d = build_models(cfg)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model_3d.load_state_dict(checkpoint["model_3d"])
    model_3d.to(device)
    print(json.dumps(evaluate_3d(model_3d, loader, device), indent=2))


if __name__ == "__main__":
    main()
