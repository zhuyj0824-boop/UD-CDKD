#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ud_cdkd.config import load_config
from ud_cdkd.data.datasets import load_volume
from ud_cdkd.engine.sliding_window import sliding_window_predict
from ud_cdkd.models.factory import build_models
from ud_cdkd.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the trained 3-D branch to an F3 or Kerry-3D NumPy volume.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input seismic volume in .npy format.")
    parser.add_argument("--output", required=True, help="Output fault-probability .npy file.")
    parser.add_argument("--axis-order", default="HWD", choices=["HWD", "DHW", "WHD"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model_3d = build_models(cfg)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model_3d.load_state_dict(checkpoint["model_3d"])
    model_3d.to(device).eval()

    volume = load_volume(args.input, args.axis_order).to(device)
    logits = sliding_window_predict(
        model_3d,
        volume,
        roi_size=tuple(int(v) for v in cfg.data.patch_size),
        overlap=float(cfg.inference.overlap),
        sw_batch_size=int(cfg.inference.sliding_window_batch_size),
    )
    probability = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, probability)


if __name__ == "__main__":
    main()
