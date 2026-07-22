#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ud_cdkd.config import load_config, save_config
from ud_cdkd.engine.trainer import UDCDKDTrainer
from ud_cdkd.utils.distributed import cleanup_distributed, initialise_distributed
from ud_cdkd.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UD-CDKD.")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint from which to resume.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    distributed = initialise_distributed()
    seed_everything(int(cfg.experiment.seed) + distributed.rank, bool(cfg.experiment.deterministic))

    output_dir = Path(cfg.experiment.output_dir)
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, output_dir / "resolved_config.yaml")

    trainer = UDCDKDTrainer(cfg, distributed)
    if args.resume:
        trainer.resume(args.resume)
    try:
        trainer.train()
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
