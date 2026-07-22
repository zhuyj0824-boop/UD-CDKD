# UD-CDKD: minimal core release

This repository contains only the core implementation required to understand and reproduce the main UD-CDKD experiments described in the manuscript. Development history, duplicate scripts, notebooks, visualisation utilities, logs, intermediate checkpoints, unit tests and separate ablation configuration files are intentionally excluded.

## Included files

```text
UD-CDKD_core_release/
├── train.py                 # Two-stage UD-CDKD training
├── evaluate.py              # Dice and IoU on the synthetic validation set
├── infer_field.py           # F3/Kerry-3D inference from a NumPy volume
├── config.yaml              # Main configuration; edit label ratio/strategy here
├── requirements.txt
├── LICENSE                  # Replace after checking upstream licence compatibility
├── NOTICE.md
└── ud_cdkd/
    ├── data/                # Data loading and 2-D/3-D conversion
    ├── models/              # SegFormer-LoRA and 3-D U-Net branches
    ├── losses/              # Supervised loss, warm-up consistency and DUCL
    ├── distillation/        # Stability-guided direction controller
    ├── engine/              # Training, evaluation and sliding-window inference
    └── utils/               # Metrics, checkpoints, seeds and distributed training
```

## Paper-to-code mapping

| Manuscript component | Implementation |
|---|---|
| 2-D slice extraction and 3-D aggregation | `ud_cdkd/data/conversion.py` |
| 2-D and 3-D segmentation branches | `ud_cdkd/models/` |
| Supervised segmentation and warm-up consistency | `ud_cdkd/losses/segmentation.py`, `supervised_consistency.py` |
| Stability-guided transfer-direction control | `ud_cdkd/distillation/direction_controller.py` |
| Dynamic uncertainty-guided consistency loss (DUCL) | `ud_cdkd/losses/ducl.py` |
| Two-stage optimisation | `ud_cdkd/engine/trainer.py` |

## Data layout

```text
/path/to/fault_dataset/
├── train/
│   ├── 0/seis.npy
│   ├── 0/fault.npy
│   └── ...
└── val/
    ├── 0/seis.npy
    ├── 0/fault.npy
    └── ...
```

Data sources used in the manuscript:

- Synthetic fault data: https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8
- Netherlands Offshore F3: https://wiki.seg.org/wiki/F3_Netherlands
- Kerry-3D: https://wiki.seg.org/wiki/Kerry-3D

## Configuration

Edit `data.root` in `config.yaml`. The main file uses the 25 per cent labelled-data setting. The same file can represent the other experiments by changing:

```yaml
data:
  label_ratio: 0.25      # 0.10, 0.25 or 1.00

distillation:
  strategy: stability_guided
  # no_distillation | fixed_alternation | one_way
  # simultaneous_bidirectional | stability_guided

ducl:
  uncertainty_weighting: true
  entropy_regularisation: true
  dynamic_scheduling: true
```

For the Table 3 direction-strategy ablation, disable all three DUCL switches and change `distillation.strategy`. For the Table 4 DUCL ablation, retain `stability_guided` and progressively enable the three DUCL switches.

## Installation and use

Install PyTorch for the local CUDA version, then run:

```bash
pip install -r requirements.txt
python train.py --config config.yaml
python evaluate.py --config config.yaml --checkpoint outputs/ud_cdkd_25pct/best_3d_checkpoint.pt
python infer_field.py --config config.yaml \
  --checkpoint outputs/ud_cdkd_25pct/best_3d_checkpoint.pt \
  --input F3.npy --output F3_fault_probability.npy --axis-order HWD
```

Distributed training can be launched with:

```bash
torchrun --nproc_per_node=4 train.py --config config.yaml
```

Only the trained 3-D branch is required for inference.

## Scope of this release

This minimal release does not include raw public data, old experimental scripts, individual ablation YAML files, plotting scripts, TensorBoard logs, intermediate model weights or private development notes. The single editable configuration exposes the switches needed to reproduce the principal and ablation settings without publishing the full development repository.
