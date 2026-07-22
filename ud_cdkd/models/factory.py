from __future__ import annotations

from .segformer_lora import SegFormerLoRABranch
from .unet3d import UNet3DBranch


def build_models(cfg):
    branch_2d = SegFormerLoRABranch(
        model_name=cfg.models.branch_2d.model_name,
        num_classes=2,
        lora_rank=int(cfg.models.branch_2d.lora_rank),
        lora_alpha=int(cfg.models.branch_2d.lora_alpha),
        lora_dropout=float(cfg.models.branch_2d.lora_dropout),
        pretrained=bool(cfg.models.branch_2d.pretrained),
    )
    branch_3d = UNet3DBranch(
        in_channels=1,
        num_classes=2,
        base_channels=int(cfg.models.branch_3d.base_channels),
    )
    return branch_2d, branch_3d
