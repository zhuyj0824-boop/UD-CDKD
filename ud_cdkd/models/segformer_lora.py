from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormerLoRABranch(nn.Module):
    """Pre-trained 2-D SegFormer branch adapted with LoRA for binary fault segmentation."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        lora_rank: int = 64,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if not pretrained:
            raise ValueError(
                "The paper configuration uses an ADE20K-pre-trained SegFormer. "
                "Set models.branch_2d.pretrained=true."
            )

        base = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        for parameter in base.parameters():
            parameter.requires_grad = False

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "dense"],
            modules_to_save=["decode_head.classifier"],
            bias="none",
        )
        self.model = get_peft_model(base, lora_config)

        # The complete decoder remains trainable, as specified in the manuscript.
        for parameter in self.model.base_model.model.decode_head.parameters():
            parameter.requires_grad = True

    def forward(self, sections: torch.Tensor) -> torch.Tensor:
        if sections.ndim != 4:
            raise ValueError(f"Expected 2-D sections with shape (N,C,H,W), got {tuple(sections.shape)}")
        if sections.shape[1] == 1:
            sections = sections.repeat(1, 3, 1, 1)
        elif sections.shape[1] != 3:
            raise ValueError("SegFormer expects one-channel seismic sections or three-channel inputs.")

        spatial_size = sections.shape[-2:]
        logits = self.model(pixel_values=sections).logits
        return F.interpolate(logits, size=spatial_size, mode="bilinear", align_corners=False)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
