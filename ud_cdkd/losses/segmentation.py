from __future__ import annotations

import torch
from torch.nn import functional as F


def fault_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss for the fault class (class index 1)."""
    probability = torch.softmax(logits, dim=1)[:, 1]
    target_fault = (target > 0).to(probability.dtype)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target_fault).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target_fault.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cross_entropy = F.cross_entropy(logits, target.long())
    dice = fault_dice_loss(logits, target)
    total = cross_entropy + dice
    return total, {"cross_entropy": cross_entropy, "dice": dice}
