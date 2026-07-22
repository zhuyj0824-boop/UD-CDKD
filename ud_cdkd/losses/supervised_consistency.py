from __future__ import annotations

import torch


def supervised_cross_dimensional_consistency(
    logits_2d: torch.Tensor,
    logits_3d: torch.Tensor,
    confidence_2d: float = 0.7,
    confidence_3d: float = 0.8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fault-focused supervised consistency term corresponding to Eqs. (6)--(8)."""
    probability_2d = torch.softmax(logits_2d, dim=1)
    probability_3d = torch.softmax(logits_3d, dim=1)

    confidence_value_2d, hard_2d = probability_2d.max(dim=1)
    confidence_value_3d, hard_3d = probability_3d.max(dim=1)
    mask = ((hard_2d == 1) & (confidence_value_2d >= confidence_2d)) | (
        (hard_3d == 1) & (confidence_value_3d >= confidence_3d)
    )
    mask = mask.to(probability_2d.dtype)
    squared_difference = (probability_2d[:, 1] - probability_3d[:, 1]).square()
    return (mask * squared_difference).sum() / (mask.sum() + eps)
