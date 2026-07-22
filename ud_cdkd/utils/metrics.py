from __future__ import annotations

import torch


def _as_binary_prediction(prediction: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if prediction.ndim >= 2 and prediction.shape[1] == 2:
        prediction = torch.softmax(prediction, dim=1)[:, 1]
    elif prediction.ndim >= 2 and prediction.shape[1] == 1:
        prediction = torch.sigmoid(prediction[:, 0])
    return (prediction >= threshold).to(torch.float32)


def _as_binary_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim >= 2 and target.shape[1] == 2:
        target = target[:, 1]
    elif target.ndim >= 2 and target.shape[1] == 1:
        target = target[:, 0]
    return (target > 0).to(torch.float32)


def dice_score(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred = _as_binary_prediction(prediction, threshold).reshape(prediction.shape[0], -1)
    tgt = _as_binary_target(target).reshape(target.shape[0], -1)
    intersection = (pred * tgt).sum(dim=1)
    return ((2 * intersection + eps) / (pred.sum(dim=1) + tgt.sum(dim=1) + eps)).mean()


def iou_score(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred = _as_binary_prediction(prediction, threshold).reshape(prediction.shape[0], -1)
    tgt = _as_binary_target(target).reshape(target.shape[0], -1)
    intersection = (pred * tgt).sum(dim=1)
    union = pred.sum(dim=1) + tgt.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()
