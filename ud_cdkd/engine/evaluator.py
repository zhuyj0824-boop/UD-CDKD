from __future__ import annotations

import torch

from ud_cdkd.utils.distributed import DistributedContext, all_reduce_sum
from ud_cdkd.utils.metrics import dice_score, iou_score


@torch.no_grad()
def evaluate_3d(model: torch.nn.Module, loader, device: torch.device, distributed: DistributedContext | None = None) -> dict[str, float]:
    model.eval()
    dice_total = torch.zeros((), dtype=torch.float64, device=device)
    iou_total = torch.zeros((), dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        logits = model(image)
        batch_size = image.shape[0]
        dice_total += dice_score(logits, target).double() * batch_size
        iou_total += iou_score(logits, target).double() * batch_size
        count += batch_size

    totals = torch.stack([dice_total, iou_total, count])
    if distributed is not None:
        totals = all_reduce_sum(totals, distributed)
    denominator = max(float(totals[2].item()), 1.0)
    return {
        "dice": float(totals[0].item() / denominator),
        "iou": float(totals[1].item() / denominator),
    }
