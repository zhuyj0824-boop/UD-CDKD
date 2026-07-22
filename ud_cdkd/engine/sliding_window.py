from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    volume: torch.Tensor,
    roi_size: tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
) -> torch.Tensor:
    """Gaussian-blended sliding-window inference for volumes larger than a training patch."""
    model.eval()
    return sliding_window_inference(
        inputs=volume,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode="gaussian",
    )
