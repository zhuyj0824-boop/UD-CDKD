from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SliceLayout:
    """Metadata required to restore slice-wise predictions to a 3-D volume."""

    batch_size: int
    channels: int
    spatial_shape: tuple[int, int, int]
    slice_axis: int


def volume_to_slices(volume: torch.Tensor, slice_axis: int = -1) -> tuple[torch.Tensor, SliceLayout]:
    """
    Convert a 5-D tensor from ``(B, C, H, W, D)`` to a sequence of 2-D sections.

    Returns
    -------
    slices
        Tensor with shape ``(B*S, C, A, B)``, where ``S`` is the number of
        sections along ``slice_axis``.
    layout
        Metadata used by :func:`slices_to_volume`.
    """
    if volume.ndim != 5:
        raise ValueError(f"Expected volume shape (B,C,H,W,D), got {tuple(volume.shape)}")

    axis = slice_axis if slice_axis >= 0 else volume.ndim + slice_axis
    if axis not in (2, 3, 4):
        raise ValueError("slice_axis must refer to one of the three spatial dimensions (2, 3 or 4).")

    b, c, h, w, d = volume.shape
    moved = torch.movedim(volume, axis, 1)  # (B, S, C, A, B)
    sections = moved.contiguous().reshape(-1, c, moved.shape[-2], moved.shape[-1])
    return sections, SliceLayout(b, c, (h, w, d), axis)


def slices_to_volume(slices: torch.Tensor, layout: SliceLayout) -> torch.Tensor:
    """Restore slice-wise logits to ``(B, C, H, W, D)``."""
    if slices.ndim != 4:
        raise ValueError(f"Expected slice tensor shape (B*S,C,H,W), got {tuple(slices.shape)}")

    num_slices = layout.spatial_shape[layout.slice_axis - 2]
    expected = layout.batch_size * num_slices
    if slices.shape[0] != expected:
        raise ValueError(f"Expected {expected} slices, received {slices.shape[0]}.")

    restored = slices.reshape(layout.batch_size, num_slices, slices.shape[1], slices.shape[2], slices.shape[3])
    restored = torch.movedim(restored, 1, layout.slice_axis)
    if restored.shape[2:] != layout.spatial_shape:
        raise ValueError(
            "Restored spatial shape does not match the source volume: "
            f"{tuple(restored.shape[2:])} vs {layout.spatial_shape}."
        )
    return restored
