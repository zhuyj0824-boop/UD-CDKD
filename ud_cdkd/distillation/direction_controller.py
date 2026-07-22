from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ud_cdkd.utils.distributed import DistributedContext, all_reduce_sum


@dataclass(frozen=True)
class DirectionDecision:
    stability: float
    normalised_improvement: float
    reverse_activation: float
    reference_branch: str
    updated_branch: str


@dataclass(frozen=True)
class TransferInstruction:
    reference_branch: str
    updated_branch: str
    scale: float


class StabilityGuidedDirectionController:
    """Fault-aware temporal stability controller corresponding to Eqs. (10)--(17)."""

    def __init__(
        self,
        prediction_momentum: float = 0.9,
        statistics_momentum: float = 0.9,
        eps: float = 1e-6,
        store_dtype: torch.dtype = torch.float16,
    ) -> None:
        if not 0 <= prediction_momentum < 1:
            raise ValueError("prediction_momentum must be in [0, 1).")
        if not 0 <= statistics_momentum < 1:
            raise ValueError("statistics_momentum must be in [0, 1).")
        self.prediction_momentum = float(prediction_momentum)
        self.statistics_momentum = float(statistics_momentum)
        self.eps = float(eps)
        self.store_dtype = store_dtype
        self.prediction_ema: dict[str, torch.Tensor] = {}
        self.stability_mean: float | None = None
        self.stability_deviation: float | None = None
        self._numerator = 0.0
        self._denominator = 0.0
        self._pending_ema: dict[str, torch.Tensor] = {}

    def begin_epoch(self) -> None:
        self._numerator = 0.0
        self._denominator = 0.0
        self._pending_ema = {}

    @torch.no_grad()
    def observe(self, sample_ids: Iterable[str], fault_probabilities: torch.Tensor) -> None:
        if fault_probabilities.ndim != 4:
            raise ValueError(
                "fault_probabilities must have shape (B,H,W,D), got "
                f"{tuple(fault_probabilities.shape)}"
            )
        ids = list(sample_ids)
        if len(ids) != fault_probabilities.shape[0]:
            raise ValueError("Number of sample identifiers does not match the batch size.")

        for sample_id, current in zip(ids, fault_probabilities.detach()):
            current = current.float()
            previous_cpu = self.prediction_ema.get(str(sample_id))
            if previous_cpu is None:
                updated = current
                # The first observation initialises the EMA; z is forced to zero at epoch end.
                self._denominator += float(current.sum().item())
            else:
                previous = previous_cpu.to(device=current.device, dtype=current.dtype)
                weight = torch.maximum(current, previous)
                self._numerator += float((weight * (current - previous).abs()).sum().item())
                self._denominator += float(weight.sum().item())
                updated = self.prediction_momentum * previous + (1.0 - self.prediction_momentum) * current
            self._pending_ema[str(sample_id)] = updated.to("cpu", dtype=self.store_dtype)

    def end_epoch(self, distributed: DistributedContext | None = None) -> DirectionDecision:
        device = distributed.device if distributed is not None else torch.device("cpu")
        totals = torch.tensor([self._numerator, self._denominator], dtype=torch.float64, device=device)
        if distributed is not None:
            totals = all_reduce_sum(totals, distributed)
        numerator, denominator = (float(v) for v in totals.cpu().tolist())
        stability = 1.0 - numerator / (denominator + self.eps)
        stability = max(0.0, min(1.0, stability))

        first_epoch = self.stability_mean is None or self.stability_deviation is None
        if first_epoch:
            z_score = 0.0
            self.stability_mean = stability
            self.stability_deviation = self.eps
        else:
            z_score = (stability - self.stability_mean) / (self.stability_deviation + self.eps)
            new_mean = (
                self.statistics_momentum * self.stability_mean
                + (1.0 - self.statistics_momentum) * stability
            )
            new_deviation = (
                self.statistics_momentum * self.stability_deviation
                + (1.0 - self.statistics_momentum) * abs(stability - new_mean)
            )
            self.stability_mean = new_mean
            self.stability_deviation = max(new_deviation, self.eps)

        positive = max(0.0, z_score)
        reverse_activation = positive / (1.0 + positive)
        if reverse_activation > 0:
            reference, updated = "3d", "2d"
        else:
            reference, updated = "2d", "3d"

        self.prediction_ema.update(self._pending_ema)
        self._pending_ema = {}
        return DirectionDecision(
            stability=stability,
            normalised_improvement=z_score,
            reverse_activation=reverse_activation,
            reference_branch=reference,
            updated_branch=updated,
        )

    def state_dict(self, include_prediction_ema: bool = True) -> dict:
        state = {
            "prediction_momentum": self.prediction_momentum,
            "statistics_momentum": self.statistics_momentum,
            "eps": self.eps,
            "stability_mean": self.stability_mean,
            "stability_deviation": self.stability_deviation,
        }
        if include_prediction_ema:
            state["prediction_ema"] = self.prediction_ema
        return state

    def load_state_dict(self, state: dict) -> None:
        self.stability_mean = state.get("stability_mean")
        self.stability_deviation = state.get("stability_deviation")
        self.prediction_ema = state.get("prediction_ema", {})


def select_transfer_instructions(
    strategy: str,
    distillation_epoch: int,
    decision: DirectionDecision | None = None,
) -> list[TransferInstruction]:
    """Return the active transfer direction(s) for the Table 3 variants."""
    if strategy == "no_distillation":
        return []
    if strategy == "one_way":
        return [TransferInstruction("2d", "3d", 1.0)]
    if strategy == "fixed_alternation":
        if distillation_epoch % 2 == 1:
            return [TransferInstruction("2d", "3d", 1.0)]
        return [TransferInstruction("3d", "2d", 1.0)]
    if strategy == "simultaneous_bidirectional":
        return [
            TransferInstruction("2d", "3d", 1.0),
            TransferInstruction("3d", "2d", 1.0),
        ]
    if strategy == "stability_guided":
        if decision is None:
            raise ValueError("A stability decision is required for the stability_guided strategy.")
        scale = decision.reverse_activation if decision.reference_branch == "3d" else 1.0
        return [TransferInstruction(decision.reference_branch, decision.updated_branch, scale)]
    raise ValueError(f"Unknown distillation strategy: {strategy}")
