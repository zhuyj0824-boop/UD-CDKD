from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DUCLComponents:
    total: torch.Tensor
    pseudo_label: torch.Tensor
    entropy_regularisation: torch.Tensor
    ramp_weight: float
    mean_reliability: torch.Tensor


def normalised_binary_entropy(probability: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = probability.clamp(eps, 1.0 - eps)
    entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )
    return entropy / torch.log(torch.tensor(2.0, device=probability.device, dtype=probability.dtype))


class DynamicUncertaintyGuidedConsistencyLoss(nn.Module):
    """Dynamic uncertainty-guided consistency loss (DUCL), Eqs. (18)--(26)."""

    def __init__(
        self,
        eta: float = 5.0,
        tau: float = 0.5,
        beta0: float = 0.1,
        delta0: float = 0.5,
        uncertainty_weighting: bool = True,
        entropy_regularisation: bool = True,
        dynamic_scheduling: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eta = float(eta)
        self.tau = float(tau)
        self.beta0 = float(beta0)
        self.delta0 = float(delta0)
        self.uncertainty_weighting = bool(uncertainty_weighting)
        self.entropy_regularisation = bool(entropy_regularisation)
        self.dynamic_scheduling = bool(dynamic_scheduling)
        self.eps = float(eps)

    def ramp_weight(self, epoch: int, total_epochs: int) -> float:
        if not self.dynamic_scheduling:
            return 1.0
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive.")
        t = min(max(int(epoch), 1), int(total_epochs))
        progress = t / float(total_epochs)
        return float(torch.exp(torch.tensor(-self.eta * (1.0 - progress) ** 2)).item())

    def forward(
        self,
        updated_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        epoch: int,
        total_epochs: int,
        direction_scale: float = 1.0,
    ) -> DUCLComponents:
        if updated_logits.shape != reference_logits.shape:
            raise ValueError(
                "Updated and reference logits must have identical shapes, got "
                f"{tuple(updated_logits.shape)} and {tuple(reference_logits.shape)}."
            )

        updated_probability = torch.softmax(updated_logits, dim=1)[:, 1]
        reference_distribution = torch.softmax(reference_logits.detach(), dim=1)
        reference_probability = reference_distribution[:, 1]
        pseudo_label = reference_distribution.argmax(dim=1).to(updated_probability.dtype)

        updated_entropy = normalised_binary_entropy(updated_probability, self.eps)
        reference_entropy = normalised_binary_entropy(reference_probability, self.eps)

        lam = self.ramp_weight(epoch, total_epochs)
        beta = self.beta0 * lam if self.entropy_regularisation else 0.0
        delta = self.delta0 * lam if self.uncertainty_weighting else 0.0

        if self.uncertainty_weighting:
            reliability = torch.exp(-reference_entropy / self.tau)
            reliability = reliability * (
                1.0 + delta * torch.relu(updated_entropy - reference_entropy)
            )
        else:
            reliability = torch.ones_like(reference_entropy)

        weighted_intersection = (reliability * updated_probability * pseudo_label).sum()
        weighted_prediction = (reliability * updated_probability).sum()
        weighted_target = (reliability * pseudo_label).sum()
        pseudo_label_loss = 1.0 - (
            2.0 * weighted_intersection + self.eps
        ) / (weighted_prediction + weighted_target + self.eps)

        if self.entropy_regularisation:
            entropy_loss = (reliability * updated_entropy).sum() / (reliability.sum() + self.eps)
        else:
            entropy_loss = torch.zeros((), device=updated_logits.device, dtype=updated_logits.dtype)

        total = float(direction_scale) * (lam * pseudo_label_loss + beta * entropy_loss)
        return DUCLComponents(
            total=total,
            pseudo_label=pseudo_label_loss,
            entropy_regularisation=entropy_loss,
            ramp_weight=lam,
            mean_reliability=reliability.mean(),
        )
