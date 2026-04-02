"""Composite weighted loss with schedule and adaptive balancing."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from extract_dvcs_cff.utils.config import LossConfig, LossWeightsConfig


WEIGHT_TO_TERM: dict[str, str] = {
    "w_data": "L_DVCS",
    "w_CFF": "L_transform",
    "w_fwd": "L_forward",
    "w_sum": "L_sumrule",
    "w_poly": "L_polynomiality",
    "w_pos": "L_positivity",
    "w_evol": "L_evolution",
    "w_smooth": "L_smooth",
    "w_reg": "L_regularization",
}


@dataclass
class AdaptiveLossBalancer:
    """Simple inverse-EMA magnitude balancer for multi-term losses."""

    beta: float
    eps: float
    ema: dict[str, float] = field(default_factory=dict)

    def scaled_weight(self, term_name: str, base_weight: float, value: torch.Tensor) -> float:
        magnitude = float(torch.detach(torch.abs(value)).cpu().item())
        previous = self.ema.get(term_name, magnitude)
        current = self.beta * previous + (1.0 - self.beta) * magnitude
        self.ema[term_name] = current
        return base_weight / max(current, self.eps)


class CompositeLoss(nn.Module):
    """Scheduled composite objective for physics-informed training."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.adaptive = (
            AdaptiveLossBalancer(beta=config.adaptive_beta, eps=config.adaptive_eps)
            if config.adaptive_weighting
            else None
        )

    @staticmethod
    def _weights_to_dict(weights: LossWeightsConfig) -> dict[str, float]:
        return {
            "w_data": weights.w_data,
            "w_CFF": weights.w_CFF,
            "w_fwd": weights.w_fwd,
            "w_sum": weights.w_sum,
            "w_poly": weights.w_poly,
            "w_pos": weights.w_pos,
            "w_evol": weights.w_evol,
            "w_smooth": weights.w_smooth,
            "w_reg": weights.w_reg,
        }

    def weights_for_epoch(self, epoch: int) -> dict[str, float]:
        """Compute scheduled scalar weights for a given epoch."""
        base = self._weights_to_dict(self.config.base_weights)

        phase_weights = None
        for phase in self.config.phases:
            if phase.start_epoch <= epoch <= phase.end_epoch:
                phase_weights = self._weights_to_dict(phase.multipliers)
                break

        if phase_weights is None:
            return base

        return {name: base[name] * phase_weights[name] for name in base}

    def forward(
        self,
        loss_terms: dict[str, torch.Tensor],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        """
        Combine terms into total loss.

        Returns
        -------
        total_loss, weighted_terms, effective_weights
        """
        scheduled = self.weights_for_epoch(epoch)

        weighted_terms: dict[str, torch.Tensor] = {}
        effective_weights: dict[str, float] = {}

        total = torch.zeros((), device=next(iter(loss_terms.values())).device)
        for weight_name, term_name in WEIGHT_TO_TERM.items():
            value = loss_terms.get(term_name)
            if value is None:
                continue

            base_weight = scheduled[weight_name]
            if self.adaptive is not None:
                eff_weight = self.adaptive.scaled_weight(term_name, base_weight, value)
            else:
                eff_weight = base_weight

            weighted = float(eff_weight) * value
            weighted_terms[term_name] = weighted
            effective_weights[term_name] = float(eff_weight)
            total = total + weighted

        return total, weighted_terms, effective_weights
