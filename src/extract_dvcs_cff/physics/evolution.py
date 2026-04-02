"""
Differentiable Q2 evolution layers for GPD channels.

This module supports two modes:
- differentiable: analytic multiplicative scaling with configurable channel exponents
- surrogate: learnable residual correction over the differentiable baseline
"""

from __future__ import annotations

import torch
from torch import nn

from extract_dvcs_cff.utils.config import EvolutionConfig


class Q2EvolutionLayer(nn.Module):
    """
    Channel-wise Q2 evolution operator.

    Parameters
    ----------
    config:
        Evolution settings including mode and reference scale.

    Inputs
    ------
    gpd:
        Tensor of shape [..., 4] ordered [H, E, Htilde, Etilde].
    q2_target:
        Tensor of shape [...] with target Q2 values.
    q2_source:
        Optional tensor of shape [...] with source Q2 values.
        If omitted, config.reference_q2 is used.
    """

    def __init__(self, config: EvolutionConfig) -> None:
        super().__init__()
        self.config = config

        gamma = torch.tensor(config.channel_anomalous_dims, dtype=torch.float32)
        self.log_gamma = nn.Parameter(torch.log(torch.clamp(gamma, min=1e-6)))

        self.surrogate = None
        if config.mode == "surrogate":
            hidden = config.surrogate_hidden_dim
            layers: list[nn.Module] = [nn.Linear(5, hidden), nn.GELU()]
            for _ in range(config.surrogate_depth - 1):
                layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
            layers.append(nn.Linear(hidden, 4))
            self.surrogate = nn.Sequential(*layers)

    def _log_q2_ratio(self, q2_target: torch.Tensor, q2_source: torch.Tensor) -> torch.Tensor:
        ratio = torch.log(torch.clamp(q2_target, min=1e-8)) - torch.log(torch.clamp(q2_source, min=1e-8))
        return torch.clamp(ratio, -self.config.clamp_log_q2_ratio, self.config.clamp_log_q2_ratio)

    def _differentiable_evolution(
        self,
        gpd: torch.Tensor,
        q2_target: torch.Tensor,
        q2_source: torch.Tensor,
    ) -> torch.Tensor:
        log_ratio = self._log_q2_ratio(q2_target, q2_source).unsqueeze(-1)
        gamma = torch.exp(self.log_gamma).to(device=gpd.device, dtype=gpd.dtype)
        scale = torch.exp(-gamma * log_ratio)
        return gpd * scale

    def evolve(
        self,
        gpd: torch.Tensor,
        q2_target: torch.Tensor,
        q2_source: torch.Tensor,
    ) -> torch.Tensor:
        """Evolve GPDs from q2_source to q2_target."""
        if not self.config.enabled:
            return gpd

        baseline = self._differentiable_evolution(gpd, q2_target, q2_source)
        if self.surrogate is None:
            return baseline

        log_ratio = self._log_q2_ratio(q2_target, q2_source).unsqueeze(-1)
        surrogate_in = torch.cat([gpd, log_ratio], dim=-1)
        correction = self.surrogate(surrogate_in)
        return baseline + correction

    def forward(self, gpd: torch.Tensor, q2_target: torch.Tensor, q2_source: torch.Tensor | None = None) -> torch.Tensor:
        if q2_source is None:
            q2_source = torch.full_like(q2_target, float(self.config.reference_q2))
        return self.evolve(gpd, q2_target=q2_target, q2_source=q2_source)

    def backward_consistency_penalty(self, gpd: torch.Tensor, q2_target: torch.Tensor) -> torch.Tensor:
        """
        Penalize mismatch after forward evolution then backward evolution.

        This is a soft consistency regularizer, not an exact inverse guarantee.
        """
        q2_ref = torch.full_like(q2_target, float(self.config.reference_q2))
        forward = self.evolve(gpd, q2_target=q2_target, q2_source=q2_ref)
        backward = self.evolve(forward, q2_target=q2_ref, q2_source=q2_target)
        return torch.mean((backward - gpd) ** 2)
