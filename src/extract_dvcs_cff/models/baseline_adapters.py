"""
Optional baseline adapters used for comparison studies.

These adapters are not the main extraction path.
"""

from __future__ import annotations

import torch
from torch import nn


class CFFOnlyBaselineAdapter(nn.Module):
    """Directly fit CFFs from kinematics as a baseline."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 96, output_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, kinematics: torch.Tensor) -> torch.Tensor:
        return self.net(kinematics)


class ParametricGPDBaselineAdapter(nn.Module):
    """
    Simple analytic GPD baseline with learnable coefficients.

    This is intentionally compact and interpretable for closure/benchmark tests.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_amp = nn.Parameter(torch.zeros(4))
        self.alpha = nn.Parameter(torch.ones(4))
        self.beta = nn.Parameter(torch.ones(4))
        self.t_slope = nn.Parameter(torch.full((4,), 2.0))

    def forward(self, x: torch.Tensor, xi: torch.Tensor, t: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
        xi = torch.clamp(xi, 1e-6, 1.0 - 1e-6)
        q_log = torch.log(torch.clamp(q2, min=1e-6)).unsqueeze(-1)

        base = torch.exp(self.log_amp).unsqueeze(0)
        x_term = torch.pow(x.unsqueeze(-1), torch.relu(self.alpha).unsqueeze(0))
        one_minus_x = torch.pow(1.0 - x.unsqueeze(-1), torch.relu(self.beta).unsqueeze(0))
        xi_term = 1.0 + 0.2 * xi.unsqueeze(-1)
        t_term = torch.exp(torch.clamp(t.unsqueeze(-1), max=0.0) * torch.relu(self.t_slope).unsqueeze(0))
        q_term = 1.0 + 0.05 * q_log

        return base * x_term * one_minus_x * xi_term * t_term * q_term


class PureDataFitBaselineAdapter(nn.Module):
    """Pure data-driven observable regressor baseline."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, n_outputs: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
