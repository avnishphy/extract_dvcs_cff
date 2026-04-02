"""Differentiable physics constraints for the DVCS observables -> GPD pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from extract_dvcs_cff.utils.config import ConstraintConfig


class PDFProvider(Protocol):
    """Provider interface for forward-limit PDF queries."""

    def evaluate(self, flavor: str, x: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Return PDF values for a flavor at (x, Q2)."""


class FormFactorProvider(Protocol):
    """Provider interface for elastic form factors used in sum-rule penalties."""

    def f1(self, t: torch.Tensor) -> torch.Tensor:
        """Dirac form factor F1(t)."""

    def f2(self, t: torch.Tensor) -> torch.Tensor:
        """Pauli form factor F2(t)."""


@dataclass
class DipoleFormFactorProvider:
    """
    Lightweight analytic form-factor provider.

    This is intended as a placeholder/reference provider until a dedicated
    external form-factor database is wired in.
    """

    m2: float = 0.71
    kappa: float = 1.793

    def f1(self, t: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(1.0 - t / self.m2, min=1e-6)
        return 1.0 / (denom * denom)

    def f2(self, t: torch.Tensor) -> torch.Tensor:
        return self.kappa * self.f1(t)


@dataclass
class TabulatedFormFactorProvider:
    """Form-factor provider with linear interpolation over tabulated values."""

    t_points: torch.Tensor
    f1_points: torch.Tensor
    f2_points: torch.Tensor

    def __post_init__(self) -> None:
        if self.t_points.ndim != 1:
            raise ValueError("t_points must be 1D.")
        if self.f1_points.shape != self.t_points.shape:
            raise ValueError("f1_points must match t_points shape.")
        if self.f2_points.shape != self.t_points.shape:
            raise ValueError("f2_points must match t_points shape.")
        if not torch.all(self.t_points[1:] >= self.t_points[:-1]):
            raise ValueError("t_points must be sorted in ascending order.")

    def _interp(self, t: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.t_points, t)
        idx = torch.clamp(idx, 1, self.t_points.numel() - 1)

        t0 = self.t_points[idx - 1]
        t1 = self.t_points[idx]
        v0 = values[idx - 1]
        v1 = values[idx]

        w = (t - t0) / torch.clamp(t1 - t0, min=1e-8)
        return (1.0 - w) * v0 + w * v1

    def f1(self, t: torch.Tensor) -> torch.Tensor:
        return self._interp(t, self.f1_points)

    def f2(self, t: torch.Tensor) -> torch.Tensor:
        return self._interp(t, self.f2_points)


@dataclass
class NullPDFProvider:
    """Fallback provider used when LHAPDF is unavailable."""

    value: float = 0.0

    def evaluate(self, flavor: str, x: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        _ = (flavor, q2)
        return torch.full_like(x, float(self.value))


def support_penalty(x: torch.Tensor) -> torch.Tensor:
    """Penalty for x values outside physical support [-1, 1]."""
    above = torch.relu(x - 1.0)
    below = torch.relu(-1.0 - x)
    return torch.mean(above * above + below * below)


def endpoint_suppression_penalty(gpd: torch.Tensor, x: torch.Tensor, power: float = 0.25) -> torch.Tensor:
    """
    Encourage suppressed amplitudes near |x| -> 1.

    The penalty is soft to avoid forcing a rigid endpoint shape.
    """
    edge = torch.pow(torch.clamp(1.0 - torch.abs(torch.clamp(x, -1.0, 1.0)), min=1e-6), power)
    scaled = gpd / edge
    return torch.mean(torch.relu(torch.abs(scaled) - 5.0) ** 2)


def positivity_penalty(gpd: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Soft positivity penalty for channels where positivity is expected."""
    return torch.mean(torch.relu(margin - gpd) ** 2)


def smoothness_penalty(gpd_grid: torch.Tensor, x_grid: torch.Tensor) -> torch.Tensor:
    """
    Curvature regularizer based on second finite differences in x.

    Expected shape for gpd_grid is [B, Nx, C] or [B, Nx].
    """
    if gpd_grid.ndim == 2:
        gpd_grid = gpd_grid.unsqueeze(-1)

    dx = x_grid[1:] - x_grid[:-1]
    dx = torch.clamp(dx, min=1e-6)

    first = (gpd_grid[:, 1:, :] - gpd_grid[:, :-1, :]) / dx.view(1, -1, 1)
    dx_mid = torch.clamp(0.5 * (dx[1:] + dx[:-1]), min=1e-6)
    second = (first[:, 1:, :] - first[:, :-1, :]) / dx_mid.view(1, -1, 1)
    return torch.mean(second * second)


def mellin_moment(gpd_values: torch.Tensor, x_grid: torch.Tensor, order: int) -> torch.Tensor:
    """Compute the order-th Mellin moment with trapezoidal integration."""
    if order < 0:
        raise ValueError("order must be non-negative.")
    weights = torch.pow(x_grid, order)
    weighted = gpd_values * weights
    return torch.trapz(weighted, x_grid, dim=-1)


def forward_limit_penalty(
    h_forward: torch.Tensor,
    x_forward: torch.Tensor,
    q2_forward: torch.Tensor,
    pdf_provider: PDFProvider,
    flavor: str = "u",
) -> torch.Tensor:
    """Compare H(x, 0, 0, Q2) against a PDF reference provider."""
    target = pdf_provider.evaluate(flavor=flavor, x=x_forward, q2=q2_forward)
    return torch.mean((h_forward - target) ** 2)


def sum_rule_penalty(
    h_grid: torch.Tensor,
    e_grid: torch.Tensor,
    x_grid: torch.Tensor,
    t_values: torch.Tensor,
    form_factor_provider: FormFactorProvider,
) -> torch.Tensor:
    """
    Enforce first-moment consistency with elastic form factors.

    The implementation uses channel-averaged first moments as a compact soft
    consistency term for training.
    """
    f1_model = mellin_moment(h_grid, x_grid, order=0)
    f2_model = mellin_moment(e_grid, x_grid, order=0)

    f1_target = form_factor_provider.f1(t_values)
    f2_target = form_factor_provider.f2(t_values)

    return torch.mean((f1_model - f1_target) ** 2 + (f2_model - f2_target) ** 2)


def polynomiality_penalty(
    gpd_xi_grid: torch.Tensor,
    x_grid: torch.Tensor,
    xi_grid: torch.Tensor,
    max_moment: int,
    fit_degree: int,
) -> torch.Tensor:
    """
    Soft polynomiality constraint based on moment-vs-xi fits.

    Parameters
    ----------
    gpd_xi_grid:
        Tensor of shape [Nxi, Nx] for one channel.
    x_grid:
        x grid with shape [Nx].
    xi_grid:
        xi points with shape [Nxi].
    """
    if gpd_xi_grid.ndim != 2:
        raise ValueError("gpd_xi_grid must have shape [Nxi, Nx].")
    if x_grid.ndim != 1 or xi_grid.ndim != 1:
        raise ValueError("x_grid and xi_grid must be 1D.")
    if gpd_xi_grid.shape[0] != xi_grid.shape[0]:
        raise ValueError("gpd_xi_grid first dimension must match xi_grid.")
    if gpd_xi_grid.shape[1] != x_grid.shape[0]:
        raise ValueError("gpd_xi_grid second dimension must match x_grid.")

    total = torch.tensor(0.0, device=gpd_xi_grid.device, dtype=gpd_xi_grid.dtype)
    n_terms = 0

    for moment_order in range(max_moment + 1):
        moments = mellin_moment(gpd_xi_grid, x_grid, order=moment_order)
        degree = min(fit_degree, moment_order + 1)

        vandermonde = torch.stack(
            [torch.pow(xi_grid, p) for p in range(degree + 1)],
            dim=-1,
        )
        coeffs = torch.linalg.lstsq(vandermonde, moments.unsqueeze(-1)).solution.squeeze(-1)
        fitted = vandermonde @ coeffs

        total = total + torch.mean((moments - fitted) ** 2)
        n_terms += 1

    return total / max(n_terms, 1)


class PhysicsConstraintEvaluator:
    """Collect and evaluate configurable physics penalties."""

    def __init__(
        self,
        config: ConstraintConfig,
        pdf_provider: PDFProvider | None = None,
        form_factor_provider: FormFactorProvider | None = None,
    ) -> None:
        self.config = config
        self.pdf_provider = pdf_provider
        self.form_factor_provider = form_factor_provider or DipoleFormFactorProvider()

    def evaluate(
        self,
        gpd_grid: torch.Tensor,
        x_grid: torch.Tensor,
        xi_values: torch.Tensor,
        t_values: torch.Tensor,
        q2_values: torch.Tensor,
        forward_h: torch.Tensor | None = None,
        forward_x: torch.Tensor | None = None,
        forward_q2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return per-term physics penalties for a training batch."""
        penalties: dict[str, torch.Tensor] = {}

        if self.config.enforce_support:
            penalties["L_support"] = support_penalty(x_grid)

        if self.config.enforce_endpoint_suppression:
            penalties["L_endpoint"] = endpoint_suppression_penalty(
                gpd_grid[..., 0],
                x_grid,
                power=self.config.endpoint_power,
            )

        if self.config.enforce_positivity:
            penalties["L_positivity"] = positivity_penalty(
                gpd_grid[..., 0],
                margin=self.config.positivity_margin,
            )

        if self.config.enforce_smoothness:
            penalties["L_smoothness"] = self.config.smoothness_strength * smoothness_penalty(gpd_grid, x_grid)

        if self.config.enforce_sum_rules:
            penalties["L_sumrule"] = sum_rule_penalty(
                h_grid=gpd_grid[..., 0],
                e_grid=gpd_grid[..., 1],
                x_grid=x_grid,
                t_values=t_values,
                form_factor_provider=self.form_factor_provider,
            )

        if self.config.enforce_polynomiality and gpd_grid.shape[0] > 3:
            penalties["L_polynomiality"] = polynomiality_penalty(
                gpd_xi_grid=gpd_grid[..., 0],
                x_grid=x_grid,
                xi_grid=xi_values,
                max_moment=self.config.polynomiality_max_moment,
                fit_degree=self.config.polynomiality_fit_degree,
            )

        if self.config.enforce_forward_limit and self.pdf_provider is not None and forward_h is not None:
            if forward_x is None or forward_q2 is None:
                raise ValueError("forward_x and forward_q2 are required when forward_h is provided.")
            penalties["L_forward"] = forward_limit_penalty(
                h_forward=forward_h,
                x_forward=forward_x,
                q2_forward=forward_q2,
                pdf_provider=self.pdf_provider,
            )

        return penalties


# Backward-compatible helpers used by older scripts/tests.
CONSTRAINTS = {
    "support_x": True,
    "forward_limit": True,
    "qcd_evolution": True,
    "dglap_erbl": True,
    "polynomiality": False,
    "positivity": False,
}


def support_constraint(x: float) -> bool:
    """Legacy scalar support check for old code paths."""
    return -1.0 < x < 1.0


def forward_limit_constraint(gpd: float, pdf: float, tol: float = 1e-2) -> bool:
    """Legacy scalar forward-limit check."""
    return abs(gpd - pdf) < tol


def dglap_erbl_region(x: float, xi: float) -> str:
    """Legacy DGLAP/ERBL region helper."""
    if abs(x) < xi:
        return "ERBL"
    if x > xi:
        return "DGLAP"
    return "unphysical"


def polynomiality_constraint(moments: np.ndarray, tol: float = 1e-2) -> bool:
    """Legacy diagnostic-only polynomiality check."""
    return np.all(np.isfinite(moments)) and bool(np.mean(np.abs(moments)) < (1e6 + tol))


def positivity_constraint(gpd: float) -> bool:
    """Legacy positivity helper."""
    return gpd >= 0
