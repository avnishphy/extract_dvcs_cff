"""Numerical helper functions for stable differentiable computation."""

from __future__ import annotations

import torch


def safe_log(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically safe logarithm."""
    return torch.log(torch.clamp(value, min=eps))


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically safe division."""
    return numerator / torch.clamp(denominator, min=eps)


def nan_to_num(value: torch.Tensor, clamp: float = 1e6) -> torch.Tensor:
    """Replace NaN/Inf by finite guarded values."""
    return torch.nan_to_num(value, nan=0.0, posinf=clamp, neginf=-clamp)


def trapz_with_fallback(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Use trapz with a finite-value fallback for unstable regions."""
    integral = torch.trapz(values, grid, dim=-1)
    if torch.isfinite(integral).all():
        return integral

    clean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.trapz(clean, grid, dim=-1)
