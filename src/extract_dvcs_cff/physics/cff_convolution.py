"""
Differentiable CFF convolution from GPD grids.

This module implements a stable principal-value style integral with optional
analytic singular-term handling near x = xi.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from extract_dvcs_cff.utils.config import ConvolutionConfig


@dataclass(frozen=True)
class ConvolutionResult:
    """Structured output for CFF channels."""

    stacked: torch.Tensor
    H_real: torch.Tensor
    H_imag: torch.Tensor
    E_real: torch.Tensor
    E_imag: torch.Tensor
    Htilde_real: torch.Tensor
    Htilde_imag: torch.Tensor
    Etilde_real: torch.Tensor
    Etilde_imag: torch.Tensor


class DifferentiableCFFConvolution(nn.Module):
    """
    Principal-value convolution layer.

    Input convention:
    - gpd_grid has shape [B, Nx, 4] with channels [H, E, Htilde, Etilde]
    - xi has shape [B]
    - x_grid has shape [Nx] if provided
    """

    CHANNELS = ("H", "E", "Htilde", "Etilde")

    def __init__(self, config: ConvolutionConfig) -> None:
        super().__init__()
        self.config = config
        self._grid_cache: dict[tuple[str, str], torch.Tensor] = {}

    def get_x_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return cached x-grid on the requested device/dtype."""
        key = (str(device), str(dtype))
        if self.config.cache_grids and key in self._grid_cache:
            return self._grid_cache[key]

        eps = 1e-6
        grid = torch.linspace(-1.0 + eps, 1.0 - eps, self.config.x_grid_size, device=device, dtype=dtype)
        if self.config.cache_grids:
            self._grid_cache[key] = grid
        return grid

    @staticmethod
    def _interp_at_xi(gpd_channel: torch.Tensor, x_grid: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of GPD channel values at x=xi."""
        idx = torch.searchsorted(x_grid, xi)
        idx = torch.clamp(idx, 1, x_grid.numel() - 1)

        x0 = x_grid[idx - 1]
        x1 = x_grid[idx]

        row = torch.arange(gpd_channel.shape[0], device=gpd_channel.device)
        y0 = gpd_channel[row, idx - 1]
        y1 = gpd_channel[row, idx]

        weight = (xi - x0) / torch.clamp(x1 - x0, min=1e-8)
        return (1.0 - weight) * y0 + weight * y1

    def _principal_value_real(
        self,
        gpd_channel: torch.Tensor,
        x_grid: torch.Tensor,
        xi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute real part via principal-value-like regularization."""
        xi = torch.clamp(xi, min=-0.999, max=0.999)
        gpd_xi = self._interp_at_xi(gpd_channel, x_grid, xi)

        diff = x_grid.unsqueeze(0) - xi.unsqueeze(-1)

        if self.config.use_analytic_singularity_term:
            integrand = (gpd_channel - gpd_xi.unsqueeze(-1)) / torch.clamp(
                diff,
                min=-1e6,
                max=1e6,
            )
            regular = torch.trapz(integrand, x_grid, dim=-1)
            analytic = gpd_xi * torch.log(
                torch.clamp((1.0 - xi) / torch.clamp(1.0 + xi, min=1e-8), min=1e-8)
            )
            real_part = regular + analytic
            return real_part, gpd_xi

        # Fallback integration by masking a neighborhood around x=xi.
        mask = torch.abs(diff) > self.config.pv_eps
        integrand = torch.where(mask, gpd_channel / torch.clamp(diff, min=1e-8), torch.zeros_like(gpd_channel))
        real_part = torch.trapz(integrand, x_grid, dim=-1)
        return real_part, gpd_xi

    def forward(
        self,
        gpd_grid: torch.Tensor,
        xi: torch.Tensor,
        x_grid: torch.Tensor | None = None,
    ) -> ConvolutionResult:
        """Convolve GPDs into CFFs with differentiable operations."""
        if gpd_grid.ndim != 3 or gpd_grid.shape[-1] != 4:
            raise ValueError("gpd_grid must have shape [B, Nx, 4].")
        if xi.ndim != 1 or xi.shape[0] != gpd_grid.shape[0]:
            raise ValueError("xi must have shape [B] matching gpd_grid batch dimension.")

        if x_grid is None:
            x_grid = self.get_x_grid(device=gpd_grid.device, dtype=gpd_grid.dtype)

        if x_grid.ndim != 1 or x_grid.shape[0] != gpd_grid.shape[1]:
            raise ValueError("x_grid must have shape [Nx] and match gpd_grid's x dimension.")

        results = []
        imag_scale = torch.pi

        for channel_idx in range(4):
            channel = gpd_grid[..., channel_idx]
            real_part, gpd_xi = self._principal_value_real(channel, x_grid, xi)
            imag_part = imag_scale * gpd_xi
            results.append(torch.stack([real_part, imag_part], dim=-1))

        stacked = torch.stack(results, dim=1)
        return ConvolutionResult(
            stacked=stacked,
            H_real=stacked[:, 0, 0],
            H_imag=stacked[:, 0, 1],
            E_real=stacked[:, 1, 0],
            E_imag=stacked[:, 1, 1],
            Htilde_real=stacked[:, 2, 0],
            Htilde_imag=stacked[:, 2, 1],
            Etilde_real=stacked[:, 3, 0],
            Etilde_imag=stacked[:, 3, 1],
        )
