"""
Configurable residual backbone for GPD function approximation.

Each hidden block is fully independently configurable and can be replaced
without changing other blocks.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from extract_dvcs_cff.utils.config import BlockConfig, GPDBackboneConfig


def build_activation(name: str) -> nn.Module:
    """Return activation module from config name."""
    normalized = name.lower().strip()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unsupported activation: {name}")


def build_normalization(name: str, width: int) -> nn.Module:
    """Return normalization module from config name."""
    normalized = name.lower().strip()
    if normalized == "none":
        return nn.Identity()
    if normalized == "layernorm":
        return nn.LayerNorm(width)
    if normalized == "batchnorm":
        return nn.BatchNorm1d(width)
    raise ValueError(f"Unsupported normalization: {name}")


class ResidualMLPBlock(nn.Module):
    """
    One independently configurable residual block.

    The block applies an MLP stack and blends it with a residual projection using
    a configurable residual scale.
    """

    def __init__(self, input_dim: int, config: BlockConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = config.width
        self.use_checkpoint = config.checkpoint

        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(config.depth):
            layers.append(nn.Linear(current, config.width))
            layers.append(build_normalization(config.normalization, config.width))
            layers.append(build_activation(config.activation))
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            current = config.width

        self.mlp = nn.Sequential(*layers)
        self.skip = nn.Identity() if input_dim == config.width else nn.Linear(input_dim, config.width)

    def _forward_impl(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        hidden = self.mlp(inputs)
        return residual + self.config.residual_scale * hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, inputs, use_reentrant=False)
        return self._forward_impl(inputs)


class GPDBackbone(nn.Module):
    """Residual MLP backbone with per-block configurability."""

    def __init__(self, config: GPDBackboneConfig) -> None:
        super().__init__()
        self.config = config

        blocks: list[nn.Module] = []
        current_dim = config.input_dim
        for block_cfg in config.blocks:
            block = ResidualMLPBlock(current_dim, block_cfg)
            blocks.append(block)
            current_dim = block.output_dim

        self.blocks = nn.ModuleList(blocks)
        self.final_layer = (
            nn.Identity()
            if current_dim == config.final_dim
            else nn.Linear(current_dim, config.final_dim)
        )
        self.output_dim = config.final_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.final_layer(hidden)
        return torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)
