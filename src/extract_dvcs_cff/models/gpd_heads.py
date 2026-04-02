"""
Prediction heads for H, E, Htilde, Etilde and optional auxiliary outputs.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from extract_dvcs_cff.models.gpd_backbone import GPDBackbone, build_activation
from extract_dvcs_cff.models.kinematics_encoder import KinematicsEncoder
from extract_dvcs_cff.utils.config import GPDHeadsConfig, PipelineConfig


class _Tower(nn.Module):
    """Small configurable MLP tower used by channel and auxiliary heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        activation: str,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class GPDHeads(nn.Module):
    """
    Predict the four leading quark GPD channels from shared backbone features.

    Outputs
    -------
    Dictionary with keys:
    - gpd: [N, 4] tensor ordered as [H, E, Htilde, Etilde]
    - channels: named tensors per channel
    - optional auxiliary tensors (if enabled)
    """

    def __init__(self, input_dim: int, config: GPDHeadsConfig) -> None:
        super().__init__()
        self.config = config
        self.channel_names = list(config.channels)

        if config.shared_tower:
            self.shared_tower = _Tower(
                input_dim=input_dim,
                hidden_dim=config.tower_hidden_dim,
                depth=config.tower_depth,
                activation=config.activation,
                dropout=config.dropout,
                output_dim=config.tower_hidden_dim,
            )
            channel_input_dim = config.tower_hidden_dim
            self.channel_towers = None
        else:
            self.shared_tower = None
            self.channel_towers = nn.ModuleDict(
                {
                    name: _Tower(
                        input_dim=input_dim,
                        hidden_dim=config.tower_hidden_dim,
                        depth=config.tower_depth,
                        activation=config.activation,
                        dropout=config.dropout,
                        output_dim=config.tower_hidden_dim,
                    )
                    for name in self.channel_names
                }
            )
            channel_input_dim = config.tower_hidden_dim

        self.channel_outputs = nn.ModuleDict(
            {name: nn.Linear(channel_input_dim, 1) for name in self.channel_names}
        )

        aux_cfg = config.auxiliary
        self.aux_cff_head = (
            nn.Linear(input_dim, aux_cfg.cff_output_dim)
            if aux_cfg.enable_cff_head and aux_cfg.cff_output_dim > 0
            else None
        )
        self.aux_mellin_head = (
            nn.Linear(input_dim, aux_cfg.mellin_output_dim)
            if aux_cfg.enable_mellin_head and aux_cfg.mellin_output_dim > 0
            else None
        )
        self.aux_observable_head = (
            nn.Linear(input_dim, aux_cfg.observable_proxy_output_dim)
            if aux_cfg.enable_observable_proxy_head and aux_cfg.observable_proxy_output_dim > 0
            else None
        )

    def _endpoint_factor(self, x: torch.Tensor) -> torch.Tensor:
        x_abs = torch.abs(torch.clamp(x, -1.0, 1.0))
        base = torch.clamp(1.0 - x_abs, min=self.config.support_clamp_margin)
        power = self.config.endpoint_suppression_power
        return torch.pow(base, power)

    def forward(self, features: torch.Tensor, x: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError("features must have shape [N, D].")

        endpoint = None
        if x is not None:
            endpoint = self._endpoint_factor(x).unsqueeze(-1)

        channel_values: OrderedDict[str, torch.Tensor] = OrderedDict()

        if self.shared_tower is not None:
            shared = self.shared_tower(features)
            for name in self.channel_names:
                value = self.channel_outputs[name](shared)
                if endpoint is not None:
                    value = value * endpoint
                channel_values[name] = value.squeeze(-1)
        else:
            assert self.channel_towers is not None
            for name in self.channel_names:
                local = self.channel_towers[name](features)
                value = self.channel_outputs[name](local)
                if endpoint is not None:
                    value = value * endpoint
                channel_values[name] = value.squeeze(-1)

        stacked = torch.stack([channel_values[name] for name in self.channel_names], dim=-1)
        outputs: dict[str, torch.Tensor] = {
            "gpd": stacked,
            "channels": channel_values,
        }

        if self.aux_cff_head is not None:
            outputs["aux_cff"] = self.aux_cff_head(features)
        if self.aux_mellin_head is not None:
            outputs["aux_mellin"] = self.aux_mellin_head(features)
        if self.aux_observable_head is not None:
            outputs["aux_observable_proxy"] = self.aux_observable_head(features)

        return outputs


class DVCSGPDModel(nn.Module):
    """
    End-to-end neural model for predicting the four leading GPD channels.

    The model is deliberately representation-agnostic: no hard-coded parametric
    GPD ansatz is imposed in the primary pathway.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = KinematicsEncoder(config.encoder)
        self.backbone = GPDBackbone(
            config.backbone.model_copy(update={"input_dim": self.encoder.output_dim})
        )
        self.heads = GPDHeads(self.backbone.output_dim, config.heads)

    def forward(
        self,
        kinematics: torch.Tensor,
        process_id: torch.Tensor | None = None,
        flavor_id: torch.Tensor | None = None,
        observable_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(
            kinematics,
            process_id=process_id,
            flavor_id=flavor_id,
            observable_id=observable_id,
        )
        hidden = self.backbone(encoded)
        x = kinematics[:, 0] if kinematics.shape[-1] >= 1 else None
        return self.heads(hidden, x=x)

    def predict_gpd(
        self,
        kinematics: torch.Tensor,
        process_id: torch.Tensor | None = None,
        flavor_id: torch.Tensor | None = None,
        observable_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return only the [N, 4] GPD tensor."""
        return self.forward(
            kinematics,
            process_id=process_id,
            flavor_id=flavor_id,
            observable_id=observable_id,
        )["gpd"]

    def predict_gpd_on_grid(
        self,
        x_grid: torch.Tensor,
        xi: torch.Tensor,
        t: torch.Tensor,
        q2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate GPDs on an x-grid for each event in a batch.

        Parameters
        ----------
        x_grid:
            [Nx] or [B, Nx] tensor.
        xi, t, q2:
            [B] tensors.

        Returns
        -------
        torch.Tensor
            [B, Nx, 4] tensor with channels [H, E, Htilde, Etilde].
        """
        if xi.ndim != 1 or t.ndim != 1 or q2.ndim != 1:
            raise ValueError("xi, t, q2 must all have shape [B].")
        if not (xi.shape == t.shape == q2.shape):
            raise ValueError("xi, t, q2 must have identical shapes.")

        batch = xi.shape[0]
        if x_grid.ndim == 1:
            x_points = x_grid.unsqueeze(0).expand(batch, -1)
        elif x_grid.ndim == 2 and x_grid.shape[0] == batch:
            x_points = x_grid
        else:
            raise ValueError("x_grid must have shape [Nx] or [B, Nx].")

        nx = x_points.shape[1]

        xi_grid = xi.unsqueeze(1).expand(batch, nx)
        t_grid = t.unsqueeze(1).expand(batch, nx)
        q2_grid = q2.unsqueeze(1).expand(batch, nx)

        stacked = torch.stack([x_points, xi_grid, t_grid, q2_grid], dim=-1)
        flat = stacked.reshape(batch * nx, 4)
        flat_pred = self.predict_gpd(flat)
        return flat_pred.reshape(batch, nx, 4)
