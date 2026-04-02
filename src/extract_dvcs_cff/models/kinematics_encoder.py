"""
Kinematics encoder for DVCS/GPD learning.

The encoder performs minimal, explicit preprocessing on
(x, xi, t, Q2) and optional categorical metadata before feeding a learnable
feature stack.
"""

from __future__ import annotations

import torch
from torch import nn

from extract_dvcs_cff.utils.config import KinematicsEncoderConfig


def _safe_logit(value: torch.Tensor, eps: float) -> torch.Tensor:
    clipped = torch.clamp(value, min=eps, max=1.0 - eps)
    return torch.log(clipped / (1.0 - clipped))


def _signed_log1p_abs(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


class KinematicsEncoder(nn.Module):
    """
    Encode kinematics into a dense representation.

    Inputs
    ------
    kinematics:
        Tensor of shape [N, 4] in the convention [x, xi, t, Q2].
    process_id/flavor_id/observable_id:
        Optional integer tensors of shape [N] used for small learned embeddings.
    """

    def __init__(self, config: KinematicsEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.feature_proj = nn.Sequential(
            nn.Linear(4, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        self.process_embedding = (
            nn.Embedding(config.process_vocab_size, config.process_embedding_dim)
            if config.use_process_embedding
            else None
        )
        self.flavor_embedding = (
            nn.Embedding(config.flavor_vocab_size, config.flavor_embedding_dim)
            if config.use_flavor_embedding
            else None
        )
        self.observable_embedding = (
            nn.Embedding(config.observable_vocab_size, config.observable_embedding_dim)
            if config.use_observable_embedding
            else None
        )

        final_dim = config.output_dim
        if self.process_embedding is not None:
            final_dim += config.process_embedding_dim
        if self.flavor_embedding is not None:
            final_dim += config.flavor_embedding_dim
        if self.observable_embedding is not None:
            final_dim += config.observable_embedding_dim

        self.output_dim = final_dim

    def _encode_core(self, kinematics: torch.Tensor) -> torch.Tensor:
        if kinematics.ndim != 2 or kinematics.shape[-1] < 4:
            raise ValueError("kinematics must have shape [N, 4] or [N, >=4].")
        if not torch.isfinite(kinematics).all():
            raise ValueError("kinematics contains NaN/Inf values.")

        x = kinematics[:, 0]
        xi = kinematics[:, 1]
        t = kinematics[:, 2]
        q2 = kinematics[:, 3]

        transformed = torch.stack(
            [
                _safe_logit(x, self.config.eps),
                _safe_logit(xi, self.config.eps),
                _signed_log1p_abs(t),
                torch.log(torch.clamp(q2, min=self.config.eps)),
            ],
            dim=-1,
        )
        return self.feature_proj(transformed)

    def forward(
        self,
        kinematics: torch.Tensor,
        process_id: torch.Tensor | None = None,
        flavor_id: torch.Tensor | None = None,
        observable_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode kinematics and optional categorical metadata."""
        encoded = [self._encode_core(kinematics)]

        if self.process_embedding is not None:
            if process_id is None:
                raise ValueError("process_id is required when process embedding is enabled.")
            encoded.append(self.process_embedding(process_id.long()))

        if self.flavor_embedding is not None:
            if flavor_id is None:
                raise ValueError("flavor_id is required when flavor embedding is enabled.")
            encoded.append(self.flavor_embedding(flavor_id.long()))

        if self.observable_embedding is not None:
            if observable_id is None:
                raise ValueError("observable_id is required when observable embedding is enabled.")
            encoded.append(self.observable_embedding(observable_id.long()))

        return torch.cat(encoded, dim=-1)
