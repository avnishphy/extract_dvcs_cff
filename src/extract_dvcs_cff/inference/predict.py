"""Inference pipeline for GPDs, CFFs, and observables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from extract_dvcs_cff.models.gpd_heads import DVCSGPDModel
from extract_dvcs_cff.physics.cff_convolution import DifferentiableCFFConvolution
from extract_dvcs_cff.physics.observables import TorchDVCSObservableLayer
from extract_dvcs_cff.utils.config import PipelineConfig


@dataclass
class PredictorOutput:
    """Container for inference outputs."""

    gpd_grid: torch.Tensor
    cff_stacked: torch.Tensor
    observables: torch.Tensor


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_for_inference(
    config: PipelineConfig,
    checkpoint_path: str | Path,
    device: str = "auto",
) -> DVCSGPDModel:
    """Load a trained model checkpoint for inference."""
    model = DVCSGPDModel(config)
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()

    target_device = _resolve_device(device if device != "auto" else config.runtime.device)
    return model.to(target_device)


class DVCSPredictor:
    """Predictor that computes GPDs, CFFs, and observables."""

    def __init__(self, config: PipelineConfig, model: DVCSGPDModel) -> None:
        self.config = config
        self.model = model
        self.convolution = DifferentiableCFFConvolution(config.convolution)
        self.observable_layer = TorchDVCSObservableLayer().to(next(model.parameters()).device)

    @torch.no_grad()
    def predict(
        self,
        kinematics: torch.Tensor,
        observable_id: torch.Tensor,
    ) -> PredictorOutput:
        """
        Predict full physics outputs for a batch.

        kinematics convention: [xB, xi, t, Q2, phi].
        """
        device = next(self.model.parameters()).device
        kinematics = kinematics.to(device)
        observable_id = observable_id.to(device)

        xi = kinematics[:, 1]
        t = kinematics[:, 2]
        q2 = kinematics[:, 3]

        x_grid = self.convolution.get_x_grid(device=device, dtype=kinematics.dtype)
        gpd_grid = self.model.predict_gpd_on_grid(x_grid=x_grid, xi=xi, t=t, q2=q2)
        cff = self.convolution(gpd_grid=gpd_grid, xi=xi, x_grid=x_grid)
        obs = self.observable_layer(cff.stacked, kinematics, observable_id)

        return PredictorOutput(
            gpd_grid=gpd_grid,
            cff_stacked=cff.stacked,
            observables=obs,
        )
