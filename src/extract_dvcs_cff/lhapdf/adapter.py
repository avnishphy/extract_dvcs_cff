"""
Adapter for LHAPDF external physics code.
"""
from __future__ import annotations

import importlib
import logging
from typing import Dict

import torch

class LHAPDFAdapter:
    """Adapter for LHAPDF backend with optional analytic fallback."""

    FLAVOR_TO_PID = {
        "u": 2,
        "d": 1,
        "s": 3,
        "c": 4,
        "b": 5,
        "ubar": -2,
        "dbar": -1,
        "sbar": -3,
    }

    def __init__(self, set_name: str = "CT18NNLO", member: int = 0):
        self.set_name = set_name
        self.member = member
        self._backend = None
        self._pdf = None
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            self._backend = importlib.import_module("lhapdf")
        except ModuleNotFoundError:
            self._backend = None
            return False

        try:
            self._pdf = self._backend.mkPDF(self.set_name, self.member)
        except Exception as exc:  # pragma: no cover - backend-dependent
            logging.warning("LHAPDF import succeeded but PDF set loading failed: %s", exc)
            self._pdf = None
            return False
        return True

    @staticmethod
    def _analytic_fallback(flavor: str, x: float, q2: float) -> float:
        x = max(min(float(x), 1.0 - 1e-8), 1e-8)
        q2 = max(float(q2), 1e-6)

        flavor_scale = {
            "u": 1.0,
            "d": 0.6,
            "s": 0.2,
            "c": 0.08,
            "b": 0.02,
            "ubar": 0.15,
            "dbar": 0.12,
            "sbar": 0.08,
        }.get(flavor, 0.1)

        return flavor_scale * x ** (-0.25) * (1.0 - x) ** 3.0 * (1.0 + 0.05 * torch.log(torch.tensor(q2)).item())

    def get_pdf(self, flavor: str, x: float, Q2: float) -> float:
        flavor_key = flavor.strip().lower()
        if not self.available or self._pdf is None:
            return float(self._analytic_fallback(flavor_key, x, Q2))

        pid = self.FLAVOR_TO_PID.get(flavor_key)
        if pid is None:
            raise ValueError(f"Unsupported flavor '{flavor}'.")

        x_safe = max(min(float(x), 1.0 - 1e-8), 1e-8)
        q2_safe = max(float(Q2), 1e-6)
        # LHAPDF returns x * f(x, Q), so divide by x to get f(x, Q).
        return float(self._pdf.xfxQ2(pid, x_safe, q2_safe) / x_safe)

    def evaluate(self, flavor: str, x: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Vectorized interface compatible with physics constraint providers."""
        x_cpu = x.detach().cpu().flatten().tolist()
        q_cpu = q2.detach().cpu().flatten().tolist()
        values = [self.get_pdf(flavor, xi, qi) for xi, qi in zip(x_cpu, q_cpu)]
        return torch.tensor(values, device=x.device, dtype=x.dtype).reshape_as(x)

    def compare_forward_limit(self, gpd: Dict[str, float], pdf: Dict[str, float]) -> bool:
        keys = set(gpd) & set(pdf)
        if not keys:
            raise ValueError("No overlapping keys between gpd and pdf dictionaries.")
        max_rel_error = 0.0
        for key in keys:
            denom = max(abs(pdf[key]), 1e-12)
            max_rel_error = max(max_rel_error, abs(gpd[key] - pdf[key]) / denom)
        return bool(max_rel_error < 0.2)
