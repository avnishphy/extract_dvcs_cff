"""Physics-aware differentiable loss terms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from extract_dvcs_cff.physics.constraints import PhysicsConstraintEvaluator
from extract_dvcs_cff.physics.evolution import Q2EvolutionLayer


def data_misfit_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor | None = None,
    covariance: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute weighted data-space misfit.

    Supports either diagonal uncertainties or a full covariance matrix.
    """
    if covariance is not None:
        residual = target - prediction
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be a square matrix.")
        if covariance.shape[0] != residual.shape[0]:
            raise ValueError("covariance dimension must match residual length.")
        chol = torch.linalg.cholesky(covariance)
        solved = torch.cholesky_solve(residual.unsqueeze(-1), chol).squeeze(-1)
        chi2 = torch.dot(residual, solved)
        return chi2 / residual.shape[0]

    sigma = torch.clamp(sigma, min=1e-8)
    residual = (prediction - target) / sigma
    if mask is not None:
        residual = residual * mask
        denom = torch.clamp(mask.sum(), min=1.0)
    else:
        denom = float(residual.numel())
    return torch.sum(residual * residual) / denom


def transform_consistency_loss(
    aux_cff: torch.Tensor | None,
    cff_stacked: torch.Tensor,
) -> torch.Tensor:
    """Consistency between auxiliary CFF head and convolution-derived CFFs."""
    if aux_cff is None:
        return torch.zeros((), device=cff_stacked.device, dtype=cff_stacked.dtype)

    expected = cff_stacked.reshape(cff_stacked.shape[0], -1)
    if aux_cff.shape != expected.shape:
        min_dim = min(aux_cff.shape[-1], expected.shape[-1])
        return torch.mean((aux_cff[:, :min_dim] - expected[:, :min_dim]) ** 2)
    return torch.mean((aux_cff - expected) ** 2)


def regularization_loss(module: nn.Module) -> torch.Tensor:
    """L2 regularization over trainable parameters."""
    terms = [torch.sum(param * param) for param in module.parameters() if param.requires_grad]
    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).mean()


@dataclass
class PhysicsLossTermComputer:
    """Compute all loss components required by the composite objective."""

    constraints: PhysicsConstraintEvaluator
    evolution: Q2EvolutionLayer | None = None

    def compute(
        self,
        *,
        model: nn.Module,
        pred_observables: torch.Tensor,
        target_observables: torch.Tensor,
        sigma: torch.Tensor,
        mask: torch.Tensor,
        cff_stacked: torch.Tensor,
        aux_cff: torch.Tensor | None,
        gpd_grid: torch.Tensor,
        x_grid: torch.Tensor,
        xi_values: torch.Tensor,
        t_values: torch.Tensor,
        q2_values: torch.Tensor,
        forward_h: torch.Tensor | None = None,
        forward_x: torch.Tensor | None = None,
        forward_q2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return all scalar loss terms."""
        terms: dict[str, torch.Tensor] = {}

        terms["L_DVCS"] = data_misfit_loss(
            prediction=pred_observables,
            target=target_observables,
            sigma=sigma,
            mask=mask,
        )

        terms["L_transform"] = transform_consistency_loss(aux_cff=aux_cff, cff_stacked=cff_stacked)

        constraint_terms = self.constraints.evaluate(
            gpd_grid=gpd_grid,
            x_grid=x_grid,
            xi_values=xi_values,
            t_values=t_values,
            q2_values=q2_values,
            forward_h=forward_h,
            forward_x=forward_x,
            forward_q2=forward_q2,
        )

        if "L_forward" in constraint_terms:
            terms["L_forward"] = constraint_terms["L_forward"]
        if "L_sumrule" in constraint_terms:
            terms["L_sumrule"] = constraint_terms["L_sumrule"]
        if "L_polynomiality" in constraint_terms:
            terms["L_polynomiality"] = constraint_terms["L_polynomiality"]
        if "L_positivity" in constraint_terms:
            terms["L_positivity"] = constraint_terms["L_positivity"]

        smooth = constraint_terms.get("L_smoothness")
        terms["L_smooth"] = (
            smooth if smooth is not None else torch.zeros((), device=pred_observables.device, dtype=pred_observables.dtype)
        )

        if self.evolution is not None:
            # Use channel values evaluated at x=xi as a consistency anchor.
            center_idx = x_grid.shape[0] // 2
            gpd_center = gpd_grid[:, center_idx, :]
            terms["L_evolution"] = self.evolution.backward_consistency_penalty(gpd_center, q2_values)
        else:
            terms["L_evolution"] = torch.zeros((), device=pred_observables.device, dtype=pred_observables.dtype)

        terms["L_regularization"] = regularization_loss(model)

        return terms
