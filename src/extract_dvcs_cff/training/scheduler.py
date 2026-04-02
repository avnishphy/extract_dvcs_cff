"""Scheduling helpers for loss curriculum and learning-rate updates."""

from __future__ import annotations

import torch

from extract_dvcs_cff.utils.config import LossConfig, TrainingConfig


def current_loss_phase_name(loss_config: LossConfig, epoch: int) -> str:
    """Return a human-readable loss curriculum phase name for logging."""
    for idx, phase in enumerate(loss_config.phases):
        if phase.start_epoch <= epoch <= phase.end_epoch:
            return f"phase_{idx + 1}"
    return "phase_final"


def build_lr_scheduler(optimizer: torch.optim.Optimizer, training_cfg: TrainingConfig) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a conservative cosine scheduler for stable multi-term optimization."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(training_cfg.epochs, 1),
        eta_min=0.1 * training_cfg.learning_rate,
    )
