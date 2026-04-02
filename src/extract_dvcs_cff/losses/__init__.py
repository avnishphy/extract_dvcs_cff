"""Loss modules for end-to-end DVCS->GPD training."""

from .composite import CompositeLoss, WEIGHT_TO_TERM
from .physics_terms import PhysicsLossTermComputer, data_misfit_loss

__all__ = [
    "CompositeLoss",
    "WEIGHT_TO_TERM",
    "PhysicsLossTermComputer",
    "data_misfit_loss",
]
