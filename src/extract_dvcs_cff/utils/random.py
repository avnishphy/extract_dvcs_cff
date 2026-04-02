"""
Deterministic seeding utilities for extract_dvcs_cff.
"""
import os
import random

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is an optional runtime dependency in tests.
    torch = None


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and optionally PyTorch backends."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
