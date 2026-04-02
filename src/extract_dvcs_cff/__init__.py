"""
extract_dvcs_cff: DVCS/GPD/CFF Extraction Framework
"""

from __future__ import annotations

import importlib

__all__ = [
    "config",
    "data",
    "physics",
    "partons",
    "apfel",
    "lhapdf",
    "simulation",
    "evaluation",
    "plotting",
    "utils",
    "models",
    "losses",
    "training",
    "inference",
]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
