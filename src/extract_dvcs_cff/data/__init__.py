"""
extract_dvcs_cff.data package init.
"""
from __future__ import annotations

import importlib

from .schemas import KinematicPoint, ObservableRecord, DatasetRecord, TheoryPoint, CovarianceMatrixContainer

__all__ = [
    "KinematicPoint",
    "ObservableRecord",
    "DatasetRecord",
    "TheoryPoint",
    "CovarianceMatrixContainer",
    "GlobalDVCSDataset",
    "DatasetMappings",
]


def __getattr__(name: str):
    if name in {"GlobalDVCSDataset", "DatasetMappings"}:
        module = importlib.import_module(".dvcs_dataset", __name__)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
