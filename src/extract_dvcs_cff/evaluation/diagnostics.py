"""
Diagnostics for DVCS/GPD/CFF evaluation.
"""
from typing import Dict, Any
import numpy as np

def residuals(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return obs - pred

def pulls(obs: np.ndarray, pred: np.ndarray, errors: np.ndarray) -> np.ndarray:
    return (obs - pred) / errors

# Add more diagnostics as needed
