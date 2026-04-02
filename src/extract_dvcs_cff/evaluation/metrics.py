"""
Metrics and diagnostics for DVCS/GPD/CFF evaluation.
"""
from typing import Dict
import numpy as np


def _validate_shapes(obs: np.ndarray, pred: np.ndarray) -> None:
    if obs.ndim != 1 or pred.ndim != 1:
        raise ValueError("obs and pred must both be 1D arrays")
    if obs.shape != pred.shape:
        raise ValueError("obs and pred must have identical shapes")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(pred)):
        raise ValueError("obs and pred must contain only finite values")


def compute_pointwise_error(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    _validate_shapes(obs, pred)
    return obs - pred


def compute_reduced_chi2(obs: np.ndarray, pred: np.ndarray, errors: np.ndarray, dof: int) -> float:
    _validate_shapes(obs, pred)
    if errors.ndim != 1 or errors.shape != obs.shape:
        raise ValueError("errors must be a 1D array matching obs/pred shape")
    if np.any(~np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError("errors must be finite and strictly positive")
    if dof <= 0:
        raise ValueError("dof must be positive")
    chi2 = np.sum(((obs - pred) / errors) ** 2)
    return chi2 / dof


def compute_coverage(obs: np.ndarray, pred: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    _validate_shapes(obs, pred)
    if lower.ndim != 1 or upper.ndim != 1 or lower.shape != obs.shape or upper.shape != obs.shape:
        raise ValueError("lower/upper must be 1D arrays matching obs/pred shape")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("lower/upper must contain only finite values")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    within = (obs >= lower) & (obs <= upper)
    return float(np.mean(within))


def compute_metrics(obs: np.ndarray, pred: np.ndarray, errors: np.ndarray) -> Dict[str, float]:
    """Return a compact metric summary for CLI and diagnostics reports."""
    _validate_shapes(obs, pred)
    if errors.ndim != 1 or errors.shape != obs.shape:
        raise ValueError("errors must be a 1D array matching obs/pred shape")
    if np.any(~np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError("errors must be finite and strictly positive")

    residuals = obs - pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    dof = max(1, int(obs.shape[0] - 1))
    chi2_red = compute_reduced_chi2(obs, pred, errors, dof=dof)

    return {
        "rmse": rmse,
        "mae": mae,
        "chi2_reduced": float(chi2_red),
    }


def compute_replica_statistics(
    replica_predictions: np.ndarray,
    lower_quantile: float = 0.16,
    upper_quantile: float = 0.84,
) -> Dict[str, np.ndarray]:
    """Compute central tendency and uncertainty bands from replica predictions."""
    arr = np.asarray(replica_predictions, dtype=float)
    if arr.ndim != 2:
        raise ValueError("replica_predictions must have shape [n_replicas, n_points].")
    if arr.shape[0] < 2:
        raise ValueError("At least two replicas are required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("replica_predictions contains non-finite values.")

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    lower = np.quantile(arr, lower_quantile, axis=0)
    upper = np.quantile(arr, upper_quantile, axis=0)

    return {
        "mean": mean,
        "std": std,
        "lower": lower,
        "upper": upper,
    }


def compute_closure_recovery_score(
    truth: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
) -> float:
    """Return average absolute pull for closure tests."""
    truth = np.asarray(truth, dtype=float)
    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_std = np.asarray(pred_std, dtype=float)

    _validate_shapes(truth, pred_mean)
    if pred_std.shape != truth.shape:
        raise ValueError("pred_std must match truth shape.")
    if np.any(~np.isfinite(pred_std)) or np.any(pred_std <= 0):
        raise ValueError("pred_std must be finite and strictly positive.")

    pulls = np.abs(truth - pred_mean) / pred_std
    return float(np.mean(pulls))
