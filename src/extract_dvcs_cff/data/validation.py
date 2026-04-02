"""
Validation utilities for DVCS/GPD/CFF datasets.
"""
from .schemas import DatasetRecord, KinematicPoint
import numpy as np

def validate_kinematic_point(point: KinematicPoint) -> None:
    if not (0 < point.xB < 1):
        raise ValueError(f"xB out of range: {point.xB}")
    if point.Q2 <= 0:
        raise ValueError(f"Q2 must be > 0: {point.Q2}")
    if point.t > 0:
        raise ValueError(f"t must be <= 0 in DVCS convention: {point.t}")
    if not (0 <= point.phi <= 360):
        raise ValueError(f"phi out of range: {point.phi}")
    if np.isnan(point.t) or np.isinf(point.t):
        raise ValueError(f"t is NaN or Inf: {point.t}")
    if point.y is not None and (np.isnan(point.y) or np.isinf(point.y)):
        raise ValueError(f"y is NaN or Inf: {point.y}")
    if point.y is not None and not (0 <= point.y <= 1):
        raise ValueError(f"y is outside physical range [0, 1]: {point.y}")
    if point.beam_energy is not None and (np.isnan(point.beam_energy) or np.isinf(point.beam_energy)):
        raise ValueError(f"beam_energy is NaN or Inf: {point.beam_energy}")
    if point.beam_energy is not None and point.beam_energy <= 0:
        raise ValueError(f"beam_energy must be > 0: {point.beam_energy}")

def validate_dataset_record(record: DatasetRecord) -> None:
    if not record.dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")
    if not record.experiment_name.strip():
        raise ValueError("experiment_name must be non-empty")
    if len(record.observables) == 0:
        raise ValueError("dataset has no observables")
    if len(record.kinematics) == 0:
        raise ValueError("dataset has no kinematic points")
    if len(record.observables) != len(record.kinematics):
        raise ValueError("observables and kinematics lengths must match")

    for point in record.kinematics:
        validate_kinematic_point(point)

    for obs in record.observables:
        if not obs.observable_name.strip():
            raise ValueError("observable_name must be non-empty")
        if not np.isfinite(obs.value):
            raise ValueError(f"observable value is non-finite: {obs.value}")
        if obs.stat_error is not None:
            if not np.isfinite(obs.stat_error) or obs.stat_error < 0:
                raise ValueError(f"invalid stat_error: {obs.stat_error}")
        if obs.sys_error is not None:
            if not np.isfinite(obs.sys_error) or obs.sys_error < 0:
                raise ValueError(f"invalid sys_error: {obs.sys_error}")
        if obs.total_error is not None:
            if not np.isfinite(obs.total_error) or obs.total_error < 0:
                raise ValueError(f"invalid total_error: {obs.total_error}")
