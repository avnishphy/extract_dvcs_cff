"""
Pydantic schemas for DVCS/GPD/CFF data and metadata.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import numpy as np

class KinematicPoint(BaseModel):
    xB: float = Field(..., gt=0, lt=1, description="Bjorken x (0 < xB < 1)")
    Q2: float = Field(..., gt=0, description="Photon virtuality Q^2 (GeV^2)")
    t: float = Field(..., description="Momentum transfer t (GeV^2)")
    phi: float = Field(..., ge=0, le=360, description="Azimuthal angle phi (deg)")
    beam_energy: Optional[float] = Field(None, description="Beam energy (GeV)")
    y: Optional[float] = Field(None, description="Inelasticity y")
    units: Optional[Dict[str, str]] = Field(default_factory=dict, description="Units metadata")

    @field_validator("xB")
    @classmethod
    def xb_range(cls, v):
        if not (0 < v < 1):
            raise ValueError("xB must be in (0, 1)")
        return v

    @field_validator("Q2")
    @classmethod
    def q2_positive(cls, v):
        if v <= 0:
            raise ValueError("Q2 must be > 0")
        return v

    @field_validator("phi")
    @classmethod
    def phi_range(cls, v):
        if not (0 <= v <= 360):
            raise ValueError("phi must be in [0, 360] degrees")
        return v

    @field_validator("t")
    @classmethod
    def t_finite_and_physical(cls, v):
        if not np.isfinite(v):
            raise ValueError("t must be finite")
        if v > 0:
            raise ValueError("t must be <= 0 in DVCS convention")
        return v

    @field_validator("beam_energy")
    @classmethod
    def beam_energy_positive(cls, v):
        if v is None:
            return v
        if not np.isfinite(v):
            raise ValueError("beam_energy must be finite")
        if v <= 0:
            raise ValueError("beam_energy must be > 0")
        return v

    @field_validator("y")
    @classmethod
    def y_range(cls, v):
        if v is None:
            return v
        if not np.isfinite(v):
            raise ValueError("y must be finite")
        if not (0 <= v <= 1):
            raise ValueError("y must be in [0, 1]")
        return v

class ObservableRecord(BaseModel):
    observable_name: str
    value: float
    stat_error: Optional[float] = None
    sys_error: Optional[float] = None
    total_error: Optional[float] = None
    covariance_id: Optional[str] = None
    channel: Optional[str] = None

    @field_validator("observable_name")
    @classmethod
    def observable_name_nonempty(cls, v):
        name = v.strip()
        if not name:
            raise ValueError("observable_name must be non-empty")
        return name

    @field_validator("value")
    @classmethod
    def value_finite(cls, v):
        if not np.isfinite(v):
            raise ValueError("value must be finite")
        return v

    @field_validator("stat_error", "sys_error", "total_error")
    @classmethod
    def errors_nonnegative_finite(cls, v):
        if v is None:
            return v
        if not np.isfinite(v):
            raise ValueError("errors must be finite when provided")
        if v < 0:
            raise ValueError("errors must be non-negative")
        return v

    @model_validator(mode="after")
    def check_total_error_consistency(self):
        if self.total_error is None:
            return self
        if self.stat_error is None and self.sys_error is None:
            return self

        stat = 0.0 if self.stat_error is None else self.stat_error
        sys = 0.0 if self.sys_error is None else self.sys_error
        expected = float(np.hypot(stat, sys))
        tolerance = 1e-8 + 1e-6 * expected
        # total_error may include additional terms (e.g. normalization uncertainty),
        # but it must never be smaller than quadrature(stat, sys).
        if self.total_error + tolerance < expected:
            raise ValueError("total_error must be >= quadrature(stat_error, sys_error)")
        return self

class DatasetRecord(BaseModel):
    experiment_name: str
    dataset_id: str
    publication: Optional[str] = None
    observables: List[ObservableRecord]
    kinematics: List[KinematicPoint]
    comments: Optional[str] = None

    @model_validator(mode="after")
    def check_dataset_integrity(self):
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if len(self.observables) == 0:
            raise ValueError("observables must contain at least one entry")
        if len(self.kinematics) == 0:
            raise ValueError("kinematics must contain at least one entry")
        if len(self.observables) != len(self.kinematics):
            raise ValueError("observables and kinematics must have the same number of entries")
        return self

class TheoryPoint(BaseModel):
    benchmark_model: str
    cff_values: Optional[Dict[str, float]] = None
    gpd_values: Optional[Dict[str, float]] = None
    kinematics: KinematicPoint

class CovarianceMatrixContainer(BaseModel):
    covariance: Optional[np.ndarray] = None
    correlation: Optional[np.ndarray] = None
    observable_names: Optional[List[str]] = None
    dataset_id: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("covariance", "correlation", mode="before")
    @classmethod
    def check_array(cls, v):
        if v is not None and not isinstance(v, np.ndarray):
            return np.array(v)
        return v

    @model_validator(mode="after")
    def validate_matrix_shapes(self):
        covariance = self.covariance
        correlation = self.correlation

        if covariance is not None:
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError("covariance must be a square 2D matrix")
            if not np.all(np.isfinite(covariance)):
                raise ValueError("covariance must contain only finite values")

        if correlation is not None:
            if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
                raise ValueError("correlation must be a square 2D matrix")
            if not np.all(np.isfinite(correlation)):
                raise ValueError("correlation must contain only finite values")

        if covariance is not None and correlation is not None and covariance.shape != correlation.shape:
            raise ValueError("covariance and correlation must have identical shapes")

        if self.observable_names is not None:
            n_obs = len(self.observable_names)
            if covariance is not None and covariance.shape[0] != n_obs:
                raise ValueError("observable_names length must match covariance size")
            if correlation is not None and correlation.shape[0] != n_obs:
                raise ValueError("observable_names length must match correlation size")

        return self
