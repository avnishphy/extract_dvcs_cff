# """
# Observable definitions and helpers for DVCS/GPD/CFF analysis.
# """
# from typing import Dict, List

# def map_observable_label(label: str) -> str:

# # Canonical observable types and database label mapping
# # This list is derived from the OpenGPD database summary and covers all observables present in the datasets.
# OBSERVABLE_TYPES = [
#     # Cross sections
#     "cross_section",
#     "CrossSectionUU",
#     "CrossSectionDifferenceLU",
#     "CrossSectionUUVirtualPhotoProduction",
#     # Asymmetries
#     "beam_spin_asymmetry",
#     "charge_asymmetry",
#     "AcCos0Phi", "AcCos1Phi", "AcCos2Phi", "AcCos3Phi",
#     "ALUIntSin1Phi", "ALUDVCSSin1Phi", "ALUIntSin2Phi",
#     "ALU", "ALL", "ALUSin1Phi", "ALUSin2Phi", "AULSin1Phi", "AULSin2Phi",
#     "Ac", "BSA", "CA",
#     # Slopes and other
#     "TSlope",
# ]

# # Map database labels to internal names for consistency
# OBSERVABLE_LABEL_MAP = {
#     # Cross sections
#     "CrossSectionUU": "cross_section_uu",
#     "CrossSectionDifferenceLU": "cross_section_difference_lu",
#     "CrossSectionUUVirtualPhotoProduction": "cross_section_uu_virtual_photoproduction",
#     # Asymmetries
#     "AcCos0Phi": "ac_cos0phi",
#     "AcCos1Phi": "ac_cos1phi",
#     "AcCos2Phi": "ac_cos2phi",
#     "AcCos3Phi": "ac_cos3phi",
#     "ALUIntSin1Phi": "alu_int_sin1phi",
#     "ALUDVCSSin1Phi": "alu_dvcs_sin1phi",
#     "ALUIntSin2Phi": "alu_int_sin2phi",
#     "ALU": "alu",
#     "ALL": "all",
#     "ALUSin1Phi": "alu_sin1phi",
#     "ALUSin2Phi": "alu_sin2phi",
#     "AULSin1Phi": "aul_sin1phi",
#     "AULSin2Phi": "aul_sin2phi",
#     "Ac": "ac",
#     "BSA": "beam_spin_asymmetry",
#     "CA": "charge_asymmetry",
#     # Slopes
#     "TSlope": "t_slope",
# }

# def map_observable_label(label: str) -> str:
#     """
#     Map a database observable label to a canonical internal name.
#     If not found, returns the label unchanged.
#     """
#     return OBSERVABLE_LABEL_MAP.get(label, label)

# # Utility: get all unique observable names from the canonical list
# def get_all_observable_names() -> list:
#     return list(OBSERVABLE_LABEL_MAP.keys())


"""
Observable definitions, label mapping, kinematics containers, and a toy DVCS
observable calculator for DVCS/GPD/CFF analysis.

This module is designed to do two jobs at once:

1) provide canonical names and label mapping for observables that appear in the
   gpd database (https://opengpd.github.io/gpddatabase/index.html);
   
2) provide a controlled toy forward model that is smooth, vectorized, and
   sufficiently rich to validate the likelihood and fitting pipeline before
   integrating PARTONS.

The toy model is not meant to be phenomenologically exact. It is meant to be:
- deterministic
- smooth in kinematics and CFFs
- vectorized
- positive where required
- bounded for asymmetries
- compatible with GaussianLikelihood
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "OBSERVABLE_TYPES",
    "DATABASE_OBSERVABLE_LABELS",
    "OBSERVABLE_LABEL_MAP",
    "map_observable_label",
    "get_all_observable_names",
    "get_all_database_labels",
    "is_supported_observable",
    "Kinematics",
    "KinematicsBatch",
    "ToyCFFParameters",
    "xi_from_xB",
    "generate_toy_cffs",
    "scale_toy_cffs",
    "ObservableCalculator",
    "ToyDVCSObservableCalculator",
    "TORCH_OBSERVABLE_INDEX",
    "observable_name_to_index",
    "TorchDVCSObservableLayer",
]


_TWO_PI = 2.0 * np.pi
_EPS = 1e-12


def _normalize_label(label: str) -> str:
    """
    Normalize an observable label for robust matching.

    This removes whitespace and makes matching case-insensitive.
    """
    return "".join(label.split()).lower()


@dataclass(frozen=True)
class ObservableDefinition:
    """
    Metadata for an observable family.

    Attributes
    ----------
    canonical_name:
        Internal canonical name used by the codebase.
    database_labels:
        Labels that may appear in the OpenGPD database for this family.
    description:
        Short physics-oriented description.
    """

    canonical_name: str
    database_labels: tuple[str, ...]
    description: str


# ---------------------------------------------------------------------------
# Canonical observable names
# ---------------------------------------------------------------------------

OBSERVABLE_TYPES: tuple[str, ...] = (
    "cross_section_uu",
    "cross_section_difference_lu",
    "cross_section_uu_virtual_photoproduction",
    "beam_spin_asymmetry",
    "beam_spin_asymmetry_sin2",
    "beam_charge_asymmetry",
    "beam_charge_asymmetry_cos0",
    "beam_charge_asymmetry_cos1",
    "beam_charge_asymmetry_cos2",
    "beam_charge_asymmetry_cos3",
    "target_spin_asymmetry_sin1",
    "target_spin_asymmetry_sin2",
    "double_spin_asymmetry",
    "t_slope",
)

DATABASE_OBSERVABLE_LABELS: tuple[str, ...] = (
    "CrossSectionUU",
    "CrossSectionDifferenceLU",
    "CrossSectionUUVirtualPhotoProduction",
    "AcCos0Phi",
    "AcCos1Phi",
    "AcCos2Phi",
    "AcCos3Phi",
    "ALUIntSin1Phi",
    "ALUDVCSSin1Phi",
    "ALUIntSin2Phi",
    "ALU",
    "ALL",
    "ALUSin1Phi",
    "ALUSin2Phi",
    "AULSin1Phi",
    "AULSin2Phi",
    "Ac",
    "BSA",
    "CA",
    "TSlope",
)

OBSERVABLE_DEFINITIONS: dict[str, ObservableDefinition] = {
    "cross_section_uu": ObservableDefinition(
        canonical_name="cross_section_uu",
        database_labels=("CrossSectionUU",),
        description="Unpolarized DVCS cross section.",
    ),
    "cross_section_difference_lu": ObservableDefinition(
        canonical_name="cross_section_difference_lu",
        database_labels=("CrossSectionDifferenceLU",),
        description="Beam-helicity cross-section difference.",
    ),
    "cross_section_uu_virtual_photoproduction": ObservableDefinition(
        canonical_name="cross_section_uu_virtual_photoproduction",
        database_labels=("CrossSectionUUVirtualPhotoProduction",),
        description="Unpolarized cross section in virtual photoproduction convention.",
    ),
    "beam_spin_asymmetry": ObservableDefinition(
        canonical_name="beam_spin_asymmetry",
        database_labels=("BSA", "ALU", "ALUSin1Phi", "ALUIntSin1Phi", "ALUDVCSSin1Phi"),
        description="Beam-spin asymmetry with leading sin(phi) structure.",
    ),
    "beam_spin_asymmetry_sin2": ObservableDefinition(
        canonical_name="beam_spin_asymmetry_sin2",
        database_labels=("ALUSin2Phi", "ALUIntSin2Phi"),
        description="Beam-spin asymmetry with sin(2phi) structure.",
    ),
    "beam_charge_asymmetry": ObservableDefinition(
        canonical_name="beam_charge_asymmetry",
        database_labels=("Ac", "CA"),
        description="Beam-charge asymmetry with leading cos(phi) structure.",
    ),
    "beam_charge_asymmetry_cos0": ObservableDefinition(
        canonical_name="beam_charge_asymmetry_cos0",
        database_labels=("AcCos0Phi",),
        description="Beam-charge asymmetry with constant harmonic.",
    ),
    "beam_charge_asymmetry_cos1": ObservableDefinition(
        canonical_name="beam_charge_asymmetry_cos1",
        database_labels=("AcCos1Phi",),
        description="Beam-charge asymmetry with cos(phi) structure.",
    ),
    "beam_charge_asymmetry_cos2": ObservableDefinition(
        canonical_name="beam_charge_asymmetry_cos2",
        database_labels=("AcCos2Phi",),
        description="Beam-charge asymmetry with cos(2phi) structure.",
    ),
    "beam_charge_asymmetry_cos3": ObservableDefinition(
        canonical_name="beam_charge_asymmetry_cos3",
        database_labels=("AcCos3Phi",),
        description="Beam-charge asymmetry with cos(3phi) structure.",
    ),
    "target_spin_asymmetry_sin1": ObservableDefinition(
        canonical_name="target_spin_asymmetry_sin1",
        database_labels=("AULSin1Phi",),
        description="Target-spin asymmetry with sin(phi) structure.",
    ),
    "target_spin_asymmetry_sin2": ObservableDefinition(
        canonical_name="target_spin_asymmetry_sin2",
        database_labels=("AULSin2Phi",),
        description="Target-spin asymmetry with sin(2phi) structure.",
    ),
    "double_spin_asymmetry": ObservableDefinition(
        canonical_name="double_spin_asymmetry",
        database_labels=("ALL",),
        description="Longitudinal double-spin asymmetry.",
    ),
    "t_slope": ObservableDefinition(
        canonical_name="t_slope",
        database_labels=("TSlope",),
        description="Effective t-slope observable.",
    ),
}

# Build a robust alias map from database labels and canonical names to canonical names.
OBSERVABLE_LABEL_MAP: dict[str, str] = {}

for canonical_name, definition in OBSERVABLE_DEFINITIONS.items():
    OBSERVABLE_LABEL_MAP[_normalize_label(canonical_name)] = canonical_name
    for label in definition.database_labels:
        OBSERVABLE_LABEL_MAP[_normalize_label(label)] = canonical_name

# Additional common aliases that appear in informal usage.
OBSERVABLE_LABEL_MAP.update(
    {
        _normalize_label("cross_section"): "cross_section_uu",
        _normalize_label("sigma"): "cross_section_uu",
        _normalize_label("unpolarized_cross_section"): "cross_section_uu",
        _normalize_label("beam_spin_asymmetry"): "beam_spin_asymmetry",
        _normalize_label("bsa"): "beam_spin_asymmetry",
        _normalize_label("beam_charge_asymmetry"): "beam_charge_asymmetry",
        _normalize_label("charge_asymmetry"): "beam_charge_asymmetry",
        _normalize_label("ca"): "beam_charge_asymmetry",
        _normalize_label("tslope"): "t_slope",
    }
)


def map_observable_label(label: str) -> str:
    """
    Map a database or informal observable label to a canonical internal name.

    Unknown labels are returned in normalized form.
    """
    normalized = _normalize_label(label)
    return OBSERVABLE_LABEL_MAP.get(normalized, normalized)


def get_all_observable_names() -> list[str]:
    """Return all canonical observable names understood by this module."""
    return list(OBSERVABLE_TYPES)


def get_all_database_labels() -> list[str]:
    """Return all raw database labels known to the mapping layer."""
    return list(DATABASE_OBSERVABLE_LABELS)


def is_supported_observable(label: str) -> bool:
    """Return True if the label can be mapped to a supported observable."""
    return _normalize_label(label) in OBSERVABLE_LABEL_MAP


@dataclass(frozen=True)
class Kinematics:
    """
    Single DVCS kinematic point in canonical convention.

    Conventions:
    - xB in (0, 1)
    - Q2 > 0
    - t <= 0
    - phi in radians, in [0, 2π]
    """

    xB: float
    Q2: float
    t: float
    phi: float
    beam_energy: Optional[float] = None

    def __post_init__(self) -> None:
        _validate_scalar_kinematics(self.xB, self.Q2, self.t, self.phi)
        if self.beam_energy is not None:
            if not np.isfinite(self.beam_energy):
                raise ValueError("beam_energy must be finite if provided.")
            if self.beam_energy <= 0.0:
                raise ValueError("beam_energy must be positive if provided.")


@dataclass(frozen=True)
class KinematicsBatch:
    """
    Vectorized kinematic container.

    All fields must be 1D arrays of identical length.
    """

    xB: np.ndarray
    Q2: np.ndarray
    t: np.ndarray
    phi: np.ndarray
    beam_energy: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        xB = np.asarray(self.xB, dtype=float)
        Q2 = np.asarray(self.Q2, dtype=float)
        t = np.asarray(self.t, dtype=float)
        phi = np.asarray(self.phi, dtype=float)

        if xB.ndim != 1 or Q2.ndim != 1 or t.ndim != 1 or phi.ndim != 1:
            raise ValueError("xB, Q2, t, and phi must all be 1D arrays.")
        if not (xB.shape == Q2.shape == t.shape == phi.shape):
            raise ValueError("xB, Q2, t, and phi must have identical shapes.")
        if not np.all(np.isfinite(xB)):
            raise ValueError("xB contains non-finite values.")
        if not np.all(np.isfinite(Q2)):
            raise ValueError("Q2 contains non-finite values.")
        if not np.all(np.isfinite(t)):
            raise ValueError("t contains non-finite values.")
        if not np.all(np.isfinite(phi)):
            raise ValueError("phi contains non-finite values.")

        _validate_vector_kinematics(xB, Q2, t, phi)

        be = None
        if self.beam_energy is not None:
            be = np.asarray(self.beam_energy, dtype=float)
            if be.ndim != 1:
                raise ValueError("beam_energy must be a 1D array if provided.")
            if be.shape != xB.shape:
                raise ValueError("beam_energy must have the same shape as xB.")
            if not np.all(np.isfinite(be)):
                raise ValueError("beam_energy contains non-finite values.")
            if np.any(be <= 0.0):
                raise ValueError("beam_energy must be positive if provided.")

        object.__setattr__(self, "xB", xB)
        object.__setattr__(self, "Q2", Q2)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "beam_energy", be)

    @classmethod
    def from_sequences(
        cls,
        xB: Sequence[float],
        Q2: Sequence[float],
        t: Sequence[float],
        phi: Sequence[float],
        beam_energy: Optional[Sequence[float]] = None,
    ) -> "KinematicsBatch":
        return cls(
            xB=np.asarray(xB, dtype=float),
            Q2=np.asarray(Q2, dtype=float),
            t=np.asarray(t, dtype=float),
            phi=np.asarray(phi, dtype=float),
            beam_energy=None if beam_energy is None else np.asarray(beam_energy, dtype=float),
        )

    @property
    def n_points(self) -> int:
        return int(self.xB.shape[0])

    def __len__(self) -> int:
        return self.n_points


@dataclass(frozen=True)
class ToyCFFParameters:
    """
    Parameters controlling the toy CFF generator.

    These are not meant to be physical constants.
    They are chosen to produce smooth but nontrivial kinematic dependence.
    """

    h_real_0: float = 1.00
    h_imag_0: float = 0.65
    e_real_0: float = 0.28
    e_imag_0: float = 0.18

    x_power: float = 1.25
    q2_slope: float = 0.18
    t_slope: float = 2.5

    h_real_xi: float = 0.30
    h_imag_xi: float = 0.22
    e_real_xi: float = 0.12
    e_imag_xi: float = 0.08

    h_real_t: float = 0.50
    h_imag_t: float = 0.35
    e_real_t: float = 0.18
    e_imag_t: float = 0.15

    scale: float = 1.0


def _validate_scalar_kinematics(xB: float, Q2: float, t: float, phi: float) -> None:
    if not np.isfinite(xB) or not np.isfinite(Q2) or not np.isfinite(t) or not np.isfinite(phi):
        raise ValueError("Kinematics must be finite.")
    if not (0.0 < xB < 1.0):
        raise ValueError("xB must be in (0, 1).")
    if Q2 <= 0.0:
        raise ValueError("Q2 must be strictly positive.")
    if t > 0.0:
        raise ValueError("t must be non-positive in the canonical convention.")
    if not (0.0 <= phi <= _TWO_PI):
        raise ValueError("phi must be in radians and lie in [0, 2π].")


def _validate_vector_kinematics(xB: np.ndarray, Q2: np.ndarray, t: np.ndarray, phi: np.ndarray) -> None:
    if np.any((xB <= 0.0) | (xB >= 1.0)):
        raise ValueError("All xB values must be in (0, 1).")
    if np.any(Q2 <= 0.0):
        raise ValueError("All Q2 values must be strictly positive.")
    if np.any(t > 0.0):
        raise ValueError("All t values must be non-positive in the canonical convention.")
    if np.any((phi < 0.0) | (phi > _TWO_PI)):
        raise ValueError("All phi values must lie in [0, 2π].")


def xi_from_xB(xB: np.ndarray) -> np.ndarray:
    """
    Convert xB to the usual DVCS skewness proxy ξ ≈ xB / (2 - xB).
    """
    xB = np.asarray(xB, dtype=float)
    return xB / np.clip(2.0 - xB, _EPS, None)


def generate_toy_cffs(
    kinematics: KinematicsBatch,
    params: ToyCFFParameters = ToyCFFParameters(),
) -> Dict[str, np.ndarray]:
    """
    Generate smooth, kinematics-dependent toy CFFs.

    The CFFs depend on (xB, Q2, t) but not on φ. This is intentional:
    φ-dependence should enter the observables, not the CFFs.
    """
    xB = kinematics.xB
    Q2 = kinematics.Q2
    t = kinematics.t

    xi = xi_from_xB(xB)
    x_shape = np.power(1.0 - xB, params.x_power)
    q_shape = 1.0 + params.q2_slope * np.log1p(Q2)
    t_shape = np.exp(params.t_slope * t)  # t <= 0, so this decays smoothly

    base = params.scale * x_shape * q_shape * t_shape

    H_real = base * (
        params.h_real_0
        + params.h_real_xi * xi
        + params.h_real_t * (-t)
    )
    H_imag = base * (
        params.h_imag_0
        + params.h_imag_xi * np.sqrt(np.clip(xi, 0.0, None) + _EPS)
        + params.h_imag_t * (-t)
    )
    E_real = base * (
        params.e_real_0
        + params.e_real_xi * xi
        + params.e_real_t * (-t)
    )
    E_imag = base * (
        params.e_imag_0
        + params.e_imag_xi * np.sqrt(np.clip(xi, 0.0, None) + _EPS)
        + params.e_imag_t * (-t)
    )

    return {
        "H_real": H_real,
        "H_imag": H_imag,
        "E_real": E_real,
        "E_imag": E_imag,
    }


def scale_toy_cffs(
    cffs: Mapping[str, np.ndarray],
    *,
    h_real_scale: float = 1.0,
    h_imag_scale: float = 1.0,
    e_real_scale: float = 1.0,
    e_imag_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Return a scaled copy of a toy CFF dictionary.
    """
    return {
        "H_real": h_real_scale * np.asarray(cffs["H_real"], dtype=float),
        "H_imag": h_imag_scale * np.asarray(cffs["H_imag"], dtype=float),
        "E_real": e_real_scale * np.asarray(cffs["E_real"], dtype=float),
        "E_imag": e_imag_scale * np.asarray(cffs["E_imag"], dtype=float),
    }


class ObservableCalculator:
    """
    Abstract observable calculator interface.
    """

    def compute(
        self,
        observable_name: str,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_observable(
        self,
        observable_name: str,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Convenience alias for compute().
        """
        return self.compute(observable_name, kinematics, cffs)


class ToyDVCSObservableCalculator(ObservableCalculator):
    """
    Toy DVCS observable calculator.

    backend:
        - "toy": fully implemented toy model
        - "partons": reserved stub for later integration
    """

    def __init__(self, backend: str = "toy") -> None:
        backend = backend.lower().strip()
        if backend not in {"toy", "partons"}:
            raise ValueError("backend must be either 'toy' or 'partons'.")
        self.backend = backend

    def compute(
        self,
        observable_name: str,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Dispatch to the requested observable.

        The input may be a canonical name or a database label.
        """
        normalized = _normalize_label(observable_name)

        if self.backend == "partons":
            raise NotImplementedError(
                "PARTONS backend is not wired in yet. Use backend='toy' for validation."
            )

        if normalized in {
            "cross_section_uu",
            "cross_section",
            "sigma",
            "unpolarized_cross_section",
        }:
            return self.compute_cross_section_uu(kinematics, cffs)

        if normalized in {"cross_section_difference_lu"}:
            return self.compute_cross_section_difference_lu(kinematics, cffs)

        if normalized in {"cross_section_uu_virtual_photoproduction"}:
            return self.compute_cross_section_uu_virtual_photoproduction(kinematics, cffs)

        if normalized in {
            "beam_spin_asymmetry",
            "bsa",
            "alu",
            "alusin1phi",
            "aluintsin1phi",
            "aludvcssin1phi",
        }:
            return self.compute_beam_spin_asymmetry(kinematics, cffs, harmonic_order=1)

        if normalized in {"beam_spin_asymmetry_sin2", "alusin2phi", "aluintsin2phi"}:
            return self.compute_beam_spin_asymmetry(kinematics, cffs, harmonic_order=2)

        if normalized in {
            "beam_charge_asymmetry",
            "charge_asymmetry",
            "ca",
            "ac",
        }:
            return self.compute_beam_charge_asymmetry(kinematics, cffs, harmonic_order=1)

        if normalized == "beam_charge_asymmetry_cos0" or normalized == "accos0phi":
            return self.compute_beam_charge_asymmetry(kinematics, cffs, harmonic_order=0)

        if normalized == "beam_charge_asymmetry_cos1" or normalized == "accos1phi":
            return self.compute_beam_charge_asymmetry(kinematics, cffs, harmonic_order=1)

        if normalized == "beam_charge_asymmetry_cos2" or normalized == "accos2phi":
            return self.compute_beam_charge_asymmetry(kinematics, cffs, harmonic_order=2)

        if normalized == "beam_charge_asymmetry_cos3" or normalized == "accos3phi":
            return self.compute_beam_charge_asymmetry(kinematics, cffs, harmonic_order=3)

        if normalized in {"target_spin_asymmetry_sin1", "aulsin1phi"}:
            return self.compute_target_spin_asymmetry(kinematics, cffs, harmonic_order=1)

        if normalized in {"target_spin_asymmetry_sin2", "aulsin2phi"}:
            return self.compute_target_spin_asymmetry(kinematics, cffs, harmonic_order=2)

        if normalized in {"double_spin_asymmetry", "all"}:
            return self.compute_double_spin_asymmetry(kinematics, cffs)

        if normalized in {"t_slope", "tslope"}:
            return self.compute_t_slope(kinematics, cffs)

        raise ValueError(f"Unknown observable_name='{observable_name}'.")

    def compute_all(
        self,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Compute the core toy observable set at once.
        """
        return {
            "cross_section_uu": self.compute_cross_section_uu(kinematics, cffs),
            "cross_section_difference_lu": self.compute_cross_section_difference_lu(kinematics, cffs),
            "cross_section_uu_virtual_photoproduction": self.compute_cross_section_uu_virtual_photoproduction(
                kinematics, cffs
            ),
            "beam_spin_asymmetry": self.compute_beam_spin_asymmetry(kinematics, cffs, harmonic_order=1),
            "beam_charge_asymmetry": self.compute_beam_charge_asymmetry(kinematics, cffs, harmonic_order=1),
            "double_spin_asymmetry": self.compute_double_spin_asymmetry(kinematics, cffs),
            "t_slope": self.compute_t_slope(kinematics, cffs),
        }

    def compute_cross_section_uu(
        self,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Toy unpolarized cross section.

        It is designed to be:
        - positive
        - smooth in kinematics
        - nonlinear in CFFs
        - sensitive to H and E simultaneously
        """
        h_r, h_i, e_r, e_i = self._unpack_cffs(kinematics, cffs)
        phi = kinematics.phi
        xB = kinematics.xB
        Q2 = kinematics.Q2
        t = kinematics.t

        xi = xi_from_xB(xB)
        kin_scale = np.sqrt(np.clip((1.0 - xB) * (1.0 + (-t) / Q2), 0.0, None))
        acc = 1.0 + 0.18 * np.cos(phi) + 0.05 * np.cos(2.0 * phi)
        norm = 1.0 + 0.12 * np.log1p(Q2) + 0.04 * (-t) + 0.03 * xi

        amp = (
            0.35 * (h_r**2 + h_i**2)
            + 0.12 * (e_r**2 + e_i**2)
            + 0.06 * (h_r * e_r + h_i * e_i)
            + 0.08 * kin_scale * h_r
            + 0.05 * kin_scale * e_r
            + 0.02 * np.cos(phi) * h_i
            + 0.01 * np.sin(phi) * e_i
        )

        sigma = norm * acc * (1.0 + amp)
        return np.clip(sigma, 1e-10, None)

    def compute_cross_section(
        self,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Backward-compatible alias for historical tests/scripts.

        Prefer compute_cross_section_uu() in new code.
        """
        return self.compute_cross_section_uu(kinematics, cffs)

    def compute_cross_section_difference_lu(
        self,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Toy beam-helicity cross-section difference.

        This is generated as the cross section times a helicity asymmetry-like
        factor so that it carries the right dimensional behavior.
        """
        sigma = self.compute_cross_section_uu(kinematics, cffs)
        asym = self.compute_beam_spin_asymmetry(kinematics, cffs, harmonic_order=1)
        return sigma * asym

    def compute_cross_section_uu_virtual_photoproduction(
        self,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Toy virtual photoproduction cross section.

        Uses the same underlying structure as the unpolarized cross section but
        with an additional mild Q2 suppression.
        """
        sigma = self.compute_cross_section_uu(kinematics, cffs)
        q2 = kinematics.Q2
        return sigma * (1.0 + 0.12 / (1.0 + q2))

    def compute_beam_spin_asymmetry(
        self,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
        harmonic_order: int = 1,
    ) -> np.ndarray:
        """
        Toy beam-spin asymmetry.

        Designed to be sensitive mainly to imaginary parts and φ-structure.

        Parameters
        ----------
        harmonic_order:
            1 gives a leading sin(phi)-like modulation.
            2 gives a sin(2phi)-like modulation.
        """
        if harmonic_order not in {1, 2}:
            raise ValueError("harmonic_order for beam spin asymmetry must be 1 or 2.")

        h_r, h_i, e_r, e_i = self._unpack_cffs(kinematics, cffs)
        phi = kinematics.phi
        xB = kinematics.xB
        Q2 = kinematics.Q2
        t = kinematics.t

        xi = xi_from_xB(xB)
        kin_scale = np.sqrt(np.clip((1.0 - xB) * (1.0 + (-t) / Q2), 0.0, None))

        if harmonic_order == 1:
            sin_harm = np.sin(phi)
            second_harm = np.sin(2.0 * phi)
            numerator = (
                0.60 * h_i * sin_harm
                + 0.14 * e_i * sin_harm
                + 0.05 * (h_r * e_i - h_i * e_r) * np.cos(phi)
                + 0.02 * kin_scale * second_harm * (h_i + 0.5 * e_i)
            )
        else:
            sin_harm = np.sin(2.0 * phi)
            numerator = (
                0.42 * h_i * sin_harm
                + 0.11 * e_i * sin_harm
                + 0.03 * kin_scale * np.sin(3.0 * phi) * (h_i + 0.35 * e_i)
                + 0.02 * (h_r * e_i - h_i * e_r) * np.cos(2.0 * phi)
            )

        denom = (
            1.0
            + 0.25 * (h_r**2 + h_i**2)
            + 0.10 * (e_r**2 + e_i**2)
            + 0.05 * xi
            + 0.03 * np.abs(t)
            + 0.02 * np.log1p(Q2)
        )
        bsa = numerator / np.clip(denom, _EPS, None)
        return np.clip(bsa, -0.95, 0.95)

    def compute_beam_charge_asymmetry(
        self,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
        harmonic_order: int = 1,
    ) -> np.ndarray:
        """
        Toy beam-charge asymmetry.

        Designed to be sensitive mainly to real parts and cos(phi)-like structure.
        """
        if harmonic_order not in {0, 1, 2, 3}:
            raise ValueError("harmonic_order for beam charge asymmetry must be 0, 1, 2, or 3.")

        h_r, h_i, e_r, e_i = self._unpack_cffs(kinematics, cffs)
        phi = kinematics.phi
        xB = kinematics.xB
        Q2 = kinematics.Q2
        t = kinematics.t

        xi = xi_from_xB(xB)
        kin_scale = np.sqrt(np.clip((1.0 - xB) * (1.0 + (-t) / Q2), 0.0, None))

        if harmonic_order == 0:
            cos_harm = np.ones_like(phi)
        else:
            cos_harm = np.cos(harmonic_order * phi)

        numerator = (
            0.55 * h_r * cos_harm
            + 0.16 * e_r * cos_harm
            + 0.04 * (h_r * e_i + h_i * e_r) * np.sin(phi)
            + 0.03 * kin_scale * np.cos(2.0 * phi) * (h_r + 0.5 * e_r)
        )
        denom = (
            1.0
            + 0.20 * (h_r**2 + h_i**2)
            + 0.08 * (e_r**2 + e_i**2)
            + 0.03 * xi
            + 0.02 * np.abs(t)
            + 0.02 * np.log1p(Q2)
        )
        bca = numerator / np.clip(denom, _EPS, None)
        return np.clip(bca, -0.95, 0.95)

    def compute_target_spin_asymmetry(
        self,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
        harmonic_order: int = 1,
    ) -> np.ndarray:
        """
        Toy target-spin asymmetry.

        This uses a mix of imaginary and real parts and sin(phi)-type harmonics.
        """
        if harmonic_order not in {1, 2}:
            raise ValueError("harmonic_order for target spin asymmetry must be 1 or 2.")

        h_r, h_i, e_r, e_i = self._unpack_cffs(kinematics, cffs)
        phi = kinematics.phi
        xB = kinematics.xB
        Q2 = kinematics.Q2
        t = kinematics.t

        xi = xi_from_xB(xB)
        kin_scale = np.sqrt(np.clip((1.0 - xB) * (1.0 + (-t) / Q2), 0.0, None))

        if harmonic_order == 1:
            harmonic = np.sin(phi)
            numerator = (
                0.46 * e_i * harmonic
                + 0.18 * e_r * harmonic
                + 0.06 * h_i * np.cos(phi)
                + 0.03 * kin_scale * np.sin(2.0 * phi) * (e_i + 0.5 * h_i)
            )
        else:
            harmonic = np.sin(2.0 * phi)
            numerator = (
                0.34 * e_i * harmonic
                + 0.14 * e_r * harmonic
                + 0.04 * h_i * np.cos(2.0 * phi)
                + 0.02 * kin_scale * np.sin(3.0 * phi) * (e_i + 0.35 * h_i)
            )

        denom = (
            1.0
            + 0.18 * (h_r**2 + h_i**2)
            + 0.12 * (e_r**2 + e_i**2)
            + 0.03 * xi
            + 0.02 * np.abs(t)
            + 0.02 * np.log1p(Q2)
        )
        asym = numerator / np.clip(denom, _EPS, None)
        return np.clip(asym, -0.95, 0.95)

    def compute_double_spin_asymmetry(
        self,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Toy longitudinal double-spin asymmetry.

        Designed to depend on both real and imaginary parts.
        """
        h_r, h_i, e_r, e_i = self._unpack_cffs(kinematics, cffs)
        phi = kinematics.phi
        xB = kinematics.xB
        Q2 = kinematics.Q2
        t = kinematics.t

        xi = xi_from_xB(xB)
        kin_scale = np.sqrt(np.clip((1.0 - xB) * (1.0 + (-t) / Q2), 0.0, None))

        numerator = (
            0.32 * h_i * np.sin(phi)
            + 0.10 * h_r * np.cos(phi)
            + 0.08 * e_i * np.sin(phi)
            + 0.05 * e_r * np.cos(phi)
            + 0.03 * kin_scale * np.sin(2.0 * phi) * (h_i + 0.5 * e_i)
        )
        denom = (
            1.0
            + 0.16 * (h_r**2 + h_i**2)
            + 0.09 * (e_r**2 + e_i**2)
            + 0.03 * xi
            + 0.02 * np.abs(t)
            + 0.02 * np.log1p(Q2)
        )
        asym = numerator / np.clip(denom, _EPS, None)
        return np.clip(asym, -0.95, 0.95)

    def compute_t_slope(
        self,
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Toy effective t-slope observable.

        Returns a positive slope-like quantity.
        """
        h_r, _h_i, e_r, _e_i = self._unpack_cffs(kinematics, cffs)
        xB = kinematics.xB
        Q2 = kinematics.Q2
        t = kinematics.t

        xi = xi_from_xB(xB)
        slope = (
            0.80
            + 0.18 * np.log1p(Q2)
            + 0.35 * (-t)
            + 0.08 * xi
            + 0.03 * np.abs(h_r)
            + 0.02 * np.abs(e_r)
        )
        return np.clip(slope, 1e-10, None)

    @staticmethod
    def _unpack_cffs(
        kinematics: KinematicsBatch,
        cffs: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = kinematics.n_points
        required = ("H_real", "H_imag", "E_real", "E_imag")
        missing = [k for k in required if k not in cffs]
        if missing:
            raise ValueError(f"Missing required CFF keys: {missing}")

        h_r = np.asarray(cffs["H_real"], dtype=float)
        h_i = np.asarray(cffs["H_imag"], dtype=float)
        e_r = np.asarray(cffs["E_real"], dtype=float)
        e_i = np.asarray(cffs["E_imag"], dtype=float)

        for name, arr in {
            "H_real": h_r,
            "H_imag": h_i,
            "E_real": e_r,
            "E_imag": e_i,
        }.items():
            if arr.ndim != 1:
                raise ValueError(f"{name} must be a 1D array.")
            if arr.shape != (n,):
                raise ValueError(f"{name} must have the same length as the kinematics batch.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values.")

        return h_r, h_i, e_r, e_i


# ---------------------------------------------------------------------------
# Torch observable layer for end-to-end differentiable training
# ---------------------------------------------------------------------------

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - torch-less environments.
    torch = None

    class _NNFallback:
        class Module:
            pass

    nn = _NNFallback()


TORCH_OBSERVABLE_INDEX: dict[str, int] = {
    "cross_section_uu": 0,
    "cross_section_difference_lu": 1,
    "beam_spin_asymmetry": 2,
    "beam_charge_asymmetry": 3,
    "double_spin_asymmetry": 4,
}


def observable_name_to_index(name: str) -> int:
    """Map observable name/label to a compact integer index."""
    canonical = map_observable_label(name)
    if canonical not in TORCH_OBSERVABLE_INDEX:
        raise ValueError(f"Unsupported observable for torch layer: {name}")
    return TORCH_OBSERVABLE_INDEX[canonical]


class TorchDVCSObservableLayer(nn.Module):
    """
    Differentiable DVCS observable layer based on CFF inputs.

    Inputs
    ------
    cff_stacked:
        Tensor with shape [B, 4, 2], where channels are [H, E, Htilde, Etilde]
        and the last dimension is [real, imag].
    kinematics:
        Tensor with shape [B, 5] in the convention [xB, xi, t, Q2, phi].
    """

    def __init__(self) -> None:
        if torch is None:
            raise ModuleNotFoundError("TorchDVCSObservableLayer requires torch to be installed.")
        super().__init__()

    @staticmethod
    def _split_cffs(cff_stacked: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if cff_stacked.ndim != 3 or cff_stacked.shape[1:] != (4, 2):
            raise ValueError("cff_stacked must have shape [B, 4, 2].")
        h_r = cff_stacked[:, 0, 0]
        h_i = cff_stacked[:, 0, 1]
        e_r = cff_stacked[:, 1, 0]
        e_i = cff_stacked[:, 1, 1]
        ht_r = cff_stacked[:, 2, 0]
        ht_i = cff_stacked[:, 2, 1]
        et_r = cff_stacked[:, 3, 0]
        et_i = cff_stacked[:, 3, 1]
        return h_r, h_i, e_r, e_i, ht_r, ht_i, et_r, et_i

    @staticmethod
    def _split_kinematics(kinematics: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if kinematics.ndim != 2 or kinematics.shape[1] < 5:
            raise ValueError("kinematics must have shape [B, 5] with [xB, xi, t, Q2, phi].")
        x_b = kinematics[:, 0]
        xi = kinematics[:, 1]
        t = kinematics[:, 2]
        q2 = kinematics[:, 3]
        phi = kinematics[:, 4]
        return x_b, xi, t, q2, phi

    def compute_cross_section_uu(self, cff_stacked: torch.Tensor, kinematics: torch.Tensor) -> torch.Tensor:
        h_r, h_i, e_r, e_i, ht_r, ht_i, _et_r, _et_i = self._split_cffs(cff_stacked)
        _x_b, xi, t, q2, phi = self._split_kinematics(kinematics)

        mag_h = h_r * h_r + h_i * h_i
        mag_e = e_r * e_r + e_i * e_i
        mag_ht = ht_r * ht_r + ht_i * ht_i

        kinematic_prefactor = 1.0 + 0.14 * torch.log1p(torch.clamp(q2, min=1e-8)) + 0.05 * torch.abs(t) + 0.04 * xi
        harmonics = 1.0 + 0.20 * torch.cos(phi) + 0.05 * torch.cos(2.0 * phi)
        cff_term = 0.40 * mag_h + 0.20 * mag_e + 0.10 * mag_ht + 0.08 * (h_r * e_r + h_i * e_i)
        sigma = kinematic_prefactor * harmonics * (1.0 + cff_term)
        return torch.clamp(sigma, min=1e-8)

    def compute_cross_section_difference_lu(
        self,
        cff_stacked: torch.Tensor,
        kinematics: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self.compute_cross_section_uu(cff_stacked, kinematics)
        asym = self.compute_beam_spin_asymmetry(cff_stacked, kinematics)
        return sigma * asym

    def compute_beam_spin_asymmetry(self, cff_stacked: torch.Tensor, kinematics: torch.Tensor) -> torch.Tensor:
        h_r, h_i, e_r, e_i, _ht_r, ht_i, _et_r, et_i = self._split_cffs(cff_stacked)
        _x_b, xi, t, q2, phi = self._split_kinematics(kinematics)

        numerator = (
            (0.72 * h_i + 0.18 * e_i) * torch.sin(phi)
            + 0.08 * ht_i * torch.sin(2.0 * phi)
            + 0.03 * et_i * torch.sin(2.0 * phi)
            + 0.04 * (h_r * e_i - h_i * e_r) * torch.cos(phi)
        )
        denominator = (
            1.0
            + 0.30 * (h_r * h_r + h_i * h_i)
            + 0.14 * (e_r * e_r + e_i * e_i)
            + 0.04 * xi
            + 0.03 * torch.abs(t)
            + 0.02 * torch.log1p(torch.clamp(q2, min=1e-8))
        )
        asym = numerator / torch.clamp(denominator, min=1e-8)
        return torch.clamp(asym, min=-0.99, max=0.99)

    def compute_beam_charge_asymmetry(self, cff_stacked: torch.Tensor, kinematics: torch.Tensor) -> torch.Tensor:
        h_r, h_i, e_r, e_i, ht_r, _ht_i, et_r, _et_i = self._split_cffs(cff_stacked)
        _x_b, xi, t, q2, phi = self._split_kinematics(kinematics)

        numerator = (
            (0.68 * h_r + 0.20 * e_r) * torch.cos(phi)
            + 0.06 * ht_r * torch.cos(2.0 * phi)
            + 0.03 * et_r * torch.cos(2.0 * phi)
            + 0.03 * (h_r * e_i + h_i * e_r) * torch.sin(phi)
        )
        denominator = (
            1.0
            + 0.26 * (h_r * h_r + h_i * h_i)
            + 0.12 * (e_r * e_r + e_i * e_i)
            + 0.05 * (ht_r * ht_r)
            + 0.04 * xi
            + 0.02 * torch.abs(t)
            + 0.02 * torch.log1p(torch.clamp(q2, min=1e-8))
        )
        asym = numerator / torch.clamp(denominator, min=1e-8)
        return torch.clamp(asym, min=-0.99, max=0.99)

    def compute_double_spin_asymmetry(self, cff_stacked: torch.Tensor, kinematics: torch.Tensor) -> torch.Tensor:
        h_r, h_i, e_r, e_i, ht_r, ht_i, _et_r, _et_i = self._split_cffs(cff_stacked)
        _x_b, xi, t, q2, phi = self._split_kinematics(kinematics)

        numerator = (
            0.30 * h_i * torch.sin(phi)
            + 0.12 * h_r * torch.cos(phi)
            + 0.08 * e_i * torch.sin(phi)
            + 0.04 * e_r * torch.cos(phi)
            + 0.06 * ht_i * torch.sin(2.0 * phi)
            + 0.04 * ht_r * torch.cos(2.0 * phi)
        )
        denominator = (
            1.0
            + 0.24 * (h_r * h_r + h_i * h_i)
            + 0.12 * (e_r * e_r + e_i * e_i)
            + 0.08 * (ht_r * ht_r + ht_i * ht_i)
            + 0.04 * xi
            + 0.03 * torch.abs(t)
            + 0.02 * torch.log1p(torch.clamp(q2, min=1e-8))
        )
        asym = numerator / torch.clamp(denominator, min=1e-8)
        return torch.clamp(asym, min=-0.99, max=0.99)

    def compute_all(self, cff_stacked: torch.Tensor, kinematics: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the main observables used in training and diagnostics."""
        return {
            "cross_section_uu": self.compute_cross_section_uu(cff_stacked, kinematics),
            "cross_section_difference_lu": self.compute_cross_section_difference_lu(cff_stacked, kinematics),
            "beam_spin_asymmetry": self.compute_beam_spin_asymmetry(cff_stacked, kinematics),
            "beam_charge_asymmetry": self.compute_beam_charge_asymmetry(cff_stacked, kinematics),
            "double_spin_asymmetry": self.compute_double_spin_asymmetry(cff_stacked, kinematics),
        }

    def forward(
        self,
        cff_stacked: torch.Tensor,
        kinematics: torch.Tensor,
        observable_id: torch.Tensor,
    ) -> torch.Tensor:
        """Select predicted values for per-row observable identifiers."""
        if observable_id.ndim != 1:
            raise ValueError("observable_id must be a 1D tensor.")
        if observable_id.shape[0] != cff_stacked.shape[0]:
            raise ValueError("observable_id length must match batch size.")

        all_obs = self.compute_all(cff_stacked, kinematics)
        out = torch.zeros(
            cff_stacked.shape[0],
            device=cff_stacked.device,
            dtype=cff_stacked.dtype,
        )

        for name, idx in TORCH_OBSERVABLE_INDEX.items():
            mask = observable_id == idx
            if torch.any(mask):
                out[mask] = all_obs[name][mask]

        max_known = max(TORCH_OBSERVABLE_INDEX.values())
        if torch.any((observable_id < 0) | (observable_id > max_known)):
            raise ValueError("observable_id contains unsupported values.")

        return out

        return h_r, h_i, e_r, e_i