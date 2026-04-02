"""
Kinematics conventions and helpers for DVCS/GPD/CFF analysis.
"""
from typing import Dict
import numpy as np


# Canonical conventions for kinematic variables in DVCS/GPD datasets
# This list is derived from the OpenGPD database summary and covers all variables present in the datasets.
KINEMATIC_LABELS = [
    "xB",         # Bjorken x
    "Q2",         # Photon virtuality (GeV^2)
    "t",          # Momentum transfer (GeV^2)
    "phi",        # Azimuthal angle (deg)
    "xi",         # Skewness parameter
    "mu2",        # Factorization/renormalization scale (GeV^2)
    "nu",         # Energy transfer (GeV)
    "W",          # Invariant mass (GeV)
    "beam_energy",# Beam energy (GeV)
    "y",          # Inelasticity
]

# Dictionary of kinematic variable descriptions for documentation and validation
KINEMATIC_DESCRIPTIONS = {
    "xB": "Bjorken x (0 < xB < 1)",
    "Q2": "Photon virtuality Q^2 (GeV^2)",
    "t": "Momentum transfer t (GeV^2, typically negative)",
    "phi": "Azimuthal angle phi (deg, 0-360)",
    "xi": "Skewness parameter (0 < xi < 1)",
    "mu2": "Factorization/renormalization scale mu^2 (GeV^2)",
    "nu": "Energy transfer nu (GeV)",
    "W": "Invariant mass W (GeV)",
    "beam_energy": "Beam energy (GeV)",
    "y": "Inelasticity (0 < y < 1)",
}

# Sign conventions for t: negative definite (t < 0) or absolute value
T_SIGN_CONVENTION = "negative"  # or "absolute"

# Helper functions for kinematic region classification and conventions
def is_dglap_region(x: float, xi: float) -> bool:
    """
    Return True if x > xi (DGLAP region).
    DGLAP: quark/antiquark emission/absorption, x > xi.
    """
    return x > xi

def is_erbl_region(x: float, xi: float) -> bool:
    """
    Return True if |x| < xi (ERBL region).
    ERBL: quark-antiquark pair creation/annihilation, |x| < xi.
    """
    return abs(x) < xi

def convert_t_sign(t: float, convention: str = T_SIGN_CONVENTION) -> float:
    """
    Convert t to the chosen sign convention.
    By default, t is negative definite in most DVCS conventions.
    """
    if convention == "negative":
        return -abs(t)
    elif convention == "absolute":
        return abs(t)
    else:
        raise ValueError(f"Unknown t sign convention: {convention}")

# Utility: get allowed range for a kinematic variable from a phase space dictionary
def get_kinematic_range(var: str, phase_space: dict) -> tuple:
    """
    Return (min, max) for a given kinematic variable from a phase space dictionary.
    """
    return phase_space.get(var, (None, None))
