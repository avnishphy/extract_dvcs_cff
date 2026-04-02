import pytest
from extract_dvcs_cff.data.validation import validate_kinematic_point
from extract_dvcs_cff.data.schemas import KinematicPoint

def test_validate_kinematic_point_valid():
    kp = KinematicPoint(xB=0.3, Q2=1.5, t=-0.2, phi=90.0)
    validate_kinematic_point(kp)

def test_validate_kinematic_point_invalid():
    with pytest.raises(ValueError):
        KinematicPoint(xB=0.3, Q2=1.5, t=float('nan'), phi=90.0)
