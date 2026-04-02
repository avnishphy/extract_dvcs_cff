import pytest
from extract_dvcs_cff.data.schemas import KinematicPoint, ObservableRecord, DatasetRecord

def test_kinematic_point_valid():
    kp = KinematicPoint(xB=0.2, Q2=2.0, t=-0.3, phi=45.0)
    assert kp.xB == 0.2
    assert kp.Q2 == 2.0
    assert kp.t == -0.3
    assert kp.phi == 45.0

def test_kinematic_point_invalid():
    import pytest
    with pytest.raises(ValueError):
        KinematicPoint(xB=-0.1, Q2=2.0, t=-0.3, phi=45.0)
    with pytest.raises(ValueError):
        KinematicPoint(xB=0.2, Q2=-1.0, t=-0.3, phi=45.0)
    with pytest.raises(ValueError):
        KinematicPoint(xB=0.2, Q2=2.0, t=-0.3, phi=400.0)
