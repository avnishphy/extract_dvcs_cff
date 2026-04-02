import pytest

torch = pytest.importorskip("torch")

from extract_dvcs_cff.physics.constraints import positivity_penalty


def test_positivity_penalty_zero_when_positive():
    gpd = torch.tensor([0.1, 0.2, 0.3])
    penalty = positivity_penalty(gpd)
    assert torch.isclose(penalty, torch.tensor(0.0))


def test_positivity_penalty_positive_when_negative_values_present():
    gpd = torch.tensor([0.1, -0.2, 0.3])
    penalty = positivity_penalty(gpd)
    assert penalty > 0
