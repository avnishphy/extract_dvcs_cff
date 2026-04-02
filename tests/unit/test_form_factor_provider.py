import pytest

torch = pytest.importorskip("torch")

from extract_dvcs_cff.physics.constraints import DipoleFormFactorProvider, TabulatedFormFactorProvider


def test_dipole_form_factor_provider_positive():
    provider = DipoleFormFactorProvider()
    t = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32)

    f1 = provider.f1(t)
    f2 = provider.f2(t)

    assert torch.all(f1 > 0)
    assert torch.all(f2 > 0)


def test_tabulated_form_factor_interpolation():
    t_points = torch.tensor([-0.4, -0.2, 0.0], dtype=torch.float32)
    f1_points = torch.tensor([0.7, 0.85, 1.0], dtype=torch.float32)
    f2_points = torch.tensor([1.2, 1.4, 1.8], dtype=torch.float32)

    provider = TabulatedFormFactorProvider(t_points=t_points, f1_points=f1_points, f2_points=f2_points)
    t_query = torch.tensor([-0.3, -0.1], dtype=torch.float32)

    f1 = provider.f1(t_query)
    f2 = provider.f2(t_query)

    assert f1.shape == t_query.shape
    assert f2.shape == t_query.shape
    assert torch.isfinite(f1).all()
    assert torch.isfinite(f2).all()
