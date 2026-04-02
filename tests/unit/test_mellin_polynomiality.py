import pytest

torch = pytest.importorskip("torch")

from extract_dvcs_cff.physics.constraints import mellin_moment, polynomiality_penalty


def test_mellin_moment_known_function():
    x = torch.linspace(0.0, 1.0, 200)
    gpd = x**2

    m0 = mellin_moment(gpd, x, order=0)
    expected = torch.tensor(1.0 / 3.0)
    assert torch.isclose(m0, expected, atol=1e-3)


def test_polynomiality_penalty_small_for_polynomial_moments():
    xi = torch.linspace(0.05, 0.35, 8)
    x = torch.linspace(0.0, 1.0, 160)

    # Construct xi-dependent GPD with polynomial xi structure.
    coeff0 = 1.0 + 0.5 * xi + 0.2 * xi**2
    coeff1 = 0.3 + 0.1 * xi
    gpd = coeff0.unsqueeze(-1) * (1.0 - x.unsqueeze(0)) + coeff1.unsqueeze(-1) * x.unsqueeze(0)

    penalty = polynomiality_penalty(
        gpd_xi_grid=gpd,
        x_grid=x,
        xi_grid=xi,
        max_moment=2,
        fit_degree=3,
    )
    assert penalty < 1e-4
