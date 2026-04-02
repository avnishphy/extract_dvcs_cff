import pytest

torch = pytest.importorskip("torch")

from extract_dvcs_cff.physics.cff_convolution import DifferentiableCFFConvolution
from extract_dvcs_cff.utils.config import ConvolutionConfig


def test_cff_convolution_shapes_and_finite():
    config = ConvolutionConfig(x_grid_size=129)
    layer = DifferentiableCFFConvolution(config)

    batch = 4
    x_grid = torch.linspace(-0.9, 0.9, config.x_grid_size)
    xi = torch.tensor([0.1, 0.2, 0.15, 0.05], dtype=torch.float32)

    gpd_grid = torch.ones(batch, config.x_grid_size, 4)
    out = layer(gpd_grid=gpd_grid, xi=xi, x_grid=x_grid)

    assert out.stacked.shape == (batch, 4, 2)
    assert torch.isfinite(out.stacked).all()


def test_cff_convolution_imaginary_part_matches_constant_limit():
    config = ConvolutionConfig(x_grid_size=129)
    layer = DifferentiableCFFConvolution(config)

    batch = 2
    x_grid = torch.linspace(-0.9, 0.9, config.x_grid_size)
    xi = torch.tensor([0.1, 0.2], dtype=torch.float32)
    const = 0.7

    gpd_grid = torch.full((batch, config.x_grid_size, 4), const)
    out = layer(gpd_grid=gpd_grid, xi=xi, x_grid=x_grid)

    expected_imag = torch.full((batch,), torch.pi * const)
    assert torch.allclose(out.H_imag, expected_imag, atol=5e-2)
