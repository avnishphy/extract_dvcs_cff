import pytest

torch = pytest.importorskip("torch")

from extract_dvcs_cff.physics.observables import TORCH_OBSERVABLE_INDEX, TorchDVCSObservableLayer


def _make_inputs(batch: int = 6):
    cffs = torch.randn(batch, 4, 2) * 0.2
    x_b = torch.linspace(0.15, 0.35, batch)
    xi = x_b / (2.0 - x_b)
    t = -torch.linspace(0.1, 0.3, batch)
    q2 = torch.linspace(2.0, 4.0, batch)
    phi = torch.linspace(0.0, 3.0, batch)
    kin = torch.stack([x_b, xi, t, q2, phi], dim=-1)
    return cffs, kin


def test_torch_observable_layer_forward_selects_by_id():
    layer = TorchDVCSObservableLayer()
    cffs, kin = _make_inputs()

    observable_id = torch.tensor(
        [
            TORCH_OBSERVABLE_INDEX["cross_section_uu"],
            TORCH_OBSERVABLE_INDEX["beam_spin_asymmetry"],
            TORCH_OBSERVABLE_INDEX["beam_charge_asymmetry"],
            TORCH_OBSERVABLE_INDEX["double_spin_asymmetry"],
            TORCH_OBSERVABLE_INDEX["cross_section_difference_lu"],
            TORCH_OBSERVABLE_INDEX["cross_section_uu"],
        ],
        dtype=torch.long,
    )

    out = layer(cffs, kin, observable_id)
    assert out.shape == (kin.shape[0],)
    assert torch.isfinite(out).all()


def test_torch_observable_asymmetry_bounds():
    layer = TorchDVCSObservableLayer()
    cffs, kin = _make_inputs()

    bsa = layer.compute_beam_spin_asymmetry(cffs, kin)
    bca = layer.compute_beam_charge_asymmetry(cffs, kin)
    dsa = layer.compute_double_spin_asymmetry(cffs, kin)

    assert torch.all((bsa >= -0.99) & (bsa <= 0.99))
    assert torch.all((bca >= -0.99) & (bca <= 0.99))
    assert torch.all((dsa >= -0.99) & (dsa <= 0.99))
