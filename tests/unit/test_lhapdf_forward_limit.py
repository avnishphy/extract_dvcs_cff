import pytest

torch = pytest.importorskip("torch")

from extract_dvcs_cff.lhapdf.adapter import LHAPDFAdapter
from extract_dvcs_cff.physics.constraints import forward_limit_penalty


def test_lhapdf_adapter_evaluate_shape_and_finite():
    adapter = LHAPDFAdapter()
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    q2 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)

    values = adapter.evaluate("u", x, q2)
    assert values.shape == x.shape
    assert torch.isfinite(values).all()


def test_forward_limit_penalty_nonnegative():
    adapter = LHAPDFAdapter()
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    q2 = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    pdf = adapter.evaluate("u", x, q2)

    penalty_exact = forward_limit_penalty(pdf, x, q2, pdf_provider=adapter, flavor="u")
    assert torch.isclose(penalty_exact, torch.tensor(0.0), atol=1e-6)

    shifted = pdf + 0.1
    penalty_shifted = forward_limit_penalty(shifted, x, q2, pdf_provider=adapter, flavor="u")
    assert penalty_shifted > penalty_exact
