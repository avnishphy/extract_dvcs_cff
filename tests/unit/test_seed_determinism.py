import pytest

torch = pytest.importorskip("torch")

from extract_dvcs_cff.models.gpd_heads import DVCSGPDModel
from extract_dvcs_cff.utils.config import PipelineConfig
from extract_dvcs_cff.utils.random import set_seed


def test_model_determinism_under_fixed_seed():
    cfg = PipelineConfig()

    set_seed(2026)
    model_a = DVCSGPDModel(cfg)
    inp = torch.tensor(
        [[0.2, 0.1, -0.2, 2.0], [0.3, 0.15, -0.1, 3.0]],
        dtype=torch.float32,
    )
    out_a = model_a.predict_gpd(inp)

    set_seed(2026)
    model_b = DVCSGPDModel(cfg)
    out_b = model_b.predict_gpd(inp)

    assert torch.allclose(out_a, out_b)
