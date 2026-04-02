import pytest

torch = pytest.importorskip("torch")

from extract_dvcs_cff.models.kinematics_encoder import KinematicsEncoder
from extract_dvcs_cff.utils.config import KinematicsEncoderConfig


def test_kinematics_encoder_output_shape_and_finite():
    cfg = KinematicsEncoderConfig()
    encoder = KinematicsEncoder(cfg)

    kin = torch.tensor(
        [
            [0.2, 0.1, -0.2, 2.0],
            [0.3, 0.15, -0.1, 3.0],
            [0.4, 0.2, -0.3, 4.0],
        ],
        dtype=torch.float32,
    )

    out = encoder(kin)
    assert out.shape == (3, encoder.output_dim)
    assert torch.isfinite(out).all()


def test_kinematics_encoder_with_embeddings():
    cfg = KinematicsEncoderConfig(
        use_process_embedding=True,
        use_flavor_embedding=True,
        use_observable_embedding=True,
    )
    encoder = KinematicsEncoder(cfg)

    kin = torch.tensor(
        [[0.2, 0.1, -0.2, 2.0], [0.3, 0.15, -0.1, 3.0]],
        dtype=torch.float32,
    )
    process_id = torch.tensor([1, 2])
    flavor_id = torch.tensor([0, 1])
    observable_id = torch.tensor([2, 3])

    out = encoder(kin, process_id=process_id, flavor_id=flavor_id, observable_id=observable_id)
    assert out.shape[0] == 2
    assert out.shape[1] == encoder.output_dim
