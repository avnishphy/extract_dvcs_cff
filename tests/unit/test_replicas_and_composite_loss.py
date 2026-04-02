import pytest

torch = pytest.importorskip("torch")

from extract_dvcs_cff.data.dvcs_dataset import DatasetMappings, GlobalDVCSDataset
from extract_dvcs_cff.losses.composite import CompositeLoss
from extract_dvcs_cff.training.replicas import build_replica_datasets, generate_replicas
from extract_dvcs_cff.utils.config import LossConfig


def _make_dataset(n: int = 10) -> GlobalDVCSDataset:
    kinematics = torch.rand(n, 5)
    values = torch.rand(n)
    errors = 0.1 * torch.ones(n)
    observable_id = torch.zeros(n, dtype=torch.long)
    experiment_id = torch.zeros(n, dtype=torch.long)
    mask = torch.ones(n)
    metadata = [{"dataset_id": "d", "experiment": "HERMES", "observable": "cross_section_uu", "channel": "", "covariance_id": ""} for _ in range(n)]
    mappings = DatasetMappings(observable_to_id={"cross_section_uu": 0}, experiment_to_id={"HERMES": 0})
    return GlobalDVCSDataset(kinematics, values, errors, observable_id, experiment_id, mask, metadata, mappings)


def test_generate_replicas_deterministic_with_seed():
    values = torch.tensor([1.0, 2.0, 3.0])
    errors = torch.tensor([0.1, 0.2, 0.3])

    r1, _ = generate_replicas(values, errors, n_replicas=3, seed=123)
    r2, _ = generate_replicas(values, errors, n_replicas=3, seed=123)

    assert torch.allclose(r1, r2)


def test_build_replica_datasets_count():
    dataset = _make_dataset(8)
    replica_values, _ = generate_replicas(dataset.values, dataset.errors, n_replicas=4, seed=7)
    replica_datasets = build_replica_datasets(dataset, replica_values)
    assert len(replica_datasets) == 4
    assert all(len(ds) == len(dataset) for ds in replica_datasets)


def test_composite_loss_combines_terms():
    composite = CompositeLoss(LossConfig())
    loss_terms = {
        "L_DVCS": torch.tensor(2.0),
        "L_transform": torch.tensor(1.0),
        "L_forward": torch.tensor(0.5),
        "L_sumrule": torch.tensor(0.5),
        "L_polynomiality": torch.tensor(0.25),
        "L_positivity": torch.tensor(0.1),
        "L_evolution": torch.tensor(0.2),
        "L_smooth": torch.tensor(0.3),
        "L_regularization": torch.tensor(0.01),
    }
    total, weighted, effective = composite(loss_terms, epoch=0)

    assert torch.isfinite(total)
    assert total > 0
    assert "L_DVCS" in weighted
    assert "L_DVCS" in effective
