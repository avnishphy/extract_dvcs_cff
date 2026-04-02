"""Replica generation for uncertainty quantification."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from extract_dvcs_cff.data.dvcs_dataset import GlobalDVCSDataset


@dataclass(frozen=True)
class ReplicaMetadata:
    """Metadata for one pseudo-data replica."""

    replica_index: int
    seed: int
    mode: str


def generate_replicas(
    values: torch.Tensor,
    errors: torch.Tensor,
    n_replicas: int,
    seed: int,
    covariance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[ReplicaMetadata]]:
    """
    Generate pseudo-data replicas from diagonal errors or covariance matrix.

    Returns
    -------
    replicas:
        Tensor with shape [n_replicas, N].
    metadata:
        Per-replica metadata list.
    """
    if n_replicas <= 0:
        raise ValueError("n_replicas must be positive.")

    generator = torch.Generator(device=values.device)
    generator.manual_seed(seed)

    metadata: list[ReplicaMetadata] = []
    out: list[torch.Tensor] = []

    if covariance is not None:
        distribution = torch.distributions.MultivariateNormal(values, covariance_matrix=covariance)
        for replica_idx in range(n_replicas):
            draw = distribution.sample()
            out.append(draw)
            metadata.append(ReplicaMetadata(replica_index=replica_idx, seed=seed + replica_idx, mode="covariance"))
    else:
        sigma = torch.clamp(errors, min=1e-8)
        for replica_idx in range(n_replicas):
            noise = torch.randn(values.shape, generator=generator, device=values.device, dtype=values.dtype)
            draw = values + sigma * noise
            out.append(draw)
            metadata.append(ReplicaMetadata(replica_index=replica_idx, seed=seed + replica_idx, mode="diagonal"))

    return torch.stack(out, dim=0), metadata


def build_replica_datasets(
    dataset: GlobalDVCSDataset,
    replica_values: torch.Tensor,
) -> list[GlobalDVCSDataset]:
    """Clone a base dataset into a list of datasets with replica targets."""
    if replica_values.ndim != 2 or replica_values.shape[1] != len(dataset):
        raise ValueError("replica_values must have shape [n_replicas, len(dataset)].")

    datasets: list[GlobalDVCSDataset] = []
    for replica in replica_values:
        datasets.append(
            GlobalDVCSDataset(
                kinematics=dataset.kinematics.clone(),
                values=replica.clone(),
                errors=dataset.errors.clone(),
                observable_id=dataset.observable_id.clone(),
                experiment_id=dataset.experiment_id.clone(),
                mask=dataset.mask.clone(),
                metadata=list(dataset.metadata),
                mappings=dataset.mappings,
            )
        )
    return datasets
