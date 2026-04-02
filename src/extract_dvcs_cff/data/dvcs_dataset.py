"""
Dataset adapters for heterogeneous global DVCS training.

This layer abstracts experiment-specific records into a unified tensor dataset
consumed by the end-to-end trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from extract_dvcs_cff.data.schemas import DatasetRecord
from extract_dvcs_cff.physics.observables import TORCH_OBSERVABLE_INDEX, map_observable_label


SUPPORTED_DVCS_EXPERIMENTS = (
    "HERMES",
    "CLAS",
    "Hall A",
    "H1",
    "COMPASS",
)


def xi_from_xb_torch(x_b: torch.Tensor) -> torch.Tensor:
    """Compute xi from xB with safe denominator clamping."""
    return x_b / torch.clamp(2.0 - x_b, min=1e-8)


def _normalize_experiment_name(name: str) -> str:
    normalized = " ".join(name.strip().split())
    lower = normalized.lower()

    aliases = {
        "halla": "Hall A",
        "hall-a": "Hall A",
        "hall a": "Hall A",
        "hermes": "HERMES",
        "clas": "CLAS",
        "h1": "H1",
        "compass": "COMPASS",
    }
    return aliases.get(lower, normalized)


@dataclass(frozen=True)
class DatasetMappings:
    """Integer mappings used by the tensor dataset."""

    observable_to_id: dict[str, int]
    experiment_to_id: dict[str, int]


class GlobalDVCSDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Unified tensor dataset for mixed-experiment DVCS observables.

    Each row corresponds to one observable point, preserving its uncertainty,
    experiment identity, and observable identity.
    """

    def __init__(
        self,
        kinematics: torch.Tensor,
        values: torch.Tensor,
        errors: torch.Tensor,
        observable_id: torch.Tensor,
        experiment_id: torch.Tensor,
        mask: torch.Tensor,
        metadata: list[dict[str, str]],
        mappings: DatasetMappings,
    ) -> None:
        self.kinematics = kinematics
        self.values = values
        self.errors = errors
        self.observable_id = observable_id
        self.experiment_id = experiment_id
        self.mask = mask
        self.metadata = metadata
        self.mappings = mappings

    @classmethod
    def from_records(
        cls,
        records: Iterable[DatasetRecord],
        include_experiments: list[str] | None = None,
        strict_observable_support: bool = False,
    ) -> "GlobalDVCSDataset":
        include = (
            {_normalize_experiment_name(exp) for exp in include_experiments}
            if include_experiments is not None
            else {_normalize_experiment_name(exp) for exp in SUPPORTED_DVCS_EXPERIMENTS}
        )

        observable_to_id = dict(TORCH_OBSERVABLE_INDEX)
        experiment_to_id: dict[str, int] = {}

        kin_rows: list[list[float]] = []
        values: list[float] = []
        errors: list[float] = []
        observable_ids: list[int] = []
        experiment_ids: list[int] = []
        masks: list[float] = []
        metadata: list[dict[str, str]] = []

        for record in records:
            exp_name = _normalize_experiment_name(record.experiment_name)
            if exp_name not in include:
                continue

            if exp_name not in experiment_to_id:
                experiment_to_id[exp_name] = len(experiment_to_id)

            exp_id = experiment_to_id[exp_name]

            for observable, kin in zip(record.observables, record.kinematics):
                obs_name = map_observable_label(observable.observable_name)
                if obs_name not in observable_to_id:
                    if strict_observable_support:
                        raise ValueError(f"Unsupported observable: {observable.observable_name}")
                    continue

                obs_id = observable_to_id[obs_name]

                x_b = float(kin.xB)
                q2 = float(kin.Q2)
                t = float(kin.t)
                phi_rad = np.deg2rad(float(kin.phi))
                xi = float(x_b / max(2.0 - x_b, 1e-8))

                stat = 0.0 if observable.stat_error is None else float(observable.stat_error)
                sys = 0.0 if observable.sys_error is None else float(observable.sys_error)
                total = observable.total_error
                sigma = float(np.hypot(stat, sys)) if total is None else float(total)
                sigma = max(sigma, 1e-6)

                kin_rows.append([x_b, xi, t, q2, phi_rad])
                values.append(float(observable.value))
                errors.append(sigma)
                observable_ids.append(obs_id)
                experiment_ids.append(exp_id)
                masks.append(1.0)
                metadata.append(
                    {
                        "dataset_id": record.dataset_id,
                        "experiment": exp_name,
                        "observable": obs_name,
                        "channel": observable.channel or "",
                        "covariance_id": observable.covariance_id or "",
                    }
                )

        if not kin_rows:
            raise ValueError("No training points found after applying experiment/observable filters.")

        mappings = DatasetMappings(
            observable_to_id=observable_to_id,
            experiment_to_id=experiment_to_id,
        )

        return cls(
            kinematics=torch.tensor(kin_rows, dtype=torch.float32),
            values=torch.tensor(values, dtype=torch.float32),
            errors=torch.tensor(errors, dtype=torch.float32),
            observable_id=torch.tensor(observable_ids, dtype=torch.long),
            experiment_id=torch.tensor(experiment_ids, dtype=torch.long),
            mask=torch.tensor(masks, dtype=torch.float32),
            metadata=metadata,
            mappings=mappings,
        )

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "kinematics": self.kinematics[index],
            "target": self.values[index],
            "sigma": self.errors[index],
            "observable_id": self.observable_id[index],
            "experiment_id": self.experiment_id[index],
            "mask": self.mask[index],
        }

    def split(
        self,
        validation_fraction: float,
        seed: int,
    ) -> tuple[Subset[dict[str, torch.Tensor]], Subset[dict[str, torch.Tensor]]]:
        if not (0.0 <= validation_fraction < 1.0):
            raise ValueError("validation_fraction must satisfy 0 <= f < 1.")

        n_total = len(self)
        n_val = int(round(validation_fraction * n_total))
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(n_total, generator=generator)

        train_indices = permutation[:n_train].tolist()
        val_indices = permutation[n_train:].tolist()

        return Subset(self, train_indices), Subset(self, val_indices)

    def make_dataloaders(
        self,
        batch_size: int,
        validation_fraction: float,
        seed: int,
        num_workers: int,
        pin_memory: bool,
    ) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
        train_subset, val_subset = self.split(validation_fraction=validation_fraction, seed=seed)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader
