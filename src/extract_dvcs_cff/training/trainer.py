"""End-to-end trainer for DVCS observables -> GPD extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from extract_dvcs_cff.data.dvcs_dataset import GlobalDVCSDataset
from extract_dvcs_cff.losses.composite import CompositeLoss
from extract_dvcs_cff.losses.physics_terms import PhysicsLossTermComputer
from extract_dvcs_cff.models.gpd_heads import DVCSGPDModel
from extract_dvcs_cff.physics.cff_convolution import DifferentiableCFFConvolution
from extract_dvcs_cff.physics.constraints import PhysicsConstraintEvaluator
from extract_dvcs_cff.physics.evolution import Q2EvolutionLayer
from extract_dvcs_cff.physics.observables import TorchDVCSObservableLayer
from extract_dvcs_cff.training.scheduler import build_lr_scheduler, current_loss_phase_name
from extract_dvcs_cff.training.replicas import build_replica_datasets, generate_replicas
from extract_dvcs_cff.utils.config import PipelineConfig
from extract_dvcs_cff.utils.random import set_seed
from extract_dvcs_cff.utils.tracking import JSONLTracker, save_artifact_manifest


@dataclass
class TrainingResult:
    """Artifacts returned by training."""

    history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path


class DVCSGPDTrainer:
    """Trainer for physics-informed end-to-end DVCS->GPD learning."""

    def __init__(
        self,
        config: PipelineConfig,
        dataset: GlobalDVCSDataset,
        constraint_evaluator: PhysicsConstraintEvaluator,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.constraint_evaluator = constraint_evaluator

        set_seed(config.runtime.seed)
        self.device = self._resolve_device(config.runtime.device)

        self.output_dir = Path(config.paths.output_dir)
        self.checkpoint_dir = Path(config.paths.checkpoint_dir)
        self.artifact_dir = Path(config.paths.artifact_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.model = DVCSGPDModel(config).to(self.device)
        self.evolution = Q2EvolutionLayer(config.evolution).to(self.device)
        self.convolution = DifferentiableCFFConvolution(config.convolution)
        self.observable_layer = TorchDVCSObservableLayer().to(self.device)

        if config.runtime.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode=config.runtime.compile_mode)

        self.model = self._maybe_wrap_ddp(self.model)

        self.loss_terms = PhysicsLossTermComputer(
            constraints=self.constraint_evaluator,
            evolution=self.evolution if config.evolution.enabled else None,
        )
        self.composite_loss = CompositeLoss(config.losses)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.lr_scheduler = build_lr_scheduler(self.optimizer, config.training)

        self.metric_tracker = JSONLTracker(self.output_dir / "training_metrics.jsonl")
        self.best_val_loss = float("inf")
        self.best_epoch = -1

        self._save_run_metadata()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable.")
            return torch.device("cuda")
        if requested == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is unavailable.")
            return torch.device("mps")

        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _maybe_wrap_ddp(self, model: nn.Module) -> nn.Module:
        if not self.config.runtime.ddp_enabled:
            return model
        if not torch.distributed.is_available():
            return model
        if not torch.distributed.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            torch.distributed.init_process_group(backend=backend)
        if self.device.type == "cuda":
            return DistributedDataParallel(model, device_ids=[self.device.index or 0])
        return DistributedDataParallel(model)

    def _model_for_state_dict(self) -> nn.Module:
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def _save_run_metadata(self) -> None:
        model_repr = str(self._model_for_state_dict())
        model_hash = hashlib.sha256(model_repr.encode("utf-8")).hexdigest()

        manifest = {
            "seed": self.config.runtime.seed,
            "model_hash": model_hash,
            "dataset_size": len(self.dataset),
            "observable_map": self.dataset.mappings.observable_to_id,
            "experiment_map": self.dataset.mappings.experiment_to_id,
            "config": self.config.model_dump(mode="json"),
        }
        save_artifact_manifest(self.artifact_dir / "run_manifest.json", manifest)

    def _checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"

    def save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        payload = {
            "epoch": epoch,
            "model": self._model_for_state_dict().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "val_loss": val_loss,
            "config": self.config.model_dump(mode="json"),
        }
        path = self._checkpoint_path(epoch)
        torch.save(payload, path)
        return path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        payload = torch.load(checkpoint_path, map_location=self.device)
        self._model_for_state_dict().load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.lr_scheduler.load_state_dict(payload["scheduler"])
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        self.best_epoch = int(payload.get("best_epoch", -1))
        return int(payload["epoch"]) + 1

    def _predict_gpd_grid(self, x_grid: torch.Tensor, xi: torch.Tensor, t: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        model = self._model_for_state_dict()

        if self.config.runtime.use_vmap and hasattr(torch, "vmap"):
            x_points = x_grid.unsqueeze(0).expand(xi.shape[0], -1)

            def _single_eval(x_row: torch.Tensor, xi_i: torch.Tensor, t_i: torch.Tensor, q2_i: torch.Tensor) -> torch.Tensor:
                kin = torch.stack(
                    [
                        x_row,
                        torch.full_like(x_row, xi_i),
                        torch.full_like(x_row, t_i),
                        torch.full_like(x_row, q2_i),
                    ],
                    dim=-1,
                )
                return model.predict_gpd(kin)

            return torch.vmap(_single_eval)(x_points, xi, t, q2)

        return model.predict_gpd_on_grid(x_grid=x_grid, xi=xi, t=t, q2=q2)

    def _run_epoch(self, loader: DataLoader[dict[str, torch.Tensor]], epoch: int, train: bool) -> dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        running = {
            "loss": 0.0,
            "L_DVCS": 0.0,
            "L_transform": 0.0,
            "L_forward": 0.0,
            "L_sumrule": 0.0,
            "L_polynomiality": 0.0,
            "L_positivity": 0.0,
            "L_evolution": 0.0,
            "L_smooth": 0.0,
            "L_regularization": 0.0,
        }

        n_batches = 0
        for batch in loader:
            kinematics = batch["kinematics"].to(self.device)
            targets = batch["target"].to(self.device)
            sigma = batch["sigma"].to(self.device)
            observable_id = batch["observable_id"].to(self.device)
            mask = batch["mask"].to(self.device)

            x_b = kinematics[:, 0]
            xi = kinematics[:, 1]
            t = kinematics[:, 2]
            q2 = kinematics[:, 3]

            x_grid = self.convolution.get_x_grid(self.device, dtype=kinematics.dtype)
            gpd_grid = self._predict_gpd_grid(x_grid=x_grid, xi=xi, t=t, q2=q2)

            if self.config.evolution.enabled:
                flat = gpd_grid.reshape(-1, 4)
                q2_flat = q2.unsqueeze(1).expand(-1, x_grid.shape[0]).reshape(-1)
                evolved = self.evolution(flat, q2_target=q2_flat)
                gpd_grid = evolved.reshape_as(gpd_grid)

            cff = self.convolution(gpd_grid=gpd_grid, xi=xi, x_grid=x_grid)
            pred_obs = self.observable_layer(cff.stacked, kinematics, observable_id)

            forward_x = torch.clamp(torch.abs(x_b), min=1e-5, max=0.95)
            forward_kin = torch.stack(
                [
                    forward_x,
                    torch.zeros_like(forward_x),
                    torch.zeros_like(forward_x),
                    q2,
                ],
                dim=-1,
            )
            forward_h = self._model_for_state_dict().predict_gpd(forward_kin)[:, 0]

            center_kin = torch.stack([xi, xi, t, q2], dim=-1)
            aux_output = self._model_for_state_dict().forward(center_kin)
            aux_cff = aux_output.get("aux_cff")

            terms = self.loss_terms.compute(
                model=self._model_for_state_dict(),
                pred_observables=pred_obs,
                target_observables=targets,
                sigma=sigma,
                mask=mask,
                cff_stacked=cff.stacked,
                aux_cff=aux_cff,
                gpd_grid=gpd_grid,
                x_grid=x_grid,
                xi_values=xi,
                t_values=t,
                q2_values=q2,
                forward_h=forward_h,
                forward_x=forward_x,
                forward_q2=q2,
            )

            total_loss, _weighted, _effective_weights = self.composite_loss(terms, epoch)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model_for_state_dict().parameters(), self.config.training.grad_clip_norm)
                self.optimizer.step()

            running["loss"] += float(total_loss.detach().cpu().item())
            for key in list(running.keys())[1:]:
                if key in terms:
                    running[key] += float(terms[key].detach().cpu().item())

            n_batches += 1

        if n_batches == 0:
            return {key: float("nan") for key in running}

        for key in running:
            running[key] /= n_batches
        return running

    def train(self) -> TrainingResult:
        train_loader, val_loader = self.dataset.make_dataloaders(
            batch_size=self.config.training.batch_size,
            validation_fraction=self.config.training.validation_split,
            seed=self.config.runtime.seed,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
        )

        start_epoch = 0
        if self.config.training.resume_from is not None:
            start_epoch = self.load_checkpoint(Path(self.config.training.resume_from))

        history: list[dict[str, float]] = []
        patience_count = 0
        last_checkpoint = self._checkpoint_path(start_epoch)

        for epoch in range(start_epoch, self.config.training.epochs):
            train_metrics = self._run_epoch(train_loader, epoch=epoch, train=True)
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, epoch=epoch, train=False)

            self.lr_scheduler.step()

            row = {
                "epoch": float(epoch),
                "phase": current_loss_phase_name(self.config.losses, epoch),
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_L_DVCS": train_metrics["L_DVCS"],
                "val_L_DVCS": val_metrics["L_DVCS"],
            }
            history.append(row)
            self.metric_tracker.log(row)
            self.metric_tracker.flush()

            val_loss = val_metrics["loss"]
            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_count = 0
                last_checkpoint = self.save_checkpoint(epoch, val_loss)
            else:
                patience_count += 1
                if epoch % self.config.training.checkpoint_every == 0:
                    last_checkpoint = self.save_checkpoint(epoch, val_loss)

            if patience_count >= self.config.training.early_stopping_patience:
                break

        summary_path = self.artifact_dir / "training_history.json"
        summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        return TrainingResult(
            history=history,
            best_epoch=self.best_epoch,
            best_val_loss=self.best_val_loss,
            checkpoint_path=last_checkpoint,
        )


def train_with_optional_replicas(
    config: PipelineConfig,
    dataset: GlobalDVCSDataset,
    constraint_evaluator: PhysicsConstraintEvaluator,
) -> dict[str, Any]:
    """
    Train one model or an ensemble over pseudo-data replicas.

    Returns
    -------
    Dictionary with summary metrics and checkpoint locations.
    """
    if not config.replicas.enabled:
        trainer = DVCSGPDTrainer(config=config, dataset=dataset, constraint_evaluator=constraint_evaluator)
        result = trainer.train()
        return {
            "mode": "single",
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "checkpoints": [str(result.checkpoint_path)],
        }

    replica_values, metadata = generate_replicas(
        values=dataset.values,
        errors=dataset.errors,
        n_replicas=config.replicas.n_replicas,
        seed=config.replicas.seed,
    )
    replica_datasets = build_replica_datasets(dataset, replica_values)

    checkpoints: list[str] = []
    val_losses: list[float] = []
    for replica_idx, replica_dataset in enumerate(replica_datasets):
        replica_config = config.model_copy(deep=True)
        replica_config.runtime.seed = metadata[replica_idx].seed
        replica_config.paths.checkpoint_dir = Path(config.paths.checkpoint_dir) / f"replica_{replica_idx:03d}"
        replica_config.paths.artifact_dir = Path(config.paths.artifact_dir) / f"replica_{replica_idx:03d}"

        trainer = DVCSGPDTrainer(
            config=replica_config,
            dataset=replica_dataset,
            constraint_evaluator=constraint_evaluator,
        )
        result = trainer.train()
        checkpoints.append(str(result.checkpoint_path))
        val_losses.append(float(result.best_val_loss))

    return {
        "mode": "replicas",
        "n_replicas": len(replica_datasets),
        "mean_best_val_loss": float(sum(val_losses) / max(len(val_losses), 1)),
        "checkpoints": checkpoints,
    }
