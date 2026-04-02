"""Example script: train and evaluate the DVCS observables -> GPD pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("This example requires torch to be installed.") from exc

from extract_dvcs_cff.data.dvcs_dataset import GlobalDVCSDataset
from extract_dvcs_cff.data.schemas import DatasetRecord, KinematicPoint, ObservableRecord
from extract_dvcs_cff.evaluation.metrics import compute_replica_statistics
from extract_dvcs_cff.inference.predict import DVCSPredictor, load_checkpoint_for_inference
from extract_dvcs_cff.physics.constraints import NullPDFProvider, PhysicsConstraintEvaluator
from extract_dvcs_cff.physics.observables import (
    KinematicsBatch,
    TORCH_OBSERVABLE_INDEX,
    ToyDVCSObservableCalculator,
    generate_toy_cffs,
)
from extract_dvcs_cff.plotting import plot_cff_comparison, plot_gpd_slice, plot_loss_curves, plot_replica_band
from extract_dvcs_cff.training.replicas import generate_replicas
from extract_dvcs_cff.training.trainer import DVCSGPDTrainer
from extract_dvcs_cff.utils.config import PipelineConfig


def _build_synthetic_record(n_points: int, seed: int = 2026) -> DatasetRecord:
    rng = np.random.default_rng(seed)

    x_b = rng.uniform(0.1, 0.5, size=n_points)
    q2 = rng.uniform(1.5, 5.0, size=n_points)
    t = -rng.uniform(0.05, 0.5, size=n_points)
    phi_deg = rng.uniform(0.0, 360.0, size=n_points)

    kin_batch = KinematicsBatch.from_sequences(xB=x_b, Q2=q2, t=t, phi=np.deg2rad(phi_deg))
    cffs = generate_toy_cffs(kin_batch)
    calculator = ToyDVCSObservableCalculator()

    observable_names = list(TORCH_OBSERVABLE_INDEX.keys())

    observables: list[ObservableRecord] = []
    kinematics: list[KinematicPoint] = []

    for idx in range(n_points):
        name = observable_names[idx % len(observable_names)]

        one_kin = KinematicsBatch.from_sequences(
            xB=[float(x_b[idx])],
            Q2=[float(q2[idx])],
            t=[float(t[idx])],
            phi=[float(np.deg2rad(phi_deg[idx]))],
        )
        one_cff = {key: np.asarray([value[idx]]) for key, value in cffs.items()}
        value = float(calculator.compute(name, one_kin, one_cff)[0])

        sigma = 0.04 * abs(value) + 0.02
        noisy_value = float(value + rng.normal(0.0, sigma))

        observables.append(
            ObservableRecord(
                observable_name=name,
                value=noisy_value,
                stat_error=float(0.7 * sigma),
                sys_error=float(0.3 * sigma),
                total_error=float(sigma),
                covariance_id=None,
                channel="synthetic",
            )
        )

        kinematics.append(
            KinematicPoint(
                xB=float(x_b[idx]),
                Q2=float(q2[idx]),
                t=float(t[idx]),
                phi=float(phi_deg[idx]),
            )
        )

    return DatasetRecord(
        experiment_name="HERMES",
        dataset_id="synthetic_dvcs_demo",
        publication="synthetic",
        observables=observables,
        kinematics=kinematics,
        comments="Synthetic closure-like dataset for pipeline smoke testing.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact DVCS->GPD example workflow.")
    parser.add_argument("--config", type=Path, default=Path("configs/gpd_pipeline_example.yaml"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--replicas", type=int, default=30)
    args = parser.parse_args()

    config = PipelineConfig.from_file(args.config)
    config.training.epochs = args.epochs
    config.replicas.enabled = False

    record = _build_synthetic_record(args.points, seed=config.runtime.seed)
    dataset = GlobalDVCSDataset.from_records([record], include_experiments=["HERMES"])

    constraint_eval = PhysicsConstraintEvaluator(
        config=config.constraints,
        pdf_provider=NullPDFProvider(value=0.0),
    )

    trainer = DVCSGPDTrainer(config=config, dataset=dataset, constraint_evaluator=constraint_eval)
    result = trainer.train()

    model = load_checkpoint_for_inference(config, result.checkpoint_path)
    predictor = DVCSPredictor(config, model)

    sample_count = min(64, len(dataset))
    kin = dataset.kinematics[:sample_count]
    obs_id = dataset.observable_id[:sample_count]
    pred = predictor.predict(kinematics=kin, observable_id=obs_id)

    replicas, _ = generate_replicas(
        values=dataset.values[:sample_count],
        errors=dataset.errors[:sample_count],
        n_replicas=args.replicas,
        seed=config.runtime.seed,
    )
    bands = compute_replica_statistics(replicas.cpu().numpy())

    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(result.history, output_dir / "loss_curves.png")

    x_grid = predictor.convolution.get_x_grid(device=pred.gpd_grid.device, dtype=pred.gpd_grid.dtype)
    plot_gpd_slice(
        x_grid=x_grid.detach().cpu().numpy(),
        gpd_values=pred.gpd_grid[0, :, 0].detach().cpu().numpy(),
        output_path=output_dir / "gpd_slice_H.png",
        channel_name="H",
        title="Predicted H(x, xi, t, Q2) slice",
    )

    cff_pred = pred.cff_stacked[:, 0, 0].detach().cpu().numpy()

    kin_np = kin.detach().cpu().numpy()
    kin_batch = KinematicsBatch.from_sequences(
        xB=kin_np[:, 0],
        Q2=kin_np[:, 3],
        t=kin_np[:, 2],
        phi=kin_np[:, 4],
    )
    cff_ref = generate_toy_cffs(kin_batch)["H_real"]

    plot_cff_comparison(
        cff_pred=cff_pred,
        cff_ref=cff_ref,
        output_path=output_dir / "cff_H_real_comparison.png",
        label_pred="predicted H_real",
        label_ref="toy reference H_real",
    )

    indices = np.arange(sample_count)
    plot_replica_band(
        x=indices,
        mean=bands["mean"],
        lower=bands["lower"],
        upper=bands["upper"],
        data=dataset.values[:sample_count].cpu().numpy(),
        output_path=output_dir / "replica_band.png",
        y_label="Observable",
    )

    summary = {
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "checkpoint": str(result.checkpoint_path),
        "outputs": str(output_dir),
    }
    (output_dir / "example_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Example pipeline finished.")
    print(summary)


if __name__ == "__main__":
    main()
