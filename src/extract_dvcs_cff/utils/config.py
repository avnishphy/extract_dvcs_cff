"""
Typed configuration objects for the DVCS observables -> GPD extraction pipeline.

The design keeps all major sub-systems independently configurable:
- model architecture (per-block)
- physics layers (evolution, convolution, constraints)
- losses and schedules
- dataset loading and replica generation
- trainer/runtime behavior

All classes are explicit Pydantic models so config validation errors are
actionable and caught early.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


ActivationName = Literal[
    "relu",
    "gelu",
    "silu",
    "tanh",
    "elu",
    "leaky_relu",
]

NormalizationName = Literal["none", "layernorm", "batchnorm"]


class BlockConfig(BaseModel):
    """Configuration for one independently swappable hidden block."""

    width: int = Field(default=192, gt=0)
    depth: int = Field(default=2, ge=1)
    activation: ActivationName = "gelu"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalization: NormalizationName = "layernorm"
    residual_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    checkpoint: bool = False


class KinematicsEncoderConfig(BaseModel):
    """Configuration for the kinematics encoder."""

    input_dim: int = Field(default=4, ge=4)
    hidden_dim: int = Field(default=96, gt=0)
    output_dim: int = Field(default=128, gt=0)
    use_process_embedding: bool = False
    process_vocab_size: int = Field(default=8, ge=1)
    process_embedding_dim: int = Field(default=8, ge=1)
    use_flavor_embedding: bool = False
    flavor_vocab_size: int = Field(default=8, ge=1)
    flavor_embedding_dim: int = Field(default=8, ge=1)
    use_observable_embedding: bool = False
    observable_vocab_size: int = Field(default=32, ge=1)
    observable_embedding_dim: int = Field(default=8, ge=1)
    eps: float = Field(default=1e-6, gt=0.0)


class GPDBackboneConfig(BaseModel):
    """Configuration for the residual MLP trunk/backbone."""

    input_dim: int = Field(default=128, gt=0)
    blocks: list[BlockConfig] = Field(default_factory=lambda: [
        BlockConfig(width=192, depth=2, activation="gelu", dropout=0.05),
        BlockConfig(width=192, depth=2, activation="gelu", dropout=0.05),
        BlockConfig(width=160, depth=2, activation="silu", dropout=0.03),
    ])
    final_dim: int = Field(default=160, gt=0)


class AuxiliaryHeadsConfig(BaseModel):
    """Optional auxiliary diagnostic/consistency heads."""

    enable_cff_head: bool = True
    cff_output_dim: int = Field(default=8, ge=0)
    enable_mellin_head: bool = True
    mellin_output_dim: int = Field(default=8, ge=0)
    enable_observable_proxy_head: bool = False
    observable_proxy_output_dim: int = Field(default=8, ge=0)


class GPDHeadsConfig(BaseModel):
    """Head layout for predicting H/E/Htilde/Etilde and optional auxiliaries."""

    channels: tuple[str, str, str, str] = ("H", "E", "Htilde", "Etilde")
    shared_tower: bool = True
    tower_hidden_dim: int = Field(default=128, gt=0)
    tower_depth: int = Field(default=2, ge=1)
    activation: ActivationName = "gelu"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    endpoint_suppression_power: float = Field(default=0.5, ge=0.0)
    support_clamp_margin: float = Field(default=1e-6, gt=0.0)
    auxiliary: AuxiliaryHeadsConfig = Field(default_factory=AuxiliaryHeadsConfig)


class EvolutionConfig(BaseModel):
    """Configuration for Q2 evolution."""

    enabled: bool = True
    mode: Literal["differentiable", "surrogate"] = "differentiable"
    reference_q2: float = Field(default=2.0, gt=0.0)
    channel_anomalous_dims: tuple[float, float, float, float] = (0.18, 0.20, 0.16, 0.16)
    surrogate_hidden_dim: int = Field(default=64, gt=0)
    surrogate_depth: int = Field(default=2, ge=1)
    clamp_log_q2_ratio: float = Field(default=8.0, gt=0.0)
    cache_kernels: bool = True


class ConstraintConfig(BaseModel):
    """Physics-constraint toggles and hyperparameters."""

    enforce_support: bool = True
    enforce_forward_limit: bool = True
    enforce_sum_rules: bool = True
    enforce_polynomiality: bool = True
    enforce_positivity: bool = True
    enforce_endpoint_suppression: bool = True
    enforce_smoothness: bool = True
    enforce_symmetry: bool = False
    positivity_margin: float = Field(default=0.0)
    smoothness_strength: float = Field(default=1e-4, ge=0.0)
    endpoint_power: float = Field(default=0.25, ge=0.0)
    polynomiality_max_moment: int = Field(default=3, ge=1)
    polynomiality_fit_degree: int = Field(default=3, ge=1)


class ConvolutionConfig(BaseModel):
    """Configuration for differentiable CFF convolution."""

    x_grid_size: int = Field(default=161, ge=33)
    pv_eps: float = Field(default=1e-5, gt=0.0)
    use_analytic_singularity_term: bool = True
    cache_grids: bool = True
    integration_fallback: Literal["clamp", "exclude-neighborhood"] = "exclude-neighborhood"


class ObservablesConfig(BaseModel):
    """Observable layer behavior and supported labels."""

    enabled_observables: list[str] = Field(
        default_factory=lambda: [
            "cross_section_uu",
            "beam_spin_asymmetry",
            "beam_charge_asymmetry",
            "double_spin_asymmetry",
        ]
    )


class LossWeightsConfig(BaseModel):
    """Per-term base weights for the composite objective."""

    w_data: float = Field(default=1.0, ge=0.0)
    w_CFF: float = Field(default=0.2, ge=0.0)
    w_fwd: float = Field(default=0.2, ge=0.0)
    w_sum: float = Field(default=0.2, ge=0.0)
    w_poly: float = Field(default=0.15, ge=0.0)
    w_pos: float = Field(default=0.1, ge=0.0)
    w_evol: float = Field(default=0.1, ge=0.0)
    w_smooth: float = Field(default=0.05, ge=0.0)
    w_reg: float = Field(default=1e-4, ge=0.0)


class LossPhaseConfig(BaseModel):
    """One curriculum phase with explicit epoch window and multipliers."""

    start_epoch: int = Field(ge=0)
    end_epoch: int = Field(ge=0)
    multipliers: LossWeightsConfig

    @model_validator(mode="after")
    def _check_window(self) -> "LossPhaseConfig":
        if self.end_epoch < self.start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        return self


class LossConfig(BaseModel):
    """Composite loss and scheduling configuration."""

    base_weights: LossWeightsConfig = Field(default_factory=LossWeightsConfig)
    phases: list[LossPhaseConfig] = Field(
        default_factory=lambda: [
            LossPhaseConfig(
                start_epoch=0,
                end_epoch=9,
                multipliers=LossWeightsConfig(
                    w_data=1.0,
                    w_CFF=0.05,
                    w_fwd=0.05,
                    w_sum=0.05,
                    w_poly=0.02,
                    w_pos=0.02,
                    w_evol=0.02,
                    w_smooth=0.02,
                    w_reg=1e-4,
                ),
            ),
            LossPhaseConfig(
                start_epoch=10,
                end_epoch=24,
                multipliers=LossWeightsConfig(
                    w_data=1.0,
                    w_CFF=0.15,
                    w_fwd=0.15,
                    w_sum=0.10,
                    w_poly=0.08,
                    w_pos=0.05,
                    w_evol=0.05,
                    w_smooth=0.04,
                    w_reg=1e-4,
                ),
            ),
            LossPhaseConfig(
                start_epoch=25,
                end_epoch=49,
                multipliers=LossWeightsConfig(
                    w_data=1.0,
                    w_CFF=0.25,
                    w_fwd=0.25,
                    w_sum=0.20,
                    w_poly=0.15,
                    w_pos=0.10,
                    w_evol=0.10,
                    w_smooth=0.06,
                    w_reg=1e-4,
                ),
            ),
            LossPhaseConfig(
                start_epoch=50,
                end_epoch=100000,
                multipliers=LossWeightsConfig(),
            ),
        ]
    )
    adaptive_weighting: bool = True
    adaptive_beta: float = Field(default=0.95, ge=0.0, lt=1.0)
    adaptive_eps: float = Field(default=1e-8, gt=0.0)


class DatasetConfig(BaseModel):
    """Dataset and adapter options."""

    experiments: list[str] = Field(
        default_factory=lambda: ["HERMES", "CLAS", "Hall A", "H1", "COMPASS"]
    )
    include_lattice_auxiliary: bool = False
    include_structure_function_auxiliary: bool = False
    strict_missing_observable_handling: bool = True
    observable_mask_fill: float = 0.0
    use_covariance_when_available: bool = True


class ReplicaConfig(BaseModel):
    """Replica-based uncertainty configuration."""

    enabled: bool = True
    n_replicas: int = Field(default=20, ge=1)
    seed: int = Field(default=1234, ge=0)
    train_separate_models: bool = True


class RuntimeConfig(BaseModel):
    """Runtime/performance settings."""

    seed: int = Field(default=2026, ge=0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    use_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    use_checkpointing: bool = False
    use_vmap: bool = True
    ddp_enabled: bool = False


class TrainingConfig(BaseModel):
    """Trainer options."""

    epochs: int = Field(default=60, ge=1)
    batch_size: int = Field(default=64, ge=1)
    validation_split: float = Field(default=0.15, ge=0.0, lt=0.95)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    grad_clip_norm: float = Field(default=1.0, gt=0.0)
    early_stopping_patience: int = Field(default=15, ge=1)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = True
    checkpoint_every: int = Field(default=5, ge=1)
    resume_from: Path | None = None


class BaselineAdapterConfig(BaseModel):
    """Optional baselines for benchmarking against the main method."""

    enable_cff_only: bool = True
    enable_parametric_gpd: bool = True
    enable_pure_data_fit: bool = True


class PathsConfig(BaseModel):
    """Input/output path configuration."""

    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    artifact_dir: Path = Path("outputs/artifacts")


class PipelineConfig(BaseModel):
    """Top-level configuration for the end-to-end DVCS->GPD pipeline."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    encoder: KinematicsEncoderConfig = Field(default_factory=KinematicsEncoderConfig)
    backbone: GPDBackboneConfig = Field(default_factory=GPDBackboneConfig)
    heads: GPDHeadsConfig = Field(default_factory=GPDHeadsConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    convolution: ConvolutionConfig = Field(default_factory=ConvolutionConfig)
    observables: ObservablesConfig = Field(default_factory=ObservablesConfig)
    losses: LossConfig = Field(default_factory=LossConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    replicas: ReplicaConfig = Field(default_factory=ReplicaConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    baselines: BaselineAdapterConfig = Field(default_factory=BaselineAdapterConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        """Load a pipeline config from JSON or YAML."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".json":
            payload = json.loads(path.read_text())
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise ValueError("Loading YAML configs requires pyyaml.") from exc
            payload = yaml.safe_load(path.read_text())
        else:
            raise ValueError("Config path must have .json, .yaml, or .yml extension.")

        if payload is None:
            payload = {}
        return cls.model_validate(payload)

    def save(self, path: str | Path) -> None:
        """Save the config to JSON or YAML based on file extension."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        suffix = path.suffix.lower()

        if suffix == ".json":
            path.write_text(json.dumps(payload, indent=2))
            return
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise ValueError("Saving YAML configs requires pyyaml.") from exc
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
            return
        raise ValueError("Config path must have .json, .yaml, or .yml extension.")
