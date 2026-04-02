"""
Default configuration and config models for extract_dvcs_cff.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from pathlib import Path

class PathsConfig(BaseModel):
    data_dir: Path = Field(default=Path("data"), description="Path to data directory.")
    output_dir: Path = Field(default=Path("outputs"), description="Path to output directory.")
    config_dir: Path = Field(default=Path("configs"), description="Path to config directory.")
    partons_path: Optional[Path] = Field(default=None, description="Path to PARTONS executable.")
    apfel_path: Optional[Path] = Field(default=None, description="Path to APFEL executable.")
    lhapdf_path: Optional[Path] = Field(default=None, description="Path to LHAPDF installation.")

class PhysicsConstraintsConfig(BaseModel):
    support_x: bool = True
    forward_limit: bool = True
    qcd_evolution: bool = True
    dglap_erbl: bool = True
    polynomiality: bool = False
    positivity: bool = False

class UncertaintyConfig(BaseModel):
    handle_stat: bool = True
    handle_sys: bool = True
    handle_cov: bool = False
    ensemble_size: int = 1

class BenchmarkConfig(BaseModel):
    model_name: str = "KM15"
    seed: int = 42

class IngestionConfig(BaseModel):
    dataset_files: List[Path] = Field(default_factory=list)
    kinematic_filters: Optional[dict] = None
    gpddatabase_root: Optional[Path] = Field(
        default=None,
        description="Path to local gpddatabase repository root.",
    )
    gpddatabase_data_type: Optional[str] = Field(
        default="DVCS",
        description="Filter gpddatabase entries by data_type (e.g. DVCS).",
    )
    gpddatabase_collaboration: Optional[str] = Field(
        default=None,
        description="Optional collaboration filter for gpddatabase entries.",
    )
    gpddatabase_uuid: Optional[str] = Field(
        default=None,
        description="Optional UUID filter for gpddatabase entries.",
    )
    include_pseudodata: bool = Field(
        default=False,
        description="Whether to include pseudodata entries from gpddatabase.",
    )
    strict_kinematics: bool = Field(
        default=True,
        description="If true, skip gpddatabase points not compatible with core DVCS kinematic schema.",
    )

class MainConfig(BaseModel):
    paths: PathsConfig = PathsConfig()
    physics_constraints: PhysicsConstraintsConfig = PhysicsConstraintsConfig()
    uncertainty: UncertaintyConfig = UncertaintyConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    ingestion: IngestionConfig = IngestionConfig()


def get_default_config() -> MainConfig:
    """Return a default configuration object."""
    return MainConfig()
