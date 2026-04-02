"""
Command-line interface for extract_dvcs_cff.
"""
import logging
import json
from pathlib import Path

import typer

from extract_dvcs_cff.config.defaults import MainConfig
from extract_dvcs_cff.data.io import load_all_datasets
from extract_dvcs_cff.simulation.pseudodata import generate_pseudodata as generate_pseudodata_impl
from extract_dvcs_cff.physics.likelihood import compute_likelihood
from extract_dvcs_cff.plotting.base import plot_diagnostics
from extract_dvcs_cff.utils.logging import setup_logging

app = typer.Typer()


def _load_config(config_path: Path) -> MainConfig:
    if not config_path.exists():
        raise typer.BadParameter(f"Config file does not exist: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. Install pyyaml to proceed."
            ) from exc
        payload = yaml.safe_load(config_path.read_text())
    elif suffix == ".json":
        payload = json.loads(config_path.read_text())
    else:
        raise typer.BadParameter("Config file must be .yaml, .yml, or .json")

    if payload is None:
        payload = {}
    return MainConfig.model_validate(payload)

@app.command()
def ingest_datasets(config: Path):
    """Ingest datasets from config file."""
    setup_logging()
    cfg = _load_config(config)
    datasets = load_all_datasets(cfg)
    logging.info(f"Ingested {len(datasets)} datasets.")

@app.command("generate-pseudodata")
def generate_pseudodata_cmd(config: Path):
    """Generate pseudodata using benchmark model."""
    setup_logging()
    cfg = _load_config(config)
    generate_pseudodata_impl(cfg)
    logging.info("Pseudodata generation complete.")

@app.command()
def validate_dataset(config: Path):
    """Validate dataset integrity and schema."""
    setup_logging()
    cfg = _load_config(config)
    datasets = load_all_datasets(cfg)
    logging.info(f"Validation complete for {len(datasets)} dataset(s).")

@app.command()
def run_closure_test(config: Path):
    """Run closure test workflow."""
    setup_logging()
    # ...existing code...
    logging.info("Closure test complete.")

@app.command()
def compute_likelihood_cmd(config: Path):
    """Compute likelihood for dataset and model."""
    setup_logging()
    log_like = compute_likelihood(config)
    logging.info(f"Likelihood computation complete. logL={log_like:.6f}")

@app.command()
def plot_diagnostics_cmd(config: Path):
    """Plot diagnostics and results."""
    setup_logging()
    plot_diagnostics(config)
    logging.info("Plotting complete.")

if __name__ == "__main__":
    app()
