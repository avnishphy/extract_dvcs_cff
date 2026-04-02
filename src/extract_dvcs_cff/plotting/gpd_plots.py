"""Plotting helpers for training diagnostics and physics outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss_curves(history: list[dict[str, float]], output_path: str | Path) -> None:
    """Plot train/validation loss curves from trainer history."""
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Composite Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_gpd_slice(
    x_grid: np.ndarray,
    gpd_values: np.ndarray,
    output_path: str | Path,
    channel_name: str,
    title: str | None = None,
) -> None:
    """Plot a single GPD channel as a function of x at fixed (xi, t, Q2)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_grid, gpd_values, color="#1b4d66")
    ax.set_xlabel("x")
    ax.set_ylabel(channel_name)
    ax.set_title(title or f"GPD slice: {channel_name}")
    ax.grid(True, alpha=0.3)

    path = _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cff_comparison(
    cff_pred: np.ndarray,
    cff_ref: np.ndarray,
    output_path: str | Path,
    label_pred: str = "predicted",
    label_ref: str = "reference",
) -> None:
    """Compare predicted and reference CFF values."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(cff_pred.shape[0])
    ax.plot(x, cff_pred, marker="o", label=label_pred)
    ax.plot(x, cff_ref, marker="s", label=label_ref)
    ax.set_xlabel("Point index")
    ax.set_ylabel("CFF value")
    ax.set_title("CFF comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_replica_band(
    x: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    data: np.ndarray | None,
    output_path: str | Path,
    y_label: str,
) -> None:
    """Plot replica mean and uncertainty band, with optional data overlay."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, mean, color="#154c79", label="replica mean")
    ax.fill_between(x, lower, upper, color="#67a9cf", alpha=0.35, label="68% band")
    if data is not None:
        ax.scatter(x, data, s=20, color="#b2182b", label="data", zorder=3)
    ax.set_xlabel("Point index")
    ax.set_ylabel(y_label)
    ax.set_title("Replica uncertainty band")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
