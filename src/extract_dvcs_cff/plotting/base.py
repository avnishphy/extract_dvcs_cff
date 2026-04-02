"""
Base plotting utilities for DVCS/GPD/CFF analysis.
"""
import matplotlib.pyplot as plt
from typing import Any, Optional

def plot_observables_vs_kinematics(x, y, yerr=None, label=None, xlabel="Kinematic variable", ylabel="Observable", ax: Optional[Any] = None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label:
        ax.legend()
    return ax

def plot_residuals(x, residuals, ax: Optional[Any] = None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, residuals, "o")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_xlabel("Kinematic variable")
    ax.set_ylabel("Residuals")
    return ax

# Add more plotting utilities as needed

def plot_diagnostics(config):
    # Placeholder for CLI integration
    pass
