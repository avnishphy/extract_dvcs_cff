# PYTHONPATH=../src python likelihood_diagnostics.py

import numpy as np
import matplotlib.pyplot as plt

from extract_dvcs_cff.physics.likelihood import GaussianLikelihood


def generate_data(n_points=20, noise_level=0.1, seed=42):
    np.random.seed(seed)

    x = np.linspace(0, 1, n_points)

    # True model (simple linear function for intuition)
    true_param = 2.0
    theory_true = true_param * x

    noise = np.random.normal(0, noise_level, size=n_points)
    data = theory_true + noise

    errors = np.full_like(data, noise_level)

    return x, data, errors, true_param


def scan_parameter(x, data, errors, param_range):
    chi2_values = []

    for p in param_range:
        theory = p * x
        llh = GaussianLikelihood(data, theory, stat_errors=errors)
        chi2_values.append(llh.chi2())

    return np.array(chi2_values)


def plot_results(x, data, errors, true_param, param_range, chi2_values):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: data vs theory ---
    axes[0].errorbar(x, data, yerr=errors, fmt='o', label="Data")
    axes[0].plot(x, true_param * x, label="True theory", linewidth=2)
    axes[0].set_title("Data vs True Theory")
    axes[0].legend()

    # --- Plot 2: chi2 scan ---
    axes[1].plot(param_range, chi2_values)
    axes[1].axvline(true_param, linestyle="--", label="True param")
    axes[1].set_title("Chi2 vs Parameter")
    axes[1].set_xlabel("Parameter")
    axes[1].set_ylabel("Chi2")
    axes[1].legend()

    # --- Plot 3: residuals ---
    best_param = param_range[np.argmin(chi2_values)]
    best_theory = best_param * x
    residuals = data - best_theory

    axes[2].scatter(x, residuals)
    axes[2].axhline(0, linestyle="--")
    axes[2].set_title("Residuals at Best Fit")

    plt.tight_layout()
    plt.savefig("likelihood_diagnostics.png")


def main():
    x, data, errors, true_param = generate_data()

    param_range = np.linspace(0, 4, 100)
    chi2_values = scan_parameter(x, data, errors, param_range)

    plot_results(x, data, errors, true_param, param_range, chi2_values)


if __name__ == "__main__":
    main()