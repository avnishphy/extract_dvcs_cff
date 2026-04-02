#!/usr/bin/env python3
"""
Toy χ² diagnostics for the DVCS/GPD/CFF framework.

This script validates the full toy pipeline visually:
- kinematics handling
- toy observable generation
- stacked multi-observable likelihood
- correlated covariance
- normalization profiling
- 1D and 2D χ² scans

Run from the repository root.

Example:
    PYTHONPATH=src python scripts/toy_chi2_scan.py --output outputs/figures/toy_scan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

# Allow running before editable installation during development.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extract_dvcs_cff.physics.likelihood import GaussianLikelihood
from extract_dvcs_cff.physics.observables import (
    KinematicsBatch,
    ToyCFFParameters,
    ToyDVCSObservableCalculator,
    generate_toy_cffs,
    scale_toy_cffs,
)


def build_kinematics(n_points: int = 24) -> KinematicsBatch:
    """
    Build a smooth kinematic batch with enough variation to exercise the toy model.
    """
    phi = np.linspace(0.15, 2.0 * np.pi - 0.15, n_points)
    xB = np.linspace(0.16, 0.42, n_points)
    Q2 = np.linspace(1.8, 4.2, n_points)
    t = -np.linspace(0.08, 0.42, n_points)
    return KinematicsBatch.from_sequences(xB=xB, Q2=Q2, t=t, phi=phi)


def make_correlated_covariance(errors: np.ndarray, rho: float = 0.35) -> np.ndarray:
    """
    Build a simple correlated covariance matrix.

    C_ij = sigma_i sigma_j rho^{|i-j|}
    """
    errors = np.asarray(errors, dtype=float)
    n = errors.shape[0]
    cov = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            cov[i, j] = errors[i] * errors[j] * (rho ** abs(i - j))
    return cov


def make_pseudodata(
    calculator: ToyDVCSObservableCalculator,
    kinematics: KinematicsBatch,
    true_cffs: dict[str, np.ndarray],
    seed: int = 12345,
) -> dict[str, np.ndarray]:
    """
    Generate noisy toy pseudodata for cross section and BSA.
    """
    rng = np.random.default_rng(seed)

    sigma_true = calculator.compute("cross_section_uu", kinematics, true_cffs)
    bsa_true = calculator.compute("beam_spin_asymmetry", kinematics, true_cffs)

    # A realistic-enough toy error model.
    sigma_err = 0.05 * sigma_true + 0.03
    bsa_err = 0.03 + 0.10 * np.abs(bsa_true) + 0.01

    sigma_cov = make_correlated_covariance(sigma_err, rho=0.35)
    bsa_cov = np.diag(bsa_err**2)

    sigma_data = rng.multivariate_normal(mean=sigma_true, cov=sigma_cov)
    bsa_data = rng.multivariate_normal(mean=bsa_true, cov=bsa_cov)

    return {
        "sigma_true": sigma_true,
        "bsa_true": bsa_true,
        "sigma_data": sigma_data,
        "bsa_data": bsa_data,
        "sigma_err": sigma_err,
        "bsa_err": bsa_err,
        "sigma_cov": sigma_cov,
        "bsa_cov": bsa_cov,
    }


def stack_observables(sigma: np.ndarray, bsa: np.ndarray) -> np.ndarray:
    """
    Concatenate observables into a single vector for a global chi2.
    """
    return np.concatenate([np.asarray(sigma, dtype=float), np.asarray(bsa, dtype=float)])


def stack_covariances(sigma_cov: np.ndarray, bsa_cov: np.ndarray) -> np.ndarray:
    """
    Block-diagonal covariance for stacked observables.
    """
    return block_diag(sigma_cov, bsa_cov)


def chi2_scan_1d(
    calculator: ToyDVCSObservableCalculator,
    kinematics: KinematicsBatch,
    data: dict[str, np.ndarray],
    true_cffs: dict[str, np.ndarray],
    scan_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute χ² scans versus a single scaling parameter applied to H_imag.

    Returns:
        chi2_values, chi2_profiled_values
    """
    chi2_values = []
    chi2_profiled_values = []

    for s in scan_values:
        test_cffs = scale_toy_cffs(true_cffs, h_imag_scale=s)

        sigma_th = calculator.compute("cross_section_uu", kinematics, test_cffs)
        llh_sigma = GaussianLikelihood(
            data=data["sigma_data"],
            theory=sigma_th,
            covariance=data["sigma_cov"],
        )
        chi2_values.append(llh_sigma.chi2())

        # Profile an overall normalization nuisance on the cross-section block.
        chi2_prof, _eta_hat = llh_sigma.profile_normalization(norm_sigma=0.03)
        chi2_profiled_values.append(chi2_prof)

    return np.asarray(chi2_values), np.asarray(chi2_profiled_values)


def chi2_scan_2d(
    calculator: ToyDVCSObservableCalculator,
    kinematics: KinematicsBatch,
    data: dict[str, np.ndarray],
    true_cffs: dict[str, np.ndarray],
    hr_values: np.ndarray,
    hi_values: np.ndarray,
) -> np.ndarray:
    """
    Compute a 2D χ² surface using stacked cross section + BSA observables.
    """
    sigma_data = data["sigma_data"]
    bsa_data = data["bsa_data"]
    cov = stack_covariances(data["sigma_cov"], data["bsa_cov"])
    data_stack = stack_observables(sigma_data, bsa_data)

    chi2_grid = np.zeros((len(hi_values), len(hr_values)), dtype=float)

    for i, hi_scale in enumerate(hi_values):
        for j, hr_scale in enumerate(hr_values):
            test_cffs = scale_toy_cffs(
                true_cffs,
                h_real_scale=hr_scale,
                h_imag_scale=hi_scale,
            )
            sigma_th = calculator.compute("cross_section_uu", kinematics, test_cffs)
            bsa_th = calculator.compute("beam_spin_asymmetry", kinematics, test_cffs)
            theory_stack = stack_observables(sigma_th, bsa_th)
            llh = GaussianLikelihood(data_stack, theory_stack, covariance=cov)
            chi2_grid[i, j] = llh.chi2()

    return chi2_grid


def plot_outputs(
    outdir: Path,
    kinematics: KinematicsBatch,
    data: dict[str, np.ndarray],
    true_cffs: dict[str, np.ndarray],
    calculator: ToyDVCSObservableCalculator,
    chi2_1d: np.ndarray,
    chi2_1d_prof: np.ndarray,
    scan_values: np.ndarray,
    chi2_2d: np.ndarray,
    hr_values: np.ndarray,
    hi_values: np.ndarray,
) -> None:
    """
    Create and save the diagnostic plots.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    sigma_true = data["sigma_true"]
    bsa_true = data["bsa_true"]
    sigma_data = data["sigma_data"]
    bsa_data = data["bsa_data"]

    # Best-fit point from 1D scan for the sigma channel.
    best_idx = int(np.argmin(chi2_1d))
    best_hi_scale = float(scan_values[best_idx])
    best_cffs_1d = scale_toy_cffs(true_cffs, h_imag_scale=best_hi_scale)
    sigma_best = calculator.compute("cross_section_uu", kinematics, best_cffs_1d)
    bsa_best_1d = calculator.compute("beam_spin_asymmetry", kinematics, best_cffs_1d)

    # Best-fit point from the 2D surface.
    flat_idx = int(np.argmin(chi2_2d))
    best_i, best_j = np.unravel_index(flat_idx, chi2_2d.shape)
    best_hr_scale = float(hr_values[best_j])
    best_hi_scale_2d = float(hi_values[best_i])
    best_cffs_2d = scale_toy_cffs(
        true_cffs,
        h_real_scale=best_hr_scale,
        h_imag_scale=best_hi_scale_2d,
    )
    sigma_best_2d = calculator.compute("cross_section_uu", kinematics, best_cffs_2d)
    bsa_best_2d = calculator.compute("beam_spin_asymmetry", kinematics, best_cffs_2d)

    # Figure 1: data vs truth vs best-fit
    fig1, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    axes[0].errorbar(
        np.arange(kinematics.n_points),
        sigma_data,
        yerr=np.sqrt(np.diag(data["sigma_cov"])),
        fmt="o",
        label="Pseudodata",
    )
    axes[0].plot(np.arange(kinematics.n_points), sigma_true, label="Truth", linewidth=2)
    axes[0].plot(np.arange(kinematics.n_points), sigma_best, label="Best 1D fit", linewidth=2, linestyle="--")
    axes[0].plot(np.arange(kinematics.n_points), sigma_best_2d, label="Best 2D fit", linewidth=2, linestyle=":")
    axes[0].set_ylabel("Cross section (toy units)")
    axes[0].set_title("Toy cross section: pseudodata vs truth vs fit")
    axes[0].legend()

    axes[1].errorbar(
        np.arange(kinematics.n_points),
        bsa_data,
        yerr=np.sqrt(np.diag(data["bsa_cov"])),
        fmt="o",
        label="Pseudodata",
    )
    axes[1].plot(np.arange(kinematics.n_points), bsa_true, label="Truth", linewidth=2)
    axes[1].plot(np.arange(kinematics.n_points), bsa_best_1d, label="Best 1D fit", linewidth=2, linestyle="--")
    axes[1].plot(np.arange(kinematics.n_points), bsa_best_2d, label="Best 2D fit", linewidth=2, linestyle=":")
    axes[1].set_ylabel("BSA (toy units)")
    axes[1].set_xlabel("Data point index")
    axes[1].set_title("Toy beam-spin asymmetry: pseudodata vs truth vs fit")
    axes[1].legend()

    fig1.tight_layout()
    fig1.savefig(outdir / "toy_observables_fit.png", dpi=200)
    fig1.savefig(outdir / "toy_observables_fit.pdf")
    plt.close(fig1)

    # Figure 2: 1D χ² scan
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(scan_values, chi2_1d, label=r"$\chi^2$ (cross section only)")
    ax2.plot(scan_values, chi2_1d_prof, label=r"$\chi^2$ profiled over normalization")
    ax2.axvline(1.0, color="black", linestyle="--", label="Truth")
    ax2.axvline(best_hi_scale, color="tab:red", linestyle=":", label="Best unprofiled")
    ax2.set_xlabel(r"Scaling of $H_{\mathrm{imag}}$")
    ax2.set_ylabel(r"$\chi^2$")
    ax2.set_title(r"1D $\chi^2$ scan in $H_{\mathrm{imag}}$ scaling")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(outdir / "toy_chi2_scan_1d.png", dpi=200)
    fig2.savefig(outdir / "toy_chi2_scan_1d.pdf")
    plt.close(fig2)

    # Figure 3: 2D χ² surface
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    levels = 25
    cs = ax3.contourf(hr_values, hi_values, chi2_2d, levels=levels, cmap="viridis")
    cbar = fig3.colorbar(cs, ax=ax3)
    cbar.set_label(r"$\chi^2$")
    ax3.scatter([1.0], [1.0], marker="x", s=100, color="white", label="Truth")
    ax3.scatter([best_hr_scale], [best_hi_scale_2d], marker="o", s=70, color="red", label="Best fit")
    ax3.set_xlabel(r"$H_{\mathrm{real}}$ scale")
    ax3.set_ylabel(r"$H_{\mathrm{imag}}$ scale")
    ax3.set_title(r"2D $\chi^2$ contour from stacked observables")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(outdir / "toy_chi2_contour_2d.png", dpi=200)
    fig3.savefig(outdir / "toy_chi2_contour_2d.pdf")
    plt.close(fig3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy DVCS χ² diagnostics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "toy_scan",
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=24,
        help="Number of kinematic points in the toy batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for pseudodata generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    kinematics = build_kinematics(args.n_points)
    calculator = ToyDVCSObservableCalculator(backend="toy")
    true_cffs = generate_toy_cffs(kinematics, ToyCFFParameters())

    data = make_pseudodata(calculator, kinematics, true_cffs, seed=args.seed)

    # 1D scan over H_imag scaling.
    scan_values = np.linspace(0.6, 1.4, 121)
    chi2_1d, chi2_1d_prof = chi2_scan_1d(
        calculator=calculator,
        kinematics=kinematics,
        data=data,
        true_cffs=true_cffs,
        scan_values=scan_values,
    )

    # 2D scan over H_real and H_imag scaling.
    hr_values = np.linspace(0.7, 1.3, 61)
    hi_values = np.linspace(0.7, 1.3, 61)
    chi2_2d = chi2_scan_2d(
        calculator=calculator,
        kinematics=kinematics,
        data=data,
        true_cffs=true_cffs,
        hr_values=hr_values,
        hi_values=hi_values,
    )

    plot_outputs(
        outdir=args.output,
        kinematics=kinematics,
        data=data,
        true_cffs=true_cffs,
        calculator=calculator,
        chi2_1d=chi2_1d,
        chi2_1d_prof=chi2_1d_prof,
        scan_values=scan_values,
        chi2_2d=chi2_2d,
        hr_values=hr_values,
        hi_values=hi_values,
    )

    best_1d = scan_values[int(np.argmin(chi2_1d))]
    best_2d_idx = int(np.argmin(chi2_2d))
    best_2d_i, best_2d_j = np.unravel_index(best_2d_idx, chi2_2d.shape)

    print(f"Saved plots to: {args.output}")
    print(f"Best 1D H_imag scale: {best_1d:.4f}")
    print(f"Best 2D scales: H_real={hr_values[best_2d_j]:.4f}, H_imag={hi_values[best_2d_i]:.4f}")
    print(f"Minimum 2D chi2: {chi2_2d[best_2d_i, best_2d_j]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())