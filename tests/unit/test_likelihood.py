"""
Unit tests for the Gaussian likelihood used in DVCS/GPD/CFF analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from extract_dvcs_cff.physics.likelihood import GaussianLikelihood


def test_diagonal_covariance_exact_chi2():
    """
    Verify exact residuals, diagonal covariance construction, chi2,
    log-likelihood, and pulls when only stat/sys errors are provided.
    """
    data = np.array([1.0, 2.0, 3.0])
    theory = np.array([1.1, 1.9, 3.2])
    stat_errors = np.array([0.1, 0.2, 0.1])
    sys_errors = np.array([0.05, 0.05, 0.05])

    llh = GaussianLikelihood(data, theory, stat_errors=stat_errors, sys_errors=sys_errors)

    residuals = data - theory
    expected_var = stat_errors**2 + sys_errors**2
    expected_cov = np.diag(expected_var)

    assert np.allclose(llh.residuals(), residuals)
    assert np.allclose(llh.covariance_matrix(), expected_cov)

    expected_chi2 = np.sum((residuals**2) / expected_var)
    assert np.isclose(llh.chi2(), expected_chi2, atol=1e-12)
    assert np.isclose(llh.log_likelihood(), -0.5 * expected_chi2, atol=1e-12)

    expected_pulls = residuals / np.sqrt(expected_var)
    assert np.allclose(llh.standardized_residuals(), expected_pulls)


def test_full_covariance_exact_chi2():
    """
    Verify chi2 and log-likelihood against a manual computation
    using a full positive-definite covariance matrix.
    """
    data = np.array([1.0, 2.0])
    theory = np.array([0.9, 2.1])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])

    llh = GaussianLikelihood(data, theory, covariance=cov)

    residual = data - theory
    expected_chi2 = float(residual.T @ np.linalg.solve(cov, residual))

    assert np.allclose(llh.covariance_matrix(), cov)
    assert np.isclose(llh.chi2(), expected_chi2, atol=1e-12)
    assert np.isclose(llh.log_likelihood(), -0.5 * expected_chi2, atol=1e-12)


def test_log_likelihood_with_constant_term():
    """
    Verify that the optional Gaussian normalization term is included correctly.
    """
    data = np.array([1.0, 2.0])
    theory = np.array([0.9, 2.1])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])

    llh = GaussianLikelihood(data, theory, covariance=cov)

    residual = data - theory
    chi2 = float(residual.T @ np.linalg.solve(cov, residual))
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0

    expected = -0.5 * (chi2 + logdet + len(data) * np.log(2.0 * np.pi))
    assert np.isclose(llh.log_likelihood(include_constant=True), expected, atol=1e-12)


def test_profiled_normalization():
    """
    Verify analytic profiling over a multiplicative normalization nuisance parameter.
    """
    data = np.array([1.0, 2.0])
    theory = np.array([1.2, 1.8])
    stat_errors = np.array([0.1, 0.1])
    norm_sigma = 0.05

    llh = GaussianLikelihood(data, theory, stat_errors=stat_errors)

    cov = np.diag(stat_errors**2)
    Cinv = np.linalg.inv(cov)
    d = data
    t = theory

    # Analytic solution:
    # chi2(eta) = (d - (1 + eta)t)^T C^-1 (d - (1 + eta)t) + (eta/sigma)^2
    # eta_hat = [t^T C^-1 (d - t)] / [t^T C^-1 t + 1/sigma^2]
    A = float(t.T @ Cinv @ t + 1.0 / (norm_sigma**2))
    B = float(t.T @ Cinv @ (d - t))
    eta_hat_expected = B / A

    residual = d - (1.0 + eta_hat_expected) * t
    chi2_expected = float(residual.T @ Cinv @ residual + (eta_hat_expected / norm_sigma) ** 2)

    chi2_profiled, eta_hat = llh.profile_normalization(norm_sigma)

    assert np.isclose(eta_hat, eta_hat_expected, atol=1e-12)
    assert np.isclose(chi2_profiled, chi2_expected, atol=1e-12)
    assert chi2_profiled <= llh.chi2() + 1e-12


def test_shape_mismatch_raises():
    """
    Data and theory vectors must have the same shape.
    """
    data = np.array([1.0, 2.0])
    theory = np.array([1.0, 2.0, 3.0])
    stat_errors = np.array([0.1, 0.1])

    with pytest.raises(ValueError, match="same shape"):
        GaussianLikelihood(data, theory, stat_errors=stat_errors)


def test_invalid_covariance_shape_raises():
    """
    Covariance must be square and match the observable dimension.
    """
    data = np.array([1.0, 2.0])
    theory = np.array([1.0, 2.0])

    bad_cov = np.array([[0.1, 0.0]])
    with pytest.raises(ValueError, match="square"):
        GaussianLikelihood(data, theory, covariance=bad_cov)

    bad_cov2 = np.eye(3)
    with pytest.raises(ValueError, match="size"):
        GaussianLikelihood(data, theory, covariance=bad_cov2)


def test_missing_uncertainties_raises():
    """
    If no covariance is given, stat_errors are required.
    """
    data = np.array([1.0, 2.0])
    theory = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="stat_errors must be provided"):
        GaussianLikelihood(data, theory)


def test_nonfinite_input_raises():
    """
    NaN and inf values must be rejected.
    """
    data_nan = np.array([1.0, np.nan])
    theory = np.array([1.0, 2.0])
    stat_errors = np.array([0.1, 0.1])

    with pytest.raises(ValueError, match="data contains non-finite values"):
        GaussianLikelihood(data_nan, theory, stat_errors=stat_errors)

    with pytest.raises(ValueError, match="theory contains non-finite values"):
        GaussianLikelihood(theory, data_nan, stat_errors=stat_errors)

    with pytest.raises(ValueError, match="non-finite values"):
        GaussianLikelihood(theory, theory, stat_errors=np.array([0.1, np.inf]))

    bad_cov = np.array([[0.1, np.nan], [0.0, 0.1]])
    with pytest.raises(ValueError, match="non-finite values"):
        GaussianLikelihood(theory, theory, covariance=bad_cov)


def test_reduced_chi2():
    """
    Check reduced chi2 and error handling when degrees of freedom are non-positive.
    """
    data = np.array([1.0, 2.0, 3.0])
    theory = np.array([1.0, 2.0, 3.0])
    stat_errors = np.array([0.1, 0.1, 0.1])

    llh = GaussianLikelihood(data, theory, stat_errors=stat_errors)

    assert np.isclose(llh.reduced_chi2(1), 0.0, atol=1e-12)

    with pytest.raises(ValueError, match="Degrees of freedom"):
        llh.reduced_chi2(3)


def test_covariance_symmetry_validation():
    """
    Covariance matrices must be symmetric.
    """
    data = np.array([1.0, 2.0])
    theory = np.array([1.0, 2.0])
    nonsym = np.array([[1.0, 0.1], [0.0, 1.0]])

    with pytest.raises(ValueError, match="symmetric"):
        GaussianLikelihood(data, theory, covariance=nonsym)


def test_positive_definite_covariance_required():
    """
    Singular or indefinite covariance matrices must be rejected.
    """
    data = np.array([1.0, 2.0])
    theory = np.array([1.0, 2.0])

    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        GaussianLikelihood(data, theory, covariance=singular)