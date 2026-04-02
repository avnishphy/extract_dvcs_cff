"""
Likelihood computation for DVCS/GPD/CFF analysis.

This module implements a Gaussian likelihood for comparing theory predictions
to experimental observables, with support for either:
- diagonal uncertainties from statistical/systematic errors, or
- a full covariance matrix.

It also provides analytic profiling over a multiplicative normalization nuisance
parameter, which is a standard treatment for overall normalization uncertainties
in high-energy physics analyses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError


class GaussianLikelihood:
    """
    Gaussian likelihood for observable vectors.

    The likelihood is

        L(d | t) ∝ exp[-1/2 (d - t)^T C^{-1} (d - t)]

    where:
      - d is the data vector,
      - t is the theory vector,
      - C is the covariance matrix.

    Parameters
    ----------
    data:
        Experimental measurements as a 1D array.
    theory:
        Theory/model predictions as a 1D array of the same shape as data.
    stat_errors:
        Statistical uncertainties as a 1D array. Required if covariance is not supplied.
    sys_errors:
        Systematic uncertainties as a 1D array. Optional if covariance is not supplied.
    covariance:
        Full covariance matrix. If provided, stat_errors/sys_errors must not also be provided.
    """

    def __init__(
        self,
        data: np.ndarray,
        theory: np.ndarray,
        stat_errors: Optional[np.ndarray] = None,
        sys_errors: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> None:
        self.data = np.asarray(data, dtype=float)
        self.theory = np.asarray(theory, dtype=float)
        self.stat_errors = None if stat_errors is None else np.asarray(stat_errors, dtype=float)
        self.sys_errors = None if sys_errors is None else np.asarray(sys_errors, dtype=float)
        self.covariance = None if covariance is None else np.asarray(covariance, dtype=float)

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate shapes, finiteness, and covariance consistency."""
        if self.data.ndim != 1:
            raise ValueError("data must be a 1D array.")
        if self.theory.ndim != 1:
            raise ValueError("theory must be a 1D array.")
        if self.data.shape != self.theory.shape:
            raise ValueError("data and theory must have the same shape.")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("data contains non-finite values.")
        if not np.all(np.isfinite(self.theory)):
            raise ValueError("theory contains non-finite values.")

        n = self.data.shape[0]

        if self.covariance is not None:
            if self.stat_errors is not None or self.sys_errors is not None:
                raise ValueError(
                    "Provide either covariance or stat_errors/sys_errors, not both."
                )
            if self.covariance.ndim != 2:
                raise ValueError("covariance must be a 2D square matrix.")
            if self.covariance.shape[0] != self.covariance.shape[1]:
                raise ValueError("covariance must be square.")
            if self.covariance.shape[0] != n:
                raise ValueError("covariance size must match data/theory length.")
            if not np.all(np.isfinite(self.covariance)):
                raise ValueError("covariance contains non-finite values.")
            if not np.allclose(self.covariance, self.covariance.T, atol=1e-12, rtol=0.0):
                raise ValueError("covariance matrix must be symmetric.")
            self._validate_positive_definite(self.covariance)

        else:
            if self.stat_errors is None:
                raise ValueError("stat_errors must be provided if covariance is not given.")
            if self.stat_errors.ndim != 1:
                raise ValueError("stat_errors must be a 1D array.")
            if self.stat_errors.shape != (n,):
                raise ValueError("stat_errors must have the same length as data.")
            if not np.all(np.isfinite(self.stat_errors)):
                raise ValueError("stat_errors contains non-finite values.")
            if np.any(self.stat_errors <= 0):
                raise ValueError("stat_errors must be strictly positive.")

            if self.sys_errors is not None:
                if self.sys_errors.ndim != 1:
                    raise ValueError("sys_errors must be a 1D array.")
                if self.sys_errors.shape != (n,):
                    raise ValueError("sys_errors must have the same length as data.")
                if not np.all(np.isfinite(self.sys_errors)):
                    raise ValueError("sys_errors contains non-finite values.")
                if np.any(self.sys_errors < 0):
                    raise ValueError("sys_errors must be non-negative.")

    @staticmethod
    def _validate_positive_definite(covariance: np.ndarray) -> None:
        """Raise ValueError if covariance is not positive definite."""
        try:
            cho_factor(covariance, check_finite=True)
        except LinAlgError as exc:
            raise ValueError("covariance matrix must be positive definite.") from exc

    def residuals(self) -> np.ndarray:
        """Return residuals d - t."""
        return self.data - self.theory

    def covariance_matrix(self) -> np.ndarray:
        """
        Return the total covariance matrix.

        Returns
        -------
        np.ndarray
            The covariance matrix used by the likelihood.
        """
        if self.covariance is not None:
            return self.covariance

        var = self.stat_errors**2
        if self.sys_errors is not None:
            var = var + self.sys_errors**2
        return np.diag(var)

    @staticmethod
    def _quadratic_form(vector: np.ndarray, covariance: np.ndarray) -> float:
        """
        Compute v^T C^{-1} v using a Cholesky factorization.
        """
        try:
            c_factor, lower = cho_factor(covariance, check_finite=True)
            return float(vector.T @ cho_solve((c_factor, lower), vector))
        except LinAlgError as exc:
            raise ValueError("covariance matrix must be positive definite.") from exc

    def chi2(self) -> float:
        """Return chi-squared: (d - t)^T C^{-1} (d - t)."""
        return self._quadratic_form(self.residuals(), self.covariance_matrix())

    def log_likelihood(self, include_constant: bool = False) -> float:
        """
        Return the Gaussian log-likelihood.

        Parameters
        ----------
        include_constant:
            If False, return -0.5 * chi2.
            If True, include the Gaussian normalization term:
                -0.5 * [chi2 + log(det(C)) + N log(2π)]
        """
        chi2_val = self.chi2()
        if not include_constant:
            return -0.5 * chi2_val

        logdet = self.logdet_covariance()
        n = self.data.shape[0]
        return -0.5 * (chi2_val + logdet + n * np.log(2.0 * np.pi))

    def logdet_covariance(self) -> float:
        """Return log(det(C)) computed stably via Cholesky decomposition."""
        covariance = self.covariance_matrix()
        try:
            c_factor, lower = cho_factor(covariance, check_finite=True)
        except LinAlgError as exc:
            raise ValueError("covariance matrix must be positive definite.") from exc

        diag = np.diag(c_factor)
        if np.any(diag <= 0):
            raise ValueError("Invalid Cholesky factor encountered.")
        return 2.0 * float(np.sum(np.log(diag)))

    def standardized_residuals(self) -> np.ndarray:
        """
        Return diagonal pull values: (d - t) / sqrt(diag(C)).

        For correlated covariances, this is a useful diagnostic but not a full
        whitening transformation.
        """
        covariance = self.covariance_matrix()
        diag = np.diag(covariance)
        if np.any(diag <= 0):
            raise ValueError("Covariance diagonal must be strictly positive.")
        return self.residuals() / np.sqrt(diag)

    def reduced_chi2(self, n_params: int) -> float:
        """
        Return reduced chi2 = chi2 / dof.

        Raises
        ------
        ValueError
            If n_params >= N.
        """
        n = self.data.shape[0]
        dof = n - n_params
        if dof <= 0:
            raise ValueError("Degrees of freedom must be positive (n_params < N).")
        return self.chi2() / dof

    def profile_normalization(self, norm_sigma: float) -> Tuple[float, float]:
        """
        Profile out a multiplicative normalization nuisance parameter eta.

        The model is:
            t -> (1 + eta) t

        The profiled objective is:
            chi2(eta) = (d - (1 + eta)t)^T C^{-1} (d - (1 + eta)t) + (eta / norm_sigma)^2

        The best-fit eta is obtained analytically.

        Parameters
        ----------
        norm_sigma:
            Standard deviation of the normalization nuisance parameter.

        Returns
        -------
        profiled_chi2:
            Minimum chi2 after profiling over eta.
        eta_hat:
            Best-fit normalization shift.

        Notes
        -----
        This is a standard profile-likelihood treatment for an overall
        multiplicative normalization uncertainty.
        """
        if norm_sigma <= 0:
            raise ValueError("norm_sigma must be positive.")

        covariance = self.covariance_matrix()
        d = self.data
        t = self.theory

        try:
            c_factor, lower = cho_factor(covariance, check_finite=True)
            Cinv_d = cho_solve((c_factor, lower), d)
            Cinv_t = cho_solve((c_factor, lower), t)
        except LinAlgError as exc:
            raise ValueError("covariance matrix must be positive definite.") from exc

        # Expand chi2(eta) as:
        # chi2(eta) = A eta^2 - 2 B eta + const
        # where:
        #   A = t^T C^-1 t + 1/sigma^2
        #   B = t^T C^-1 (d - t)
        #
        # The minimizer is eta_hat = B / A.
        A = float(t.T @ Cinv_t + 1.0 / (norm_sigma**2))
        B = float(t.T @ (Cinv_d - Cinv_t))

        eta_hat = B / A

        residual = d - (1.0 + eta_hat) * t
        profiled_chi2 = float(residual.T @ cho_solve((c_factor, lower), residual) + (eta_hat / norm_sigma) ** 2)

        return profiled_chi2, eta_hat


def _load_likelihood_payload(spec: Path | str | dict) -> dict:
    if isinstance(spec, dict):
        return spec

    path = Path(spec)
    if not path.exists():
        raise ValueError(f"Likelihood config file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ValueError(
                "YAML likelihood config requires PyYAML. Install pyyaml or use JSON."
            ) from exc
        payload = yaml.safe_load(path.read_text())
        return {} if payload is None else payload

    raise ValueError("Likelihood config must be JSON or YAML.")


def compute_likelihood(spec: Path | str | dict, include_constant: bool = False) -> float:
    """
    Compute Gaussian log-likelihood from a lightweight spec.

    Required fields:
      - data
      - theory

    One uncertainty specification must be supplied:
      - covariance
      OR
      - stat_errors (and optional sys_errors)
    """
    payload = _load_likelihood_payload(spec)

    if "data" not in payload or "theory" not in payload:
        raise ValueError("Likelihood spec must include 'data' and 'theory'.")

    data = np.asarray(payload["data"], dtype=float)
    theory = np.asarray(payload["theory"], dtype=float)
    stat_errors = payload.get("stat_errors")
    sys_errors = payload.get("sys_errors")
    covariance = payload.get("covariance")

    llh = GaussianLikelihood(
        data=data,
        theory=theory,
        stat_errors=None if stat_errors is None else np.asarray(stat_errors, dtype=float),
        sys_errors=None if sys_errors is None else np.asarray(sys_errors, dtype=float),
        covariance=None if covariance is None else np.asarray(covariance, dtype=float),
    )
    return llh.log_likelihood(include_constant=include_constant)