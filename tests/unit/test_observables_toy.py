

import numpy as np
import pytest

from extract_dvcs_cff.physics.observables import (
    KinematicsBatch,
    ToyDVCSObservableCalculator,
    generate_toy_cffs,
    map_observable_label,
    is_supported_observable,
)
from extract_dvcs_cff.physics.likelihood import GaussianLikelihood




def _make_test_batch():
    """Helper to create a standard test kinematics batch."""
    return KinematicsBatch.from_sequences(
        xB=[0.2, 0.25, 0.3],
        Q2=[2.0, 2.5, 3.0],
        t=[-0.1, -0.2, -0.3],
        phi=[0.3, 1.2, 2.4],
    )



def test_observable_shapes_and_ranges():
    """
    Test that main toy observables return correct shapes and physical ranges.
    """
    batch = _make_test_batch()
    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()
    observables = [
        "cross_section_uu",
        "beam_spin_asymmetry",
        "beam_charge_asymmetry",
    ]
    for obs in observables:
        out = calc.compute(obs, batch, cffs)
        assert out.shape == (batch.n_points,)
        assert np.all(np.isfinite(out))
        if "cross_section" in obs:
            assert np.all(out > 0)
        if "asymmetry" in obs:
            assert np.all((out >= -1.0) & (out <= 1.0))



def test_likelihood_pipeline():
    """
    Test that toy observable output is compatible with GaussianLikelihood.
    """
    batch = _make_test_batch()
    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()
    sigma = calc.compute("cross_section_uu", batch, cffs)
    errors = 0.05 * sigma + 0.02
    llh = GaussianLikelihood(sigma, sigma, stat_errors=errors)
    assert np.isclose(llh.chi2(), 0.0, atol=1e-12)
    assert np.isclose(llh.log_likelihood(), -0.5 * llh.chi2(), atol=1e-12)



def test_determinism():
    """
    The observable function must be deterministic: same inputs → identical outputs.
    """
    batch = _make_test_batch()
    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()
    pred1 = calc.compute("cross_section_uu", batch, cffs)
    pred2 = calc.compute("cross_section_uu", batch, cffs)
    assert np.allclose(pred1, pred2, atol=1e-12)



def test_continuity_in_kinematics():
    """
    Observables should vary smoothly with kinematics.
    """
    batch = _make_test_batch()
    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()
    epsilon = 1e-6
    direction = np.array([1.0, -1.0, 0.5])
    batch_shifted = KinematicsBatch(
        xB=batch.xB + epsilon * direction[0],
        Q2=batch.Q2 + epsilon * direction[1],
        t=batch.t + epsilon * direction[2],
        phi=batch.phi,
    )
    pred = calc.compute("cross_section_uu", batch, cffs)
    pred_shifted = calc.compute("cross_section_uu", batch_shifted, cffs)
    delta = np.abs(pred_shifted - pred)
    assert np.all(delta < 1e-4 * (1.0 + np.abs(pred)))



def test_sensitivity_to_cffs():
    """
    Observables should respond smoothly and nontrivially to CFF perturbations.
    """
    batch = _make_test_batch()
    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()
    pred = calc.compute("cross_section_uu", batch, cffs)
    for key in cffs:
        cffs_shifted = dict(cffs)
        cffs_shifted[key] = cffs[key] + 1e-6
        pred_shifted = calc.compute("cross_section_uu", batch, cffs_shifted)
        delta = np.abs(pred_shifted - pred)
        assert np.all(np.isfinite(delta))
        assert np.any(delta > 0)
        relative_change = delta / (np.abs(pred) + 1e-12)
        assert np.all(relative_change < 1e-2)



def test_observable_label_mapping():
    """
    Test that database labels and canonical names map correctly.
    """
    cases = [
        ("BSA", "beam_spin_asymmetry"),
        ("CrossSectionUU", "cross_section_uu"),
        ("beam_spin_asymmetry", "beam_spin_asymmetry"),
        ("t_slope", "t_slope"),
    ]
    for label, expected in cases:
        mapped = map_observable_label(label)
        assert mapped == expected or mapped == expected.lower()
    assert is_supported_observable("BSA")
    assert is_supported_observable("beam_spin_asymmetry")
    assert not is_supported_observable("nonsense_label")



def test_compute_accepts_database_labels():
    """
    Test that compute() accepts both database and canonical labels.
    """
    batch = _make_test_batch()
    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()
    out1 = calc.compute("BSA", batch, cffs)
    out2 = calc.compute("beam_spin_asymmetry", batch, cffs)
    assert np.allclose(out1, out2, atol=1e-12)



def test_compute_all_outputs():
    """
    Test that compute_all returns all expected observables with correct shapes.
    """
    batch = _make_test_batch()
    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()
    out = calc.compute_all(batch, cffs)
    expected = [
        "cross_section_uu",
        "cross_section_difference_lu",
        "cross_section_uu_virtual_photoproduction",
        "beam_spin_asymmetry",
        "beam_charge_asymmetry",
        "double_spin_asymmetry",
        "t_slope",
    ]
    for key in expected:
        assert key in out
        arr = out[key]
        assert arr.shape == (batch.n_points,)
        assert np.all(np.isfinite(arr))



def test_input_validation():
    """
    Test that invalid kinematics and CFFs raise ValueError.
    """
    batch = _make_test_batch()
    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()
    # Bad kinematics: mismatched lengths
    with pytest.raises(ValueError):
        KinematicsBatch.from_sequences(
            xB=[0.2, 0.25], Q2=[2.0, 2.5, 3.0], t=[-0.1, -0.2, -0.3], phi=[0.3, 1.2, 2.4]
        )
    # Bad kinematics: xB out of range
    with pytest.raises(ValueError):
        KinematicsBatch.from_sequences(
            xB=[-0.1, 0.25, 0.3], Q2=[2.0, 2.5, 3.0], t=[-0.1, -0.2, -0.3], phi=[0.3, 1.2, 2.4]
        )
    # Bad kinematics: t > 0
    with pytest.raises(ValueError):
        KinematicsBatch.from_sequences(
            xB=[0.2, 0.25, 0.3], Q2=[2.0, 2.5, 3.0], t=[0.1, -0.2, -0.3], phi=[0.3, 1.2, 2.4]
        )
    # Bad kinematics: phi out of range
    with pytest.raises(ValueError):
        KinematicsBatch.from_sequences(
            xB=[0.2, 0.25, 0.3], Q2=[2.0, 2.5, 3.0], t=[-0.1, -0.2, -0.3], phi=[-1.0, 1.2, 2.4]
        )
    # Bad CFFs: wrong length
    with pytest.raises(ValueError):
        bad_cffs = dict(cffs)
        bad_cffs["H_real"] = bad_cffs["H_real"][:2]
        calc.compute("cross_section_uu", batch, bad_cffs)


def test_toy_likelihood_pipeline():
    batch = KinematicsBatch.from_sequences(
        xB=[0.2, 0.25, 0.3],
        Q2=[2.0, 2.5, 3.0],
        t=[-0.1, -0.2, -0.3],
        phi=[0.3, 1.2, 2.4],
    )
    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()

    sigma = calc.compute_cross_section(batch, cffs)
    errors = 0.05 * sigma + 0.02
    llh = GaussianLikelihood(sigma, sigma, stat_errors=errors)

    assert np.isclose(llh.chi2(), 0.0, atol=1e-12)
    assert np.isclose(llh.log_likelihood(), 0.0, atol=1e-12)

def test_observables_deterministic():
    """
    The observable function must be deterministic:
    same inputs → identical outputs.
    """
    batch = KinematicsBatch.from_sequences(
        xB=[0.2, 0.25, 0.3],
        Q2=[2.0, 2.5, 3.0],
        t=[-0.1, -0.2, -0.3],
        phi=[0.3, 1.2, 2.4],
    )

    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()

    pred1 = calc.compute_cross_section(batch, cffs)
    pred2 = calc.compute_cross_section(batch, cffs)

    assert np.allclose(pred1, pred2, atol=1e-12)

def test_observables_smoothness_kin():
    """
    Observables should vary smoothly with kinematics.
    Small change in kinematics → small change in output.
    """
    batch = KinematicsBatch.from_sequences(
        xB=[0.2, 0.25, 0.3],
        Q2=[2.0, 2.5, 3.0],
        t=[-0.1, -0.2, -0.3],
        phi=[0.3, 1.2, 2.4],
    )

    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()

    epsilon = 1e-6
    direction = np.array([1.0, -1.0, 0.5])

    pred = calc.compute_cross_section(batch, cffs)

    batch_shifted = KinematicsBatch(
        xB=batch.xB + epsilon * direction[0],
        Q2=batch.Q2 + epsilon * direction[1],
        t=batch.t + epsilon * direction[2],
        phi=batch.phi,
    )
    
    pred_shifted = calc.compute_cross_section(batch_shifted, cffs)

    delta = np.abs(pred_shifted - pred)

    assert np.all(delta < 1e-4 * (1.0 + np.abs(pred)))

def test_observable_smoothness_in_cffs():
    batch = KinematicsBatch.from_sequences(
        xB=[0.2, 0.25, 0.3],
        Q2=[2.0, 2.5, 3.0],
        t=[-0.1, -0.2, -0.3],
        phi=[0.3, 1.2, 2.4],
    )

    cffs = generate_toy_cffs(batch)
    calc = ToyDVCSObservableCalculator()

    pred = calc.compute_cross_section(batch, cffs)

    for key in cffs:
        cffs_shifted = dict(cffs)
        cffs_shifted[key] = cffs[key] + 1e-6

        pred_shifted = calc.compute_cross_section(batch, cffs_shifted)

        delta = np.abs(pred_shifted - pred)

        assert np.all(np.isfinite(delta))
        assert np.any(delta > 0)   # must affect observable

        relative_change = delta / (np.abs(pred) + 1e-12)
        assert np.all(relative_change < 1e-2)