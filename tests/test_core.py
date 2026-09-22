"""
tests/test_core.py — automated correctness checks for the drift_monitor
package. Run with:

    pytest tests/ -v

These don't replace the evaluation grid (which measures how WELL the
system detects real shift) -- they check that the building blocks behave
correctly in isolation, on cases simple enough to know the right answer
by construction. Everything here was previously only checked by eyeballing
plots and printed summaries; this is the same logic, made repeatable and
automatic.
"""

import numpy as np
import pandas as pd
import pytest

from drift_monitor.metrics import psi, kl_divergence, multi_feature_psi, multi_feature_kl, PSIKLMonitor
from drift_monitor.injection import severity_schedule, make_batches
from drift_monitor.diagnosis import classify_shift
from drift_monitor.evaluate import wilson_ci
from drift_monitor.benchmark import record_system_info


# ---------------------------------------------------------------------------
# PSI / KL divergence: basic correctness
# ---------------------------------------------------------------------------

def test_psi_zero_for_identical_distribution():
    """If nothing has changed, PSI must be ~0 -- this is the most basic
    correctness property a drift metric has to satisfy."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 5000)
    assert psi(data, data) == pytest.approx(0.0, abs=1e-9)


def test_kl_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 5000)
    assert kl_divergence(data, data) == pytest.approx(0.0, abs=1e-9)


def test_psi_increases_with_shift_magnitude():
    """PSI should be monotonically larger for a bigger shift -- this is
    the property the whole severity-calibration approach depends on."""
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    small_shift = rng.normal(0.5, 1, 5000)
    big_shift = rng.normal(2.0, 1, 5000)
    psi_small = psi(reference, small_shift)
    psi_big = psi(reference, big_shift)
    assert 0 < psi_small < psi_big


def test_psikl_monitor_matches_uncached_functions():
    """Regression test for the caching optimization (README 'Performance'
    section): PSIKLMonitor must produce numerically identical results to
    the original per-call functions, not just similar ones."""
    rng = np.random.default_rng(1)
    features = ["f1", "f2", "f3"]
    reference_df = pd.DataFrame({f: rng.normal(0, 1, 2000) for f in features})
    batch_df = pd.DataFrame({f: rng.normal(0.3, 1, 500) for f in features})

    old_psi = multi_feature_psi(reference_df, batch_df, features)
    old_kl = multi_feature_kl(reference_df, batch_df, features)

    monitor = PSIKLMonitor(features).fit(reference_df)
    new_scores = monitor.scores(batch_df)

    for f in features:
        assert old_psi[f] == pytest.approx(new_scores["psi"][f], abs=1e-10)
        assert old_kl[f] == pytest.approx(new_scores["kl"][f], abs=1e-10)


# ---------------------------------------------------------------------------
# Injection timing: severity_schedule shape
# ---------------------------------------------------------------------------

def test_abrupt_onset_is_a_step_function():
    sched = severity_schedule(n_batches=20, onset_batch=10, onset="abrupt")
    assert np.all(sched[:10] == 0.0)
    assert np.all(sched[10:] == 1.0)


def test_gradual_onset_ramps_up_smoothly():
    sched = severity_schedule(n_batches=20, onset_batch=10, onset="gradual", ramp_batches=5)
    assert np.all(sched[:10] == 0.0)
    # strictly increasing during the ramp window
    ramp = sched[10:15]
    assert np.all(np.diff(ramp) > 0)
    # fully ramped up and held after the ramp window ends
    assert np.all(sched[15:] == 1.0)


def test_make_batches_preserves_all_rows():
    df = pd.DataFrame({"x": range(103)})
    batches = make_batches(df, n_batches=10, shuffle=False)
    assert sum(len(b) for b in batches) == len(df)
    assert len(batches) == 10


# ---------------------------------------------------------------------------
# diagnosis.py: precedence logic
# ---------------------------------------------------------------------------

def _alert_row(psi="none", kl="none", pred="none", concept="none"):
    return {"psi_agg_alert": psi, "kl_agg_alert": kl,
            "pred_prop_psi_alert": pred, "concept_signal_alert": concept}


def test_diagnosis_covariate_takes_precedence():
    """This is the exact bug diagnosis.py was built to fix: PSI/KL firing
    must win even when concept_signal ALSO fires (the residual leakage
    case documented in the report)."""
    row = _alert_row(psi="significant", concept="moderate")
    assert classify_shift(row) == "covariate"


def test_diagnosis_label_requires_both_pred_and_concept():
    row = _alert_row(pred="moderate", concept="moderate")
    assert classify_shift(row) == "label"


def test_diagnosis_concept_alone_is_ambiguous_with_mild_label():
    row = _alert_row(concept="moderate")
    assert classify_shift(row) == "concept_or_mild_label"


def test_diagnosis_no_alerts_means_none():
    row = _alert_row()
    assert classify_shift(row) == "none"


# ---------------------------------------------------------------------------
# Wilson confidence intervals: sanity bounds
# ---------------------------------------------------------------------------

def test_wilson_ci_is_within_valid_probability_range():
    low, high = wilson_ci(successes=3, n=5)
    assert 0.0 <= low <= high <= 1.0


def test_wilson_ci_perfect_score_has_high_end_at_one():
    low, high = wilson_ci(successes=5, n=5)
    assert high == pytest.approx(1.0)
    assert low < 1.0  # even a perfect small sample isn't 100% certain


def test_wilson_ci_zero_score_has_low_end_at_zero():
    low, high = wilson_ci(successes=0, n=5)
    assert low == pytest.approx(0.0)
    assert high > 0.0


def test_wilson_ci_widens_with_fewer_samples():
    """The whole point of reporting these: n=5 must give a wider interval
    than n=30 for the same observed proportion."""
    low_small, high_small = wilson_ci(successes=3, n=5)
    low_big, high_big = wilson_ci(successes=18, n=30)  # same 0.6 proportion
    assert (high_small - low_small) > (high_big - low_big)


# ---------------------------------------------------------------------------
# Environment fingerprint: provenance, not just results
# ---------------------------------------------------------------------------

def test_system_info_records_all_expected_fields():
    """Every result in this project should be traceable to the exact
    environment that produced it -- this is what would have caught the
    cross-environment reproducibility bug (main README, bug #4) on the
    first attempt instead of the fourth."""
    info = record_system_info()
    expected_fields = {"timestamp_utc", "platform", "python_version",
                        "numpy_version", "pandas_version", "sklearn_version",
                        "scipy_version", "matplotlib_version"}
    assert expected_fields.issubset(info.keys())
    assert all(v for v in info.values())  # nothing blank
