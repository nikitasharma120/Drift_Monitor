"""
tests/test_core.py — automated correctness checks for the drift_monitor
package. Run with:

    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from drift_monitor.config import DriftMonitorConfig
from drift_monitor.metrics import psi, kl_divergence, multi_feature_psi, multi_feature_kl, PSIKLMonitor, EmbeddingDriftDetector
from drift_monitor.reference_model import Winsorizer, train_reference_model, PredictedProportionDetector
from drift_monitor.performance_monitor import PerformanceMonitor
from drift_monitor.injection import severity_schedule, make_batches, inject_label_shift, inject_concept_drift
from drift_monitor.diagnosis import classify_shift
from drift_monitor.evaluate import wilson_ci
from drift_monitor.benchmark import record_system_info


# ---------------------------------------------------------------------------
# PSI / KL divergence: basic correctness
# ---------------------------------------------------------------------------

def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 5000)
    assert psi(data, data) == pytest.approx(0.0, abs=1e-9)


def test_kl_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 5000)
    assert kl_divergence(data, data) == pytest.approx(0.0, abs=1e-9)


def test_psi_increases_with_shift_magnitude():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    small_shift = rng.normal(0.5, 1, 5000)
    big_shift = rng.normal(2.0, 1, 5000)
    psi_small = psi(reference, small_shift)
    psi_big = psi(reference, big_shift)
    assert 0 < psi_small < psi_big


def test_psikl_monitor_matches_uncached_functions():
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
    ramp = sched[10:15]
    assert np.all(np.diff(ramp) > 0)
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
    assert low < 1.0


def test_wilson_ci_zero_score_has_low_end_at_zero():
    low, high = wilson_ci(successes=0, n=5)
    assert low == pytest.approx(0.0)
    assert high > 0.0


def test_wilson_ci_widens_with_fewer_samples():
    low_small, high_small = wilson_ci(successes=3, n=5)
    low_big, high_big = wilson_ci(successes=18, n=30)
    assert (high_small - low_small) > (high_big - low_big)


# ---------------------------------------------------------------------------
# Environment fingerprint: provenance
# ---------------------------------------------------------------------------

def test_system_info_records_all_expected_fields():
    info = record_system_info()
    expected_fields = {"timestamp_utc", "platform", "python_version",
                        "numpy_version", "pandas_version", "sklearn_version",
                        "scipy_version", "matplotlib_version"}
    assert expected_fields.issubset(info.keys())
    assert all(v for v in info.values())


# ---------------------------------------------------------------------------
# Additional Rigour Tests: Winsorizer, Detectors, Injectors & Zero-IQR
# ---------------------------------------------------------------------------

def test_winsorizer_clips_extremes_to_reference_bounds():
    ref = pd.DataFrame({"x": np.linspace(0, 100, 101)})
    winsor = Winsorizer(0.05, 0.95).fit(ref)
    
    test = pd.DataFrame({"x": [-50.0, 50.0, 1000.0]})
    clipped = winsor.transform(test)
    assert clipped[0, 0] == pytest.approx(5.0)
    assert clipped[1, 0] == pytest.approx(50.0)
    assert clipped[2, 0] == pytest.approx(95.0)


def test_psi_handles_zero_inflated_features_without_crashing():
    ref = np.zeros(1000)
    ref[900:] = np.linspace(1, 100, 100)
    cur = np.zeros(1000)
    cur[850:] = np.linspace(1, 100, 150)
    score = psi(ref, cur, n_bins=10)
    assert np.isfinite(score)
    assert score >= 0.0


def test_label_shift_injector_leaves_pre_onset_batches_untouched():
    df = pd.DataFrame({"feat": range(200), "label": [0]*180 + [1]*20})
    cfg = DriftMonitorConfig(n_batches=10, onset_batch=5)
    batches = [df.iloc[i*20:(i+1)*20].copy() for i in range(10)]
    shifted = inject_label_shift(batches, "label", "severe", 5, cfg)
    
    for i in range(5):
        pd.testing.assert_frame_equal(batches[i], shifted[i])


def test_performance_monitor_detects_brier_degradation():
    cfg = DriftMonitorConfig(features=["x"], label_col="y")
    ref_df = pd.DataFrame({"x": np.linspace(-2, 2, 200), "y": (np.linspace(-2, 2, 200) > 0).astype(int)})
    model = train_reference_model(ref_df, cfg)
    perf = PerformanceMonitor(model, ["x"], "y", labeled_fraction=1.0, seed=42).fit(ref_df)
    
    drift_df = ref_df.copy()
    drift_df["y"] = 1 - drift_df["y"]
    score = perf.drift_score(drift_df)
    assert score["brier_delta"] > 0.2
    assert score["concept_signal"] > 0.2


def test_concept_drift_injector_preserves_pre_onset_and_supports_pure_swap():
    df = pd.DataFrame({"x": np.linspace(-5, 5, 200), "y": [0]*100 + [1]*100})
    cfg = DriftMonitorConfig(n_batches=10, onset_batch=5)
    batches = [df.iloc[i*20:(i+1)*20].copy() for i in range(10)]
    
    # 1. Unconstrained: pre-onset must be untouched
    shifted = inject_concept_drift(batches, "x", "y", "severe", 5, cfg, pure=False)
    for i in range(5):
        pd.testing.assert_frame_equal(batches[i], shifted[i])
    
    # 2. Pure swap: marginal positive label rate across all batches must be strictly invariant
    shifted_pure = inject_concept_drift(batches, "x", "y", "severe", 5, cfg, pure=True)
    orig_pos = sum(b["y"].sum() for b in batches)
    pure_pos = sum(b["y"].sum() for b in shifted_pure)
    assert orig_pos == pure_pos

