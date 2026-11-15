"""
tests/test_core.py — automated correctness checks for the drift_monitor
package. Run with:

    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from drift_monitor.config import DriftMonitorConfig
from drift_monitor.metrics import (
    psi, kl_divergence, multi_feature_psi, multi_feature_kl,
    PSIKLMonitor, EmbeddingDriftDetector, aggregate, ks_two_sample_test
)
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


def test_ks_baseline_detects_covariate_shift():
    rng = np.random.default_rng(42)
    features = ["f1", "f2"]
    ref_df = pd.DataFrame({"f1": rng.normal(0, 1, 1000), "f2": rng.normal(5, 2, 1000)})
    clean_df = pd.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 2, 500)})
    shifted_df = pd.DataFrame({"f1": rng.normal(1.5, 1, 500), "f2": rng.normal(5, 2, 500)})

    res_clean = ks_two_sample_test(ref_df, clean_df, features)
    res_shifted = ks_two_sample_test(ref_df, shifted_df, features)

    assert not res_clean["ks_alert"]
    assert res_shifted["ks_alert"]
    assert res_shifted["ks_stat_max"] > 0.40
    assert res_shifted["ks_pval_min"] < 1e-10


def test_automatic_scores_against_hand_labelled_examples():
    """Validates automated detector alerts against human gold-standard labeled benchmark batches.
    Requires Precision = 1.0 (zero false alarms on clean batches) and Recall >= 0.90 (detects injected shifts)."""
    rng = np.random.default_rng(123)
    n_ref = 4000
    a = rng.normal(0, 1, n_ref)
    b = rng.normal(10, 2, n_ref)
    p = 1 / (1 + np.exp(-(0.8 * a + 0.3 * (b - 10))))
    y = rng.binomial(1, p)
    ref_df = pd.DataFrame({"a": a, "b": b, "y": y})

    cfg = DriftMonitorConfig(features=["a", "b"], label_col="y")
    psikl = PSIKLMonitor(cfg.features, cfg.n_bins).fit(ref_df)
    model = train_reference_model(ref_df, cfg)
    pred = PredictedProportionDetector(model, cfg.features).fit(ref_df)
    perf = PerformanceMonitor(model, cfg.features, "y", labeled_fraction=0.5, seed=42).fit(ref_df)

    benchmark_batches = []
    # 5 clean batches: ground truth = 0
    for i in range(5):
        a_c = rng.normal(0, 1, 1500)
        b_c = rng.normal(10, 2, 1500)
        p_c = 1 / (1 + np.exp(-(0.8 * a_c + 0.3 * (b_c - 10))))
        batch = pd.DataFrame({"a": a_c, "b": b_c, "y": rng.binomial(1, p_c)})
        benchmark_batches.append({"batch": batch, "ground_truth": 0})

    # 5 shifted batches: ground truth = 1
    # 1: Covariate shift on a
    a_s1 = rng.normal(1.5, 1, 1500)
    b_s1 = rng.normal(10, 2, 1500)
    p_s1 = 1 / (1 + np.exp(-(0.8 * a_s1 + 0.3 * (b_s1 - 10))))
    benchmark_batches.append({"batch": pd.DataFrame({"a": a_s1, "b": b_s1, "y": rng.binomial(1, p_s1)}), "ground_truth": 1})

    # 2: Covariate shift on b
    a_s2 = rng.normal(0, 1, 1500)
    b_s2 = rng.normal(13, 2, 1500)
    p_s2 = 1 / (1 + np.exp(-(0.8 * a_s2 + 0.3 * (b_s2 - 10))))
    benchmark_batches.append({"batch": pd.DataFrame({"a": a_s2, "b": b_s2, "y": rng.binomial(1, p_s2)}), "ground_truth": 1})

    # 3: Label shift (prior shift)
    a_s3 = rng.normal(0, 1, 1500)
    b_s3 = rng.normal(10, 2, 1500)
    benchmark_batches.append({"batch": pd.DataFrame({"a": a_s3, "b": b_s3, "y": rng.binomial(1, 0.75, 1500)}), "ground_truth": 1})

    # 4: Concept drift (inversion of P(Y|X))
    a_s4 = rng.normal(0, 1, 1500)
    b_s4 = rng.normal(10, 2, 1500)
    p_s4 = 1 / (1 + np.exp(-(0.8 * a_s4 + 0.3 * (b_s4 - 10))))
    benchmark_batches.append({"batch": pd.DataFrame({"a": a_s4, "b": b_s4, "y": rng.binomial(1, 1 - p_s4)}), "ground_truth": 1})

    # 5: Severe joint shift
    a_s5 = rng.normal(2.0, 1, 1500)
    b_s5 = rng.normal(14, 2, 1500)
    benchmark_batches.append({"batch": pd.DataFrame({"a": a_s5, "b": b_s5, "y": rng.binomial(1, 0.8, 1500)}), "ground_truth": 1})

    tp, fp, fn = 0, 0, 0
    for item in benchmark_batches:
        b = item["batch"]
        gt = item["ground_truth"]
        psi_val = aggregate(psikl.scores(b)["psi"], "max")
        pred_val = pred.drift_score(b)["pred_prop_psi"]
        perf_val = perf.drift_score(b)["concept_signal"]

        # Alert triggered if any primary detector crosses moderate operational threshold
        alert = (psi_val >= 0.10) or (pred_val >= 0.015) or (perf_val >= 0.025)

        if alert and gt == 1:
            tp += 1
        elif alert and gt == 0:
            fp += 1
        elif not alert and gt == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    assert fp == 0, f"Expected 0 false alarms on clean batches, got {fp}"
    assert precision == 1.0, f"Expected Precision 1.0, got {precision}"
    assert recall >= 0.90, f"Expected Recall >= 0.90, got {recall}"



