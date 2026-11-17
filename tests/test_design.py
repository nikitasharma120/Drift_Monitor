"""
tests/test_design.py - tests for the experimental-design guarantees:
disjoint reference/stream, pure injectors, calibrated thresholds, trade-off curves,
the named baseline, the PSI/KL identity, diagnosis, and the hand-labelled check.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_monitor.config import DriftMonitorConfig
from drift_monitor.injection import (make_reference_and_stream, batch_size_for, make_scenario,
                                      inject_label_shift, inject_concept_drift, make_batches)
from drift_monitor.metrics import psi, kl_divergence
from drift_monitor.baselines import KSBaseline
from drift_monitor.monitor import apply_alerts, confirmed_flags, METRICS, ALL_METRICS
from drift_monitor.evaluate import calibrate_thresholds
from drift_monitor.tradeoff import operating_curve
from drift_monitor.diagnosis import classify_shift


def _toy_df(n=24_000, seed=0, with_id=True):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    hours = rng.integers(30, 50, n).astype(float)          # integer-valued, like Adult's hours-per-week
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * x1 - x2 - 1.5)))).astype(int)
    df = pd.DataFrame({"f1": x1, "f2": x2, "hours": hours, "label": y})
    if with_id:
        df["row_id"] = np.arange(n)
    return df


def _cfg(**kw):
    base = dict(features=["f1", "f2", "hours"], label_col="label", target_feature="hours",
                n_batches=20, n_reference_batches=10, onset_batch=10, embedding_components=3)
    base.update(kw)
    return DriftMonitorConfig(**base)


# --- reference and stream share no rows; every run draws its own reference -------------

def test_reference_and_stream_share_no_rows():
    df, cfg = _toy_df(), _cfg()
    ref, batches = make_reference_and_stream(df, cfg, seed=3)
    stream_ids = set(pd.concat(batches)["row_id"])
    assert not (set(ref["row_id"]) & stream_ids)
    assert len(ref) == cfg.n_reference_batches * batch_size_for(len(df), cfg)
    assert all(len(b) == batch_size_for(len(df), cfg) for b in batches)


def test_every_seed_draws_a_new_reference():
    df, cfg = _toy_df(), _cfg()
    r1, _ = make_reference_and_stream(df, cfg, seed=1)
    r2, _ = make_reference_and_stream(df, cfg, seed=2)
    assert set(r1["row_id"]) != set(r2["row_id"])


# --- label-shift injector leaves pre-onset batches untouched ------------------------

def test_label_shift_leaves_pre_onset_batches_untouched_and_raises_rate_after():
    df, cfg = _toy_df(), _cfg()
    _, clean = make_reference_and_stream(df, cfg, seed=4)
    shifted = inject_label_shift(clean, "label", "severe", cfg.onset_batch, cfg, onset="abrupt", seed=4)
    for i in range(cfg.onset_batch):
        pd.testing.assert_frame_equal(clean[i], shifted[i])           # bit-for-bit, no resampling noise
    base = pd.concat(clean[:cfg.onset_batch])["label"].mean()
    assert shifted[-1]["label"].mean() == pytest.approx(base + cfg.severity_label["severe"], abs=0.01)


# --- concept drift is pure; severities differ even on integer-valued targets ----------

@pytest.mark.parametrize("severity", ["mild", "moderate", "severe"])
def test_concept_drift_is_pure(severity):
    """P(X) untouched (every feature column identical) and P(Y) untouched (same positives per batch)."""
    df, cfg = _toy_df(), _cfg()
    _, clean = make_reference_and_stream(df, cfg, seed=5)
    drifted = inject_concept_drift(clean, "hours", "label", severity, cfg.onset_batch, cfg, onset="abrupt", seed=5)
    for c, d in zip(clean, drifted):
        for f in cfg.features:
            np.testing.assert_array_equal(c[f].to_numpy(), d[f].to_numpy())
        assert c["label"].sum() == d["label"].sum()
    changed = sum((c["label"].to_numpy() != d["label"].to_numpy()).sum() for c, d in zip(clean[cfg.onset_batch:], drifted[cfg.onset_batch:]))
    assert changed > 0
    assert all((c["label"] == d["label"]).all() for c, d in zip(clean[:cfg.onset_batch], drifted[:cfg.onset_batch]))


def test_concept_severities_differ_even_on_integer_valued_target():
    df, cfg = _toy_df(), _cfg()
    _, clean = make_reference_and_stream(df, cfg, seed=6)
    n_changed = {}
    for sev in ["mild", "moderate", "severe"]:
        d = inject_concept_drift(clean, "hours", "label", sev, cfg.onset_batch, cfg, onset="abrupt", seed=6)
        n_changed[sev] = sum((c["label"].to_numpy() != x["label"].to_numpy()).sum() for c, x in zip(clean, d))
    assert n_changed["mild"] < n_changed["moderate"] < n_changed["severe"]


def test_concept_drift_actually_moves_the_feature_label_relationship():
    df, cfg = _toy_df(), _cfg()
    _, clean = make_reference_and_stream(df, cfg, seed=8)
    d = inject_concept_drift(clean, "hours", "label", "severe", cfg.onset_batch, cfg, onset="abrupt", seed=8)
    med = pd.concat(clean[:cfg.onset_batch])["hours"].median()
    hi_before = pd.concat(clean[cfg.onset_batch:]).query("hours >= @med")["label"].mean()
    hi_after = pd.concat(d[cfg.onset_batch:]).query("hours >= @med")["label"].mean()
    assert hi_after < hi_before * 0.2          # severe: (almost) all high-region positives relabelled


# --- thresholds are calibrated per dataset ---------------------------------

def test_calibration_raises_thresholds_when_batches_are_small_and_never_goes_below_floor():
    small = _toy_df(n=6_000)       # batch size 200: noisy detectors
    cfg = _cfg(calibration_reps=15)
    thr, info = calibrate_thresholds(small, cfg)
    for m in METRICS:
        assert thr[m]["moderate"] >= cfg.threshold_floors[m]["moderate"] - 1e-12
        assert thr[m]["significant"] >= cfg.threshold_floors[m]["significant"] - 1e-12
    assert info["raised_above_floor"].any()                 # at 200 rows/batch at least one floor sits under the noise
    assert thr["ks_baseline"] == cfg.alert_thresholds["ks_baseline"]    # the baseline is not tuned


def test_calibrated_thresholds_give_near_zero_false_alarms_on_fresh_clean_runs():
    from drift_monitor.monitor import fit_detectors, score_batches
    df = _toy_df(n=12_000)
    cfg = _cfg(calibration_reps=25)
    thr, _ = calibrate_thresholds(df, cfg)
    flagged = total = 0
    for seed in range(50, 62):                                # seeds never used in calibration
        ref, batches = make_scenario("none", "mild", "abrupt", seed, df, cfg)
        sc = score_batches(batches, fit_detectors(ref, cfg, seed))
        res = apply_alerts(sc, cfg, thr)
        flagged += int(res["any_alert"].sum()); total += len(res)
    assert flagged / total < 0.03


# --- trade-off curve ------------------------------------------------------------

def test_operating_curve_is_monotone_and_spans_zero_to_high():
    rng = np.random.default_rng(0)
    rows = []
    for rep in range(30):
        for b in range(20):
            rows.append({"shift_type": "covariate", "severity": "severe", "onset": "abrupt", "rep": rep, "batch": b,
                         "psi_agg": rng.normal(0, 1) + (3 if b >= 10 else 0)})
    raw = pd.DataFrame(rows)
    c = operating_curve(raw, _cfg(), "psi_agg", "covariate").sort_values("threshold", ascending=False)
    assert c["false_alarm_rate"].is_monotonic_increasing
    assert c["sensitivity"].is_monotonic_increasing
    assert c.iloc[0]["false_alarm_rate"] == 0 and c.iloc[0]["sensitivity"] == 0     # threshold above every score
    assert c["sensitivity"].max() > 0.95


# --- named baseline ------------------------------------------------------------------

def test_ks_baseline_is_quiet_on_identical_distributions_and_loud_on_a_shift():
    rng = np.random.default_rng(1)
    feats = ["a", "b"]
    ref = pd.DataFrame({f: rng.normal(size=5000) for f in feats})
    same = pd.DataFrame({f: rng.normal(size=800) for f in feats})
    moved = pd.DataFrame({"a": rng.normal(0.5, 1, 800), "b": rng.normal(size=800)})
    ks = KSBaseline(feats).fit(ref)
    assert ks.drift_score(same)["ks_baseline"] < 2.0          # below alpha = 0.01
    assert ks.drift_score(moved)["ks_baseline"] > 10


def test_ks_statistic_matches_scipy():
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(2)
    r, c = rng.normal(size=3000), rng.normal(0.2, 1, 700)
    ks = KSBaseline(["a"]).fit(pd.DataFrame({"a": r}))
    d, _, _ = ks._ks_stat("a", c)
    assert d == pytest.approx(ks_2samp(r, c).statistic, abs=1e-12)


# --- PSI versus KL -------------------------------------------------------

def test_psi_is_the_symmetrised_kl_so_one_directional_kl_never_exceeds_it():
    """PSI = KL(cur||ref) + KL(ref||cur). The repo's kl_divergence is only KL(cur||ref),
    so with the same threshold it can never fire when PSI does not (it is not 'a log-ratio
    penalty instead of a linear one' -- both use the log ratio)."""
    rng = np.random.default_rng(3)
    ref = rng.normal(size=6000)
    for shift in [0.0, 0.1, 0.5, 2.0]:
        cur = rng.normal(shift, 1, 1500)
        assert psi(ref, cur) >= kl_divergence(ref, cur) - 1e-12


def test_psi_equals_forward_plus_reverse_kl_on_shared_bins():
    ref = np.arange(100)                                           # 2 bins at the median: 50/50
    cur = np.r_[np.arange(25), np.arange(50, 125)]                 # 25 below, 75 above -> 25/75
    # hand computation: (0.25-0.5)ln(0.5) + (0.75-0.5)ln(1.5) = 0.274653...
    assert psi(ref, cur, n_bins=2) == pytest.approx(0.25 * np.log(2) + 0.25 * np.log(1.5), abs=1e-9)
    assert psi(ref, cur, n_bins=2) == pytest.approx(0.274653, abs=1e-6)
    # KL(cur||ref) = 0.25 ln(0.5) + 0.75 ln(1.5) = 0.130812...
    assert kl_divergence(ref, cur, n_bins=2) == pytest.approx(0.130812, abs=1e-6)
    # KL(ref||cur) = 0.5 ln(2) + 0.5 ln(2/3) = 0.143841...  and the two add up to PSI
    reverse = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
    assert psi(ref, cur, n_bins=2) == pytest.approx(kl_divergence(ref, cur, n_bins=2) + reverse, abs=1e-9)


# --- diagnosis: an alert must never be reported as "none" -----------------------------------------

def test_pred_prop_alert_alone_is_not_reported_as_none():
    row = {"psi_agg_alert": "none", "kl_agg_alert": "none", "pred_prop_psi_alert": "moderate", "concept_signal_alert": "none"}
    assert classify_shift(row) == "label_or_subtle_covariate"


def test_consecutive_rule():
    assert confirmed_flags([0, 5, 0, 5, 5, 5], moderate=1, k=2).tolist() == [False, False, False, False, True, True]


# --- the hand-labelled check (validation/) ---------------------------------------------------------

def test_hand_labelled_clean_stream_raises_no_alert_and_shifted_streams_are_caught():
    from validation.hand_labelled_check import run_hand_labelled_check
    d = run_hand_labelled_check()
    s = d[d["scored"]]
    assert not s[s["stream"] == "clean"]["any_alert"].any()
    for stream in ["covariate", "concept", "label"]:
        post = s[(s["stream"] == stream) & (s["batch"] >= 12)]
        assert post["any_alert"].mean() >= 0.6, stream


# --- binning, embedding and interval statistics -----------------------------------------------------

def test_psi_sees_a_shift_on_a_binary_feature_and_ignores_duplicate_quantiles():
    from drift_monitor.metrics import quantile_edges
    rng = np.random.default_rng(0)
    ref = (rng.random(20000) < 0.3).astype(float)
    cur = (rng.random(1000) < 0.6).astype(float)
    assert psi(ref, cur) > 0.2                         # a binary feature must not collapse to one bin
    zero_inflated = np.where(rng.random(20000) < 0.8, 0.0, rng.exponential(1000, 20000))
    edges = quantile_edges(zero_inflated)
    assert np.all(np.diff(edges[1:-1]) > 0)            # no zero-width bins
    assert edges[0] == -np.inf and edges[-1] == np.inf


def test_embedding_detector_is_robust_to_one_extreme_outlier_column():
    from drift_monitor.metrics import EmbeddingDriftDetector
    rng = np.random.default_rng(1)
    n = 5000
    ref = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "ratio": rng.normal(size=n)})
    ref.loc[:4, "ratio"] = 1e6                          # a handful of absurd values, like DebtRatio
    det = EmbeddingDriftDetector(["a", "b", "ratio"], 2).fit(ref)
    same = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500), "ratio": rng.normal(size=500)})
    moved = same.assign(a=same["a"] + 3.0)
    assert det.drift_score(moved)["centroid_dist"] > 3 * det.drift_score(same)["centroid_dist"]


def test_pooled_wilson_interval_is_computed_from_pooled_counts_not_averaged_bounds():
    from drift_monitor.evaluate import pooled_sensitivity, wilson_ci
    summary = pd.DataFrame({"shift_type": ["label"] * 3, "metric": ["m"] * 3, "n_reps": [20, 20, 20],
                            "detections": [0, 10, 20], "mean_false_alarm_rate": [0.0] * 3, "runs_with_false_alarm": [0.0] * 3})
    row = pooled_sensitivity(summary, ["shift_type", "metric"]).iloc[0]
    lo, hi = wilson_ci(30, 60)
    assert (row["sensitivity"], row["n_runs"]) == (0.5, 60)
    assert row["sensitivity_ci_low"] == pytest.approx(lo) and row["sensitivity_ci_high"] == pytest.approx(hi)


def test_debounce_table_more_consecutive_batches_never_adds_false_alarms():
    from drift_monitor.tradeoff import debounce_table
    rng = np.random.default_rng(2)
    rows = []
    for rep in range(40):
        for b in range(20):
            rows.append({"shift_type": "covariate", "severity": "severe", "onset": "abrupt", "rep": rep, "batch": b,
                         **{m: rng.normal() + (3 if b >= 10 else 0) for m in ALL_METRICS}})
    thr = {m: {"moderate": 1.0, "significant": 2.0} for m in ALL_METRICS}
    deb = debounce_table(pd.DataFrame(rows), _cfg(shift_types=["covariate"]), thr)
    d = deb[deb["metric"] == "psi_agg"].sort_values("k")
    assert d["false_alarm_rate"].is_monotonic_decreasing
