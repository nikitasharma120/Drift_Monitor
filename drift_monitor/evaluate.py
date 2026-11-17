"""evaluate.py - threshold calibration, the shift-type x severity x onset x rep grid, summaries.

Pipeline for one dataset:
  1. calibrate_thresholds()  -- measure each detector's noise floor on CLEAN runs (seeds disjoint from evaluation)
  2. run_full_grid()         -- score every run ONCE (raw scores, threshold-free), then apply the calibrated thresholds
  3. summarize_grid()        -- sensitivity (Wilson CI), lag, batch-level and run-level false-alarm rates
The raw scores are returned too, so tradeoff.py can re-apply many thresholds to the same runs (ROC curves).
"""

import time
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .injection import make_scenario
from .monitor import (METRICS, ALL_METRICS, fit_detectors, score_batches, confirmed_flags)


# ---------------------------------------------------------------------------
# Threshold calibration (per dataset)
# ---------------------------------------------------------------------------

def calibrate_thresholds(df: pd.DataFrame, config, n_reps=None, verbose=False):
    """Measures each of the five detectors' noise floor on CLEAN (no-shift)
    runs and sets thresholds just above it.

    Why per dataset: PSI-type and Brier-type scores have a noise floor that
    depends on batch size (rows per batch = N // 40, so Give Me Some Credit
    ~3,000 rows vs Taiwan 750). A threshold that sits 6x above the noise at
    3,000 rows can sit BELOW it at 750 rows and then fire constantly.

    Rule:  moderate    = max(floor, calibration_margin * q99(clean batch scores))
           significant = max(floor_significant, 2 * moderate)
    where `floor` is config.threshold_floors (the conventional default values, kept
    as a minimum so calibration can only make a detector LESS trigger-happy
    than the floor, never more). `ks_baseline` is left at its fixed alpha:
    a baseline is not tuned.

    Calibration runs use seeds >= config.calibration_seed_base, disjoint from
    every evaluation seed, so the false-alarm rates measured in the grid are
    out-of-sample with respect to the calibration runs."""
    n_reps = config.calibration_reps if n_reps is None else n_reps
    frames = []
    for r in range(n_reps):
        seed = config.calibration_seed_base + r
        ref, batches = make_scenario("none", "mild", "abrupt", seed, df, config)
        suite = fit_detectors(ref, config, seed)
        frames.append(score_batches(batches, suite)[["batch"] + METRICS])
        if verbose and (r + 1) % 10 == 0:
            print(f"  calibration run {r + 1}/{n_reps}")
    clean = pd.concat(frames, ignore_index=True)

    thresholds = {m: dict(config.alert_thresholds[m]) for m in ALL_METRICS}
    rows = []
    for m in METRICS:
        floor = config.threshold_floors[m]
        q = float(clean[m].quantile(config.calibration_quantile))
        moderate = max(floor["moderate"], config.calibration_margin * q)
        significant = max(floor["significant"], 2 * moderate)
        thresholds[m] = {"moderate": float(moderate), "significant": float(significant)}
        rows.append({"metric": m, "clean_median": float(clean[m].median()), "clean_q99": q,
                     "floor_moderate": floor["moderate"], "calibrated_moderate": float(moderate),
                     "calibrated_significant": float(significant),
                     "raised_above_floor": bool(moderate > floor["moderate"] + 1e-12),
                     "n_clean_batches": int(len(clean))})
    return thresholds, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def _deterministic_seed(rep: int, shift_type: str, severity: str, onset: str) -> int:
    """CRC32, not Python's hash() - hash() is process-randomized by default
    and would silently break reproducibility across runs/machines."""
    tag = f"{shift_type}_{severity}_{onset}".encode()
    return 1000 * rep + zlib.crc32(tag) % 1000


@dataclass
class GridResult:
    records: pd.DataFrame      # one row per (run, metric): detected, lag, false_alarm (batch-level), false_alarm_run
    raw_scores: pd.DataFrame   # one row per (run, batch): all raw detector scores, threshold-free
    thresholds: dict           # the thresholds that produced `records`


def score_grid(df: pd.DataFrame, config, verbose=True) -> pd.DataFrame:
    """Scores every run once. Each (shift_type, severity, onset, rep) gets its OWN
    disjoint reference/stream split and its own freshly fitted detectors."""
    frames, t0 = [], time.time()
    total = len(config.shift_types) * len(config.severities) * len(config.onsets) * config.n_reps
    done = 0
    for shift_type in config.shift_types:
        for severity in config.severities:
            for onset in config.onsets:
                for rep in range(config.n_reps):
                    seed = _deterministic_seed(rep, shift_type, severity, onset)
                    ref, batches = make_scenario(shift_type, severity, onset, seed, df, config)
                    suite = fit_detectors(ref, config, seed)
                    sc = score_batches(batches, suite)[["batch"] + ALL_METRICS]
                    sc.insert(0, "rep", rep); sc.insert(0, "onset", onset)
                    sc.insert(0, "severity", severity); sc.insert(0, "shift_type", shift_type)
                    frames.append(sc)
                    done += 1
        if verbose:
            print(f"  scored {done}/{total} runs ({time.time() - t0:.0f}s)")
    return pd.concat(frames, ignore_index=True)


def records_from_scores(raw: pd.DataFrame, config, thresholds=None) -> pd.DataFrame:
    thresholds = config.alert_thresholds if thresholds is None else thresholds
    k = config.consecutive_batches_required
    recs = []
    for (st, sev, on, rep), g in raw.groupby(["shift_type", "severity", "onset", "rep"], sort=False):
        g = g.sort_values("batch")
        batch = g["batch"].to_numpy()
        pre, post = batch < config.onset_batch, batch >= config.onset_batch
        for m in ALL_METRICS:
            flag = confirmed_flags(g[m].to_numpy(), thresholds[m]["moderate"], k)
            hit = np.flatnonzero(flag & post)
            detected = len(hit) > 0
            recs.append({"shift_type": st, "severity": sev, "onset": on, "rep": rep, "metric": m,
                         "detected": detected,
                         "lag": int(batch[hit[0]] - config.onset_batch) if detected else None,
                         "false_alarm": float(flag[pre].mean()) if pre.any() else 0.0,
                         "false_alarm_run": bool(flag[pre].any())})
    return pd.DataFrame(recs)


def run_full_grid(df: pd.DataFrame, config, thresholds=None, verbose=True) -> GridResult:
    """Calibrates thresholds on this dataset (unless `thresholds` is given), scores
    the whole grid once, and applies the thresholds."""
    if thresholds is None:
        if verbose:
            print(f"Calibrating thresholds on {config.calibration_reps} clean runs ...")
        thresholds, _ = calibrate_thresholds(df, config, verbose=verbose)
    raw = score_grid(df, config, verbose=verbose)
    records = records_from_scores(raw, config, thresholds)
    return GridResult(records, raw, thresholds)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion. Far better than a
    normal-approximation CI for small n and for proportions near 0 or 1
    (many of ours are exactly 0.0 or 1.0), where the normal approximation
    breaks down badly."""
    if n == 0:
        return (np.nan, np.nan)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize_grid(records: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        detected = g["detected"].astype(bool)
        lags = g.loc[detected, "lag"]
        n = len(detected)
        ci_low, ci_high = wilson_ci(int(detected.sum()), n)
        fa_run = g["false_alarm_run"].astype(bool)
        fa_lo, fa_hi = wilson_ci(int(fa_run.sum()), n)
        return pd.Series({
            "detections": int(detected.sum()),
            "sensitivity": detected.mean(),
            "sensitivity_ci_low": ci_low,
            "sensitivity_ci_high": ci_high,
            "n_reps": n,
            "mean_lag_when_detected": lags.mean() if len(lags) else np.nan,
            "mean_false_alarm_rate": g["false_alarm"].mean(),
            "runs_with_false_alarm": fa_run.mean(),
            "runs_with_false_alarm_ci_low": fa_lo,
            "runs_with_false_alarm_ci_high": fa_hi,
        })
    return records.groupby(["shift_type", "severity", "onset", "metric"]).apply(agg, include_groups=False).reset_index()


def find_failure_modes(summary: pd.DataFrame, sensitivity_threshold: float = 0.5,
                        lag_threshold: float = 5.0) -> pd.DataFrame:
    failures = summary[(summary["sensitivity"] < sensitivity_threshold) |
                        (summary["mean_lag_when_detected"] > lag_threshold)].copy()
    failures["failure_type"] = np.where(failures["sensitivity"] < sensitivity_threshold,
                                         "low sensitivity", "slow detection")
    return failures.sort_values(["shift_type", "severity", "onset", "metric"])


def pooled_sensitivity(summary: pd.DataFrame, by) -> pd.DataFrame:
    """Sensitivity aggregated over grid cells, with a Wilson interval computed from the
    POOLED counts (total detections out of total runs). Averaging the per-cell interval
    bounds would not be a valid confidence interval; pooling the counts is."""
    rows = []
    for key, g in summary.groupby(by):
        k, n = int(g["detections"].sum()), int(g["n_reps"].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({**dict(zip([by] if isinstance(by, str) else by, key if isinstance(key, tuple) else (key,))),
                     "detections": k, "n_runs": n, "sensitivity": k / n,
                     "sensitivity_ci_low": lo, "sensitivity_ci_high": hi,
                     "mean_false_alarm_rate": g["mean_false_alarm_rate"].mean(),
                     "runs_with_false_alarm": g["runs_with_false_alarm"].mean()})
    return pd.DataFrame(rows)


def _per_metric_table(summary: pd.DataFrame) -> pd.DataFrame:
    return pooled_sensitivity(summary, ["shift_type", "metric"])


def best_detector_per_shift_type(summary: pd.DataFrame) -> pd.DataFrame:
    """For each shift type, the one of the FIVE detectors with the highest mean
    sensitivity (averaged over severity + onset), with the false-alarm rates of
    that same detector next to it. The named baseline is excluded here and
    compared in baseline_comparison()."""
    ranked = _per_metric_table(summary[summary["metric"].isin(METRICS)])
    ranked["_o"] = ranked["metric"].map({m: i for i, m in enumerate(METRICS)})
    ranked = ranked.sort_values(["shift_type", "_o"]).drop(columns="_o").reset_index(drop=True)   # ties -> METRICS order
    best = ranked.loc[ranked.groupby("shift_type")["sensitivity"].idxmax()]
    return best.rename(columns={"metric": "recommended_detector", "sensitivity": "mean_sensitivity"}).reset_index(drop=True)


def baseline_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Best of the five detectors vs. the named KS+Bonferroni baseline, per shift type."""
    t = _per_metric_table(summary)
    rows = []
    for st, g in t.groupby("shift_type"):
        five = g[g["metric"].isin(METRICS)].set_index("metric").loc[METRICS].sort_values(
            "sensitivity", ascending=False, kind="stable").reset_index().iloc[0]
        base = g[g["metric"] == "ks_baseline"].iloc[0]
        rows.append({"shift_type": st,
                     "best_of_five": five["metric"], "best_of_five_sensitivity": five["sensitivity"],
                     "best_of_five_false_alarm": five["mean_false_alarm_rate"],
                     "ks_baseline_sensitivity": base["sensitivity"],
                     "ks_baseline_false_alarm": base["mean_false_alarm_rate"],
                     "sensitivity_gain_vs_baseline": five["sensitivity"] - base["sensitivity"]})
    return pd.DataFrame(rows)
