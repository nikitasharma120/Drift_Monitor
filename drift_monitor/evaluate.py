"""evaluate.py — the full shift-type x severity x onset x rep evaluation grid."""

import time
import zlib
import numpy as np
import pandas as pd

from .injection import make_batches, inject
from .metrics import EmbeddingDriftDetector, PSIKLMonitor, aggregate
from .reference_model import train_reference_model, PredictedProportionDetector
from .performance_monitor import PerformanceMonitor
from .monitor import METRICS


def _per_metric_alert_series(batches, reference_df, embedding_detector, config, pred_detector=None, perf_detector=None) -> pd.DataFrame:
    rows = []
    consecutive = {m: 0 for m in METRICS}
    psikl = PSIKLMonitor(config.features, config.n_bins).fit(reference_df)  # fit ONCE, not per batch
    for i, batch in enumerate(batches):
        scores = psikl.scores(batch)
        psi_feat, kl_feat = scores["psi"], scores["kl"]
        emb = embedding_detector.drift_score(batch)
        pred = pred_detector.drift_score(batch) if pred_detector is not None else {"pred_prop_psi": 0.0}
        perf = perf_detector.drift_score(batch) if perf_detector is not None else {"concept_signal": 0.0}
        vals = {"psi_agg": aggregate(psi_feat, "max"), "kl_agg": aggregate(kl_feat, "max"),
                "centroid_dist": emb["centroid_dist"], "pred_prop_psi": pred["pred_prop_psi"],
                "concept_signal": perf["concept_signal"]}
        row = {"batch": i, **vals}
        for m in METRICS:
            th = config.alert_thresholds[m]
            crossed = vals[m] >= th["moderate"]
            consecutive[m] = consecutive[m] + 1 if crossed else 0
            row[f"{m}_alert"] = consecutive[m] >= config.consecutive_batches_required
        rows.append(row)
    return pd.DataFrame(rows)


def _deterministic_seed(rep: int, shift_type: str, severity: str, onset: str) -> int:
    """CRC32, not Python's hash() — hash() is process-randomized by default
    and would silently break reproducibility across runs/machines."""
    tag = f"{shift_type}_{severity}_{onset}".encode()
    return 1000 * rep + zlib.crc32(tag) % 1000


def run_full_grid(df: pd.DataFrame, config, include_pred_detector: bool = True, include_perf_detector: bool = True) -> pd.DataFrame:
    """Runs every (shift_type, severity, onset) combination in config,
    config.n_reps times each, and returns one row per (config, rep, metric).
    Set include_pred_detector/include_perf_detector=False to reproduce
    earlier (smaller) grids without those detectors."""
    ref_batches = make_batches(df, config.n_batches, seed=0)
    reference_df = pd.concat(ref_batches[:config.onset_batch]).reset_index(drop=True)
    emb_detector = EmbeddingDriftDetector(config.features, config.embedding_components).fit(reference_df)

    pred_detector = None
    perf_detector = None
    if include_pred_detector or include_perf_detector:
        model = train_reference_model(reference_df, config)
        if include_pred_detector:
            pred_detector = PredictedProportionDetector(model, config.features).fit(reference_df)
        if include_perf_detector:
            perf_detector = PerformanceMonitor(model, config.features, config.label_col,
                                                config.labeled_trickle_fraction).fit(reference_df)

    records = []
    t0 = time.time()
    n_base_configs = len(config.shift_types) * len(config.severities) * len(config.onsets)
    total = n_base_configs * config.n_reps
    for shift_type in config.shift_types:
        for severity in config.severities:
            for onset in config.onsets:
                for rep in range(config.n_reps):
                    seed = _deterministic_seed(rep, shift_type, severity, onset)
                    batches = inject(shift_type, severity, onset, seed, df, config)
                    results = _per_metric_alert_series(batches, reference_df, emb_detector, config, pred_detector, perf_detector)
                    for m in METRICS:
                        alerts_after = results[(results["batch"] >= config.onset_batch) & results[f"{m}_alert"]]
                        detected = len(alerts_after) > 0
                        lag = int(alerts_after["batch"].iloc[0] - config.onset_batch) if detected else None
                        pre_onset = results[results["batch"] < config.onset_batch]
                        far = float(pre_onset[f"{m}_alert"].mean()) if len(pre_onset) else 0.0
                        records.append({
                            "shift_type": shift_type, "severity": severity, "onset": onset,
                            "rep": rep, "metric": m, "detected": detected, "lag": lag, "false_alarm": far,
                        })
    elapsed = time.time() - t0
    print(f"Ran {n_base_configs} scenarios x {config.n_reps} reps = {total} total runs "
          f"in {elapsed:.1f}s -> {len(records)} metric-level rows")
    return pd.DataFrame(records)


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion. Far better than a
    normal-approximation CI for small n (we only have 5 reps per grid
    cell) and for proportions near 0 or 1 (many of ours are exactly 0.0
    or 1.0), where the normal approximation breaks down badly."""
    if n == 0:
        return (np.nan, np.nan)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize_grid(records: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        detected = g["detected"]
        lags = g.loc[detected, "lag"]
        n = len(detected)
        ci_low, ci_high = wilson_ci(int(detected.sum()), n)
        return pd.Series({
            "sensitivity": detected.mean(),
            "sensitivity_ci_low": ci_low,
            "sensitivity_ci_high": ci_high,
            "n_reps": n,
            "mean_lag_when_detected": lags.mean() if len(lags) else np.nan,
            "mean_false_alarm_rate": g["false_alarm"].mean(),
        })
    return records.groupby(["shift_type", "severity", "onset", "metric"]).apply(agg, include_groups=False).reset_index()


def find_failure_modes(summary: pd.DataFrame, sensitivity_threshold: float = 0.5,
                        lag_threshold: float = 5.0) -> pd.DataFrame:
    failures = summary[(summary["sensitivity"] < sensitivity_threshold) |
                        (summary["mean_lag_when_detected"] > lag_threshold)].copy()
    failures["failure_type"] = np.where(failures["sensitivity"] < sensitivity_threshold,
                                         "low sensitivity", "slow detection")
    return failures.sort_values(["shift_type", "severity", "onset", "metric"])


def best_detector_per_shift_type(summary: pd.DataFrame) -> pd.DataFrame:
    """Recommendation table: for each shift type, which metric has the
    highest mean sensitivity (averaged across severity + onset)? Also
    reports the average of the per-cell Wilson CIs, so the table doesn't
    imply more precision than 5 reps/cell actually supports."""
    ranked = summary.groupby(["shift_type", "metric"]).agg(
        sensitivity=("sensitivity", "mean"),
        sensitivity_ci_low=("sensitivity_ci_low", "mean"),
        sensitivity_ci_high=("sensitivity_ci_high", "mean"),
    ).reset_index()
    best = ranked.loc[ranked.groupby("shift_type")["sensitivity"].idxmax()]
    return best.rename(columns={"metric": "recommended_detector", "sensitivity": "mean_sensitivity"}).reset_index(drop=True)
