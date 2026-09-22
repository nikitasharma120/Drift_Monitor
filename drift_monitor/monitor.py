"""monitor.py — runs PSI/KL/embedding-drift (and, optionally, predicted-proportion and performance drift) over a batch stream and applies alert thresholds."""

import pandas as pd

from .metrics import PSIKLMonitor, aggregate

METRICS = ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"]


def run_monitor(batches, reference_df, embedding_detector, config, pred_detector=None, perf_detector=None) -> pd.DataFrame:
    """One row per batch: raw metric values + per-metric alert level
    ('none' / 'moderate' / 'significant'), plus an `any_alert` flag.
    Requires config.consecutive_batches_required consecutive moderate+
    crossings before an alert is confirmed, to cut down single-batch noise.

    pred_detector is optional: pass a fitted PredictedProportionDetector
    (see reference_model.py) to also track predicted-probability drift.
    perf_detector is optional: pass a fitted PerformanceMonitor (see
    performance_monitor.py) to also track the concept-drift signature via
    a simulated labeled trickle. Both default to reporting 0.0 (never
    alerts) when omitted."""
    rows = []
    consecutive = {m: 0 for m in METRICS}
    psikl = PSIKLMonitor(config.features, config.n_bins).fit(reference_df)  # fit ONCE, not per batch

    for i, batch in enumerate(batches):
        scores = psikl.scores(batch)
        psi_feat, kl_feat = scores["psi"], scores["kl"]
        emb = embedding_detector.drift_score(batch)
        pred = pred_detector.drift_score(batch) if pred_detector is not None else {"pred_prop_psi": 0.0, "pred_rate_delta": 0.0, "pred_rate": None}
        perf = perf_detector.drift_score(batch) if perf_detector is not None else {"concept_signal": 0.0, "brier_delta": 0.0, "confidence_delta": 0.0}

        vals = {
            "psi_agg": aggregate(psi_feat, "max"),
            "kl_agg": aggregate(kl_feat, "max"),
            "centroid_dist": emb["centroid_dist"],
            "pred_prop_psi": pred["pred_prop_psi"],
            "concept_signal": perf["concept_signal"],
        }
        row = {"batch": i, **vals, "embedding_psi": emb["embedding_psi"],
               "pred_rate_delta": pred["pred_rate_delta"], "pred_rate": pred["pred_rate"],
               "brier_delta": perf["brier_delta"], "confidence_delta": perf["confidence_delta"],
               "top_psi_feature": max(psi_feat, key=psi_feat.get)}

        for m in METRICS:
            th = config.alert_thresholds[m]
            crossed = vals[m] >= th["moderate"]
            consecutive[m] = consecutive[m] + 1 if crossed else 0
            confirmed = consecutive[m] >= config.consecutive_batches_required
            row[f"{m}_alert"] = ("significant" if vals[m] >= th["significant"] else "moderate") if confirmed else "none"

        row["any_alert"] = any(row[f"{m}_alert"] != "none" for m in METRICS)
        rows.append(row)

    return pd.DataFrame(rows)


def _as_bool(series: pd.Series) -> pd.Series:
    """Per-metric alert columns hold 'none'/'moderate'/'significant' strings;
    any_alert is already boolean. Normalize either to a boolean mask."""
    if series.dtype == bool:
        return series
    return series != "none"


def detection_lag(results_df: pd.DataFrame, onset_batch: int, metric_col: str = "any_alert"):
    mask = _as_bool(results_df[metric_col])
    alerts_after = results_df[(results_df["batch"] >= onset_batch) & mask]
    if len(alerts_after) == 0:
        return None
    return int(alerts_after["batch"].iloc[0] - onset_batch)


def false_alarm_rate(results_df: pd.DataFrame, onset_batch: int, metric_col: str = "any_alert") -> float:
    pre_onset = results_df[results_df["batch"] < onset_batch]
    if len(pre_onset) == 0:
        return 0.0
    return float(_as_bool(pre_onset[metric_col]).mean())
