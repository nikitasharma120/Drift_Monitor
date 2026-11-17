"""monitor.py - fit every detector on a reference window, score a batch stream, apply alert thresholds.

Three small steps, deliberately separable so the evaluation grid can score
a stream ONCE and then re-apply many different thresholds to the saved raw
scores (that is how the ROC / trade-off curves are drawn):

    suite   = fit_detectors(reference_df, config, seed)     # fit on the reference window only
    scores  = score_batches(batches, suite)                 # raw per-batch scores, no thresholds
    results = apply_alerts(scores, config, thresholds)      # 2-consecutive rule -> alert levels

run_monitor() does all three. The five detectors in METRICS feed
`any_alert` and diagnosis.py; `ks_baseline` is the named baseline and is
reported alongside but never feeds `any_alert`.
"""

import numpy as np
import pandas as pd

from .metrics import PSIKLMonitor, EmbeddingDriftDetector, aggregate
from .reference_model import train_reference_model, PredictedProportionDetector
from .performance_monitor import PerformanceMonitor
from .baselines import KSBaseline

METRICS = ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"]
BASELINE_METRICS = ["ks_baseline"]
ALL_METRICS = METRICS + BASELINE_METRICS


class DetectorSuite:
    """Every detector, fitted on ONE reference window."""

    def __init__(self, config):
        self.config = config
        self.psikl = self.emb = self.model = self.pred = self.perf = self.ks = None

    def fit(self, reference_df, seed=0):
        c = self.config
        self.psikl = PSIKLMonitor(c.features, c.n_bins).fit(reference_df)
        self.emb = EmbeddingDriftDetector(c.features, c.embedding_components).fit(reference_df)
        self.model = train_reference_model(reference_df, c)
        self.pred = PredictedProportionDetector(self.model, c.features).fit(reference_df)
        self.perf = PerformanceMonitor(self.model, c.features, c.label_col,
                                       c.labeled_trickle_fraction, seed=seed).fit(reference_df)
        self.ks = KSBaseline(c.features).fit(reference_df)
        return self

    def score_batch(self, batch):
        c = self.config
        sc = self.psikl.scores(batch)
        emb = self.emb.drift_score(batch)
        probs = self.model.predict_proba(batch[c.features])[:, 1]      # model scored once per batch
        pred = self.pred.drift_score(batch, probs=probs)
        perf = self.perf.drift_score(batch, probs=probs)
        ks = self.ks.drift_score(batch)
        return {
            "psi_agg": aggregate(sc["psi"], "max"), "kl_agg": aggregate(sc["kl"], "max"),
            "centroid_dist": emb["centroid_dist"], "pred_prop_psi": pred["pred_prop_psi"],
            "concept_signal": perf["concept_signal"], "ks_baseline": ks["ks_baseline"],
            # diagnostics (dashboard / report)
            "embedding_psi": emb["embedding_psi"], "pred_rate_delta": pred["pred_rate_delta"],
            "pred_rate": pred["pred_rate"], "brier_delta": perf["brier_delta"],
            "confidence_delta": perf["confidence_delta"],
            "top_psi_feature": max(sc["psi"], key=sc["psi"].get),
        }


def fit_detectors(reference_df, config, seed=0) -> DetectorSuite:
    return DetectorSuite(config).fit(reference_df, seed=seed)


def score_batches(batches, suite: DetectorSuite) -> pd.DataFrame:
    """Raw per-batch scores (no thresholds applied)."""
    return pd.DataFrame([{"batch": i, **suite.score_batch(b)} for i, b in enumerate(batches)])


def confirmed_flags(values, moderate: float, k: int) -> np.ndarray:
    """True where `values` has been >= moderate for k consecutive batches (ending at that batch)."""
    crossed = np.asarray(values) >= moderate
    out = np.zeros(len(crossed), dtype=bool)
    run = 0
    for i, c in enumerate(crossed):
        run = run + 1 if c else 0
        out[i] = run >= k
    return out


def apply_alerts(scores: pd.DataFrame, config, thresholds=None) -> pd.DataFrame:
    """Adds `<metric>_alert` ('none'/'moderate'/'significant') for all six metrics
    and `any_alert` (any of the five detectors; the baseline is excluded)."""
    thresholds = config.alert_thresholds if thresholds is None else thresholds
    k = config.consecutive_batches_required
    df = scores.copy()
    for m in ALL_METRICS:
        th = thresholds[m]
        conf = confirmed_flags(df[m].to_numpy(), th["moderate"], k)
        level = np.where(df[m].to_numpy() >= th["significant"], "significant", "moderate")
        df[f"{m}_alert"] = np.where(conf, level, "none")
    df["any_alert"] = np.any([df[f"{m}_alert"] != "none" for m in METRICS], axis=0)
    return df


def run_monitor(batches, reference_df, config, thresholds=None, seed=0, suite=None) -> pd.DataFrame:
    """Fit (unless a fitted `suite` is passed), score, and apply alerts."""
    suite = fit_detectors(reference_df, config, seed) if suite is None else suite
    return apply_alerts(score_batches(batches, suite), config, thresholds)


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
    """Fraction of PRE-onset batches (no shift present) that sit in an alert state."""
    pre_onset = results_df[results_df["batch"] < onset_batch]
    if len(pre_onset) == 0:
        return 0.0
    return float(_as_bool(pre_onset[metric_col]).mean())
