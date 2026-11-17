"""tradeoff.py - sensitivity vs. false-alarm trade-off curves from the saved raw scores.

The alert rule is binary (alert / no alert at one threshold), so a single
evaluation cell is just one POINT on a trade-off curve. Because the grid
saves every raw score, we can re-apply the full 2-consecutive-batch alert
rule at many thresholds to the SAME runs and trace the whole curve:

  x = false-alarm rate : fraction of pre-onset (clean) batches in an alert state
  y = sensitivity      : fraction of runs that alert at/after onset

Also provided: a threshold-free score-level AUC (pre-onset batches vs.
fully-ramped post-onset batches), and sensitivity at fixed false-alarm budgets.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .monitor import ALL_METRICS


def _confirmed_matrix(crossed: np.ndarray, k: int) -> np.ndarray:
    """crossed: (T, R, B) bool -> bool array, True where crossed for k consecutive batches."""
    run = np.zeros(crossed.shape[:2], dtype=np.int32)
    out = np.zeros_like(crossed)
    for b in range(crossed.shape[2]):
        run = np.where(crossed[:, :, b], run + 1, 0)
        out[:, :, b] = run >= k
    return out


def operating_curve(raw: pd.DataFrame, config, metric: str, shift_type: str,
                    severity=None, n_thresholds: int = 120) -> pd.DataFrame:
    sub = raw[raw["shift_type"] == shift_type]
    if severity is not None:
        sub = sub[sub["severity"] == severity]
    runs = sub.groupby(["severity", "onset", "rep"], sort=False)
    mats = [g.sort_values("batch")[metric].to_numpy() for _, g in runs]
    S = np.vstack(mats)                                     # (R, B)
    batch = np.arange(S.shape[1])
    pre, post = batch < config.onset_batch, batch >= config.onset_batch
    qs = np.unique(np.quantile(S, np.linspace(0, 1, n_thresholds)))
    thr = np.concatenate([qs, [qs[-1] + 1e-9 + abs(qs[-1]) * 1e-6]])   # last one: never alerts -> (0, 0)
    flags = _confirmed_matrix(S[None, :, :] >= thr[:, None, None], config.consecutive_batches_required)
    far = flags[:, :, pre].mean(axis=(1, 2))
    sens = flags[:, :, post].any(axis=2).mean(axis=1)
    return pd.DataFrame({"threshold": thr, "false_alarm_rate": far, "sensitivity": sens})


def sensitivity_at_far(curve: pd.DataFrame, far_budget: float) -> float:
    ok = curve[curve["false_alarm_rate"] <= far_budget]
    return float(ok["sensitivity"].max()) if len(ok) else 0.0


def score_auc(raw: pd.DataFrame, config, metric: str, shift_type: str) -> float:
    """Threshold-free separability: AUC of pre-onset batch scores (negatives) vs.
    batches where the shift is fully on (positives: batch >= onset + ramp)."""
    sub = raw[raw["shift_type"] == shift_type]
    neg = sub.loc[sub["batch"] < config.onset_batch, metric]
    pos = sub.loc[sub["batch"] >= config.onset_batch + config.ramp_batches, metric]
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return float(roc_auc_score(y, np.r_[neg.to_numpy(), pos.to_numpy()]))


def build_tradeoff_tables(raw: pd.DataFrame, config, metrics=None, far_budgets=(0.01, 0.05)):
    """Returns (curves, summary).
    curves : shift_type, severity('all' or level), metric, threshold, false_alarm_rate, sensitivity
    summary: shift_type, metric, score_auc, sensitivity_at_far_<budget> (pooled over severities/onsets)"""
    metrics = ALL_METRICS if metrics is None else metrics
    curves, rows = [], []
    for st in config.shift_types:
        for m in metrics:
            c = operating_curve(raw, config, m, st)
            c.insert(0, "metric", m); c.insert(0, "severity", "all"); c.insert(0, "shift_type", st)
            curves.append(c)
            row = {"shift_type": st, "metric": m, "score_auc": score_auc(raw, config, m, st)}
            for b in far_budgets:
                row[f"sensitivity_at_far_{int(b * 100)}pct"] = sensitivity_at_far(c, b)
            rows.append(row)
            for sev in config.severities:
                cs = operating_curve(raw, config, m, st, severity=sev)
                cs.insert(0, "metric", m); cs.insert(0, "severity", sev); cs.insert(0, "shift_type", st)
                curves.append(cs)
    return pd.concat(curves, ignore_index=True), pd.DataFrame(rows)


def debounce_table(raw: pd.DataFrame, config, thresholds, ks=(1, 2, 3, 4)) -> pd.DataFrame:
    """Effect of the confirmation rule. For k = 1..4 consecutive batches above the calibrated
    threshold: sensitivity, false-alarm rate (clean pre-onset batches in alert) and mean detection lag
    (batches after onset, detected runs only), per shift type and detector."""
    rows = []
    for st in config.shift_types:
        sub = raw[raw["shift_type"] == st]
        groups = [g.sort_values("batch") for _, g in sub.groupby(["severity", "onset", "rep"], sort=False)]
        batch = groups[0]["batch"].to_numpy()
        pre, post = batch < config.onset_batch, batch >= config.onset_batch
        post_idx = np.flatnonzero(post)
        for m in ALL_METRICS:
            S = np.vstack([g[m].to_numpy() for g in groups])
            crossed = (S >= thresholds[m]["moderate"])[None, :, :]
            for k in ks:
                flags = _confirmed_matrix(crossed, k)[0]
                det = flags[:, post].any(axis=1)
                first = flags[:, post].argmax(axis=1)
                rows.append({"shift_type": st, "metric": m, "k": k,
                             "sensitivity": float(det.mean()),
                             "false_alarm_rate": float(flags[:, pre].mean()),
                             "mean_lag": float(first[det].mean()) if det.any() else np.nan})
    return pd.DataFrame(rows)
