"""injection.py — batching, timing control, and the three shift injectors, all config-driven."""

import numpy as np
import pandas as pd


def make_batches(df: pd.DataFrame, n_batches: int, shuffle: bool = True, seed: int = 42):
    data = df.sample(frac=1.0, random_state=seed).reset_index(drop=True) if shuffle else df.reset_index(drop=True)
    idx_splits = np.array_split(np.arange(len(data)), n_batches)
    return [data.iloc[idx].reset_index(drop=True) for idx in idx_splits]


def severity_schedule(n_batches: int, onset_batch: int, onset: str = "abrupt", ramp_batches: int = 5) -> np.ndarray:
    sched = np.zeros(n_batches)
    if onset == "abrupt":
        sched[onset_batch:] = 1.0
    elif onset == "gradual":
        ramp_end = min(onset_batch + ramp_batches, n_batches)
        ramp_len = max(ramp_end - onset_batch, 1)
        sched[onset_batch:ramp_end] = np.linspace(1.0 / ramp_len, 1.0, ramp_len)
        sched[ramp_end:] = 1.0
    else:
        raise ValueError("onset must be 'abrupt' or 'gradual'")
    return sched


def inject_covariate_shift(batches, feature, severity, onset_batch, config, onset="abrupt",
                            ramp_batches=None, seed=42):
    """Shifts `feature`'s central tendency by severity * a ROBUST scale
    estimate (IQR / 1.349, the normal-equivalent of std derived from the
    interquartile range). Raw std is deliberately avoided here: a single
    extreme outlier (common in real financial data — e.g. Give Me Some
    Credit's known DebtRatio/RevolvingUtilization data-quality artifacts)
    can inflate std by 100-1000x, which would blow "mild" shift up into
    an unrealistic, severity-saturating jump. IQR/1.349 is unaffected by
    tail outliers, so severity levels stay meaningfully differentiated
    regardless of how contaminated the raw feature is."""
    ramp_batches = config.ramp_batches if ramp_batches is None else ramp_batches
    n = len(batches)
    ref = pd.concat(batches[:max(1, onset_batch)])[feature]
    robust_scale = (ref.quantile(0.75) - ref.quantile(0.25)) / 1.349
    std_shift = config.severity_covariate[severity]
    sched = severity_schedule(n, onset_batch, onset, ramp_batches)
    out = []
    for i, batch in enumerate(batches):
        b = batch.copy()
        b[feature] = b[feature] + std_shift * sched[i] * robust_scale
        out.append(b)
    return out


def inject_label_shift(batches, label_col, severity, onset_batch, config, onset="abrupt",
                        ramp_batches=None, seed=42):
    ramp_batches = config.ramp_batches if ramp_batches is None else ramp_batches
    rng = np.random.default_rng(seed)
    n = len(batches)
    base = pd.concat(batches[:max(1, onset_batch)])[label_col].mean()
    pp_shift = config.severity_label[severity]
    sched = severity_schedule(n, onset_batch, onset, ramp_batches)
    out = []
    for i, batch in enumerate(batches):
        target_rate = np.clip(base + pp_shift * sched[i], 0.01, 0.99)
        pos = batch[batch[label_col] == 1]
        neg = batch[batch[label_col] == 0]
        n_total = len(batch)
        n_pos_target = int(round(target_rate * n_total))
        n_neg_target = n_total - n_pos_target
        pos_sample = pos.sample(n=n_pos_target, replace=True, random_state=int(rng.integers(1e6))) if n_pos_target > 0 and len(pos) > 0 else pos.iloc[0:0]
        neg_sample = neg.sample(n=n_neg_target, replace=True, random_state=int(rng.integers(1e6))) if n_neg_target > 0 and len(neg) > 0 else neg.iloc[0:0]
        b = pd.concat([pos_sample, neg_sample]).sample(frac=1.0, random_state=int(rng.integers(1e6))).reset_index(drop=True)
        out.append(b)
    return out


def inject_concept_drift(batches, feature, label_col, severity, onset_batch, config, onset="abrupt",
                          ramp_batches=None, seed=42):
    ramp_batches = config.ramp_batches if ramp_batches is None else ramp_batches
    n = len(batches)
    ref_data = pd.concat(batches[:max(1, onset_batch)])[feature]
    median, iqr = ref_data.median(), ref_data.quantile(0.75) - ref_data.quantile(0.25)
    frac_shift = config.severity_concept[severity]
    sched = severity_schedule(n, onset_batch, onset, ramp_batches)
    out = []
    for i, batch in enumerate(batches):
        b = batch.copy()
        band_width = frac_shift * sched[i] * iqr
        if band_width > 0:
            lo, hi = median, median + band_width
            mask = (b[feature] >= lo) & (b[feature] <= hi)
            b.loc[mask, label_col] = 1 - b.loc[mask, label_col]
        out.append(b)
    return out


def inject(shift_type, severity, onset, seed, df, config):
    """Convenience dispatcher used by evaluate.py and the dashboard."""
    batches = make_batches(df, config.n_batches, seed=seed)
    if shift_type == "covariate":
        return inject_covariate_shift(batches, config.target_feature, severity, config.onset_batch,
                                       config, onset=onset, seed=seed)
    elif shift_type == "label":
        return inject_label_shift(batches, config.label_col, severity, config.onset_batch,
                                   config, onset=onset, seed=seed)
    elif shift_type == "concept":
        return inject_concept_drift(batches, config.target_feature, config.label_col, severity,
                                     config.onset_batch, config, onset=onset, seed=seed)
    raise ValueError(f"Unknown shift_type: {shift_type}")
