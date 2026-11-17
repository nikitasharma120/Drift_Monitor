"""injection.py - batching, timing control, and the three shift injectors, all config-driven."""

import numpy as np
import pandas as pd


def make_batches(df: pd.DataFrame, n_batches: int, shuffle: bool = True, seed: int = 42):
    data = df.sample(frac=1.0, random_state=seed).reset_index(drop=True) if shuffle else df.reset_index(drop=True)
    idx_splits = np.array_split(np.arange(len(data)), n_batches)
    return [data.iloc[idx].reset_index(drop=True) for idx in idx_splits]


def batch_size_for(n_rows: int, config) -> int:
    """Rows per batch when the reference window AND the stream must both fit
    disjointly into the data: n_rows // (n_reference_batches + n_batches)."""
    return n_rows // (config.n_reference_batches + config.n_batches)


def make_reference_and_stream(df: pd.DataFrame, config, seed: int):
    """Shuffle once, then cut DISJOINT row ranges: the first
    n_reference_batches*B rows are the reference window (used to fit every
    detector and the reference model), the next n_batches*B rows are the
    monitored stream. No row appears in both, so pre-onset stream
    batches are genuinely unseen data and the false-alarm rate is not
    understated by reference rows leaking into the stream. A fresh split
    is drawn for every seed, so repetitions do not share one reference."""
    B = batch_size_for(len(df), config)
    data = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_ref = config.n_reference_batches * B
    reference_df = data.iloc[:n_ref].reset_index(drop=True)
    stream = data.iloc[n_ref: n_ref + config.n_batches * B]
    batches = [stream.iloc[i * B:(i + 1) * B].reset_index(drop=True) for i in range(config.n_batches)]
    return reference_df, batches


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
    extreme outlier (common in real financial data - e.g. Give Me Some
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
    """Resamples post-onset batches to a higher positive rate (P(X|Y) fixed).
    Pre-onset batches (schedule == 0) are returned UNTOUCHED, so no sampling
    noise is added to batches that are supposed to be clean. Sampling is
    without replacement whenever the batch has enough rows of that class,
    with replacement only when it must."""
    ramp_batches = config.ramp_batches if ramp_batches is None else ramp_batches
    rng = np.random.default_rng(seed)
    n = len(batches)
    base = pd.concat(batches[:max(1, onset_batch)])[label_col].mean()
    pp_shift = config.severity_label[severity]
    sched = severity_schedule(n, onset_batch, onset, ramp_batches)
    out = []
    for i, batch in enumerate(batches):
        if sched[i] == 0:
            out.append(batch.copy())
            continue
        target_rate = np.clip(base + pp_shift * sched[i], 0.01, 0.99)
        pos = batch[batch[label_col] == 1]
        neg = batch[batch[label_col] == 0]
        n_total = len(batch)
        n_pos_target = int(round(target_rate * n_total))
        n_neg_target = n_total - n_pos_target
        pos_sample = (pos.sample(n=n_pos_target, replace=n_pos_target > len(pos), random_state=int(rng.integers(1_000_000)))
                      if n_pos_target > 0 and len(pos) > 0 else pos.iloc[0:0])
        neg_sample = (neg.sample(n=n_neg_target, replace=n_neg_target > len(neg), random_state=int(rng.integers(1_000_000)))
                      if n_neg_target > 0 and len(neg) > 0 else neg.iloc[0:0])
        b = pd.concat([pos_sample, neg_sample]).sample(frac=1.0, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
        out.append(b)
    return out


def inject_concept_drift(batches, feature, label_col, severity, onset_batch, config, onset="abrupt",
                          ramp_batches=None, seed=42):
    """PURE concept drift: P(Y|X) changes while P(X) and P(Y) do not.

    In every post-onset batch let R = rows with feature >= the pre-onset
    median. A fraction f = severity_concept[severity] * schedule of the
    positives inside R is relabelled 0, and EXACTLY the same number of
    negatives outside R (feature < median) is relabelled 1. Consequences,
    all verified in tests/test_design.py:
      * every feature column is bit-for-bit unchanged  -> P(X) fixed
      * the number of positives per batch is unchanged -> P(Y) fixed
      * the feature->label relationship moves: risk that sat in the
        high-feature region moves to the low-feature region.
    Because severity is a fraction of rows, the three levels differ even
    on integer-valued targets such as Adult's hours-per-week."""
    ramp_batches = config.ramp_batches if ramp_batches is None else ramp_batches
    rng = np.random.default_rng(seed)
    n = len(batches)
    median = pd.concat(batches[:max(1, onset_batch)])[feature].median()
    frac = config.severity_concept[severity]
    sched = severity_schedule(n, onset_batch, onset, ramp_batches)
    out = []
    for i, batch in enumerate(batches):
        b = batch.copy()
        if sched[i] > 0:
            region = (b[feature] >= median).to_numpy()
            y = b[label_col].to_numpy().copy()
            pos_idx = np.flatnonzero(region & (y == 1))
            neg_idx = np.flatnonzero(~region & (y == 0))
            m = int(round(frac * sched[i] * len(pos_idx)))
            m = min(m, len(pos_idx), len(neg_idx))
            if m > 0:
                y[rng.choice(pos_idx, size=m, replace=False)] = 0
                y[rng.choice(neg_idx, size=m, replace=False)] = 1
                b[label_col] = y
        out.append(b)
    return out


def make_scenario(shift_type, severity, onset, seed, df, config):
    """Returns (reference_df, batches): a fresh disjoint reference/stream split for
    this seed, with `shift_type` injected into the stream. shift_type="none"
    returns the clean stream (used for threshold calibration)."""
    reference_df, batches = make_reference_and_stream(df, config, seed)
    if shift_type == "none":
        return reference_df, batches
    if shift_type == "covariate":
        batches = inject_covariate_shift(batches, config.target_feature, severity, config.onset_batch,
                                          config, onset=onset, seed=seed)
    elif shift_type == "label":
        batches = inject_label_shift(batches, config.label_col, severity, config.onset_batch,
                                      config, onset=onset, seed=seed)
    elif shift_type == "concept":
        batches = inject_concept_drift(batches, config.target_feature, config.label_col, severity,
                                        config.onset_batch, config, onset=onset, seed=seed)
    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")
    return reference_df, batches


def inject(shift_type, severity, onset, seed, df, config):
    """Convenience wrapper returning only the stream batches (see make_scenario)."""
    return make_scenario(shift_type, severity, onset, seed, df, config)[1]
