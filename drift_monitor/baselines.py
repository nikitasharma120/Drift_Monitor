"""
baselines.py - the NAMED baseline the five detectors are compared against.

Baseline: **per-feature two-sample Kolmogorov-Smirnov test with a
Bonferroni correction** ("ks_baseline"). This is the textbook, tuning-free
way to ask "did any input feature's distribution change?":

  1. for each feature, KS statistic D between the reference window and the batch
  2. asymptotic p-value from the Kolmogorov distribution
  3. Bonferroni: p_adj = min(1, n_features * min_p)
  4. score = -log10(p_adj)   (higher = more drift; alpha = 0.01 <=> score 2)

It looks only at X, so it shares PSI/KL's blind spot for label shift and
concept drift by construction. It is NOT calibrated per dataset (a test with
a fixed alpha is the point of a baseline), and it is run through exactly the
same 2-consecutive-batch alert rule as the five detectors, on exactly the
same runs. The question it answers: "did PSI / KL / embedding / the two
model-based detectors buy anything over an off-the-shelf test?"
"""

import numpy as np
from scipy.stats import kstwobign


class KSBaseline:
    def __init__(self, features):
        self.features = features
        self._ref_sorted = {}
        self._ref_cdf_at_ref = {}

    def fit(self, reference_df):
        for f in self.features:
            r = np.sort(reference_df[f].to_numpy(dtype=float))
            self._ref_sorted[f] = r
            self._ref_cdf_at_ref[f] = np.searchsorted(r, r, side="right") / len(r)
        return self

    def _ks_stat(self, f, cur):
        r = self._ref_sorted[f]
        c = np.sort(cur)
        n_r, n_c = len(r), len(c)
        d1 = np.abs(np.searchsorted(r, c, side="right") / n_r - np.searchsorted(c, c, side="right") / n_c).max()
        d2 = np.abs(self._ref_cdf_at_ref[f] - np.searchsorted(c, r, side="right") / n_c).max()
        return float(max(d1, d2)), n_r, n_c

    def drift_score(self, batch_df) -> dict:
        p_min = 1.0
        for f in self.features:
            d, n_r, n_c = self._ks_stat(f, batch_df[f].to_numpy(dtype=float))
            ne = n_r * n_c / (n_r + n_c)
            p = float(kstwobign.sf(np.sqrt(ne) * d))
            p_min = min(p_min, p)
        p_adj = min(1.0, len(self.features) * p_min)
        return {"ks_baseline": float(-np.log10(max(p_adj, 1e-300)))}
