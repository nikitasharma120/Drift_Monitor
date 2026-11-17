"""metrics.py - PSI, KL divergence, and PCA-based embedding drift."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .preprocessing import Winsorizer


def quantile_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bin edges with duplicates removed.

    On zero-inflated or low-cardinality features (Adult's capital-gain, binary
    flags) many quantiles are identical, which would create zero-width bins.
    Only the INTERIOR edges are de-duplicated, then -inf and +inf are added as
    outer edges. Deduplicating all edges first and then overwriting the ends with
    +/-inf would collapse a binary feature to a single bin and make its PSI
    identically 0 (tests/test_design.py checks this case)."""
    inner = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1))[1:-1])
    return np.concatenate([[-np.inf], inner, [np.inf]])


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    edges = quantile_edges(reference, n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def kl_divergence(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """KL(current || reference)."""
    edges = quantile_edges(reference, n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_p = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_p = np.clip(cur_counts / max(len(current), 1), 1e-6, None)
    return float(np.sum(cur_p * np.log(cur_p / ref_p)))


def multi_feature_psi(reference_df, current_df, features, n_bins=10) -> dict:
    return {f: psi(reference_df[f].values, current_df[f].values, n_bins) for f in features}


def multi_feature_kl(reference_df, current_df, features, n_bins=10) -> dict:
    return {f: kl_divergence(reference_df[f].values, current_df[f].values, n_bins) for f in features}


class PSIKLMonitor:
    """Cached PSI/KL: precomputes bin edges and the reference window's own
    histogram percentages ONCE at fit time, reusing them for every future
    batch. multi_feature_psi()/multi_feature_kl() above recompute both
    from scratch on every call, even though the reference data never
    changes -- np.quantile() is a sort, so this is wasted work repeated
    every batch. Benchmarked: this cached version cuts per-batch PSI+KL
    cost from ~23ms to ~2ms combined on the 10-feature Give Me Some Credit
    setup (see benchmark.py), with numerically identical results -- this
    is a pure performance fix, not a behavior change."""

    def __init__(self, features, n_bins=10):
        self.features = features
        self.n_bins = n_bins
        self._edges = {}
        self._ref_pct = {}

    def fit(self, reference_df):
        for f in self.features:
            ref = reference_df[f].values
            edges = quantile_edges(ref, self.n_bins)
            ref_counts, _ = np.histogram(ref, bins=edges)
            self._edges[f] = edges
            self._ref_pct[f] = np.clip(ref_counts / max(len(ref), 1), 1e-6, None)
        return self

    def scores(self, batch_df) -> dict:
        psi_scores, kl_scores = {}, {}
        for f in self.features:
            cur = batch_df[f].values
            cur_counts, _ = np.histogram(cur, bins=self._edges[f])
            cur_pct = np.clip(cur_counts / max(len(cur), 1), 1e-6, None)
            ref_pct = self._ref_pct[f]
            psi_scores[f] = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
            kl_scores[f] = float(np.sum(cur_pct * np.log(cur_pct / ref_pct)))
        return {"psi": psi_scores, "kl": kl_scores}


def aggregate(metric_dict: dict, how: str = "max") -> float:
    vals = list(metric_dict.values())
    return float(np.max(vals)) if how == "max" else float(np.mean(vals))


class EmbeddingDriftDetector:
    """PCA embedding fit on the reference window; tracks centroid distance
    and PSI-on-leading-component drift for incoming batches."""

    def __init__(self, features, n_components=3):
        self.features = features
        self.winsorizer = Winsorizer(0.01, 0.99)   # robust to extreme outliers (e.g. DebtRatio) before PCA
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.ref_embedding = None
        self.ref_centroid = None

    def fit(self, reference_df):
        clipped = self.winsorizer.fit_transform(reference_df[self.features])
        X = self.scaler.fit_transform(clipped)
        self.ref_embedding = self.pca.fit_transform(X)
        self.ref_centroid = self.ref_embedding.mean(axis=0)
        return self

    def transform(self, batch_df):
        X = self.scaler.transform(self.winsorizer.transform(batch_df[self.features]))
        return self.pca.transform(X)

    def drift_score(self, batch_df) -> dict:
        emb = self.transform(batch_df)
        centroid_dist = float(np.linalg.norm(emb.mean(axis=0) - self.ref_centroid))
        emb_psi = psi(self.ref_embedding[:, 0], emb[:, 0])
        return {"centroid_dist": centroid_dist, "embedding_psi": emb_psi}
