"""metrics.py — PSI, KL divergence, and PCA-based embedding drift."""

import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Computes Population Stability Index (Jeffreys Divergence).
    Deduplicates quantile bin edges to prevent zero-width bin collapse on
    zero-inflated or discrete features."""
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def kl_divergence(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """KL(current || reference).
    Deduplicates quantile bin edges to prevent zero-width bin collapse."""
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
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
    """Cached PSI/KL: precomputes bin edges and reference window histogram
    percentages once at fit time, reusing them across batches. Handles duplicate
    quantile edges safely."""

    def __init__(self, features, n_bins=10):
        self.features = features
        self.n_bins = n_bins
        self._edges = {}
        self._ref_pct = {}

    def fit(self, reference_df):
        for f in self.features:
            ref = reference_df[f].values
            edges = np.quantile(ref, np.linspace(0, 1, self.n_bins + 1))
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([-np.inf, np.inf])
            else:
                edges[0], edges[-1] = -np.inf, np.inf
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
    and PSI-on-leading-component drift. Uses Winsorization before scaling
    to prevent extreme outliers (e.g. DebtRatio > 100,000) from corrupting PC1."""

    def __init__(self, features, n_components=3):
        self.features = features
        self.n_components = n_components
        self.winsorizer = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.ref_embedding = None
        self.ref_centroid = None

    def fit(self, reference_df):
        from .reference_model import Winsorizer
        self.winsorizer = Winsorizer(0.01, 0.99)
        clipped_ref = self.winsorizer.fit_transform(reference_df[self.features])
        X = self.scaler.fit_transform(clipped_ref)
        self.ref_embedding = self.pca.fit_transform(X)
        self.ref_centroid = self.ref_embedding.mean(axis=0)
        return self

    def transform(self, batch_df):
        clipped = self.winsorizer.transform(batch_df[self.features])
        X = self.scaler.transform(clipped)
        return self.pca.transform(X)

    def drift_score(self, batch_df) -> dict:
        emb = self.transform(batch_df)
        centroid_dist = float(np.linalg.norm(emb.mean(axis=0) - self.ref_centroid))
        emb_psi = psi(self.ref_embedding[:, 0], emb[:, 0])
        return {"centroid_dist": centroid_dist, "embedding_psi": emb_psi}


def ks_two_sample_test(reference_df, batch_df, features, alpha: float = 0.05) -> dict:
    """Classical two-sample Kolmogorov-Smirnov test baseline.
    Computes supremum divergence D and p-value per feature with Bonferroni correction."""
    stat_dict, pval_dict = {}, {}
    bonferroni_alpha = alpha / max(len(features), 1)
    for f in features:
        ref_v = reference_df[f].dropna().values
        cur_v = batch_df[f].dropna().values
        if len(ref_v) > 0 and len(cur_v) > 0:
            res = ks_2samp(ref_v, cur_v)
            stat_dict[f] = float(res.statistic)
            pval_dict[f] = float(res.pvalue)
        else:
            stat_dict[f] = 0.0
            pval_dict[f] = 1.0

    max_stat = float(np.max(list(stat_dict.values()))) if stat_dict else 0.0
    min_pval = float(np.min(list(pval_dict.values()))) if pval_dict else 1.0
    return {
        "ks_stat_max": max_stat,
        "ks_pval_min": min_pval,
        "ks_alert": bool(min_pval < bonferroni_alpha),
        "ks_stats": stat_dict,
        "ks_pvals": pval_dict
    }


def calibrate_empirical_thresholds(clean_batches, psikl_monitor, emb_detector, pred_detector=None, perf_detector=None, k_sigma: float = 3.0) -> dict:
    """Measures empirical noise floor on clean unshifted reference batches and calibrates
    detection thresholds at mean + k_sigma * std. Prevents uncalibrated threshold failure."""
    scores = {"psi_agg": [], "kl_agg": [], "centroid_dist": []}
    if pred_detector is not None:
        scores["pred_prop_psi"] = []
    if perf_detector is not None:
        scores["concept_signal"] = []

    for b in clean_batches:
        psikl_res = psikl_monitor.scores(b)
        scores["psi_agg"].append(aggregate(psikl_res["psi"], "max"))
        scores["kl_agg"].append(aggregate(psikl_res["kl"], "max"))
        scores["centroid_dist"].append(emb_detector.drift_score(b)["centroid_dist"])
        if pred_detector is not None:
            scores["pred_prop_psi"].append(pred_detector.drift_score(b)["pred_prop_psi"])
        if perf_detector is not None:
            scores["concept_signal"].append(perf_detector.drift_score(b)["concept_signal"])

    thresholds = {}
    for m, vals in scores.items():
        arr = np.array(vals)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        mod_th = float(max(mu + k_sigma * sigma, 1e-4))
        sig_th = float(max(mu + (k_sigma + 2.0) * sigma, mod_th * 1.5))
        thresholds[m] = {"moderate": round(mod_th, 4), "significant": round(sig_th, 4), "noise_floor": round(mu, 4)}
    return thresholds

