"""
performance_monitor.py — the concept-drift detector.

PSI, KL, and embedding drift all monitor X (or model outputs on X).
Concept drift is defined as a change in P(Y|X) that need not move X at all.
The only way to catch it is to compare model predictions against true outcomes.

In production, ground truth feedback is often partial and delayed. We simulate
this via a 'labeled trickle': a random sample of each incoming batch (20%)
reveals true labels, while the rest remains unlabeled.
"""

import numpy as np


class PerformanceMonitor:
    def __init__(self, model, features, label_col, labeled_fraction=0.2, seed=42):
        self.model = model
        self.features = features
        self.label_col = label_col
        self.labeled_fraction = labeled_fraction
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.ref_brier = None
        self.ref_confidence = None

    def fit(self, reference_df):
        probs = self.model.predict_proba(reference_df[self.features])[:, 1]
        y = reference_df[self.label_col].values
        self.ref_brier = float(np.mean((probs - y) ** 2))
        self.ref_confidence = float(np.mean(np.maximum(probs, 1 - probs)))
        return self

    def reset_seed(self, seed: int):
        """Allows resetting PRNG per evaluation repetition for reproducible sampling."""
        self.rng = np.random.default_rng(seed)

    def drift_score(self, batch_df) -> dict:
        # Full-batch confidence (requires no labels)
        probs_full = self.model.predict_proba(batch_df[self.features])[:, 1]
        confidence = float(np.mean(np.maximum(probs_full, 1 - probs_full)))
        confidence_delta = confidence - self.ref_confidence

        # Labeled trickle: randomly sampled fraction of rows reveal ground truth
        n = len(batch_df)
        n_labeled = max(1, int(round(n * self.labeled_fraction)))
        idx = self.rng.choice(batch_df.index.values, size=n_labeled, replace=False)
        sample = batch_df.loc[idx]
        probs_sample = self.model.predict_proba(sample[self.features])[:, 1]
        y_sample = sample[self.label_col].values

        # Brier score: mean squared error of predicted probabilities vs binary outcome
        brier = float(np.mean((probs_sample - y_sample) ** 2))
        brier_delta = brier - self.ref_brier

        # Concept drift signal: strictly measures calibration degradation on labeled trickle
        concept_signal = max(brier_delta, 0.0)

        return {
            "brier": brier,
            "brier_delta": brier_delta,
            "confidence": confidence,
            "confidence_delta": confidence_delta,
            "concept_signal": concept_signal,
        }
