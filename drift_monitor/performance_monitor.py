"""
performance_monitor.py - the concept-drift detector.

PSI, KL, embedding drift, and even the predicted-proportion detector all
watch X (or a fixed model's predictions on X). Concept drift is defined as
a change in P(Y|X) that need not move X at all -- so by construction, none
of those signals can see it. The only way to catch it is to compare the
model's predictions against what actually happened.

In production this means waiting for outcomes to arrive -- e.g. did this
loan actually default? That feedback is usually delayed and partial, not
instant and complete. We simulate that here: each incoming batch reveals
true labels for only a small, randomly sampled fraction of its rows (a
"labeled trickle"), and everything else stays unlabeled, exactly like a
delayed-feedback production setting.

Two signals are tracked:
  - brier_delta: how much worse the model's calibration has gotten on the
    labeled trickle, vs. its reference-window Brier score. This is what
    actually responds to concept drift, since it directly compares
    predictions to real outcomes.
  - confidence_delta: how much the model's own confidence in its
    predictions has shifted, computed on the FULL batch (no labels
    needed). For concept drift specifically, X hasn't moved, so a
    deterministic model's confidence should stay flat even while its
    predictions are quietly becoming wrong -- that mismatch (rising
    brier_delta with flat confidence_delta) is the concept-drift
    signature this detector is built to catch.
"""

import numpy as np


class PerformanceMonitor:
    def __init__(self, model, features, label_col, labeled_fraction=0.2, seed=42):
        self.model = model
        self.features = features
        self.label_col = label_col
        self.labeled_fraction = labeled_fraction
        self.rng = np.random.default_rng(seed)
        self.ref_brier = None
        self.ref_confidence = None

    def fit(self, reference_df):
        probs = self.model.predict_proba(reference_df[self.features])[:, 1]
        y = reference_df[self.label_col].values
        self.ref_brier = float(np.mean((probs - y) ** 2))
        self.ref_confidence = float(np.mean(np.maximum(probs, 1 - probs)))
        return self

    def drift_score(self, batch_df, probs=None) -> dict:
        # Full-batch confidence needs no labels -- cheap, always available.
        # probs: optional precomputed predict_proba()[:,1] for this batch.
        probs_full = self.model.predict_proba(batch_df[self.features])[:, 1] if probs is None else probs
        confidence = float(np.mean(np.maximum(probs_full, 1 - probs_full)))
        confidence_delta = confidence - self.ref_confidence

        # Labeled trickle: only a random fraction of rows reveal ground truth.
        n = len(batch_df)
        n_labeled = max(1, int(round(n * self.labeled_fraction)))
        idx = self.rng.choice(n, size=n_labeled, replace=False)   # positional: the model is row-wise,
        probs_sample = probs_full[idx]                            # so sub-sampling the full-batch probs is identical
        y_sample = batch_df[self.label_col].to_numpy()[idx]       # to re-scoring the sampled rows
        brier = float(np.mean((probs_sample - y_sample) ** 2))
        brier_delta = brier - self.ref_brier

        # The concept-drift signature: performance got worse (brier_delta > 0)
        # without a corresponding drop in the model's own confidence -- i.e.
        # it's now confidently wrong, not just less sure of itself.
        concept_signal = max(brier_delta - max(confidence_delta, 0), 0.0)

        return {
            "brier": brier, "brier_delta": brier_delta,
            "confidence": confidence, "confidence_delta": confidence_delta,
            "concept_signal": concept_signal,
        }
