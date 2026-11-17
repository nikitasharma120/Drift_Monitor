"""
reference_model.py - the model being "watched" by the monitor, plus a
detector that tracks its predicted positive-rate / predicted-probability
distribution over time.

This directly targets label shift: PSI/KL/embedding drift all watch X,
and label shift (P(Y) changes, P(X|Y) fixed) doesn't have to move X much
at all. A trained model's predicted probability, in contrast, is exactly
the kind of signal that should move when the mix of "risky-looking" vs
"safe-looking" rows in a batch changes -- which is what label-shift
resampling actually does under the hood.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .metrics import psi


from .preprocessing import Winsorizer  # noqa: E402,F401  (kept importable from here)


def train_reference_model(reference_df, config) -> Pipeline:
    """A simple, fixed classifier -- deliberately not the object of study,
    just the thing being monitored (see Phase 1 design doc, Section 4).
    Pipeline: winsorize (clip outliers to reference-window percentiles) ->
    standardize -> logistic regression. Both the winsorizer's clip bounds
    and the scaler's mean/std are learned ONCE on the reference window and
    reused unchanged on every future batch, so the model itself never
    moves -- only the incoming data does."""
    X = reference_df[config.features]
    y = reference_df[config.label_col]
    model = Pipeline([
        ("winsorize", Winsorizer(0.01, 0.99)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X, y)
    return model


class PredictedProportionDetector:
    """Tracks two related signals against a fixed reference model:
    - pred_rate_delta: |predicted positive rate now - predicted positive
      rate in the reference window| (simple, interpretable, in probability
      units 0-1)
    - pred_prop_psi: PSI applied to the predicted-probability distribution
      itself (more sensitive than the rate alone, since it responds to
      shifts in the shape of the score distribution, not just where it
      crosses 0.5)
    """

    def __init__(self, model, features):
        self.model = model
        self.features = features
        self.ref_rate = None
        self.ref_probs = None

    def fit(self, reference_df):
        self.ref_probs = self.model.predict_proba(reference_df[self.features])[:, 1]
        self.ref_rate = float((self.ref_probs >= 0.5).mean())
        return self

    def drift_score(self, batch_df, probs=None) -> dict:
        """probs: optional precomputed predict_proba()[:,1] for this batch, so
        the monitor scores each batch with the model only once."""
        if probs is None:
            probs = self.model.predict_proba(batch_df[self.features])[:, 1]
        pred_rate = float((probs >= 0.5).mean())
        rate_delta = abs(pred_rate - self.ref_rate)
        prob_psi = psi(self.ref_probs, probs)
        return {"pred_rate_delta": rate_delta, "pred_prop_psi": prob_psi, "pred_rate": pred_rate}
