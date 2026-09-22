"""
reference_model.py — the model being "watched" by the monitor, plus a
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


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips each feature to the [lower_pct, upper_pct] range learned from
    the fit data, and reuses those SAME bounds on any future data (never
    refit on new batches). This is not just a scaling issue: this dataset's
    DebtRatio still sits at 130+ standard deviations from the mean even
    after StandardScaler, because scaling doesn't remove outlier leverage,
    it only changes the units it's measured in. That leaves the logistic
    regression's loss surface highly sensitive to those few extreme rows --
    small floating-point differences (different BLAS backends, thread
    counts, or library versions) can steer the optimizer to a meaningfully
    different fit on identical data, which is what caused the cross-
    environment reproducibility failure documented in Appendix A. Clipping
    to the 1st/99th percentile brings DebtRatio down to ~9 std-units and
    similarly tames every other feature, which is what actually stabilizes
    the model -- pinning library versions alone was necessary but not
    sufficient."""

    def __init__(self, lower_pct=0.01, upper_pct=0.99):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = X.quantile(self.lower_pct)
        self.upper_ = X.quantile(self.upper_pct)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1).values


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

    def drift_score(self, batch_df) -> dict:
        probs = self.model.predict_proba(batch_df[self.features])[:, 1]
        pred_rate = float((probs >= 0.5).mean())
        rate_delta = abs(pred_rate - self.ref_rate)
        prob_psi = psi(self.ref_probs, probs)
        return {"pred_rate_delta": rate_delta, "pred_prop_psi": prob_psi, "pred_rate": pred_rate}
