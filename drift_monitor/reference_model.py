"""
reference_model.py — the model being "watched" by the monitor, plus a
detector that tracks its predicted positive-rate / predicted-probability
distribution over time.
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
    refit on new batches). Prevents extreme outlier leverage (e.g. DebtRatio > 100,000)
    from destabilizing linear models and variance estimators."""

    def __init__(self, lower_pct=0.01, upper_pct=0.99):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = X.quantile(self.lower_pct)
        self.upper_ = X.quantile(self.upper_pct)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1).values


def train_reference_model(reference_df, config, random_state: int = 42) -> Pipeline:
    """A fixed classifier being monitored. Pipeline:
    Winsorize (1st/99th percentile) -> Standardize -> Logistic Regression.
    Trained strictly on the reference window."""
    X = reference_df[config.features]
    y = reference_df[config.label_col]
    model = Pipeline([
        ("winsorize", Winsorizer(0.01, 0.99)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    model.fit(X, y)
    return model


class PredictedProportionDetector:
    """Tracks two signals against a fixed reference model:
    - pred_rate_delta: |predicted positive rate now - reference positive rate|
    - pred_prop_psi: PSI on predicted probability distribution.
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
