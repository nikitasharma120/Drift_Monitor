"""preprocessing.py - shared preprocessing used by the reference model AND the PCA embedding detector."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
