"""data.py - load a real CSV via config, or fall back to synthetic demo data."""

import os
import numpy as np
import pandas as pd


def load_dataset(config) -> pd.DataFrame:
    """Loads config.csv_path if set and found, applying config.column_rename.
    Falls back to a synthetic Give-Me-Some-Credit-shaped dataset otherwise,
    so the pipeline always runs end to end even without real data on hand."""
    if config.csv_path and os.path.exists(config.csv_path):
        df = pd.read_csv(config.csv_path)
        if config.column_rename:
            df = df.rename(columns=config.column_rename)
        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
        missing = [c for c in [config.label_col] + config.features if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing expected columns after rename: {missing}. "
                f"Check config.column_rename against your CSV's actual header."
            )
        df = df[[config.label_col] + config.features].dropna().reset_index(drop=True)
        return df

    return _make_synthetic_dataset(config)


def _make_synthetic_dataset(config, n_rows: int = 20000) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    df = pd.DataFrame({
        "RevolvingUtilization": rng.beta(2, 5, n_rows),
        "age": rng.normal(45, 12, n_rows).clip(21, 90),
        "NumTime30_59DaysPastDue": rng.poisson(0.3, n_rows),
        "DebtRatio": rng.gamma(2, 0.3, n_rows),
        "MonthlyIncome": rng.lognormal(8.5, 0.6, n_rows),
        "NumOpenCreditLines": rng.poisson(8, n_rows),
        "NumTimes90DaysLate": rng.poisson(0.1, n_rows),
        "NumRealEstateLoans": rng.poisson(1.2, n_rows),
        "NumTime60_89DaysPastDue": rng.poisson(0.1, n_rows),
        "NumDependents": rng.poisson(0.9, n_rows),
    })
    risk_score = (
        1.5 * df["RevolvingUtilization"] + 0.8 * df["NumTime30_59DaysPastDue"]
        + 0.5 * df["DebtRatio"] - 0.02 * (df["age"] - 45) - 0.00002 * df["MonthlyIncome"]
        + rng.normal(0, 0.5, n_rows)
    )
    threshold = np.quantile(risk_score, 0.93)
    df["label"] = (risk_score > threshold).astype(int)

    if not set(config.features).issubset(df.columns) or config.label_col != "label":
        print("NOTE: config.features/label_col don't match the synthetic dataset's "
              "default schema (RevolvingUtilization, age, ... / 'label'). Returning "
              "the synthetic data as-is - set config.csv_path to use your real schema.")
        return df
    return df[["label"] + config.features]
