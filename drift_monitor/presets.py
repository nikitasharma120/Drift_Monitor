"""presets.py - the three dataset configurations in ONE place.

The generators, run_all.py and the dashboard all import these, so every
entry point uses identical settings. Nothing in DriftMonitorConfig's
defaults is edited per dataset: each preset passes overrides.
"""

import os

from .config import DriftMonitorConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRESETS = {
    "gmsc": {
        "title": "Give Me Some Credit",
        "folder": "give_me_some_credit",
        "csv": "cs-training.csv",
        "assets": "report_assets",
    },
    "taiwan": {
        "title": "Taiwan Credit Default",
        "folder": "taiwan_credit_default",
        "csv": "UCI_Credit_Card.csv",
        "assets": "report_assets_taiwan",
    },
    "adult": {
        "title": "Adult Income",
        "folder": "adult_income",
        "csv": "adult_income.csv",
        "assets": "report_assets_adult",
    },
}


def dataset_dir(key: str) -> str:
    return os.path.join(REPO_ROOT, "datasets", PRESETS[key]["folder"])


def assets_dir(key: str) -> str:
    return os.path.join(dataset_dir(key), PRESETS[key]["assets"])


def get_preset(key: str, **overrides) -> DriftMonitorConfig:
    csv_path = os.path.join(dataset_dir(key), PRESETS[key]["csv"])
    if key == "gmsc":
        cfg = dict(csv_path=csv_path)
    elif key == "taiwan":
        cfg = dict(
            csv_path=csv_path, label_col="label",
            column_rename={"default.payment.next.month": "label"},
            features=["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                      "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                      "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
            target_feature="BILL_AMT1",   # most recent bill amount -- closest analog to DebtRatio
        )
    elif key == "adult":
        cfg = dict(
            csv_path=csv_path, label_col="label", column_rename={},
            features=["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"],
            target_feature="hours-per-week",   # NOT capital-gain/-loss: their IQR is exactly 0
        )
    else:
        raise KeyError(f"unknown preset {key!r}; choose from {list(PRESETS)}")
    cfg.update(overrides)
    if not os.path.exists(cfg["csv_path"]):
        raise FileNotFoundError(f"{cfg['csv_path']} not found -- refusing to fall back to synthetic data silently")
    return DriftMonitorConfig(**cfg)
