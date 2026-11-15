"""
config.py — every knob the pipeline needs, in one place.

To point this pipeline at a NEW dataset/model, you only need to change
a DriftMonitorConfig instance — nothing else in the package requires
editing.
"""

from dataclasses import dataclass, field
import os


@dataclass
class DriftMonitorConfig:
    # --- Data ---
    csv_path: str = ""                     # path to your CSV; "" -> synthetic demo data
    label_col: str = "label"               # binary (0/1) label column name
    features: list = field(default_factory=lambda: [
        "RevolvingUtilization", "age", "NumTime30_59DaysPastDue",
        "DebtRatio", "MonthlyIncome", "NumOpenCreditLines",
        "NumTimes90DaysLate", "NumRealEstateLoans", "NumTime60_89DaysPastDue",
        "NumDependents",
    ])
    column_rename: dict = field(default_factory=lambda: {
        "SeriousDlqin2yrs": "label",
        "RevolvingUtilizationOfUnsecuredLines": "RevolvingUtilization",
        "NumberOfTime30-59DaysPastDueNotWorse": "NumTime30_59DaysPastDue",
        "NumberOfOpenCreditLinesAndLoans": "NumOpenCreditLines",
        "NumberOfTimes90DaysLate": "NumTimes90DaysLate",
        "NumberRealEstateLoansOrLines": "NumRealEstateLoans",
        "NumberOfTime60-89DaysPastDueNotWorse": "NumTime60_89DaysPastDue",
        "NumberOfDependents": "NumDependents",
    })

    # --- Batching / timing ---
    n_batches: int = 30
    onset_batch: int = 10
    ramp_batches: int = 6                  # gradual-onset ramp length
    target_feature: str = "DebtRatio"      # feature used for covariate shift + concept drift
    max_detection_window: int = 15         # maximum batches after onset for valid detection

    # --- Severity levels ---
    severity_covariate: dict = field(default_factory=lambda: {"mild": 0.5, "moderate": 1.0, "severe": 2.0})
    severity_label: dict = field(default_factory=lambda: {"mild": 0.05, "moderate": 0.10, "severe": 0.20})
    severity_concept: dict = field(default_factory=lambda: {"mild": 0.05, "moderate": 0.15, "severe": 0.30})

    # --- Detection ---
    n_bins: int = 10                       # histogram bins for PSI/KL
    embedding_components: int = 3          # PCA dimensions for embedding drift
    alert_thresholds: dict = field(default_factory=lambda: {
        "psi_agg": {"moderate": 0.10, "significant": 0.25},
        "kl_agg": {"moderate": 0.10, "significant": 0.25},
        "centroid_dist": {"moderate": 1.0, "significant": 2.0},
        "pred_prop_psi": {"moderate": 0.015, "significant": 0.03},
        "concept_signal": {"moderate": 0.025, "significant": 0.05},
    })
    labeled_trickle_fraction: float = 0.2  # fraction of each batch with "delayed" labels available
    consecutive_batches_required: int = 2  # confirmation window before an alert fires
    pure_concept: bool = True              # pure concept drift preserving marginal P(Y)

    # --- Evaluation grid ---
    shift_types: list = field(default_factory=lambda: ["covariate", "label", "concept"])
    severities: list = field(default_factory=lambda: ["mild", "moderate", "severe"])
    onsets: list = field(default_factory=lambda: ["abrupt", "gradual"])
    n_reps: int = 20                       # repetitions per grid cell
    seed: int = 42


def get_preset_config(preset_name: str = "give_me_some_credit", **overrides) -> DriftMonitorConfig:
    """Convenience factory returning fully configured presets for the 3 benchmark datasets."""
    preset = preset_name.lower().replace(" ", "_").replace("-", "_")
    
    if preset in ["give_me_some_credit", "credit", "dataset_a"]:
        cfg = DriftMonitorConfig(
            csv_path="cs-training.csv",
            label_col="label",
            features=[
                "RevolvingUtilization", "age", "NumTime30_59DaysPastDue",
                "DebtRatio", "MonthlyIncome", "NumOpenCreditLines",
                "NumTimes90DaysLate", "NumRealEstateLoans", "NumTime60_89DaysPastDue",
                "NumDependents",
            ],
            column_rename={
                "SeriousDlqin2yrs": "label",
                "RevolvingUtilizationOfUnsecuredLines": "RevolvingUtilization",
                "NumberOfTime30-59DaysPastDueNotWorse": "NumTime30_59DaysPastDue",
                "NumberOfOpenCreditLinesAndLoans": "NumOpenCreditLines",
                "NumberOfTimes90DaysLate": "NumTimes90DaysLate",
                "NumberRealEstateLoansOrLines": "NumRealEstateLoans",
                "NumberOfTime60-89DaysPastDueNotWorse": "NumTime60_89DaysPastDue",
                "NumberOfDependents": "NumDependents",
            },
            target_feature="DebtRatio",
        )
    elif preset in ["taiwan_credit_default", "taiwan", "dataset_b"]:
        cfg = DriftMonitorConfig(
            csv_path="UCI_Credit_Card.csv",
            label_col="label",
            column_rename={"default.payment.next.month": "label"},
            features=[
                "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
            ],
            target_feature="BILL_AMT1",
        )
    elif preset in ["adult_income", "adult", "census", "dataset_c"]:
        cfg = DriftMonitorConfig(
            csv_path="adult_income.csv",
            label_col="label",
            column_rename={},
            features=["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"],
            target_feature="hours-per-week",
        )
    elif preset in ["synthetic", "demo"]:
        cfg = DriftMonitorConfig(csv_path="")
    else:
        raise ValueError(f"Unknown preset name: {preset_name}")

    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
