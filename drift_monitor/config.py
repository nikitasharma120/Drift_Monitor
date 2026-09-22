"""
config.py — every knob the pipeline needs, in one place.

To point this pipeline at a NEW dataset/model, you only need to change
a DriftMonitorConfig instance — nothing else in the package requires
editing. See README.md for a worked example.
"""

from dataclasses import dataclass, field


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
        # real-CSV-column -> internal-name. Edit/replace for a new dataset;
        # leave {} if your CSV's columns already match `features` + `label_col`.
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

    # --- Severity levels (tune per-dataset — see README "Recalibrating severities") ---
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

    # --- Evaluation grid ---
    shift_types: list = field(default_factory=lambda: ["covariate", "label", "concept"])
    severities: list = field(default_factory=lambda: ["mild", "moderate", "severe"])
    onsets: list = field(default_factory=lambda: ["abrupt", "gradual"])
    n_reps: int = 20                       # repetitions per grid cell (raised from 5 -- see README "Confidence Intervals")
    seed: int = 42
