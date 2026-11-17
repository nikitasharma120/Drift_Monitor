"""
diagnosis.py - turns the five raw detector alerts into a single shift-type
diagnosis, instead of reading each metric in isolation.

Why this exists: concept_signal (the Brier-based performance monitor) has
a small amount of genuine, expected leakage onto covariate shift at
moderate/severe severity -- a fixed model's calibration CAN degrade under
covariate shift too, since it's now scoring inputs it wasn't trained on.
That is real model behavior, not a defect, but read in isolation it can make
a covariate-shift batch look like it might be concept drift.

The answer is not to suppress the leakage but to never read concept_signal
in isolation. PSI/KL are the specific, high-precision detector for
covariate shift in our grids (see diagnosis_confusion.csv in each
report_assets folder for how often precedence actually resolves it), so
giving them precedence resolves most of that ambiguity using detectors we
already have. This module formalizes the label-shift/concept-drift
differential diagnosis (REPORT.md) into one function instead of
leaving the combination as a manual reading exercise.

Precedence order (checked in this order, first match wins):
  1. psi_agg or kl_agg alert  -> "covariate"
  2. pred_prop_psi AND concept_signal both alert -> "label"
  3. concept_signal alert alone -> "concept_or_mild_label"
     (see REPORT.md: this ambiguity is not resolved by any detector we have)
  4. pred_prop_psi alert alone -> "label_or_subtle_covariate"
     (an alert must never be reported as "none": the model's output distribution moved but neither
     PSI/KL nor the Brier signal confirmed it. That pattern is what label
     shift looks like when the labelled trickle is too small to confirm it,
     and what a covariate shift too small for PSI/KL looks like when the
     model is sensitive to that feature -- the two cannot be told apart.)
  5. none of the above -> "none"
"""

import pandas as pd


def classify_shift(row) -> str:
    """row: a single row from run_monitor()'s output (or any dict/Series
    with the *_alert columns: psi_agg_alert, kl_agg_alert, pred_prop_psi_alert,
    concept_signal_alert, each 'none'/'moderate'/'significant')."""
    psi_fired = row["psi_agg_alert"] != "none"
    kl_fired = row["kl_agg_alert"] != "none"
    pred_fired = row["pred_prop_psi_alert"] != "none"
    concept_fired = row["concept_signal_alert"] != "none"

    if psi_fired or kl_fired:
        return "covariate"
    if pred_fired and concept_fired:
        return "label"
    if concept_fired:
        return "concept_or_mild_label"
    if pred_fired:
        return "label_or_subtle_covariate"
    return "none"


def add_diagnosis(results_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'diagnosis' column to a run_monitor() results dataframe."""
    results_df = results_df.copy()
    results_df["diagnosis"] = results_df.apply(classify_shift, axis=1)
    return results_df


def diagnostic_accuracy(records_with_truth: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe with columns ['true_shift_type', 'diagnosis'],
    returns the confusion matrix (fraction of alerted batches assigned
    each diagnosis, per true shift type). Use this to verify the
    precedence rule actually resolves ambiguity across the full grid,
    not just anecdotally."""
    label_map = {"covariate": "covariate", "label": "label",
                 "concept_or_mild_label": "concept_or_mild_label",
                 "label_or_subtle_covariate": "label_or_subtle_covariate", "none": "none"}
    records_with_truth = records_with_truth.copy().reset_index(drop=True)
    records_with_truth["diagnosis"] = records_with_truth["diagnosis"].map(label_map)
    confusion = pd.crosstab(records_with_truth["true_shift_type"],
                             records_with_truth["diagnosis"], normalize="index")
    return confusion.round(3)
