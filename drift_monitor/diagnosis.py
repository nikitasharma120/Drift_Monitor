"""
diagnosis.py — turns the five raw detector alerts into a single shift-type
diagnosis, instead of reading each metric in isolation.

Precedence order:
  1. psi_agg or kl_agg alert -> "covariate" (high-precision input detector)
  2. pred_prop_psi AND concept_signal both alert -> "label"
  3. pred_prop_psi alert alone (without performance degradation) -> "covariate_or_mild_label"
     (captures high-leverage covariate shift where PSI was insensitive, e.g. Adult Income)
  4. concept_signal alert alone -> "concept_or_mild_label"
  5. none of the above -> "none"
"""

import pandas as pd


def classify_shift(row) -> str:
    """row: a single row from run_monitor()'s output (or dict/Series
    with the *_alert columns)."""
    psi_fired = row["psi_agg_alert"] != "none"
    kl_fired = row["kl_agg_alert"] != "none"
    pred_fired = row["pred_prop_psi_alert"] != "none"
    concept_fired = row["concept_signal_alert"] != "none"

    if psi_fired or kl_fired:
        return "covariate"
    if pred_fired and concept_fired:
        return "label"
    if pred_fired and not concept_fired:
        return "covariate_or_mild_label"
    if concept_fired:
        return "concept_or_mild_label"
    return "none"


def add_diagnosis(results_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'diagnosis' column to a run_monitor() results dataframe."""
    results_df = results_df.copy()
    results_df["diagnosis"] = results_df.apply(classify_shift, axis=1)
    return results_df


def diagnostic_accuracy(records_with_truth: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe with columns ['true_shift_type', 'diagnosis'],
    returns the confusion matrix (fraction of alerted batches assigned
    each diagnosis, per true shift type)."""
    records_with_truth = records_with_truth.copy().reset_index(drop=True)
    confusion = pd.crosstab(records_with_truth["true_shift_type"],
                             records_with_truth["diagnosis"], normalize="index")
    return confusion.round(3)
