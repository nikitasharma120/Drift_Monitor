"""generate_report_assets_adult.py — ADULT / CENSUS INCOME dataset variant.

This is the THIRD-dataset generalization test, and the first one in a
genuinely different domain: predicting whether a person's income exceeds
$50K/year from census/demographic data, not credit risk. Give Me Some
Credit and Taiwan Credit Default are both financial-risk problems with
overlapping feature semantics (debt, utilization, payment history); this
dataset shares no feature semantics with either.

Dataset: UCI "Adult" / "Census Income" (Becker & Kohavi, 1996), 48,842
rows, 23.9% positive rate (>$50K), no missing numeric values. Source used
here: a GitHub mirror (adult.csv), since the UCI repository itself isn't
reachable from this environment -- verify against the official UCI page
(https://archive.ics.uci.edu/dataset/2/adult) if you need provenance
beyond this mirror. The original "income" text column (">50K"/"<=50K")
was converted to a binary "label" column (adult_income.csv, included in
this folder) before use -- drift_monitor's loader expects an already-
binary label, and we chose to do this conversion once here rather than
add string-label handling to the shared package.

A genuinely new finding from this dataset (see this folder's README):
two of its six numeric features, capital-gain and capital-loss, have an
INTERQUARTILE RANGE OF EXACTLY ZERO -- most people report $0, so the
25th and 75th percentiles are both 0. Our robust, IQR-based severity
scaling (see the main README, bug #1) divides shift magnitude by this
value; a zero IQR means "mild", "moderate", and "severe" injected shift
would all be exactly zero, silently injecting no shift at all. We
deliberately chose hours-per-week as the target_feature instead (a
well-behaved, non-degenerate continuous feature) specifically to avoid
this failure mode. This is a sharper, more literal version of the outlier
problem the main dataset had, on a feature type that first dataset never
had -- reported honestly rather than quietly avoided without comment.

Run this from INSIDE this folder (datasets/adult_income/):
    cd datasets/adult_income
    python generate_report_assets_adult.py

It reaches the shared drift_monitor/ package two directories up (the repo
root) -- that package is intentionally NOT duplicated per dataset, since
all three datasets in this repo use the exact same, unmodified detection
code. Only this script's configuration differs between datasets.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drift_monitor import DriftMonitorConfig, load_dataset
from drift_monitor.injection import make_batches, inject
from drift_monitor.metrics import EmbeddingDriftDetector
from drift_monitor.reference_model import train_reference_model, PredictedProportionDetector
from drift_monitor.performance_monitor import PerformanceMonitor
from drift_monitor.monitor import run_monitor, detection_lag, false_alarm_rate
from drift_monitor.evaluate import run_full_grid, summarize_grid, find_failure_modes, best_detector_per_shift_type
from drift_monitor.viz import plot_monitor_dashboard, sensitivity_heatmap, metric_comparison_bar
from drift_monitor.benchmark import run_benchmark, record_system_info

OUT = "report_assets_adult"
os.makedirs(OUT, exist_ok=True)

# ---- EDIT THIS to point at your local CSV ----
CSV_PATH = "adult_income.csv"   # already has a binary "label" column -- see docstring above
# -----------------------------------------------

# Every setting below is script-local, exactly per the README's rule:
# never edit drift_monitor/config.py's shared default. This IS the
# generalization test -- the package code itself is completely unchanged
# from both other datasets; only this configuration differs.
config = DriftMonitorConfig(
    csv_path=CSV_PATH,
    label_col="label",
    column_rename={},   # adult_income.csv already has a "label" column
    features=["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"],
    target_feature="hours-per-week",   # deliberately NOT capital-gain/-loss -- see docstring: zero IQR there
)
df = load_dataset(config)
if CSV_PATH and not os.path.exists(CSV_PATH):
    print(f"WARNING: CSV_PATH='{CSV_PATH}' does not exist -- silently "
          f"using synthetic demo data instead. Every number below will "
          f"be for synthetic data, not the real Adult Income dataset.")
print(f"Loaded: {df.shape[0]} rows, positive rate = {df[config.label_col].mean():.3f}")

ref_batches = make_batches(df, config.n_batches, seed=0)
reference_df = pd.concat(ref_batches[:config.onset_batch]).reset_index(drop=True)
detector = EmbeddingDriftDetector(config.features, config.embedding_components).fit(reference_df)
ref_model = train_reference_model(reference_df, config)
pred_detector = PredictedProportionDetector(ref_model, config.features).fit(reference_df)
perf_detector = PerformanceMonitor(ref_model, config.features, config.label_col, config.labeled_trickle_fraction).fit(reference_df)

# --- 1. Demo dashboards: one per shift type, severe/abrupt ---
for shift_type in config.shift_types:
    batches = inject(shift_type, "severe", "abrupt", 42, df, config)
    results = run_monitor(batches, reference_df, detector, config, pred_detector=pred_detector, perf_detector=perf_detector)
    fig = plot_monitor_dashboard(results, config.onset_batch, config,
                                  title=f"{shift_type.capitalize()} shift (severe, abrupt)")
    fig.savefig(f"{OUT}/dashboard_{shift_type}.png", dpi=130)
    plt.close(fig)
    lag = detection_lag(results, config.onset_batch)
    far = false_alarm_rate(results, config.onset_batch)
    print(f"{shift_type}: lag={lag}, false_alarm_rate={far:.2%}")

# --- 2. Full evaluation grid ---
records = run_full_grid(df, config)
summary = summarize_grid(records)
summary.to_csv(f"{OUT}/evaluation_summary.csv", index=False)

failures = find_failure_modes(summary)
failures.to_csv(f"{OUT}/failure_modes.csv", index=False)

recommendation = best_detector_per_shift_type(summary)
recommendation.to_csv(f"{OUT}/recommendation.csv", index=False)
print("\nRecommendation table:\n", recommendation)

# --- 3. Heatmaps ---
METRICS = ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"]
for onset in config.onsets:
    fig = sensitivity_heatmap(summary, config.shift_types, config.severities, METRICS, onset)
    fig.savefig(f"{OUT}/sensitivity_heatmap_{onset}.png", dpi=130)
    plt.close(fig)

# --- 4. Metric comparison bar chart ---
fig, comp_table = metric_comparison_bar(summary, config.shift_types, METRICS)
fig.savefig(f"{OUT}/metric_comparison.png", dpi=130)
plt.close(fig)
comp_table.to_csv(f"{OUT}/metric_comparison_table.csv")

# --- 5. System cost benchmark (latency, throughput, memory per detector) ---
benchmark = run_benchmark(df, config)
benchmark.to_csv(f"{OUT}/benchmark.csv", index=False)

import json
system_info = record_system_info()
with open(f"{OUT}/system_info.json", "w") as f:
    json.dump(system_info, f, indent=2)
print("\nEnvironment fingerprint (saved to system_info.json):")
for k, v in system_info.items():
    print(f"  {k}: {v}")
print("\nSystem cost benchmark:\n", benchmark.to_string(index=False))

print("\nAll report assets written to", os.path.abspath(OUT))
