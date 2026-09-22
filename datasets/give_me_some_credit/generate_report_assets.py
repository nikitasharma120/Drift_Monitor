"""generate_report_assets.py — runs the full pipeline via drift_monitor and saves
every figure/table the Phase 5 report embeds.

Run this from INSIDE this folder (datasets/give_me_some_credit/):
    cd datasets/give_me_some_credit
    python generate_report_assets.py

It reaches the shared drift_monitor/ package two directories up (the repo
root) -- that package is intentionally NOT duplicated per dataset, since
both datasets use the exact same, unmodified detection code. Only this
script's configuration differs between datasets.
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

OUT = "report_assets"
os.makedirs(OUT, exist_ok=True)

# ---- EDIT THIS to point at your local CSV ----
# Edit HERE, not drift_monitor/config.py's default: that default is the
# shared library default for anyone importing this package, and editing
# it (as happened during this project) makes the reusable package silently
# fall back to synthetic data for everyone else, with no error, no warning
# they'd notice. This script-local override is the one place to point at
# your own machine's data path.
CSV_PATH = "cs-training.csv"   # sits in this same folder; only needs changing if you move the file
# -----------------------------------------------

config = DriftMonitorConfig(csv_path=CSV_PATH)
df = load_dataset(config)
if CSV_PATH:
    import os as _os
    if not _os.path.exists(CSV_PATH):
        print(f"WARNING: CSV_PATH='{CSV_PATH}' does not exist -- silently "
              f"using synthetic demo data instead. Every number below will "
              f"be for synthetic data, not your real dataset.")

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
