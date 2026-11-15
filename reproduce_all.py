"""
reproduce_all.py — The single command to regenerate all main results, benchmarks,
and trade-off curves across all three datasets.

Usage:
    python reproduce_all.py          # Full run across all 3 datasets
    python reproduce_all.py --fast   # Fast verification run (3 reps per scenario)
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drift_monitor import DriftMonitorConfig, load_dataset
from drift_monitor.config import get_preset_config
from drift_monitor.injection import make_batches, inject
from drift_monitor.metrics import EmbeddingDriftDetector, PSIKLMonitor, aggregate, calibrate_empirical_thresholds
from drift_monitor.reference_model import train_reference_model, PredictedProportionDetector
from drift_monitor.performance_monitor import PerformanceMonitor
from drift_monitor.monitor import run_monitor, detection_lag, false_alarm_rate
from drift_monitor.evaluate import run_full_grid, summarize_grid, find_failure_modes, best_detector_per_shift_type
from drift_monitor.benchmark import run_benchmark, record_system_info
from drift_monitor.viz import (
    sensitivity_heatmap,
    metric_comparison_bar,
    plot_roc_tradeoff,
    plot_debounce_tradeoff,
    plot_lag_vs_severity,
    plot_cost_frontier
)


def compute_roc_tradeoff_data(df, config):
    """Sweeps detection thresholds using disjoint train/stream splits and standard ROC analysis."""
    from sklearn.metrics import roc_curve, roc_auc_score

    n_half = len(df) // 2
    ref_df = df.iloc[:n_half].reset_index(drop=True)
    stream_pool = df.iloc[n_half:].reset_index(drop=True)

    psikl = PSIKLMonitor(config.features, config.n_bins).fit(ref_df)
    detector = EmbeddingDriftDetector(config.features, config.embedding_components).fit(ref_df)
    model = train_reference_model(ref_df, config)
    pred_detector = PredictedProportionDetector(model, config.features).fit(ref_df)
    perf_detector = PerformanceMonitor(model, config.features, config.label_col).fit(ref_df)

    # Collect raw metric values on stream across diverse shifts
    metric_records = []
    for shift in ["covariate", "label", "concept"]:
        stream_batch = inject(shift, "severe", "abrupt", 42, stream_pool, config)
        for i, b in enumerate(stream_batch):
            scores = psikl.scores(b)
            emb = detector.drift_score(b)
            pred = pred_detector.drift_score(b)
            perf = perf_detector.drift_score(b)
            metric_records.append({
                "batch": i,
                "is_drift": 1 if i >= config.onset_batch else 0,
                "psi_agg": aggregate(scores["psi"], "max"),
                "kl_agg": aggregate(scores["kl"], "max"),
                "centroid_dist": emb["centroid_dist"],
                "pred_prop_psi": pred["pred_prop_psi"],
                "concept_signal": perf["concept_signal"]
            })
    m_df = pd.DataFrame(metric_records)

    roc_dict = {}
    for m in ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"]:
        fpr, tpr, _ = roc_curve(m_df["is_drift"], m_df[m])
        auc = roc_auc_score(m_df["is_drift"], m_df[m])
        roc_dict[f"{m} (AUC={auc:.2f})"] = (fpr, tpr)
    return roc_dict


def compute_debounce_curve(df, config, n_seeds=5):
    """Computes Trade-Off Curve 2: False Alarm Rate vs. Detection Lag across window sizes (k=1 to 5)
    using calibrated threshold to demonstrate genuine single-batch noise vs. sustained drift trade-off."""
    n_half = len(df) // 2
    ref_df = df.iloc[:n_half].reset_index(drop=True)
    stream_pool = df.iloc[n_half:].reset_index(drop=True)

    clean_batches = make_batches(ref_df, 20, seed=42)
    psikl = PSIKLMonitor(config.features, config.n_bins).fit(ref_df)
    detector = EmbeddingDriftDetector(config.features, config.embedding_components).fit(ref_df)
    
    calibrated = calibrate_empirical_thresholds(clean_batches, psikl, detector, k_sigma=3.2)
    th = calibrated["psi_agg"]["moderate"]

    records = {k: {"far": [], "lag": []} for k in [1, 2, 3, 4, 5]}
    seeds = [10, 20, 30, 40, 50][:n_seeds]
    for s in seeds:
        batches = inject("covariate", "moderate", "abrupt", s, stream_pool, config)
        for k in [1, 2, 3, 4, 5]:
            c = get_preset_config("give_me_some_credit", consecutive_batches_required=k)
            c.alert_thresholds["psi_agg"]["moderate"] = th
            res = run_monitor(batches, ref_df, detector, c)
            far = false_alarm_rate(res, config.onset_batch, "psi_agg_alert")
            lag = detection_lag(res, config.onset_batch, "psi_agg_alert")
            records[k]["far"].append(far)
            records[k]["lag"].append(lag if lag is not None else 10)

    rows = []
    for k in [1, 2, 3, 4, 5]:
        mean_far = float(np.mean(records[k]["far"]))
        mean_lag = float(np.mean(records[k]["lag"]))
        rows.append({"window_size": k, "false_alarm_rate": mean_far, "mean_lag": mean_lag})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Reproduce all MLSE Drift Monitor results.")
    parser.add_argument("--fast", action="store_true", help="Fast evaluation (3 reps instead of 20)")
    parser.add_argument("--out", default="report_assets_combined", help="Combined output directory")
    args = parser.parse_args()

    n_reps = 3 if args.fast else 20
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("A SYSTEM THAT NOTICES — FULL REPRODUCTION PIPELINE")
    print(f"Repetitions per scenario cell: {n_reps} | Output: {os.path.abspath(out_dir)}")
    print("=" * 80)

    datasets = [
        ("Give Me Some Credit", "give_me_some_credit", "datasets/give_me_some_credit/cs-training.csv"),
        ("Taiwan Credit Default", "taiwan_credit_default", "datasets/taiwan_credit_default/UCI_Credit_Card.csv"),
        ("Adult Census Income", "adult_income", "datasets/adult_income/adult_income.csv"),
    ]

    all_recommendations = []
    all_summaries = []
    all_benchmarks = []
    t_start = time.time()

    for display_name, preset_key, path in datasets:
        print(f"\n---> Processing: {display_name} ({path})")
        config = get_preset_config(preset_key, csv_path=path, n_reps=n_reps)
        df = load_dataset(config)
        print(f"     Loaded {len(df):,} rows | Positive rate: {df[config.label_col].mean():.1%}")

        # Benchmark
        bench = run_benchmark(df, config, n_calls=25)
        bench["dataset"] = display_name
        all_benchmarks.append(bench)

        # Full Grid Evaluation
        records = run_full_grid(df, config)
        summary = summarize_grid(records)
        summary["dataset"] = display_name
        all_summaries.append(summary)

        rec = best_detector_per_shift_type(records)
        rec["dataset"] = display_name
        all_recommendations.append(rec)

        # Save individual dataset assets into its own directory
        per_dataset_dir = {
            "give_me_some_credit": "datasets/give_me_some_credit/report_assets",
            "taiwan_credit_default": "datasets/taiwan_credit_default/report_assets_taiwan",
            "adult_income": "datasets/adult_income/report_assets_adult",
        }.get(preset_key, f"datasets/{preset_key}/report_assets")
        os.makedirs(per_dataset_dir, exist_ok=True)

        summary.to_csv(f"{per_dataset_dir}/evaluation_summary.csv", index=False)
        bench.to_csv(f"{per_dataset_dir}/benchmark.csv", index=False)
        rec.to_csv(f"{per_dataset_dir}/recommendation.csv", index=False)

        failures = find_failure_modes(summary)
        failures.to_csv(f"{per_dataset_dir}/failure_modes.csv", index=False)

        fig_bar, comp = metric_comparison_bar(summary, config.shift_types,
                                             ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"])
        fig_bar.savefig(f"{per_dataset_dir}/metric_comparison.png", dpi=120)
        plt.close(fig_bar)
        comp.to_csv(f"{per_dataset_dir}/metric_comparison_table.csv")

        # Individual dataset heatmaps
        for onset in config.onsets:
            fig = sensitivity_heatmap(summary, config.shift_types, config.severities,
                                      ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"], onset)
            fig.savefig(f"{per_dataset_dir}/heatmap_{onset}.png", dpi=120)
            fig.savefig(f"{out_dir}/heatmap_{preset_key}_{onset}.png", dpi=120)
            plt.close(fig)

    # -----------------------------------------------------------------------
    # Consolidated Outputs
    # -----------------------------------------------------------------------
    combined_rec = pd.concat(all_recommendations, ignore_index=True)
    combined_rec.to_csv(f"{out_dir}/recommendations_all.csv", index=False)

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(f"{out_dir}/evaluation_summary_all.csv", index=False)

    combined_bench = pd.concat(all_benchmarks, ignore_index=True)
    combined_bench.to_csv(f"{out_dir}/benchmark_all.csv", index=False)

    # -----------------------------------------------------------------------
    # The 4 Trade-Off Curves
    # -----------------------------------------------------------------------
    print("\n---> Generating the 4 Mandatory Trade-Off Curves...")
    
    # 1. ROC Curve on primary dataset
    credit_cfg = get_preset_config("give_me_some_credit", csv_path="datasets/give_me_some_credit/cs-training.csv")
    credit_df = load_dataset(credit_cfg)
    roc_data = compute_roc_tradeoff_data(credit_df, credit_cfg)
    fig_roc = plot_roc_tradeoff(roc_data)
    fig_roc.savefig(f"{out_dir}/tradeoff_1_roc_curves.png", dpi=130)
    plt.close(fig_roc)

    # 2. Debounce window trade-off curve
    debounce_df = compute_debounce_curve(credit_df, credit_cfg)
    fig_debounce = plot_debounce_tradeoff(debounce_df)
    fig_debounce.savefig(f"{out_dir}/tradeoff_2_debounce_lag_vs_far.png", dpi=130)
    plt.close(fig_debounce)

    # 3. Lag vs Severity curve (on primary dataset)
    credit_summary = all_summaries[0]
    fig_lag = plot_lag_vs_severity(credit_summary)
    fig_lag.savefig(f"{out_dir}/tradeoff_3_lag_vs_severity.png", dpi=130)
    plt.close(fig_lag)

    # 4. Cost vs Capability Frontier
    fig_frontier = plot_cost_frontier(combined_bench[combined_bench["dataset"] == "Give Me Some Credit"])
    fig_frontier.savefig(f"{out_dir}/tradeoff_4_cost_capability_frontier.png", dpi=130)
    plt.close(fig_frontier)

    # Environment fingerprint
    sys_info = record_system_info()
    with open(f"{out_dir}/system_info.json", "w") as f:
        json.dump(sys_info, f, indent=2)

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"REPRODUCTION COMPLETE IN {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 80)
    print("\nCONSOLIDATED RECOMMENDATION TABLE ACROSS ALL 3 DATASETS:")
    pivot_rec = combined_rec.pivot(index="shift_type", columns="dataset", values="recommended_detector")
    print(pivot_rec.to_string())
    print(f"\nAll unified assets and trade-off curves written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
