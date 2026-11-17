"""report.py - regenerate EVERY artifact in a dataset's report_assets*/ folder.

    python -m drift_monitor.report gmsc            # or: taiwan | adult
    python run_all.py                              # all three datasets + docs, one command

Everything written here (CSV, JSON, PNG) comes from one call, so the numbers
in the CSVs, the charts and (via build_docs.py) the README tables cannot
drift apart. Saved per dataset:

  calibrated_thresholds.json   per-dataset thresholds + the clean noise floor they came from
  evaluation_summary.csv       sensitivity (Wilson CI), lag, batch-level AND run-level false-alarm rates
  run_records.csv              one row per (run, metric)
  raw_scores.csv.gz            every raw detector score of every batch of every run (threshold-free)
  tradeoff_curves.csv/.png     sensitivity vs false-alarm curves (threshold swept), + tradeoff_summary.csv (AUC, sens@FAR)
  severity_curves.png          sensitivity vs severity
  baseline_comparison.csv      best of the five detectors vs the named KS+Bonferroni baseline
  diagnosis_confusion.csv      diagnosis.py's verdict vs. the injected truth
  recommendation.csv, failure_modes.csv, metric_comparison_table.csv, heatmaps, dashboard_*.png
  model_quality.json, benchmark.csv, system_info.json, run_manifest.json
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

from .benchmark import run_benchmark, record_system_info
from .data import load_dataset
from .diagnosis import classify_shift
from .evaluate import (calibrate_thresholds, score_grid, records_from_scores, summarize_grid,
                       find_failure_modes, best_detector_per_shift_type, baseline_comparison)
from .injection import make_scenario, make_reference_and_stream, batch_size_for
from .monitor import METRICS, ALL_METRICS, fit_detectors, run_monitor, apply_alerts, detection_lag, false_alarm_rate
from .presets import get_preset, assets_dir, PRESETS
from .reference_model import train_reference_model
from .tradeoff import build_tradeoff_tables, debounce_table
from .viz import (plot_monitor_dashboard, sensitivity_heatmap, metric_comparison_bar,
                  plot_tradeoff_curves, plot_severity_curves, plot_debounce_tradeoff,
                  plot_lag_vs_severity, plot_cost_frontier)


def _diagnosis_confusion(raw: pd.DataFrame, config, thresholds) -> pd.DataFrame:
    """For every post-onset batch where ANY of the five detectors is in an alert state,
    what does diagnosis.classify_shift say, per true injected shift type?"""
    rows = []
    for (st, sev, on, rep), g in raw.groupby(["shift_type", "severity", "onset", "rep"], sort=False):
        a = apply_alerts(g.sort_values("batch").reset_index(drop=True), config, thresholds)
        a = a[(a["batch"] >= config.onset_batch) & a["any_alert"]]
        for _, r in a.iterrows():
            rows.append({"true_shift_type": st, "diagnosis": classify_shift(r)})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    return pd.crosstab(d["true_shift_type"], d["diagnosis"], normalize="index").round(3)


def generate_report_assets(key: str, n_reps=None, calibration_reps=None, out_dir=None, quick=False, verbose=True):
    t_start = time.time()
    overrides = {}
    if quick:
        overrides.update(n_reps=3, calibration_reps=10)
    if n_reps is not None:
        overrides["n_reps"] = n_reps
    if calibration_reps is not None:
        overrides["calibration_reps"] = calibration_reps
    config = get_preset(key, **overrides)
    out = out_dir or assets_dir(key)
    os.makedirs(out, exist_ok=True)

    df = load_dataset(config)
    B = batch_size_for(len(df), config)
    print(f"[{key}] loaded {len(df)} rows, positive rate {df[config.label_col].mean():.4f}, "
          f"batch size {B}, reference window {config.n_reference_batches * B} rows, stream {config.n_batches} x {B}")

    # ---- 1. reference model quality on a clean, DISJOINT stream ----
    ref_df, clean_batches = make_reference_and_stream(df, config, seed=0)
    test_df = pd.concat(clean_batches).reset_index(drop=True)
    ref_model = train_reference_model(ref_df, config)
    test_probs = ref_model.predict_proba(test_df[config.features])[:, 1]
    ref_probs = ref_model.predict_proba(ref_df[config.features])[:, 1]
    model_quality = {
        "held_out_auc": round(float(roc_auc_score(test_df[config.label_col], test_probs)), 4),
        "held_out_accuracy": round(float(accuracy_score(test_df[config.label_col], (test_probs >= 0.5).astype(int))), 4),
        "held_out_mean_pred_prob": round(float(test_probs.mean()), 4),
        "held_out_true_base_rate": round(float(test_df[config.label_col].mean()), 4),
        "ref_mean_pred_prob": round(float(ref_probs.mean()), 4),
        "ref_true_base_rate": round(float(ref_df[config.label_col].mean()), 4),
        "held_out_rows": int(len(test_df)),
        "reference_rows": int(len(ref_df)),
    }
    json.dump(model_quality, open(f"{out}/model_quality.json", "w"), indent=2)

    # ---- 2. per-dataset threshold calibration ----
    print(f"[{key}] calibrating thresholds on {config.calibration_reps} clean runs ...")
    thresholds, calib_info = calibrate_thresholds(df, config, verbose=verbose)
    json.dump({"batch_size": B, "n_features": len(config.features),
               "calibration_reps": config.calibration_reps, "quantile": config.calibration_quantile,
               "margin": config.calibration_margin, "thresholds": thresholds,
               "noise_floor": calib_info.to_dict(orient="records")},
              open(f"{out}/calibrated_thresholds.json", "w"), indent=2)
    calib_info.to_csv(f"{out}/calibration_noise_floor.csv", index=False)

    # ---- 3. demo dashboards (severe, abrupt) ----
    demo_rows = []
    for shift_type in config.shift_types:
        ref, batches = make_scenario(shift_type, "severe", "abrupt", 42, df, config)
        results = run_monitor(batches, ref, config, thresholds=thresholds, seed=42)
        fig = plot_monitor_dashboard(results, config.onset_batch, config, thresholds=thresholds,
                                     title=f"{shift_type.capitalize()} shift (severe, abrupt) - calibrated thresholds")
        fig.savefig(f"{out}/dashboard_{shift_type}.png", dpi=130)
        plt.close(fig)
        row = {"shift_type": shift_type, "detection_lag_batches_any_alert": detection_lag(results, config.onset_batch),
               "pre_onset_false_alarm_rate_any_alert": false_alarm_rate(results, config.onset_batch)}
        for m in ALL_METRICS:
            row[f"{m}_pre_onset_far"] = false_alarm_rate(results, config.onset_batch, f"{m}_alert")
        demo_rows.append(row)
    pd.DataFrame(demo_rows).to_csv(f"{out}/demo_runs.csv", index=False)

    # ---- 4. full grid: score once, apply calibrated thresholds ----
    print(f"[{key}] scoring grid: {len(config.shift_types) * len(config.severities) * len(config.onsets)} scenarios x {config.n_reps} reps ...")
    raw = score_grid(df, config, verbose=verbose)
    raw.to_csv(f"{out}/raw_scores.csv.gz", index=False, float_format="%.6g")
    records = records_from_scores(raw, config, thresholds)
    records.to_csv(f"{out}/run_records.csv", index=False)
    summary = summarize_grid(records)
    summary.to_csv(f"{out}/evaluation_summary.csv", index=False)
    find_failure_modes(summary).to_csv(f"{out}/failure_modes.csv", index=False)
    best_detector_per_shift_type(summary).to_csv(f"{out}/recommendation.csv", index=False)
    baseline_comparison(summary).to_csv(f"{out}/baseline_comparison.csv", index=False)
    _diagnosis_confusion(raw, config, thresholds).to_csv(f"{out}/diagnosis_confusion.csv")

    for onset in config.onsets:
        fig = sensitivity_heatmap(summary, config.shift_types, config.severities, ALL_METRICS, onset)
        fig.savefig(f"{out}/sensitivity_heatmap_{onset}.png", dpi=130)
        plt.close(fig)
    fig, comp = metric_comparison_bar(summary, config.shift_types, ALL_METRICS)
    fig.savefig(f"{out}/metric_comparison.png", dpi=130)
    plt.close(fig)
    comp.to_csv(f"{out}/metric_comparison_table.csv")

    # ---- 5. trade-off curves ----
    curves, tsum = build_tradeoff_tables(raw, config)
    curves.to_csv(f"{out}/tradeoff_curves.csv", index=False, float_format="%.6g")
    tsum.to_csv(f"{out}/tradeoff_summary.csv", index=False)
    op = (records.groupby(["shift_type", "metric"])
          .agg(false_alarm_rate=("false_alarm", "mean"), sensitivity=("detected", "mean")).reset_index())
    op.to_csv(f"{out}/operating_points.csv", index=False)
    fig = plot_tradeoff_curves(curves, tsum, op, config.shift_types,
                               title=f"{PRESETS[key]['title']}: sensitivity vs. false-alarm trade-off")
    fig.savefig(f"{out}/tradeoff_curves.png", dpi=130); plt.close(fig)
    fig = plot_severity_curves(summary, config.shift_types, config.severities, ALL_METRICS,
                               title=f"{PRESETS[key]['title']}: sensitivity vs. severity")
    fig.savefig(f"{out}/severity_curves.png", dpi=130); plt.close(fig)

    # ---- 5b. confirmation-rule trade-off and lag vs severity ----
    deb = debounce_table(raw, config, thresholds)
    deb.to_csv(f"{out}/debounce_tradeoff.csv", index=False)
    fig = plot_debounce_tradeoff(deb, config.shift_types, ALL_METRICS,
                                 title=f"{PRESETS[key]['title']}: effect of the confirmation rule")
    fig.savefig(f"{out}/debounce_tradeoff.png", dpi=130); plt.close(fig)
    fig = plot_lag_vs_severity(summary, config.shift_types, config.severities, ALL_METRICS,
                               title=f"{PRESETS[key]['title']}: detection lag vs. severity")
    fig.savefig(f"{out}/lag_vs_severity.png", dpi=130); plt.close(fig)

    # ---- 6. cost benchmark + provenance ----
    bench = run_benchmark(df, config)
    bench.to_csv(f"{out}/benchmark.csv", index=False)
    redraw_cost_frontier(out)
    info = record_system_info()
    json.dump(info, open(f"{out}/system_info.json", "w"), indent=2)
    manifest = {"dataset": key, "rows": int(len(df)), "batch_size": B, "n_reps": config.n_reps,
                "n_scenarios": len(config.shift_types) * len(config.severities) * len(config.onsets),
                "n_runs": int(len(raw.groupby(['shift_type', 'severity', 'onset', 'rep']))),
                "calibration_reps": config.calibration_reps,
                "severity_covariate": config.severity_covariate, "severity_label": config.severity_label,
                "severity_concept": config.severity_concept, "target_feature": config.target_feature,
                "consecutive_batches_required": config.consecutive_batches_required,
                "runtime_seconds": round(time.time() - t_start, 1), "quick_mode": bool(quick)}
    json.dump(manifest, open(f"{out}/run_manifest.json", "w"), indent=2)
    print(f"[{key}] done in {time.time() - t_start:.0f}s -> {out}")
    return out


_COST_ROW = {"psi_agg": 0, "kl_agg": 0, "centroid_dist": 1, "pred_prop_psi": 2, "concept_signal": 3, "ks_baseline": 5}


def redraw_cost_frontier(out: str):
    """Cost-vs-capability scatter from benchmark.csv and evaluation_summary.csv."""
    bench = pd.read_csv(f"{out}/benchmark.csv")
    summ = pd.read_csv(f"{out}/evaluation_summary.csv")
    sens = summ.groupby("metric")["sensitivity"].mean()
    pts = pd.DataFrame([{"metric": m, "ms_per_batch": float(bench.iloc[i]["per_batch_ms"]), "mean_sensitivity": float(sens[m])}
                        for m, i in _COST_ROW.items()])
    pts.to_csv(f"{out}/cost_capability.csv", index=False)
    fig = plot_cost_frontier(pts, title="Cost vs. capability")
    fig.savefig(f"{out}/cost_capability.png", dpi=130); plt.close(fig)


def redraw_figures(key: str):
    """Redraw the summary charts from the saved CSVs (no re-running the grid)."""
    config = get_preset(key)
    out = assets_dir(key)
    summary = pd.read_csv(f"{out}/evaluation_summary.csv")
    fig = plot_severity_curves(summary, config.shift_types, config.severities, ALL_METRICS,
                               title=f"{PRESETS[key]['title']}: sensitivity vs. severity")
    fig.savefig(f"{out}/severity_curves.png", dpi=130); plt.close(fig)
    if os.path.exists(f"{out}/benchmark.csv"):
        redraw_cost_frontier(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(PRESETS))
    ap.add_argument("--n-reps", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="3 reps, 10 calibration runs: smoke test only")
    a = ap.parse_args()
    generate_report_assets(a.dataset, n_reps=a.n_reps, quick=a.quick)
