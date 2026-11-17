"""
tools/build_docs.py - writes every NUMBER in the docs from the committed CSV/JSON files.

    python tools/build_docs.py          (or: python run_all.py, which calls it after regenerating the assets)

Writes:
  REPORT.md                        full generated report (headline, thresholds, curves, baseline, diagnosis, validation, provenance)
  datasets/<d>/RESULTS.md          the per-dataset section of the report
  README.md / datasets/<d>/README.md   only the blocks between  <!-- BEGIN:x -->  and  <!-- END:x -->  markers

Reading ONLY files means a sentence in the docs can no longer disagree with a file in report_assets*/:
the sentence is produced from it. Hand-written prose in the READMEs makes no numeric claim.
"""

import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drift_monitor.presets import PRESETS, assets_dir, dataset_dir  # noqa: E402
from drift_monitor.evaluate import pooled_sensitivity  # noqa: E402

METRIC_ORDER = ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal", "ks_baseline"]
SHIFTS = ["covariate", "label", "concept"]
SEVS = ["mild", "moderate", "severe"]


# ----------------------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("-" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pct(x, d=1):
    return "-" if pd.isna(x) else f"{100 * x:.{d}f}%"


def load(key):
    a = assets_dir(key)
    rd = lambda n: pd.read_csv(os.path.join(a, n)) if os.path.exists(os.path.join(a, n)) else None
    js = lambda n: json.load(open(os.path.join(a, n))) if os.path.exists(os.path.join(a, n)) else None
    return {"key": key, "title": PRESETS[key]["title"], "assets": a, "assets_rel": f"datasets/{PRESETS[key]['folder']}/{PRESETS[key]['assets']}",
            "summary": rd("evaluation_summary.csv"), "records": rd("run_records.csv"), "reco": rd("recommendation.csv"),
            "base": rd("baseline_comparison.csv"), "tsum": rd("tradeoff_summary.csv"), "diag": rd("diagnosis_confusion.csv"),
            "fail": rd("failure_modes.csv"), "bench": rd("benchmark.csv"), "noise": rd("calibration_noise_floor.csv"),
            "demo": rd("demo_runs.csv"), "deb": rd("debounce_tradeoff.csv"), "cost": rd("cost_capability.csv"),
            "thr": js("calibrated_thresholds.json"), "quality": js("model_quality.json"),
            "sysinfo": js("system_info.json"), "manifest": js("run_manifest.json")}


def available(keys=None):
    out = []
    for k in (keys or PRESETS):
        d = load(k)
        if d["summary"] is not None and d["manifest"] is not None:
            out.append(d)
    return out


# ----------------------------------------------------------------------------- tables
def headline_table(datasets) -> str:
    rows = []
    for d in datasets:
        t = pooled_sensitivity(d["summary"], ["shift_type", "metric"]).rename(
            columns={"sensitivity": "s", "sensitivity_ci_low": "lo", "sensitivity_ci_high": "hi",
                     "mean_false_alarm_rate": "far", "runs_with_false_alarm": "runfa"})
        for st in SHIFTS:
            gi = t[t["shift_type"] == st].set_index("metric")
            five = gi.loc[METRIC_ORDER[:5]].sort_values("s", ascending=False, kind="stable").reset_index().iloc[0]   # ties -> METRIC_ORDER
            ks = gi.loc["ks_baseline"]
            rows.append({"Dataset": d["title"], "Shift": st, "Best of the five": five["metric"],
                         "Sensitivity [95% CI]": f"{five['s']:.2f} [{five['lo']:.2f}, {five['hi']:.2f}]",
                         "False alarms (batches)": pct(five["far"], 2), "Runs with a false alarm": pct(five["runfa"], 2),
                         "KS baseline sens. [95% CI]": f"{ks['s']:.2f} [{ks['lo']:.2f}, {ks['hi']:.2f}]",
                         "KS baseline false alarms": pct(ks["far"], 2)})
    return md_table(pd.DataFrame(rows))


def detector_grid(d) -> str:
    s = d["summary"]
    sens = s.groupby(["shift_type", "severity", "metric"])["sensitivity"].mean().unstack().reindex(columns=METRIC_ORDER)
    sens = sens.reindex(pd.MultiIndex.from_product([SHIFTS, SEVS])).reset_index()
    sens.columns = ["shift", "severity"] + METRIC_ORDER
    return md_table(sens)


def far_grid(d) -> str:
    s = d["summary"]
    far = s.groupby(["shift_type", "metric"]).agg(b=("mean_false_alarm_rate", "mean"), r=("runs_with_false_alarm", "mean")).reset_index()
    rows = []
    for st in SHIFTS:
        g = far[far["shift_type"] == st].set_index("metric")
        rows.append({"shift": st, **{m: f"{pct(g.loc[m, 'b'], 2)} / {pct(g.loc[m, 'r'], 2)}" for m in METRIC_ORDER}})
    return md_table(pd.DataFrame(rows))


def thresholds_table(d) -> str:
    n = d["noise"]
    t = pd.DataFrame({"detector": n["metric"], "default floor": n["floor_moderate"].map(lambda v: f"{v:g}"),
                      "clean median": n["clean_median"].map(lambda v: f"{v:.4f}"),
                      "clean 99th pct": n["clean_q99"].map(lambda v: f"{v:.4f}"),
                      "calibrated 'moderate'": n["calibrated_moderate"].map(lambda v: f"{v:.4f}"),
                      "calibrated 'significant'": n["calibrated_significant"].map(lambda v: f"{v:.4f}"),
                      "raised above floor?": n["raised_above_floor"].map(lambda v: "yes" if v else "no")})
    return md_table(t)


def tradeoff_table(d) -> str:
    t = d["tsum"].copy()
    t["metric"] = pd.Categorical(t["metric"], METRIC_ORDER, ordered=True)
    t["shift_type"] = pd.Categorical(t["shift_type"], SHIFTS, ordered=True)
    t = t.sort_values(["shift_type", "metric"])
    t = t.rename(columns={"shift_type": "shift", "score_auc": "score AUC", "sensitivity_at_far_1pct": "sens. @ 1% FAR",
                          "sensitivity_at_far_5pct": "sens. @ 5% FAR"})
    return md_table(t)


def baseline_table(d) -> str:
    b = d["base"].copy()
    b = b.rename(columns={"shift_type": "shift", "best_of_five": "best of the five", "best_of_five_sensitivity": "its sens.",
                          "best_of_five_false_alarm": "its false alarms", "ks_baseline_sensitivity": "KS baseline sens.",
                          "ks_baseline_false_alarm": "KS false alarms", "sensitivity_gain_vs_baseline": "gain vs. baseline"})
    for c in ("its false alarms", "KS false alarms"):
        b[c] = b[c].map(lambda v: pct(v, 2))
    return md_table(b)


def debounce_md(d) -> str:
    """For each shift type, the best of the five detectors (by sensitivity at k=2), at k = 1..4."""
    deb = d["deb"]
    rows = []
    for st in SHIFTS:
        g2 = deb[(deb["shift_type"] == st) & (deb["k"] == 2) & (deb["metric"].isin(METRIC_ORDER[:5]))]
        g2 = g2.set_index("metric").loc[METRIC_ORDER[:5]]
        best = g2["sensitivity"].idxmax()
        for _, r in deb[(deb["shift_type"] == st) & (deb["metric"] == best)].sort_values("k").iterrows():
            rows.append({"shift": st, "detector": best, "k": int(r["k"]), "sensitivity": f"{r['sensitivity']:.2f}",
                         "false alarms": pct(r["false_alarm_rate"], 2),
                         "mean lag (batches)": "-" if pd.isna(r["mean_lag"]) else f"{r['mean_lag']:.1f}"})
    return md_table(pd.DataFrame(rows))


def diag_table(d) -> str:
    if d["diag"] is None or d["diag"].empty:
        return "_no post-onset alerts_"
    t = d["diag"].copy()
    for c in t.columns[1:]:
        t[c] = t[c].map(lambda v: pct(v))
    return md_table(t)


def bench_table(d) -> str:
    b = d["bench"][["detector", "per_batch_ms", "one_time_fit_ms", "throughput_batches_per_sec", "approx_memory_kb"]]
    return md_table(b, "{:.2f}")


def quality_table(d) -> str:
    q = d["quality"]
    return md_table(pd.DataFrame([{"reference base rate": f"{q['ref_true_base_rate']:.4f}", "reference mean pred.": f"{q['ref_mean_pred_prob']:.4f}",
                                   "held-out base rate": f"{q['held_out_true_base_rate']:.4f}", "held-out mean pred.": f"{q['held_out_mean_pred_prob']:.4f}",
                                   "held-out accuracy": f"{q['held_out_accuracy']:.4f}", "held-out ROC-AUC": f"{q['held_out_auc']:.4f}"}]))


def findings(datasets) -> str:
    """Programmatic, so it cannot contradict the tables."""
    out = []
    for d in datasets:
        s = d["summary"]
        t = pooled_sensitivity(s, ["shift_type", "metric"]).rename(columns={"sensitivity": "s", "mean_false_alarm_rate": "far"})
        worst = t[t["metric"].isin(METRIC_ORDER[:5])].sort_values("far", ascending=False).iloc[0]
        total_runs = d["manifest"]["n_runs"]
        runfa = d["records"][d["records"]["metric"].isin(METRIC_ORDER[:5])].groupby(["shift_type", "severity", "onset", "rep"])["false_alarm_run"].any().mean()
        out.append(f"- **{d['title']}**: {total_runs} runs; the highest batch-level false-alarm rate of any of the five detectors in any shift family is "
                   f"**{pct(worst['far'], 2)}** (`{worst['metric']}`, {worst['shift_type']} runs); **{pct(runfa, 2)}** of runs had at least one pre-onset alert from at least one of the five detectors.")
        for st in SHIFTS:
            g = t[t["shift_type"] == st].set_index("metric")
            five = g.loc[METRIC_ORDER[:5]].sort_values("s", ascending=False, kind="stable")
            beats = five["s"].iloc[0] - g.loc["ks_baseline", "s"]
            verb = "beats" if beats > 0.005 else ("ties" if abs(beats) <= 0.005 else "is beaten by")
            line = (f"  - {st}: best of the five is `{five.index[0]}` ({five['s'].iloc[0]:.2f}); it {verb} the KS baseline "
                    f"({g.loc['ks_baseline', 's']:.2f}; difference {beats:+.2f}).")
            deb = d.get("deb")
            if deb is not None:
                m = five.index[0]
                r1 = deb[(deb["shift_type"] == st) & (deb["metric"] == m) & (deb["k"] == 1)].iloc[0]
                r2 = deb[(deb["shift_type"] == st) & (deb["metric"] == m) & (deb["k"] == 2)].iloc[0]
                line += (f" With k = 1 instead of k = 2 consecutive batches that detector's sensitivity goes from {r2['sensitivity']:.2f} "
                         f"to {r1['sensitivity']:.2f} and its false alarms from {pct(r2['false_alarm_rate'], 2)} to {pct(r1['false_alarm_rate'], 2)}.")
            out.append(line)
    return "\n".join(out)


# ----------------------------------------------------------------------------- sections
def dataset_section(d, level="##") -> str:
    m, q = d["manifest"], d["quality"]
    h = level
    parts = [f"{h} {d['title']}",
             f"{m['rows']:,} rows - batch size {m['batch_size']:,} - reference window {10 * m['batch_size']:,} rows, disjoint from the {30 * m['batch_size']:,}-row stream - "
             f"{m['n_scenarios']} scenarios x {m['n_reps']} reps = **{m['n_runs']} runs**, each with its own fresh split, reference and fitted detectors - "
             f"thresholds calibrated on {m['calibration_reps']} separate clean runs. Target feature: `{m['target_feature']}`.",
             f"{h}# Calibrated thresholds",
             "Each detector's noise floor is measured on clean runs whose seeds are disjoint from every evaluation seed; "
             "`moderate = max(default floor, 1.25 x 99th percentile of clean scores)`, `significant = max(floor, 2 x moderate)`. The KS baseline is never tuned.",
             thresholds_table(d),
             f"{h}# Sensitivity by shift type and severity (mean over onsets)",
             detector_grid(d),
             f"{h}# False alarms: batch-level / share of runs with at least one",
             "Batch-level = fraction of the 10 pre-onset (clean) batches in a confirmed alert state. Runs-with-a-false-alarm = fraction of runs in which any pre-onset batch alerted.",
             far_grid(d),
             f"{h}# Trade-off curves",
             f"![trade-off curves]({d['assets_rel']}/tradeoff_curves.png)",
             f"![severity curves]({d['assets_rel']}/severity_curves.png)",
             "Score AUC = threshold-free separation of pre-onset batches from fully-ramped post-onset batches. Sensitivity @ x% FAR = best sensitivity reachable with at most x% false alarms.",
             tradeoff_table(d),
             f"{h}# The confirmation rule (k consecutive batches), lag and cost",
             "An alert needs k consecutive batches above threshold. k = 2 is used everywhere else; this shows what k buys, for the best detector of each shift type, with the calibrated thresholds. Lag = batches after onset, detected runs only.",
             debounce_md(d),
             f"![confirmation rule]({d['assets_rel']}/debounce_tradeoff.png)",
             f"![lag vs severity]({d['assets_rel']}/lag_vs_severity.png)",
             f"![cost vs capability]({d['assets_rel']}/cost_capability.png)",
             f"{h}# Against the named baseline",
             "Baseline = per-feature two-sample Kolmogorov-Smirnov test, Bonferroni-corrected, alpha 0.01, same 2-consecutive-batch rule, same runs, no tuning.",
             baseline_table(d),
             f"{h}# What diagnosis.py says when something alerts",
             "Share of post-onset alerting batches by verdict, per true injected shift.",
             diag_table(d),
             f"{h}# Reference model",
             quality_table(d),
             f"{h}# Cost",
             bench_table(d),
             f"{h}# Failure modes",
             f"{len(d['fail'])} (shift, severity, onset, detector) cells have sensitivity below 0.5 or mean lag above 5 batches - see `{d['assets_rel']}/failure_modes.csv`.",
             f"{h}# Provenance",
             f"`{d['assets_rel']}/system_info.json`: " + ", ".join(f"{k}={v}" for k, v in d["sysinfo"].items() if k != "timestamp_utc") + f", generated {d['sysinfo']['timestamp_utc'][:19]}Z. Run time {m['runtime_seconds']:.0f}s."]
    return "\n\n".join(parts)


def validation_section() -> str:
    p = os.path.join(ROOT, "validation", "hand_labelled_check.csv")
    if not os.path.exists(p):
        return "_run `python -m validation.hand_labelled_check`_"
    d = pd.read_csv(p)
    s = d[d["scored"]].copy()
    s["agree"] = s["agree"].astype(bool); s["alert_ok"] = s["alert_ok"].astype(bool)
    t = s.groupby("stream").agg(batches_scored=("agree", "size"), alert_correct=("alert_ok", "mean"), diagnosis_in_accepted_set=("agree", "mean")).reset_index()
    for c in ("alert_correct", "diagnosis_in_accepted_set"):
        t[c] = t[c].map(pct)
    return md_table(t)


# ----------------------------------------------------------------------------- marker patching
def patch_markers(path, blocks: dict):
    if not os.path.exists(path):
        return
    s = open(path, encoding="utf-8").read()
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
        if pat.search(s):
            s = pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", s)
    open(path, "w", encoding="utf-8", newline="\n").write(s)


def build_all():
    ds = available()
    if not ds:
        print("no assets found - run python run_all.py first")
        return
    from drift_monitor.report import redraw_figures
    for d in ds:
        redraw_figures(d["key"])
    head, find = headline_table(ds), findings(ds)
    for d in ds:
        body = dataset_section(d, "##")
        open(os.path.join(dataset_dir(d["key"]), "RESULTS.md"), "w", encoding="utf-8", newline="\n").write(
            f"# {d['title']} - generated results\n\n_Generated by `tools/build_docs.py` from `{d['assets_rel']}/`. Do not edit by hand._\n\n"
            + body.split("\n\n", 1)[1].replace("![trade-off curves](datasets", "![trade-off curves](../../datasets").replace("![severity curves](datasets", "![severity curves](../../datasets").replace("](datasets/", "](../../datasets/") + "\n")
        patch_markers(os.path.join(dataset_dir(d["key"]), "README.md"),
                      {"dataset_results": dataset_section(d, "###").replace("](datasets/", "](../../datasets/")})
    sections = "\n\n".join(dataset_section(d, "##") for d in ds)
    tpl_path = os.path.join(ROOT, "tools", "report_template.md")
    tpl = open(tpl_path, encoding="utf-8").read() if os.path.exists(tpl_path) else "{{headline}}\n\n{{findings}}\n\n{{sections}}\n\n{{validation}}\n"
    report = (tpl.replace("{{headline}}", head).replace("{{findings}}", find)
                 .replace("{{sections}}", sections).replace("{{validation}}", validation_section()))
    open(os.path.join(ROOT, "REPORT.md"), "w", encoding="utf-8", newline="\n").write(report)
    patch_markers(os.path.join(ROOT, "README.md"), {"headline": head, "findings": find, "validation": validation_section()})
    print(f"docs rebuilt from {len(ds)} dataset(s)")


if __name__ == "__main__":
    build_all()
