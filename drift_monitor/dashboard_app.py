"""
dashboard_app.py - interactive dashboard for "A System That Notices"

Run with:
    streamlit run drift_monitor/dashboard_app.py

How to read the false-alarm numbers:
  * The monitored stream is DISJOINT from the reference window, so pre-onset batches are
    genuinely unseen data.
  * Thresholds are the per-dataset CALIBRATED ones (calibrated_thresholds.json, or calibrated
    on the fly).
  * The top card is ONE run, pre-onset batches only. The table beneath it gives the
    false-alarm rate per detector over every evaluation run (360 per dataset), so a single
    lucky run is not mistaken for the system's false-alarm rate.
  * All three datasets can be selected in the sidebar.
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from drift_monitor import load_dataset
from drift_monitor.evaluate import (calibrate_thresholds, score_grid, records_from_scores, summarize_grid,
                                    find_failure_modes, best_detector_per_shift_type, baseline_comparison)
from drift_monitor.injection import make_scenario, batch_size_for
from drift_monitor.monitor import run_monitor, detection_lag, false_alarm_rate, METRICS, ALL_METRICS
from drift_monitor.presets import PRESETS, assets_dir, get_preset
from drift_monitor.tradeoff import build_tradeoff_tables
from drift_monitor.viz import (plot_monitor_dashboard, sensitivity_heatmap, metric_comparison_bar,
                               plot_tradeoff_curves, plot_severity_curves)

st.set_page_config(page_title="A System That Notices", layout="wide")


@st.cache_data(show_spinner="Loading dataset ...")
def get_data(key: str):
    return load_dataset(get_preset(key))


@st.cache_data(show_spinner="Calibrating thresholds on clean runs ...")
def get_thresholds(key: str):
    path = os.path.join(assets_dir(key), "calibrated_thresholds.json")
    if os.path.exists(path):
        return json.load(open(path))["thresholds"], "loaded from calibrated_thresholds.json"
    config = get_preset(key)
    thr, _ = calibrate_thresholds(get_data(key), config)
    return thr, f"calibrated just now on {config.calibration_reps} clean runs"


def read_asset(key: str, name: str):
    p = os.path.join(assets_dir(key), name)
    return pd.read_csv(p) if os.path.exists(p) else None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("A System That Notices")
st.sidebar.caption("DS635 - Distribution Shift Monitor")
key = st.sidebar.selectbox("Dataset", list(PRESETS), format_func=lambda k: PRESETS[k]["title"])
config = get_preset(key)
st.sidebar.header("Scenario")
shift_type = st.sidebar.selectbox("Shift type", config.shift_types, index=0)
severity = st.sidebar.selectbox("Severity", config.severities, index=2)
onset = st.sidebar.selectbox("Onset speed", config.onsets, index=0)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

df = get_data(key)
thresholds, thr_note = get_thresholds(key)
B = batch_size_for(len(df), config)
st.sidebar.caption(f"{len(df):,} rows - batch size {B:,} - reference window {config.n_reference_batches * B:,} rows "
                   f"(disjoint from the stream)")

tab1, tab2, tab3 = st.tabs(["Live Monitor", "Trade-off curves", "Evaluation Grid"])

# ---------------------------------------------------------------------------
# Tab 1 - live monitor
# ---------------------------------------------------------------------------
with tab1:
    reference_df, batches = make_scenario(shift_type, severity, onset, int(seed), df, config)
    results = run_monitor(batches, reference_df, config, thresholds=thresholds, seed=int(seed))
    lag = detection_lag(results, config.onset_batch)
    far_run = false_alarm_rate(results, config.onset_batch)

    summary = read_asset(key, "evaluation_summary.csv")
    far_grid = None
    if summary is not None:
        far_grid = summary[summary["metric"].isin(METRICS)].groupby("metric")["mean_false_alarm_rate"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detection lag", f"{lag} batches" if lag is not None else "NOT DETECTED")
    c2.metric("False alarms - this run, pre-onset", f"{far_run:.1%}",
              help=f"Share of the {config.onset_batch} pre-onset batches of THIS ONE run in which any of the five detectors "
                   f"was in a confirmed alert state. One run of {config.onset_batch} batches cannot resolve rates below 10%.")
    c3.metric("Shift onset", f"batch {config.onset_batch}")
    c4.metric("Alerts fired", int(results["any_alert"].sum()))
    st.caption(f"Thresholds: {thr_note}. False-alarm rate = fraction of PRE-onset (clean) batches in a confirmed alert "
               f"state; confirmation needs {config.consecutive_batches_required} consecutive batches above threshold.")

    fa = pd.DataFrame({"detector": ALL_METRICS,
                       "this run (pre-onset)": [false_alarm_rate(results, config.onset_batch, f"{m}_alert") for m in ALL_METRICS]})
    if summary is not None:
        per = summary.groupby("metric")[["mean_false_alarm_rate", "runs_with_false_alarm"]].mean()
        fa["all evaluation runs (batch-level)"] = fa["detector"].map(per["mean_false_alarm_rate"])
        fa["runs with >=1 false alarm"] = fa["detector"].map(per["runs_with_false_alarm"])
    fa.loc[fa["detector"] == "ks_baseline", "detector"] = "ks_baseline (named baseline)"
    st.markdown("**False-alarm rate by detector**")
    st.dataframe(fa.style.format({c: "{:.1%}" for c in fa.columns if c != "detector"}), width="stretch", hide_index=True)

    fig = plot_monitor_dashboard(results, config.onset_batch, config, thresholds=thresholds,
                                 title=f"{PRESETS[key]['title']} - {shift_type} shift, {severity}, {onset} onset")
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Raw per-batch results"):
        st.dataframe(results, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2 - trade-off curves (from the saved raw scores)
# ---------------------------------------------------------------------------
with tab2:
    curves, tsum, op = (read_asset(key, n) for n in ("tradeoff_curves.csv", "tradeoff_summary.csv", "operating_points.csv"))
    if curves is None:
        st.info("Run `python run_all.py` (or `python -m drift_monitor.report " + key + "`) to generate the curves.")
    else:
        st.markdown("Each line re-applies the full 2-consecutive-batch alert rule at every threshold to the same runs. "
                    "The filled marker is the calibrated operating point reported everywhere else.")
        fig = plot_tradeoff_curves(curves, tsum, op, config.shift_types, title=f"{PRESETS[key]['title']}")
        st.pyplot(fig); plt.close(fig)
        st.markdown("**Score-level AUC and sensitivity at fixed false-alarm budgets**")
        st.dataframe(tsum, width="stretch", hide_index=True)
        if summary is not None:
            fig = plot_severity_curves(summary, config.shift_types, config.severities, ALL_METRICS)
            st.pyplot(fig); plt.close(fig)

# ---------------------------------------------------------------------------
# Tab 3 - evaluation grid
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Full evaluation grid")
    st.caption("Showing the committed results (20 reps per cell). The button re-runs a smaller grid live.")
    if "live_summary" not in st.session_state or st.session_state.get("live_key") != key:
        st.session_state["live_summary"], st.session_state["live_key"] = None, key
    n_reps = st.slider("Repetitions per grid cell (live re-run)", 1, 10, 3)
    if st.button("Re-run evaluation grid live"):
        with st.spinner("Running grid ..."):
            cfg = get_preset(key, n_reps=n_reps)
            raw = score_grid(df, cfg, verbose=False)
            st.session_state["live_summary"] = summarize_grid(records_from_scores(raw, cfg, thresholds))
    shown = st.session_state["live_summary"] if st.session_state["live_summary"] is not None else summary
    if shown is None:
        st.info("No committed results found - run `python run_all.py`, or use the live re-run button.")
    else:
        st.markdown("**Recommended detector per shift type** (five detectors; baseline compared below)")
        st.dataframe(best_detector_per_shift_type(shown), width="stretch", hide_index=True)
        st.markdown("**Five detectors vs. the named KS + Bonferroni baseline**")
        st.dataframe(baseline_comparison(shown), width="stretch", hide_index=True)
        col1, col2 = st.columns(2)
        for col, on in zip((col1, col2), config.onsets):
            with col:
                f = sensitivity_heatmap(shown, config.shift_types, config.severities, ALL_METRICS, on)
                st.pyplot(f); plt.close(f)
        f, _ = metric_comparison_bar(shown, config.shift_types, ALL_METRICS)
        st.pyplot(f); plt.close(f)
        st.markdown("**Failure modes** (sensitivity < 0.5 or mean lag > 5 batches)")
        st.dataframe(find_failure_modes(shown), width="stretch")
