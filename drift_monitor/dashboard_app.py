"""
dashboard_app.py — polished interactive dashboard for "A System That Notices"

Run with:
    streamlit run drift_monitor/dashboard_app.py

Lets you pick a shift scenario from the sidebar, watch the monitor detect
(or fail to detect) it batch by batch, and browse the full evaluation grid.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from drift_monitor import DriftMonitorConfig, load_dataset
from drift_monitor.injection import make_batches, inject
from drift_monitor.metrics import EmbeddingDriftDetector
from drift_monitor.reference_model import train_reference_model, PredictedProportionDetector
from drift_monitor.performance_monitor import PerformanceMonitor
from drift_monitor.monitor import run_monitor, detection_lag, false_alarm_rate
from drift_monitor.evaluate import run_full_grid, summarize_grid, find_failure_modes, best_detector_per_shift_type
from drift_monitor.viz import plot_monitor_dashboard, sensitivity_heatmap, metric_comparison_bar

st.set_page_config(page_title="A System That Notices", layout="wide")

# ---------------------------------------------------------------------------
# Cached setup
# ---------------------------------------------------------------------------

@st.cache_resource
def get_config():
    return DriftMonitorConfig()


@st.cache_data
def get_data(csv_path: str):
    config = get_config()
    config.csv_path = csv_path
    return load_dataset(config)


@st.cache_resource
def get_reference_and_detector(csv_path: str):
    config = get_config()
    df = get_data(csv_path)
    ref_batches = make_batches(df, config.n_batches, seed=0)
    reference_df = pd.concat(ref_batches[:config.onset_batch]).reset_index(drop=True)
    detector = EmbeddingDriftDetector(config.features, config.embedding_components).fit(reference_df)
    ref_model = train_reference_model(reference_df, config)
    pred_detector = PredictedProportionDetector(ref_model, config.features).fit(reference_df)
    perf_detector = PerformanceMonitor(ref_model, config.features, config.label_col, config.labeled_trickle_fraction).fit(reference_df)
    return reference_df, detector, pred_detector, perf_detector


@st.cache_data
def get_grid_summary(csv_path: str, n_reps: int):
    config = get_config()
    config.n_reps = n_reps
    df = get_data(csv_path)
    records = run_full_grid(df, config)
    return summarize_grid(records)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("A System That Notices")
st.sidebar.caption("DS635 — Distribution Shift Monitor")

csv_path = st.sidebar.text_input("CSV path (blank = synthetic demo data)", value="")
config = get_config()
config.csv_path = csv_path

st.sidebar.header("Scenario")
shift_type = st.sidebar.selectbox("Shift type", config.shift_types, index=0)
severity = st.sidebar.selectbox("Severity", config.severities, index=2)
onset = st.sidebar.selectbox("Onset speed", config.onsets, index=0)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

tab1, tab2 = st.tabs(["Live Monitor", "Evaluation Grid"])

# ---------------------------------------------------------------------------
# Tab 1 — live monitor for the selected scenario
# ---------------------------------------------------------------------------

with tab1:
    df = get_data(csv_path)
    reference_df, detector, pred_detector, perf_detector = get_reference_and_detector(csv_path)
    batches = inject(shift_type, severity, onset, int(seed), df, config)
    results = run_monitor(batches, reference_df, detector, config, pred_detector=pred_detector, perf_detector=perf_detector)

    lag = detection_lag(results, config.onset_batch)
    far = false_alarm_rate(results, config.onset_batch)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detection lag", f"{lag} batches" if lag is not None else "NOT DETECTED")
    c2.metric("False alarm rate (pre-onset)", f"{far:.1%}")
    c3.metric("Shift onset", f"batch {config.onset_batch}")
    c4.metric("Alerts fired", int(results["any_alert"].sum()))

    fig = plot_monitor_dashboard(results, config.onset_batch, config,
                                  title=f"{shift_type} shift — {severity}, {onset} onset")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Raw per-batch results"):
        st.dataframe(results, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2 — evaluation grid explorer
# ---------------------------------------------------------------------------

with tab2:
    st.subheader("Full evaluation grid")
    n_reps = st.slider("Repetitions per grid cell", 1, 10, 5)
    if st.button("Run evaluation grid"):
        with st.spinner("Running grid — a few dozen seconds..."):
            summary = get_grid_summary(csv_path, n_reps)
        st.session_state["summary"] = summary

    if "summary" in st.session_state:
        summary = st.session_state["summary"]

        st.markdown("**Recommended detector per shift type**")
        st.dataframe(best_detector_per_shift_type(summary), width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            fig1 = sensitivity_heatmap(summary, config.shift_types, config.severities,
                                        ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"], "abrupt")
            st.pyplot(fig1)
            plt.close(fig1)
        with col2:
            fig2 = sensitivity_heatmap(summary, config.shift_types, config.severities,
                                        ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"], "gradual")
            st.pyplot(fig2)
            plt.close(fig2)

        fig3, _ = metric_comparison_bar(summary, config.shift_types, ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"])
        st.pyplot(fig3)
        plt.close(fig3)

        st.markdown("**Failure modes** (sensitivity < 0.5 or mean lag > 5 batches)")
        st.dataframe(find_failure_modes(summary), width="stretch")
    else:
        st.info("Click **Run evaluation grid** to populate this tab.")
