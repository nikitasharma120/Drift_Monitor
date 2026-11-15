"""
dashboard_app.py — interactive dashboard for "A System That Notices"

Run with:
    streamlit run drift_monitor/dashboard_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from drift_monitor import DriftMonitorConfig, load_dataset
from drift_monitor.config import get_preset_config
from drift_monitor.injection import make_batches, inject
from drift_monitor.metrics import EmbeddingDriftDetector
from drift_monitor.reference_model import train_reference_model, PredictedProportionDetector
from drift_monitor.performance_monitor import PerformanceMonitor
from drift_monitor.monitor import run_monitor, detection_lag, false_alarm_rate
from drift_monitor.evaluate import run_full_grid, summarize_grid, find_failure_modes, best_detector_per_shift_type
from drift_monitor.viz import plot_monitor_dashboard, sensitivity_heatmap, metric_comparison_bar

st.set_page_config(page_title="A System That Notices", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar Dataset Configuration
# ---------------------------------------------------------------------------

st.sidebar.title("A System That Notices")
st.sidebar.caption("DS635 — Distribution Shift Monitor")

dataset_choice = st.sidebar.selectbox(
    "Choose Benchmark Dataset",
    [
        "Give Me Some Credit (cs-training.csv)",
        "Taiwan Credit Default (UCI_Credit_Card.csv)",
        "Adult Census Income (adult_income.csv)",
        "Synthetic Demo Data"
    ],
    index=0
)

# Resolve preset configuration and default paths
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dataset_choice.startswith("Give Me Some Credit"):
    preset_key = "give_me_some_credit"
    default_path = os.path.join(repo_root, "datasets", "give_me_some_credit", "cs-training.csv")
elif dataset_choice.startswith("Taiwan Credit"):
    preset_key = "taiwan_credit_default"
    default_path = os.path.join(repo_root, "datasets", "taiwan_credit_default", "UCI_Credit_Card.csv")
elif dataset_choice.startswith("Adult Census"):
    preset_key = "adult_income"
    default_path = os.path.join(repo_root, "datasets", "adult_income", "adult_income.csv")
else:
    preset_key = "synthetic"
    default_path = ""

csv_path = st.sidebar.text_input("CSV Path", value=default_path)
config = get_preset_config(preset_key, csv_path=csv_path)

st.sidebar.header("Shift Scenario")
shift_type = st.sidebar.selectbox("Shift Type", config.shift_types, index=0)
severity = st.sidebar.selectbox("Severity", config.severities, index=2)
onset = st.sidebar.selectbox("Onset Speed", config.onsets, index=0)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

tab1, tab2 = st.tabs(["Live Monitor", "Evaluation Grid"])

# ---------------------------------------------------------------------------
# Tab 1 — Live Monitor
# ---------------------------------------------------------------------------

with tab1:
    df = load_dataset(config)
    ref_batches = make_batches(df, config.n_batches, seed=0)
    reference_df = pd.concat(ref_batches[:config.onset_batch]).reset_index(drop=True)
    
    detector = EmbeddingDriftDetector(config.features, config.embedding_components).fit(reference_df)
    ref_model = train_reference_model(reference_df, config, random_state=42)
    pred_detector = PredictedProportionDetector(ref_model, config.features).fit(reference_df)
    perf_detector = PerformanceMonitor(ref_model, config.features, config.label_col, config.labeled_trickle_fraction, seed=42).fit(reference_df)

    batches = inject(shift_type, severity, onset, int(seed), df, config)
    results = run_monitor(batches, reference_df, detector, config, pred_detector=pred_detector, perf_detector=perf_detector)

    lag = detection_lag(results, config.onset_batch)
    far = false_alarm_rate(results, config.onset_batch)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detection Lag", f"{lag} batches" if lag is not None else "NOT DETECTED")
    c2.metric("Pre-onset False Alarm Rate", f"{far:.1%}")
    c3.metric("Shift Onset", f"batch {config.onset_batch}")
    c4.metric("Alerts Fired", int(results["any_alert"].sum()))

    fig = plot_monitor_dashboard(results, config.onset_batch, config,
                                  title=f"{shift_type.capitalize()} shift ({severity}, {onset} onset) — {dataset_choice.split()[0]}")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Raw per-batch results table"):
        st.dataframe(results, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2 — Evaluation Grid Explorer
# ---------------------------------------------------------------------------

with tab2:
    st.subheader(f"Evaluation Grid — {dataset_choice.split()[0]}")
    n_reps = st.slider("Repetitions per scenario cell", 1, 10, 5)
    
    if st.button("Run Evaluation Grid"):
        with st.spinner("Running grid across scenarios..."):
            config.n_reps = n_reps
            records = run_full_grid(df, config)
            summary = summarize_grid(records)
            st.session_state["summary"] = summary
            st.session_state["records"] = records

    if "summary" in st.session_state:
        summary = st.session_state["summary"]
        records = st.session_state.get("records", summary)

        st.markdown("**Recommended Detector per Shift Type (Pooled Wilson 95% CIs)**")
        st.dataframe(best_detector_per_shift_type(records), use_container_width=True)

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

        st.markdown("**Failure Modes** (Sensitivity < 0.5 or Lag > 5 batches)")
        st.dataframe(find_failure_modes(summary), use_container_width=True)
    else:
        st.info("Click **Run Evaluation Grid** to populate this tab.")
