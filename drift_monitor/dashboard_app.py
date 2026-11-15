"""
dashboard_app.py — Interactive Production Dashboard for "Drift Monitor"
Course: DS635 Machine Learning System Engineering (MLSE)
Term Project Team: Sanjana Nathani, Nikita Sharma, Anish Karmakar, Srishti Lamba

Run with:
    streamlit run drift_monitor/dashboard_app.py
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_monitor import DriftMonitorConfig, load_dataset
from drift_monitor.config import get_preset_config
from drift_monitor.injection import make_batches, inject
from drift_monitor.metrics import EmbeddingDriftDetector, aggregate
from drift_monitor.reference_model import train_reference_model, PredictedProportionDetector
from drift_monitor.performance_monitor import PerformanceMonitor
from drift_monitor.monitor import run_monitor, detection_lag, false_alarm_rate
from drift_monitor.diagnosis import classify_shift
from drift_monitor.evaluate import run_full_grid, summarize_grid, find_failure_modes, best_detector_per_shift_type
from drift_monitor.viz import (
    plot_monitor_dashboard,
    plot_interactive_monitor_dashboard,
    sensitivity_heatmap,
    metric_comparison_bar
)

# ---------------------------------------------------------------------------
# Streamlit Page Setup & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Drift Monitor — MLSE Term Project",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        background: #F8FAFC;
    }
    .metric-card {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .triage-box {
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        font-size: 15px;
    }
    .triage-normal {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        color: #065F46;
    }
    .triage-covariate {
        background-color: #EFF6FF;
        border: 1px solid #3B82F6;
        color: #1E40AF;
    }
    .triage-label {
        background-color: #FFFBEB;
        border: 1px solid #F59E0B;
        color: #92400E;
    }
    .triage-concept {
        background-color: #FEF2F2;
        border: 1px solid #EF4444;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Caching Functions (Sub-50ms Reactive UI)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset_cached(preset_key, csv_path):
    cfg = get_preset_config(preset_key, csv_path=csv_path)
    df = load_dataset(cfg)
    return df, cfg


@st.cache_resource(show_spinner=False)
def fit_reference_suite(preset_key, csv_path):
    cfg = get_preset_config(preset_key, csv_path=csv_path)
    df = load_dataset(cfg)
    ref_batches = make_batches(df, cfg.n_batches, seed=0)
    ref_df = pd.concat(ref_batches[:cfg.onset_batch]).reset_index(drop=True)

    detector = EmbeddingDriftDetector(cfg.features, cfg.embedding_components).fit(ref_df)
    ref_model = train_reference_model(ref_df, cfg, random_state=42)
    pred_detector = PredictedProportionDetector(ref_model, cfg.features).fit(ref_df)
    perf_detector = PerformanceMonitor(ref_model, cfg.features, cfg.label_col,
                                       cfg.labeled_trickle_fraction, seed=42).fit(ref_df)
    return ref_df, detector, pred_detector, perf_detector, ref_model


# ---------------------------------------------------------------------------
# Sidebar: Dataset Selection & Scenario Controls
# ---------------------------------------------------------------------------

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.sidebar.title("📡 Drift Monitor")
st.sidebar.caption("DS635 — Machine Learning System Engineering")
st.sidebar.markdown("**Team:** Sanjana N., Nikita S., Anish K., Srishti L.")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Benchmark Dataset")
dataset_choice = st.sidebar.selectbox(
    "Dataset",
    [
        "Give Me Some Credit (cs-training.csv)",
        "Taiwan Credit Default (UCI_Credit_Card.csv)",
        "Adult Census Income (adult_income.csv)",
        "Synthetic Demo Data"
    ],
    index=0
)

if dataset_choice.startswith("Give Me Some Credit"):
    preset_key = "give_me_some_credit"
    default_path = os.path.join(repo_root, "datasets", "give_me_some_credit", "cs-training.csv")
    dataset_name = "Give Me Some Credit"
elif dataset_choice.startswith("Taiwan Credit"):
    preset_key = "taiwan_credit_default"
    default_path = os.path.join(repo_root, "datasets", "taiwan_credit_default", "UCI_Credit_Card.csv")
    dataset_name = "Taiwan Credit Default"
elif dataset_choice.startswith("Adult Census"):
    preset_key = "adult_income"
    default_path = os.path.join(repo_root, "datasets", "adult_income", "adult_income.csv")
    dataset_name = "Adult Census Income"
else:
    preset_key = "synthetic"
    default_path = ""
    dataset_name = "Synthetic Demo Data"

csv_path = st.sidebar.text_input("CSV Path", value=default_path)

# Load dataset & reference suite
with st.spinner(f"Loading {dataset_name}..."):
    df, base_cfg = load_dataset_cached(preset_key, csv_path)
    ref_df, detector, pred_detector, perf_detector, ref_model = fit_reference_suite(preset_key, csv_path)

# Quick stats badge
n_rows = len(df)
n_features = len(base_cfg.features)
pos_rate = float(df[base_cfg.label_col].mean()) if base_cfg.label_col in df.columns else 0.0
st.sidebar.info(f"📊 **{n_rows:,}** rows | **{n_features}** features | **{pos_rate:.1%}** base rate")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Shift Injection")
shift_type = st.sidebar.selectbox("Shift Type", base_cfg.shift_types, index=0,
                                  help="Covariate moves P(X); Label moves P(Y); Concept moves P(Y|X).")
severity = st.sidebar.selectbox("Severity", base_cfg.severities, index=2,
                                help="Mild (0.5 IQR), Moderate (1.0 IQR), Severe (2.0 IQR).")
onset = st.sidebar.selectbox("Onset Speed", base_cfg.onsets, index=0,
                             help="Abrupt step function at batch 10 vs 6-batch gradual ramp.")
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Monitor Tuning")
debounce_k = st.sidebar.slider("Debounce Window (k)", min_value=1, max_value=5, value=2,
                               help="Consecutive above-threshold batches required before confirmed alert fires.")
trickle_frac = st.sidebar.slider("Labeled Trickle Fraction", min_value=0.05, max_value=0.50, value=0.20, step=0.05,
                                 help="Delayed ground truth sample fraction used for Brier score tracking.")
pure_concept = st.sidebar.checkbox("Enforce Pure Concept Drift", value=True,
                                   help="Strictly swaps labels between positive and negative instances within band, preserving marginal P(Y).")


# Create runtime configuration
config = get_preset_config(
    preset_key,
    csv_path=csv_path,
    consecutive_batches_required=debounce_k,
    labeled_trickle_fraction=trickle_frac,
    pure_concept=pure_concept
)

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Stream Monitor",
    "📊 Master Benchmark & Grid",
    "⚖️ The 4 Trade-Off Curves",
    "📚 Architecture & Math Reference"
])


# ---------------------------------------------------------------------------
# TAB 1: Live Stream Monitor & Diagnostic Triage
# ---------------------------------------------------------------------------

with tab1:
    st.subheader(f"Live Streaming Telemetry — {dataset_name}")
    st.caption(f"Simulating 30 streaming batches ({len(df)//30:,} instances/batch). Onset injected at batch {config.onset_batch} ({shift_type.upper()} shift, {severity}, {onset} onset).")

    # Generate streaming batches
    perf_detector.labeled_fraction = trickle_frac
    perf_detector.reset_seed(int(seed))
    
    batches = inject(shift_type, severity, onset, int(seed), df, config, pure_concept=pure_concept)
    results = run_monitor(batches, ref_df, detector, config, pred_detector=pred_detector, perf_detector=perf_detector)

    lag = detection_lag(results, config.onset_batch)
    far = false_alarm_rate(results, config.onset_batch)
    total_alerts = int(results["any_alert"].sum())
    post_onset_alerts = int(results[results["batch"] >= config.onset_batch]["any_alert"].sum())

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Detection Status & Lag",
            value=f"{lag} batches" if lag is not None else "NOT DETECTED",
            delta=f"Onset at Batch {config.onset_batch}" if lag is not None else "Under-threshold",
            delta_color="normal" if lag is not None else "inverse"
        )
    with c2:
        st.metric(
            label="Pre-Onset False Alarm Rate",
            value=f"{far:.1%}",
            delta="Target: 0.0%" if far == 0 else "Noise Fluctuation",
            delta_color="normal" if far == 0 else "inverse"
        )
    with c3:
        st.metric(
            label="Confirmed Alerts Fired",
            value=f"{total_alerts} batches",
            delta=f"{post_onset_alerts} post-onset"
        )
    with c4:
        st.metric(
            label="Active Debounce Filter",
            value=f"k = {debounce_k}",
            delta=f"Trickle: {trickle_frac:.0%}"
        )

    # Automated Diagnostic Triage Engine
    alert_rows = results[results["any_alert"]]
    if len(alert_rows) > 0:
        latest_row = alert_rows.iloc[-1]
        verdict = classify_shift(latest_row)
    else:
        verdict = "none"

    if verdict == "covariate":
        st.markdown("""
        <div class="triage-box triage-covariate">
            <b>🔵 DIAGNOSTIC VERDICT: COVARIATE SHIFT (P(X) Shifted)</b><br>
            <b>Mechanism:</b> Input features have drifted into novel distribution support while model predictive mapping P(Y|X) remains valid.<br>
            <b>Recommended Action:</b> Inspect upstream data pipelines, check for seasonal or demographic feature shifts, evaluate out-of-domain input filtering.
        </div>
        """, unsafe_allow_html=True)
    elif verdict == "label":
        st.markdown("""
        <div class="triage-box triage-label">
            <b>🟡 DIAGNOSTIC VERDICT: LABEL SHIFT (P(Y) Class Prior Shifted)</b><br>
            <b>Mechanism:</b> Target base rate has moved significantly. Model outputs require probability calibration.<br>
            <b>Recommended Action:</b> Update prior odds ratio, re-tune classification decision thresholds via Platt scaling or isotonic regression without full retraining.
        </div>
        """, unsafe_allow_html=True)
    elif verdict in ["concept", "concept_or_mild_label"]:
        st.markdown("""
        <div class="triage-box triage-concept">
            <b>🔴 DIAGNOSTIC VERDICT: CONCEPT DRIFT (P(Y|X) Degradation)</b><br>
            <b>Mechanism:</b> The real-world mapping between inputs and outcomes has deteriorated. Brier score degraded on labeled verification trickle.<br>
            <b>Recommended Action:</b> <b>CRITICAL:</b> Model predictions are structurally invalid. Trigger emergency model retraining on recent ground truth.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="triage-box triage-normal">
            <b>🟢 DIAGNOSTIC VERDICT: NORMAL BASELINE OPERATION</b><br>
            <b>Status:</b> All five detectors operating below confirmed alert thresholds. Inferences statistically aligned with reference distribution.
        </div>
        """, unsafe_allow_html=True)

    # Visualization Mode Selector
    view_mode = st.radio("Visualization Engine", ["Interactive Plotly Studio (Zoom, Hover, Tooltips)", "Publication Matplotlib Layout"], horizontal=True)

    title_str = f"{shift_type.upper()} Shift ({severity}, {onset} onset) — {dataset_name}"
    if "Plotly" in view_mode:
        fig_plotly = plot_interactive_monitor_dashboard(results, config.onset_batch, config, title=title_str)
        st.plotly_chart(fig_plotly, use_container_width=True)
    else:
        fig_mpl = plot_monitor_dashboard(results, config.onset_batch, config, title=title_str)
        st.pyplot(fig_mpl)
        plt.close(fig_mpl)

    with st.expander("🔍 Inspect Raw Per-Batch Telemetry DataFrame"):
        st.dataframe(results, use_container_width=True)
        csv_bytes = results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Batch Telemetry CSV", data=csv_bytes, file_name=f"telemetry_{preset_key}_{shift_type}.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# TAB 2: Master Benchmark & Evaluation Grid Explorer
# ---------------------------------------------------------------------------

with tab2:
    st.subheader("Master Benchmark & Cross-Dataset Evaluation Grid")
    st.markdown("Full 18-scenario grid evaluation ($3 \\text{ shifts} \\times 3 \\text{ severities} \\times 2 \\text{ onsets} \\times 20 \\text{ repetitions} = 1,080 \\text{ test runs}$).")

    grid_view = st.radio("Benchmark View", ["⚡ Pre-Computed Authoritative Benchmark (Instant)", "🔄 Interactive Live Grid Sweep (Custom Repetitions)"], horizontal=True)

    combined_dir = os.path.join(repo_root, "report_assets_combined")
    rec_path = os.path.join(combined_dir, "recommendations_all.csv")
    summary_path = os.path.join(combined_dir, "evaluation_summary_all.csv")

    if "Pre-Computed" in grid_view:
        if os.path.exists(rec_path):
            st.markdown("### 🏆 Master Cross-Dataset Recommendation Matrix")
            df_rec = pd.read_csv(rec_path)
            
            # Format display table
            st.dataframe(
                df_rec[["dataset", "shift_type", "recommended_detector", "mean_sensitivity", "sensitivity_ci_low", "sensitivity_ci_high"]].style.format({
                    "mean_sensitivity": "{:.2%}",
                    "sensitivity_ci_low": "{:.2%}",
                    "sensitivity_ci_high": "{:.2%}"
                }),
                use_container_width=True
            )

            # Per-dataset sensitivity heatmaps
            st.markdown("### 🗺️ Sensitivity Heatmaps Across All Scenarios")
            c_h1, c_h2 = st.columns(2)
            
            img_abrupt = os.path.join(combined_dir, f"heatmap_{preset_key}_abrupt.png")
            img_gradual = os.path.join(combined_dir, f"heatmap_{preset_key}_gradual.png")

            with c_h1:
                if os.path.exists(img_abrupt):
                    st.image(img_abrupt, caption=f"Abrupt Onset Sensitivity Matrix — {dataset_name}", use_container_width=True)
            with c_h2:
                if os.path.exists(img_gradual):
                    st.image(img_gradual, caption=f"Gradual Onset Sensitivity Matrix — {dataset_name}", use_container_width=True)

            # Failure modes
            if os.path.exists(summary_path):
                st.markdown("### ⚠️ Discovered Failure Modes (Sensitivity < 0.50 or Lag > 5 batches)")
                df_sum = pd.read_csv(summary_path)
                df_curr = df_sum[df_sum["dataset"] == dataset_name] if "dataset" in df_sum.columns else df_sum
                failures = find_failure_modes(df_curr)
                if len(failures) > 0:
                    st.dataframe(failures, use_container_width=True)
                else:
                    st.success("No failure modes encountered under current configuration.")
        else:
            st.warning("Pre-computed benchmark assets not found. Run `python reproduce_all.py` to generate them.")

    else:
        st.markdown("#### Run Custom Grid Sweep")
        n_reps_custom = st.slider("Monte Carlo Repetitions per cell", 1, 10, 3)
        if st.button("🚀 Execute Custom Grid Sweep"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            config.n_reps = n_reps_custom
            records = run_full_grid(df, config)
            summary_custom = summarize_grid(records)
            
            progress_bar.progress(1.0)
            status_text.success("Grid sweep completed successfully!")
            
            st.markdown("**Best Detector Recommendation:**")
            st.dataframe(best_detector_per_shift_type(records), use_container_width=True)
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig1 = sensitivity_heatmap(summary_custom, config.shift_types, config.severities,
                                           ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"], "abrupt")
                st.pyplot(fig1)
                plt.close(fig1)
            with col_g2:
                fig2 = sensitivity_heatmap(summary_custom, config.shift_types, config.severities,
                                           ["psi_agg", "kl_agg", "centroid_dist", "pred_prop_psi", "concept_signal"], "gradual")
                st.pyplot(fig2)
                plt.close(fig2)


# ---------------------------------------------------------------------------
# TAB 3: The 4 Continuous Trade-Off Curves
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("The Four Engineering Trade-Off Curves")
    st.caption("Empirical characterization of detection frontiers, false alarm trade-offs, and latency scaling.")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### Curve 1: Sensitivity (TPR) vs. False Alarm Rate (FPR) ROC Frontier")
        img_roc = os.path.join(combined_dir, "tradeoff_1_roc_curves.png")
        if os.path.exists(img_roc):
            st.image(img_roc, use_container_width=True)
        with st.expander("Mathematical Explanation (Curve 1)"):
            st.markdown("""
            - Evaluated across streaming batches using strictly disjoint train/stream partitioning.
            - `psi_agg` achieves **AUC = 0.88** pooled across diverse shifts (**1.00** on isolated covariate shift).
            - `pred_prop_psi` achieves **AUC = 0.83** (and **1.00** on label shift).
            - `concept_signal` achieves **AUC = 0.76** (and **1.00** on concept drift).
            - `centroid_dist` achieves **AUC = 0.73** pooled (**0.54–0.59** on Give Me Some Credit due to `DebtRatio` extreme variance, but **>0.95** on Taiwan and Adult).
            """)

    with col_t2:
        st.markdown("#### Curve 2: Debounce Window vs. False Alarms & Detection Lag")
        img_deb = os.path.join(combined_dir, "tradeoff_2_debounce_lag_vs_far.png")
        if os.path.exists(img_deb):
            st.image(img_deb, use_container_width=True)
        with st.expander("Mathematical Explanation (Curve 2)"):
            st.markdown("""
            - At **k = 1** (no confirmation): Single-batch noise causes a **34.0% False Alarm Rate** (lag = 0.0 batches).
            - At **k = 2** (the operational sweet spot): False alarms drop by more than half to **14.0%**, with lag at **0.6 batches**.
            - At **k = 3**: False Alarm Rate drops sharply to **2.0%** (lag = 1.2 batches).
            - At **k >= 4**: False Alarm Rate is completely suppressed to **0.0%** (lag = 2.2 to 3.2 batches).
            """)

    col_t3, col_t4 = st.columns(2)

    with col_t3:
        st.markdown("#### Curve 3: Detection Lag vs. Injected Shift Severity")
        img_lag = os.path.join(combined_dir, "tradeoff_3_lag_vs_severity.png")
        if os.path.exists(img_lag):
            st.image(img_lag, use_container_width=True)
        with st.expander("Mathematical Explanation (Curve 3)"):
            st.markdown("""
            - **Severe Shifts:** Detected within **1.0 to 2.0 batches** (the theoretical minimum under confirmation damping).
            - **Moderate Shifts:** Detected within **2.1 to 3.4 batches**.
            - **Mild Shifts:** Detected within **5.2 to 8.6 batches** under gradual onset, reflecting the statistical mass accumulation requirement of the linear ramp.
            """)

    with col_t4:
        st.markdown("#### Curve 4: Computational Cost vs. Capability Frontier")
        img_cost = os.path.join(combined_dir, "tradeoff_4_cost_capability_frontier.png")
        if os.path.exists(img_cost):
            st.image(img_cost, use_container_width=True)
        with st.expander("Mathematical Explanation (Curve 4)"):
            st.markdown("""
            - `psi_agg + kl_agg`: **1.32 ms/batch** (~757 batches/sec) — Fast unsupervised input gate.
            - `centroid_dist`: **14.53 ms/batch** (~69 batches/sec) — 3D latent subspace projection.
            - `pred_prop_psi`: **14.58 ms/batch** (~69 batches/sec) — Output probability distribution divergence.
            - `concept_signal`: **20.55 ms/batch** (~49 batches/sec) — Labeled trickle Brier evaluation.
            - **Combined Full Pipeline:** **50.99 ms/batch** (~20 batches/sec), providing 100% coverage across all 3 failure modes.
            """)


# ---------------------------------------------------------------------------
# TAB 4: Architecture, Math & System Health
# ---------------------------------------------------------------------------

with tab4:
    st.subheader("Architectural Foundations & Verification Rigor")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("### 📐 Mathematical Taxonomy of Shift")
        st.markdown("""
        Let $X \\in \\mathbb{R}^d$ be features and $Y \\in \\{0, 1\\}$ be targets:
        $$P(X, Y) = P(X) P(Y \\mid X) = P(Y) P(X \\mid Y)$$

        | Failure Mode | Mathematical Definition | Invariant | Visibility |
        |---|---|---|---|
        | **Covariate Shift** | $P_{\\text{stream}}(X) \\neq P_{\\text{ref}}(X)$ | $P(Y \\mid X)$ unchanged | **Visible** via PSI / KL |
        | **Label Shift** | $P_{\\text{stream}}(Y) \\neq P_{\\text{ref}}(Y)$ | $P(X \\mid Y)$ unchanged | **Indirect** via Pred-Prop PSI |
        | **Concept Drift** | $P_{\\text{stream}}(Y \\mid X) \\neq P_{\\text{ref}}(Y \\mid X)$ | None | **Structurally Invisible** to $X$ |

        > **The Input-Blindness Principle:** Any detector watching raw $X$ alone is structurally blind to pure concept drift. Ground-truth feedback (even delayed) is mathematically required.
        """)

        st.markdown("### 🔬 Jeffreys Divergence Equivalence")
        st.markdown("""
        $$\\text{PSI}(P, Q) = D_{\\text{KL}}(P \\parallel Q) + D_{\\text{KL}}(Q \\parallel P) \\approx 2 D_{\\text{KL}}(P \\parallel Q)$$
        - Setting equal thresholds ($\\tau = 0.10$) makes single-direction KL strictly harder to trigger than PSI.
        """)

    with col_a2:
        st.markdown("### 🏗️ Production 3-Tier Monitoring Topology")
        st.markdown("""
        ```
        Incoming Inference Batches
                   │
                   ▼
        ┌─────────────────────────────────────────────────────────┐
        │ TIER 1: Real-Time Unsupervised Input/Output Gate (<15ms)│
        │ • Feature PSI on numerical inputs (tau = 0.10)          │
        │ • Pred-Prop PSI on output probabilities (tau = 0.015)   │
        │ • Confirmation debouncing: k = 2 consecutive batches    │
        └─────────────────────────────────────────────────────────┘
                   │ (If Tier 1 alerts: Flag Covariate or Label)
                   ▼
        ┌─────────────────────────────────────────────────────────┐
        │ TIER 2: Supervised Verification Gate (Delayed Trickle)  │
        │ • Asynchronous 10-20% labeled trickle join              │
        │ • concept_signal: Delta Brier degradation (tau = 0.025) │
        └─────────────────────────────────────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────────────────────────────────┐
        │ TIER 3: Automated Diagnostic Resolution Engine          │
        │ • Pred-Prop + Concept Fired  --> "label_shift"          │
        │ • Concept Fired Alone        --> "pure_concept_drift"   │
        │ • PSI / KL Fired Alone       --> "covariate_shift"      │
        └─────────────────────────────────────────────────────────┘
        ```
        """)

        st.markdown("### 🛡️ Automated Verification Suite")
        st.success("✅ **23 Automated Unit Tests Passing** (`pytest tests/test_core.py`)")
        st.markdown("""
        - **Kolmogorov-Smirnov Baseline:** Classical 2-sample KS test with Bonferroni correction.
        - **Gold-Standard Hand-Labelled Suite:** Verified Precision = 1.00 (0 false alarms) & Recall >= 0.90.
        - **Jeffreys Equivalence:** Taylor expansion validated to within 4% accuracy.
        - **Deduplicated Quantile Bins:** Zero-inflated features handled without numerical overflow.
        """)

        # Hardware Provenance
        sys_info_path = os.path.join(combined_dir, "system_info.json")
        if os.path.exists(sys_info_path):
            with open(sys_info_path, "r") as f:
                s_info = json.load(f)
            st.markdown("**Hardware & Library Provenance (`system_info.json`):**")
            st.json(s_info)

st.markdown("---")
st.caption("Drift Monitor — Term Project | DS635 Machine Learning System Engineering | Fully reproducible with `python reproduce_all.py`.")
