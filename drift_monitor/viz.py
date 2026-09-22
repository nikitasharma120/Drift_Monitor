"""viz.py — dashboard panel + evaluation-grid heatmaps, shared by the report generator and the Streamlit app."""

import numpy as np
import matplotlib.pyplot as plt


def plot_monitor_dashboard(results_df, onset_batch, config, title="Drift Monitor", ax_set=None):
    """5-panel dashboard: distribution-drift metrics, embedding drift,
    predicted-proportion drift, concept-drift performance signal, and
    alert markers. Pass ax_set=(ax0..ax4) to draw into an existing figure
    (e.g. Streamlit); otherwise creates and returns a new figure."""
    if ax_set is None:
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    else:
        axes = ax_set
        fig = axes[0].figure

    axes[0].plot(results_df["batch"], results_df["psi_agg"], label="PSI (max over features)", color="tab:blue")
    axes[0].plot(results_df["batch"], results_df["kl_agg"], label="KL divergence (max over features)", color="tab:orange")
    axes[0].axhline(config.alert_thresholds["psi_agg"]["significant"], color="red", linestyle=":", linewidth=1, label="significant threshold")
    axes[0].set_ylabel("Distribution drift")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(results_df["batch"], results_df["centroid_dist"], color="tab:green", label="Embedding centroid distance")
    axes[1].axhline(config.alert_thresholds["centroid_dist"]["significant"], color="red", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Embedding drift")
    axes[1].legend(loc="upper left", fontsize=8)

    if "pred_prop_psi" in results_df.columns:
        axes[2].plot(results_df["batch"], results_df["pred_prop_psi"], color="tab:purple", label="Predicted-probability PSI")
        axes[2].axhline(config.alert_thresholds["pred_prop_psi"]["significant"], color="red", linestyle=":", linewidth=1)
        axes[2].set_ylabel("Predicted-\nproportion drift")
        axes[2].legend(loc="upper left", fontsize=8)

    if "concept_signal" in results_df.columns:
        axes[3].plot(results_df["batch"], results_df["concept_signal"], color="tab:brown", label="Concept-drift signal (Brier-based)")
        axes[3].axhline(config.alert_thresholds["concept_signal"]["significant"], color="red", linestyle=":", linewidth=1)
        axes[3].set_ylabel("Concept-drift\nsignal")
        axes[3].legend(loc="upper left", fontsize=8)

    alert_batches = results_df.loc[results_df["any_alert"], "batch"]
    axes[4].scatter(alert_batches, [1] * len(alert_batches), marker="v", color="red", s=60, label="Alert fired")
    axes[4].set_ylim(0.5, 1.5)
    axes[4].set_yticks([])
    axes[4].set_xlabel("Batch index")
    axes[4].legend(loc="upper left", fontsize=8)

    for ax in axes:
        ax.axvline(onset_batch, color="gray", linestyle="--", linewidth=1)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def sensitivity_heatmap(summary, shift_types, severities, metrics, onset_filter, ax_row=None):
    sub = summary[summary["onset"] == onset_filter]
    if ax_row is None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.5), sharey=True)
    else:
        axes = ax_row
        fig = axes[0].figure

    im = None
    for ax, metric in zip(axes, metrics):
        pivot = sub[sub["metric"] == metric].pivot(index="shift_type", columns="severity", values="sensitivity")
        pivot = pivot.reindex(index=shift_types, columns=severities)
        im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(severities))); ax.set_xticklabels(severities)
        ax.set_yticks(range(len(shift_types))); ax.set_yticklabels(shift_types)
        ax.set_title(metric)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.1f}" if not np.isnan(val) else "-", ha="center", va="center", fontsize=10)
    fig.suptitle(f"Sensitivity (detection rate) — {onset_filter} onset")
    if ax_row is None:
        fig.colorbar(im, ax=axes, shrink=0.8, label="sensitivity")
    return fig


def metric_comparison_bar(summary, shift_types, metrics):
    comp = summary.groupby(["shift_type", "metric"])["sensitivity"].mean().unstack()
    comp = comp.reindex(index=shift_types, columns=metrics)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comp.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean sensitivity (across severities + onsets)")
    ax.set_title("Which metric catches which shift type?")
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig, comp
