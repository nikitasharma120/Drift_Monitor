"""viz.py - dashboard panel + evaluation-grid heatmaps + trade-off curves, shared by the report generator and the Streamlit app."""

import numpy as np
import matplotlib.pyplot as plt

from .evaluate import pooled_sensitivity

_COLORS = {"psi_agg": "tab:blue", "kl_agg": "tab:orange", "centroid_dist": "tab:green",
           "pred_prop_psi": "tab:purple", "concept_signal": "tab:brown", "ks_baseline": "black"}


def plot_monitor_dashboard(results_df, onset_batch, config, title="Drift Monitor", ax_set=None, thresholds=None):
    """5-panel dashboard: distribution-drift metrics, embedding drift,
    predicted-proportion drift, concept-drift performance signal, and
    alert markers. Pass ax_set=(ax0..ax4) to draw into an existing figure
    (e.g. Streamlit); otherwise creates and returns a new figure."""
    thresholds = config.alert_thresholds if thresholds is None else thresholds   # draw the thresholds actually used
    if ax_set is None:
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    else:
        axes = ax_set
        fig = axes[0].figure

    axes[0].plot(results_df["batch"], results_df["psi_agg"], label="PSI (max over features)", color="tab:blue")
    axes[0].plot(results_df["batch"], results_df["kl_agg"], label="KL divergence (max over features)", color="tab:orange")
    axes[0].axhline(thresholds["psi_agg"]["significant"], color="red", linestyle=":", linewidth=1, label="significant threshold")
    axes[0].set_ylabel("Distribution drift")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(results_df["batch"], results_df["centroid_dist"], color="tab:green", label="Embedding centroid distance")
    axes[1].axhline(thresholds["centroid_dist"]["significant"], color="red", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Embedding drift")
    axes[1].legend(loc="upper left", fontsize=8)

    if "pred_prop_psi" in results_df.columns:
        axes[2].plot(results_df["batch"], results_df["pred_prop_psi"], color="tab:purple", label="Predicted-probability PSI")
        axes[2].axhline(thresholds["pred_prop_psi"]["significant"], color="red", linestyle=":", linewidth=1)
        axes[2].set_ylabel("Predicted-\nproportion drift")
        axes[2].legend(loc="upper left", fontsize=8)

    if "concept_signal" in results_df.columns:
        axes[3].plot(results_df["batch"], results_df["concept_signal"], color="tab:brown", label="Concept-drift signal (Brier-based)")
        axes[3].axhline(thresholds["concept_signal"]["significant"], color="red", linestyle=":", linewidth=1)
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
    fig.suptitle(f"Sensitivity (detection rate) - {onset_filter} onset")
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


def plot_tradeoff_curves(curves, summary, operating_points, shift_types, title="Sensitivity vs. false-alarm trade-off"):
    """One panel per shift type; one line per detector (the named baseline in black, dashed).
    A filled marker is the detector's CALIBRATED operating point (what the grid reports).
    curves: tradeoff.build_tradeoff_tables()[0]; summary: ...[1] (for AUC in the legend);
    operating_points: DataFrame with shift_type, metric, false_alarm_rate, sensitivity."""
    fig, axes = plt.subplots(1, len(shift_types), figsize=(5.2 * len(shift_types), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, shift_types):
        for m, col in _COLORS.items():
            c = curves[(curves["shift_type"] == st) & (curves["metric"] == m) & (curves["severity"] == "all")]
            if c.empty:
                continue
            auc = summary[(summary["shift_type"] == st) & (summary["metric"] == m)]["score_auc"].iloc[0]
            ls = "--" if m == "ks_baseline" else "-"
            ax.plot(c["false_alarm_rate"], c["sensitivity"], ls, color=col, lw=1.6, label=f"{m} (AUC {auc:.2f})")
            op = operating_points[(operating_points["shift_type"] == st) & (operating_points["metric"] == m)]
            if len(op):
                ax.scatter(op["false_alarm_rate"], op["sensitivity"], color=col, s=55, zorder=5, edgecolor="white")
        ax.set_xlim(-0.01, 0.35); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False-alarm rate (clean pre-onset batches in alert)")
        ax.set_title(f"{st} shift")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("Sensitivity (runs detected)")
    fig.suptitle(title + "  - filled marker = calibrated operating point; AUC = score-level, threshold-free")
    fig.tight_layout()
    return fig


def plot_severity_curves(summary, shift_types, severities, metrics, title="Sensitivity vs. severity (calibrated thresholds)"):
    """Sensitivity (mean over onsets) as shift severity grows, one panel per shift type.
    Shaded band = 95% Wilson interval computed from the pooled counts (40 runs per point)."""
    fig, axes = plt.subplots(1, len(shift_types), figsize=(5.2 * len(shift_types), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(severities))
    for ax, st in zip(axes, shift_types):
        for m in metrics:
            sub = summary[(summary["shift_type"] == st) & (summary["metric"] == m)]
            g = pooled_sensitivity(sub, "severity").set_index("severity").reindex(severities)
            col = _COLORS.get(m)
            ax.plot(x, g["sensitivity"].values, "o--" if m == "ks_baseline" else "o-", color=col, label=m)
            ax.fill_between(x, g["sensitivity_ci_low"].values, g["sensitivity_ci_high"].values, color=col, alpha=0.10, linewidth=0)
        ax.set_xticks(x); ax.set_xticklabels(severities)
        ax.set_ylim(-0.02, 1.02); ax.set_title(f"{st} shift"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Sensitivity (band = 95% Wilson CI)")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_debounce_tradeoff(deb, shift_types, metrics, title="Effect of the confirmation rule (k consecutive batches)"):
    """Top row: false-alarm rate vs k. Bottom row: sensitivity vs k. Calibrated thresholds throughout."""
    fig, axes = plt.subplots(2, len(shift_types), figsize=(5.2 * len(shift_types), 7), sharex=True)
    for j, st in enumerate(shift_types):
        for m in metrics:
            d = deb[(deb["shift_type"] == st) & (deb["metric"] == m)].sort_values("k")
            ls = "--" if m == "ks_baseline" else "-"
            axes[0, j].plot(d["k"], 100 * d["false_alarm_rate"], "o" + ls, color=_COLORS.get(m), label=m)
            axes[1, j].plot(d["k"], d["sensitivity"], "o" + ls, color=_COLORS.get(m), label=m)
        axes[0, j].set_title(f"{st} shift"); axes[0, j].grid(alpha=0.3)
        axes[1, j].set_xlabel("k = consecutive batches required"); axes[1, j].grid(alpha=0.3)
        axes[1, j].set_xticks(sorted(deb["k"].unique())); axes[1, j].set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("False-alarm rate (%)"); axes[1, 0].set_ylabel("Sensitivity")
    axes[1, -1].legend(fontsize=7, loc="lower left")
    fig.suptitle(title); fig.tight_layout()
    return fig


def plot_lag_vs_severity(summary, shift_types, severities, metrics, title="Detection lag vs. severity (detected runs only)"):
    """Mean lag in batches after onset, by severity. Points only where the detector detected at least once."""
    fig, axes = plt.subplots(1, len(shift_types), figsize=(5.2 * len(shift_types), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(severities))
    for ax, st in zip(axes, shift_types):
        for m in metrics:
            sub = summary[(summary["shift_type"] == st) & (summary["metric"] == m)]
            y = sub.groupby("severity")["mean_lag_when_detected"].mean().reindex(severities)
            if y.notna().any():
                ax.plot(x, y.values, "o--" if m == "ks_baseline" else "o-", color=_COLORS.get(m), label=m)
        ax.set_xticks(x); ax.set_xticklabels(severities); ax.set_title(f"{st} shift"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean lag (batches after onset)")
    axes[-1].legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout()
    return fig


def plot_cost_frontier(points, title="Cost vs. capability"):
    """points: DataFrame(metric, ms_per_batch, mean_sensitivity). x = per-batch cost, y = mean sensitivity over all shift types."""
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for _, r in points.iterrows():
        ax.scatter(r["ms_per_batch"], r["mean_sensitivity"], s=70, color=_COLORS.get(r["metric"]), zorder=3)
        ax.annotate(r["metric"], (r["ms_per_batch"], r["mean_sensitivity"]), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("Cost per batch (ms; psi_agg and kl_agg share one cached computation)")
    ax.set_ylabel("Mean sensitivity over covariate, label and concept")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.set_title(title); fig.tight_layout()
    return fig
