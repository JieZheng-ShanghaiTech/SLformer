from __future__ import annotations

from pathlib import Path

from itertools import combinations
import textwrap

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra


PAPER_BLUE = "#0072B2"
PAPER_ORANGE = "#D55E00"
PAPER_SKY = "#56B4E9"
PAPER_GREEN = "#009E73"
PAPER_LIGHT_GRAY = "#D9D9D9"
PAPER_GRAY = "#666666"
MODEL_COLORS = {
    "SLformer": PAPER_BLUE,
    "Geneformer": PAPER_ORANGE,
    "Geneformer-Probing": PAPER_ORANGE,
    "Gene2Vec": PAPER_GREEN,
}
MODEL_LABELS = {
    "SLformer": "SLformer",
    "Geneformer": "Geneformer",
    "Geneformer-Probing": "Geneformer-Probing",
    "Gene2Vec": "Gene2Vec",
}
SCOPE_LABELS = {"mix-cancer": "mix-cancer", "cancer-specific": "cancer-specific"}


def load_metrics(metrics_csv: str | Path) -> pd.DataFrame:
    """Load epoch-level SAE training metrics.

    Expected columns include: epoch, train_loss, val_loss.
    """
    p = Path(metrics_csv).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Missing metrics.csv: {p}")
    df = pd.read_csv(p)
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    # Coerce all other columns to numeric when possible.
    numeric_cols = [c for c in df.columns if c != "epoch"]
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def plot_training_summary(
    metrics: pd.DataFrame,
    *,
    title: str = "SAE training: loss + latent metrics",
    show_val: bool = True,
) -> None:
    """Plot detailed loss components and latent metrics vs epoch.

    Left: total loss and its main components (recon, gate_term, orth_term).
    Right: topK-aware diagnostics (dead_frac, active-only activation, raw regularizers).
    """
    df = metrics.sort_values("epoch") if "epoch" in metrics.columns else metrics
    if "epoch" not in df.columns:
        raise ValueError("metrics must contain an 'epoch' column")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax0, ax1, ax2, ax3 = axes.ravel()

    loss_series = [
        ("train_loss", "train_total"),
        ("train_recon", "train_recon"),
    ]
    for col, label in loss_series:
        if col in df.columns:
            ax0.plot(df["epoch"], df[col], label=label)

    if show_val:
        val_loss_series = [
            ("val_loss", "val_total"),
            ("val_recon", "val_recon"),
        ]
        for col, label in val_loss_series:
            if col in df.columns and df[col].notna().any():
                ax0.plot(df["epoch"], df[col], linestyle="--", alpha=0.9, label=label)

    reg_series = [
        ("train_gate_term", "train_gate_term (weighted)"),
        ("train_l1_term", "train_l1_term (weighted)"),
        ("train_orth_term", "train_orth_term (weighted)"),
    ]
    for col, label in reg_series:
        if col in df.columns and df[col].notna().any():
            ax0.plot(df["epoch"], df[col], linestyle=":", linewidth=2.0, alpha=0.95, label=label)

    if show_val:
        reg_val_series = [
            ("val_gate_term", "val_gate_term (weighted)"),
            ("val_l1_term", "val_l1_term (weighted)"),
            ("val_orth_term", "val_orth_term (weighted)"),
        ]
        for col, label in reg_val_series:
            if col in df.columns and df[col].notna().any():
                ax0.plot(df["epoch"], df[col], linestyle="--", linewidth=1.4, alpha=0.85, label=label)

    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax0.set_title("Loss components")
    ax0.legend(fontsize=8, loc="best")

    def _is_nonconstant(series: pd.Series, tol: float = 1e-8) -> bool:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return False
        return (float(s.max()) - float(s.min())) > tol

    for col, label in [("train_gate_raw", "train_gate_raw"), ("train_orth_raw", "train_orth_raw")]:
        if col in df.columns and df[col].notna().any():
            y = df[col].clip(lower=1e-12)
            ax1.plot(df["epoch"], y, label=label)
    if show_val:
        for col, label in [("val_gate_raw", "val_gate_raw"), ("val_orth_raw", "val_orth_raw")]:
            if col in df.columns and df[col].notna().any():
                y = df[col].clip(lower=1e-12)
                ax1.plot(df["epoch"], y, linestyle="--", alpha=0.9, label=label)
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("raw value (log)")
    ax1.set_title("Regularizers (raw)")
    ax1.legend(fontsize=8, loc="best")

    for col, label in [("train_dead_frac", "train_dead_frac")]:
        if col in df.columns and df[col].notna().any():
            ax2.plot(df["epoch"], df[col], label=label)
    if show_val:
        for col, label in [("val_dead_frac", "val_dead_frac")]:
            if col in df.columns and df[col].notna().any():
                ax2.plot(df["epoch"], df[col], linestyle="--", alpha=0.9, label=label)
    for col, label in [("train_active_frac", "train_active_frac")]:
        if col in df.columns and _is_nonconstant(df[col]):
            ax2.plot(df["epoch"], df[col], label=label)
    if show_val:
        for col, label in [("val_active_frac", "val_active_frac")]:
            if col in df.columns and _is_nonconstant(df[col]):
                ax2.plot(df["epoch"], df[col], linestyle="--", alpha=0.9, label=label)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("fraction")
    ax2.set_title("Latent support dynamics")
    ax2.legend(fontsize=8, loc="best")

    for col, label in [("train_mean_act_active", "train_mean_abs_act_active"), ("train_mean_act_all", "train_mean_abs_act_all")]:
        if col in df.columns and df[col].notna().any():
            ax3.plot(df["epoch"], df[col], label=label)
    if show_val:
        for col, label in [("val_mean_act_active", "val_mean_abs_act_active"), ("val_mean_act_all", "val_mean_abs_act_all")]:
            if col in df.columns and df[col].notna().any():
                ax3.plot(df["epoch"], df[col], linestyle="--", alpha=0.9, label=label)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("mean abs act")
    ax3.set_title("Latent: mean abs activation")
    ax3.legend(fontsize=8, loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def plot_loss_curves(
    metrics: pd.DataFrame,
    *,
    title: str = "SAE training: loss curves",
) -> None:
    """Plot train/val total loss vs epoch (reference-style)."""
    df = metrics.sort_values("epoch") if "epoch" in metrics.columns else metrics

    plt.figure(figsize=(8.2, 4.2))
    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def manifold_ordering_delta(summary: pd.DataFrame, *, baseline_models: list[str] | tuple[str, ...]) -> pd.DataFrame:
    delta = summary.pivot_table(
        index=["scenario", "k"],
        columns="model",
        values=["mean_rho_manifold_score", "mean_rho_manifold_label"],
    ).reset_index()
    delta.columns = ["_".join([str(x) for x in col if x]) for col in delta.columns.to_flat_index()]
    rows = []
    for baseline_model in baseline_models:
        baseline_delta = delta[["scenario", "k"]].copy()
        baseline_delta["baseline_model"] = baseline_model
        baseline_delta["delta_score"] = (
            delta["mean_rho_manifold_score_SLformer"] - delta[f"mean_rho_manifold_score_{baseline_model}"]
        )
        baseline_delta["delta_label"] = (
            delta["mean_rho_manifold_label_SLformer"] - delta[f"mean_rho_manifold_label_{baseline_model}"]
        )
        rows.append(baseline_delta)
    return pd.concat(rows, ignore_index=True)


def plot_multi_anchor_manifold_ordering(
    summary: pd.DataFrame,
    *,
    anchor_count: int,
    ks: list[int] | tuple[int, ...],
    distance_label: str = "$d_{\\mathcal{G}}$",
    rho_label: str = "$\\rho_{\\mathcal{G}}$",
    title: str | None = None,
    out_path: str | Path | None = None,
) -> plt.Figure:
    plot_summary = summary.copy()
    plot_summary["model_label"] = plot_summary["model"].map(MODEL_LABELS)
    plot_summary["scope_label"] = plot_summary["scenario"].map(SCOPE_LABELS)
    model_names = list(plot_summary["model"].drop_duplicates())
    baseline_models = [model for model in model_names if model != "SLformer"]
    ordered_models = ["SLformer"] + baseline_models

    previous_rcparams = plt.rcParams.copy()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 260,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(3, 6), sharex=True, sharey=True, constrained_layout=False)

    for ax, scenario, panel_id in zip(axes, ["cancer-specific", "mix-cancer"], ["a", "b"]):
        scenario_data = plot_summary[plot_summary["scenario"] == scenario]

        for model_name in ordered_models:
            group = scenario_data[scenario_data["model"] == model_name]
            ax.plot(
                group["k"],
                group["mean_rho_manifold_score"],
                color=MODEL_COLORS[model_name],
                marker="o",
                linewidth=1.25,
                markersize=3.2,
                label=f"{MODEL_LABELS[model_name]}: score",
            )
            ax.plot(
                group["k"],
                group["mean_rho_manifold_label"],
                color=MODEL_COLORS[model_name],
                marker="s",
                linestyle="--",
                linewidth=0.95,
                markersize=3.0,
                alpha=0.78,
                label=f"{MODEL_LABELS[model_name]}: label",
            )
        ax.axhline(0, color=PAPER_LIGHT_GRAY, linewidth=0.8, zorder=0)
        ax.set_ylim(-0.12, 0.55)
        ax.set_title(panel_id, loc="left", fontweight="bold")
        ax.set_title(f"{SCOPE_LABELS[scenario]} ordering", loc="center")
        ax.set_xticks(ks)
        ax.tick_params(axis="x", length=3)
        ax.tick_params(axis="y", length=3)
        ax.grid(axis="y", color=PAPER_LIGHT_GRAY, linewidth=0.55, alpha=0.65)
    axes[-1].set_xlabel(f"K nearest neighbours by {distance_label}")
    for ax in axes:
        ax.set_ylabel(f"mean {rho_label}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.27, hspace=0.28)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.8,
    )
    title_text = title or f"Multi-anchor Mapper-nerve manifold-distance SL ordering (n={anchor_count} anchors)"
    fig.suptitle(
        "\n".join(textwrap.wrap(title_text, width=48)),
        x=0.16,
        y=0.965,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
    )
    if out_path is not None:
        fig.savefig(Path(out_path))
    plt.show()
    plt.rcParams.update(previous_rcparams)
    return fig


def plot_anchor_group_manifold_ordering(
    anchor_group_summary: pd.DataFrame,
    *,
    ks: list[int] | tuple[int, ...],
    high_label: str = "high-score positive anchors",
    low_label: str = "low-score negative anchors",
) -> None:
    plot_summary = anchor_group_summary.copy()
    plot_summary["model_label"] = plot_summary["model"].map(MODEL_LABELS)
    plot_summary["scope_label"] = plot_summary["scenario"].map(SCOPE_LABELS)
    high_n = int(plot_summary[plot_summary["anchor_group"] == high_label]["n_anchors"].iloc[0])
    low_n = int(plot_summary[plot_summary["anchor_group"] == low_label]["n_anchors"].iloc[0])

    rho_vals = plot_summary[["mean_rho_manifold_score", "mean_rho_manifold_label"]].to_numpy()
    y_lo = min(-0.14, np.floor(rho_vals.min() / 0.05) * 0.05 - 0.06)
    y_hi = max(0.55, np.ceil(rho_vals.max() / 0.05) * 0.05 + 0.06)

    previous_rcparams = plt.rcParams.copy()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 260,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex="col", constrained_layout=False)

    panel_ids = [["a", "b"], ["c", "d"]]
    col_group_names = {high_label: "high-score anchors", low_label: "low-score anchors"}

    for row, scenario in enumerate(["cancer-specific", "mix-cancer"]):
        for col, anchor_group in enumerate([high_label, low_label]):
            group_data = plot_summary[
                (plot_summary["scenario"] == scenario)
                & (plot_summary["anchor_group"] == anchor_group)
            ]

            ax = axes[row, col]
            for model_name in ["SLformer", "Geneformer"]:
                model_data = group_data[group_data["model"] == model_name]
                ax.plot(
                    model_data["k"],
                    model_data["mean_rho_manifold_score"],
                    color=MODEL_COLORS[model_name],
                    marker="o",
                    linewidth=1.25,
                    markersize=3.2,
                    label=f"{MODEL_LABELS[model_name]}: score",
                )
                ax.plot(
                    model_data["k"],
                    model_data["mean_rho_manifold_label"],
                    color=MODEL_COLORS[model_name],
                    marker="s",
                    linestyle="--",
                    linewidth=0.95,
                    markersize=3.0,
                    alpha=0.78,
                    label=f"{MODEL_LABELS[model_name]}: label",
                )
            ax.axhline(0, color=PAPER_LIGHT_GRAY, linewidth=0.8, zorder=0)
            ax.set_ylim(y_lo, y_hi)
            ax.set_ylabel("mean $\\rho_{\\mathcal{G}}$")
            ax.set_title(panel_ids[row][col], loc="left", fontweight="bold")
            ax.set_title(f"{SCOPE_LABELS[scenario]}, {col_group_names[anchor_group]}", loc="center")

    for row in range(2):
        for col in range(2):
            axes[row, col].set_xticks(ks)
            axes[row, col].tick_params(axis="x", length=3)
            axes[row, col].tick_params(axis="y", length=3)
            axes[row, col].grid(axis="y", color=PAPER_LIGHT_GRAY, linewidth=0.55, alpha=0.65)
            axes[row, col].set_xlabel("K nearest neighbours by $d_{\\mathcal{G}}$" if row == 1 else "")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.23, wspace=0.32, hspace=0.36)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.9,
    )
    fig.suptitle(
        f"Anchor-conditioned manifold-distance SL ordering (high={high_n}, low={low_n} anchors)",
        x=0.10,
        y=0.95,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
    )
    plt.show()
    plt.rcParams.update(previous_rcparams)


def plot_local_manifold_score_contours(chart: pd.DataFrame, *, levels: int = 8, refine_subdiv: int = 3) -> None:
    plot_df = chart.copy()
    models = list(plot_df["model"].drop_duplicates())
    scenario = str(plot_df["scenario"].iloc[0])
    anchor_count = int(plot_df[plot_df["is_anchor"]]["index"].nunique())
    anchor_cancers = sorted(plot_df[plot_df["is_anchor"]]["cancer"].astype(str).unique().tolist())
    if "anchor_group" in plot_df.columns:
        anchor_groups = list(plot_df["anchor_group"].drop_duplicates())
        group_anchor_counts = (
            plot_df[plot_df["is_anchor"]]
            .groupby("anchor_group")["index"]
            .nunique()
            .astype(int)
        )
        anchor_layout_text = (
            f"{int(group_anchor_counts.iloc[0])} anchors per row"
            if group_anchor_counts.nunique() == 1
            else "grouped anchors"
        )
    else:
        anchor_groups = [None]
        anchor_layout_text = "one panel per model"
    score_cmap = mcolors.LinearSegmentedColormap.from_list(
        "score_blue_grey_purple",
        ["#F7FBFF", "#D7E8F5", "#AEB7C2", "#8176A8", "#4B2E83"],
    )
    contour_model_colors = {
        "SLformer": "#2166AC",
        "Geneformer": "#7A8190",
        "Geneformer-Probing": "#7A8190",
        "Gene2Vec": "#5E3C99",
    }
    contour_levels = np.linspace(0.0, 1.0, int(levels) + 1)

    previous_rcparams = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 260,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    n_rows = len(anchor_groups)
    n_cols = len(models)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.15 * n_cols, 2.55 * n_rows),
        squeeze=False,
        constrained_layout=False,
    )

    for row, anchor_group in enumerate(anchor_groups):
        for col, model_name in enumerate(models):
            ax = axes[row, col]
            model_df = plot_df[plot_df["model"] == model_name]
            if anchor_group is not None:
                model_df = model_df[model_df["anchor_group"] == anchor_group]
            neighbors = model_df[~model_df["is_anchor"]]
            anchors = model_df[model_df["is_anchor"]]

            triangulation = mtri.Triangulation(neighbors["m1"], neighbors["m2"])
            refined_tri, refined_score = mtri.UniformTriRefiner(triangulation).refine_field(
                neighbors["score"].to_numpy(dtype=float), subdiv=int(refine_subdiv))

            ax.tricontourf(refined_tri, refined_score, levels=contour_levels,
                           cmap=score_cmap, vmin=0.0, vmax=1.0, alpha=0.72)
            ax.tricontour(refined_tri, refined_score, levels=contour_levels,
                          colors="#51515F", linewidths=0.45, alpha=0.70)

            point_colors = np.where(neighbors["label"].to_numpy(dtype=int) == 1,
                                    "#4B2E83", "#F3F5F8")
            ax.scatter(neighbors["m1"], neighbors["m2"], c=point_colors, s=10,
                       edgecolor="#5A6270", linewidth=0.28, alpha=0.84)
            ax.scatter(anchors["m1"], anchors["m2"], marker="*", s=82,
                       color=contour_model_colors[model_name],
                       edgecolor="white", linewidth=0.7, zorder=5)
            if "anchor_label" in anchors.columns:
                for anchor in anchors.itertuples(index=False):
                    ax.annotate(
                        str(anchor.anchor_label),
                        (float(anchor.m1), float(anchor.m2)),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=6.2,
                        color=PAPER_GRAY,
                        zorder=6,
                    )

            label_one = int((neighbors["label"].to_numpy(dtype=int) == 1).sum())
            panel_score_label = str(model_df["score_field"].iloc[0]) if "score_field" in model_df.columns else "score"
            score_text = (
                f"{panel_score_label} mean={neighbors['score'].mean():.3f}; "
                f"SL True={label_one}/{len(neighbors)}"
            )
            title = f"{model_name}\n{score_text}" if row == 0 else score_text
            ax.set_title(title, fontsize=7.4)

            ax.set_xlabel("$m_1$" if row == n_rows - 1 else "")
            if col == 0:
                if "anchor_group_label" in model_df.columns:
                    group_label = str(model_df["anchor_group_label"].iloc[0])
                    ax.set_ylabel(f"{group_label}\n$m_2$")
                else:
                    ax.set_ylabel("$m_2$")
            else:
                ax.set_ylabel("")
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(axis="both", length=3)

    cbar_ax = fig.add_axes([0.92, 0.16, 0.012, 0.68])
    sm = plt.cm.ScalarMappable(cmap=score_cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("panel prediction score")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#F3F5F8",
               markeredgecolor="#5A6270", markersize=5, label="SL label False"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#4B2E83",
               markeredgecolor="#5A6270", markersize=5, label="SL label True"),
        Line2D([0], [0], marker="*", linestyle="", markerfacecolor=PAPER_BLUE,
               markeredgecolor="white", markersize=9, label="anchor"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=False, handletextpad=0.35, columnspacing=1.3)

    fig.subplots_adjust(left=0.075, right=0.90, top=0.90, bottom=0.10, wspace=0.26, hspace=0.52)
    cancer_text = ",".join(anchor_cancers[:3]) + ("..." if len(anchor_cancers) > 3 else "")
    fig.suptitle(
        f"Local {scenario} manifold score contours "
        f"({anchor_count} anchors, {anchor_layout_text}, cancer={cancer_text})",
        x=0.075, y=0.98, ha="left", fontsize=9.5, fontweight="bold")
    plt.show()
    plt.rcParams.update(previous_rcparams)


def compact_manifold_ordering_table(
    summary: pd.DataFrame,
    *,
    key_k: list[int] | tuple[int, ...],
    rho_label: str = "$\\rho_{\\mathcal{G}}$",
) -> pd.DataFrame:
    plot_summary = summary.copy()
    plot_summary["model_label"] = plot_summary["model"].map(MODEL_LABELS)
    plot_summary["scope_label"] = plot_summary["scenario"].map(SCOPE_LABELS)
    return plot_summary.loc[
        plot_summary["k"].isin(key_k),
        [
            "model_label",
            "scope_label",
            "k",
            "n",
            "label_valid",
            "mean_rho_manifold_score",
            "median_rho_manifold_score",
            "mean_rho_manifold_label",
            "median_rho_manifold_label",
            "mean_score",
            "mean_positive_fraction",
            "mean_same_cancer_fraction",
        ],
    ].rename(
        columns={
            "model_label": "model",
            "scope_label": "scope",
            "mean_rho_manifold_score": f"mean {rho_label} score",
            "median_rho_manifold_score": f"median {rho_label} score",
            "mean_rho_manifold_label": f"mean {rho_label} label",
            "median_rho_manifold_label": f"median {rho_label} label",
            "mean_score": "mean score",
            "mean_positive_fraction": "mean positive fraction",
            "mean_same_cancer_fraction": "mean same-cancer fraction",
        }
    ).round(3)


def _discrete_cmap(n_colors: int, base_cmap: str = "tab20") -> mcolors.ListedColormap:
    """Return a discrete colour map with *n_colors* distinct entries."""
    base = plt.get_cmap(base_cmap)
    colours = [base(i / max(n_colors - 1, 1)) for i in range(n_colors)]
    return mcolors.ListedColormap(colours[:n_colors])



def _mapper_chart_xy(entry: dict) -> tuple[np.ndarray, np.ndarray, str, str]:
    nerve = entry["nerve"]
    meta = entry["meta"]
    lens_dims = list(nerve["lens_dims"])
    chart_columns = ["score_axis", "residual_axis_1", "residual_axis_2"]
    x_col, y_col = [chart_columns[int(dim)] for dim in lens_dims]
    points = meta[[x_col, y_col]].to_numpy(dtype=float)
    nodes = nerve["node_centers"][:, lens_dims].astype(float)
    return points, nodes, x_col, y_col


def _mapper_anchor_rows(meta: pd.DataFrame) -> np.ndarray:
    return np.flatnonzero(meta["is_anchor"].to_numpy(dtype=bool))


def _mapper_node_summary(entry: dict) -> pd.DataFrame:
    nerve = entry["nerve"]
    meta = entry["meta"]
    scores = meta["score"].to_numpy(dtype=float)
    labels = meta["label"].to_numpy(dtype=float)
    rows = []
    for node_id, members in enumerate(nerve["node_members"]):
        box_i, box_j = nerve["cover_boxes"][node_id]
        rows.append({
            "node": int(node_id),
            "box_i": int(box_i),
            "box_j": int(box_j),
            "n": int(len(members)),
            "mean_score": float(scores[members].mean()),
            "mean_label": float(labels[members].mean()),
        })
    return pd.DataFrame(rows)


def _mapper_anchor_nodes(entry: dict) -> dict[int, list[str]]:
    meta = entry["meta"]
    point_node_memberships = entry["nerve"]["point_node_memberships"]
    anchor_nodes: dict[int, list[str]] = {}
    for rank, row in enumerate(_mapper_anchor_rows(meta), start=1):
        node_id = int(point_node_memberships[int(row)][0])
        label = str(meta["anchor_label"].iloc[int(row)]) if "anchor_label" in meta.columns else f"A{rank}"
        anchor_nodes.setdefault(node_id, []).append(label)
    return anchor_nodes


def _mapper_score_route(entry: dict) -> list[int]:
    """Return the adjacent-region path with the largest monotone score gain."""
    summary = _mapper_node_summary(entry)
    if len(summary) < 2:
        return summary["node"].astype(int).tolist()
    graph = entry["nerve"]["graph"]
    scores = summary.set_index("node")["mean_score"].to_dict()
    counts = summary.set_index("node")["n"].to_dict()
    edge_pairs = np.argwhere(np.triu(np.isfinite(graph) & (graph > 0), k=1))

    children: dict[int, list[int]] = {int(node): [] for node in summary["node"]}
    for u, v in edge_pairs:
        u = int(u)
        v = int(v)
        if scores[v] > scores[u]:
            children[u].append(v)
        elif scores[u] > scores[v]:
            children[v].append(u)

    path_by_end: dict[int, list[int]] = {int(node): [int(node)] for node in summary["node"]}
    for node in summary.sort_values("mean_score")["node"].astype(int):
        current_path = path_by_end[int(node)]
        for child in children[int(node)]:
            candidate = current_path + [int(child)]
            old = path_by_end[int(child)]
            candidate_gain = scores[candidate[-1]] - scores[candidate[0]]
            old_gain = scores[old[-1]] - scores[old[0]]
            candidate_support = sum(counts[n] for n in candidate)
            old_support = sum(counts[n] for n in old)
            if (candidate_gain, len(candidate), candidate_support) > (old_gain, len(old), old_support):
                path_by_end[int(child)] = candidate

    return max(
        path_by_end.values(),
        key=lambda path: (scores[path[-1]] - scores[path[0]], len(path), sum(counts[n] for n in path)),
    )


def _format_anchor_labels(labels: list[str]) -> str:
    numbers = sorted(int(label[1:]) for label in labels if label.startswith("A") and label[1:].isdigit())
    if len(numbers) == len(labels) and len(numbers) > 1:
        if numbers == list(range(numbers[0], numbers[-1] + 1)):
            return f"A{numbers[0]}-A{numbers[-1]}"
        return ",".join(f"A{number}" for number in numbers)
    return ",".join(sorted(labels))


def plot_manifold_topology_map(
    nerve_data: dict[str, dict],
    *,
    model_names: list[str] | None = None,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Option 1: CAML-like manifold maps with sparse dominant topology routes."""
    if model_names is None:
        model_names = list(nerve_data.keys())

    score_cmap = mcolors.LinearSegmentedColormap.from_list(
        "manifold_score_blue_orange", ["#F7FBFF", "#9ECAE1", PAPER_BLUE, PAPER_ORANGE]
    )
    previous_rcparams = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.6,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.0,
        "figure.dpi": 260,
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, len(model_names), figsize=(4.15 * len(model_names), 3.95), squeeze=False)
    for mi, model_name in enumerate(model_names):
        ax = axes[0, mi]
        entry = nerve_data[model_name]
        meta = entry["meta"]
        points, nodes, x_col, y_col = _mapper_chart_xy(entry)
        scores = meta["score"].to_numpy(dtype=float)
        labels = meta["label"].to_numpy(dtype=int)
        anchor_rows = _mapper_anchor_rows(meta)
        summary = _mapper_node_summary(entry)
        node_score = summary.set_index("node")["mean_score"].to_dict()
        route = _mapper_score_route(entry)

        triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
        refined_tri, refined_score = mtri.UniformTriRefiner(triangulation).refine_field(scores, subdiv=3)
        ax.tricontourf(refined_tri, refined_score, levels=np.linspace(0, 1, 9), cmap=score_cmap, alpha=0.72)
        ax.tricontour(refined_tri, refined_score, levels=np.linspace(0, 1, 7), colors="#4E555E", linewidths=0.35, alpha=0.38)
        ax.scatter(points[:, 0], points[:, 1], s=9, c="#404852", alpha=0.18, edgecolor="none", rasterized=True)
        positive = labels == 1
        ax.scatter(points[positive, 0], points[positive, 1], s=15, facecolor="none", edgecolor=PAPER_ORANGE,
                   linewidth=0.34, alpha=0.62)

        route_xy = nodes[np.array(route, dtype=int)]
        if len(route_xy) > 1:
            ax.plot(route_xy[:, 0], route_xy[:, 1], color=PAPER_BLUE, linewidth=2.55, alpha=0.95, zorder=4)
            ax.scatter(route_xy[:, 0], route_xy[:, 1], s=42, color=PAPER_BLUE, edgecolor="white", linewidth=0.65, zorder=5)
            ax.scatter(route_xy[0, 0], route_xy[0, 1], marker="s", s=54, color="#F7FBFF",
                       edgecolor=PAPER_BLUE, linewidth=1.2, zorder=6)
            ax.scatter(route_xy[-1, 0], route_xy[-1, 1], marker="D", s=58, color=PAPER_ORANGE,
                       edgecolor="white", linewidth=0.7, zorder=6)
            for step, node_id in enumerate(route, start=1):
                ax.annotate(
                    str(step),
                    route_xy[step - 1],
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color="white" if step not in {1, len(route)} else PAPER_BLUE,
                    zorder=7,
                )

        ax.scatter(points[anchor_rows, 0], points[anchor_rows, 1], marker="*", s=130, color=PAPER_ORANGE,
                   edgecolor="white", linewidth=0.75, zorder=6)
        for row in anchor_rows:
            ax.annotate(str(meta["anchor_label"].iloc[int(row)]), points[int(row)], xytext=(4, 4),
                        textcoords="offset points", fontsize=6.4, color="#2F3338", zorder=7)

        route_gain = node_score[route[-1]] - node_score[route[0]]
        ax.set_title(
            f"{model_name}\n{len(meta)} rows, {len(entry['nerve']['node_members'])} regions; "
            f"route gain={route_gain:.3f}"
        )
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(y_col.replace("_", " ") if mi == 0 else "")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(x=0.14, y=0.20)
        ax.set_aspect("equal", adjustable="datalim")

    cbar_ax = fig.add_axes([0.935, 0.28, 0.010, 0.44])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=score_cmap, norm=plt.Normalize(0, 1)), cax=cbar_ax)
    cbar.set_label("SL score $s_i$")
    legend_handles = [
        Line2D([0], [0], color=PAPER_BLUE, linewidth=2.3, label="monotone score route"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#F7FBFF",
               markeredgecolor=PAPER_BLUE, markersize=6, label="route start"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=PAPER_ORANGE,
               markeredgecolor="white", markersize=6, label="route end"),
        Line2D([0], [0], marker="*", linestyle="", markerfacecolor=PAPER_ORANGE,
               markeredgecolor="white", markersize=8, label="anchor"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=PAPER_ORANGE,
               markersize=5, label="$y_i=1$"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.48, 0.045), ncol=5, frameon=False)
    fig.subplots_adjust(left=0.055, right=0.905, top=0.76, bottom=0.22, wspace=0.26)
    fig.suptitle("Option 1: manifold map with monotone topology score route", x=0.055, y=0.94,
                 ha="left", fontsize=10.0, fontweight="bold")
    if out_path is not None:
        fig.savefig(Path(out_path), bbox_inches="tight")
    plt.rcParams.update(previous_rcparams)
    return fig


def plot_topology_region_matrix(
    nerve_data: dict[str, dict],
    *,
    model_names: list[str] | None = None,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Option 3: aligned cover-region matrices with a mapped monotone route."""
    if model_names is None:
        model_names = list(nerve_data.keys())

    score_cmap = mcolors.LinearSegmentedColormap.from_list(
        "region_score_blue_orange", ["#F7FBFF", "#9ECAE1", PAPER_BLUE, PAPER_ORANGE]
    )
    previous_rcparams = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.0,
        "figure.dpi": 260,
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, len(model_names), figsize=(3.60 * len(model_names), 3.95), squeeze=False)
    for mi, model_name in enumerate(model_names):
        ax = axes[0, mi]
        entry = nerve_data[model_name]
        summary = _mapper_node_summary(entry)
        route = _mapper_score_route(entry)
        n_x = int(summary["box_i"].max()) + 1
        n_y = int(summary["box_j"].max()) + 1
        max_n = float(summary["n"].max())
        score_by_node = summary.set_index("node")["mean_score"].to_dict()

        ax.set_xlim(-0.5, n_x - 0.5)
        ax.set_ylim(-0.5, n_y - 0.5)
        for _, row in summary.iterrows():
            alpha = 0.30 + 0.62 * float(row["n"]) / max_n
            rect = Rectangle(
                (float(row["box_i"]) - 0.45, float(row["box_j"]) - 0.45),
                0.90,
                0.90,
                facecolor=score_cmap(float(row["mean_score"])),
                edgecolor="#555B64",
                linewidth=0.55,
                alpha=alpha,
                zorder=2,
            )
            ax.add_patch(rect)

        box_by_node = summary.set_index("node")[["box_i", "box_j"]].to_dict("index")
        def route_color(node_id: int):
            score = float(score_by_node[int(node_id)])
            return "#6E737D" if score < 0.16 else score_cmap(score)

        for u, v in zip(route[:-1], route[1:]):
            p0 = box_by_node[int(u)]
            p1 = box_by_node[int(v)]
            ax.annotate(
                "",
                xy=(p1["box_i"], p1["box_j"]),
                xytext=(p0["box_i"], p0["box_j"]),
                arrowprops={"arrowstyle": "-|>", "color": "#F7FBFF", "linewidth": 3.5, "shrinkA": 10, "shrinkB": 10},
                zorder=4.8,
            )
            ax.annotate(
                "",
                xy=(p1["box_i"], p1["box_j"]),
                xytext=(p0["box_i"], p0["box_j"]),
                arrowprops={"arrowstyle": "-|>", "color": route_color(v), "linewidth": 2.1, "shrinkA": 10, "shrinkB": 10},
                zorder=5,
            )

        for node_id in route:
            box = box_by_node[int(node_id)]
            route_halo = Rectangle(
                (float(box["box_i"]) - 0.50, float(box["box_j"]) - 0.50),
                1.00,
                1.00,
                facecolor="none",
                edgecolor="#F7FBFF",
                linewidth=3.0,
                zorder=6.4,
            )
            route_border = Rectangle(
                (float(box["box_i"]) - 0.49, float(box["box_j"]) - 0.49),
                0.98,
                0.98,
                facecolor="none",
                edgecolor=route_color(node_id),
                linewidth=1.9,
                zorder=6.5,
            )
            ax.add_patch(route_halo)
            ax.add_patch(route_border)
        start_box = box_by_node[int(route[0])]
        end_box = box_by_node[int(route[-1])]
        ax.scatter(start_box["box_i"], start_box["box_j"], marker="s", s=74, color=route_color(route[0]),
                   edgecolor="#33383F", linewidth=0.9, zorder=8)
        ax.scatter(end_box["box_i"], end_box["box_j"], marker="D", s=78, color=route_color(route[-1]),
                   edgecolor="#33383F", linewidth=0.9, zorder=8)

        route_members = np.unique(np.concatenate([entry["nerve"]["node_members"][int(node)] for node in route]))
        route_gain = float(summary.loc[summary["node"] == route[-1], "mean_score"].iloc[0] -
                           summary.loc[summary["node"] == route[0], "mean_score"].iloc[0])
        ax.set_title(f"{model_name}\ngain={route_gain:.3f}; hops={len(route) - 1}; route n={len(route_members)}")
        ax.set_xlabel("cover residual axis 1")
        ax.set_ylabel("cover residual axis 2" if mi == 0 else "")
        ax.set_xticks(range(n_x))
        ax.set_yticks(range(n_y))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#C9CED6", linewidth=0.38, alpha=0.70)

    cbar_ax = fig.add_axes([0.935, 0.28, 0.010, 0.44])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=score_cmap, norm=plt.Normalize(0, 1)), cax=cbar_ax)
    cbar.set_label("cover mean SL score $\\bar{s}_C$")
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=score_cmap(0.75), edgecolor="#555B64", label="cover region"),
        Line2D([0], [0], color=score_cmap(0.75), linewidth=2.0, label="route colored by $\\bar{s}_C$"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#F7FBFF",
               markeredgecolor="#33383F", markersize=6, label="route start"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=score_cmap(0.95),
               markeredgecolor="#33383F", markersize=6, label="route end"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.48, 0.04), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.055, right=0.905, top=0.74, bottom=0.22, wspace=0.32)
    fig.suptitle("Option 3: topology region matrix with mapped score route", x=0.055, y=0.94,
                 ha="left", fontsize=10.0, fontweight="bold")
    if out_path is not None:
        fig.savefig(Path(out_path), bbox_inches="tight")
    plt.show()
    plt.rcParams.update(previous_rcparams)
    return fig
