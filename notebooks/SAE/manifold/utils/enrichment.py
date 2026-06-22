from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from SAE.manifold.core import local_manifold_projection, ordered_spearman


PAPER_BLUE = "#0072B2"
PAPER_ORANGE = "#D55E00"
PAPER_SKY = "#56B4E9"
PAPER_GREEN = "#009E73"
PAPER_LIGHT_GRAY = "#D9D9D9"
PAPER_GRAY = "#666666"
MODEL_COLORS = {
    "SLformer": PAPER_BLUE,
    "Geneformer": PAPER_ORANGE,
    "Gene2Vec": PAPER_GREEN,
}


def configure_publication_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 260,
            "savefig.dpi": 300,
        }
    )


def region_gene_set(meta: pd.DataFrame, row_indices: Sequence[int]) -> list[str]:
    region = meta.iloc[np.asarray(row_indices, dtype=int)]
    genes = pd.concat(
        [region["primary_gene"].astype(str), region["partner_gene"].astype(str)],
        ignore_index=True,
    )
    return sorted(genes.drop_duplicates().tolist())


def gene_level_matrix(meta: pd.DataFrame, X_pair: np.ndarray, *, cancer: str) -> tuple[pd.DataFrame, np.ndarray]:
    cancer_mask = meta["cancer"].astype(str).to_numpy() == str(cancer)
    meta_cancer = meta.loc[cancer_mask].reset_index(drop=True)
    X_cancer = np.asarray(X_pair[cancer_mask], dtype=np.float32)
    half_dim = X_cancer.shape[1] // 2

    gene_rows = pd.concat(
        [
            pd.DataFrame({"gene": meta_cancer["primary_gene"].astype(str), "row": np.arange(len(meta_cancer)), "half": 0}),
            pd.DataFrame({"gene": meta_cancer["partner_gene"].astype(str), "row": np.arange(len(meta_cancer)), "half": 1}),
        ],
        ignore_index=True,
    )
    vectors = np.concatenate([X_cancer[:, :half_dim], X_cancer[:, half_dim:]], axis=0)
    gene_rows["vector"] = list(vectors)

    grouped = gene_rows.groupby("gene", sort=True)["vector"].apply(lambda values: np.mean(np.stack(values), axis=0))
    counts = gene_rows.groupby("gene", sort=True).size().to_numpy(dtype=int)
    genes = grouped.index.to_numpy(dtype=str)
    X_gene = np.stack(grouped.to_numpy()).astype(np.float32)
    return pd.DataFrame({"gene": genes, "n_occurrences": counts}), X_gene


def load_go_symbol_annotations(gaf_path: str | Path, id_mapping_tsv: str | Path) -> dict[str, set[str]]:
    from goatools.associations import read_gaf

    annotations = read_gaf(str(gaf_path))
    id_mapping = pd.read_csv(id_mapping_tsv, sep="\t")
    uniprot_to_symbol = dict(zip(id_mapping["From"].astype(str), id_mapping["To"].astype(str)))

    symbol_annotations: dict[str, set[str]] = {}
    for uniprot_id, go_terms in annotations.items():
        if str(uniprot_id) in uniprot_to_symbol:
            symbol_annotations[uniprot_to_symbol[str(uniprot_id)]] = set(go_terms)
    return symbol_annotations


def region_go_coherence(
    genes: Sequence[str],
    symbol_annotations: Mapping[str, set[str]],
    *,
    min_shared_terms: int,
) -> dict[str, float | int]:
    annotated_genes = [gene for gene in genes if gene in symbol_annotations]
    counts = np.array(
        [
            len(set(symbol_annotations[gene1]).intersection(symbol_annotations[gene2]))
            for gene1, gene2 in combinations(annotated_genes, 2)
        ],
        dtype=float,
    )
    return {
        "n_annotated_genes": int(len(annotated_genes)),
        "n_gene_pairs": int(counts.size),
        "mean_shared_go_terms": float(counts.mean()) if counts.size else np.nan,
        "high_shared_pair_fraction": float((counts >= int(min_shared_terms)).mean()) if counts.size else np.nan,
    }


def go_pair_counts(genes: Sequence[str], symbol_annotations: Mapping[str, set[str]]) -> np.ndarray:
    annotated_genes = [gene for gene in genes if gene in symbol_annotations]
    return np.array(
        [
            len(set(symbol_annotations[gene1]).intersection(symbol_annotations[gene2]))
            for gene1, gene2 in combinations(annotated_genes, 2)
        ],
        dtype=float,
    )


def run_gmm_gene_clustering(
    gene_table: pd.DataFrame,
    X_gene: np.ndarray,
    *,
    n_components: int,
    seed: int,
) -> dict[str, list[str]]:
    X_scaled = StandardScaler().fit_transform(np.asarray(X_gene, dtype=np.float64))
    gmm = GaussianMixture(
        n_components=int(n_components),
        covariance_type="diag",
        random_state=int(seed),
        max_iter=500,
        reg_covar=1e-4,
        init_params="k-means++",
        tol=1e-3,
    )
    labels = gmm.fit_predict(X_scaled)
    clustered = gene_table.copy()
    clustered["cluster"] = labels.astype(str)
    return {
        cluster: rows["gene"].astype(str).tolist()
        for cluster, rows in clustered.groupby("cluster", sort=True)
    }


def cluster_go_background_summary(
    clusters: Mapping[str, Sequence[str]],
    symbol_annotations: Mapping[str, set[str]],
    *,
    model: str,
    background_samples: int,
    min_cluster_genes: int,
    min_shared_terms: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rng = np.random.default_rng(int(seed))
    gene_pool = sorted({gene for genes in clusters.values() for gene in genes if gene in symbol_annotations})
    background_pairs = list(combinations(gene_pool, 2))
    sampled = rng.choice(len(background_pairs), size=min(int(background_samples), len(background_pairs)), replace=False)
    background_counts = np.array(
        [
            len(symbol_annotations[background_pairs[int(i)][0]].intersection(symbol_annotations[background_pairs[int(i)][1]]))
            for i in sampled
        ],
        dtype=float,
    )
    background = {
        "background_mean": float(background_counts.mean()),
        "background_median": float(np.median(background_counts)),
        "background_q975": float(np.quantile(background_counts, 0.975)),
        "background_count": int(background_counts.size),
    }

    rows = []
    for cluster_id, genes in clusters.items():
        annotated = [gene for gene in genes if gene in symbol_annotations]
        if len(annotated) < int(min_cluster_genes):
            continue
        counts = go_pair_counts(annotated, symbol_annotations)
        u_stat, u_pvalue = mannwhitneyu(counts, background_counts, alternative="greater")
        rows.append(
            {
                "model": str(model),
                "cluster": str(cluster_id),
                "n_genes": int(len(genes)),
                "n_annotated_genes": int(len(annotated)),
                "n_gene_pairs": int(counts.size),
                "mean_shared_go_terms": float(counts.mean()),
                "high_shared_pair_fraction": float((counts >= int(min_shared_terms)).mean()),
                "background_mean": background["background_mean"],
                "background_q975": background["background_q975"],
                "go_lift": float(counts.mean() - background["background_mean"]),
                "mannwhitney_p": float(u_pvalue),
            }
        )
    return pd.DataFrame(rows), background


def best_cluster_summary(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    idx = cluster_summary.groupby("model")["go_lift"].idxmax()
    return cluster_summary.loc[idx].sort_values("go_lift", ascending=False).reset_index(drop=True)


def cluster_failure_diagnostic(gene_tables: Mapping[str, pd.DataFrame], cluster_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, gene_table in gene_tables.items():
        model_clusters = cluster_summary[cluster_summary["model"] == model]
        rows.append(
            {
                "model": model,
                "n_genes": int(len(gene_table)),
                "median_gene_occurrences": float(gene_table["n_occurrences"].median()),
                "single_occurrence_fraction": float((gene_table["n_occurrences"] == 1).mean()),
                "best_cluster_gene_fraction": float(model_clusters["n_annotated_genes"].max() / len(gene_table)),
                "median_cluster_go_mean": float(model_clusters["mean_shared_go_terms"].median()),
                "max_cluster_go_mean": float(model_clusters["mean_shared_go_terms"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_brca1_mapper_panel(
    coherence: pd.DataFrame,
    null_summary: pd.DataFrame,
    *,
    out_path: str | Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(3.7, 5.2), sharex=True, constrained_layout=False)
    ax = axes[0]
    ax.fill_between(
        null_summary["k"],
        null_summary["null_q025"],
        null_summary["null_q975"],
        color=PAPER_LIGHT_GRAY,
        alpha=0.62,
        linewidth=0,
        label="global null 95%",
    )
    ax.plot(null_summary["k"], null_summary["null_median"], color=PAPER_GRAY, linewidth=1.15, label="null median")
    for model, color in MODEL_COLORS.items():
        rows = coherence[coherence["model"] == model].sort_values("k")
        ax.plot(rows["k"], rows["mean_shared_go_terms"], marker="o", linewidth=1.25, markersize=3.2, color=color, label=model)
    ax.grid(axis="y", color=PAPER_LIGHT_GRAY, linewidth=0.55, alpha=0.65)
    ax.set_xlabel("")
    ax.set_ylabel("mean shared GO terms")
    ax.set_title("a", loc="left", fontweight="bold")
    ax.set_title("BRCA1-PARP1 GO coherence", loc="center")

    ax = axes[1]
    for model, color in MODEL_COLORS.items():
        rows = coherence[coherence["model"] == model].sort_values("k")
        ax.plot(rows["k"], rows["positive_fraction"], marker="s", linewidth=1.05, markersize=3.0, color=color, alpha=0.88, label=model)
    ax.grid(axis="y", color=PAPER_LIGHT_GRAY, linewidth=0.55, alpha=0.65)
    ax.set_xlabel("K nearest rows by $d_{\\mathcal{G}}$")
    ax.set_ylabel("positive-label fraction")
    ax.set_title("b", loc="left", fontweight="bold")
    ax.set_title("SL-label composition", loc="center")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.18, right=0.98, top=0.93, bottom=0.25, hspace=0.30)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        handlelength=2.0,
        columnspacing=0.8,
    )
    if out_path is not None:
        fig.savefig(Path(out_path))
    return fig


def plot_mapper_model_summary(
    cluster_best: pd.DataFrame,
    target_k_summary: pd.DataFrame,
    *,
    k: int,
    out_path: str | Path | None = None,
) -> plt.Figure:
    ordered_models = [model for model in MODEL_COLORS if model in set(target_k_summary["model"])]
    model_summary = (
        target_k_summary.groupby("model", as_index=False)
        .agg(
            mean_shared_go_terms=("mean_shared_go_terms", "mean"),
            mean_positive_fraction=("positive_fraction", "mean"),
            mean_score=("mean_score", "mean"),
        )
        .set_index("model")
        .loc[ordered_models]
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.7, 5.2), constrained_layout=False)
    ax = axes[0]
    x = np.arange(len(model_summary))
    heights = model_summary["mean_shared_go_terms"].to_numpy(dtype=float)
    ax.bar(
        x,
        heights,
        color=[MODEL_COLORS[model] for model in model_summary["model"]],
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_xticks(x, labels=model_summary["model"], rotation=0, ha="center")
    ax.set_ylabel("mean shared GO terms")
    ax.set_title("a", loc="left", fontweight="bold")
    ax.set_title(f"BRCA/PARP targets, top-{k}", loc="center")
    ax.grid(axis="y", color=PAPER_LIGHT_GRAY, linewidth=0.55, alpha=0.65)

    ax = axes[1]
    for model, color in MODEL_COLORS.items():
        rows = target_k_summary[target_k_summary["model"] == model]
        if rows.empty:
            continue
        ax.scatter(
            rows["positive_fraction"],
            rows["mean_shared_go_terms"],
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=model,
        )
    ax.set_xlabel("positive-label fraction")
    ax.set_ylabel("mean shared GO terms")
    ax.set_title("b", loc="left", fontweight="bold")
    ax.set_title("GO signal vs SL-label concentration", loc="center")
    ax.grid(axis="y", color=PAPER_LIGHT_GRAY, linewidth=0.55, alpha=0.65)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.93, bottom=0.17, hspace=0.48)
    if out_path is not None:
        fig.savefig(Path(out_path))
    return fig


def mapper_order_for_scores(
    X_model: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_idx: int,
    fit_k: int,
    tangent_dim: int,
    ridge_rho: float,
    nerve_lens_dims: Sequence[int],
    nerve_covers: int,
    nerve_overlap: float,
    nerve_cover_mode: str,
    nerve_cover_core_fraction: float,
    nerve_cluster_mode: str,
    nerve_dbscan_eps_quantile: float,
    nerve_dbscan_min_samples: int,
    nerve_fallback_unreachable: bool,
) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=int(fit_k) + 1, metric="cosine").fit(X_model)
    neighbours = nn.kneighbors(X_model[int(target_idx) : int(target_idx) + 1], return_distance=False)[0]
    fit_indices = neighbours[neighbours != int(target_idx)][: int(fit_k)]
    mapper = local_manifold_projection(
        X_model,
        scores,
        labels,
        anchor_index=int(target_idx),
        fit_indices=fit_indices,
        display_k=int(fit_k),
        tangent_dim=int(tangent_dim),
        ridge_rho=float(ridge_rho),
        distance_method="mapper-nerve",
        nerve_lens_dims=nerve_lens_dims,
        nerve_covers=int(nerve_covers),
        nerve_overlap=float(nerve_overlap),
        nerve_cover_mode=str(nerve_cover_mode),
        nerve_cover_core_fraction=float(nerve_cover_core_fraction),
        nerve_cluster_mode=str(nerve_cluster_mode),
        nerve_dbscan_eps_quantile=float(nerve_dbscan_eps_quantile),
        nerve_dbscan_min_samples=int(nerve_dbscan_min_samples),
        nerve_fallback_unreachable=bool(nerve_fallback_unreachable),
    )
    return mapper.sort_values("manifold_distance")["index"].to_numpy(dtype=int)


def mapper_permutation_stress_test(
    X_model: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    target_indices: Sequence[int],
    *,
    model: str,
    ks: Sequence[int],
    fit_k: int,
    n_permutations: int,
    seed: int,
    tangent_dim: int,
    ridge_rho: float,
    nerve_lens_dims: Sequence[int],
    nerve_covers: int,
    nerve_overlap: float,
    nerve_cover_mode: str,
    nerve_cover_core_fraction: float,
    nerve_cluster_mode: str,
    nerve_dbscan_eps_quantile: float,
    nerve_dbscan_min_samples: int,
    nerve_fallback_unreachable: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    observed_rows = []
    null_rows = []
    for target_idx in target_indices:
        observed_order = mapper_order_for_scores(
            X_model,
            scores,
            labels,
            target_idx=int(target_idx),
            fit_k=fit_k,
            tangent_dim=tangent_dim,
            ridge_rho=ridge_rho,
            nerve_lens_dims=nerve_lens_dims,
            nerve_covers=nerve_covers,
            nerve_overlap=nerve_overlap,
            nerve_cover_mode=nerve_cover_mode,
            nerve_cover_core_fraction=nerve_cover_core_fraction,
            nerve_cluster_mode=nerve_cluster_mode,
            nerve_dbscan_eps_quantile=nerve_dbscan_eps_quantile,
            nerve_dbscan_min_samples=nerve_dbscan_min_samples,
            nerve_fallback_unreachable=nerve_fallback_unreachable,
        )
        for k in ks:
            idx = observed_order[: int(k)]
            observed_rows.append(
                {
                    "model": model,
                    "target_idx": int(target_idx),
                    "k": int(k),
                    "rho_score": ordered_spearman(scores[idx]),
                    "rho_label": ordered_spearman(labels[idx]),
                    "positive_fraction": float(labels[idx].mean()),
                    "mean_score": float(scores[idx].mean()),
                }
            )
        for permutation_id in range(int(n_permutations)):
            permuted_scores = rng.permutation(scores)
            permuted_order = mapper_order_for_scores(
                X_model,
                permuted_scores,
                labels,
                target_idx=int(target_idx),
                fit_k=fit_k,
                tangent_dim=tangent_dim,
                ridge_rho=ridge_rho,
                nerve_lens_dims=nerve_lens_dims,
                nerve_covers=nerve_covers,
                nerve_overlap=nerve_overlap,
                nerve_cover_mode=nerve_cover_mode,
                nerve_cover_core_fraction=nerve_cover_core_fraction,
                nerve_cluster_mode=nerve_cluster_mode,
                nerve_dbscan_eps_quantile=nerve_dbscan_eps_quantile,
                nerve_dbscan_min_samples=nerve_dbscan_min_samples,
                nerve_fallback_unreachable=nerve_fallback_unreachable,
            )
            for k in ks:
                idx = permuted_order[: int(k)]
                null_rows.append(
                    {
                        "model": model,
                        "target_idx": int(target_idx),
                        "permutation": int(permutation_id),
                        "k": int(k),
                        "rho_score": ordered_spearman(scores[idx]),
                        "rho_label": ordered_spearman(labels[idx]),
                        "positive_fraction": float(labels[idx].mean()),
                        "mean_score": float(scores[idx].mean()),
                    }
                )
    return pd.DataFrame(observed_rows), pd.DataFrame(null_rows)


def summarize_permutation_stress(observed: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    observed_summary = observed.groupby(["model", "k"], as_index=False).agg(
        observed_rho_score=("rho_score", "mean"),
        observed_rho_label=("rho_label", "mean"),
        observed_positive_fraction=("positive_fraction", "mean"),
        observed_mean_score=("mean_score", "mean"),
    )
    null_summary = null.groupby(["model", "k"], as_index=False).agg(
        null_rho_score_median=("rho_score", "median"),
        null_rho_score_q025=("rho_score", lambda values: float(np.nanquantile(values, 0.025))),
        null_rho_score_q975=("rho_score", lambda values: float(np.nanquantile(values, 0.975))),
        null_rho_label_median=("rho_label", "median"),
        null_positive_fraction_median=("positive_fraction", "median"),
        null_mean_score_median=("mean_score", "median"),
    )
    summary = observed_summary.merge(null_summary, on=["model", "k"], how="inner")
    summary["rho_score_excess"] = summary["observed_rho_score"] - summary["null_rho_score_median"]
    summary["label_fraction_excess"] = summary["observed_positive_fraction"] - summary["null_positive_fraction_median"]
    return summary


def plot_permutation_stress(summary: pd.DataFrame, *, out_path: str | Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(3.7, 5.2), sharex=True, constrained_layout=False)
    ax = axes[0]
    null_label_used = False
    for model, color in MODEL_COLORS.items():
        rows = summary[summary["model"] == model].sort_values("k")
        if rows.empty:
            continue
        ax.fill_between(
            rows["k"],
            rows["null_rho_score_q025"],
            rows["null_rho_score_q975"],
            color=PAPER_LIGHT_GRAY,
            alpha=0.42,
            linewidth=0,
            label="permuted 95%" if not null_label_used else None,
        )
        null_label_used = True
        ax.plot(rows["k"], rows["observed_rho_score"], marker="o", linewidth=1.25, markersize=3.2, color=color, label=model)
        ax.plot(rows["k"], rows["null_rho_score_median"], linestyle="--", linewidth=1.0, color=color, alpha=0.78)
    ax.axhline(0.0, color=PAPER_LIGHT_GRAY, linewidth=0.8)
    ax.grid(axis="y", color=PAPER_LIGHT_GRAY, linewidth=0.55, alpha=0.65)
    ax.set_xlabel("")
    ax.set_ylabel("mean $\\rho(s_i)$")
    ax.set_title("a", loc="left", fontweight="bold")
    ax.set_title("Permutation stress test", loc="center")

    ax = axes[1]
    for model, color in MODEL_COLORS.items():
        rows = summary[summary["model"] == model].sort_values("k")
        if rows.empty:
            continue
        ax.plot(rows["k"], rows["rho_score_excess"], marker="o", linewidth=1.25, markersize=3.2, color=color, label=model)
    ax.axhline(0.0, color=PAPER_GRAY, linewidth=0.85)
    ax.grid(axis="y", color=PAPER_LIGHT_GRAY, linewidth=0.55, alpha=0.65)
    ax.set_xlabel("K nearest rows by score-conditioned $d_{\\mathcal{G}}$")
    ax.set_ylabel("observed minus permuted median")
    ax.set_title("b", loc="left", fontweight="bold")
    ax.set_title("Residual ordering evidence", loc="center")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.18, right=0.98, top=0.93, bottom=0.25, hspace=0.30)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        handlelength=2.0,
        columnspacing=0.8,
    )
    if out_path is not None:
        fig.savefig(Path(out_path))
    return fig
