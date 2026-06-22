"""Utilities for single-pair SAE feature explanation notebooks."""


from __future__ import annotations


import re


from pathlib import Path
from typing import Iterable, Sequence


import numpy as np


import pandas as pd


from .general import format_pair_evidence, sample_name, same_unordered_pair


def activation_feature_tables(
    *,
    candidate_features: Iterable[int],
    Z: np.ndarray,
    meta: pd.DataFrame,
    target_idx: int,
    context_cancer: str,
    sample_scope: str,
    top_m: int,
    explain_exemplars: int,
) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    rows = []
    explain_indices: dict[int, list[int]] = {}
    sample_scope = str(sample_scope).strip()
    if sample_scope not in {"same_cancer", "all_cancer"}:
        raise ValueError(f"sample_scope must be same_cancer or all_cancer, got {sample_scope!r}")
    primary = meta["primary_gene"].astype(str).to_numpy()
    partner = meta["partner_gene"].astype(str).to_numpy()
    cancers = meta["cancer"].astype(str).to_numpy()
    target_genes = {primary[int(target_idx)], partner[int(target_idx)]}
    same_pair = np.asarray([{primary[index], partner[index]} == target_genes for index in range(len(meta))], dtype=bool)
    scope_mask = np.ones(len(meta), dtype=bool) if sample_scope == "all_cancer" else cancers == str(context_cancer)
    eligible = np.flatnonzero(scope_mask & ~same_pair)
    n_high = min(int(top_m), int(eligible.size))

    for feature in candidate_features:
        activations = np.abs(Z[eligible, int(feature)])
        if n_high < int(eligible.size):
            high_unsorted = np.argpartition(activations, -n_high)[-n_high:]
            high_order = high_unsorted[np.argsort(activations[high_unsorted])[::-1]]
        else:
            high_order = np.argsort(activations)[::-1]
        high = eligible[high_order]
        explain = high[: int(explain_exemplars)].astype(int).tolist()

        explain_indices[int(feature)] = explain
        rows.append(
            {
                "feature": int(feature),
                "n_explain_pairs": int(len(explain)),
            }
        )

    return pd.DataFrame(rows), explain_indices

def top_feature_pair_lines(
    *,
    feature: int,
    Z: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    target_idx: int,
    context_cancer: str,
    sample_scope: str,
    n_pairs: int,
) -> str:
    ordered = np.argsort(np.abs(Z[:, int(feature)]))[::-1]
    sample_scope = str(sample_scope).strip()
    if sample_scope not in {"same_cancer", "all_cancer"}:
        raise ValueError(f"sample_scope must be same_cancer or all_cancer, got {sample_scope!r}")
    indices = [
        int(index)
        for index in ordered
        if int(index) != int(target_idx)
        and (sample_scope == "all_cancer" or str(meta.loc[int(index), "cancer"]) == str(context_cancer))
        and not same_unordered_pair(meta, int(index), int(target_idx))
    ][: int(n_pairs)]
    lines = []
    for number, index in enumerate(indices, start=1):
        lines.append(
            f"{number}. {sample_name(meta, index)} | SL score={float(y[index]):.4f} | feature activation={float(abs(Z[index, int(feature)])):.6f}"
        )
    return "\n".join(lines)


def rank_normalized(values: pd.Series) -> pd.Series:
    ranks = values.astype(float).rank(method="average")
    return (ranks - 1.0) / max(len(values) - 1, 1)


def rank_candidate_features(candidate_table: pd.DataFrame, evidence_table: pd.DataFrame) -> pd.DataFrame:
    ranked = candidate_table.merge(evidence_table, on="feature", how="inner")
    ranked["rank_jvp"] = rank_normalized(ranked["jvp"].abs())
    ranked["rank_decoder"] = rank_normalized(ranked["c_star"].abs())
    ranked["joint_rank_score"] = (ranked["rank_jvp"] + ranked["rank_decoder"]) / 2.0
    return ranked.sort_values("joint_rank_score", ascending=False).reset_index(drop=True)


def build_explainer_prompt(
    *,
    feature: int,
    feature_rank: pd.DataFrame,
    explain_indices: Sequence[int],
    Z: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    target_idx: int,
    target_cancer: str,
    sample_scope: str,
    prompt_dir: str | Path,
) -> str:
    row = feature_rank.loc[feature_rank["feature"] == int(feature)].iloc[0]
    exemplars = format_pair_evidence(
        indices=explain_indices,
        feature=feature,
        Z=Z,
        y=y,
        meta=meta,
        include_activation=True,
    )
    sample_scope = str(sample_scope).strip()
    evidence_scope_label = "all-cancer SAE manifold" if sample_scope == "all_cancer" else f"same-cancer {target_cancer} subset"
    top_pairs = top_feature_pair_lines(
        feature=feature,
        Z=Z,
        y=y,
        meta=meta,
        target_idx=target_idx,
        context_cancer=target_cancer,
        sample_scope=sample_scope,
        n_pairs=12,
    )
    template = (Path(prompt_dir) / "explainer_prompt_template.txt").read_text(encoding="utf-8")
    return template.format(
        target_cancer=target_cancer,
        evidence_scope_label=evidence_scope_label,
        feature=int(feature),
        z0=float(row["z0"]),
        dot_z=float(row["jvp"]),
        delta_z=float(row["delta_z"]),
        c_star=float(row["c_star"]),
        joint_rank_score=float(row["joint_rank_score"]),
        top_pairs=top_pairs,
        exemplars=exemplars,
    )




def target_relevance_note(feature_row: pd.Series) -> str:
    return (
        f"z0={float(feature_row['z0']):.4g}, "
        f"dot_z={float(feature_row['jvp']):.4g}, "
        f"Delta_z={float(feature_row['delta_z']):.4g}, "
        f"c_star={float(feature_row['c_star']):.4g}"
    )


def extract_explainer_fields(text: str) -> dict[str, str]:
    hypothesis = re.search(r"HYPOTHESIS:\s*(.+)", text)
    confidence = re.search(r"CONFIDENCE:\s*(.+)", text)
    rationale = re.search(r"RATIONALE:\s*([\s\S]+)", text)
    if hypothesis is None or confidence is None or rationale is None:
        raise ValueError(f"Explainer response missing required fields; response starts with: {text[:500]!r}")
    return {
        "hypothesis": hypothesis.group(1).strip(),
        "confidence": confidence.group(1).strip().lower(),
        "rationale": rationale.group(1).strip(),
    }
