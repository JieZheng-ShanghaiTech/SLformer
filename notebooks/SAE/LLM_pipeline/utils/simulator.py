"""Simulator prompt and validation helpers for SAE feature explanations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .general import format_pair_evidence


def actual_feature_activations(*, indices: Sequence[int], feature: int, Z: np.ndarray) -> np.ndarray:
    return np.asarray([float(abs(Z[int(index), int(feature)])) for index in indices], dtype=np.float32)


def validation_set_profile(actual: np.ndarray) -> dict[str, Any]:
    values = np.asarray(actual, dtype=np.float32)
    positive = values > 0.0
    zero = values == 0.0
    if values.size < 2:
        status = "too_few_test_pairs"
    elif positive.any() and zero.any():
        status = "positive_zero_contrast"
    elif zero.all():
        status = "zero_control_only"
    else:
        status = "positive_constant_only"
    return {
        "validation_status": status,
    }


def simulator_validation_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.float32)
    act = np.asarray(actual, dtype=np.float32)
    act_max = float(act.max()) if act.size > 0 else 0.0
    act_0_10 = 10.0 * act / act_max if act_max > 0.0 else act
    metrics = validation_set_profile(act)
    metrics["predicted_activation_0_10"] = pred.tolist()
    metrics["predicted_unique_values"] = int(np.unique(pred).size) if pred.size > 0 else 0
    metrics["predicted_activation_mean"] = float(pred.mean()) if pred.size > 0 else float("nan")
    metrics["predicted_activation_max"] = float(pred.max()) if pred.size > 0 else float("nan")
    metrics["spearman_r"] = float(spearmanr(pred, act).statistic) if pred.size >= 2 and np.unique(pred).size > 1 and np.unique(act).size > 1 else float("nan")
    metrics["pearson_r"] = float(pearsonr(pred, act).statistic) if pred.size >= 2 and np.unique(pred).size > 1 and np.unique(act).size > 1 else float("nan")
    metrics["mae_0_10_vs_scaled_activation"] = float(np.mean(np.abs(pred - act_0_10))) if pred.size > 0 else float("nan")

    positive = act > 0.0
    zero = act == 0.0
    metrics["positive_control_margin"] = float(pred[positive].mean() - pred[zero].mean()) if positive.any() and zero.any() else "not_applicable"
    metrics["zero_control_mean_prediction"] = float(pred[zero].mean()) if zero.any() else "not_applicable"
    metrics["zero_control_max_prediction"] = float(pred[zero].max()) if zero.any() else "not_applicable"
    return metrics


def target_relevance_note(feature_row: pd.Series) -> str:
    return (
        f"z0={float(feature_row['z0']):.4g}, "
        f"dot_z={float(feature_row['jvp']):.4g}, "
        f"Delta_z={float(feature_row['delta_z']):.4g}, "
        f"c_star={float(feature_row['c_star']):.4g}"
    )


def chunk_indices(indices: Sequence[int], chunk_size: int) -> list[list[int]]:
    values = [int(index) for index in indices]
    return [values[start : start + int(chunk_size)] for start in range(0, len(values), int(chunk_size))]


def build_simulator_prompt(
    *,
    feature: int,
    hypothesis: str,
    test_indices: Sequence[int],
    Z: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    target_cancer: str,
    sample_scope: str,
    prompt_dir: str | Path,
) -> str:
    test_pairs = format_pair_evidence(
        indices=test_indices,
        feature=feature,
        Z=Z,
        y=y,
        meta=meta,
        include_activation=False,
    )
    sample_scope = str(sample_scope).strip()
    if sample_scope not in {"same_cancer", "all_cancer"}:
        raise ValueError(f"sample_scope must be same_cancer or all_cancer, got {sample_scope!r}")
    evidence_scope_label = "all-cancer SAE manifold" if sample_scope == "all_cancer" else f"same-cancer {target_cancer} subset"
    template = (Path(prompt_dir) / "simulator_prompt_template.txt").read_text(encoding="utf-8")
    return template.format(
        feature=int(feature),
        hypothesis=hypothesis,
        target_cancer=target_cancer,
        evidence_scope_label=evidence_scope_label,
        test_pairs=test_pairs,
    )


def clean_pair_name(name: str) -> str:
    value = str(name).strip().strip("`'\"")
    value = re.sub(r"^\s*\d+[.)]\s*", "", value)
    value = re.sub(r"\s+", "", value)
    return value


def reversed_pair_name(name: str) -> str:
    value = clean_pair_name(name)
    if ":" in value:
        context, pair = value.split(":", 1)
        gene_a, gene_b = pair.split("-", 1)
        return f"{context}:{gene_b}-{gene_a}"
    gene_a, gene_b = value.split("-", 1)
    return f"{gene_b}-{gene_a}"


def parse_simulator_scores(text: str) -> dict[str, float]:
    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match is not None:
        obj = json.loads(object_match.group(0))
        if isinstance(obj, dict):
            return {clean_pair_name(pair): float(score) for pair, score in obj.items()}
        return {clean_pair_name(row["pair"]): float(row["predicted_activation"]) for row in obj}

    json_match = re.search(r"\[[\s\S]*\]", text)
    if json_match is not None:
        rows = json.loads(json_match.group(0))
        return {clean_pair_name(row["pair"]): float(row["predicted_activation"]) for row in rows}

    labeled = r"PAIR:\s*([A-Za-z0-9_.]+:[A-Za-z0-9_.-]+-[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+-[A-Za-z0-9_.-]+).*?PREDICTED_ACTIVATION:\s*([0-9]+(?:\.[0-9]+)?)"
    scores = {clean_pair_name(pair): float(score) for pair, score in re.findall(labeled, text, flags=re.S)}
    if scores:
        return scores

    table = r"([A-Za-z0-9_.]+:[A-Za-z0-9_.-]+-[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+-[A-Za-z0-9_.-]+)[^\n0-9]*(?:activation|score)?[^\n0-9]*([0-9]+(?:\.[0-9]+)?)"
    return {clean_pair_name(pair): float(score) for pair, score in re.findall(table, text, flags=re.I)}


def simulator_scores_for_pairs(text: str, expected_pairs: Sequence[str]) -> dict[str, float]:
    parsed = parse_simulator_scores(text)
    predicted = {}
    missing = []
    for pair in expected_pairs:
        pair_key = clean_pair_name(pair)
        reverse_key = reversed_pair_name(pair_key)
        if pair_key in parsed:
            predicted[pair] = parsed[pair_key]
        elif reverse_key in parsed:
            predicted[pair] = parsed[reverse_key]
        else:
            missing.append(pair)
    if missing:
        parsed_preview = ", ".join(sorted(parsed)[:12])
        raise KeyError(
            f"Simulator response missing {len(missing)}/{len(expected_pairs)} expected pairs; "
            f"first missing={missing[:5]}; parsed keys preview={parsed_preview}"
        )
    return predicted
