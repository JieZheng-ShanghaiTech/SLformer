"""Composite metrics for SAE interpretation evaluation."""

from __future__ import annotations

from typing import Mapping

from .eval_common import clamp01


def harmonic_mean_2(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * float(a) * float(b) / (float(a) + float(b))


def grounded_feature_parts(feature_recall: float | None, checks: Mapping[str, object]) -> dict[str, float]:
    recall = float(feature_recall or 0.0)
    gt_faithfulness = float(checks.get("faithfulness_score") or 0.0)
    evidence_faithfulness = float(checks.get("kg_faithfulness") or 0.0)
    combined = harmonic_mean_2(gt_faithfulness, evidence_faithfulness)
    return {
        "recall": recall,
        "gt_faithfulness": gt_faithfulness,
        "kg_faithfulness": evidence_faithfulness,
        "combined_faithfulness": combined,
        "score": clamp01(recall * combined),
    }


def grounded_feature_score(feature_recall: float | None, checks: Mapping[str, object]) -> float:
    return float(grounded_feature_parts(feature_recall, checks)["score"])
