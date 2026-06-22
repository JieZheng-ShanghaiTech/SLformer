"""Payload and text metric helpers for SAE LLM evaluation."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

from .eval_common import count_citations
from .expert_judge import ExpertJudgeSettings, judge_checks
from .explanation_scoring import feature_embedding_prf1_by_coverage
from .hallucination_scoring import compute_hallucination_metrics


def hash_text(text: str) -> str:
    return sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def make_pair_payload(
    *,
    gene_a: str,
    gene_b: str,
    prompt_text: str,
    model_payload: Mapping[str, object],
    ground_truth_available: bool,
    ground_truth_features: Sequence[str],
    ground_truth_explanation: str,
    prompt_path: str | None = None,
    prompt_hash: str | None = None,
) -> dict[str, object]:
    return {
        "prompt_path": prompt_path,
        "gene_a": gene_a,
        "gene_b": gene_b,
        "prompt_hash": prompt_hash if prompt_hash is not None else hash_text(prompt_text),
        "model": dict(model_payload),
        "ground_truth": {
            "available": bool(ground_truth_available),
            "features": list(ground_truth_features),
            "explanation": str(ground_truth_explanation or ""),
        },
        "texts": {"prompt": prompt_text},
        "metrics": {},
    }


def score_text_metrics(
    *,
    ground_truth_features: Sequence[str],
    ground_truth_explanation: str,
    text: str,
    effective_model: object = None,
    prompt_context: str = "",
    feature_embed_model_path: str | None = None,
    judge_model_path: str | None = None,
    nli_model_path: str | None = None,
    evidence_embed_model_path: str | None = None,
    device: str = "cpu",
    feature_similarity_mode: str = "adjusted",
    feature_similarity_threshold: float = 0.6,
    feature_similarity_threshold_adjusted: float = 0.4,
    feature_score_scope: str = "section2",
    feature_candidate_backend: str = "lexicon",
    feature_tokencls_model_path: str | None = None,
    feature_max_candidates: int = 800,
    feature_soft_weight: float = 0.5,
) -> dict[str, Any]:
    feature_eval = feature_embedding_prf1_by_coverage(
        ground_truth_features=ground_truth_features,
        text=text,
        model_path=feature_embed_model_path,
        device=device,
        similarity_mode=feature_similarity_mode,
        similarity_threshold=feature_similarity_threshold,
        similarity_threshold_adjusted=feature_similarity_threshold_adjusted,
        score_scope=feature_score_scope,
        candidate_backend=feature_candidate_backend,
        tokencls_model_path=feature_tokencls_model_path,
        max_candidates=feature_max_candidates,
        soft_weight=feature_soft_weight,
    )
    checks = judge_checks(
        text,
        prompt_context=prompt_context,
        settings=ExpertJudgeSettings(model_name=str(judge_model_path), model_path=judge_model_path) if judge_model_path else None,
    )
    checks["judge_backend_used"] = str(checks["judge_backend"])
    citation_count, unique_citation_count = count_citations(text)
    checks["citation_count"] = citation_count
    checks["unique_citation_count"] = unique_citation_count

    hallucination_score, hallucination_details = compute_hallucination_metrics(
        text=text,
        ground_truth_explanation=ground_truth_explanation,
        prompt_context=prompt_context,
        nli_model_path=nli_model_path,
        embed_model_path=evidence_embed_model_path,
        device=device,
    )
    checks["hallucination_score"] = hallucination_score
    checks["hallucination_details"] = hallucination_details
    checks["faithfulness_score"] = hallucination_details["faithfulness_score"]
    checks["gt_faithfulness"] = hallucination_details["gt_faithfulness"]
    if citation_count < 2:
        checks["grounding_ok"] = False

    raw_recall = feature_eval["recall"] or 0.0
    gate = float(checks["format_score"]) if checks["grounding_ok"] and citation_count >= 2 else 0.0
    return {
        "feature_embed_f1_raw": raw_recall,
        "feature_embed_recall_only": raw_recall,
        "feature_embed_f1_raw_full": feature_eval["f1"],
        "feature_embed_f1_raw_topk_p50": feature_eval["topk"]["p50"]["f1"],
        "feature_embed_f1_raw_topk_p75": feature_eval["topk"]["p75"]["f1"],
        "hallucination_score": checks["hallucination_score"],
        "faithfulness_score": checks["faithfulness_score"],
        "feature_embed_f1": float(raw_recall) * gate,
        "feature_embed_gate": gate,
        "feature_embed_details": feature_eval,
        "feature_embed_precision_raw": feature_eval["precision"],
        "feature_embed_recall_raw": feature_eval["recall"],
        "checks": checks,
        "effective_model": effective_model,
    }
