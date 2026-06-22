"""Faithfulness and hallucination scoring for SAE interpretation text."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from .eval_common import clamp01, count_citations, extract_evidence_sentences, extract_mechanism_section, lexical_similarity, split_sentences, strip_citations


def cosine_to_unit_interval(cosine_similarity: float) -> float:
    value = max(-1.0, min(1.0, float(cosine_similarity)))
    return 1.0 - (math.acos(value) / math.pi)


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 300) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if len(value) <= int(chunk_size):
        return [value]
    chunks = []
    start = 0
    while start < len(value):
        chunk = value[start : start + int(chunk_size)].strip()
        if chunk:
            chunks.append(chunk)
        start += int(chunk_size) - int(overlap)
    return chunks


@lru_cache(maxsize=2)
def load_nli_pipeline(model_path: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    device_arg = -1
    if str(device).startswith("cuda"):
        device_arg = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=Path(model_path).exists())
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=Path(model_path).exists())
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_arg,
        top_k=None,
    )


def nli_scores_for_premise(
    *,
    premise: str,
    hypotheses: Sequence[str],
    nli_model_path: str,
    device: str,
    max_sentences: int,
) -> list[dict[str, float]]:
    pipe = load_nli_pipeline(str(nli_model_path), str(device))
    rows = []
    for hypothesis in list(hypotheses)[: int(max_sentences)]:
        result = pipe(f"{premise} </s></s> {hypothesis[:400]}", truncation=True, max_length=512)
        score_rows = result[0] if isinstance(result, list) and result else result
        scores = {str(row["label"]).lower(): float(row["score"]) for row in score_rows}
        rows.append(
            {
                "entailment": scores.get("entailment", scores.get("entail", 0.0)),
                "neutral": scores.get("neutral", 0.0),
                "contradiction": scores.get("contradiction", scores.get("contradict", 0.0)),
            }
        )
    return rows


def chunked_nli_scores(
    *,
    premise: str,
    hypotheses: Sequence[str],
    nli_model_path: str,
    device: str,
    max_sentences: int = 40,
    worst_k: int = 6,
) -> dict[str, Any]:
    hypotheses = list(hypotheses)[: int(max_sentences)]
    chunks = chunk_text(premise)
    if not hypotheses or not chunks:
        return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0, "n_sentences": 0, "n_chunks": len(chunks)}

    best_entailment = [0.0] * len(hypotheses)
    best_neutral = [1.0] * len(hypotheses)
    best_contradiction = [1.0] * len(hypotheses)
    for chunk in chunks:
        scores = nli_scores_for_premise(
            premise=chunk,
            hypotheses=hypotheses,
            nli_model_path=nli_model_path,
            device=device,
            max_sentences=max_sentences,
        )
        for index, row in enumerate(scores):
            if row["entailment"] > best_entailment[index]:
                best_entailment[index] = row["entailment"]
                best_neutral[index] = row["neutral"]
                best_contradiction[index] = row["contradiction"]

    n = len(hypotheses)
    sentence_scores = [
        {
            "sentence": hypothesis[:100] + "..." if len(hypothesis) > 100 else hypothesis,
            "entailment": best_entailment[index],
            "neutral": best_neutral[index],
            "contradiction": best_contradiction[index],
        }
        for index, hypothesis in enumerate(hypotheses)
    ]
    return {
        "entailment": float(sum(best_entailment) / n),
        "neutral": float(sum(best_neutral) / n),
        "contradiction": float(sum(best_contradiction) / n),
        "n_sentences": n,
        "n_chunks": len(chunks),
        "worst_sentences": sorted(sentence_scores, key=lambda row: row["entailment"])[: int(worst_k)],
        "means": {
            "entailment": float(sum(best_entailment) / n),
            "neutral": float(sum(best_neutral) / n),
            "contradiction": float(sum(best_contradiction) / n),
        },
    }


def lexical_alignment_score(source_text: str, target_sentences: Sequence[str]) -> tuple[float, dict[str, Any]]:
    source_sentences = split_sentences(source_text, min_chars=15)
    if not source_sentences or not target_sentences:
        return 0.0, {"backend": "lexical", "n_source_sentences": len(source_sentences), "n_target_sentences": len(target_sentences)}
    best_scores = []
    for target in target_sentences:
        best_scores.append(max(lexical_similarity(target, source) for source in source_sentences))
    return float(sum(best_scores) / len(best_scores)), {
        "backend": "lexical",
        "n_source_sentences": len(source_sentences),
        "n_target_sentences": len(target_sentences),
        "mean_best_similarity": float(sum(best_scores) / len(best_scores)),
        "min_best_similarity": float(min(best_scores)),
    }


def compute_faithfulness_score(
    *,
    text: str,
    ground_truth_explanation: str,
    prompt_context: str = "",
    nli_model_path: str | None = None,
    device: str = "cpu",
    max_sentences: int = 40,
    worst_k: int = 6,
) -> tuple[float, dict[str, Any]]:
    text_s = str(text or "").strip()
    gt_s = str(ground_truth_explanation or "").strip()
    citation_count, unique_count = count_citations(text_s)
    base = {
        "citation_count": citation_count,
        "unique_citation_count": unique_count,
        "has_citations": citation_count >= 2,
        "gt_faithfulness": 0.0,
        "faithfulness_score": 0.0,
    }
    if not gt_s:
        return 0.0, {**base, "backend": "none", "reason": "no_ground_truth", "enabled": False}

    generated_sentences = split_sentences(extract_mechanism_section(text_s))
    if not generated_sentences:
        return 0.0, {**base, "backend": "none", "reason": "no_sentences", "enabled": False}

    gt_sentences = split_sentences(gt_s, min_chars=15)
    if nli_model_path:
        fwd = chunked_nli_scores(
            premise=gt_s,
            hypotheses=generated_sentences,
            nli_model_path=nli_model_path,
            device=device,
            max_sentences=max_sentences,
            worst_k=worst_k,
        )
        fwd_score = clamp01((float(fwd["entailment"]) - float(fwd["contradiction"]) + 1.0) / 2.0)
        if gt_sentences:
            rev = chunked_nli_scores(
                premise=strip_citations(text_s),
                hypotheses=gt_sentences,
                nli_model_path=nli_model_path,
                device=device,
                max_sentences=max_sentences,
                worst_k=worst_k,
            )
            rev_score = clamp01((float(rev["entailment"]) - float(rev["contradiction"]) + 1.0) / 2.0)
        else:
            rev = {}
            rev_score = fwd_score
        score = clamp01(0.5 * fwd_score + 0.5 * rev_score)
        return score, {
            **base,
            "backend": "chunked_nli",
            "model_path": str(nli_model_path),
            "device": str(device),
            "n_sentences": fwd["n_sentences"],
            "n_gt_sentences": len(gt_sentences),
            "n_chunks_fwd": fwd["n_chunks"],
            "n_chunks_rev": rev.get("n_chunks", 0) if isinstance(rev, dict) else 0,
            "means": fwd["means"],
            "worst_sentences": fwd["worst_sentences"],
            "components": {"forward_faithfulness": fwd_score, "reverse_faithfulness": rev_score},
            "gt_faithfulness": score,
            "faithfulness_score": score,
            "enabled": True,
        }

    forward, forward_details = lexical_alignment_score(gt_s, generated_sentences)
    reverse, reverse_details = lexical_alignment_score(text_s, gt_sentences) if gt_sentences else (forward, {})
    score = clamp01(0.5 * forward + 0.5 * reverse)
    return score, {
        **base,
        "backend": "lexical",
        "n_sentences": len(generated_sentences),
        "n_gt_sentences": len(gt_sentences),
        "components": {"forward_faithfulness": forward, "reverse_faithfulness": reverse},
        "forward_details": forward_details,
        "reverse_details": reverse_details,
        "gt_faithfulness": score,
        "faithfulness_score": score,
        "enabled": True,
    }


@lru_cache(maxsize=2)
def load_sentence_embedder(model_path: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path))


def compute_kg_faithfulness(
    *,
    text: str,
    prompt_context: str,
    embed_model_path: str | None = None,
    max_sentences: int = 40,
    grounding_threshold: float = 0.35,
) -> tuple[float, dict[str, Any]]:
    sentences = split_sentences(extract_mechanism_section(text))
    evidence = extract_evidence_sentences(prompt_context)
    if not sentences or not evidence:
        return 0.0, {"reason": "no_sentences_or_evidence", "kg_faithfulness": 0.0, "n_sentences": len(sentences), "n_kg_edges": len(evidence)}

    sentences = sentences[: int(max_sentences)]
    if embed_model_path:
        import numpy as np

        embedder = load_sentence_embedder(str(embed_model_path))
        sent_vecs = embedder.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        evidence_vecs = embedder.encode(evidence, normalize_embeddings=True, show_progress_bar=False)
        similarities = np.dot(sent_vecs, evidence_vecs.T).max(axis=1)
        raw = float(np.mean(similarities))
        score = cosine_to_unit_interval(raw)
        n_grounded = int(np.sum(similarities >= float(grounding_threshold)))
        return score, {
            "kg_faithfulness": score,
            "kg_faithfulness_raw": raw,
            "normalization": {"type": "angular_cosine", "formula": "1 - arccos(cos_sim)/pi"},
            "n_sentences": len(sentences),
            "n_kg_edges": len(evidence),
            "n_grounded_sentences": n_grounded,
            "grounding_ratio": n_grounded / max(len(sentences), 1),
            "per_sent_max_sim": {
                "mean": raw,
                "median": float(np.median(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities)),
            },
            "method": "summac_embedding",
        }

    best_scores = [max(lexical_similarity(sentence, item) for item in evidence) for sentence in sentences]
    score = float(sum(best_scores) / len(best_scores))
    n_grounded = sum(1 for value in best_scores if value >= float(grounding_threshold))
    return score, {
        "kg_faithfulness": score,
        "kg_faithfulness_raw": score,
        "n_sentences": len(sentences),
        "n_kg_edges": len(evidence),
        "n_grounded_sentences": int(n_grounded),
        "grounding_ratio": n_grounded / max(len(sentences), 1),
        "per_sent_max_sim": {
            "mean": score,
            "median": sorted(best_scores)[len(best_scores) // 2],
            "min": min(best_scores),
            "max": max(best_scores),
        },
        "method": "lexical_evidence_similarity",
    }


def compute_hallucination_metrics(
    *,
    text: str,
    ground_truth_explanation: str,
    prompt_context: str = "",
    nli_model_path: str | None = None,
    embed_model_path: str | None = None,
    device: str = "cpu",
    max_sentences: int = 40,
) -> tuple[float, dict[str, Any]]:
    gt_faithfulness, details = compute_faithfulness_score(
        text=text,
        ground_truth_explanation=ground_truth_explanation,
        prompt_context=prompt_context,
        nli_model_path=nli_model_path,
        device=device,
        max_sentences=max_sentences,
    )
    kg_faithfulness, kg_details = compute_kg_faithfulness(
        text=text,
        prompt_context=prompt_context,
        embed_model_path=embed_model_path,
        max_sentences=max_sentences,
    )
    details["hallucination_score"] = 1.0 - gt_faithfulness
    details["faithfulness_score"] = gt_faithfulness
    details["gt_faithfulness"] = gt_faithfulness
    details["kg_faithfulness"] = kg_faithfulness
    details["kg_details"] = kg_details
    return 1.0 - gt_faithfulness, details
