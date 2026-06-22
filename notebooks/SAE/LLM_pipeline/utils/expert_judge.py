"""Format and grounding checks for SAE LLM answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .eval_common import clamp01, cosine, count_citations


EXPECTED_SECTION_NUMBERS = {1, 2, 3, 4, 5, 6, 7, 8}
SECTION_NUMBER_RE = re.compile(r"(?:^|\n)\s*(\d+)\s*\)", flags=re.MULTILINE)
FORMAT_COMPONENTS = [
    ("mechanism", "Mechanistic explanation linking gene A and gene B through pathway or causal chain."),
    ("evidence", "Evidence description with supporting interactions or concrete support."),
    ("direction", "Directionality or effect description such as activates inhibits increases decreases."),
    ("context", "Biological context assumptions condition scope and caveats."),
    ("caveats", "Limitations uncertainty and alternative hypotheses."),
    ("validation", "Suggested validation experiments perturbations checks and follow-ups."),
]
STRUCTURED_MARKERS = frozenset({"mechanistic summary", "evidence", "references", "output format", "required format", "json"})
KNOWN_HEADINGS = ("mechanistic summary", "evidence", "references", "caveat", "limitations")


def split_paragraphs(text: str, *, max_paragraphs: int = 24) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", str(text or "")) if chunk.strip()]
    if len(chunks) <= 1:
        chunks = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return chunks[: int(max_paragraphs)]


@lru_cache(maxsize=2)
def load_embedder(model_path: str):
    from sentence_transformers import SentenceTransformer

    resolved = Path(model_path)
    return SentenceTransformer(str(resolved if resolved.exists() else model_path))


def compute_hybrid_format_score(text: str, model_path: str | None = None) -> tuple[float, dict[str, Any]]:
    value = str(text or "").strip()
    if not value:
        return 0.0, {"reason": "empty_text", "component_coverage": 0.0, "structural_score": 0.0}

    citation_count, unique_count = count_citations(value)
    detected_sections = {int(number) for number in SECTION_NUMBER_RE.findall(value) if int(number) in EXPECTED_SECTION_NUMBERS}
    structural_compliance = len(detected_sections) / len(EXPECTED_SECTION_NUMBERS)
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    has_numbered = 1.0 if detected_sections else 0.0
    has_bullets = 1.0 if any(line.startswith(("- ", "* ", "•")) for line in lines) else 0.0
    has_colon_headings = 1.0 if sum(1 for line in lines[:40] if line.endswith(":")) >= 2 else 0.0
    length_score = clamp01(min(1.0, len(value) / 800.0))
    structural_score = clamp01(0.35 * has_numbered + 0.25 * has_colon_headings + 0.20 * has_bullets + 0.20 * length_score)

    paragraphs = split_paragraphs(value)
    if model_path:
        embedder = load_embedder(str(model_path))
        paragraph_vecs = embedder.encode([paragraph[:600] for paragraph in paragraphs] if paragraphs else [value[:600]], normalize_embeddings=True, show_progress_bar=False)
        component_vecs = embedder.encode([description for _, description in FORMAT_COMPONENTS], normalize_embeddings=True, show_progress_bar=False)
        component_scores: dict[str, float] = {}
        for (name, _), component_vec in zip(FORMAT_COMPONENTS, component_vecs):
            best = max(float(cosine(paragraph_vec, component_vec)) for paragraph_vec in paragraph_vecs)
            component_scores[name] = clamp01((best - 0.25) / 0.35)
    else:
        lower = value.lower()
        component_scores = {
            "mechanism": 1.0 if any(word in lower for word in ["mechanism", "pathway", "causal", "dependence", "synthetic"]) else 0.0,
            "evidence": 1.0 if any(word in lower for word in ["evidence", "exemplar", "support", "activation", "citation"]) else 0.0,
            "direction": 1.0 if any(word in lower for word in ["increase", "decrease", "activate", "inhibit", "positive", "negative", "buffer"]) else 0.0,
            "context": 1.0 if any(word in lower for word in ["cancer", "tumor", "context", "condition", "scope"]) else 0.0,
            "caveats": 1.0 if any(word in lower for word in ["caveat", "uncertain", "limitation", "alternative", "moderate"]) else 0.0,
            "validation": 1.0 if any(word in lower for word in ["validate", "validation", "assay", "perturb", "experiment"]) else 0.0,
        }

    component_coverage = sum(component_scores.values()) / len(component_scores)
    final_score = clamp01(0.75 * component_coverage + 0.25 * structural_score)
    return final_score, {
        "component_scores": component_scores,
        "component_coverage": float(component_coverage),
        "structural_score": float(structural_score),
        "structural_compliance": float(structural_compliance),
        "detected_sections": sorted(detected_sections),
        "citation_count": citation_count,
        "unique_citation_count": unique_count,
        "method": "embedding_format_components" if model_path else "lexical_format_components",
    }


def heuristic_checks(text: str, *, prompt_context: str = "") -> dict[str, object]:
    value = str(text or "")
    context = str(prompt_context or "").lower()
    citation_count, unique_count = count_citations(value)
    requires_structure = any(marker in context for marker in STRUCTURED_MARKERS)
    citations_expected = ("key=" in context) or ("citation" in context and "kg" in context) or ("evidence chains" in context)
    required = [heading for heading in KNOWN_HEADINGS if heading in context]
    if requires_structure and not required:
        required = ["mechanistic summary", "evidence"]
    present = [heading for heading in required if heading in value.lower()]

    score = 0.2 if value.strip() else 0.0
    if required:
        score += 0.55 * clamp01(len(present) / len(required))
    elif requires_structure:
        score += 0.1
    return {
        "format_score": clamp01(score),
        "grounding_ok": bool(citation_count >= 2) if citations_expected else False,
        "citation_count": citation_count,
        "unique_citation_count": unique_count,
        "judge_backend": "heuristic",
        "judge_model": "heuristic",
        "judge_parse_ok": True,
    }


@dataclass(frozen=True)
class ExpertJudgeSettings:
    model_name: str
    model_path: str | None = None


def judge_checks(
    text: str,
    *,
    prompt_context: str = "",
    settings: ExpertJudgeSettings | None = None,
) -> dict[str, object]:
    if settings is None or not settings.model_path:
        return heuristic_checks(text, prompt_context=prompt_context)

    score, details = compute_hybrid_format_score(text, settings.model_path)
    citation_count = int(details["citation_count"])
    structural_compliance = float(details["structural_compliance"])
    context = str(prompt_context or "").lower()
    citations_expected = ("key=" in context) or ("citation" in context and "kg" in context) or ("evidence chains" in context)
    return {
        "format_score": float(score),
        "grounding_ok": bool(citations_expected and citation_count >= 2 and structural_compliance >= 0.5),
        "citation_count": citation_count,
        "unique_citation_count": int(details["unique_citation_count"]),
        "judge_backend": "embedding_format_components",
        "judge_model": settings.model_name,
        "judge_parse_ok": True,
        "hybrid_details": details,
    }
