"""Shared text helpers for SAE LLM evaluation."""

from __future__ import annotations

import re
from typing import Sequence


KG_EDGE_CIT_RE = re.compile(
    r"\([^)]*(?:->|<->)[^)]*\|[^)]*\|[^)]*(?:\bkey\s*=\s*\d+)?[^)]*\)",
    flags=re.IGNORECASE | re.DOTALL,
)
KG_KEY_RE = re.compile(r"\bkey\s*=\s*(\d+)\b", flags=re.IGNORECASE)
KG_EDGE_RE = re.compile(
    r"([A-Za-z0-9_\-/\s.]+?)\s*(?:->|<->)\s*([A-Za-z0-9_\-/\s.]+?)\s*\|\s*([^|]+?)\s*\|",
    flags=re.IGNORECASE,
)
KG_NODE_ID_RE = re.compile(
    r"\b(?:gene|pathway|protein|drug|disease|cohort)\s*:\s*[a-z0-9_.\-]+\b",
    flags=re.IGNORECASE,
)
KG_REACTOME_ID_RE = re.compile(r"\bR-HSA-\d+\b", flags=re.IGNORECASE)

URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
BRACKET_CIT_RE = re.compile(r"\[(?:\s*\d+\s*(?:[,;\-]\s*\d+\s*)*)\]", flags=re.IGNORECASE)
PMID_RE = re.compile(r"\bpmid\s*[:#]?\s*\d+\b", flags=re.IGNORECASE)
DOI_RE = re.compile(r"\bdoi\s*[:#]?\s*\S+", flags=re.IGNORECASE)
REF_SECTION_RE = re.compile(r"\b(references|bibliography)\b", flags=re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
WORD_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)

MECHANISM_SECTION_NAMES = frozenset({"mechanism", "mechanistic summary"})
KNOWN_SECTION_NAMES = frozenset(
    {
        "analysis",
        "mechanism",
        "mechanistic summary",
        "sae geometry evidence",
        "evidence",
        "evidence chain",
        "caveats",
        "validation",
        "validation readout",
        "limitations",
        "final interpretation",
        "trace",
        "initial draft",
        "self refine feedback",
        "self refine revised draft",
        "cove questions",
        "cove answers",
    }
)


SCORING_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "these",
        "this",
        "to",
        "was",
        "were",
        "with",
        "without",
        "while",
        "will",
        "would",
    }
)


def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def normalize_section_heading(line: str) -> tuple[str, str]:
    value = str(line or "").strip()
    value = re.sub(r"^#+\s*", "", value)
    value = re.sub(r"^\[|\]$", "", value)
    value = re.sub(r"^\d+\s*[\).:-]\s*", "", value)
    tail = ""
    if ":" in value:
        head, tail = value.split(":", 1)
    else:
        head = value
    head = normalize_text(re.sub(r"[^A-Za-z0-9\s/-]+", " ", head)).lower()
    return head, tail.strip()


def extract_mechanism_section(text: str) -> str:
    lines = str(text or "").splitlines()
    start = None
    chunk: list[str] = []
    for index, line in enumerate(lines):
        heading, tail = normalize_section_heading(line)
        if heading in MECHANISM_SECTION_NAMES:
            start = index
            if tail:
                chunk.append(tail)
            break
    if start is None:
        return ""

    for line in lines[start + 1 :]:
        heading, _ = normalize_section_heading(line)
        stripped = line.strip()
        if stripped and (stripped.startswith("#") or heading in KNOWN_SECTION_NAMES):
            break
        chunk.append(line)
    return "\n".join(chunk).strip()


def tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(str(text or ""))]


def content_tokens(text: str) -> list[str]:
    return [token for token in tokens(text) if token not in SCORING_STOPWORDS and not token.isdigit()]


def count_citations(text: str) -> tuple[int, int]:
    citations = KG_EDGE_CIT_RE.findall(str(text or ""))
    keys = set()
    for citation in citations:
        match = KG_KEY_RE.search(citation)
        if match is not None:
            keys.add(match.group(1))
    return len(citations), len(keys)


def strip_citations(text: str) -> str:
    value = str(text or "")
    ref_match = REF_SECTION_RE.search(value)
    if ref_match is not None:
        value = value[: ref_match.start()]
    value = URL_RE.sub(" ", value)
    value = BRACKET_CIT_RE.sub(" ", value)
    value = PMID_RE.sub(" ", value)
    value = DOI_RE.sub(" ", value)
    value = KG_EDGE_CIT_RE.sub(" ", value)
    value = KG_NODE_ID_RE.sub(" ", value)
    value = KG_REACTOME_ID_RE.sub(" ", value)
    return value


def split_sentences(text: str, *, min_chars: int = 25) -> list[str]:
    cleaned = normalize_text(strip_citations(text))
    if not cleaned:
        return []
    sentences = SENTENCE_RE.split(cleaned)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= int(min_chars)]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def lexical_similarity(a: str, b: str) -> float:
    a_tokens = set(content_tokens(a))
    b_tokens = set(content_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return 0.6 + 0.4 * overlap / union if overlap else 0.0


def extract_evidence_sentences(prompt_context: str, *, max_items: int = 200) -> list[str]:
    context = str(prompt_context or "")
    evidence: list[str] = []
    seen: set[str] = set()

    for match in KG_EDGE_RE.finditer(context):
        src = normalize_text(match.group(1))
        dst = normalize_text(match.group(2))
        rel = normalize_text(match.group(3).replace("_", " "))
        item = f"{src} {rel} {dst}"
        key = item.lower()
        if len(item) >= 8 and key not in seen:
            seen.add(key)
            evidence.append(item)

    for line in context.splitlines():
        item = normalize_text(line)
        if len(item) < 15 or item.startswith(("##", "===", "---")):
            continue
        lower = item.lower()
        if any(marker in lower for marker in ["sl score=", "feature activation=", "hypothesis", "rationale", "evidence", "pair", "gene"]):
            key = item.lower()[:160]
            if key not in seen:
                seen.add(key)
                evidence.append(item)

    return evidence[: int(max_items)]
