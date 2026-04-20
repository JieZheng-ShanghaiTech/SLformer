from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


_DIM_RE = re.compile(r"\bdim_(\d{1,4})\b", flags=re.IGNORECASE)

# Fold-change patterns seen in some LLM outputs.
_FOLD_X_RE = re.compile(r"\bx\s*(\d+(?:\.\d+)?)\b", flags=re.IGNORECASE)
_FOLD_WORD_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*[- ]?fold\b", flags=re.IGNORECASE)

_TARGET_CTX_RE = re.compile(r"\*\*Target Analysis Context:\*\*\s*(.+)")


def _safe_div(n: float, d: float) -> float:
    if d == 0:
        return float("nan")
    return n / d


def extract_target_context_code(prompt: str) -> str:
    """Extract the target context code (e.g., 'LUAD') from the generated prompt."""
    m = _TARGET_CTX_RE.search(prompt or "")
    return (m.group(1).strip() if m else "")


def extract_dim_ids(text: str) -> Set[int]:
    """Extract integer dim ids from tokens like 'dim_24'."""
    out: Set[int] = set()
    for m in _DIM_RE.finditer(text or ""):
        try:
            out.add(int(m.group(1)))
        except Exception:
            continue
    return out


def extract_prompt_dim_ids(prompt: str, *, context: Optional[str] = None) -> Set[int]:
    """Extract dim ids mentioned in the prompt.

    If context is provided, filters to lines that look like per-context summaries
    for that context code.
    """
    dims: Set[int] = set()
    ctx = (context or "").strip()

    for raw_line in (prompt or "").splitlines():
        line = raw_line.strip()
        if "Top dimensions" not in line:
            continue
        if ctx:
            # The generator formats as: "  CESC (SL-Score: ...): Top dimensions - ..."
            if not line.startswith(ctx):
                continue
        dims |= extract_dim_ids(line)

    return dims


def extract_prompt_fold_tokens(prompt: str) -> Set[str]:
    """Extract fold tokens present in the prompt, e.g. 'x2.9'."""
    toks: Set[str] = set()
    for m in _FOLD_X_RE.finditer(prompt or ""):
        toks.add(f"x{m.group(1)}")
    for m in _FOLD_WORD_RE.finditer(prompt or ""):
        toks.add(f"{m.group(1)}-fold")
    return toks


def extract_output_fold_tokens(text: str) -> Set[str]:
    toks: Set[str] = set()
    for m in _FOLD_X_RE.finditer(text or ""):
        toks.add(f"x{m.group(1)}")
    for m in _FOLD_WORD_RE.finditer(text or ""):
        toks.add(f"{m.group(1)}-fold")
    return toks


@dataclass(frozen=True)
class CoverageResult:
    expected: int
    mentioned: int
    covered: int
    recall: float
    covered_dims: Tuple[int, ...] = ()
    missing_dims: Tuple[int, ...] = ()


@dataclass(frozen=True)
class HallucinationResult:
    mentioned: int
    hallucinated: int
    hallucination_rate: float
    hallucinated_dims: Tuple[int, ...] = ()


def feature_coverage_recall(*, prompt: str, output: str) -> CoverageResult:
    """Compute 'feature coverage recall' for embedding-dimension features.

    Definition here (KG_LLM_XAI-style, but prompt-driven):
    - Expected features: dim ids reported for the target context in the prompt.
    - Covered: those expected dim ids that appear in the output text.
    - Recall: covered / expected.

    Notes:
    - If the prompt contains no dims for the target context, recall is NaN.
    """
    ctx = extract_target_context_code(prompt)
    expected_dims = extract_prompt_dim_ids(prompt, context=ctx)
    mentioned_dims = extract_dim_ids(output)

    covered = sorted(expected_dims & mentioned_dims)
    missing = sorted(expected_dims - mentioned_dims)

    recall = float("nan")
    if expected_dims:
        recall = _safe_div(float(len(covered)), float(len(expected_dims)))

    return CoverageResult(
        expected=len(expected_dims),
        mentioned=len(mentioned_dims),
        covered=len(covered),
        recall=recall,
        covered_dims=tuple(covered),
        missing_dims=tuple(missing),
    )


def hallucination_score(*, prompt: str, output: str) -> HallucinationResult:
    """Compute a simple hallucination score for embedding-dimension claims.

    Definition:
    - Any dim id mentioned in output that does NOT appear anywhere in the prompt
      is treated as hallucinated (unsupported by provided evidence).
    - hallucination_rate = hallucinated / mentioned (0 if mentioned==0)

    This targets the common failure mode where a model invents dim ids.
    """
    prompt_dims_all = extract_prompt_dim_ids(prompt, context=None)
    mentioned_dims = extract_dim_ids(output)

    hallucinated = sorted(mentioned_dims - prompt_dims_all)
    rate = 0.0
    if mentioned_dims:
        rate = float(len(hallucinated)) / float(len(mentioned_dims))

    return HallucinationResult(
        mentioned=len(mentioned_dims),
        hallucinated=len(hallucinated),
        hallucination_rate=rate,
        hallucinated_dims=tuple(hallucinated),
    )


def fold_hallucination_rate(*, prompt: str, output: str) -> float:
    """Rate of fold-change tokens in output that are not present in the prompt.

    This flags unsupported claims like "100-fold" when the prompt doesn't contain
    that fold token. It is intentionally conservative: derived fold-changes may be
    correct mathematically but are treated as unsupported unless the prompt states
    the fold token explicitly.

    Returns NaN if output contains no fold tokens.
    """
    out_toks = extract_output_fold_tokens(output)
    if not out_toks:
        return float("nan")
    prompt_toks = extract_prompt_fold_tokens(prompt)
    unsupported = out_toks - prompt_toks
    return _safe_div(float(len(unsupported)), float(len(out_toks)))


def evaluate_explanation(
    *,
    prompt: str,
    output: str,
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a compact set of evaluation metrics.

    Returns a JSON-serializable dict.
    """
    cov = feature_coverage_recall(prompt=prompt, output=output)
    hallu = hallucination_score(prompt=prompt, output=output)
    fold_hallu = fold_hallucination_rate(prompt=prompt, output=output)

    results: Dict[str, Any] = {
        "target_context": extract_target_context_code(prompt),
        "feature_coverage_recall": {
            "expected_dims": cov.expected,
            "mentioned_dims": cov.mentioned,
            "covered_dims": cov.covered,
            "recall": cov.recall,
        },
        "hallucination": {
            "mentioned_dims": hallu.mentioned,
            "hallucinated_dims": hallu.hallucinated,
            "hallucination_rate": hallu.hallucination_rate,
        },
        "fold_hallucination_rate": fold_hallu,
    }

    # Optional similarity vs ground truth (if available) is computed elsewhere
    # (prompt_api/enrichment.py) to avoid heavy model imports here.
    if ground_truth:
        results["ground_truth_available"] = True
    else:
        results["ground_truth_available"] = False

    return results
