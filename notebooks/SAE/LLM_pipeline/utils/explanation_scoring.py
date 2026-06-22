"""Feature coverage scoring for SAE interpretation text."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from .eval_common import content_tokens, cosine, extract_mechanism_section, lexical_similarity, normalize_text, strip_citations, tokens


def normalize_feature(feature: str) -> str:
    return normalize_text(feature).lower()


def split_feature_field(feature_field: str) -> list[str]:
    features: list[str] = []
    for line in str(feature_field or "").splitlines():
        for chunk in re.split(r"[;,]", line):
            feature = normalize_feature(chunk)
            if feature:
                features.append(feature)
    return features


def feature_in_text(text: str, feature: str, *, max_span: int = 80) -> bool:
    text_norm = normalize_feature(text)
    feature_norm = normalize_feature(feature)
    feature_tokens = tokens(feature_norm)
    if not text_norm or not feature_tokens:
        return False
    if len(feature_tokens) == 1:
        return re.search(rf"\b{re.escape(feature_tokens[0])}\b", text_norm) is not None
    if len(feature_tokens) == 2:
        a, b = (re.escape(feature_tokens[0]), re.escape(feature_tokens[1]))
        return (
            re.search(rf"\b{a}\b.{{0,{max_span}}}\b{b}\b", text_norm) is not None
            or re.search(rf"\b{b}\b.{{0,{max_span}}}\b{a}\b", text_norm) is not None
        )
    return all(re.search(rf"\b{re.escape(token)}\b", text_norm) is not None for token in feature_tokens)


def extract_lexicon_features(text: str, feature_pool: Sequence[str], *, max_span: int = 80) -> list[str]:
    hits = []
    for feature in feature_pool:
        normalized = normalize_feature(feature)
        if normalized and feature_in_text(text, normalized, max_span=max_span):
            hits.append(normalized)
    return sorted(set(hits))


def extract_explicit_key_phrases(text: str) -> list[str]:
    patterns = [
        r"[Kk]ey\s+process\s+phrases?[:\s]+([^\n]+)",
        r"[Kk]ey\s+phrases?[:\s]+([^\n]+)",
        r"[Pp]rocess\s+phrases?[:\s]+([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text or ""))
        if match is None:
            continue
        line = match.group(1).strip()
        parts = line.split(";") if ";" in line else line.split(",")
        phrases = []
        for part in parts:
            phrase = re.sub(r"\s*\([^)]*\)\s*$", "", part.strip()).rstrip(".")
            phrase = normalize_feature(phrase)
            if phrase and len(phrase) >= 3 and len(phrase.split()) <= 8:
                phrases.append(phrase)
        if phrases:
            return phrases
    return []


BIOPHRASE_META_TOKENS = frozenset(
    {
        "slformer",
        "sae",
        "geometry",
        "embedding",
        "embed",
        "latent",
        "latents",
        "feature",
        "features",
        "score",
        "scores",
        "model",
        "internal",
        "evidence",
        "prediction",
        "predicted",
        "direction",
        "directions",
        "condition",
        "pair",
        "pairs",
        "active",
        "activation",
        "activations",
        "baseline",
        "signal",
        "signals",
        "state",
        "states",
        "z0",
        "dot",
        "delta",
        "star",
        "norm",
        "true",
        "random",
        "zero",
    }
)


BIOPHRASE_ANCHOR_TOKENS = frozenset(
    {
        "apoptosis",
        "apoptotic",
        "cell",
        "checkpoint",
        "chromatin",
        "cycle",
        "damage",
        "death",
        "deficiency",
        "deficient",
        "defects",
        "dna",
        "g1",
        "g2",
        "growth",
        "hif",
        "homologous",
        "hr",
        "inhibition",
        "inhibitor",
        "blockade",
        "loss",
        "mapk",
        "mitotic",
        "mitosis",
        "mtor",
        "pathway",
        "p53",
        "pi3k",
        "proliferation",
        "proteasome",
        "proteostasis",
        "production",
        "repair",
        "replication",
        "response",
        "ros",
        "signaling",
        "spindle",
        "splicing",
        "stress",
        "translation",
        "vulnerability",
    }
)


def extract_biomedical_ngram_candidates(text: str, *, max_phrases: int = 240) -> list[str]:
    value = re.sub(r"[-/]", " ", str(text or ""))
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", value)
    norm_tokens = [token.lower() for token in raw_tokens]
    candidates: list[str] = []
    seen: set[str] = set()

    for ngram_size in range(2, 6):
        for start in range(0, max(0, len(raw_tokens) - ngram_size + 1)):
            phrase_tokens = raw_tokens[start : start + ngram_size]
            phrase_norm = norm_tokens[start : start + ngram_size]
            content = [token for token in phrase_norm if token not in BIOPHRASE_META_TOKENS]
            if len(content) < 2:
                continue
            if not any(token in BIOPHRASE_ANCHOR_TOKENS for token in content):
                continue
            if any(token in BIOPHRASE_META_TOKENS for token in phrase_norm):
                continue
            if sum(token in BIOPHRASE_META_TOKENS for token in phrase_norm) > 1:
                continue
            if phrase_norm[0] in SCORING_BOUNDARY_WORDS or phrase_norm[-1] in SCORING_BOUNDARY_WORDS:
                continue
            phrase = normalize_feature(" ".join(phrase_tokens))
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                candidates.append(phrase)
            if len(candidates) >= int(max_phrases):
                return candidates
    return candidates


SCORING_BOUNDARY_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "alone",
        "not",
        "of",
        "on",
        "or",
        "so",
        "than",
        "that",
        "the",
        "this",
        "to",
        "through",
        "yet",
        "while",
        "with",
        "without",
    }
)


def primary_scoring_text(text: str, *, scope: str = "section2") -> str:
    if str(scope).strip().lower() in {"full", "all", "entire"}:
        return str(text or "")
    return extract_mechanism_section(text)


def filter_keyphrase_candidates(candidates: Sequence[str], ground_truth_features: Sequence[str]) -> list[str]:
    gt_norm = [normalize_feature(feature) for feature in ground_truth_features if normalize_feature(feature)]
    gene_symbol_re = re.compile(r"^[A-Z0-9][A-Z0-9_\-]{1,9}$")
    stop = {"none", "null", "na", "n/a", "unknown", "is", "are", "the", "a", "an", "and", "or", "of", "to"}

    related_out: list[str] = []
    other_out: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        phrase = normalize_text(raw).strip(".,;: ")
        if not phrase:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        phrase_tokens = set(tokens(key))
        has_meta = bool(phrase_tokens & BIOPHRASE_META_TOKENS)
        has_bio_anchor = bool(phrase_tokens & BIOPHRASE_ANCHOR_TOKENS)
        related = any(key in gt or gt in key for gt in gt_norm)
        strong_bio_anchors = BIOPHRASE_ANCHOR_TOKENS - {"pathway", "response", "signaling", "vulnerability"}
        if has_meta and not (phrase_tokens & strong_bio_anchors):
            continue
        if not related and not has_bio_anchor:
            continue
        if not related:
            if len(phrase) < 3 or key in stop or not re.search(r"[A-Za-z]", phrase) or gene_symbol_re.fullmatch(phrase):
                continue
        (related_out if related else other_out).append(phrase)
    return related_out + other_out


@lru_cache(maxsize=2)
def load_token_cls_pipeline(model_path: str, device: str):
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    device_arg = -1
    if str(device).startswith("cuda"):
        device_arg = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=Path(model_path).exists(), use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=Path(model_path).exists())
    return pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device_arg)


def extract_keyphrases_token_cls(text: str, *, model_path: str, device: str = "cpu", max_phrases: int = 800) -> list[str]:
    pipe = load_token_cls_pipeline(str(model_path), str(device))
    predictions = pipe(str(text or ""))
    phrases = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        phrase = str(prediction.get("word") or "").strip()
        if not phrase and prediction.get("start") is not None and prediction.get("end") is not None:
            phrase = str(text or "")[int(prediction["start"]) : int(prediction["end"])].strip()
        if phrase:
            phrases.append(phrase)

    out: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        cleaned = normalize_text(phrase).strip(".,;: ")
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out[: int(max_phrases)]


@lru_cache(maxsize=2)
def load_hf_encoder(model_path: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=Path(model_path).exists())
    model = AutoModel.from_pretrained(model_path, local_files_only=Path(model_path).exists()).to(device).eval()
    return tokenizer, model, device


def embed_texts_hf(
    texts: Sequence[str],
    *,
    model_path: str,
    device: str = "cpu",
    max_length: int = 64,
    batch_size: int = 16,
) -> list[list[float]]:
    import torch

    tokenizer, model, device_name = load_hf_encoder(str(model_path), str(device))
    vectors: list[list[float]] = []
    for start in range(0, len(texts), int(batch_size)):
        batch = [str(text or "") for text in texts[start : start + int(batch_size)]]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")
        encoded = {key: value.to(device_name) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            mask = encoded["attention_mask"].unsqueeze(-1).type_as(hidden)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.extend(pooled.detach().cpu().tolist())
    return vectors


def adjusted_similarity(feature: str, phrase: str, cosine_sim: float) -> float:
    feature_tokens = content_tokens(feature)
    phrase_tokens = content_tokens(phrase)
    if not feature_tokens or not phrase_tokens:
        return 0.0
    if set(feature_tokens) & set(phrase_tokens):
        overlap_factor = 1.0
    elif cosine_sim >= 0.75:
        overlap_factor = 0.70
    elif cosine_sim >= 0.60:
        overlap_factor = 0.50
    else:
        overlap_factor = 0.35
    length_factor = 0.65 + 0.35 * min(1.0, len(phrase_tokens) / max(1, len(feature_tokens)))
    return max(0.0, min(0.9, float(cosine_sim) * overlap_factor * length_factor))


def _score_from_vectors(
    *,
    gt: Sequence[str],
    candidates: Sequence[str],
    gt_vecs: Sequence[Sequence[float]],
    cand_vecs: Sequence[Sequence[float]],
    similarity_threshold: float,
    similarity_mode: str,
    soft_weight: float,
) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    covered: list[str] = []
    missed: list[str] = []
    best_sims: list[float] = []

    for feature, feature_vec in zip(gt, gt_vecs):
        best_similarity = -math.inf
        best_phrase = None
        for phrase, phrase_vec in zip(candidates, cand_vecs):
            raw = cosine(feature_vec, phrase_vec)
            similarity = adjusted_similarity(feature, phrase, raw) if similarity_mode == "adjusted" else float(raw)
            if similarity > best_similarity:
                best_similarity = similarity
                best_phrase = phrase
        covered_hit = bool(best_phrase is not None and best_similarity >= similarity_threshold)
        matches.append(
            {
                "feature": feature,
                "best_phrase": best_phrase,
                "best_similarity": float(best_similarity),
                "covered": covered_hit,
            }
        )
        best_sims.append(float(best_similarity))
        (covered if covered_hit else missed).append(feature)

    aligned = 0
    cand_best_sims: list[float] = []
    for phrase, phrase_vec in zip(candidates, cand_vecs):
        best_similarity = -math.inf
        best_feature = None
        for feature, feature_vec in zip(gt, gt_vecs):
            raw = cosine(phrase_vec, feature_vec)
            similarity = adjusted_similarity(feature, phrase, raw) if similarity_mode == "adjusted" else float(raw)
            if similarity > best_similarity:
                best_similarity = similarity
                best_feature = feature
        cand_best_sims.append(float(best_similarity))
        if best_feature is not None and best_similarity >= similarity_threshold:
            aligned += 1

    p_soft = float(sum(cand_best_sims) / len(cand_best_sims)) if cand_best_sims else 0.0
    r_soft = float(sum(best_sims) / len(best_sims)) if best_sims else 0.0
    f1_soft = 2.0 * p_soft * r_soft / (p_soft + r_soft) if p_soft + r_soft else 0.0
    p_hard = float(aligned) / len(candidates) if candidates else 0.0
    r_hard = float(len(covered)) / len(gt) if gt else 0.0
    f1_hard = 2.0 * p_hard * r_hard / (p_hard + r_hard) if p_hard + r_hard else 0.0

    precision = soft_weight * p_soft + (1.0 - soft_weight) * p_hard
    recall = soft_weight * r_soft + (1.0 - soft_weight) * r_hard
    f1 = soft_weight * f1_soft + (1.0 - soft_weight) * f1_hard

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "soft": {"precision": float(p_soft), "recall": float(r_soft), "f1": float(f1_soft)},
        "hard": {"precision": float(p_hard), "recall": float(r_hard), "f1": float(f1_hard)},
        "mean_best_similarity": float(sum(best_sims) / len(best_sims)) if best_sims else None,
        "min_best_similarity": float(min(best_sims)) if best_sims else None,
        "ground_truth": {"total": len(gt), "covered": covered, "missed": missed, "matches": matches},
        "candidates": {"total": len(candidates), "aligned": int(aligned), "phrases_preview": list(candidates[:50])},
    }


def _prf1_by_lexical_overlap(
    *,
    ground_truth_features: Sequence[str],
    candidates: Sequence[str],
    similarity_threshold: float,
    soft_weight: float,
) -> dict[str, object]:
    gt = [normalize_feature(feature) for feature in ground_truth_features if normalize_feature(feature)]

    def lexical_score(feature: str, phrase: str) -> float:
        return lexical_similarity(feature, phrase)

    matches: list[dict[str, object]] = []
    covered: list[str] = []
    missed: list[str] = []
    best_sims: list[float] = []
    for feature in gt:
        best_phrase = None
        best_similarity = -math.inf
        for phrase in candidates:
            similarity = lexical_score(feature, phrase)
            if similarity > best_similarity:
                best_similarity = similarity
                best_phrase = phrase
        hit = bool(best_phrase is not None and best_similarity >= similarity_threshold)
        matches.append({"feature": feature, "best_phrase": best_phrase, "best_similarity": float(best_similarity), "covered": hit})
        best_sims.append(float(best_similarity))
        (covered if hit else missed).append(feature)

    aligned = 0
    cand_best_sims: list[float] = []
    for phrase in candidates:
        best_similarity = max((lexical_score(feature, phrase) for feature in gt), default=0.0)
        cand_best_sims.append(float(best_similarity))
        if best_similarity >= similarity_threshold:
            aligned += 1

    p_soft = float(sum(cand_best_sims) / len(cand_best_sims)) if cand_best_sims else 0.0
    r_soft = float(sum(best_sims) / len(best_sims)) if best_sims else 0.0
    f1_soft = 2.0 * p_soft * r_soft / (p_soft + r_soft) if p_soft + r_soft else 0.0
    p_hard = float(aligned) / len(candidates) if candidates else 0.0
    r_hard = float(len(covered)) / len(gt) if gt else 0.0
    f1_hard = 2.0 * p_hard * r_hard / (p_hard + r_hard) if p_hard + r_hard else 0.0
    precision = soft_weight * p_soft + (1.0 - soft_weight) * p_hard
    recall = soft_weight * r_soft + (1.0 - soft_weight) * r_hard
    f1 = soft_weight * f1_soft + (1.0 - soft_weight) * f1_hard

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "soft": {"precision": float(p_soft), "recall": float(r_soft), "f1": float(f1_soft)},
        "hard": {"precision": float(p_hard), "recall": float(r_hard), "f1": float(f1_hard)},
        "mean_best_similarity": float(sum(best_sims) / len(best_sims)) if best_sims else None,
        "min_best_similarity": float(min(best_sims)) if best_sims else None,
        "ground_truth": {"total": len(gt), "covered": covered, "missed": missed, "matches": matches},
        "candidates": {"total": len(candidates), "aligned": int(aligned), "phrases_preview": list(candidates[:50])},
    }


def feature_embedding_prf1_by_coverage(
    *,
    ground_truth_features: Sequence[str],
    text: str,
    model_path: str | None = None,
    device: str = "cpu",
    max_length: int = 64,
    batch_size: int = 16,
    similarity_threshold: float = 0.6,
    similarity_threshold_adjusted: float = 0.4,
    similarity_mode: str = "adjusted",
    score_scope: str = "section2",
    candidate_backend: str = "lexicon",
    tokencls_model_path: str | None = None,
    max_candidates: int = 800,
    soft_weight: float = 0.5,
) -> dict[str, object]:
    gt = sorted(set(normalize_feature(feature) for feature in ground_truth_features if normalize_feature(feature)))
    scoring_text = strip_citations(primary_scoring_text(text, scope=score_scope))
    mode = str(similarity_mode).strip().lower()
    threshold = float(similarity_threshold_adjusted if mode == "adjusted" else similarity_threshold)
    backend = str(candidate_backend).strip().lower()

    explicit_phrases = extract_explicit_key_phrases(scoring_text)
    ngram_candidates = extract_biomedical_ngram_candidates(scoring_text, max_phrases=min(int(max_candidates), 240))
    if backend == "token_cls":
        model_candidates = extract_keyphrases_token_cls(
            scoring_text,
            model_path=str(tokencls_model_path),
            device=device,
            max_phrases=int(max_candidates),
        )
    else:
        model_candidates = []
    lexicon_hits = extract_lexicon_features(scoring_text, gt)
    candidates = explicit_phrases + model_candidates + ngram_candidates + lexicon_hits
    candidates = filter_keyphrase_candidates(candidates, gt)
    debug = {
        "backend": backend,
        "mode": mode,
        "threshold": threshold,
        "threshold_raw": float(similarity_threshold),
        "feature_embed_model_path": str(model_path) if model_path else None,
        "tokencls_model_path": str(tokencls_model_path) if tokencls_model_path else None,
        "explicit_phrases": explicit_phrases,
        "ngram_candidate_count": len(ngram_candidates),
        "model_candidate_count": len(model_candidates),
        "lexicon_hit_count": len(lexicon_hits),
        "candidate_mode": "symmetric",
    }

    if not gt:
        return {
            "threshold": threshold,
            "threshold_raw": float(similarity_threshold),
            "similarity_mode": mode,
            "precision": None,
            "recall": None,
            "f1": None,
            "ground_truth": {"total": 0},
            "candidates": {"total": len(candidates), "backend": backend, "debug": debug},
        }

    if not candidates:
        return {
            "threshold": threshold,
            "threshold_raw": float(similarity_threshold),
            "similarity_mode": mode,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "soft": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "hard": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "topk": {
                "p50": {"k": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "p75": {"k": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            },
            "ground_truth": {
                "total": len(gt),
                "covered": [],
                "missed": list(gt),
                "matches": [{"feature": feature, "best_phrase": None, "best_similarity": -math.inf, "covered": False} for feature in gt],
            },
            "candidates": {"total": 0, "aligned": 0, "backend": backend, "debug": debug, "phrases_preview": []},
        }

    if model_path:
        gt_vecs = embed_texts_hf(gt, model_path=model_path, device=device, max_length=max_length, batch_size=batch_size)
        cand_vecs = embed_texts_hf(candidates, model_path=model_path, device=device, max_length=max_length, batch_size=batch_size)
        result = _score_from_vectors(
            gt=gt,
            candidates=candidates,
            gt_vecs=gt_vecs,
            cand_vecs=cand_vecs,
            similarity_threshold=threshold,
            similarity_mode=mode,
            soft_weight=float(soft_weight),
        )
    else:
        result = _prf1_by_lexical_overlap(
            ground_truth_features=gt,
            candidates=candidates,
            similarity_threshold=threshold,
            soft_weight=float(soft_weight),
        )

    result["threshold"] = threshold
    result["threshold_raw"] = float(similarity_threshold)
    result["similarity_mode"] = mode
    result["soft_weight"] = float(soft_weight)
    result["topk"] = {
        "p50": {"k": max(1, int(0.5 * len(candidates))), "precision": result["precision"], "recall": result["recall"], "f1": result["f1"]},
        "p75": {"k": max(1, int(0.75 * len(candidates))), "precision": result["precision"], "recall": result["recall"], "f1": result["f1"]},
    }
    result["candidates"]["backend"] = backend
    result["candidates"]["debug"] = debug
    return result
