"""Small shared utilities for SAE notebooks and analysis code."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PairTextMap = dict[tuple[str, str], str]


def find_repo_root(start: Path) -> Path:
    path = start.resolve()
    for candidate in (path, *path.parents):
        if (candidate / "src" / "SAE").exists():
            return candidate
    raise FileNotFoundError(f"Could not identify repository root from {path}")


def find_pair_index(meta: pd.DataFrame, primary_gene: str, partner_gene: str, cancer: str) -> int:
    primary = meta["primary_gene"].astype(str)
    partner = meta["partner_gene"].astype(str)
    cancer_values = meta["cancer"].astype(str)
    mask = ((primary == primary_gene) & (partner == partner_gene)) | ((primary == partner_gene) & (partner == primary_gene))
    mask = mask & (cancer_values == str(cancer))
    return int(np.flatnonzero(mask.to_numpy())[0])


def sanitize_groundtruth_text(text: str) -> str:
    cleaned = re.sub(r"\[\^\d+\^\](?:\s*\[\d+\])?", "", str(text))
    cleaned = re.sub(r"\[\^\d+\]", "", cleaned)
    cleaned = re.sub(r"(?<![A-Za-z0-9])\[\d+\](?!:)", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\[\d+\]:.*$", "", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_groundtruth_texts(groundtruth_dir: Path) -> PairTextMap:
    pair_texts: PairTextMap = {}
    for pair_folder in sorted(Path(groundtruth_dir).glob("*-*")):
        gene_a, gene_b = pair_folder.name.split("-", 1)
        text = sanitize_groundtruth_text((pair_folder / "explanation.txt").read_text(encoding="utf-8"))
        pair_texts[(gene_a, gene_b)] = text
        pair_texts[(gene_b, gene_a)] = text
    return pair_texts


def pair_key(meta: pd.DataFrame, index: int) -> tuple[str, str]:
    return str(meta.loc[int(index), "primary_gene"]), str(meta.loc[int(index), "partner_gene"])


def pair_name(meta: pd.DataFrame, index: int) -> str:
    gene_a, gene_b = pair_key(meta, index)
    return f"{gene_a}-{gene_b}"


def same_unordered_pair(meta: pd.DataFrame, index_a: int, index_b: int) -> bool:
    return set(pair_key(meta, index_a)) == set(pair_key(meta, index_b))


def sample_name(meta: pd.DataFrame, index: int) -> str:
    return f"{str(meta.loc[int(index), 'cancer'])}:{pair_name(meta, index)}"


def unit_vector(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return x / (np.linalg.norm(x) + 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def short_text(text: str, n_chars: int) -> str:
    return " ".join(sanitize_groundtruth_text(text).split())[: int(n_chars)]


def format_exemplars(
    *,
    indices: Sequence[int],
    feature: int,
    Z: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    pair_texts: PairTextMap,
    include_activation: bool,
    text_chars: int,
) -> str:
    lines = []
    for number, index in enumerate(indices, start=1):
        score = float(y[int(index)])
        text = short_text(pair_texts[pair_key(meta, int(index))], text_chars)
        activation = float(abs(Z[int(index), int(feature)]))
        evidence_role = "positive exemplar" if activation > 0.0 else "zero-control exemplar"
        prefix = f"{number}. {sample_name(meta, int(index))} | SL score={score:.4f}"
        if include_activation:
            prefix += f" | feature activation={activation:.6f} | {evidence_role}"
        lines.append(prefix + "\n" + text)
    return "\n\n".join(lines)


def format_pair_evidence(
    *,
    indices: Sequence[int],
    feature: int,
    Z: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    include_activation: bool,
) -> str:
    lines = []
    for number, index in enumerate(indices, start=1):
        score = float(y[int(index)])
        activation = float(abs(Z[int(index), int(feature)]))
        evidence_role = "positive exemplar" if activation > 0.0 else "zero-control exemplar"
        prefix = f"{number}. {sample_name(meta, int(index))} | SL score={score:.4f}"
        if include_activation:
            prefix += f" | feature activation={activation:.6f} | {evidence_role}"
        lines.append(prefix)
    return "\n".join(lines)
