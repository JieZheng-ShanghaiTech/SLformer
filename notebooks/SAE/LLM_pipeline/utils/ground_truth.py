"""Ground-truth loaders for quick SAE interpretation validation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .eval_payload import score_text_metrics
from .explanation_scoring import split_feature_field
from .general import sanitize_groundtruth_text


NORMALIZED_COLUMNS = [
    "gene_a",
    "gene_b",
    "primary_gene",
    "explanation",
    "reference",
    "pubmed_id",
    "source",
    "label",
    "corrected_explanation",
    "important_features",
    "note",
]


def _find_header_row(raw: pd.DataFrame) -> int:
    for index, row in raw.iterrows():
        values = [str(value).strip().lower() for value in row.tolist()]
        if "gene a 🧬" in values or "gene a" in values:
            return int(index)
    return 0


def _normalize_sheet_columns(table: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "gene A 🧬": "gene_a",
        "gene A": "gene_a",
        "gene_a": "gene_a",
        "geneB": "gene_b",
        "gene B 🧬": "gene_b",
        "gene B": "gene_b",
        "gene_b": "gene_b",
        "genes more likely to mutate": "primary_gene",
        "primary_gene": "primary_gene",
        "answer": "explanation",
        "explanation": "explanation",
        "reference": "reference",
        "pubmed_id": "pubmed_id",
        "source": "source",
        "the answer could be a label? (0/1)": "label",
        "label": "label",
        "if need modify": "corrected_explanation",
        "important feature (gene functions related to cell death)": "important_features",
        "important_features": "important_features",
        "note": "note",
    }
    normalized_names = []
    for column in table.columns:
        key = str(column).strip()
        normalized_names.append(rename[key] if key in rename else key)
    table = table.copy()
    table.columns = normalized_names
    for column in NORMALIZED_COLUMNS:
        if column not in table.columns:
            table[column] = ""
    return table[NORMALIZED_COLUMNS]


def load_nexleth_ground_truth(path: str | Path, *, accepted_labels: Sequence[int] = (1,)) -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_csv(path, header=None)
    header_row = _find_header_row(raw)
    table = pd.read_csv(path, header=header_row)
    table = _normalize_sheet_columns(table)
    table = table[table["gene_a"].notna() & table["gene_b"].notna()].copy()
    table["gene_a"] = table["gene_a"].astype(str).str.strip()
    table["gene_b"] = table["gene_b"].astype(str).str.strip()
    table = table[(table["gene_a"] != "") & (table["gene_b"] != "")]

    labels = pd.to_numeric(table["label"], errors="coerce")
    table = table[labels.isin([int(label) for label in accepted_labels])].copy()
    table["label"] = labels[table.index].astype(int)
    corrected = table["corrected_explanation"]
    corrected_text = corrected.fillna("").astype(str).str.strip()
    use_corrected = corrected.notna() & (corrected_text != "")
    table["ground_truth_explanation"] = table["explanation"]
    table.loc[use_corrected, "ground_truth_explanation"] = corrected.loc[use_corrected]
    table["ground_truth_explanation"] = table["ground_truth_explanation"].fillna("").map(
        sanitize_groundtruth_text
    )
    table["ground_truth_features"] = table["important_features"].map(split_feature_field)
    return table.reset_index(drop=True)


def find_ground_truth_pair(table: pd.DataFrame, gene_a: str, gene_b: str) -> pd.Series:
    a = str(gene_a).strip().upper()
    b = str(gene_b).strip().upper()
    left = table["gene_a"].astype(str).str.upper()
    right = table["gene_b"].astype(str).str.upper()
    mask = ((left == a) & (right == b)) | ((left == b) & (right == a))
    return table.loc[mask].iloc[0]


def score_text_against_nexleth(
    *,
    text: str,
    gene_a: str,
    gene_b: str,
    ground_truth_csv: str | Path,
    prompt_context: str = "",
) -> dict[str, object]:
    table = load_nexleth_ground_truth(ground_truth_csv)
    row = find_ground_truth_pair(table, gene_a, gene_b)
    return score_text_metrics(
        ground_truth_features=row["ground_truth_features"],
        ground_truth_explanation=row["ground_truth_explanation"],
        text=text,
        prompt_context=prompt_context,
    )
