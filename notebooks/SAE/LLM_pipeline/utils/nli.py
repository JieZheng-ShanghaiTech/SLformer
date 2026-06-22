"""Local NLI utilities for comparing SAE interpretations with curated text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]


def is_nli_config(config_path: Path) -> bool:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "id2label" not in config:
        return False
    labels = [str(label).lower() for label in config["id2label"].values()]
    return any("entail" in label for label in labels) and any("contrad" in label for label in labels)


def nli_model_path(model_root: Path) -> Path:
    model_root = Path(model_root)
    if (model_root / "config.json").exists() and is_nli_config(model_root / "config.json"):
        return model_root
    for config_path in sorted(model_root.glob("*/config.json")):
        if is_nli_config(config_path):
            return config_path.parent
    return next(
        path
        for path in sorted(model_root.iterdir())
        if path.is_dir() and ("nli" in path.name.lower() or "mnli" in path.name.lower())
    )


def nli_label_columns(id2label: dict[int, str]) -> dict[str, int]:
    labels = {str(label).lower(): int(index) for index, label in id2label.items()}
    return {
        "contradiction": next(index for label, index in labels.items() if "contrad" in label),
        "neutral": next(index for label, index in labels.items() if "neutral" in label),
        "entailment": next(index for label, index in labels.items() if "entail" in label),
    }


def classify_nli_hypothesis(text: str) -> str:
    sentence = str(text).lower()
    if any(token in sentence for token in ["feature ", "z0=", "dot_z", "c_star", "sae", "model's", "model ", "latent", "atom pattern", "validation_status", "zero_control"]):
        return "model_internal_claim"
    if any(token in sentence for token in ["wet_lab_alignment", "readout prediction", "counterfactual prediction", "should reveal", "treated with", "inhibitor"]):
        return "experimental_prediction"
    if any(token in sentence for token in ["limitations", "cannot prove", "cannot rule out"]):
        return "limitation"
    return "biological_inference"


def score_nli_pairs(
    *,
    premises: Sequence[str],
    hypotheses: Sequence[str],
    model_path: Path,
    device: str,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
    label_columns = nli_label_columns(model.config.id2label)
    rows = []

    for start in range(0, len(hypotheses), int(batch_size)):
        batch_premises = list(premises[start : start + int(batch_size)])
        batch_hypotheses = list(hypotheses[start : start + int(batch_size)])
        encoded = tokenizer(
            batch_premises,
            batch_hypotheses,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probabilities = model(**encoded).logits.softmax(dim=-1).detach().cpu().numpy()
        for premise, hypothesis, probability in zip(batch_premises, batch_hypotheses, probabilities):
            rows.append(
                {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "claim_type": classify_nli_hypothesis(hypothesis),
                    "contradiction": float(probability[label_columns["contradiction"]]),
                    "neutral": float(probability[label_columns["neutral"]]),
                    "entailment": float(probability[label_columns["entailment"]]),
                }
            )

    table = pd.DataFrame(rows)
    table["prediction"] = table[["contradiction", "neutral", "entailment"]].idxmax(axis=1)
    return table


def compare_interpretation_to_groundtruth(
    *,
    final_text: str,
    groundtruth_text: str,
    model_path: Path,
    device: str,
    batch_size: int = 8,
    max_length: int = 512,
) -> pd.DataFrame:
    final_sentences = split_sentences(final_text)
    groundtruth_sentences = split_sentences(groundtruth_text)

    support = score_nli_pairs(
        premises=[groundtruth_text] * len(final_sentences),
        hypotheses=final_sentences,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    support.insert(0, "comparison", "sae_claim_supported_by_groundtruth")

    coverage = score_nli_pairs(
        premises=[final_text] * len(groundtruth_sentences),
        hypotheses=groundtruth_sentences,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    coverage.insert(0, "comparison", "groundtruth_covered_by_sae")

    return pd.concat([support, coverage], ignore_index=True)
