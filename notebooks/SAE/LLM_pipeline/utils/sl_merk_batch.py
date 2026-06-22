"""Batch utilities for SL-MERK SAE pair explanations."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from SAE.LLM_pipeline.utils.eval_payload import score_text_metrics
from SAE.LLM_pipeline.utils.explainer import (
    activation_feature_tables,
    build_explainer_prompt,
    extract_explainer_fields,
    rank_candidate_features,
    target_relevance_note,
)
from SAE.LLM_pipeline.utils.final_interpretation import (
    build_final_interpretation_prompt,
    format_dictionary_atom_table,
    format_feature_evidence_blocks,
)
from SAE.LLM_pipeline.utils.general import sanitize_groundtruth_text, write_json
from SAE.LLM_pipeline.utils.llm_strategy import (
    final_strategy_text,
    run_llm_strategy,
    strategy_call_count,
    strategy_report_text,
    strategy_total_tokens,
)
from SAE.SAE_training.model import SAEConfig, SparseAutoencoder
from SAE.SAE_training.utils.data import build_artifacts, extract_concat_matrix
from SAE.manifold.utils.projection import candidate_feature_table, projection_state
from prompt_api.client import AigcBestChatClient


def run_name_from_model_config(model_cfg: SAEConfig) -> str:
    return f"hidden{model_cfg.d_hidden}_gate{model_cfg.gate_weight}_orth{model_cfg.orth_weight}_k{model_cfg.topk}"


def load_sl_merk_ground_truth(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table.loc[table["label_true"].astype(int) == 1].copy().reset_index(drop=True)
    table["geneA"] = table["geneA"].astype(str).str.upper()
    table["geneB"] = table["geneB"].astype(str).str.upper()
    table["ground_truth_explanation"] = table["explanation"].map(sanitize_groundtruth_text)
    table["ground_truth_features"] = table["important_features"].fillna("").map(split_feature_lines)
    table["pair_key"] = table.apply(lambda row: unordered_key(row["geneA"], row["geneB"]), axis=1)
    return table


def split_feature_lines(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").replace(";", "\n").splitlines() if part.strip()]


def unordered_key(gene_a: str, gene_b: str) -> str:
    return "-".join(sorted([str(gene_a).upper(), str(gene_b).upper()]))


def load_curated_context_rows(groundtruth_dir: str | Path, sl_merk: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = Path(groundtruth_dir)
    for gt_row in sl_merk.itertuples(index=False):
        gene_a = str(gt_row.geneA).upper()
        gene_b = str(gt_row.geneB).upper()
        exact_dir = root / f"{gene_a}-{gene_b}"
        reverse_dir = root / f"{gene_b}-{gene_a}"
        if exact_dir.exists():
            pair_dir = exact_dir
        elif reverse_dir.exists():
            pair_dir = reverse_dir
        else:
            continue
        score_text = (pair_dir / "slformer_context_scores.txt").read_text(encoding="utf-8")
        table_text = "\n".join(line for line in score_text.splitlines() if line.strip() and not line.startswith("pair:"))
        score_table = pd.read_fwf(io.StringIO(table_text))
        for row in score_table.itertuples(index=False):
            rows.append(
                {
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "pair_folder": pair_dir.name,
                    "pair_key": str(gt_row.pair_key),
                    "cancer": str(row.cancer),
                    "score_file": float(row.mean),
                    "ground_truth_features": list(gt_row.ground_truth_features),
                    "ground_truth_explanation": str(gt_row.ground_truth_explanation),
                }
            )
    return pd.DataFrame(rows)


def sl_merk_rows_from_artifact(sl_merk: pd.DataFrame, meta: pd.DataFrame, groundtruth_dir: str | Path) -> pd.DataFrame:
    meta_lookup = meta.copy().reset_index().rename(columns={"index": "target_idx"})
    meta_lookup["pair_key"] = meta_lookup.apply(lambda row: unordered_key(row["primary_gene"], row["partner_gene"]), axis=1)
    meta_lookup["target_score"] = meta_lookup["score"].astype(float)

    gt = sl_merk.drop_duplicates("pair_key", keep="first")[
        ["geneA", "geneB", "pair_key", "ground_truth_features", "ground_truth_explanation"]
    ].copy()
    gt = gt.rename(columns={"geneA": "gene_a", "geneB": "gene_b"})

    matched = meta_lookup.merge(gt, on="pair_key", how="inner")
    root = Path(groundtruth_dir)
    pair_folders = []
    for row in matched.itertuples(index=False):
        exact = root / f"{row.gene_a}-{row.gene_b}"
        reverse = root / f"{row.gene_b}-{row.gene_a}"
        if exact.exists():
            pair_folders.append(exact.name)
        elif reverse.exists():
            pair_folders.append(reverse.name)
        else:
            pair_folders.append(f"{row.gene_a}-{row.gene_b}")
    matched["pair_folder"] = pair_folders
    matched["target_idx"] = matched["target_idx"].astype(int)
    matched["score_file"] = matched["target_score"]
    return matched.sort_values(["target_score", "cancer", "pair_folder"], ascending=[False, True, True]).reset_index(drop=True)


def load_sae_artifacts(project_root: Path, train_config_path: Path, sae_dir: Path) -> dict[str, Any]:
    train_cfg = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    model_cfg = SAEConfig(**dict(train_cfg["model"]))
    artifacts = build_artifacts(
        embeddings_pkl=train_cfg["paths"]["embeddings_pkl"],
        prediction_csvs=train_cfg["paths"]["prediction_csvs"],
    )
    X, y, meta = extract_concat_matrix(
        artifacts,
        cancer=train_cfg["scope"]["cancer"],
        max_samples=train_cfg["scope"]["max_samples"],
        seed=train_cfg["scope"]["seed"],
        use_score_col=train_cfg["scope"]["score_col"],
    )
    mu = np.load(sae_dir / "norm" / "mu.npy")
    sigma = np.load(sae_dir / "norm" / "sigma.npy")
    Xn = ((X - mu) / sigma).astype(np.float32)

    checkpoint = torch.load(sae_dir / "final.pt", map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(SAEConfig(**checkpoint["sae_cfg"]))
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    return {
        "train_cfg": train_cfg,
        "model_cfg": model_cfg,
        "X": X,
        "Xn": Xn,
        "y": y,
        "meta": meta.reset_index(drop=True),
        "sae": sae,
    }


def attach_target_indices(target_rows: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta_lookup = meta.copy().reset_index().rename(columns={"index": "target_idx"})
    meta_lookup["pair_key"] = meta_lookup.apply(lambda row: unordered_key(row["primary_gene"], row["partner_gene"]), axis=1)
    matched = target_rows.merge(
        meta_lookup,
        on=["pair_key", "cancer"],
        how="inner",
        suffixes=("", "_meta"),
    )
    matched["target_idx"] = matched["target_idx"].astype(int)
    matched["target_score"] = matched["score"].astype(float)
    return matched.sort_values(["target_score", "cancer", "pair_folder"], ascending=[False, True, True]).reset_index(drop=True)


def prepare_pair_geometry(
    *,
    sae: SparseAutoencoder,
    Xn: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    Z: np.ndarray,
    target_idx: int,
    projection_cfg: dict[str, Any],
    device: str,
    feature_topk: int,
    sample_scope: str,
    top_m: int,
    explain_exemplars: int,
) -> dict[str, Any]:
    state = projection_state(
        sae,
        Xn,
        y,
        target_idx=int(target_idx),
        projection_config=projection_cfg,
        device=device,
    )
    candidate_table = candidate_feature_table(
        state["z0"],
        state["jvp"],
        state["delta_z"],
        state["c_star"],
        topk=int(feature_topk),
    )
    evidence_table, explain_indices = activation_feature_tables(
        candidate_features=candidate_table["feature"].astype(int).tolist(),
        Z=Z,
        meta=meta,
        target_idx=int(target_idx),
        context_cancer=str(meta.loc[int(target_idx), "cancer"]),
        sample_scope=str(sample_scope),
        top_m=int(top_m),
        explain_exemplars=int(explain_exemplars),
    )
    feature_rank = rank_candidate_features(candidate_table, evidence_table)
    return {
        "state": state,
        "feature_rank": feature_rank,
        "explain_indices": explain_indices,
    }


def save_interpretation_state(
    *,
    output_dir: Path,
    row: pd.Series,
    feature_rank: pd.DataFrame,
    interpretations: list[dict[str, Any]],
    state: dict[str, Any],
    evidence_cfg: dict[str, Any],
) -> None:
    feature_rank.to_csv(output_dir / "feature_rank.csv", index=False)
    pd.DataFrame(interpretations).to_csv(output_dir / "llm_interpretation_summary.csv", index=False)
    local = state["local"]
    decoder_projection = state["decoder_projection"]
    write_json(
        output_dir / "interpretation_state.json",
        {
            "target": {
                "primary_gene": str(row["gene_a"]),
                "partner_gene": str(row["gene_b"]),
                "training_scope": "mix",
                "cancer": str(row["cancer"]),
                "target_idx": int(row["target_idx"]),
                "score": float(row["target_score"]),
                "ground_truth_source": "SL_MERK_groundtruth.csv",
                "label_true": 1,
            },
            "projection": {
                "neighbor_indices": local["neighbor_indices"].tolist(),
                "grad_ambient_norm": float(np.linalg.norm(local["grad_ambient"])),
                "x_tangent_norm": float(np.linalg.norm(state["x_tangent"])),
                "x_normal_norm": float(np.linalg.norm(state["x_normal"])),
                "decoder_projection": {
                    "alpha": decoder_projection["alpha"],
                    "nnz": decoder_projection["nnz"],
                    "cosine": decoder_projection["cosine"],
                    "rel_err": decoder_projection["rel_err"],
                },
            },
            "evidence": {
                "sample_scope": str(evidence_cfg["sample_scope"]),
                "top_m": int(evidence_cfg["top_m"]),
                "explain_exemplars": int(evidence_cfg["explain_exemplars"]),
            },
            "feature_rank": feature_rank.to_dict(orient="records"),
            "interpretations": interpretations,
        },
    )


def run_pair_feature_explanations(
    *,
    client: AigcBestChatClient,
    output_dir: Path,
    row: pd.Series,
    feature_rank: pd.DataFrame,
    explain_indices: dict[int, list[int]],
    Z: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    feature_strategy: str,
    n_features: int,
    sample_scope: str,
    state: dict[str, Any],
    evidence_cfg: dict[str, Any],
    resume_completed: bool,
    sleep_s: float,
) -> list[dict[str, Any]]:
    selected_features = [int(feature) for feature in feature_rank.head(int(n_features))["feature"]]
    interpretations: list[dict[str, Any]] = []
    completed_features: set[int] = set()
    state_path = output_dir / "interpretation_state.json"
    if resume_completed and state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        for item in previous["interpretations"]:
            feature = int(item["feature"])
            if feature in selected_features:
                item["feature"] = feature
                interpretations.append(item)
                completed_features.add(feature)

    for position, feature in enumerate(selected_features, start=1):
        if feature in completed_features:
            print(f"[feature] {output_dir.name} {position}/{len(selected_features)} feature={feature} reused")
            continue
        started = time.monotonic()
        prompt = build_explainer_prompt(
            feature=feature,
            feature_rank=feature_rank,
            explain_indices=explain_indices[feature],
            Z=Z,
            y=y,
            meta=meta,
            target_idx=int(row["target_idx"]),
            target_cancer=str(row["cancer"]),
            sample_scope=str(sample_scope),
            prompt_dir=client.settings.prompt_dir,
        )
        trace = run_llm_strategy(client, prompt, feature_strategy)
        text = final_strategy_text(trace)
        fields = extract_explainer_fields(text)
        feature_row = feature_rank.loc[feature_rank["feature"] == feature].iloc[0]
        interpretations.append(
            {
                "feature": feature,
                **fields,
                "target_relevance": target_relevance_note(feature_row),
                "explainer_strategy": feature_strategy,
                "evidence_sample_scope": str(sample_scope),
                "explainer_strategy_tokens": strategy_total_tokens(trace),
                "explainer_runtime_s": float(time.monotonic() - started),
            }
        )
        save_interpretation_state(
            output_dir=output_dir,
            row=row,
            feature_rank=feature_rank,
            interpretations=interpretations,
            state=state,
            evidence_cfg=evidence_cfg,
        )
        print(f"[feature] {output_dir.name} {position}/{len(selected_features)} feature={feature} saved")
        if sleep_s > 0:
            time.sleep(float(sleep_s))
    return interpretations


def run_final_interpretation(
    *,
    client: AigcBestChatClient,
    output_dir: Path,
    row: pd.Series,
    feature_rank: pd.DataFrame,
    interpretations: list[dict[str, Any]],
    template: str,
    final_strategy: str,
    resume_completed: bool,
) -> dict[str, Any]:
    final_dir = output_dir / "final_prompt"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_md = final_dir / "final_interpretation.md"
    final_response = final_dir / "final_interpretation_response.md"
    final_prompt_path = final_dir / "final_interpretation_prompt.md"
    if resume_completed and final_md.exists() and final_response.exists() and final_prompt_path.exists():
        text = final_md.read_text(encoding="utf-8")
        return {"text": text, "tokens": 0, "runtime_s": 0.0, "reused": True}

    prompt = build_final_interpretation_prompt(
        template=template,
        target_primary=str(row["gene_a"]),
        target_partner=str(row["gene_b"]),
        cancer=str(row["cancer"]),
        target_score=float(row["target_score"]),
        dictionary_atom_table=format_dictionary_atom_table(feature_rank, interpretations),
        feature_evidence=format_feature_evidence_blocks(interpretations),
    )
    final_prompt_path.write_text(prompt, encoding="utf-8")
    started = time.monotonic()
    trace = run_llm_strategy(client, prompt, final_strategy)
    text = final_strategy_text(trace).strip() + "\n"
    report = strategy_report_text(trace)
    final_response.write_text(report, encoding="utf-8")
    final_md.write_text(text, encoding="utf-8")
    return {
        "text": text,
        "tokens": strategy_total_tokens(trace),
        "runtime_s": float(time.monotonic() - started),
        "reused": False,
    }


def score_final_text(
    *,
    row: pd.Series,
    final_text: str,
    final_prompt: str,
    metric_cfg: dict[str, Any],
) -> dict[str, Any]:
    return score_text_metrics(
        ground_truth_features=row["ground_truth_features"],
        ground_truth_explanation=str(row["ground_truth_explanation"]),
        text=final_text,
        prompt_context=final_prompt,
        feature_embed_model_path=str(metric_cfg["feature_embed_model_path"]),
        feature_tokencls_model_path=str(metric_cfg["feature_tokencls_model_path"]),
        nli_model_path=str(metric_cfg["nli_model_path"]),
        evidence_embed_model_path=str(metric_cfg["evidence_embed_model_path"]),
        device=str(metric_cfg["device"]),
        feature_candidate_backend=str(metric_cfg["feature_candidate_backend"]),
        feature_score_scope="mechanism",
    )


def metric_record(row: pd.Series, output_dir: Path, metrics: dict[str, Any], final_result: dict[str, Any]) -> dict[str, Any]:
    details = metrics["feature_embed_details"]
    checks = metrics["checks"]
    return {
        "output_dir": str(output_dir),
        "pair": f"{row['gene_a']}-{row['gene_b']}",
        "cancer": str(row["cancer"]),
        "target_idx": int(row["target_idx"]),
        "label_true": 1,
        "target_score": float(row["target_score"]),
        "feature_precision": metrics["feature_embed_precision_raw"],
        "feature_recall": metrics["feature_embed_recall_raw"],
        "feature_f1": metrics["feature_embed_f1_raw_full"],
        "gt_feature_total": details["ground_truth"]["total"],
        "gt_feature_covered": len(details["ground_truth"]["covered"]),
        "candidate_phrase_total": details["candidates"]["total"],
        "feature_candidate_backend": details["candidates"]["backend"],
        "faithfulness_score": metrics["faithfulness_score"],
        "hallucination_score": metrics["hallucination_score"],
        "format_score": checks["format_score"],
        "citation_count": checks["citation_count"],
        "final_tokens": int(final_result["tokens"]),
        "final_runtime_s": float(final_result["runtime_s"]),
        "final_reused": bool(final_result["reused"]),
    }


def strategy_api_call_estimate(client: AigcBestChatClient, feature_strategy: str, final_strategy: str, n_features: int) -> dict[str, int]:
    feature_calls = strategy_call_count(
        feature_strategy,
        self_refine_rounds=client.settings.self_refine_rounds,
        cove_num_questions=client.settings.cove_num_questions,
    ) * int(n_features)
    final_calls = strategy_call_count(
        final_strategy,
        self_refine_rounds=client.settings.self_refine_rounds,
        cove_num_questions=client.settings.cove_num_questions,
    )
    return {"feature_calls": feature_calls, "final_calls": final_calls, "total_calls": feature_calls + final_calls}
