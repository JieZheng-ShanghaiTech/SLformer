"""JSON-backed configuration for the SLformer prompt API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).with_name("model_config.json")


@dataclass(frozen=True)
class PromptAPIConfig:
    base_url: str
    api_key: str
    model: str
    system_prompt: str
    temperature: float
    top_p: float
    max_tokens: int | None
    request_timeout_s: float
    embedding_base_path: Path
    output_dir: Path
    cross_subdir: str
    report_dir_name: str
    self_refine_rounds: int
    cove_num_questions: int
    gene_sets: tuple[str, ...]
    enrichment_top_k: int
    go_basic_obo_path: Path
    goa_human_gaf_path: Path
    id_mapping_path: Path
    go_anno_popular_path: Path
    medcpt_path: Path
    mpnet_path: Path
    bge_path: Path


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> PromptAPIConfig:
    data = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    llm = data["llm"]
    paths = data["paths"]
    strategy = data["strategy"]
    enrichment = data["enrichment"]
    report = data["report"]
    max_tokens = llm["max_tokens"]

    return PromptAPIConfig(
        base_url=str(llm["base_url"]).rstrip("/"),
        api_key=str(llm["api_key"]).strip(),
        model=str(llm["model"]),
        system_prompt=str(llm["system_prompt"]),
        temperature=float(llm["temperature"]),
        top_p=float(llm["top_p"]),
        max_tokens=None if max_tokens is None else int(max_tokens),
        request_timeout_s=float(llm["request_timeout_s"]),
        embedding_base_path=Path(paths["embedding_base_path"]).expanduser(),
        output_dir=Path(paths["output_dir"]).expanduser(),
        cross_subdir=str(report["cross_subdir"]),
        report_dir_name=str(report["report_dir_name"]),
        self_refine_rounds=int(strategy["self_refine_rounds"]),
        cove_num_questions=int(strategy["cove_num_questions"]),
        gene_sets=tuple(str(name) for name in enrichment["gene_sets"]),
        enrichment_top_k=int(enrichment["top_k"]),
        go_basic_obo_path=Path(paths["go_basic_obo_path"]).expanduser(),
        goa_human_gaf_path=Path(paths["goa_human_gaf_path"]).expanduser(),
        id_mapping_path=Path(paths["id_mapping_path"]).expanduser(),
        go_anno_popular_path=Path(paths["go_anno_popular_path"]).expanduser(),
        medcpt_path=Path(paths["medcpt_path"]).expanduser(),
        mpnet_path=Path(paths["mpnet_path"]).expanduser(),
        bge_path=Path(paths["bge_path"]).expanduser(),
    )


CONFIG = load_config()
