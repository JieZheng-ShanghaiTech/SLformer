"""Centralized configuration for the SLformer prompt API."""

from __future__ import annotations

import os
from pathlib import Path

# ======================================================================
# NOTE: change to your own LLM provider and model, and set API keys in your environment variables if needed.
LLM_PROVIDER = ""
AIGC_BEST_BASE_URL: str = os.environ.get("AIGC_BEST_BASE_URL", "").rstrip("/")
AIGC_API_KEY: str = os.environ.get("AIGC_BEST_API_KEY", "").strip()
MODEL = os.environ.get("LLM_MODEL", "gpt-5.4")
# ======================================================================


# Paths (repo-relative; may be overridden by notebooks/scripts)
# Important-feature exports live under /home/guoyu/SLformer_interpretation/output/embedding_saved/
EMBEDDING_BASE_PATH = Path("/home/guoyu/SLformer_interpretation/output/embedding_saved")
# Default LLM output root in this repo is under output/
OUTPUT_DIR = Path("/home/guoyu/SLformer_interpretation/output/LLM_outputs")



# Generation parameters
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.7))
TOP_P = float(os.environ.get("LLM_TOP_P", 0.95))
_raw_max = os.environ.get("LLM_MAX_TOKENS", "15000").strip()
MAX_TOKENS = int(_raw_max) if _raw_max and _raw_max.lower() != "none" else None

# Strategy defaults
SELF_REFINE_ROUNDS = int(os.environ.get("SELF_REFINE_ROUNDS", "1"))
COVE_NUM_QUESTIONS = int(os.environ.get("COVE_NUM_QUESTIONS", "5"))

# Request behavior
LLM_REQUEST_TIMEOUT_S = float(os.environ.get("LLM_REQUEST_TIMEOUT_S", "60"))

# System prompt
SYSTEM_PROMPT = (
    "You are a computational biologist specialized in mechanistic interpretation "
    "of deep learning models for synthetic lethality prediction."
)

# Gene sets for enrichment
GENE_SETS = (
    "Reactome_2022",
    "KEGG_2021_Human",
    "GO_Biological_Process_2021",
    "GO_Molecular_Function_2021",
    "GO_Cellular_Component_2021",
)
ENRICHMENT_TOP_K = 12

# Report / Output structure
REPORT_DIR_NAME = "cross_val_res"
CROSS_SUBDIR = "cross_emb"

# Data Paths (from prompt_gen.ipynb)
GENE2ID_PATH = "/home/jienihu/sc/SLformer/data/saved_data/map/gene2id.pkl"
CANCER_LIST_PATH = "/home/jienihu/sc/SLformer/data/saved_data/map/cancer_list.txt"
GO_BASIC_OBO_PATH = "/home/jienihu/sc/SLformer/data/GO/go-basic.obo"
GOA_HUMAN_GAF_PATH = "/home/jienihu/sc/SLformer/data/GO/goa_human.gaf"
ID_MAPPING_PATH = "/home/jienihu/sc/SLformer/data/GO/idmapping_2024_11_09.tsv"
GO_ANNO_POPULAR_PATH = "/home/jienihu/sc/SLformer/data/GO/go_anno_popular.csv"

# Model Paths (from enrichment.py)
MEDCPT_PATH = "/data/guoyu/HF-models/MedCPT-Query-Encoder"
MPNET_PATH = "/data/guoyu/HF-models/all-mpnet-base-v2"
BGE_PATH = "/data/guoyu/HF-models/bge-large-en-v1.5"

