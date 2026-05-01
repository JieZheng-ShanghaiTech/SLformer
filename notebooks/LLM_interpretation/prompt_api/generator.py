"""Prompt generator API."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from . import config
from .models import GenePair

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompt_template.txt"


def _load_embedding_features(gene_pair: GenePair, embedding_base_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_dir = embedding_base_path / "crossemb-important" / f"{gene_pair.primary}-{gene_pair.partner}"
    primary_features = pd.read_csv(pair_dir / "primary_important_features.csv")
    partner_features = pd.read_csv(pair_dir / "partner_important_features.csv")
    return primary_features, partner_features


def _analyze_embedding_contexts(features: pd.DataFrame, gene_type: str, score_override: Optional[float] = None) -> List[str]:
    analysis: List[str] = []
    emb_cols = [col for col in features.columns if col.startswith(f"{gene_type}_emb_")]
    avg_activations = features[emb_cols].mean().sort_values(ascending=False)

    for cancer in features["cancer"].unique():
        subset = features[features["cancer"] == cancer]
        row = subset.iloc[0]
        score = float(score_override) if score_override is not None else float(row["score"])
        top_dims = sorted([(col, row[col]) for col in emb_cols], key=lambda item: -item[1])[:5]
        top_dims_str = ", ".join(f"dim_{col.split('_')[-1]}({val:.3f})" for col, val in top_dims)
        context_specific = [
            f"dim_{col.split('_')[-1]}(x{val / float(avg_activations[col]):.1f} vs avg)"
            for col, val in top_dims
            if float(avg_activations[col]) > 0 and val > 2 * float(avg_activations[col])
        ]
        context_specific_str = "Context-specific: " + ", ".join(context_specific) if context_specific else "No strongly context-specific dimensions"
        analysis.append(f"  {cancer} (SL-Score: {score:.3f}): Top dimensions - {top_dims_str} | {context_specific_str}")
    return analysis


def _find_shared_goterms_strings(gene_pair: GenePair) -> Tuple[List[str], List[str], List[str]]:
    from goatools.anno.gaf_reader import GafReader  # type: ignore
    from goatools.obo_parser import GODag  # type: ignore

    go_basic = Path(str(config.CONFIG.go_basic_obo_path))
    goa_gaf = Path(str(config.CONFIG.goa_human_gaf_path))
    idmap_tsv = Path(str(config.CONFIG.id_mapping_path))

    quiet_stream = io.StringIO()
    with contextlib.redirect_stdout(quiet_stream), contextlib.redirect_stderr(quiet_stream):
        go_dag = GODag(str(go_basic), prt=None)
        annotations = GafReader(str(goa_gaf), prt=None, godag=go_dag).get_id2gos(namespace="BP", prt=None)

    id_mapping_df = pd.read_csv(str(idmap_tsv), sep="\t")
    id_mapping = dict(zip(id_mapping_df["From"], id_mapping_df["To"]))
    anno_mapped = {id_mapping[key]: value for key, value in annotations.items() if key in id_mapping}

    def get_go_term_names(go_terms):
        return [go_dag[term_id].name for term_id in go_terms]

    go_terms_gene1 = set(anno_mapped[gene_pair.primary])
    go_terms_gene2 = set(anno_mapped[gene_pair.partner])
    shared_go_terms = go_terms_gene1.intersection(go_terms_gene2)

    return (
        get_go_term_names(shared_go_terms),
        get_go_term_names(go_terms_gene1),
        get_go_term_names(go_terms_gene2),
    )


def generate_prompt(gene_pair: GenePair, context: Optional[str] = None, score_override: Optional[float] = None) -> str:
    """Generate the contextual embedding interpretation prompt."""

    overlaps, primary_terms, partner_terms = _find_shared_goterms_strings(gene_pair)
    primary_features, partner_features = _load_embedding_features(gene_pair, config.CONFIG.embedding_base_path)

    analysis: List[str] = [f"Primary Gene ({gene_pair.primary}) Contextual Embedding Analysis:"]
    analysis.extend(_analyze_embedding_contexts(primary_features, "primary", score_override))
    analysis.append(f"\nPartner Gene ({gene_pair.partner}) Contextual Embedding Analysis:")
    analysis.extend(_analyze_embedding_contexts(partner_features, "partner", score_override))

    available_contexts = primary_features["cancer"].tolist() + partner_features["cancer"].tolist()
    target_context = str(context) if context is not None else str(available_contexts[0])

    depth_guidance = (
        "When elaborating the main mechanism, include 2-3 concrete specifics to increase mechanistic clarity (choose those that fit the evidence):\n"
        "- If metabolic: name the pathway branch or module (e.g., glycolysis, PPP, one-carbon, fatty acid synthesis), a plausible rate-limiting or control step, the main cofactor usage/production (e.g., ATP, NAD(P)H), and subcellular compartment(s) involved (cytosol, mitochondria, peroxisome).\n"
        "- If DNA repair: specify the subpathway (HR, NHEJ, BER, MMR), key complex(es)/enzymes, damage type addressed, and how the gene pair alters repair choice or efficiency.\n"
        "- If signaling: identify the pathway axis (e.g., MAPK, PI3K/AKT/mTOR), the node(s) most likely impacted, and whether changes are upstream/downstream or via feedback.\n"
        "- If epigenetic: state the chromatin modifier class (e.g., HMT, HDAC), relevant histone/DNA marks, and expected impact on transcriptional programs.\n"
        "- If cell-cycle: indicate the checkpoint/phase, cyclin/CDK modules, and the dependency created by the pair.\n"
        "Tie each specific to the target context with 1-2 sentences, and ground claims in the provided embeddings and GO terms."
    )
    mech_ranking_hint = (
        "Briefly weigh candidate mechanism families (e.g., signaling, DNA repair, metabolic, epigenetic, cell-cycle, cofactor/redox) by available evidence; discuss the better-supported one(s) first and provide a focused deep-dive for the top family."
    )
    guardrail_line = "Avoid over-weighting any single family without convergent support and make uncertainty explicit when evidence is mixed."

    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").format(
        embedding_type_str="cross-attention",
        gene_primary=gene_pair.primary,
        gene_partner=gene_pair.partner,
        primary_terms_str=", ".join(primary_terms) if primary_terms else "N/A",
        partner_terms_str=", ".join(partner_terms) if partner_terms else "N/A",
        overlaps_str=", ".join(overlaps) if overlaps else "N/A",
        analysis_str="\n".join(analysis),
        target_context=target_context,
        available_contexts_str=", ".join(sorted(set(available_contexts))),
        topic_profile_line="",
        depth_guidance=depth_guidance,
        mech_ranking_hint=mech_ranking_hint,
        metabolic_note="",
        redox_note="",
        guardrail_line=guardrail_line,
    ).strip()
