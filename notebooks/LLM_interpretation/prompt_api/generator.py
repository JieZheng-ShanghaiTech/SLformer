"""Prompt generator API."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
import contextlib
import io
import os
import pickle as pkl
import pandas as pd

from .models import GenePair
from . import config

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompt_template.txt"


def _load_prompt_template() -> str:
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_TEMPLATE_PATH}")
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

def _load_embedding_features(gene_pair: GenePair, embedding_base_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load important embedding features for a gene pair (cross-attention only)."""
    primary_gene = gene_pair.primary
    partner_gene = gene_pair.partner

    # Repo uses `crossemb-important/` (hyphen), but older code used `crossemb_important/`.
    embedding_type_candidates = ["crossemb-important", "crossemb_important"]
    pair_candidates = [f"{primary_gene}-{partner_gene}", f"{partner_gene}-{primary_gene}"]

    for embedding_type_dir in embedding_type_candidates:
        for pair_dir in pair_candidates:
            full_path = embedding_base_path / embedding_type_dir / pair_dir
            if not full_path.exists():
                continue

            primary_features_path = full_path / "primary_important_features.csv"
            partner_features_path = full_path / "partner_important_features.csv"

            primary_features = pd.read_csv(primary_features_path) if primary_features_path.exists() else None
            partner_features = pd.read_csv(partner_features_path) if partner_features_path.exists() else None

            if (primary_features is None or primary_features.empty) and (partner_features is None or partner_features.empty):
                continue
            return primary_features, partner_features

    return None, None


def _analyze_embedding_contexts(features: pd.DataFrame, gene_type: str, score_override: Optional[float] = None) -> List[str]:
    analysis: List[str] = []
    cancer_types = features['cancer'].unique()
    for cancer in cancer_types:
        subset = features[features['cancer'] == cancer]
        if score_override is not None:
            score = score_override
        else:
            score = subset['score'].values[0] if 'score' in subset.columns else None
        
        emb_cols = [col for col in features.columns if col.startswith(f'{gene_type}_emb_')]
        row = subset.iloc[0]
        top_dims = sorted([(col, row[col]) for col in emb_cols], key=lambda x: -x[1])[:5]
        top_dims_str = ', '.join([f"dim_{col.split('_')[-1]}({val:.3f})" for col, val in top_dims])
        avg_activations = features[emb_cols].mean().sort_values(ascending=False)
        context_specific = []
        for col, val in top_dims:
            avg_val = float(avg_activations[col]) if col in avg_activations else 0.0
            if avg_val > 0 and val > 2 * avg_val:
                context_specific.append(f"dim_{col.split('_')[-1]}(x{val/avg_val:.1f} vs avg)")
        context_specific_str = (
            f"Context-specific: {', '.join(context_specific)}" if context_specific else "No strongly context-specific dimensions"
        )
        if score is not None:
            analysis.append(f"  {cancer} (SL-Score: {score:.3f}): Top dimensions - {top_dims_str} | {context_specific_str}")
        else:
            analysis.append(f"  {cancer}: Top dimensions - {top_dims_str} | {context_specific_str}")
    return analysis


def _find_shared_goterms_strings(gene_pair: GenePair) -> Tuple[List[str], List[str], List[str]]:
    """Get GO terms using goatools."""

    # This repo may not ship GO resources; treat GO terms as optional.
    try:
        from goatools.obo_parser import GODag  # type: ignore
        from goatools.anno.gaf_reader import GafReader  # type: ignore
    except Exception:
        return [], [], []

    try:
        go_basic = Path(str(config.GO_BASIC_OBO_PATH))
        goa_gaf = Path(str(config.GOA_HUMAN_GAF_PATH))
        idmap_tsv = Path(str(config.ID_MAPPING_PATH))
        if not (go_basic.exists() and goa_gaf.exists() and idmap_tsv.exists()):
            return [], [], []

        quiet_stream = io.StringIO()
        with contextlib.redirect_stdout(quiet_stream), contextlib.redirect_stderr(quiet_stream):
            go_dag = GODag(str(go_basic), prt=None)
            annotations = GafReader(str(goa_gaf), prt=None, godag=go_dag).get_id2gos(namespace="BP", prt=None)

        # Load ID mapping from UniProt to gene symbols
        id_mapping_df = pd.read_csv(str(idmap_tsv), sep='\t')
        id_mapping = dict(zip(id_mapping_df['From'], id_mapping_df['To']))

        # Map annotations to gene symbols
        anno_mapped = {}
        for k, v in annotations.items():
            if k in id_mapping:
                anno_mapped[id_mapping[k]] = v

        def get_go_term_details(go_terms):
            details = []
            for term_id in go_terms:
                if term_id in go_dag:
                    term = go_dag[term_id]
                    details.append(
                        {
                            'id': term_id,
                            'name': term.name,
                            'namespace': term.namespace,
                            'depth': term.depth,
                        }
                    )
            return details

        go_terms_gene1 = {go_id for go_id in anno_mapped.get(gene_pair.primary, [])}
        go_terms_gene2 = {go_id for go_id in anno_mapped.get(gene_pair.partner, [])}

        shared_go_terms = go_terms_gene1.intersection(go_terms_gene2)
        primary_terms = [terms['name'] for terms in get_go_term_details(go_terms_gene1)]
        partner_terms = [terms['name'] for terms in get_go_term_details(go_terms_gene2)]
        overlaps = [terms['name'] for terms in get_go_term_details(shared_go_terms)]

        return overlaps, primary_terms, partner_terms
    except Exception:
        return [], [], []


def generate_prompt(gene_pair: GenePair, context: Optional[str] = None, score_override: Optional[float] = None) -> str:
    """Generate the contextual embedding interpretation prompt."""

    overlaps, primary_terms, partner_terms = _find_shared_goterms_strings(gene_pair)
    primary_features, partner_features = _load_embedding_features(
        gene_pair, config.EMBEDDING_BASE_PATH
    )

    if (primary_features is None or primary_features.empty) and (partner_features is None or partner_features.empty):
        return (
            f"Error: No embedding features found for gene pair {gene_pair.primary}-{gene_pair.partner} "
            f"using cross-attention embeddings"
        )

    analysis: List[str] = []
    
    if primary_features is not None and not primary_features.empty:
        analysis.append(f"Primary Gene ({gene_pair.primary}) Contextual Embedding Analysis:")
        analysis.extend(_analyze_embedding_contexts(primary_features, 'primary', score_override))
    
    if partner_features is not None and not partner_features.empty:
        analysis.append(f"\nPartner Gene ({gene_pair.partner}) Contextual Embedding Analysis:")
        analysis.extend(_analyze_embedding_contexts(partner_features, 'partner', score_override))

    embedding_type_str = "cross-attention"
    available_contexts: List[str] = []
    if primary_features is not None and not primary_features.empty:
        available_contexts.extend(primary_features['cancer'].tolist())
    if partner_features is not None and not partner_features.empty:
        available_contexts.extend(partner_features['cancer'].tolist())

    target_context = context or (available_contexts[0] if available_contexts else "Unknown")
    
    primary_terms_str = ", ".join(primary_terms) if primary_terms else "N/A"
    partner_terms_str = ", ".join(partner_terms) if partner_terms else "N/A"
    overlaps_str = ", ".join(overlaps) if overlaps else "N/A"
    available_contexts_str = ", ".join(sorted(set(available_contexts))) if available_contexts else "Unknown"

    analysis_str = "\n".join(analysis)

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
    metabolic_note = ""
    redox_note = ""
    guardrail_line = (
        "Avoid over-weighting any single family without convergent support and make uncertainty explicit when evidence is mixed."
    )

    template = _load_prompt_template()
    query_prompt = template.format(
        embedding_type_str=embedding_type_str,
        gene_primary=gene_pair.primary,
        gene_partner=gene_pair.partner,
        primary_terms_str=primary_terms_str,
        partner_terms_str=partner_terms_str,
        overlaps_str=overlaps_str,
        analysis_str=analysis_str,
        target_context=target_context,
        available_contexts_str=available_contexts_str,
        topic_profile_line="",
        depth_guidance=depth_guidance,
        mech_ranking_hint=mech_ranking_hint,
        metabolic_note=metabolic_note,
        redox_note=redox_note,
        guardrail_line=guardrail_line,
    )
    return query_prompt.strip()
