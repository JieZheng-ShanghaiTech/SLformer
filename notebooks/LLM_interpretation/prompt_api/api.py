"""High-level API for SLformer embedding interpretation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List

from .models import GenePair
from .client import AigcBestChatClient, PromptProcessor


class SLformerAPI:
    """Main entry point for running SLformer analyses."""

    def __init__(self):
        self.client = AigcBestChatClient()
        self.processor = PromptProcessor(self.client)
    
    def analyze_gene_pair(
        self,
        primary_gene: str,
        partner_gene: str,
        context: str,
        score_override: Optional[float] = None,
        save_output: bool = True,
        output_dir: Optional[Path | str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        strategy: str = "baseline",
        self_refine_rounds: Optional[int] = None,
        cove_questions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze a gene pair for synthetic lethality mechanisms (Cross-Attention only)."""
        
        gene_pair = GenePair(primary=primary_gene, partner=partner_gene)
        
        return self.processor.generate_explanation(
            gene_pair=gene_pair,
            context=context,
            score_override=score_override,
            save_output=save_output,
            output_dir=output_dir,
            temperature=temperature,
            top_p=top_p,
            strategy=strategy,
            self_refine_rounds=self_refine_rounds,
            cove_questions=cove_questions,
        )

    def analyze_gene_pair_comprehensive(
        self,
        primary_gene: str,
        partner_gene: str,
        context: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Wrapper for analyze_gene_pair, providing compatibility with the old interface."""
        
        result = self.analyze_gene_pair(
            primary_gene=primary_gene,
            partner_gene=partner_gene,
            context=context,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        return {
            "cross_attention": result,
        }

