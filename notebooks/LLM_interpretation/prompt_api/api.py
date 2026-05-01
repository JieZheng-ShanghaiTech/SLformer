"""High-level API for SLformer embedding interpretation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .client import AigcBestChatClient, PromptProcessor
from .models import GenePair


class SLformerAPI:
    """Main entry point for running SLformer prompt analyses."""

    def __init__(self, *, config_path: str | Path | None = None) -> None:
        self.client = AigcBestChatClient(config_path=config_path)
        self.processor = PromptProcessor(self.client)

    def analyze_gene_pair(
        self,
        primary_gene: str,
        partner_gene: str,
        context: str,
        score_override: Optional[float] = None,
        save_output: bool = True,
        output_dir: Optional[Path | str] = None,
        strategy: str = "baseline",
    ) -> Dict[str, Any]:
        gene_pair = GenePair(primary=primary_gene, partner=partner_gene)
        return self.processor.generate_explanation(
            gene_pair=gene_pair,
            context=context,
            score_override=score_override,
            save_output=save_output,
            output_dir=output_dir,
            strategy=strategy,
        )

    def analyze_gene_pair_comprehensive(
        self,
        primary_gene: str,
        partner_gene: str,
        context: str,
        score_override: Optional[float] = None,
        save_output: bool = True,
        output_dir: Optional[Path | str] = None,
        strategy: str = "baseline",
    ) -> Dict[str, Any]:
        return {
            "cross_attention": self.analyze_gene_pair(
                primary_gene=primary_gene,
                partner_gene=partner_gene,
                context=context,
                score_override=score_override,
                save_output=save_output,
                output_dir=output_dir,
                strategy=strategy,
            )
        }
