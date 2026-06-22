"""Final pair-level interpretation prompt helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd


GEOMETRY_COLUMNS = ["z0", "jvp", "delta_z", "c_star"]
def rank_label(value: int, total: int) -> str:
    return f"{int(value)}/{int(total)}"



def signed_rank_table(feature_rank: pd.DataFrame, interpretations: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    features = [int(item["feature"]) for item in interpretations]
    table = feature_rank.loc[feature_rank["feature"].isin(features)].copy().reset_index(drop=True)
    table["abs_z0_rank"] = table["z0"].abs().rank(ascending=False, method="min").astype(int)
    table["abs_dot_z_rank"] = table["jvp"].abs().rank(ascending=False, method="min").astype(int)
    table["abs_delta_z_rank"] = table["delta_z"].abs().rank(ascending=False, method="min").astype(int)
    table["abs_c_star_rank"] = table["c_star"].abs().rank(ascending=False, method="min").astype(int)
    table["abs_c_star"] = table["c_star"].abs()
    return table.set_index("feature")


def format_dictionary_atom_table(feature_rank: pd.DataFrame, interpretations: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    geometry = signed_rank_table(feature_rank, interpretations)
    total = len(interpretations)
    for item in interpretations:
        feature = int(item["feature"])
        row = geometry.loc[feature]
        rows.append(
            " | ".join(
                [
                    f"feature={feature}",
                    f"hypothesis={item['hypothesis']}",
                    f"confidence={item['confidence']}",
                    f"z0={float(row['z0']):.4f} (activation_rank={rank_label(row['abs_z0_rank'], total)})",
                    f"dot_z={float(row['jvp']):.4f} (magnitude_rank={rank_label(row['abs_dot_z_rank'], total)})",
                    f"delta_z={float(row['delta_z']):.6f} (magnitude_rank={rank_label(row['abs_delta_z_rank'], total)})",
                    f"c_star={float(row['c_star']):.4f} (abs_rank={rank_label(row['abs_c_star_rank'], total)})",
                ]
            )
        )
    return "\n".join(rows)


def format_feature_evidence_blocks(interpretations: Sequence[Mapping[str, Any]]) -> str:
    blocks = []
    for item in interpretations:
        feature = int(item["feature"])
        blocks.append(
            f"Feature {feature}\n"
            f"Hypothesis: {item['hypothesis']}\n"
            f"Confidence: {item['confidence']}\n"
            f"Target relevance: {item['target_relevance']}\n"
            f"Rationale: {item['rationale']}"
        )
    return "\n\n".join(blocks)


def build_final_interpretation_prompt(
    *,
    template: str,
    target_primary: str,
    target_partner: str,
    cancer: str,
    target_score: float,
    dictionary_atom_table: str,
    feature_evidence: str,
) -> str:
    return template.format(
        target_primary=target_primary,
        target_partner=target_partner,
        cancer=cancer,
        dictionary_atom_table=dictionary_atom_table,
        feature_evidence=feature_evidence,
    )
