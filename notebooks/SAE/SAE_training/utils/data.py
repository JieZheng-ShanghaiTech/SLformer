from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MIX_SCOPE = "mix"
MIX_REQUIRED_CANCERS = ("KIRC", "COAD", "LAML", "OV", "BRCA", "CESC", "SKCM", "LUAD", "Glioma")


@dataclass(frozen=True)
class EmbeddingArtifacts:
    """Aligned embedding + prediction artifacts.

    `predictions` row order must match the fold-concatenated order in `embeddings`.
    """

    embeddings: list
    predictions: pd.DataFrame
    fold_sizes: Tuple[int, ...]
    fold_boundaries: Tuple[int, ...]


def load_predictions(prediction_csvs: Sequence[str | Path]) -> pd.DataFrame:
    """Load and concatenate per-fold prediction CSVs.

    Requirements:
    - `prediction_csvs` must be absolute paths (or resolvable to absolute via `.resolve()`).
    - File order must correspond to the embedding fold order.

    Returns a single DataFrame in the same row order as the fold-concatenated embeddings.
    """
    if not prediction_csvs:
        raise ValueError("prediction_csvs must be a non-empty sequence")

    parts = []
    for p in prediction_csvs:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions: {path}")
        parts.append(pd.read_csv(path))

    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def load_cross_embeddings(embeddings_pkl: str | Path) -> list:
    """Load SLformer cross-attention embeddings.

    Requirements:
    - `embeddings_pkl` must be an absolute path (or resolvable to absolute via `.resolve()`).

    Returns: list[n_folds][2 arrays] where arrays are (n_rows_fold, 512).
    """
    import pickle as pkl

    p = Path(embeddings_pkl).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Missing embeddings: {p}")
    with open(p, "rb") as f:
        embeddings = pkl.load(f)
    return embeddings


def build_artifacts(*, embeddings_pkl: str | Path, prediction_csvs: Sequence[str | Path]) -> EmbeddingArtifacts:
    """Create aligned artifacts for downstream extraction."""
    embs = load_cross_embeddings(embeddings_pkl)
    preds = load_predictions(prediction_csvs)

    fold_sizes = tuple(int(embs[i][0].shape[0]) for i in range(len(embs)))
    boundaries = [0]
    for s in fold_sizes:
        boundaries.append(boundaries[-1] + s)
    fold_boundaries = tuple(boundaries)

    if fold_boundaries[-1] != len(preds):
        raise ValueError(
            "Row alignment mismatch: sum(fold_sizes) != len(preds). "
            f"sum(fold_sizes)={fold_boundaries[-1]} len(preds)={len(preds)}"
        )

    return EmbeddingArtifacts(
        embeddings=embs,
        predictions=preds,
        fold_sizes=fold_sizes,
        fold_boundaries=fold_boundaries,
    )


def extract_concat_matrix(
    artifacts: EmbeddingArtifacts,
    *,
    cancer: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
    use_score_col: str = "score",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract samples as concatenated 1024-d vectors.

    `cancer="mix"` selects every row in the aligned artifacts and requires the
    organized 8 benchmark cancers plus Glioma to be present.

    Returns:
    - X: (N, 1024) float32
    - y: (N,) float32 (SL score)
    - meta: DataFrame with columns: primary_gene, partner_gene, cancer, fold, score
    """
    preds = artifacts.predictions
    cancer = str(cancer).strip()

    if "cancer" not in preds.columns:
        raise ValueError("predictions must contain a 'cancer' column")
    if use_score_col not in preds.columns:
        raise ValueError(f"predictions missing score column: {use_score_col!r}")

    cancer_values = preds["cancer"].astype(str)
    if cancer.lower() == MIX_SCOPE:
        observed = set(cancer_values.unique())
        missing = [name for name in MIX_REQUIRED_CANCERS if name not in observed]
        if missing:
            raise ValueError(
                "scope.cancer='mix' requires prediction/embedding artifacts containing "
                f"{list(MIX_REQUIRED_CANCERS)}; missing {missing}. "
                f"Observed cancers: {sorted(observed)}"
            )
        mask = np.ones(len(preds), dtype=bool)
    else:
        mask = (cancer_values == cancer).to_numpy()
    idx_all = np.flatnonzero(mask)

    if idx_all.size == 0:
        raise ValueError(f"No rows found for cancer={cancer!r}")

    rng = np.random.default_rng(seed)
    if max_samples is not None and idx_all.size > int(max_samples):
        idx_all = rng.choice(idx_all, size=int(max_samples), replace=False)
        idx_all.sort()

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    meta_parts: List[pd.DataFrame] = []

    boundaries = artifacts.fold_boundaries
    embs = artifacts.embeddings

    for fold in range(len(embs)):
        start, end = boundaries[fold], boundaries[fold + 1]
        in_fold = idx_all[(idx_all >= start) & (idx_all < end)]
        if in_fold.size == 0:
            continue
        local = in_fold - start

        p = embs[fold][0][local]
        q = embs[fold][1][local]
        X = np.concatenate([p, q], axis=1).astype(np.float32, copy=False)

        y = preds.loc[in_fold, use_score_col].to_numpy(dtype=np.float32, copy=False)
        meta = preds.loc[in_fold, ["primary_gene", "partner_gene", "cancer", use_score_col]].copy()
        meta["fold"] = fold

        X_parts.append(X)
        y_parts.append(y)
        meta_parts.append(meta)

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    meta_all = pd.concat(meta_parts, ignore_index=True)

    return X_all, y_all, meta_all


def concat_slformer_pair_embeddings(slformer_emb_folds: list) -> np.ndarray:
    gene1_embeddings = []
    gene2_embeddings = []
    for fold_embeddings in slformer_emb_folds:
        gene1 = fold_embeddings[0][:, 0, :] if fold_embeddings[0].ndim == 3 else fold_embeddings[0]
        gene2 = fold_embeddings[1][:, 0, :] if fold_embeddings[1].ndim == 3 else fold_embeddings[1]
        gene1_embeddings.append(gene1.astype(np.float32))
        gene2_embeddings.append(gene2.astype(np.float32))
    return np.concatenate(
        [np.concatenate(gene1_embeddings, axis=0), np.concatenate(gene2_embeddings, axis=0)],
        axis=1,
    ).astype(np.float32)


def load_cancer_id_map(cancer_list_txt: str | Path) -> dict[str, int]:
    cancers = [line.strip() for line in Path(cancer_list_txt).open() if line.strip()]
    return {cancer: index for index, cancer in enumerate(cancers)}


def concat_geneformer_pair_embeddings(
    meta: pd.DataFrame,
    geneformer_emb: list,
    gene2id_map: dict[str, int],
    cancer2id_map: dict[str, int],
) -> np.ndarray:
    return np.stack(
        [
            np.concatenate(
                [
                    geneformer_emb[cancer2id_map[str(row.cancer)]][gene2id_map[str(row.primary_gene)]],
                    geneformer_emb[cancer2id_map[str(row.cancer)]][gene2id_map[str(row.partner_gene)]],
                ]
            )
            for row in meta.itertuples(index=False)
        ]
    ).astype(np.float32)


def load_gene2vec_embeddings(gene2vec_txt: str | Path) -> dict[str, np.ndarray]:
    path = Path(gene2vec_txt).expanduser().resolve()
    vectors = {}
    with path.open(encoding="utf-8") as f:
        first = f.readline().strip().split()
        if len(first) > 2:
            vectors[first[0]] = np.asarray(first[1:], dtype=np.float32)
        for line in f:
            parts = line.rstrip().split()
            vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def gene2vec_pair_mask(meta: pd.DataFrame, gene2vec: dict[str, np.ndarray]) -> np.ndarray:
    primary_present = meta["primary_gene"].astype(str).isin(gene2vec).to_numpy()
    partner_present = meta["partner_gene"].astype(str).isin(gene2vec).to_numpy()
    return primary_present & partner_present


def concat_gene2vec_pair_embeddings(meta: pd.DataFrame, gene2vec: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [
            np.concatenate(
                [
                    gene2vec[str(row.primary_gene)],
                    gene2vec[str(row.partner_gene)],
                ]
            )
            for row in meta.itertuples(index=False)
        ]
    ).astype(np.float32)


def zscore_matrix(X: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    return ((X - mu) / (sigma + eps)).astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)
