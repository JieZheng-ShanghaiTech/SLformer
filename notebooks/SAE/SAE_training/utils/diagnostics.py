from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _zscore_rows(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    A = A / (A.std(axis=1, keepdims=True) + 1e-6)
    return A


def corr_offdiag(A: np.ndarray) -> np.ndarray:
    Az = _zscore_rows(A)
    C = (Az @ Az.T) / float(max(Az.shape[1] - 1, 1))
    tri = np.triu_indices(C.shape[0], k=1)
    return C[tri]


def effective_rank(S: np.ndarray) -> float:
    S = np.asarray(S, dtype=np.float64)
    p = (S**2) / float(np.sum(S**2) + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(np.exp(-np.sum(p * np.log(p))))


def summ(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q50": float(np.quantile(x, 0.50)),
        "q90": float(np.quantile(x, 0.90)),
        "frac>0.5": float(np.mean(x > 0.5)),
        "frac>0.8": float(np.mean(x > 0.8)),
    }


def final_metric_row(metrics_csv: str | Path, model_name: str) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_csv)
    best_idx = metrics["val_recon"].astype(float).idxmin()
    row = metrics.loc[
        [best_idx],
        [
            "epoch",
            "train_recon",
            "val_recon",
            "train_loss",
            "val_loss",
            "train_dead_frac",
            "val_dead_frac",
            "train_active_frac",
            "val_active_frac",
            "train_l0_mean",
            "val_l0_mean",
        ],
    ].copy()
    row.insert(0, "model", model_name)
    row["metrics_csv"] = str(metrics_csv)
    return row
