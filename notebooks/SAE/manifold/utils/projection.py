"""SAE latent projection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Lasso

from SAE.LLM_pipeline.utils.general import unit_vector
from SAE.SAE_training.model import SAEConfig, SparseAutoencoder, estimate_latent_direction_scores
from SAE.manifold.core import TangentSpace, estimate_point_score_direction, project_to_tangent


def encode_dataset(model: SparseAutoencoder, X: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    model = model.to(device)
    parts = []
    with torch.no_grad():
        for start in range(0, X.shape[0], int(batch_size)):
            xb = torch.from_numpy(X[start : start + int(batch_size)]).to(device)
            parts.append(model.encode(xb).detach().cpu().numpy())
    return np.concatenate(parts, axis=0)


def sparse_decoder_projection(
    decoder_weight: np.ndarray,
    direction: np.ndarray,
    *,
    base_alpha: float,
    grid_size: int,
    threshold: float = 1e-8,
) -> dict[str, Any]:
    W = np.asarray(decoder_weight, dtype=np.float32)
    v = unit_vector(direction)
    col_norm = np.linalg.norm(W, axis=0) + 1e-12
    Wn = W / col_norm
    rows = []

    for alpha in [float(base_alpha) * (0.5 ** i) for i in range(int(grid_size))]:
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=50000)
        lasso.fit(Wn, v)
        coeff = lasso.coef_.astype(np.float32) / col_norm
        recon = W @ coeff
        recon_unit = unit_vector(recon)
        rows.append(
            {
                "alpha": float(alpha),
                "c_star": coeff,
                "nnz": int(np.count_nonzero(np.abs(coeff) > float(threshold))),
                "cosine": float(np.dot(v, recon_unit)),
                "rel_err": float(np.linalg.norm(v - recon) / (np.linalg.norm(v) + 1e-12)),
            }
        )

    return max(rows, key=lambda row: row["cosine"])


def projection_state(
    model: SparseAutoencoder,
    Xn: np.ndarray,
    y: np.ndarray,
    *,
    target_idx: int,
    projection_config: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    local = estimate_point_score_direction(
        Xn,
        y.astype(np.float32),
        point_index=target_idx,
        n_neighbors=int(projection_config["default_neighbors"]),
        metric=str(projection_config["metric"]),
        tangent_dim=int(projection_config["tangent_dim"]),
        ridge_alpha=float(projection_config["ridge_alpha"]),
    )
    direction = local["grad_ambient_unit"].astype(np.float32)
    tangent = TangentSpace(basis=local["basis"], mean=local["mu"])
    x_tangent, x_normal = project_to_tangent(Xn[target_idx] - tangent.mean, tangent)

    x0 = torch.from_numpy(Xn[target_idx]).unsqueeze(0).to(device)
    v = torch.from_numpy(direction).unsqueeze(0).to(device)
    latent = estimate_latent_direction_scores(model, x0, v)
    z0 = latent["z0"].detach().cpu().numpy()[0]
    jvp = latent["jvp"].detach().cpu().numpy()[0]

    eps = float(projection_config["fd_eps"])
    with torch.no_grad():
        z_eps = model.encode(x0 + eps * v).detach().cpu().numpy()[0]

    decoder_projection = sparse_decoder_projection(
        model.decoder.weight.detach().cpu().numpy().astype(np.float32),
        direction,
        base_alpha=float(projection_config["decoder_lasso_alpha"]),
        grid_size=int(projection_config["decoder_lasso_grid_size"]),
    )

    return {
        "local": local,
        "direction": direction,
        "x_tangent": x_tangent,
        "x_normal": x_normal,
        "z0": z0,
        "jvp": jvp,
        "delta_z": z_eps - z0,
        "decoder_projection": decoder_projection,
        "c_star": decoder_projection["c_star"],
    }


def top_abs_indices(values: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.abs(values))[::-1][: int(k)]


def candidate_feature_table(z0: np.ndarray, jvp: np.ndarray, delta_z: np.ndarray, c_star: np.ndarray, *, topk: int) -> pd.DataFrame:
    top_z = np.argsort(z0)[::-1][: int(topk)]
    top_jvp = top_abs_indices(jvp, topk)
    top_decoder = top_abs_indices(c_star, topk)
    features = sorted(set(top_z.tolist()) | set(top_jvp.tolist()) | set(top_decoder.tolist()))
    return pd.DataFrame(
        {
            "feature": features,
            "in_top_z0": [feature in set(top_z.tolist()) for feature in features],
            "in_top_jvp": [feature in set(top_jvp.tolist()) for feature in features],
            "in_top_decoder": [feature in set(top_decoder.tolist()) for feature in features],
            "z0": [float(z0[feature]) for feature in features],
            "jvp": [float(jvp[feature]) for feature in features],
            "delta_z": [float(delta_z[feature]) for feature in features],
            "c_star": [float(c_star[feature]) for feature in features],
        }
    )


def reconstruct_with_sae_checkpoint(
    X: np.ndarray,
    sae_dir: str | Path,
    *,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> np.ndarray:
    sae_dir = Path(sae_dir)
    mu = np.load(sae_dir / "norm" / "mu.npy")
    sigma = np.load(sae_dir / "norm" / "sigma.npy")
    X_scaled = ((X - mu) / sigma).astype(np.float32)

    ckpt = torch.load(sae_dir / "final.pt", map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(SAEConfig(**ckpt["sae_cfg"]))
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()

    run_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae.to(run_device)

    reconstructed_batches = []
    with torch.no_grad():
        for start in range(0, X_scaled.shape[0], int(batch_size)):
            batch = torch.from_numpy(X_scaled[start : start + int(batch_size)]).to(run_device)
            batch_hat, _ = sae(batch)
            reconstructed_batches.append(batch_hat.detach().cpu().numpy())
    X_recon_scaled = np.concatenate(reconstructed_batches, axis=0).astype(np.float32)
    return (X_recon_scaled * sigma + mu).astype(np.float32)
