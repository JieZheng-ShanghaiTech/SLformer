from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SAEConfig:
    d_in: int = 1024
    d_hidden: int = 4096
    activation: str = "jumprelu"  # relu | jumprelu
    orth_weight: float = 1e2
    decoder_unit_norm: bool = True
    topk: int = 200  # Number of top activations to keep per sample
    gate_weight: float = 1e-2  # Regularization on gate magnitude
    jump_threshold: float = 0.0


@dataclass(frozen=True)
class CrossSAEConfig:
    d_in: int = 512
    d_out: int = 1024
    d_hidden: int = 4096
    activation: str = "jumprelu"
    orth_weight: float = 1.0
    decoder_unit_norm: bool = True
    topk: int = 64
    gate_weight: float = 0.5
    jump_threshold: float = 0.0


def _activation(name: str, *, threshold: float = 0.0):
    name = (name or "").lower().strip()
    if name == "relu":
        return F.relu
    if name == "jumprelu":
        return lambda x: F.relu(x - float(threshold))
    raise ValueError(f"Unsupported activation: {name}")


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for SLformer embeddings with Gated topK sparsity.

    Design goals for interpretability:
    - overcomplete hidden layer (d_hidden >> d_in)
    - sparse activations (topK gating + learned gate magnitudes)
    - near-orthogonal decoder atoms (||W^T W - I||_F^2 on column-normalized decoder)

    Notes:
    - This is intentionally simple and stable (linear encoder/decoder + pointwise nonlinearity).
    - We interpret *decoder columns* as feature directions in embedding space.
    - Gating: each latent is multiplied by a learned magnitude, then topK selection applied.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Linear(cfg.d_in, cfg.d_hidden, bias=True)
        self.decoder = nn.Linear(cfg.d_hidden, cfg.d_in, bias=True)

        # Small init to encourage stable early training.
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.decoder.bias)

        self._act = _activation(cfg.activation, threshold=cfg.jump_threshold)

        # Gate parameter: learned magnitude for each latent feature
        self.gate = nn.Parameter(torch.ones(cfg.d_hidden, dtype=torch.float32))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_pre = self.encoder(x)
        z = self._act(z_pre)
        # Apply gating: scale each latent by learned magnitude
        z_gated = z * self.gate.abs()
        # TopK selection: keep only top k activations per sample
        k = min(self.cfg.topk, z_gated.shape[1])
        vals, indices = torch.topk(z_gated.abs(), k=k, dim=1, largest=True, sorted=False)
        mask = torch.zeros_like(z_gated, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        z_masked = z_gated * mask
        return z_masked

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder_columns_(self) -> None:
        if not self.cfg.decoder_unit_norm:
            return
        w = self.decoder.weight.data  # shape: (d_in, d_hidden)
        norms = torch.norm(w, dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.data = w / norms

    def loss_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_hat, z = self.forward(x)

        recon = F.mse_loss(x_hat, x)
        gate_loss = self.gate.abs().mean()
        orth = self.orthogonality_penalty()

        total = recon + self.cfg.gate_weight * gate_loss + self.cfg.orth_weight * orth
        return {"total": total, "recon": recon, "gate": gate_loss, "orth": orth}

    def orthogonality_penalty(self) -> torch.Tensor:
        """Encourage decoder row space to be near-orthonormal.

        For overcomplete decoders (d_hidden > d_in), exact column orthogonality
        is infeasible. We therefore regularize the row Gram (W W^T) toward I.
        """
        w = self.decoder.weight
        w = F.normalize(w, dim=1, eps=1e-8)  # row-normalized
        gram = w @ w.T  # (d_in, d_in)
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        diff = gram - eye
        # Per-row Frobenius energy avoids extra 1/d_in shrink from entry-wise mean.
        return (diff * diff).sum() / gram.shape[0]


class CrossSparseAutoencoder(nn.Module):
    """Sparse encoder/decoder map between two embedding spaces.

    This shares the SAE mechanism used for SLformer embeddings: linear encoder,
    pointwise activation, learned gate magnitude, hard top-k sparsity, linear
    decoder, and decoder row-space orthogonality.  Unlike `SparseAutoencoder`,
    the input and reconstruction target dimensions may differ.
    """

    def __init__(self, cfg: CrossSAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.d_in, cfg.d_hidden, bias=True)
        self.decoder = nn.Linear(cfg.d_hidden, cfg.d_out, bias=True)
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.decoder.bias)
        self._act = _activation(cfg.activation, threshold=cfg.jump_threshold)
        self.gate = nn.Parameter(torch.ones(cfg.d_hidden, dtype=torch.float32))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self._act(self.encoder(x))
        z_gated = z * self.gate.abs()
        k = min(self.cfg.topk, z_gated.shape[1])
        _, indices = torch.topk(z_gated.abs(), k=k, dim=1, largest=True, sorted=False)
        mask = torch.zeros_like(z_gated, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        return z_gated * mask

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def orthogonality_penalty(self) -> torch.Tensor:
        w = F.normalize(self.decoder.weight, dim=1, eps=1e-8)
        gram = w @ w.T
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        diff = gram - eye
        return (diff * diff).sum() / gram.shape[0]

    def loss_components(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_hat, z = self.forward(x)
        recon = F.mse_loss(x_hat, target)
        gate = self.gate.abs().mean()
        orth = self.orthogonality_penalty()
        total = recon + float(self.cfg.gate_weight) * gate + float(self.cfg.orth_weight) * orth
        return {"total": total, "recon": recon, "gate": gate, "orth": orth, "z": z}


def compute_latent_jvp(
    model: SparseAutoencoder,
    x: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Compute Jacobian-vector product d(encode)/dx @ direction at x.

    Args:
        model: SAE model.
        x: Input point, shape (d_in,) or (1, d_in).
        direction: Ambient direction, same shape as x.

    Returns:
        jvp tensor with shape (1, d_hidden).
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if direction.ndim == 1:
        direction = direction.unsqueeze(0)

    x = x.to(dtype=torch.float32)
    direction = direction.to(device=x.device, dtype=x.dtype)

    def _encode(inp: torch.Tensor) -> torch.Tensor:
        return model.encode(inp)

    model.eval()
    x_req = x.detach().clone().requires_grad_(True)
    _, jvp = torch.autograd.functional.jvp(_encode, x_req, direction, create_graph=False, strict=False)
    return jvp


def estimate_latent_direction_scores(
    model: SparseAutoencoder,
    x: torch.Tensor,
    direction: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return baseline latent and directional sensitivity from JVP.

    The `direction` is L2-normalized for stable scaling.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if direction.ndim == 1:
        direction = direction.unsqueeze(0)

    x = x.to(dtype=torch.float32)
    direction = direction.to(device=x.device, dtype=x.dtype)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)

    model.eval()
    with torch.no_grad():
        z0 = model.encode(x)
    jvp = compute_latent_jvp(model, x, direction)
    return {"z0": z0, "jvp": jvp, "direction": direction}
