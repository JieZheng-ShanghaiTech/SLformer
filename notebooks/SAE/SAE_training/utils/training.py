from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import csv
import time
import warnings
import json
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from SAE.SAE_training.model import SAEConfig, SparseAutoencoder


_METRIC_COLUMNS = (
    "epoch", "train_loss", "val_loss", "train_recon", "val_recon",
    "train_gate_term", "val_gate_term", "train_orth_term", "val_orth_term",
    "train_gate_raw", "val_gate_raw", "train_orth_raw", "val_orth_raw",
    "train_l0_mean", "val_l0_mean", "train_l0_median", "val_l0_median",
    "train_dead_frac", "val_dead_frac", "train_active_frac", "val_active_frac",
    "train_mean_act_active", "val_mean_act_active", "train_mean_act_all", "val_mean_act_all",
    "lr", "runtime_s",
)


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 512
    lr: float = 2e-4
    weight_decay: float = 1e-2
    epochs: int = 20
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_every_epoch: bool = True
    amp: bool = False
    scheduler_type: str = "none"
    scheduler_eta_min: float = 0.0
    save_ckpt: bool = True
    ckpt_every_n_epochs: int = 10
    use_saved_ckpt: bool = False

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], *, seed: int) -> "TrainConfig":
        d = dict(cfg)
        return cls(
            seed=int(seed), batch_size=int(d["batch_size"]), lr=float(d["lr"]),
            weight_decay=float(d["weight_decay"]), epochs=int(d["epochs"]),
            grad_clip=float(d["grad_clip"]), device=str(d["device"]),
            val_every_epoch=bool(d["val_every_epoch"]), amp=bool(d["amp"]),
            scheduler_type=str(d["scheduler_type"]),
            scheduler_eta_min=float(d["scheduler_eta_min"]),
            save_ckpt=bool(d["save_ckpt"]),
            ckpt_every_n_epochs=int(d["ckpt_every_n_epochs"]),
            use_saved_ckpt=bool(d["use_saved_ckpt"]),
        )


@dataclass
class _EpochAgg:
    n: int = 0
    loss: float = 0.0
    recon: float = 0.0
    gate_term: float = 0.0
    orth_term: float = 0.0
    gate_raw: float = 0.0
    orth_raw: float = 0.0
    l0_mean: float = 0.0
    l0_median: float = 0.0
    dead_frac: float = 0.0
    active_frac: float = 0.0
    mean_act_active: float = 0.0
    mean_act_all: float = 0.0

    def add(self, bs: int, total: torch.Tensor, recon: torch.Tensor, gate_term: torch.Tensor, orth_term: torch.Tensor, gate_raw: torch.Tensor, orth_raw: torch.Tensor, lat: Dict[str, torch.Tensor]) -> None:
        self.n += int(bs)
        self.loss += float(total.detach().cpu().item()) * bs
        self.recon += float(recon.detach().cpu().item()) * bs
        self.gate_term += float(gate_term.detach().cpu().item()) * bs
        self.orth_term += float(orth_term.detach().cpu().item()) * bs
        self.gate_raw += float(gate_raw.detach().cpu().item()) * bs
        self.orth_raw += float(orth_raw.detach().cpu().item()) * bs
        self.l0_mean += float(lat["l0_mean"].detach().cpu().item()) * bs
        self.l0_median += float(lat["l0_median"].detach().cpu().item()) * bs
        self.dead_frac += float(lat["dead_frac"].detach().cpu().item()) * bs
        self.active_frac += float(lat["active_frac"].detach().cpu().item()) * bs
        self.mean_act_active += float(lat["mean_act_active"].detach().cpu().item()) * bs
        self.mean_act_all += float(lat["mean_act_all"].detach().cpu().item()) * bs

    def mean(self) -> Dict[str, float]:
        d = max(1, self.n)
        return {
            "loss": self.loss / d,
            "recon": self.recon / d,
            "gate_term": self.gate_term / d,
            "orth_term": self.orth_term / d,
            "gate_raw": self.gate_raw / d,
            "orth_raw": self.orth_raw / d,
            "l0_mean": self.l0_mean / d,
            "l0_median": self.l0_median / d,
            "dead_frac": self.dead_frac / d,
            "active_frac": self.active_frac / d,
            "mean_act_active": self.mean_act_active / d,
            "mean_act_all": self.mean_act_all / d,
        }


@dataclass
class _Runtime:
    model: SparseAutoencoder
    device: torch.device
    pin_memory: bool
    use_amp: bool
    autocast_dtype: Optional[torch.dtype]
    opt: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    
    

    def run_epoch(self, *, dl: DataLoader, train: bool, gate_weight: float, orth_weight: float, grad_clip: float) -> Dict[str, float]:
        self.model.train(mode=train)
        agg = _EpochAgg()
        ctx = torch.enable_grad() if train else torch.no_grad()
        
        with ctx:
            for (x,) in dl:
                x = x.to(device=self.device, dtype=torch.float32, non_blocking=self.pin_memory)
                if train:
                    self.opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp):
                    x_hat, z = self.model.forward(x)
                    recon = torch.nn.functional.mse_loss(x_hat, x)
                    gate_raw = self.model.gate.abs().mean()
                    orth_raw = self.model.orthogonality_penalty()
                    gate_term = float(gate_weight) * gate_raw
                    orth_term = float(orth_weight) * orth_raw
                    total = recon + gate_term + orth_term
                if train:
                    self.scaler.scale(total).backward()
                    if float(grad_clip) > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.model.normalize_decoder_columns_()
                agg.add(int(x.shape[0]), total, recon, gate_term, orth_term, gate_raw, orth_raw, _latent_metrics(z))
        return agg.mean()



# ============================= open APIs =============================

def load_sae_train_config(config_path: str | Path) -> Tuple[Dict[str, Any], SAEConfig, TrainConfig]:
    cfg = yaml.safe_load(Path(config_path).expanduser().resolve().read_text(encoding="utf-8"))
    seed = int(cfg["scope"]["seed"])
    model_cfg = SAEConfig(**cfg["model"])
    train_cfg = TrainConfig.from_dict(cfg["train"], seed=seed)
    return cfg, model_cfg, train_cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_experiment(seed: int = 42) -> None:
    set_seed(int(seed))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    warnings.filterwarnings("ignore", message=r"The epoch parameter in `scheduler\\.step\\(\\)` was not necessary.*")


def _safe_device(device: str) -> torch.device:
    dev = str(device or "cpu").lower().strip()
    return torch.device("cpu") if dev == "cuda" and not torch.cuda.is_available() else torch.device(dev)





@torch.no_grad()
def _latent_metrics(z: torch.Tensor, *, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    # For hard topK sparsity, activity should be measured directly from non-zero support.
    active = z.abs() > eps
    l0 = active.sum(dim=1).to(torch.float32)
    dead = (~active).all(dim=0)
    z_abs = z.abs()
    mean_act_active = (z_abs.sum(dim=1) / l0.clamp_min(1.0)).mean()
    return {
        "l0_mean": l0.mean(),
        "l0_median": l0.median(),
        "dead_frac": dead.to(torch.float32).mean(),
        "active_frac": active.to(torch.float32).mean(),
        "mean_act_active": mean_act_active,
        "mean_act_all": z_abs.mean(),
    }


def _write_metrics_row(metrics_csv: Path, row: Dict[str, object]) -> None:
    write_header = not metrics_csv.exists()
    with metrics_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_METRIC_COLUMNS))
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _METRIC_COLUMNS})


def fit_and_save_sae(X_train: np.ndarray, *, sae_cfg: SAEConfig, train_cfg: TrainConfig, out_dir: Path, X_val: Optional[np.ndarray] = None, wandb_run: Optional[Any] = None) -> Tuple[SparseAutoencoder, Optional[Path], Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    setup_experiment(seed=int(train_cfg.seed))
    device = _safe_device(train_cfg.device)
    pin_memory = device.type == "cuda"
    model = SparseAutoencoder(sae_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay))

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(np.asarray(X_train, dtype=np.float32))), batch_size=int(train_cfg.batch_size), shuffle=True, drop_last=False, pin_memory=pin_memory)
    dl_va = None if X_val is None else DataLoader(TensorDataset(torch.from_numpy(np.asarray(X_val, dtype=np.float32))), batch_size=int(train_cfg.batch_size), shuffle=False, drop_last=False, pin_memory=pin_memory)

    scheduler = None
    if str(train_cfg.scheduler_type).lower().strip() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(train_cfg.epochs) * max(1, len(dl_tr))), eta_min=float(train_cfg.scheduler_eta_min))

    use_amp = bool(train_cfg.amp) and device.type == "cuda"
    runtime = _Runtime(model=model, device=device, pin_memory=pin_memory, use_amp=use_amp, autocast_dtype=torch.float16 if use_amp else None, opt=opt, scaler=torch.amp.GradScaler(device=device.type, enabled=use_amp), scheduler=scheduler)

    start_epoch, best_ckpt, best_score = 0, None, None
    last_fp = ckpt_dir / "last.pt"

    if bool(train_cfg.use_saved_ckpt) and last_fp.exists():
        ckpt = torch.load(last_fp, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
        opt.load_state_dict(ckpt.get("opt_state", {}))
        if scheduler is not None and ckpt.get("sched_state") is not None:
            scheduler.load_state_dict(ckpt["sched_state"])
        if ckpt.get("scaler_state") is not None:
            runtime.scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score = ckpt.get("best_score", None)
        best_ckpt = Path(ckpt.get("best_ckpt")) if ckpt.get("best_ckpt") else None
    
    if not bool(train_cfg.use_saved_ckpt):
        for fp in ckpt_dir.glob("*.pt"):
            fp.unlink(missing_ok=True)
        if metrics_csv.exists():
            metrics_csv.unlink(missing_ok=True)

    t0 = time.time()
    for epoch in tqdm(range(start_epoch, int(train_cfg.epochs)), desc="training", leave=False):
        tr = runtime.run_epoch(dl=dl_tr, train=True, gate_weight=float(sae_cfg.gate_weight), orth_weight=float(sae_cfg.orth_weight), grad_clip=float(train_cfg.grad_clip))
        va = None if (dl_va is None or not bool(train_cfg.val_every_epoch)) else runtime.run_epoch(dl=dl_va, train=False, gate_weight=float(sae_cfg.gate_weight), orth_weight=float(sae_cfg.orth_weight), grad_clip=0.0)

        row = {
            "epoch": int(epoch),
            "train_loss": tr["loss"],
            "train_recon": tr["recon"],
            "train_gate_term": tr["gate_term"],  # WEIGHTED: gate_raw * gate_weight
            "train_orth_term": tr["orth_term"],  # WEIGHTED: orth_raw * orth_weight
            "train_gate_raw": tr["gate_raw"],    # UNWEIGHTED: raw regularizer value
            "train_orth_raw": tr["orth_raw"],    # UNWEIGHTED: raw regularizer value
            "train_l0_mean": tr["l0_mean"],
            "train_l0_median": tr["l0_median"],
            "train_dead_frac": tr["dead_frac"],
            "train_active_frac": tr["active_frac"],
            "train_mean_act_active": tr["mean_act_active"],
            "train_mean_act_all": tr["mean_act_all"],
            "val_loss": "" if va is None else va["loss"],
            "val_recon": "" if va is None else va["recon"],
            "val_gate_term": "" if va is None else va["gate_term"],  # WEIGHTED
            "val_orth_term": "" if va is None else va["orth_term"],  # WEIGHTED
            "val_gate_raw": "" if va is None else va["gate_raw"],    # UNWEIGHTED
            "val_orth_raw": "" if va is None else va["orth_raw"],    # UNWEIGHTED
            "val_l0_mean": "" if va is None else va["l0_mean"],
            "val_l0_median": "" if va is None else va["l0_median"],
            "val_dead_frac": "" if va is None else va["dead_frac"],
            "val_active_frac": "" if va is None else va["active_frac"],
            "val_mean_act_active": "" if va is None else va["mean_act_active"],
            "val_mean_act_all": "" if va is None else va["mean_act_all"],
            "lr": float(opt.param_groups[0]["lr"]),
            "runtime_s": float(time.time() - t0),
        }
        _write_metrics_row(metrics_csv, row)
        if wandb_run is not None:
            wandb_run.log({k: v for k, v in row.items() if v != ""}, step=int(epoch))

        score = float(tr["loss"]) if va is None else float(va["loss"])
        if best_score is None or score < float(best_score):
            best_score = score
            if bool(train_cfg.save_ckpt):
                best_ckpt = ckpt_dir / "best.pt"
                torch.save({"sae_cfg": asdict(sae_cfg), "train_cfg": asdict(train_cfg), "state_dict": model.state_dict(), "epoch": int(epoch), "best_score": float(best_score)}, best_ckpt)

        if bool(train_cfg.save_ckpt):
            torch.save({"sae_cfg": asdict(sae_cfg), "train_cfg": asdict(train_cfg), "state_dict": model.state_dict(), "opt_state": opt.state_dict(), "sched_state": scheduler.state_dict() if scheduler is not None else None, "scaler_state": runtime.scaler.state_dict() if use_amp else None, "epoch": int(epoch), "best_score": float(best_score) if best_score is not None else None, "best_ckpt": str(best_ckpt) if best_ckpt is not None else None}, last_fp)
            if int(train_cfg.ckpt_every_n_epochs) > 0 and ((epoch + 1) % int(train_cfg.ckpt_every_n_epochs) == 0):
                torch.save({"sae_cfg": asdict(sae_cfg), "train_cfg": asdict(train_cfg), "state_dict": model.state_dict(), "epoch": int(epoch)}, ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt")

    meta = {"sae_cfg": asdict(sae_cfg), "train_cfg": asdict(train_cfg), "n_train": int(np.asarray(X_train).shape[0]), "d_in": int(np.asarray(X_train).shape[1]), "runtime_s": float(time.time() - t0), "best_ckpt": str(best_ckpt) if best_ckpt is not None else None, "metrics_csv": str(metrics_csv)}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return model, best_ckpt, out_dir
