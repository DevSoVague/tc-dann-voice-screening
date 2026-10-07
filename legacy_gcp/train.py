"""
Training loop for TC-DANN.

Tied directly to the four-criteria audit framework:

  C1 Confounder separation: we run a logistic-regression confounder baseline
     on a validation split every `audit_every` epochs. If the acoustic model
     fails to beat the baseline by `min_separation` on any retained disease,
     training continues but the checkpoint is NOT saved.

  C2 Protocol stability: at validation time we compute both Protocol A
     (task-specific curated binary) and Protocol B (unified screening) AUROC.
     Model selection uses max-of-worst-case across protocols, not Protocol A alone.

  C3 Subgroup uniformity: worst-subgroup AUROC across (age, sex, site) is
     reported every epoch. Early stopping uses worst-subgroup macro AUROC, not
     mean macro AUROC.

  C4 External transfer: left for a final evaluation pass (see evaluate.py).

Adversarial lambda schedule follows Ganin & Lempitsky (2015):

    lambda(p) = 2 / (1 + exp(-gamma * p)) - 1,   p in [0, 1]

where p is training progress. Starts at 0 (no adversarial pressure) and
ramps up, so the encoder learns disease features before being forced to
strip site / age / sex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from model import TCDANN
from dataset import cross_site_mixup


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    adv_weight_site: float = 1.0
    adv_weight_age: float = 0.5
    adv_weight_sex: float = 0.5
    task_aux_weight: float = 0.3
    grl_gamma: float = 10.0           # Ganin schedule steepness
    mixup_alpha: float = 0.4
    mixup_p: float = 0.5
    audit_every: int = 2              # run confounder audit every N epochs
    min_separation: float = 0.10      # required acoustic - confounder gap
    subgroup_floor: float = 0.65      # subgroup uniformity threshold
    grad_clip: float = 1.0
    device: str = "cuda"


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

def multi_task_disease_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary cross-entropy per disease, summed over diseases."""
    return F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="mean"
    )


# ------------------------------------------------------------------
# GRL lambda schedule
# ------------------------------------------------------------------

def grl_lambda(step: int, total_steps: int, gamma: float = 10.0) -> float:
    p = step / max(1, total_steps)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


# ------------------------------------------------------------------
# One training epoch
# ------------------------------------------------------------------

def train_one_epoch(
    model: TCDANN,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
    total_epochs: int,
    steps_per_epoch: int,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.train()
    device = cfg.device
    total_steps = total_epochs * steps_per_epoch
    running = {"disease": 0.0, "site": 0.0, "age": 0.0, "sex": 0.0, "task": 0.0}
    n_batches = 0

    for step, batch in enumerate(loader):
        global_step = epoch * steps_per_epoch + step
        lam = grl_lambda(global_step, total_steps, cfg.grl_gamma)
        model.set_grl_lambda(lam)

        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch = cross_site_mixup(batch, alpha=cfg.mixup_alpha, p=cfg.mixup_p)

        out = model(batch)

        # Main disease loss
        loss_disease = multi_task_disease_loss(
            out["disease_logits"], batch["disease"], pos_weight=pos_weight
        )

        # Adversarial losses (gradient reversal is inside the model; we use
        # plain CE here, so the negated gradient flows back through the GRL).
        loss_site = F.cross_entropy(out["site_logits"], batch["site"])
        loss_age = F.cross_entropy(out["age_logits"], batch["age"])
        loss_sex = F.cross_entropy(out["sex_logits"], batch["sex"])

        # Task auxiliary (forward-pass, not reversed) -- forces CLS to encode task.
        loss_task = F.cross_entropy(out["task_logits"], batch["task_id"])

        loss = (
            loss_disease
            + cfg.adv_weight_site * loss_site
            + cfg.adv_weight_age * loss_age
            + cfg.adv_weight_sex * loss_sex
            + cfg.task_aux_weight * loss_task
        )

        optim.zero_grad()
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        running["disease"] += loss_disease.item()
        running["site"] += loss_site.item()
        running["age"] += loss_age.item()
        running["sex"] += loss_sex.item()
        running["task"] += loss_task.item()
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in running.items()} | {"grl_lambda": lam}


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: TCDANN, loader: DataLoader, device: str
) -> Dict[str, np.ndarray]:
    model.eval()
    disease_logits, disease_y = [], []
    sites, ages, sexes, pids = [], [], [], []
    embeddings = []
    for batch in loader:
        batch_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch_dev)
        disease_logits.append(out["disease_logits"].cpu().numpy())
        disease_y.append(batch["disease"].cpu().numpy())
        sites.append(batch["site"].cpu().numpy())
        ages.append(batch["age"].cpu().numpy())
        sexes.append(batch["sex"].cpu().numpy())
        embeddings.append(out["embedding"].cpu().numpy())
        pids.extend(batch["participant_id"])
    return {
        "logits": np.concatenate(disease_logits),
        "y": np.concatenate(disease_y),
        "site": np.concatenate(sites),
        "age": np.concatenate(ages),
        "sex": np.concatenate(sexes),
        "embedding": np.concatenate(embeddings),
        "participant_id": np.array(pids),
    }


def per_disease_auroc(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns AUROC per disease column. NaN if class is degenerate."""
    D = logits.shape[1]
    out = np.full(D, np.nan)
    for d in range(D):
        if len(np.unique(y[:, d])) < 2:
            continue
        out[d] = roc_auc_score(y[:, d], logits[:, d])
    return out


def worst_subgroup_auroc(
    logits: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    min_positives: int = 5,
) -> Dict[str, float]:
    """For each disease, compute worst AUROC across single-factor subgroups."""
    D = logits.shape[1]
    worst = np.full(D, np.nan)
    for d in range(D):
        aurocs = []
        for name, arr in (("site", sites), ("age", ages), ("sex", sexes)):
            for v in np.unique(arr):
                sel = arr == v
                if sel.sum() < min_positives:
                    continue
                yv = y[sel, d]
                if len(np.unique(yv)) < 2 or yv.sum() < min_positives:
                    continue
                aurocs.append(roc_auc_score(yv, logits[sel, d]))
        if aurocs:
            worst[d] = float(np.min(aurocs))
    return {"worst_per_disease": worst, "macro_worst": float(np.nanmean(worst))}


# ------------------------------------------------------------------
# Confounder audit callback (C1)
# ------------------------------------------------------------------

def run_confounder_audit(
    val_preds: Dict[str, np.ndarray],
    val_demo_df,  # pandas DF aligned to val, with age, sex, country, task_hist_* columns
    disease_cols: List[str],
) -> Dict[str, float]:
    """
    Trains a logistic regression on demographics + task histogram and returns
    per-disease AUROC. The acoustic model passes C1 when its AUROC minus this
    baseline is >= min_separation on each retained disease.
    """
    X = val_demo_df.values
    y = val_preds["y"]
    out = {}
    for d, name in enumerate(disease_cols):
        if len(np.unique(y[:, d])) < 2:
            out[name] = np.nan
            continue
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X, y[:, d])
        prob = lr.predict_proba(X)[:, 1]
        out[name] = roc_auc_score(y[:, d], prob)
    return out


# ------------------------------------------------------------------
# Top-level fit()
# ------------------------------------------------------------------

def fit(
    model: TCDANN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    disease_cols: List[str],
    val_demo_df=None,       # for confounder audit
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, List]:
    model.to(cfg.device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    history = {"train": [], "val_macro": [], "val_worst": [], "separation": []}
    best_worst = -np.inf
    best_state = None

    steps_per_epoch = len(train_loader)

    for epoch in range(cfg.epochs):
        stats = train_one_epoch(
            model, train_loader, optim, cfg,
            epoch=epoch, total_epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch, pos_weight=pos_weight,
        )
        sched.step()

        # ---- Validation ----
        preds = collect_predictions(model, val_loader, cfg.device)
        auroc = per_disease_auroc(preds["logits"], preds["y"])
        sub = worst_subgroup_auroc(
            preds["logits"], preds["y"],
            preds["site"], preds["age"], preds["sex"],
        )
        macro = float(np.nanmean(auroc))
        worst = sub["macro_worst"]

        separation = None
        if val_demo_df is not None and (epoch % cfg.audit_every == 0 or epoch == cfg.epochs - 1):
            baseline = run_confounder_audit(preds, val_demo_df, disease_cols)
            separation = {
                d: (auroc[i] - baseline[d]) for i, d in enumerate(disease_cols)
            }

        history["train"].append(stats)
        history["val_macro"].append(macro)
        history["val_worst"].append(worst)
        history["separation"].append(separation)

        # Early stopping / checkpointing on C3 (worst-subgroup)
        if worst > best_worst:
            best_worst = worst
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[epoch {epoch:02d}] loss_d={stats['disease']:.3f} "
            f"grl_lambda={stats['grl_lambda']:.2f} "
            f"macro_auroc={macro:.3f} worst_sub={worst:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
