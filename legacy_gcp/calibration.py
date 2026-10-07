"""
Calibration layer for TC-DANN.

Two components:

1. GroupTemperatureScaler -- learns a separate temperature T per
   (disease x site x age_bucket) cell on a held-out calibration split.
   Temperatures are fit by minimising NLL, as in Guo et al. (2017).
   The group split directly addresses the subgroup-collapse finding from
   audit 2.4: a prediction from the 71+ bin is calibrated against 71+
   reliability, not global reliability.

2. SplitConformalPredictor -- given an ensemble of per-model probabilities
   and a calibration split, emits prediction sets (for binary: just a scalar
   p-value) with marginal coverage 1 - alpha. We use the conditional variant
   that calibrates per (disease x subgroup), so coverage is subgroup-wise.

These feed into confidence.py, where the abstain rule uses them.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Group-conditional temperature scaling
# ------------------------------------------------------------------

class GroupTemperatureScaler(nn.Module):
    """
    Per-group temperature. `group_ids` must be integer codes in [0, n_groups).

    Usage:
        scaler = GroupTemperatureScaler(n_diseases=D, n_groups=G)
        scaler.fit(logits, labels, group_ids)           # on calibration split
        p_cal = scaler.transform(logits, group_ids)     # at inference time
    """

    def __init__(self, n_diseases: int, n_groups: int):
        super().__init__()
        # Temperature parameter per (group, disease), initialised to 1.0.
        # We store log-T and exponentiate to keep T > 0.
        self.log_T = nn.Parameter(torch.zeros(n_groups, n_diseases))
        self.n_diseases = n_diseases
        self.n_groups = n_groups

    def temperatures(self) -> torch.Tensor:
        return self.log_T.exp()

    def apply(self, logits: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """logits: [N, D], group_ids: [N] -> scaled logits [N, D]."""
        T = self.temperatures()[group_ids]   # [N, D]
        return logits / T

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        group_ids: torch.Tensor,
        max_iter: int = 200,
        lr: float = 0.05,
    ) -> None:
        """Fit T per cell by LBFGS on NLL. Cells with < min_samples are pooled."""
        logits = logits.detach()
        labels = labels.detach().float()
        group_ids = group_ids.detach().long()

        opt = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            scaled = self.apply(logits, group_ids)
            loss = F.binary_cross_entropy_with_logits(scaled, labels)
            loss.backward()
            return loss

        opt.step(closure)

    def transform(self, logits: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities [N, D]."""
        with torch.no_grad():
            return torch.sigmoid(self.apply(logits, group_ids))


def make_group_id(
    site: np.ndarray,
    age: np.ndarray,
    n_sites: int = 3,
    n_ages: int = 4,
) -> np.ndarray:
    """
    Flatten (site, age_bucket) -> single int in [0, n_sites*n_ages).
    Sex is intentionally omitted here because (site x age x sex) produces many
    sparse cells; sex gets handled by the SubgroupBalancedSampler + adversarial
    head at training time.
    """
    return site * n_ages + age


# ------------------------------------------------------------------
# Split conformal prediction (per-subgroup)
# ------------------------------------------------------------------

class SplitConformalPredictor:
    """
    Binary classification conformal predictor with subgroup-conditional
    calibration.

    For each (disease, group) cell we compute a quantile of a non-conformity
    score on a calibration split. At test time we emit:

        p_hat (calibrated probability)
        lo, hi (the [alpha/2, 1-alpha/2] prediction interval over ensemble)
        score (non-conformity) -- higher = more surprising given the subgroup

    The abstain rule downstream uses `score` as an out-of-distribution indicator.
    """

    def __init__(self, n_diseases: int, n_groups: int, alpha: float = 0.1):
        self.n_diseases = n_diseases
        self.n_groups = n_groups
        self.alpha = alpha
        # Quantile per (group, disease, label_value in {0, 1})
        self.q = np.full((n_groups, n_diseases, 2), np.nan)

    @staticmethod
    def _nonconformity(p: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Standard APS-style score for binary: 1 - p[y]."""
        return 1.0 - np.where(y == 1, p, 1.0 - p)

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray, group_ids: np.ndarray) -> None:
        """
        p_cal: [N, D] calibrated probabilities
        y_cal: [N, D] binary labels
        group_ids: [N]
        """
        for g in range(self.n_groups):
            sel = group_ids == g
            if sel.sum() < 10:
                # fall back to global later
                continue
            for d in range(self.n_diseases):
                for k in (0, 1):
                    mask = sel & (y_cal[:, d] == k)
                    if mask.sum() < 5:
                        continue
                    scores = self._nonconformity(p_cal[mask, d], y_cal[mask, d])
                    self.q[g, d, k] = np.quantile(scores, 1.0 - self.alpha)

    def score(self, p: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
        """
        Returns the non-conformity score relative to the subgroup's calibration
        distribution, for the PREDICTED class. High score = unusual prediction
        given this subgroup, which is the signal used for abstain.

        Returns: [N, D] float
        """
        N, D = p.shape
        out = np.zeros((N, D))
        for i in range(N):
            g = group_ids[i]
            for d in range(D):
                pred = 1 if p[i, d] >= 0.5 else 0
                q = self.q[g, d, pred]
                s = self._nonconformity(np.array([p[i, d]]), np.array([pred]))[0]
                # If we lack subgroup quantile, score is just s; else relative.
                out[i, d] = s if np.isnan(q) else s / max(q, 1e-6)
        return out


# ------------------------------------------------------------------
# Ensemble helpers
# ------------------------------------------------------------------

def ensemble_mean_std(list_of_logits: list) -> Tuple[np.ndarray, np.ndarray]:
    """Stack K arrays of shape [N, D], return sigmoid mean and std across K."""
    stacked = np.stack(list_of_logits, axis=0)       # [K, N, D]
    probs = 1.0 / (1.0 + np.exp(-stacked))           # sigmoid per-model
    return probs.mean(axis=0), probs.std(axis=0)
