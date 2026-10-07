"""
Confidence and abstain layer for TC-DANN.

This is what converts a raw probability into a clinically actionable output.
A prediction is only returned when ALL four conditions hold:

  A1 Ensemble agreement:
         std across the K-model ensemble < sigma_max
     (if models disagree strongly, abstain)

  A2 Confounder-baseline disagreement:
         |p_acoustic - p_confounder| > delta_min
     (if the acoustic model agrees with a demographics-only LR, the
     prediction carries no incremental acoustic evidence over cohort
     structure, so abstain -- directly from audit Section 5.1)

  A3 Subgroup support:
         training set has >= n_support positives in the input's
         (disease x site x age x sex) cell
     (avoids the 71+ psychiatric_history = 0.189 failure mode)

  A4 Conformal support:
         non-conformity score within subgroup calibration quantile
     (input is not out-of-distribution for its subgroup)

When any check fails, the model returns an 'abstain' flag for that disease.
Downstream reporting should treat abstained predictions as 'insufficient
evidence' rather than as a probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class AbstainConfig:
    sigma_max: float = 0.15        # ensemble std threshold
    delta_min: float = 0.10        # minimum gap vs confounder baseline
    n_support: int = 30            # minimum positives in training subgroup cell
    conformal_tau: float = 1.5     # max relative non-conformity score


class AbstainPredictor:
    """
    Wraps:
      - ensemble mean probability
      - ensemble std
      - confounder-baseline probability (pre-computed LR)
      - conformal non-conformity score
      - subgroup support counts from training

    Produces per-(example, disease) (prob, abstain_flag, reason) triples.
    """

    def __init__(
        self,
        config: AbstainConfig,
        support_counts: np.ndarray,   # [n_groups, n_diseases] positives in train
    ):
        self.cfg = config
        self.support_counts = support_counts

    def predict(
        self,
        p_mean: np.ndarray,           # [N, D] calibrated ensemble mean
        p_std: np.ndarray,            # [N, D]
        p_confounder: np.ndarray,     # [N, D] predicted by demographics-only LR
        conformal_score: np.ndarray,  # [N, D] non-conformity
        group_ids: np.ndarray,        # [N]
    ) -> Dict[str, np.ndarray]:
        N, D = p_mean.shape
        abstain = np.zeros((N, D), dtype=bool)
        reason = np.full((N, D), "", dtype=object)

        # A1 ensemble agreement
        mask1 = p_std > self.cfg.sigma_max
        abstain |= mask1
        reason[mask1] = "ensemble_disagreement"

        # A2 confounder disagreement
        gap = np.abs(p_mean - p_confounder)
        mask2 = (~abstain) & (gap < self.cfg.delta_min)
        abstain |= mask2
        reason[mask2] = "agrees_with_confounder_baseline"

        # A3 subgroup support (per-group, per-disease count from training)
        support = self.support_counts[group_ids]      # [N, D]
        mask3 = (~abstain) & (support < self.cfg.n_support)
        abstain |= mask3
        reason[mask3] = "insufficient_subgroup_support"

        # A4 conformal
        mask4 = (~abstain) & (conformal_score > self.cfg.conformal_tau)
        abstain |= mask4
        reason[mask4] = "out_of_distribution_for_subgroup"

        return {"p": p_mean, "std": p_std, "abstain": abstain, "reason": reason}

    def summarise(self, result: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Abstain rate per disease + breakdown by reason."""
        abst = result["abstain"]
        N, D = abst.shape
        out = {
            "overall_abstain_rate": float(abst.mean()),
            "per_disease_abstain_rate": abst.mean(axis=0).tolist(),
        }
        reasons, counts = np.unique(result["reason"][abst], return_counts=True)
        out["reason_counts"] = dict(zip(reasons.tolist(), counts.tolist()))
        return out
