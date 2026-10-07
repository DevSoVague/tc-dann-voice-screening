"""
Four-criteria validator for TC-DANN.

Direct implementation of audit Section 5.6:

  C1 Confounder separation: acoustic AUROC exceeds non-acoustic baseline by
     at least 0.10 per retained disease.

  C2 Protocol stability: |Protocol A AUROC - Protocol B AUROC| <= 0.10
     per retained disease.

  C3 Subgroup uniformity: every (disease x site) and (disease x age) and
     (disease x sex) cell with sufficient positives achieves AUROC >= 0.65.

  C4 External transfer: model AUROC on an external corpus (SVD / NeuroVoz /
     COUGHVID / MODMA) exceeds a random-guessing floor AND exceeds a
     demographics-only LR fitted on the external corpus.

Also includes `fair_participant_aggregate`, which collapses per-recording
predictions to per-participant only over tasks performed by both cases
and controls for the disease being evaluated. This is the deployment-correct
way to aggregate given the task-histogram shortcut (audit 2.2).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------------
# Fair aggregation
# ------------------------------------------------------------------

def fair_participant_aggregate(
    rec_df: pd.DataFrame,
    disease: str,
) -> pd.DataFrame:
    """
    rec_df must contain columns:
        participant_id, task, prob_<disease>, y_<disease>

    Returns a participant-level DataFrame with probability averaged ONLY over
    tasks that exist in both the positive and negative cohorts for this
    disease. Avoids the trivial "Parkinson's patients did task A, controls
    didn't, so task-A-existence = label" leak.
    """
    pos_tasks = set(rec_df.loc[rec_df[f"y_{disease}"] == 1, "task"])
    neg_tasks = set(rec_df.loc[rec_df[f"y_{disease}"] == 0, "task"])
    common = pos_tasks & neg_tasks
    sub = rec_df[rec_df["task"].isin(common)].copy()

    grouped = sub.groupby("participant_id").agg(
        prob=(f"prob_{disease}", "mean"),
        y=(f"y_{disease}", "max"),
        n_tasks=("task", "nunique"),
    ).reset_index()
    grouped["disease"] = disease
    return grouped


# ------------------------------------------------------------------
# Four criteria
# ------------------------------------------------------------------

@dataclass
class CriteriaThresholds:
    c1_min_separation: float = 0.10
    c2_max_protocol_gap: float = 0.10
    c3_subgroup_floor: float = 0.65
    c4_external_margin: float = 0.05


@dataclass
class CriterionResult:
    passed: bool
    value: float
    detail: Optional[dict] = None


def evaluate_c1_confounder(
    p_acoustic: np.ndarray,
    y: np.ndarray,
    demographic_features: np.ndarray,
    disease_cols: List[str],
    th: float = 0.10,
) -> Dict[str, CriterionResult]:
    out = {}
    for d, name in enumerate(disease_cols):
        if len(np.unique(y[:, d])) < 2:
            out[name] = CriterionResult(False, np.nan, {"reason": "degenerate"})
            continue
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(demographic_features, y[:, d])
        prob_conf = lr.predict_proba(demographic_features)[:, 1]
        auc_conf = roc_auc_score(y[:, d], prob_conf)
        auc_ac = roc_auc_score(y[:, d], p_acoustic[:, d])
        gap = auc_ac - auc_conf
        out[name] = CriterionResult(
            passed=bool(gap >= th),
            value=gap,
            detail={"acoustic": auc_ac, "confounder": auc_conf},
        )
    return out


def evaluate_c2_protocol(
    protA_auroc: Dict[str, float],
    protB_auroc: Dict[str, float],
    th: float = 0.10,
) -> Dict[str, CriterionResult]:
    out = {}
    for d in protA_auroc:
        if d not in protB_auroc:
            continue
        gap = abs(protA_auroc[d] - protB_auroc[d])
        out[d] = CriterionResult(
            passed=bool(gap <= th),
            value=gap,
            detail={"protA": protA_auroc[d], "protB": protB_auroc[d]},
        )
    return out


def evaluate_c3_subgroup(
    p_acoustic: np.ndarray,
    y: np.ndarray,
    site: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    disease_cols: List[str],
    th: float = 0.65,
    min_positives: int = 5,
) -> Dict[str, CriterionResult]:
    out = {}
    for d, name in enumerate(disease_cols):
        cell_aurocs = {}
        for fname, arr in (("site", site), ("age", age), ("sex", sex)):
            for v in np.unique(arr):
                sel = arr == v
                if sel.sum() < min_positives:
                    continue
                yv = y[sel, d]
                if len(np.unique(yv)) < 2 or yv.sum() < min_positives:
                    continue
                cell_aurocs[f"{fname}={v}"] = float(
                    roc_auc_score(yv, p_acoustic[sel, d])
                )
        if not cell_aurocs:
            out[name] = CriterionResult(False, np.nan, {"reason": "no_cells"})
            continue
        worst = min(cell_aurocs.values())
        out[name] = CriterionResult(
            passed=bool(worst >= th),
            value=worst,
            detail=cell_aurocs,
        )
    return out


def evaluate_c4_external(
    p_acoustic_ext: np.ndarray,
    y_ext: np.ndarray,
    demo_ext: np.ndarray,
    disease_cols: List[str],
    margin: float = 0.05,
) -> Dict[str, CriterionResult]:
    out = {}
    for d, name in enumerate(disease_cols):
        if p_acoustic_ext.shape[1] <= d or len(np.unique(y_ext[:, d])) < 2:
            out[name] = CriterionResult(False, np.nan, {"reason": "degenerate"})
            continue
        auc_ac = roc_auc_score(y_ext[:, d], p_acoustic_ext[:, d])
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(demo_ext, y_ext[:, d])
        prob_conf = lr.predict_proba(demo_ext)[:, 1]
        auc_conf = roc_auc_score(y_ext[:, d], prob_conf)
        out[name] = CriterionResult(
            passed=bool(auc_ac > 0.5 and auc_ac - auc_conf >= margin),
            value=auc_ac,
            detail={"external_acoustic": auc_ac, "external_confounder": auc_conf},
        )
    return out


# ------------------------------------------------------------------
# Top-level summary
# ------------------------------------------------------------------

def run_four_criteria(
    *,
    p_acoustic_internal: np.ndarray,
    y_internal: np.ndarray,
    demo_internal: np.ndarray,
    protA_auroc: Dict[str, float],
    protB_auroc: Dict[str, float],
    site: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    p_acoustic_external: Optional[np.ndarray] = None,
    y_external: Optional[np.ndarray] = None,
    demo_external: Optional[np.ndarray] = None,
    disease_cols: List[str] = None,
    thresholds: CriteriaThresholds = CriteriaThresholds(),
) -> Dict[str, dict]:
    c1 = evaluate_c1_confounder(
        p_acoustic_internal, y_internal, demo_internal, disease_cols,
        thresholds.c1_min_separation,
    )
    c2 = evaluate_c2_protocol(protA_auroc, protB_auroc, thresholds.c2_max_protocol_gap)
    c3 = evaluate_c3_subgroup(
        p_acoustic_internal, y_internal, site, age, sex, disease_cols,
        thresholds.c3_subgroup_floor,
    )
    c4 = None
    if p_acoustic_external is not None:
        c4 = evaluate_c4_external(
            p_acoustic_external, y_external, demo_external, disease_cols,
            thresholds.c4_external_margin,
        )

    return {
        "C1_confounder_separation": {k: asdict(v) for k, v in c1.items()},
        "C2_protocol_stability": {k: asdict(v) for k, v in c2.items()},
        "C3_subgroup_uniformity": {k: asdict(v) for k, v in c3.items()},
        "C4_external_transfer": {k: asdict(v) for k, v in c4.items()} if c4 else None,
    }
