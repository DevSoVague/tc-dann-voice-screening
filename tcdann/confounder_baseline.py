"""
confounder_baseline.py
======================
Live non-acoustic confounder baseline — mirrors Section 2.1 of the audit report.

Trains (or loads) a logistic regression on participant metadata only:
  age, sex, country, ethnicity, task_histogram
...and runs alongside MARVEL on every inference call.

If the demographic AUROC is within 0.10 of the model AUROC for any disease,
a shortcut warning is issued — matching Recommendation 5.1.

Usage (standalone):
    from confounder_baseline import ConfounderBaseline, build_confounder_vector

    baseline = ConfounderBaseline()
    baseline.load("confounder_baseline.joblib")       # or .fit(X, y, task_names)

    vec = build_confounder_vector(age=58, sex="Male", country="United States",
                                  ethnicity="White / Caucasian", tasks_completed=["simulate"]*4)
    preds = baseline.predict(vec)                     # {task_name: probability}
    flags = baseline.audit_flags(preds, marvel_preds) # list of shortcut warnings
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

log = logging.getLogger("voxclinbench.confounder")

# ── Constants ─────────────────────────────────────────────────────────────────

TASK_NAMES = [
    "parkinsons", "airway_stenosis", "laryngeal_dystonia", "vf_paralysis",
    "chronic_cough", "mtd", "benign_lesions", "glottic_insuff",
    "depression", "ptsd", "adhd", "bipolar", "cognitive_impairment",
    "psychiatric_history", "anxiety", "precancerous", "als",
    "copd_asthma", "laryngitis", "laryngeal_cancer",
]
N_TASKS = len(TASK_NAMES)

# Optional per-disease confounder AUROC priors for heuristic mode, loaded from a
# JSON file ({task_name: auroc}) produced by your own run of audit/confounder_analysis.py.
# Set CONFOUNDER_PRIORS_JSON to its path. Without it, a neutral 0.60 prior is used.
def _load_audit_priors() -> dict:
    import os
    path = os.environ.get("CONFOUNDER_PRIORS_JSON", "")
    if path and Path(path).is_file():
        try:
            return {str(k): float(v) for k, v in json.loads(Path(path).read_text()).items()}
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read CONFOUNDER_PRIORS_JSON: %s", e)
    return {}

AUDIT_REPORT_AUROC = _load_audit_priors()

# Shortcut flag threshold (Rec. 5.1: flag if confounder within 0.10 of model)
SHORTCUT_GAP_THRESHOLD = 0.10

# ── Feature builder ───────────────────────────────────────────────────────────

SEX_CODES     = {"Female": 0, "Male": 1, "Intersex / Other": 2}
COUNTRY_CODES = {
    "United States": 0, "Canada": 1, "Germany": 2, "Spain": 3,
    "United Kingdom": 4, "China": 5, "India": 6, "Other": 7,
}
ETHNICITY_CODES = {
    "White / Caucasian": 0, "Black / African American": 1,
    "Hispanic / Latino": 2, "Asian": 3, "Middle Eastern": 4,
    "Mixed / Other": 5, "Prefer not to say": 6,
}
# Task families — used to build the task histogram vector (one-hot + count)
TASK_BATTERY_CODES = {
    "Structural / Motor": 0,
    "Laryngeal / Vocal":  1,
    "Psychiatric / Cognitive": 2,
}
N_TASK_FEATURES = 3   # one value per battery family (count of tasks completed)

# Total feature vector length: age(1) + sex(1) + country(1) + ethnicity(1) + task_hist(3) = 7
FEATURE_DIM = 7


def build_confounder_vector(
    age: int | float,
    sex: str,
    country: str,
    ethnicity: str,
    task_battery: str,
    n_tasks_completed: int = 4,
) -> np.ndarray:
    """
    Build the [7] confounder feature vector from patient metadata.

    Features (matching audit report Section 2.1):
        [0] age (continuous, normalised 18-110)
        [1] sex code (0=F, 1=M, 2=other)
        [2] country code
        [3] ethnicity code
        [4] task_hist_structural  (1 if this battery, else 0) * n_tasks_completed
        [5] task_hist_laryngeal
        [6] task_hist_psychiatric
    """
    age_norm  = float(np.clip((age - 18) / (110 - 18), 0, 1))
    sex_code  = float(SEX_CODES.get(sex, 2))
    ctry_code = float(COUNTRY_CODES.get(country, 7))
    eth_code  = float(ETHNICITY_CODES.get(ethnicity, 6))

    task_hist = np.zeros(N_TASK_FEATURES, dtype=np.float32)
    bat_idx   = TASK_BATTERY_CODES.get(task_battery, -1)
    if bat_idx >= 0:
        task_hist[bat_idx] = float(n_tasks_completed)

    vec = np.array([age_norm, sex_code, ctry_code, eth_code], dtype=np.float32)
    return np.concatenate([vec, task_hist])  # shape [7]


# ── Confounder Baseline class ─────────────────────────────────────────────────

class ConfounderBaseline:
    """
    Non-acoustic confounder baseline for live audit alongside MARVEL.

    Two operating modes:
      1. Trained: load a joblib file produced by fit() — returns model predictions
      2. Heuristic: no joblib available — returns audit-report AUROC values as
         fixed probability estimates, with appropriate warnings
    """

    def __init__(self):
        self._models: dict | None = None   # task_name -> fitted LogisticRegression
        self._scalers: dict | None = None  # task_name -> StandardScaler
        self._mode: str = "heuristic"

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y_dict: dict[str, np.ndarray], task_names: list[str] | None = None):
        """
        Fit one logistic regression per task.

        Args:
            X: [N, 7] confounder feature matrix
            y_dict: {task_name: [N] binary label array}
            task_names: subset of tasks to train (None = all with data in y_dict)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self._models  = {}
        self._scalers = {}
        names = task_names or list(y_dict.keys())

        for task in names:
            if task not in y_dict:
                continue
            y = y_dict[task]
            if len(np.unique(y)) < 2:
                log.warning("Task %s has only one class — skipping", task)
                continue

            scaler = StandardScaler()
            Xs     = scaler.fit_transform(X)

            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )
            clf.fit(Xs, y)
            self._models[task]  = clf
            self._scalers[task] = scaler
            log.info("Fitted confounder baseline for task: %s", task)

        self._mode = "trained"
        return self

    def save(self, path: str | Path):
        path = Path(path)
        payload = {
            "models":  self._models,
            "scalers": self._scalers,
            "mode":    self._mode,
        }
        joblib.dump(payload, path)
        log.info("Saved confounder baseline to %s", path)

    def load(self, path: str | Path) -> "ConfounderBaseline":
        path = Path(path)
        if not path.exists():
            log.warning("Confounder baseline not found at %s — using heuristic mode", path)
            return self
        payload      = joblib.load(path)
        self._models  = payload.get("models")
        self._scalers = payload.get("scalers")
        self._mode    = payload.get("mode", "trained")
        log.info("Loaded confounder baseline from %s (mode=%s)", path, self._mode)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, confounder_vec: np.ndarray) -> dict[str, float]:
        """
        Return {task_name: probability} for all 20 tasks.

        In trained mode: runs the fitted logistic regressions.
        In heuristic mode: returns audit-report AUROC values as probability proxies
        (these represent how well demographics predict each disease, not a true
        per-patient probability — clearly labelled as such in the output).
        """
        if self._mode == "trained" and self._models:
            return self._predict_trained(confounder_vec)
        else:
            return self._predict_heuristic(confounder_vec)

    def _predict_trained(self, vec: np.ndarray) -> dict[str, float]:
        results = {}
        x = vec.reshape(1, -1)
        for task in TASK_NAMES:
            if task not in self._models:
                results[task] = 0.5  # no model for this task
                continue
            scaler = self._scalers[task]
            model  = self._models[task]
            xs     = scaler.transform(x)
            prob   = float(model.predict_proba(xs)[0, 1])
            results[task] = round(prob, 4)
        return results

    def _predict_heuristic(self, vec: np.ndarray) -> dict[str, float]:
        """
        Heuristic mode: return a confounder AUROC prior (from CONFOUNDER_PRIORS_JSON, else 0.60).
        Modulates slightly by age and country to give per-patient variation.
        """
        age_norm  = float(vec[0])   # 0-1
        ctry_code = float(vec[2])   # 0=USA, 1=Canada

        results = {}
        for task in TASK_NAMES:
            base = AUDIT_REPORT_AUROC.get(task, 0.60)
            # Slight adjustment for age (older patients -> higher psychiatric confounding)
            if task in ("depression", "ptsd", "adhd", "anxiety", "psychiatric_history"):
                base += (age_norm - 0.4) * 0.05
            # Canada bias for cognitive impairment (all positives are Canadian in B2AI v3)
            if task == "cognitive_impairment" and ctry_code == 1:
                base = min(0.99, base + 0.03)
            results[task] = round(float(np.clip(base, 0.01, 0.99)), 4)
        return results

    # ── Audit flags (Rec. 5.1) ────────────────────────────────────────────────

    def audit_flags(
        self,
        confounder_probs: dict[str, float],
        model_probs: dict[str, float],
    ) -> list[dict]:
        """
        Compare confounder and model probabilities.
        Returns audit flags per Recommendation 5.1:
          - If |confounder_prob - model_prob| < SHORTCUT_GAP_THRESHOLD → shortcut warning
          - If confounder_prob > model_prob → baseline exceeds model warning
        """
        flags = []
        mode_note = "(heuristic AUROC proxy)" if self._mode == "heuristic" else "(trained LR)"

        for task in TASK_NAMES:
            cp = confounder_probs.get(task, 0.5)
            mp = model_probs.get(task, 0.5)
            gap = mp - cp

            if cp > mp:
                flags.append({
                    "task":    task,
                    "level":   "error",
                    "type":    "baseline_exceeds_model",
                    "msg":     (f"{task}: confounder baseline {cp:.3f} EXCEEDS model {mp:.3f} "
                                f"{mode_note} -- no evidence of acoustic signal"),
                    "confounder_prob": cp,
                    "model_prob":      mp,
                    "gap":             round(gap, 4),
                })
            elif gap < SHORTCUT_GAP_THRESHOLD:
                flags.append({
                    "task":    task,
                    "level":   "warn",
                    "type":    "shortcut_risk",
                    "msg":     (f"{task}: gap = {gap:.3f} (threshold {SHORTCUT_GAP_THRESHOLD}) "
                                f"-- shortcut not ruled out {mode_note}"),
                    "confounder_prob": cp,
                    "model_prob":      mp,
                    "gap":             round(gap, 4),
                })

        return flags

    @property
    def mode(self) -> str:
        return self._mode

    def summary(self) -> dict:
        return {
            "mode":        self._mode,
            "n_tasks_fit": len(self._models) if self._models else 0,
            "feature_dim": FEATURE_DIM,
            "threshold":   SHORTCUT_GAP_THRESHOLD,
        }
