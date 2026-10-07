"""
run_tc_dann.py — Quad-model TC-DANN training & evaluation
Bridge2AI-Voice v6.0.0  |  features/ directory only (no phenotype)

Four independent TCDANN models, one per clinical domain:

  MODEL 1 · Voice+Oncological
      Diseases : laryngeal_dystonia, unilateral_vocal_fold_paralysis,
                 benign_lesions, muscle_tension_dysphonia,
                 glottic_insufficiency, laryngeal_cancer,
                 precancerous_lesions, control
      Features : ALL (MFCC, Mel, spectrogram, pitch, loudness,
                 periodicity, EMA, static)

  MODEL 2 · Neurological
      Diseases : parkinsons_disease, cognitive_impairment,
                 amyotrophic_lateral_sclerosis
      Features : ALL

  MODEL 3 · Respiratory
      Diseases : airway_stenosis, copd_and_asthma,
                 unexplained_chronic_cough, laryngitis
      Features : ALL

  MODEL 4 · Psychiatric
      Diseases : adhd_adult, ptsd_adult, anxiety, depression,
                 bipolar_disorder
      Features : ALL except PPG.
                 MFCC, Mel, and spectrogram are KEPT.
                 The earlier "zero spectrograms helps" finding was a
                 cross-domain interference artefact from the old
                 unified model.  Separate models eliminate that
                 interference, so spectral features contribute freely.

  Removed: psychiatric_history — too heterogeneous; contaminates
           every psychiatric head it shares a model with.
  All models: PPG excluded (zero contribution in all domains).

Composite confidence score  confidence = a·x + b·y − c·z + d·m + e·p
  x  subgroup cosine similarity — top-k kNN in train subgroup (k=10).
     Sparsity fallback: subgroups < MIN_SUBGROUP_FOR_COSINE → x = 0.5.
  y  model head sigmoid probability.
  z  diagnostic uncertainty — max x-score of the 2 runner-up diseases
     within the SAME model only (no cross-model leakage).
  m  demographic prior alignment.
     Always 0 for Psychiatric (shortcut contamination per audit).
  p  confounder robustness = 1 − task_confounder_auroc.

  Separate weight sets for physical vs psychiatric models.
  worst_task_auroc is NaN (not 1.0) when fewer than 2 task subgroups
  have enough samples — eliminates the spurious perfect-score sentinel.
  n_train_subgroup added to audit table for interpretability.

  Unified cross-model top-3: all four models are pooled per participant,
  re-ranked globally by confidence, top-3 selected across models.

Usage:
    python run_tc_dann.py --data_root /path/to/my_model --epochs 40

Leakage fixes (v3.1.0, carried forward):
  1. Participant-level 80/20 train/test split before ANY fitting.
  2. SimpleImputer fitted only on train rows.
  3. StandardScaler fitted only on train rows.
  4. confounder_baseline_auroc on test split only.

New in v6.0.0:
  5. worst_task_auroc: NaN sentinel instead of 1.0.
  6. n_train_subgroup in audit table.
  7. Sparse-subgroup cosine fallback to neutral 0.5.
  8. z-penalty scoped within model only.
  9. Per-model confidence weight sets.
 10. Unified cross-model top-3.
 11. psychiatric_history removed.
 12. MFCC/Mel/spectrogram kept for all models; only PPG excluded.
"""

import argparse
import os
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

# ── dependency check ──────────────────────────────────────────────────────────
MISSING = []
for pkg in ["numpy", "pandas", "pyarrow", "sklearn", "torch", "joblib"]:
    try:
        __import__(pkg)
    except ImportError:
        MISSING.append(pkg)
if MISSING:
    sys.exit(f"Missing packages: {', '.join(MISSING)}\n"
             f"Run: pip install {' '.join(MISSING)}")

import hashlib
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# psychiatric_history removed.
ALL_DISEASES = [
    "laryngeal_dystonia", "unilateral_vocal_fold_paralysis", "benign_lesions",
    "muscle_tension_dysphonia", "glottic_insufficiency",
    "parkinsons_disease", "cognitive_impairment", "amyotrophic_lateral_sclerosis",
    "airway_stenosis", "copd_and_asthma", "unexplained_chronic_cough", "laryngitis",
    "adhd_adult", "ptsd_adult", "anxiety", "depression", "bipolar_disorder",
    "laryngeal_cancer", "precancerous_lesions", "control",
]

DISEASE_CATEGORY = {
    "laryngeal_dystonia":              "Voice",
    "unilateral_vocal_fold_paralysis": "Voice",
    "benign_lesions":                  "Voice",
    "muscle_tension_dysphonia":        "Voice",
    "glottic_insufficiency":           "Voice",
    "parkinsons_disease":              "Neurological",
    "cognitive_impairment":            "Neurological",
    "amyotrophic_lateral_sclerosis":   "Neurological",
    "airway_stenosis":                 "Respiratory",
    "copd_and_asthma":                 "Respiratory",
    "unexplained_chronic_cough":       "Respiratory",
    "laryngitis":                      "Respiratory",
    "adhd_adult":                      "Psychiatric",
    "ptsd_adult":                      "Psychiatric",
    "anxiety":                         "Psychiatric",
    "depression":                      "Psychiatric",
    "bipolar_disorder":                "Psychiatric",
    "laryngeal_cancer":                "Oncological",
    "precancerous_lesions":            "Oncological",
    "control":                         "Control",
}

# ── Four-model routing ────────────────────────────────────────────────────────
MODEL_DISEASE_MAP: dict[str, list[str]] = {
    "voice_onco": [
        "laryngeal_dystonia", "unilateral_vocal_fold_paralysis", "benign_lesions",
        "muscle_tension_dysphonia", "glottic_insufficiency",
        "laryngeal_cancer", "precancerous_lesions", "control",
    ],
    "neurological": [
        "parkinsons_disease", "cognitive_impairment", "amyotrophic_lateral_sclerosis",
    ],
    "respiratory": [
        "airway_stenosis", "copd_and_asthma",
        "unexplained_chronic_cough", "laryngitis",
    ],
    "psychiatric": [
        "adhd_adult", "ptsd_adult", "anxiety", "depression", "bipolar_disorder",
    ],
}

MODEL_DISPLAY = {
    "voice_onco":   "Voice+Onco",
    "neurological": "Neurological",
    "respiratory":  "Respiratory",
    "psychiatric":  "Psychiatric",
}

# All models exclude PPG.  MFCC/Mel/spectrogram kept for all models.
PPG_EXCLUDED_PREFIXES = ("ppg", "ppgs")

# Psychiatric diseases must not receive a demographic boost.
PSYCHIATRIC_NO_DEMO = set(MODEL_DISEASE_MAP["psychiatric"])

# Confidence score default weights — physical vs psychiatric
# Physical: large feature space → cosine informative (higher a)
# Psychiatric: narrower space; confounder penalty more important (higher e)
CS_PHYSICAL = dict(a=0.25, b=0.40, c=0.15, d=0.10, e=0.10)
CS_PSYCH    = dict(a=0.15, b=0.35, c=0.20, d=0.00, e=0.30)
# b reduced 0.50→0.35: raw head prob was dominant on thin anxiety/depression data
# c raised 0.15→0.20: stronger penalty for a high-confidence wrong runner-up
# e raised 0.20→0.30: confounder robustness weighted more in psychiatric domain

# Sparse-subgroup fallback for cosine similarity (x term)
MIN_SUBGROUP_FOR_COSINE = 10
COSINE_K                = 10

DEMO_COLS = ["age", "sex", "bmi"]

TASKS    = ["prolonged-vowel", "read-speech", "free-speech", "diadochokinesis"]
TASK_ENC = {t: i for i, t in enumerate(TASKS)}

ARRAY_COLS = {"mfcc", "mel_spectrogram", "spectrograms", "ppgs",
              "ema", "pitch", "loudness", "periodicity"}

QC_N_FRAMES_MIN = 10
QC_PERIODICITY  = 0.1
QC_LOUDNESS_DB  = -60.0
EMA_MAX_CHANNELS = 12

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & AGGREGATE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _stream_array_col(path: Path, col: str, batch_size: int = 500,
                      max_rows: int = None) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    records = []
    scalar_cols = [f.name for f in pf.schema_arrow if f.name not in ARRAY_COLS]
    for batch in pf.iter_batches(batch_size=batch_size):
        for i in range(batch.num_rows):
            if max_rows and len(records) >= max_rows:
                break
            row = {c: batch.column(c)[i].as_py() for c in scalar_cols}
            arr_val = batch.column(col)[i]
            if arr_val is not None:
                arr = np.array(arr_val.as_py(), dtype=np.float32)
                if arr.ndim == 1:
                    row[f"{col}_mean"] = float(arr.mean())
                    row[f"{col}_std"]  = float(arr.std())
                    row[f"{col}_min"]  = float(arr.min())
                elif arr.ndim == 2:
                    for ch in range(min(arr.shape[0], EMA_MAX_CHANNELS)):
                        row[f"ema_ch{ch:02d}"] = float(arr[ch].mean())
            records.append(row)
        if max_rows and len(records) >= max_rows:
            break
    return pd.DataFrame(records)


def load_mfcc(path: Path, batch_size: int = 500, max_rows=None) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    records = []
    meta_cols = ["participant_id", "session_id", "task_name", "n_frames"]
    for _, batch in enumerate(pf.iter_batches(batch_size=batch_size,
                                              columns=meta_cols + ["mfcc"])):
        for i in range(batch.num_rows):
            if max_rows and len(records) >= max_rows:
                break
            meta = {c: batch.column(c)[i].as_py() for c in meta_cols}
            mfcc = np.array(batch.column("mfcc")[i].as_py(), dtype=np.float32)
            if mfcc.ndim != 2 or mfcc.shape[0] != 60:
                continue
            row = dict(meta)
            for stat, fn in [("mean", np.mean), ("std", np.std),
                              ("max",  np.max),  ("min", np.min)]:
                for ci, v in enumerate(fn(mfcc, axis=1)):
                    row[f"mfcc{ci:02d}_{stat}"] = float(v)
            if mfcc.shape[1] > 2:
                d1 = np.diff(mfcc, axis=1)
                for ci in range(60):
                    row[f"mfcc{ci:02d}_delta_mean"] = float(d1[ci].mean())
                    row[f"mfcc{ci:02d}_delta_std"]  = float(d1[ci].std())
                d2 = np.diff(d1, axis=1)
                if d2.shape[1] > 0:
                    for ci in range(60):
                        row[f"mfcc{ci:02d}_deltadelta_mean"] = float(d2[ci].mean())
                        row[f"mfcc{ci:02d}_deltadelta_std"]  = float(d2[ci].std())
            records.append(row)
        if max_rows and len(records) >= max_rows:
            break
    print(f"  MFCC: loaded {len(records):,} recordings")
    return pd.DataFrame(records)


def load_static(path: Path) -> pd.DataFrame:
    df   = pd.read_csv(path, sep="\t")
    orig = df.shape[1]
    # Protect demographic and identifier columns from any filtering
    _protect = {"participant_id", "session_id", "task_name",
                "age", "sex", "bmi", "site_id"}
    drop_high_null = [c for c in df.columns
                      if c not in _protect and df[c].isnull().mean() > 0.80]
    df = df.drop(columns=drop_high_null)
    num  = df.select_dtypes(include="number").columns
    near_const = [c for c in num
                  if c not in _protect and
                  (df[c].std(skipna=True) == 0 or df[c].nunique(dropna=True) <= 1)]
    df.drop(columns=near_const, inplace=True)
    print(f"  Static features: {orig} → {df.shape[1]} cols "
          f"(dropped {orig - df.shape[1]})")
    return df


def load_all_features(data_root: Path, max_mfcc_rows=None) -> pd.DataFrame:
    feat = data_root / "features"
    J    = ["participant_id", "session_id", "task_name"]

    print("\n[1/6] Loading MFCC features (streaming 2.3 GB)...")
    mfcc_df   = load_mfcc(feat / "torchaudio_mfcc.parquet", max_rows=max_mfcc_rows)
    print("[2/6] Loading SPARC pitch...")
    pitch_df  = _stream_array_col(feat / "sparc_pitch.parquet",       "pitch",
                                  max_rows=max_mfcc_rows)
    print("[3/6] Loading SPARC loudness...")
    loud_df   = _stream_array_col(feat / "sparc_loudness.parquet",    "loudness",
                                  max_rows=max_mfcc_rows)
    print("[4/6] Loading SPARC periodicity...")
    period_df = _stream_array_col(feat / "sparc_periodicity.parquet", "periodicity",
                                  max_rows=max_mfcc_rows)
    print("[5/6] Loading SPARC EMA...")
    ema_df    = _stream_array_col(feat / "sparc_ema.parquet",         "ema",
                                  max_rows=max_mfcc_rows)
    print("[6/6] Loading static features (TSV)...")
    static_df = load_static(feat / "static_features.tsv")

    def _coerce_pid(df):
        if "participant_id" in df.columns:
            df = df.copy()
            df["participant_id"] = df["participant_id"].astype(str)
        return df

    mfcc_df, pitch_df, loud_df, period_df, ema_df, static_df = (
        _coerce_pid(x) for x in
        [mfcc_df, pitch_df, loud_df, period_df, ema_df, static_df])

    KEEP = set(J)
    print("\nJoining feature tables...")
    master = mfcc_df.copy()
    for name, df, key in [("pitch",       pitch_df,  J),
                           ("loudness",    loud_df,   J),
                           ("periodicity", period_df, J),
                           ("ema",         ema_df,    J)]:
        if df.empty or len(df.columns) == 0:
            print(f"  + {name}: skipped (empty)")
            continue
        avail_key = [k for k in key if k in df.columns]
        if not avail_key:
            print(f"  + {name}: skipped (no join keys)")
            continue
        # Deduplicate on join keys before merging — prevents row explosion
        # when SPARC parquets have duplicate session_id values.
        df = df.drop_duplicates(subset=avail_key, keep="first")
        dup_cols  = [c for c in df.columns
                     if c in master.columns and c not in avail_key and c not in KEEP]
        master = master.merge(df.drop(columns=dup_cols, errors="ignore"),
                              on=avail_key, how="left")
        print(f"  + {name}: {master.shape}")

    if "participant_id" in static_df.columns:
        # If static_features has session_id (per-recording, e.g. SVD openSMILE output),
        # merge on participant_id + session_id to preserve all recordings.
        # If it only has participant_id (per-participant), deduplicate and merge on that.
        if "session_id" in static_df.columns:
            static_clean = static_df
            merge_on = [k for k in ["participant_id", "session_id"] if k in static_clean.columns]
        else:
            static_clean = static_df.drop_duplicates(subset="participant_id")
            merge_on = ["participant_id"]
        _prot = {"task_name", "session_id", "n_frames", "any_quality_flag",
                 "flag_short", "flag_aperiodic", "flag_near_silent", "flag_zero_mfcc"}
        conflicts = [c for c in static_clean.columns
                     if c not in merge_on and c in master.columns and c in _prot]
        master = master.merge(static_clean.drop(columns=conflicts, errors="ignore"),
                              on=merge_on, how="left")
        print(f"  + static: {master.shape}")

    master["flag_short"] = master["n_frames"] < QC_N_FRAMES_MIN
    master["flag_aperiodic"] = (
        master.get("periodicity_mean", pd.Series(1.0, index=master.index))
        < QC_PERIODICITY)
    master["flag_near_silent"] = (
        master.get("loudness_mean", pd.Series(0.0, index=master.index))
        < QC_LOUDNESS_DB)
    master["flag_zero_mfcc"]   = master["mfcc00_mean"] == 0.0
    master["any_quality_flag"] = (
        master["flag_short"] | master["flag_aperiodic"] |
        master["flag_near_silent"] | master["flag_zero_mfcc"])

    n_flagged = master["any_quality_flag"].sum()
    print(f"\nTotal recordings : {len(master):,}")
    print(f"Quality flagged  : {n_flagged:,} ({100*n_flagged/len(master):.1f}%)")
    return master


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LABELS
# ─────────────────────────────────────────────────────────────────────────────

def load_labels_from_diagnosis_tsvs(data_root: Path) -> pd.DataFrame:
    diag_dir = data_root / "phenotype" / "diagnosis"
    records  = {}
    for disease in ALL_DISEASES:
        tsv = diag_dir / f"{disease}.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t", usecols=lambda c: c == "participant_id",
                         dtype=str, on_bad_lines="skip")
        if "participant_id" not in df.columns:
            df = pd.read_csv(tsv, sep="\t", dtype=str, on_bad_lines="skip")
            if df.shape[1] > 0:
                df.columns = ["participant_id"] + list(df.columns[1:])
        for pid in df["participant_id"].dropna().str.strip().unique():
            records.setdefault(pid, {})[disease] = 1

    label_df = (pd.DataFrame.from_dict(records, orient="index")
                .reset_index()
                .rename(columns={"index": "participant_id"}))
    label_df["participant_id"] = label_df["participant_id"].astype(str)
    label_df = label_df.fillna(0).infer_objects(copy=False)
    for d in ALL_DISEASES:
        if d not in label_df.columns:
            label_df[d] = 0

    print(f"\nLabels loaded: {len(label_df):,} participants")
    for d in ALL_DISEASES:
        n = int(label_df[d].sum())
        if n > 0:
            print(f"  {d:<45} n={n:>4}")
    return label_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE COLUMN PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def build_feat_cols(merged: pd.DataFrame, disease_cols: list[str]) -> list[str]:
    """
    All float columns minus identifiers, flags, labels, and PPG.
    MFCC / Mel / spectrogram columns are KEPT for all models.
    """
    skip = {"participant_id", "session_id", "task_name", "n_frames",
            "flag_short", "flag_aperiodic", "flag_near_silent",
            "flag_zero_mfcc", "any_quality_flag", "site_id"} | set(disease_cols)
    _float_dtypes = {np.float64, np.float32,
                     np.dtype("float64"), np.dtype("float32")}
    feat_cols = [
        c for c in merged.columns
        if c not in skip
        and merged[c].dtype in _float_dtypes
        and not c.lower().startswith(PPG_EXCLUDED_PREFIXES)
    ]
    all_nan = [c for c in feat_cols if merged[c].isna().all()]
    if all_nan:
        feat_cols = [c for c in feat_cols if c not in all_nan]
        print(f"  NOTE: dropped {len(all_nan)} all-NaN feature columns")
    check = merged[feat_cols].select_dtypes(include="number")
    if check.shape[1] != len(feat_cols):
        feat_cols = list(check.columns)
        print(f"  NOTE: trimmed feat_cols to {len(feat_cols)} fully-numeric columns")
    return feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MODEL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        lam = ctx.saved_tensors[0].item()
        return -lam * grad, None


def grad_reverse(x, lam=1.0):
    return GradientReversalFn.apply(x, lam)


class FiLM(nn.Module):
    def __init__(self, task_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma = nn.Linear(task_emb_dim, hidden_dim)
        self.beta  = nn.Linear(task_emb_dim, hidden_dim)

    def forward(self, h, task_emb):
        return self.gamma(task_emb) * h + self.beta(task_emb)


class TCDANN(nn.Module):
    """
    Task-Conditioned Domain-Adversarial Neural Network.
    Instantiated independently for each clinical domain.

      Input (F features)
        → FiLM-modulated encoder FC→512→256→128
          (BN + ReLU + Dropout 0.35, task embedding per layer)
        → 128-dim site-invariant repr
             ↓ disease heads              ↓ site adversary
          D × FC(128→1)           GradRev → FC(128→64→n_sites)
    """
    def __init__(self, n_features, n_tasks=4, n_sites=5, n_diseases=4,
                 hidden=(512, 256, 128), task_emb_dim=32, dropout=0.35):
        super().__init__()
        self.task_emb = nn.Embedding(n_tasks, task_emb_dim)
        self.enc  = nn.ModuleList()
        self.film = nn.ModuleList()
        self.bn   = nn.ModuleList()
        in_d = n_features
        for out_d in hidden:
            self.enc.append(nn.Linear(in_d, out_d))
            self.film.append(FiLM(task_emb_dim, out_d))
            self.bn.append(nn.BatchNorm1d(out_d))
            in_d = out_d
        self.drop = nn.Dropout(dropout)
        repr_d = hidden[-1]
        self.site_head = nn.Sequential(
            nn.Linear(repr_d, 64), nn.ReLU(), nn.Linear(64, n_sites))
        self.disease_heads = nn.ModuleList(
            [nn.Linear(repr_d, 1) for _ in range(n_diseases)])

    def encode(self, x, task_ids):
        te = self.task_emb(task_ids)
        h  = x
        for layer, film, bn in zip(self.enc, self.film, self.bn):
            h = film(bn(F.relu(layer(h))), te)
            h = self.drop(h)
        return h

    def forward(self, x, task_ids, lam=1.0):
        h = self.encode(x, task_ids)
        disease_logits = torch.cat([head(h) for head in self.disease_heads], dim=1)
        site_logits    = self.site_head(grad_reverse(h, lam))
        return disease_logits, site_logits


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DATASET
# ─────────────────────────────────────────────────────────────────────────────

class VoiceDataset(Dataset):
    def __init__(self, df, feat_cols, disease_cols,
                 imputer=None, scaler=None, fit_scaler=False):
        if imputer is None:
            self.imputer = SimpleImputer(strategy="median")
            X = self.imputer.fit_transform(df[feat_cols].values.astype(np.float32))
        else:
            self.imputer = imputer
            X = imputer.transform(df[feat_cols].values.astype(np.float32))

        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X).astype(np.float32)
        elif scaler is not None:
            self.scaler = scaler
            self.X = scaler.transform(X).astype(np.float32)
        else:
            self.scaler = None
            self.X = X

        self.task_ids = (df["task_name"].map(TASK_ENC)
                         .fillna(0).values.astype(np.int64))

        # site_id may be a string (e.g. "svd", "site_01") or already an int.
        # Encode to a contiguous integer index regardless of input type.
        raw_site = df.get("site_id", pd.Series(0, index=df.index)).fillna(0)
        if raw_site.dtype == object or raw_site.dtype.name == "string":
            unique_sites = {s: i for i, s in enumerate(sorted(raw_site.unique()))}
            self.site_ids = raw_site.map(unique_sites).fillna(0).values.astype(np.int64)
        else:
            self.site_ids = raw_site.values.astype(np.int64)
        self.Y = df[disease_cols].fillna(0).values.astype(np.float32)

    def __len__(self):  return len(self.X)

    def __getitem__(self, idx):
        return {"x":       torch.tensor(self.X[idx]),
                "task_id": torch.tensor(self.task_ids[idx]),
                "site_id": torch.tensor(self.site_ids[idx]),
                "y":       torch.tensor(self.Y[idx])}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def dann_lambda(epoch, total, gamma=10.0):
    p = epoch / max(total, 1)
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


def train_one_epoch(model, loader, opt, lam, alpha=0.5):
    model.train()
    tot = dis = site_l = 0.0
    for b in loader:
        x, t, s, y = (b["x"].to(DEVICE), b["task_id"].to(DEVICE),
                      b["site_id"].to(DEVICE), b["y"].to(DEVICE))
        d_logits, s_logits = model(x, t, lam)
        d_loss = F.binary_cross_entropy_with_logits(d_logits, y)
        s_loss = F.cross_entropy(s_logits, s)
        loss   = d_loss + alpha * s_loss
        opt.zero_grad(); loss.backward(); opt.step()
        tot    += loss.item()
        dis    += d_loss.item()
        site_l += s_loss.item()
    n = len(loader)
    return tot / n, dis / n, site_l / n


def train_model(label, train_df, feat_cols, disease_cols, n_sites, args):
    dataset = VoiceDataset(train_df, feat_cols, disease_cols, fit_scaler=True)
    assert dataset.X.shape[1] == len(feat_cols), (
        f"[{label}] Tensor width {dataset.X.shape[1]} != feat_cols {len(feat_cols)}")

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, drop_last=True, num_workers=0)
    model  = TCDANN(n_features=len(feat_cols), n_tasks=len(TASKS),
                    n_sites=n_sites, n_diseases=len(disease_cols)).to(DEVICE)

    print(f"  [{label}] params={sum(p.numel() for p in model.parameters()):,}  "
          f"features={len(feat_cols)}  heads={len(disease_cols)}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_loss, best_state = float("inf"), None
    for epoch in range(args.epochs):
        lam = dann_lambda(epoch, args.epochs)
        loss, dis, site_l = train_one_epoch(model, loader, opt, lam)
        sched.step()
        if dis < best_loss:
            best_loss  = dis
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            marker = " ← best" if dis == best_loss else ""
            print(f"  [{label}] Epoch {epoch+1:>3}/{args.epochs}  "
                  f"total={loss:.4f}  disease={dis:.4f}  "
                  f"site={site_l:.4f}  λ={lam:.3f}{marker}")

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"  [{label}] Best disease loss: {best_loss:.4f}")
    return model, dataset


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — COMPOSITE CONFIDENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────

class SubgroupCentroidIndex:
    """
    Per-disease kNN cosine similarity over scaled training embeddings.

    Sparse fallback (v6): subgroups with < MIN_SUBGROUP_FOR_COSINE samples
    return x = 0.5 (neutral) instead of collapsing toward zero due to
    sparsity noise.  Small oncological subgroups (e.g. precancerous_lesions)
    were the primary victims of the old behaviour.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 disease_cols: list[str], k: int = COSINE_K):
        self.k            = k
        self.disease_cols = disease_cols
        self.subgroup_X   = {}
        self.subgroup_n   = {}

        norms  = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = (X / np.where(norms == 0, 1.0, norms)).astype(np.float32)

        for di, d in enumerate(disease_cols):
            mask = Y[:, di] > 0.5
            n    = int(mask.sum())
            self.subgroup_n[d] = n
            self.subgroup_X[d] = X_norm[mask] if n > 0 else None

    def query(self, X_query: np.ndarray) -> np.ndarray:
        N  = X_query.shape[0]
        D  = len(self.disease_cols)
        norms = np.linalg.norm(X_query, axis=1, keepdims=True)
        Q     = (X_query / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        scores = np.zeros((N, D), dtype=np.float32)

        for di, d in enumerate(self.disease_cols):
            sub = self.subgroup_X.get(d)
            n   = self.subgroup_n.get(d, 0)
            if sub is None or n == 0 or n < MIN_SUBGROUP_FOR_COSINE:
                scores[:, di] = 0.5    # neutral fallback
                continue
            sims  = Q @ sub.T
            k     = min(self.k, sims.shape[1])
            top_k = np.partition(sims, -k, axis=1)[:, -k:]
            scores[:, di] = top_k.mean(axis=1)
        return scores


class DemographicIndex:
    """
    Demographic prior alignment (m term).
    Always 0 for psychiatric diseases (shortcut contamination confirmed).
    """
    def __init__(self, train_df: pd.DataFrame, Y: np.ndarray,
                 disease_cols: list[str]):
        self.disease_cols = disease_cols
        self.centroids    = {}
        avail = [c for c in DEMO_COLS if c in train_df.columns]
        self._empty = (len(avail) == 0)
        if self._empty:
            print(f"  WARNING: DemographicIndex — none of {DEMO_COLS} found in "
                  f"train_df; m-scores will be 0 for all non-psychiatric diseases "
                  f"(d weight is inactive — check merged column names)")
            return
        self._avail = avail

        demo_df = train_df[avail].copy()
        for c in avail:
            if demo_df[c].dtype == object:
                demo_df[c] = (demo_df[c].str.lower()
                              .map({"male": 0, "m": 0, "female": 1, "f": 1})
                              .fillna(0.5))
        demo_df = demo_df.fillna(demo_df.median(numeric_only=True))
        demo_arr = demo_df.values.astype(np.float32)
        self._mean = demo_arr.mean(axis=0)
        self._std  = np.where(demo_arr.std(axis=0) == 0, 1.0,
                              demo_arr.std(axis=0))
        demo_norm = (demo_arr - self._mean) / self._std

        for di, d in enumerate(disease_cols):
            if d in PSYCHIATRIC_NO_DEMO:
                self.centroids[d] = None
                continue
            mask = Y[:, di] > 0.5
            if mask.sum() == 0:
                self.centroids[d] = None
                continue
            cen  = demo_norm[mask].mean(axis=0)
            norm = np.linalg.norm(cen)
            self.centroids[d] = cen / norm if norm > 0 else cen

    def query(self, test_df: pd.DataFrame) -> np.ndarray:
        N, D   = len(test_df), len(self.disease_cols)
        scores = np.zeros((N, D), dtype=np.float32)
        if self._empty:
            return scores
        avail = self._avail
        if not all(c in test_df.columns for c in avail):
            return scores
        demo_df = test_df[avail].copy()
        for c in avail:
            if demo_df[c].dtype == object:
                demo_df[c] = (demo_df[c].str.lower()
                              .map({"male": 0, "m": 0, "female": 1, "f": 1})
                              .fillna(0.5))
        demo_arr = (demo_df.fillna(demo_df.median(numeric_only=True))
                    .values.astype(np.float32))
        demo_arr = (demo_arr - self._mean) / self._std
        norms    = np.linalg.norm(demo_arr, axis=1, keepdims=True)
        Q        = demo_arr / np.where(norms == 0, 1.0, norms)
        for di, d in enumerate(self.disease_cols):
            cen = self.centroids.get(d)
            if cen is not None:
                scores[:, di] = Q @ cen
        return scores


def compute_confidence_scores(
    probs:        np.ndarray,
    x_scores:     np.ndarray,
    m_scores:     np.ndarray,
    conf_map:     dict,
    disease_cols: list[str],
    a: float, b: float, c: float, d: float, e: float,
) -> list[list[dict]]:
    """
    confidence = a·x + b·y − c·z + d·m + e·p

    z is scoped to the model's disease_cols list — runner-up diseases are
    only from the same model, preventing cross-model interference (v6 fix).
    """
    N, D  = probs.shape
    p_vec = np.array([max(0.0, 1.0 - conf_map.get(dis, 0.5))
                      for dis in disease_cols], dtype=np.float32)
    all_top3 = []

    for i in range(N):
        y = probs[i]; x = x_scores[i]; m = m_scores[i]
        y_rank = np.argsort(y)[::-1]
        z = np.zeros(D, dtype=np.float32)
        for di in range(D):
            runners = [idx for idx in y_rank if idx != di][:2]
            if runners:
                z[di] = float(np.max(x[runners]))

        score = a * x + b * y - c * z + d * m + e * p_vec

        top3_idx = np.argsort(score)[::-1][:3]
        top3 = []
        for rank, di in enumerate(top3_idx, start=1):
            top3.append({
                "rank":          rank,
                "disease":       disease_cols[di],
                "confidence":    float(np.clip(score[di],    0.0,  1.0)),
                "x_cosine":      float(np.clip(x[di],        0.0,  1.0)),
                "y_prob":        float(y[di]),
                "z_uncertainty": float(np.clip(z[di],        0.0,  1.0)),
                "m_demo":        float(np.clip(m[di],       -1.0,  1.0)),
                "p_confounder":  float(np.clip(p_vec[di],    0.0,  1.0)),
            })
        all_top3.append(top3)
    return all_top3


def top3_to_dataframe(top3_list, participant_ids, model_label) -> pd.DataFrame:
    rows = []
    for i, top3 in enumerate(top3_list):
        pid = participant_ids[i] if participant_ids is not None else i
        for pred in top3:
            rows.append({"participant_id": pid, "model": model_label, **pred})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def confounder_baseline_auroc(test_df, disease_cols):
    results      = {}
    task_dummies = pd.get_dummies(test_df["task_name"], prefix="task").values
    for d in disease_cols:
        y = test_df[d].fillna(0).values
        if y.sum() < 10:
            continue
        aurocs = []
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for tr, val in skf.split(task_dummies, y):
            clf = LogisticRegression(max_iter=300, random_state=42)
            try:
                clf.fit(task_dummies[tr], y[tr])
                aurocs.append(roc_auc_score(
                    y[val], clf.predict_proba(task_dummies[val])[:, 1]))
            except Exception:
                aurocs.append(0.5)
        results[d] = round(float(np.mean(aurocs)), 4)
    return results


def evaluate_model(
    model,
    test_df:      pd.DataFrame,
    feat_cols:    list[str],
    disease_cols: list[str],
    imputer,
    scaler,
    model_label:  str,
    train_df:     pd.DataFrame,
    train_Y:      np.ndarray,
    cs_weights:   dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (audit_df, top3_df).

    worst_task_auroc (v6 fix): NaN when fewer than 2 task subgroups have
    enough samples (≥5 total, ≥3 positives).  No longer a 1.0 sentinel.

    n_train_subgroup (v6 fix): added to every audit row.
    """
    a = cs_weights["a"]; b = cs_weights["b"]; c = cs_weights["c"]
    d = cs_weights["d"]; e = cs_weights["e"]

    model.eval()
    X_raw  = imputer.transform(test_df[feat_cols].values.astype(np.float32))
    X_test = scaler.transform(X_raw).astype(np.float32)
    t_ids  = test_df["task_name"].map(TASK_ENC).fillna(0).values.astype(np.int64)

    with torch.no_grad():
        logits, _ = model(torch.tensor(X_test).to(DEVICE),
                          torch.tensor(t_ids).to(DEVICE), lam=0.0)
    probs = torch.sigmoid(logits).cpu().numpy()

    conf_map = confounder_baseline_auroc(test_df, disease_cols)

    train_X_raw = imputer.transform(train_df[feat_cols].values.astype(np.float32))
    train_X     = scaler.transform(train_X_raw).astype(np.float32)

    centroid_idx = SubgroupCentroidIndex(train_X, train_Y, disease_cols, k=COSINE_K)
    demo_idx     = DemographicIndex(train_df, train_Y, disease_cols)

    x_scores = centroid_idx.query(X_test)
    m_scores = demo_idx.query(test_df)

    top3_list = compute_confidence_scores(
        probs, x_scores, m_scores, conf_map, disease_cols,
        a=a, b=b, c=c, d=d, e=e)
    top3_df = top3_to_dataframe(
        top3_list, test_df["participant_id"].tolist(), model_label)

    rows = []
    for di, dis in enumerate(disease_cols):
        y = test_df[dis].fillna(0).values
        if y.sum() < 10:
            continue
        try:
            auroc = roc_auc_score(y, probs[:, di])
        except Exception:
            auroc = 0.5

        conf   = conf_map.get(dis, 0.5)
        margin = auroc - conf

        # worst_task: NaN unless ≥2 task subgroups qualify
        worst_vals = []
        for task in test_df["task_name"].unique():
            mask = test_df["task_name"] == task
            if mask.sum() < 5 or y[mask].sum() < 3:
                continue
            try:
                worst_vals.append(roc_auc_score(y[mask], probs[mask, di]))
            except Exception:
                pass
        worst = (float(np.min(worst_vals))
                 if len(worst_vals) >= 2 else float("nan"))

        n_train_sub = int(centroid_idx.subgroup_n.get(dis, 0))

        top1_conf = top3_df[(top3_df["model"]   == model_label) &
                            (top3_df["rank"]    == 1) &
                            (top3_df["disease"] == dis)]["confidence"]
        mean_conf = (round(float(top1_conf.mean()), 4)
                     if len(top1_conf) else float("nan"))

        rows.append({
            "disease":               dis,
            "category":              DISEASE_CATEGORY.get(dis, "Other"),
            "model":                 model_label,
            "tc_dann_auroc":         round(auroc, 4),
            "task_confounder_auroc": round(conf, 4),
            "margin":                round(margin, 4),
            "worst_task_auroc":      round(worst, 4) if not np.isnan(worst)
                                     else float("nan"),
            "n_train_subgroup":      n_train_sub,
            "mean_confidence_top1":  mean_conf,
            "passes_audit":          margin > 0.05,
        })

    return pd.DataFrame(rows), top3_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — SAVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_model_artifacts(out_dir, stem, model, dataset,
                         feat_cols, disease_cols, n_sites, args,
                         centroid_idx=None, demo_idx=None):
    torch.save({
        "model_state":  model.state_dict(),
        "feat_cols":    feat_cols,
        "disease_cols": disease_cols,
        "scaler_mean":  dataset.scaler.mean_.tolist(),
        "scaler_std":   dataset.scaler.scale_.tolist(),
        "args":         vars(args),
    }, out_dir / f"{stem}_checkpoint.pt")

    bundle = {
        "model_state":        {k: v.cpu().numpy()
                               for k, v in model.state_dict().items()},
        "feat_cols":          feat_cols,
        "disease_cols":       disease_cols,
        "scaler_mean":        dataset.scaler.mean_,
        "scaler_std":         dataset.scaler.scale_,
        "imputer_statistics": dataset.imputer.statistics_,
        "n_features":         dataset.X.shape[1],
        "n_tasks":            len(TASKS),
        "n_sites":            n_sites,
        "n_diseases":         len(disease_cols),
        "args":               vars(args),
    }
    if centroid_idx is not None:
        bundle["centroid_index"] = centroid_idx
    if demo_idx is not None:
        bundle["demo_index"] = demo_idx
    joblib.dump(bundle, out_dir / f"{stem}_best.joblib", compress=3)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quad TC-DANN: Voice+Onco / Neurological / Respiratory / Psychiatric")
    parser.add_argument("--data_root",  default=os.environ.get("SVD_OUT_ROOT"),
                        required=os.environ.get("SVD_OUT_ROOT") is None,
                        help="Output root of svd_preprocess.py (or set SVD_OUT_ROOT)")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--test_size",  type=float, default=0.2)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--cache_dir",  type=str,   default=None)
    parser.add_argument("--no_cache",   action="store_true")
    # Physical confidence weights
    parser.add_argument("--cs_a",  type=float, default=CS_PHYSICAL["a"],
                        help="Physical: x weight (subgroup cosine)")
    parser.add_argument("--cs_b",  type=float, default=CS_PHYSICAL["b"],
                        help="Physical: y weight (head prob)")
    parser.add_argument("--cs_c",  type=float, default=CS_PHYSICAL["c"],
                        help="Physical: z weight (uncertainty penalty)")
    parser.add_argument("--cs_d",  type=float, default=CS_PHYSICAL["d"],
                        help="Physical: m weight (demographic prior)")
    parser.add_argument("--cs_e",  type=float, default=CS_PHYSICAL["e"],
                        help="Physical: p weight (confounder robustness)")
    # Psychiatric confidence weights (separate calibration)
    parser.add_argument("--cs_pa", type=float, default=CS_PSYCH["a"],
                        help="Psychiatric: x weight")
    parser.add_argument("--cs_pb", type=float, default=CS_PSYCH["b"],
                        help="Psychiatric: y weight")
    parser.add_argument("--cs_pc", type=float, default=CS_PSYCH["c"],
                        help="Psychiatric: z weight")
    parser.add_argument("--cs_pd", type=float, default=CS_PSYCH["d"],
                        help="Psychiatric: m weight (keep 0 — shortcut)")
    parser.add_argument("--cs_pe", type=float, default=CS_PSYCH["e"],
                        help="Psychiatric: p weight (confounder robustness)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_root = Path(args.data_root)

    # ── Cache ────────────────────────────────────────────────────────────────
    _cache_root   = (Path(args.cache_dir) if args.cache_dir
                     else data_root / "tc_dann_cache")
    _cache_key    = hashlib.md5(
        f"{data_root}|smoke={args.smoke_test}".encode()).hexdigest()[:10]
    _cache_master = _cache_root / f"master_df_{_cache_key}.joblib"
    _cache_labels = _cache_root / f"label_df_{_cache_key}.joblib"

    max_rows = 2000 if args.smoke_test else None
    if not args.no_cache and _cache_master.exists() and _cache_labels.exists():
        print(f"\n[cache] Loading from {_cache_root}")
        master_df = joblib.load(_cache_master)
        label_df  = joblib.load(_cache_labels)
        print(f"  master_df {master_df.shape}  label_df {label_df.shape}")
    else:
        master_df = load_all_features(data_root, max_mfcc_rows=max_rows)
        label_df  = load_labels_from_diagnosis_tsvs(data_root)
        if not args.no_cache:
            _cache_root.mkdir(parents=True, exist_ok=True)
            joblib.dump(master_df, _cache_master, compress=3)
            joblib.dump(label_df,  _cache_labels,  compress=3)
            print(f"\n[cache] Saved to {_cache_root}")

    # ── Merge ────────────────────────────────────────────────────────────────
    disease_cols = [d for d in ALL_DISEASES if label_df[d].sum() >= 10]
    _has_task    = "task_name" in master_df.columns
    merged = master_df.merge(label_df[["participant_id"] + disease_cols],
                             on="participant_id", how="inner")
    if "task_name" not in merged.columns:
        if _has_task:
            merged["task_name"] = master_df.loc[merged.index, "task_name"].values
        elif "task_name" in master_df.columns:
            merged["task_name"] = merged["participant_id"].map(
                master_df.set_index("participant_id")["task_name"])
    merged = merged[~merged["any_quality_flag"]].reset_index(drop=True)

    # ── Demographic column diagnostic ────────────────────────────────────────
    _present_demo  = [c for c in DEMO_COLS if c in merged.columns]
    _missing_demo  = [c for c in DEMO_COLS if c not in merged.columns]
    if _missing_demo:
        print(f"\n  WARNING: Demographic columns absent from merged data: "
              f"{_missing_demo}")
        print(f"           m-term will be 0 everywhere → "
              f"d-weight (physical={args.cs_d}) is wasted budget.")
        print(f"           Add age/sex/bmi to static_features.tsv or "
              f"set --cs_d 0 to suppress this warning.")
    else:
        print(f"\n  Demographic columns active: {_present_demo}")
    if "task_name" not in merged.columns or merged["task_name"].isna().all():
        merged["task_name"] = TASKS[0]
        print(f"  WARNING: task_name missing; defaulting to '{TASKS[0]}'")

    # ── Feature columns (PPG excluded, everything else kept) ─────────────────
    feat_cols = build_feat_cols(merged, disease_cols)
    print(f"\nFeature columns  : {len(feat_cols)}  (PPG excluded; MFCC/Mel/spec kept)")

    # ── Train / test split ───────────────────────────────────────────────────
    all_pids = merged["participant_id"].unique()
    train_pids, test_pids = train_test_split(
        all_pids, test_size=args.test_size, random_state=args.seed)
    train_df = merged[merged["participant_id"].isin(train_pids)].reset_index(drop=True)
    test_df  = merged[merged["participant_id"].isin(test_pids)].reset_index(drop=True)
    n_sites  = max(train_df.get("site_id", pd.Series(0)).nunique(), 2)

    # ── Resolve disease lists (intersect with available labels) ──────────────
    model_disease_cols: dict[str, list[str]] = {}
    for stem, candidates in MODEL_DISEASE_MAP.items():
        cols = [d for d in candidates if d in disease_cols]
        if cols:
            model_disease_cols[stem] = cols
        else:
            print(f"  WARNING: {stem} — no diseases with n≥10, skipping")

    print(f"\n{'─'*70}")
    print("Dataset summary")
    print(f"{'─'*70}")
    print(f"  Total samples      : {len(merged):,}  "
          f"(train {len(train_df):,} / test {len(test_df):,})")
    print(f"  Total participants : {len(all_pids):,}  "
          f"(train {len(train_pids):,} / test {len(test_pids):,})")
    for stem, cols in model_disease_cols.items():
        print(f"  {MODEL_DISPLAY[stem]:<16}: {len(cols)} — {', '.join(cols)}")
    print(f"  Sites              : {n_sites}   Device: {DEVICE}")

    phys_w = dict(a=args.cs_a,  b=args.cs_b,  c=args.cs_c,
                  d=args.cs_d,  e=args.cs_e)
    psyc_w = dict(a=args.cs_pa, b=args.cs_pb, c=args.cs_pc,
                  d=args.cs_pd, e=args.cs_pe)

    def _weights(stem):
        return psyc_w if stem == "psychiatric" else phys_w

    # ══════════════════════════════════════════════════════════════════════════
    # TRAIN
    # ══════════════════════════════════════════════════════════════════════════
    trained: dict = {}
    for stem, dis_cols in model_disease_cols.items():
        label = MODEL_DISPLAY[stem]
        w     = _weights(stem)
        print(f"\n{'═'*70}")
        print(f"Training  MODEL: {label}")
        print(f"  diseases : {', '.join(dis_cols)}")
        print(f"  conf weights: "
              f"a={w['a']} b={w['b']} c={w['c']} d={w['d']} e={w['e']}")
        print(f"{'═'*70}")
        model, dataset = train_model(label, train_df, feat_cols,
                                     dis_cols, n_sites, args)
        trained[stem] = (model, dataset)

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("Confounder Audit + Confidence Scoring  [held-out test set]")
    print(f"{'─'*70}")

    all_audit: list[pd.DataFrame] = []
    all_top3:  list[pd.DataFrame] = []

    for stem, dis_cols in model_disease_cols.items():
        model, dataset = trained[stem]
        train_Y = train_df[dis_cols].fillna(0).values.astype(np.float32)
        audit_df, top3_df = evaluate_model(
            model=model, test_df=test_df,
            feat_cols=feat_cols, disease_cols=dis_cols,
            imputer=dataset.imputer, scaler=dataset.scaler,
            model_label=MODEL_DISPLAY[stem],
            train_df=train_df, train_Y=train_Y,
            cs_weights=_weights(stem),
        )
        all_audit.append(audit_df)
        all_top3.append(top3_df)

    results = (pd.concat(all_audit, ignore_index=True)
               .sort_values("tc_dann_auroc", ascending=False))

    # ── Print audit table ─────────────────────────────────────────────────────
    hdr = (f"{'Disease':<45} {'Cat':<15} {'Model':<14} {'AUROC':<8} "
           f"{'Conf':<8} {'Margin':<9} {'WrstTask':<10} {'N_sub':<7} "
           f"{'MeanConf':<10} {'Audit'}")
    print(f"\n{hdr}")
    print("─" * 140)

    for cat in ["Voice", "Oncological", "Neurological",
                "Respiratory", "Psychiatric", "Control"]:
        sub = results[results["category"] == cat]
        if sub.empty:
            continue
        print(f"\n  [{cat}]")
        for _, r in sub.iterrows():
            flag  = "✓ PASS" if r["passes_audit"] else "✗ FAIL"
            worst = (f"{r['worst_task_auroc']:.4f}"
                     if not pd.isna(r["worst_task_auroc"]) else "  —   ")
            mconf = (f"{r['mean_confidence_top1']:.4f}"
                     if not pd.isna(r["mean_confidence_top1"]) else "  —   ")
            print(f"  {r['disease']:<43} {cat:<15} {r['model']:<14} "
                  f"{r['tc_dann_auroc']:<8} {r['task_confounder_auroc']:<8} "
                  f"{r['margin']:+.4f}   {worst:<10} "
                  f"{int(r['n_train_subgroup']):<7} {mconf:<10} {flag}")

    n_pass = results["passes_audit"].sum()
    print(f"\n  Diseases passing audit: {n_pass}/{len(results)}")
    for stem in model_disease_cols:
        label = MODEL_DISPLAY[stem]
        sub   = results[results["model"] == label]
        print(f"    {label:<16}: {sub['passes_audit'].sum()}/{len(sub)}")

    # ── Unified cross-model top-3 ─────────────────────────────────────────────
    # Pool all four model predictions per participant, re-rank by confidence.
    all_top3_combined = pd.concat(all_top3, ignore_index=True)
    unified_rows = []
    for pid in all_top3_combined["participant_id"].unique():
        sub = (all_top3_combined[all_top3_combined["participant_id"] == pid]
               .sort_values("confidence", ascending=False)
               # Drop duplicate diseases: keep the highest-confidence row per disease.
               # Without this, a participant with N recordings gets the same disease
               # repeated N times in the top-3 (one per recording).
               .drop_duplicates(subset=["disease"])
               .head(3)
               .reset_index(drop=True))
        for new_rank, (_, r) in enumerate(sub.iterrows(), start=1):
            unified_rows.append({**r.to_dict(), "rank": new_rank})
    unified_top3 = pd.DataFrame(unified_rows)

    print(f"\n{'─'*70}")
    print("Unified Cross-Model Top-3  [first 5 test participants]")
    print(f"{'─'*70}")
    for pid in unified_top3["participant_id"].unique()[:5]:
        sub = unified_top3[unified_top3["participant_id"] == pid]
        print(f"\n  Participant {pid}")
        for _, r in sub.iterrows():
            print(f"    #{int(r['rank'])}  [{r['model']:<14}]  "
                  f"{r['disease']:<38}  conf={r['confidence']:.3f}  "
                  f"(x={r['x_cosine']:.3f} y={r['y_prob']:.3f} "
                  f"z={r['z_uncertainty']:.3f} m={r['m_demo']:.3f} "
                  f"p={r['p_confounder']:.3f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = data_root / "tc_dann_results"
    out.mkdir(exist_ok=True)

    results.to_csv(out / "audit_table.csv",        index=False)
    unified_top3.to_csv(out / "top3_unified.csv",  index=False)
    all_top3_combined.to_csv(out / "top3_all_models.csv", index=False)

    for stem, dis_cols in model_disease_cols.items():
        model, dataset = trained[stem]
        label          = MODEL_DISPLAY[stem]
        train_Y        = train_df[dis_cols].fillna(0).values.astype(np.float32)
        tX_raw         = dataset.imputer.transform(
            train_df[feat_cols].values.astype(np.float32))
        tX             = dataset.scaler.transform(tX_raw).astype(np.float32)
        c_idx          = SubgroupCentroidIndex(tX, train_Y, dis_cols, k=COSINE_K)
        d_idx          = DemographicIndex(train_df, train_Y, dis_cols)

        results[results["model"] == label].to_csv(
            out / f"audit_{stem}.csv", index=False)
        all_top3_combined[all_top3_combined["model"] == label].to_csv(
            out / f"top3_{stem}.csv", index=False)

        save_model_artifacts(
            out_dir=out, stem=f"{stem}_model",
            model=model, dataset=dataset,
            feat_cols=feat_cols, disease_cols=dis_cols,
            n_sites=n_sites, args=args,
            centroid_idx=c_idx, demo_idx=d_idx,
        )

    print(f"\n{'─'*70}")
    print(f"✓ audit_table.csv       → {out}/audit_table.csv")
    print(f"✓ top3_unified.csv      → {out}/top3_unified.csv")
    print(f"✓ top3_all_models.csv   → {out}/top3_all_models.csv")
    for stem in model_disease_cols:
        print(f"✓ {stem:<22} "
              f"audit_{stem}.csv  |  top3_{stem}.csv  |  "
              f"{stem}_model_checkpoint.pt  |  {stem}_model_best.joblib")
    print("\nDone.")


if __name__ == "__main__":
    main()
