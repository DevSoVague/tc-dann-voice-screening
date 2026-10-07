"""
Dataset layer for TC-DANN.

Public interface (unchanged from original skeleton):
    BridgeVoiceDataset
    build_subgroup_sampler
    cross_site_mixup
    AGE_BUCKETS, AGE_BUCKET_NAMES, age_to_bucket

What changed: the dataset now reads directly from the Bridge2AI-Voice v3.0.0
public release parquets via a `FeatureStore`. No manual loaders are required.
The store auto-detects key columns (participant_id / session_id / task / recording_id)
with substring matching so both unprefixed and prefixed schemas work.

Design choices from the audit report, kept verbatim:
  - Recording-level items, not participant-aggregated (kills task-histogram shortcut).
  - Subgroup-balanced sampler against (site x age_bucket x sex).
  - Cross-site mixup pairs within same disease pattern but different sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler


# ------------------------------------------------------------------
# Subgroup definitions (audit 2.4)
# ------------------------------------------------------------------

AGE_BUCKETS: List[Tuple[float, float]] = [
    (-np.inf, 40),
    (40, 55),
    (55, 70),
    (70, np.inf),
]
AGE_BUCKET_NAMES = ["<=40", "41-55", "56-70", "71+"]


def age_to_bucket(age: float) -> int:
    for i, (lo, hi) in enumerate(AGE_BUCKETS):
        if lo < age <= hi:
            return i
    return len(AGE_BUCKETS) - 1


SITE_MAP = {"USA": 0, "Canada": 1, "Other": 2}
SEX_MAP = {"F": 0, "M": 1}


def site_to_code(site: str) -> int:
    return SITE_MAP.get(str(site), SITE_MAP["Other"])


def sex_to_code(sex: str) -> int:
    return SEX_MAP.get(str(sex), SEX_MAP["F"])


# ------------------------------------------------------------------
# FeatureStore: auto-detecting parquet reader
# ------------------------------------------------------------------

_KEY_PATTERNS = {
    "pid":  ["participant_id", "record_id", "subject_id"],
    "sid":  ["session_id"],
    "rid":  ["recording_id"],
    "task": ["task_name", "acoustic_task_name", "task"],
}


class FeatureStore:
    """
    Reads a b2aiprep-style parquet and returns per-recording feature tensors.

    Auto-detected columns:
        pid_col  (participant_id / record_id / subject_id, possibly prefixed)
        sid_col  (session_id)
        rid_col  (recording_id)
        task_col (task_name / acoustic_task_name / task)
    Feature columns = everything else. If there is a single feature column,
    its value is returned as-is (expected to be an ND array stored as a
    nested list / pyarrow array). If there are multiple feature columns,
    they are stacked along a new leading axis.
    """

    def __init__(self, parquet_path: Path, expected_rows: Optional[int] = None):
        self.path = Path(parquet_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.df: pd.DataFrame = pd.read_parquet(self.path)
        self.expected_rows = expected_rows

        # Detect keys
        self.cols = list(self.df.columns)
        self.pid_col = self._find(_KEY_PATTERNS["pid"])
        self.sid_col = self._find(_KEY_PATTERNS["sid"], required=False)
        self.rid_col = self._find(_KEY_PATTERNS["rid"], required=False)
        self.task_col = self._find(_KEY_PATTERNS["task"], required=False)

        key_cols = [c for c in (self.pid_col, self.sid_col, self.rid_col, self.task_col) if c]
        self.feat_cols = [c for c in self.cols if c not in key_cols]
        if not self.feat_cols:
            raise ValueError(
                f"No feature columns detected in {self.path} (all columns look like keys)"
            )

        # Normalise key column dtypes to str for reliable lookup
        for c in key_cols:
            self.df[c] = self.df[c].astype(str)

        # Build a composite string key for O(1) dict-style lookup.
        # We avoid pandas MultiIndex because `set_index` + `.loc[tuple]` hits
        # a known edge case when feature columns hold nested arrays / lists.
        self._key_cols = [c for c in (self.pid_col, self.sid_col, self.task_col, self.rid_col) if c]
        key_series = self.df[self._key_cols[0]].astype(str)
        for c in self._key_cols[1:]:
            key_series = key_series + "||" + self.df[c].astype(str)
        self._key_to_row = {}
        for i, k in enumerate(key_series.values):
            # first occurrence wins if duplicates
            if k not in self._key_to_row:
                self._key_to_row[k] = i

    def _find(self, patterns: Sequence[str], required: bool = True) -> Optional[str]:
        lower = {c: c.lower() for c in self.cols}
        for p in patterns:
            pl = p.lower()
            for c in self.cols:
                if lower[c] == pl:
                    return c
        for p in patterns:
            pl = p.lower()
            for c in self.cols:
                if pl in lower[c]:
                    return c
        if required:
            raise KeyError(f"Could not find any of {patterns} in {self.path}")
        return None

    def get(self, participant_id: str, session_id: Optional[str] = None,
            task: Optional[str] = None, recording_id: Optional[str] = None
            ) -> Optional[torch.Tensor]:
        """Return the feature tensor, or None if the recording is not present."""
        available = {
            self.pid_col: participant_id,
            self.sid_col: session_id,
            self.task_col: task,
            self.rid_col: recording_id,
        }
        parts = []
        for col in self._key_cols:
            v = available.get(col)
            if v is None:
                # Lookup with a partial key: fall back to any row matching
                # the first N key-parts by scanning (rare path).
                return self._fallback_partial_lookup(available)
            parts.append(str(v))
        key = "||".join(parts)
        idx = self._key_to_row.get(key)
        if idx is None:
            # Try progressively relaxed keys (drop recording_id, then task, etc.)
            return self._fallback_partial_lookup(available)
        row = self.df.iloc[idx]
        return self._row_to_tensor(row)

    def _fallback_partial_lookup(self, available: dict) -> Optional[torch.Tensor]:
        """Boolean-mask scan for the first matching row when the exact composite key is absent."""
        m = pd.Series(True, index=self.df.index)
        for col, val in available.items():
            if col is None or val is None:
                continue
            m &= (self.df[col].astype(str) == str(val))
        hits = self.df[m]
        if len(hits) == 0:
            return None
        return self._row_to_tensor(hits.iloc[0])

    def _row_to_tensor(self, row: pd.Series) -> Optional[torch.Tensor]:
        if len(self.feat_cols) == 1:
            val = row[self.feat_cols[0]]
            arr = _coerce_array(val)
        else:
            arrs = []
            for c in self.feat_cols:
                a = _coerce_array(row[c])
                if a is None:
                    return None
                if a.ndim == 0:
                    a = a.reshape(1)
                arrs.append(a)
            target_len = max(a.shape[-1] for a in arrs)
            arrs = [_pad_last_axis(a, target_len) for a in arrs]
            arr = np.stack(arrs, axis=0)
        if arr is None:
            return None
        if arr.dtype == object:
            # Last-ditch: force cast via tolist()
            try:
                arr = np.array(arr.tolist(), dtype=np.float32)
            except Exception:
                return None
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _coerce_array(value) -> Optional[np.ndarray]:
    """Turn a cell value (list, ndarray, nested list, pyarrow) into an ndarray.

    Handles the common pyarrow/pandas pattern where a 2D feature is stored as
    a 1D object-array of 1D arrays. In that case we stack the inner arrays.
    """
    if value is None:
        return None
    # Nested object-array case (pyarrow "large_list<list<float>>")
    if isinstance(value, np.ndarray) and value.dtype == object:
        try:
            return np.stack([np.asarray(v, dtype=np.float32) for v in value])
        except Exception:
            return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        # Might be a list-of-lists (2D) or list of floats (1D)
        try:
            return np.asarray(value, dtype=np.float32)
        except (ValueError, TypeError):
            try:
                return np.stack([np.asarray(v, dtype=np.float32) for v in value])
            except Exception:
                return None
    try:
        return np.asarray(value.tolist())      # pyarrow scalar
    except Exception:
        try:
            return np.asarray(value)
        except Exception:
            return None


def _pad_last_axis(a: np.ndarray, target_len: int) -> np.ndarray:
    if a.ndim == 0:
        a = a.reshape(1)
    pad = target_len - a.shape[-1]
    if pad <= 0:
        return a
    return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, pad)])


# ------------------------------------------------------------------
# Shape helpers
# ------------------------------------------------------------------

def _as_spec(arr: Optional[torch.Tensor], n_freq: int) -> Optional[torch.Tensor]:
    """Coerce a spectrogram tensor to shape [1, n_freq, T]."""
    if arr is None:
        return None
    if arr.ndim == 1:
        # Flat: unknown shape, skip
        return None
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    if arr.ndim != 2:
        return None
    # Orient so the first axis is frequency. Prefer exact match, else heuristic
    if arr.shape[0] == n_freq and arr.shape[1] != n_freq:
        pass
    elif arr.shape[1] == n_freq and arr.shape[0] != n_freq:
        arr = arr.T
    elif arr.shape[0] > arr.shape[1]:
        # assume freq is the larger axis
        pass
    else:
        arr = arr.T
    return arr.unsqueeze(0).contiguous()


def _as_ema(arr: Optional[torch.Tensor], n_channels: int = 12) -> Optional[torch.Tensor]:
    """Coerce EMA tensor to shape [n_channels, T]."""
    if arr is None:
        return None
    if arr.ndim == 1:
        return None
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    if arr.ndim != 2:
        return None
    if arr.shape[0] == n_channels:
        pass
    elif arr.shape[1] == n_channels:
        arr = arr.T
    elif arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr.contiguous()


def _fix_time(x: torch.Tensor, T: int) -> torch.Tensor:
    """Pad or truncate the last axis to T frames."""
    t = x.shape[-1]
    if t == T:
        return x
    if t > T:
        return x[..., :T]
    return F.pad(x, (0, T - t))


# ------------------------------------------------------------------
# BridgeVoiceDataset
# ------------------------------------------------------------------

class BridgeVoiceDataset(Dataset):
    """
    One recording per item. Returns a dict shaped like the batch expected
    by TCDANN.forward plus supervision labels.

    Parameters:
        manifest:       per-recording DataFrame (from data_prep.build_master_manifest)
        disease_cols:   list of disease code names that are columns in manifest
        task_vocab:     mapping task-name -> int id
        static:         DataFrame indexed by participant_id (static features)
        features_dir:   path to the features/ folder (parquet files live here)
        n_spec_freq:    number of spectrogram frequency bins (default 201)
        n_mel:          number of mel bins (default 80; auto-corrects at runtime)
        n_ema:          number of EMA channels (default 12)
        max_frames:     time dimension to pad/truncate to
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        disease_cols: List[str],
        task_vocab: Dict[str, int],
        static: pd.DataFrame,
        features_dir: Path,
        n_spec_freq: int = 201,
        n_mel: int = 80,
        n_ema: int = 12,
        max_frames: int = 400,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.disease_cols = list(disease_cols)
        self.task_vocab = dict(task_vocab)
        self.static = static
        self.n_spec_freq = n_spec_freq
        self.n_mel = n_mel
        self.n_ema = n_ema
        self.max_frames = max_frames

        features_dir = Path(features_dir)
        self.spec_store = FeatureStore(features_dir / "torchaudio_spectrogram.parquet")
        self.mel_store = FeatureStore(features_dir / "torchaudio_mel_spectrogram.parquet")
        self.ema_store = FeatureStore(features_dir / "sparc_ema.parquet")

        # Convenience counts (consumed by run.py)
        self.n_diseases = len(self.disease_cols)
        self.n_tasks = len(self.task_vocab)
        self.n_sites = len(SITE_MAP)
        self.n_age_buckets = len(AGE_BUCKETS)
        self.n_sex = len(SEX_MAP)
        self.n_static = int(self.static.shape[1])

        # Pre-compute subgroup codes
        m = self.manifest
        m["age_bucket"] = m["age"].map(age_to_bucket).astype(np.int64)
        m["site_code"] = m["site"].map(site_to_code).astype(np.int64)
        m["sex_code"] = m["sex"].map(sex_to_code).astype(np.int64)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        pid = str(row.participant_id)
        sid = str(row.session_id) if row.session_id else None
        rid = str(row.recording_id) if row.recording_id else None
        tname = str(row.task)

        # Features (with graceful fallback if a recording is missing from a store)
        spec = _as_spec(self.spec_store.get(pid, sid, tname, rid), self.n_spec_freq)
        mel = _as_spec(self.mel_store.get(pid, sid, tname, rid), self.n_mel)
        ema = _as_ema(self.ema_store.get(pid, sid, tname, rid), self.n_ema)

        spec_present = spec is not None
        mel_present = mel is not None
        ema_present = ema is not None
        static_present = pid in self.static.index

        if spec is None:
            spec = torch.zeros(1, self.n_spec_freq, self.max_frames)
        else:
            spec = _fix_time(spec, self.max_frames)

        if mel is None:
            mel = torch.zeros(1, self.n_mel, self.max_frames)
        else:
            mel = _fix_time(mel, self.max_frames)

        if ema is None:
            ema = torch.zeros(self.n_ema, self.max_frames)
        else:
            ema = _fix_time(ema, self.max_frames)
        ema_mask = torch.ones(ema.shape[-1], dtype=torch.float32)

        if static_present:
            s_vec = torch.tensor(self.static.loc[pid].values, dtype=torch.float32)
        else:
            s_vec = torch.zeros(self.n_static, dtype=torch.float32)

        disease = torch.tensor(
            row[self.disease_cols].values.astype(np.float32),
            dtype=torch.float32,
        )

        task_id = self.task_vocab.get(tname, 0)

        return {
            "spec": spec,
            "mel": mel,
            "ema": ema,
            "ema_mask": ema_mask,
            "static": s_vec,
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "present": torch.tensor(
                [spec_present, mel_present, ema_present, static_present],
                dtype=torch.bool,
            ),
            "disease": disease,
            "site": torch.tensor(int(row.site_code), dtype=torch.long),
            "age": torch.tensor(int(row.age_bucket), dtype=torch.long),
            "sex": torch.tensor(int(row.sex_code), dtype=torch.long),
            "participant_id": pid,
        }


# ------------------------------------------------------------------
# Subgroup-balanced sampler
# ------------------------------------------------------------------

def build_subgroup_sampler(dataset: BridgeVoiceDataset) -> WeightedRandomSampler:
    """Weight each sample by 1 / freq(site, age_bucket, sex)."""
    m = dataset.manifest
    key = list(zip(m.site_code, m.age_bucket, m.sex_code))
    counts = pd.Series(key).value_counts().to_dict()
    weights = np.array([1.0 / counts[k] for k in key], dtype=np.float64)
    weights = weights / weights.sum() * len(weights)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )


# ------------------------------------------------------------------
# Cross-site mixup
# ------------------------------------------------------------------

def cross_site_mixup(batch: Dict[str, torch.Tensor],
                     alpha: float = 0.4,
                     p: float = 0.5) -> Dict[str, torch.Tensor]:
    if torch.rand(1).item() > p:
        return batch

    B = batch["spec"].size(0)
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)

    site = batch["site"]
    disease = batch["disease"]
    pair = torch.arange(B)

    for i in range(B):
        candidates = [
            j for j in range(B)
            if j != i and site[j] != site[i]
            and torch.equal(disease[j], disease[i])
        ]
        if candidates:
            pair[i] = candidates[int(torch.randint(0, len(candidates), (1,)).item())]

    for key in ("spec", "mel", "ema", "static"):
        if key in batch:
            batch[key] = lam * batch[key] + (1.0 - lam) * batch[key][pair]

    return batch
