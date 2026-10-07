"""
data_prep.py  --  construct the training manifest directly from the
Bridge2AI-Voice v3.0.0 public release layout.

Expected layout (relative to --root_dir):

    <root>/
      features/
        torchaudio_spectrogram.parquet
        torchaudio_mel_spectrogram.parquet
        sparc_ema.parquet
        static_features.tsv
        ...
      phenotype/
        demographics/demographics.tsv
        diagnosis/<disease>.tsv     (one TSV per disease, also control.tsv)
        enrollment/participant.tsv
        task/recording.tsv, session.tsv, acoustic_task.tsv

We build:
    - a participant-level diagnosis matrix (one binary column per retained disease)
    - a participant-level demographics frame (age, sex, site)
    - a recording-level manifest (one row per recording, joined with pid + task)
    - participant-disjoint train/val/test splits

Everything is schema-defensive. `_find_col` resolves unprefixed and prefixed
column names (e.g. `participant_id` vs `mfcc_participant_id`) by substring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Disease name <-> diagnosis TSV filename
# ------------------------------------------------------------------

# Short model-side codes on the left (audit report + run.py scope).
# Right side = basename of the TSV inside phenotype/diagnosis/.
DISEASE_FILE_MAP: Dict[str, str] = {
    # structural / neuromotor / respiratory (deployable)
    "parkinsons":              "parkinsons_disease",
    "airway_stenosis":         "airway_stenosis",
    "laryngeal_dystonia":      "laryngeal_dystonia",
    "vf_paralysis":            "unilateral_vocal_fold_paralysis",
    "mtd":                     "muscle_tension_dysphonia",
    "chronic_cough":           "unexplained_chronic_cough",
    "benign_lesions":          "benign_lesions",
    "laryngitis":              "laryngitis",
    "glottic_insufficiency":   "glottic_insufficiency",
    # exploratory
    "cognitive_impairment":    "cognitive_impairment",
    "depression":              "depression",
    "ptsd":                    "ptsd_adult",
    "adhd":                    "adhd_adult",
    "psychiatric_history":     "psychiatric_history",
    "bipolar":                 "bipolar_disorder",
    "anxiety":                 "anxiety",
    # present in the release but not modelled (too few or beyond scope)
    "als":                     "amyotrophic_lateral_sclerosis",
    "copd_asthma":             "copd_and_asthma",
    "laryngeal_cancer":        "laryngeal_cancer",
    "precancerous":            "precancerous_lesions",
}


# ------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: Sequence[str],
              required: bool = True) -> Optional[str]:
    """Return the first column whose lowercase name contains any candidate substring."""
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}
    for cand in candidates:
        cl = cand.lower()
        for c in cols:
            if cl == lower[c]:                 # exact match first
                return c
        for c in cols:
            if cl in lower[c]:                 # substring (handles prefixes)
                return c
    if required:
        raise KeyError(
            f"None of {candidates} found in columns: {cols}"
        )
    return None


def _normalize_site(value) -> str:
    """Map free-text country/site strings to USA / Canada / Other."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Other"
    s = str(value).strip().upper()
    if s in {"US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"}:
        return "USA"
    if s in {"CA", "CAN", "CANADA"}:
        return "Canada"
    return "Other"


def _normalize_sex(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().upper()
    if s.startswith("F"):
        return "F"
    if s.startswith("M"):
        return "M"
    return None


# ------------------------------------------------------------------
# Phenotype loaders
# ------------------------------------------------------------------

def load_diagnosis_matrix(phenotype_dir: Path,
                          diseases: Iterable[str]) -> pd.DataFrame:
    """
    For each requested disease code, read its TSV and treat all listed
    participants as positives. Returns a DataFrame indexed by participant_id
    with one 0/1 column per disease. Participants who appear in control.tsv
    but in no disease TSV still appear in the index as all-zero rows.
    """
    diag_dir = Path(phenotype_dir) / "diagnosis"
    diseases = list(diseases)
    positives: Dict[str, set] = {}
    all_pids: set = set()

    for code in diseases:
        fname = DISEASE_FILE_MAP.get(code)
        if fname is None:
            print(f"[data_prep] warning: no file mapping for disease '{code}', skipping")
            continue
        path = diag_dir / f"{fname}.tsv"
        if not path.exists():
            print(f"[data_prep] warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path, sep="\t", dtype=str)
        pid_col = _find_col(df, ["participant_id", "record_id", "subject_id"])
        pids = set(df[pid_col].dropna().astype(str))
        positives[code] = pids
        all_pids |= pids

    # also include controls so negatives are representable
    ctrl = diag_dir / "control.tsv"
    if ctrl.exists():
        df = pd.read_csv(ctrl, sep="\t", dtype=str)
        pid_col = _find_col(df, ["participant_id", "record_id", "subject_id"])
        all_pids |= set(df[pid_col].dropna().astype(str))

    all_pids = sorted(all_pids)
    matrix = pd.DataFrame(
        0, index=pd.Index(all_pids, name="participant_id"),
        columns=list(positives.keys()), dtype=np.int8,
    )
    for code, pids in positives.items():
        matrix.loc[matrix.index.isin(pids), code] = 1
    return matrix


def load_demographics(phenotype_dir: Path) -> pd.DataFrame:
    """Return DataFrame indexed by participant_id with age, sex, site."""
    path = Path(phenotype_dir) / "demographics" / "demographics.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str)

    pid_col = _find_col(df, ["participant_id", "record_id", "subject_id"])
    age_col = _find_col(df, ["age_enrollment", "age_at_enrollment", "age"],
                        required=False)
    sex_col = _find_col(df, ["sex_at_birth", "sex", "gender"], required=False)
    site_col = _find_col(
        df,
        ["country_of_birth", "country", "enrollment_institution_country",
         "site", "institution", "enrolled_by"],
        required=False,
    )
    # Country may live in the enrollment TSV instead. Try that as a fallback.
    if site_col is None:
        enr = Path(phenotype_dir) / "enrollment" / "participant.tsv"
        if enr.exists():
            edf = pd.read_csv(enr, sep="\t", dtype=str)
            e_pid = _find_col(edf, ["participant_id", "record_id", "subject_id"],
                              required=False)
            e_site = _find_col(
                edf,
                ["country_of_birth", "country", "enrollment_institution_country",
                 "site", "institution", "enrolled_by"],
                required=False,
            )
            if e_pid and e_site:
                df = df.merge(edf[[e_pid, e_site]], left_on=pid_col,
                              right_on=e_pid, how="left", suffixes=("", "_enr"))
                site_col = e_site

    out = pd.DataFrame({
        "participant_id": df[pid_col].astype(str),
        "age": pd.to_numeric(df[age_col], errors="coerce") if age_col else np.nan,
        "sex": df[sex_col].map(_normalize_sex) if sex_col else None,
        "site": (df[site_col].map(_normalize_site) if site_col
                 else pd.Series(["Other"] * len(df))),
    })
    # sensible fallbacks
    median_age = out["age"].median()
    if not np.isfinite(median_age):
        median_age = 50.0
    out["age"] = out["age"].fillna(median_age)
    out["sex"] = out["sex"].fillna("F")   # binary adversary needs a label; picked at random-ish
    out["site"] = out["site"].fillna("Other")
    return out.drop_duplicates("participant_id").set_index("participant_id")


def load_recording_manifest(phenotype_dir: Path) -> pd.DataFrame:
    """
    Return one row per recording with columns:
        participant_id, session_id, recording_id, task

    Joins task/recording.tsv with task/session.tsv and task/acoustic_task.tsv
    as needed to resolve participant_id and human-readable task name.
    """
    task_dir = Path(phenotype_dir) / "task"
    rec = pd.read_csv(task_dir / "recording.tsv", sep="\t", dtype=str)

    rec_pid = _find_col(rec, ["participant_id", "record_id", "subject_id"],
                        required=False)
    rec_sid = _find_col(rec, ["session_id"], required=False)
    rec_rid = _find_col(rec, ["recording_id"], required=False)
    rec_task = _find_col(rec,
                         ["acoustic_task_name", "task_name", "task"],
                         required=False)
    rec_task_id = _find_col(rec, ["acoustic_task_id"], required=False)

    # Resolve task name via acoustic_task.tsv if only the id is present
    if rec_task is None and rec_task_id is not None:
        atask_path = task_dir / "acoustic_task.tsv"
        if atask_path.exists():
            atask = pd.read_csv(atask_path, sep="\t", dtype=str)
            at_id = _find_col(atask, ["acoustic_task_id"], required=False)
            at_name = _find_col(
                atask, ["acoustic_task_name", "task_name", "task"],
                required=False,
            )
            if at_id and at_name:
                mapping = dict(zip(atask[at_id], atask[at_name]))
                rec["task"] = rec[rec_task_id].map(mapping)
                rec_task = "task"

    # Resolve participant_id via session.tsv if missing from recording.tsv
    if rec_pid is None and rec_sid is not None:
        sess_path = task_dir / "session.tsv"
        if sess_path.exists():
            sess = pd.read_csv(sess_path, sep="\t", dtype=str)
            se_sid = _find_col(sess, ["session_id"], required=False)
            se_pid = _find_col(
                sess, ["participant_id", "record_id", "subject_id"],
                required=False,
            )
            if se_sid and se_pid:
                mapping = dict(zip(sess[se_sid], sess[se_pid]))
                rec["participant_id"] = rec[rec_sid].map(mapping)
                rec_pid = "participant_id"

    if rec_pid is None or rec_task is None:
        raise RuntimeError(
            "Could not resolve participant_id / task columns from recording.tsv. "
            "Run `python run.py diagnose --root_dir ..` for a full schema dump."
        )

    out = pd.DataFrame({
        "participant_id": rec[rec_pid].astype(str),
        "session_id":     rec[rec_sid].astype(str) if rec_sid else "",
        "recording_id":   rec[rec_rid].astype(str) if rec_rid else "",
        "task":           rec[rec_task].astype(str).fillna("unknown"),
    })
    return out.dropna(subset=["participant_id", "task"]).reset_index(drop=True)


def load_static_features(features_dir: Path) -> Tuple[pd.DataFrame, int]:
    """Load static_features.tsv indexed by participant_id. Returns (df, n_dim)."""
    path = Path(features_dir) / "static_features.tsv"
    df = pd.read_csv(path, sep="\t")
    pid_col = _find_col(df, ["participant_id", "record_id", "subject_id"])
    df[pid_col] = df[pid_col].astype(str)
    # Keep only numeric columns for the encoder
    feat = df.select_dtypes(include=[np.number]).copy()
    feat.index = df[pid_col].values
    feat.index.name = "participant_id"
    # Fill any NaNs with column median (robust)
    feat = feat.fillna(feat.median(numeric_only=True)).fillna(0.0)
    feat = feat.groupby(feat.index).first()           # dedup
    return feat, feat.shape[1]


# ------------------------------------------------------------------
# Build full manifest (one row per recording, joined with everything)
# ------------------------------------------------------------------

def build_master_manifest(root_dir: Path,
                          diseases: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Returns:
        manifest:  one row per recording with participant_id, session_id,
                   recording_id, task, age, sex, site, and one column per disease
        static:    pd.DataFrame indexed by participant_id (static features)
        n_static:  number of static feature dimensions
    """
    root_dir = Path(root_dir)
    pheno = root_dir / "phenotype"
    feat = root_dir / "features"

    diag = load_diagnosis_matrix(pheno, diseases)
    demo = load_demographics(pheno)
    rec = load_recording_manifest(pheno)
    static, n_static = load_static_features(feat)

    # Keep only recordings whose participant we have full metadata for
    keep = set(diag.index) & set(demo.index) & set(static.index)
    rec = rec[rec.participant_id.isin(keep)].copy()

    # Join diagnosis + demographics
    rec = rec.merge(diag, left_on="participant_id", right_index=True, how="left")
    rec = rec.merge(demo, left_on="participant_id", right_index=True, how="left")

    # Fill any residual NaNs in disease columns with 0
    disease_cols = list(diag.columns)
    rec[disease_cols] = rec[disease_cols].fillna(0).astype(np.int8)

    return rec.reset_index(drop=True), static, n_static


# ------------------------------------------------------------------
# Participant-disjoint splitting
# ------------------------------------------------------------------

def stratified_participant_split(manifest: pd.DataFrame,
                                 seed: int = 42,
                                 fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15)
                                 ) -> Dict[str, pd.DataFrame]:
    """
    Split participants into train/val/test with the target fractions.
    Splits are participant-disjoint. With 833 participants this simple
    random split is adequate; stratified multi-label splits add complexity
    without meaningful benefit at this scale.
    """
    rng = np.random.default_rng(seed)
    pids = manifest["participant_id"].drop_duplicates().sample(
        frac=1.0, random_state=seed
    ).tolist()
    n = len(pids)
    n_train = int(round(n * fracs[0]))
    n_val = int(round(n * fracs[1]))
    train = set(pids[:n_train])
    val = set(pids[n_train:n_train + n_val])
    test = set(pids[n_train + n_val:])

    return {
        "train": manifest[manifest.participant_id.isin(train)].reset_index(drop=True),
        "val":   manifest[manifest.participant_id.isin(val)].reset_index(drop=True),
        "test":  manifest[manifest.participant_id.isin(test)].reset_index(drop=True),
    }


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------

def diagnose_schemas(root_dir: Path) -> None:
    """Print a schema report so schema surprises are visible before training."""
    root_dir = Path(root_dir)
    print(f"\n=== Root directory: {root_dir.resolve()} ===")
    for sub in ("features", "phenotype"):
        p = root_dir / sub
        print(f"  {sub}/: {'found' if p.exists() else 'MISSING'}")

    # Parquets
    print("\n--- features/*.parquet ---")
    feat_dir = root_dir / "features"
    for pq in sorted(feat_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(f"  {pq.name}: FAILED to read -- {e}")
            continue
        print(f"  {pq.name}: {len(df)} rows, cols={list(df.columns)[:8]}"
              + (" ..." if len(df.columns) > 8 else ""))
        # sample first non-key cell
        try:
            sample = df.iloc[0].to_dict()
            print(f"     first row keys: {list(sample.keys())[:6]}")
        except Exception:
            pass

    # Static
    tsv = feat_dir / "static_features.tsv"
    if tsv.exists():
        df = pd.read_csv(tsv, sep="\t", nrows=5)
        print(f"\n--- static_features.tsv ({df.shape[1]} cols shown 0..5) ---")
        print(f"  {list(df.columns)[:8]}{' ...' if df.shape[1] > 8 else ''}")

    # Phenotype TSVs
    print("\n--- phenotype/diagnosis/*.tsv (row counts) ---")
    diag_dir = root_dir / "phenotype" / "diagnosis"
    for tsv in sorted(diag_dir.glob("*.tsv")):
        try:
            n = sum(1 for _ in open(tsv)) - 1
        except Exception:
            n = -1
        print(f"  {tsv.name}: {n} rows")

    demo_path = root_dir / "phenotype" / "demographics" / "demographics.tsv"
    if demo_path.exists():
        ddf = pd.read_csv(demo_path, sep="\t", nrows=5)
        print(f"\n--- demographics.tsv ---")
        print(f"  cols: {list(ddf.columns)}")

    rec_path = root_dir / "phenotype" / "task" / "recording.tsv"
    if rec_path.exists():
        rdf = pd.read_csv(rec_path, sep="\t", nrows=5)
        print(f"\n--- recording.tsv ---")
        print(f"  cols: {list(rdf.columns)}")

    # Try full manifest build
    print("\n--- Attempting to build master manifest (all diseases) ---")
    try:
        rec, static, n_stat = build_master_manifest(
            root_dir, list(DISEASE_FILE_MAP.keys())
        )
        print(f"  manifest:  {len(rec)} recordings, "
              f"{rec.participant_id.nunique()} participants")
        print(f"  static:    {len(static)} participants x {n_stat} features")
        disease_cols = [c for c in rec.columns if c in DISEASE_FILE_MAP]
        counts = rec.drop_duplicates("participant_id")[disease_cols].sum().to_dict()
        print("  positives per disease:")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {k:24s} {int(v):4d}")
        print(f"  tasks: {sorted(rec.task.unique())[:10]}"
              + (" ..." if rec.task.nunique() > 10 else ""))
        print(f"  sites: {rec.drop_duplicates('participant_id')['site'].value_counts().to_dict()}")
        print(f"  sex:   {rec.drop_duplicates('participant_id')['sex'].value_counts().to_dict()}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
