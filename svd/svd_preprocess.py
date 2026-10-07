"""
svd_preprocess.py  —  SVD → TC-DANN feature pipeline
Bridge2AI-Voice compatible output

Built exactly for the confirmed SVD subset on disk:

    data/
        Amyotrophe Lateralsklerose/   → amyotrophic_lateral_sclerosis
        Bulbärparalyse/               → unilateral_vocal_fold_paralysis
        Carcinoma in situ/            → precancerous_lesions
        GERD/                         → laryngitis
        Laryngozele/                  → benign_lesions
        Monochorditis/                → benign_lesions
        Morbus Parkinson/             → parkinsons_disease
        Poltersyndrom/                → SKIPPED (no TC-DANN mapping)

Each speaker folder contains:
    vowels/     <id>-a_n.nsp  <id>-i_n.nsp  <id>-u_n.nsp   (normal pitch only)
                <id>-a_h.nsp  <id>-a_l.nsp  <id>-a_lhl.nsp  (skipped by default)
                <id>-iau.nsp                                   (skipped)
    sentences/  <id>-phrase.nsp
    remarks/    <id>-remarks.txt   (ignored)
    *.egg files                    (EGG signal — ignored)

═══════════════════════════════════════════════════════════════════
INSTALL DEPENDENCIES
═══════════════════════════════════════════════════════════════════

    pip install torch torchaudio pyarrow pandas opensmile tqdm librosa soundfile

    # sox does NOT support NSP — use praat instead:
    brew install praat        # macOS (handles NSP natively)
    # OR: brew install ffmpeg (sometimes works)

    # SPARC (optional — librosa used as fallback if absent):
    pip install sparc-toolkit

═══════════════════════════════════════════════════════════════════
RUN
═══════════════════════════════════════════════════════════════════

    # Smoke-test first (processes first 20 NSP files):
    python svd_preprocess.py --svd_root /path/to/data --out_root /path/to/svd_out --smoke_test

    # Full run — SVD standalone:
    python svd_preprocess.py --svd_root /path/to/data --out_root /path/to/svd_out

    # Full run + auto-merge with Bridge2AI (recommended):
    python svd_preprocess.py --svd_root /path/to/data --out_root /path/to/svd_out --merge_b2ai /path/to/bridge2ai

═══════════════════════════════════════════════════════════════════
THEN RUN TC-DANN
═══════════════════════════════════════════════════════════════════

    # Standalone:
    python run_tc_dann.py --data_root /path/to/svd_out --epochs 40

    # Merged with Bridge2AI (gradient reversal treats dataset as a site):
    python run_tc_dann.py --data_root /path/to/svd_out_merged --epochs 40

═══════════════════════════════════════════════════════════════════
"""

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import sparc as _sparc
    SPARC_AVAILABLE = True
except ImportError:
    SPARC_AVAILABLE = False

try:
    import librosa as _librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

if not SPARC_AVAILABLE and not LIBROSA_AVAILABLE:
    sys.exit("Neither sparc-toolkit nor librosa found.\npip install librosa")

try:
    import opensmile as _opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    print("WARNING: opensmile not found — jitter/shimmer/alpha-ratio features will be absent.\n"
          "pip install opensmile\n")

SOX_AVAILABLE    = shutil.which("sox")   is not None
PRAAT_AVAILABLE  = shutil.which("praat") is not None
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

if not any([SOX_AVAILABLE, PRAAT_AVAILABLE, FFMPEG_AVAILABLE]):
    print("WARNING: sox/praat/ffmpeg all missing — using built-in NSP reader (less robust).\n"
          "Recommended:  brew install praat\n")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match Bridge2AI / run_tc_dann.py exactly
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR  = 16_000
N_MFCC     = 60
N_MELS     = 80
N_FFT      = 400
HOP_LENGTH = 160

# Only normal-pitch vowels + sentence on first run.
# _h/_l/_lhl variants introduce within-speaker pitch variation that
# Bridge2AI does not have — keep off unless you have a specific reason.
INCLUDE_PITCH_VARIANTS = False

VOWEL_TASK_MAP: dict[str, str] = {
    "a_n":    "prolonged-vowel",
    "i_n":    "prolonged-vowel",
    "u_n":    "prolonged-vowel",
    "phrase": "read-speech",
}
if INCLUDE_PITCH_VARIANTS:
    for v in ["a_h","a_l","a_lhl","i_h","i_l","i_lhl","u_h","u_l","u_lhl","iau"]:
        VOWEL_TASK_MAP[v] = "prolonged-vowel"

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE MAP — only what is on disk
# ─────────────────────────────────────────────────────────────────────────────

# Exact folder name (case-insensitive) → TC-DANN disease name
# Updated FOLDER_DISEASE_MAP for svd_preprocess.py
FOLDER_DISEASE_MAP: dict[str, str] = {
    # --- Neurological ---
    "amyotrophe lateralsklerose": "amyotrophic_lateral_sclerosis",
    "bulbärparalyse":             "unilateral_vocal_fold_paralysis", 
    "morbus parkinson":           "parkinsons_disease",
    "dysarthrophonie":            "parkinsons_disease",
    
    # --- Voice + Onco ---
    "phonationsknötchen":         "benign_lesions",
    "stimmlippenpolyp":           "benign_lesions",
    "laryngozele":                "benign_lesions",
    "monochorditis":              "benign_lesions",
    "carcinoma in situ":          "precancerous_lesions",
    "leukoplakie":                "precancerous_lesions",
    "stimmlippenkarzinom":        "laryngeal_cancer",
    "chordektomie":               "unilateral_vocal_fold_paralysis",
    
    # --- Respiratory ---
    "gerd":                       "laryngitis",
    "laryngitis":                 "laryngitis",
    
    # --- Psychiatric / Fluency ---
    "poltersyndrom":              "adhd_adult",  # Mapping Cluttering to Fluency/ADHD head
    "balbuties":                  "adhd_adult",  # Mapping Stuttering to Fluency/ADHD head
    "psychogene dysphonie":       "anxiety",
    
    # --- Control ---
    "gesangsstimme":              "control",
}

def folder_to_disease(name: str) -> str | None:
    return FOLDER_DISEASE_MAP.get(name.strip().lower())


# ─────────────────────────────────────────────────────────────────────────────
# NSP → WAV CONVERSION
# Kay Pentax/CSL .nsp format. sox does NOT support NSP natively despite being
# installed. Strategy order:
#   1. Praat  — handles NSP natively (brew install praat)
#   2. ffmpeg — sometimes works depending on build
#   3. Raw binary reader — parses the NSP header directly (always attempted)
# ─────────────────────────────────────────────────────────────────────────────

def _praat_convert(nsp: Path, wav: Path) -> bool:
    """Use Praat's command-line interface to convert NSP → WAV."""
    try:
        script = (
            f"sound = Read from file: \"{nsp}\"\n"
            f"Save as WAV file: \"{wav}\"\n"
            f"Remove\n"
        )
        script_path = wav.with_suffix(".praat")
        script_path.write_text(script)
        r = subprocess.run(
            ["praat", "--run", str(script_path)],
            capture_output=True, timeout=60,
        )
        script_path.unlink(missing_ok=True)
        return r.returncode == 0 and wav.exists() and wav.stat().st_size > 44
    except Exception:
        return False


def _ffmpeg_convert(nsp: Path, wav: Path) -> bool:
    """Try ffmpeg — works if built with NSP demuxer."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(nsp), str(wav)],
            capture_output=True, timeout=30,
        )
        return r.returncode == 0 and wav.exists() and wav.stat().st_size > 44
    except Exception:
        return False


def _raw_nsp_convert(nsp: Path, wav: Path) -> bool:
    """
    Direct binary parser for Kay Pentax NSP / CSL format.

    SVD uses Kay Pentax CSL (Computerized Speech Lab). The NSP header
    is 512 bytes long. Key fields (all little-endian):
      offset   0: char[8]  — magic "MDVP    " or similar
      offset   8: uint32   — sample rate (Hz), typically 50000
      offset  12: uint16   — bits per sample (16 for SVD)
      offset  14: uint16   — number of channels (1)
      offset  16: uint32   — number of samples
      offset 512: int16[]  — raw PCM data begins here

    We try the 512-byte offset first (standard CSL), then scan for the
    PCM start heuristically if that produces silence or garbage.
    """
    try:
        raw = nsp.read_bytes()
        if len(raw) < 600:
            return False

        # ── Try standard 512-byte CSL header ─────────────────────────────
        def _try_offset(off: int, sr: int, bits: int, nch: int) -> np.ndarray | None:
            dtype = {8: np.int8, 16: np.int16, 32: np.int32}.get(bits)
            if dtype is None:
                return None
            try:
                audio = np.frombuffer(raw[off:], dtype=dtype).astype(np.float32)
                if len(audio) < 100:
                    return None
                if bits == 8:    audio = (audio - 128) / 128.0
                elif bits == 16: audio /= 32_768.0
                elif bits == 32: audio /= 2_147_483_648.0
                if nch == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)
                # Sanity: non-silent
                if np.abs(audio).max() < 1e-6:
                    return None
                return audio
            except Exception:
                return None

        # Read header fields at standard CSL offsets
        try:
            sr_h   = struct.unpack_from("<I", raw, 8)[0]
            bits_h = struct.unpack_from("<H", raw, 12)[0]
            nch_h  = struct.unpack_from("<H", raw, 14)[0]
        except Exception:
            sr_h, bits_h, nch_h = 0, 0, 0

        # Validate
        sr_ok   = 8_000 <= sr_h <= 96_000
        bits_ok = bits_h in (8, 16, 32)
        nch_ok  = nch_h in (1, 2)

        audio = None

        if sr_ok and bits_ok and nch_ok:
            audio = _try_offset(512, sr_h, bits_h, nch_h)   # standard CSL
            if audio is None:
                audio = _try_offset(16, sr_h, bits_h, nch_h)  # short-header variant

        # Heuristic fallback: SVD is recorded at 50 kHz, 16-bit, mono
        if audio is None:
            for off in (512, 1024, 256, 16):
                audio = _try_offset(off, 50_000, 16, 1)
                if audio is not None:
                    sr_h = 50_000
                    break

        if audio is None:
            return False

        tensor = torch.from_numpy(audio).unsqueeze(0)   # [1, T]
        torchaudio.save(str(wav), tensor, sr_h)
        return wav.exists() and wav.stat().st_size > 44

    except Exception:
        return False
# Shared temp dir for WAV cache (cleaned up at end)
_TMP_DIR = Path(tempfile.mkdtemp(prefix="svd_wav_"))


def load_nsp(path: str) -> torch.Tensor:
    """
    Convert NSP → WAV using best available method, cache result, return [1, T].
    Conversion order: Praat → ffmpeg → raw binary parser.
    sox is intentionally NOT used — it does not support NSP format.
    """
    nsp = Path(path)
    wav = _TMP_DIR / (nsp.stem + ".wav")

    if not wav.exists():
        ok = False
        if PRAAT_AVAILABLE:
            ok = _praat_convert(nsp, wav)
        if not ok and FFMPEG_AVAILABLE:
            ok = _ffmpeg_convert(nsp, wav)
        if not ok:
            ok = _raw_nsp_convert(nsp, wav)
        if not ok:
            raise RuntimeError(
                f"Cannot convert {nsp.name} to WAV.\n"
                f"Install praat:  brew install praat\n"
                f"Then retry."
            )

    audio, sr = torchaudio.load(str(wav))
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SR)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    return audio   # [1, T]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — MANIFEST
# Walks exactly:  <disease_dir>/<speaker_id>/vowels/*.nsp
#                 <disease_dir>/<speaker_id>/sentences/*-phrase.nsp
#                 <disease_dir>/overview.csv  (age/sex metadata)
# Ignores: *.egg, remarks/, pitch-variant NSPs (unless flag set)
# ─────────────────────────────────────────────────────────────────────────────

def _read_overview(disease_dir: Path) -> dict[str, dict]:
    """Returns {zero-padded-id: {age, sex}} from overview.csv if present."""
    overview: dict[str, dict] = {}
    for name in ("overview.csv", "Overview.csv"):
        p = disease_dir / name
        if not p.exists():
            continue
        try:
            ov = pd.read_csv(p, dtype=str, sep=None, engine="python")
            ov.columns = [c.strip().lower() for c in ov.columns]
            id_c  = next((c for c in ov.columns
                          if c in ("id","speaker","nummer","pat.-nr.","pat_nr")), None)
            age_c = next((c for c in ov.columns if "age" in c or "alter" in c), None)
            sex_c = next((c for c in ov.columns
                          if c in ("sex","gender","geschlecht")), None)
            if id_c is None:
                break
            for _, r in ov.iterrows():
                sid = str(r[id_c]).strip().zfill(4)
                overview[sid] = {
                    "age": (float(r[age_c])
                            if age_c and str(r[age_c]).strip() not in ("","nan")
                            else np.nan),
                    "sex": (str(r[sex_c]).strip()
                            if sex_c and str(r[sex_c]).strip() not in ("","nan")
                            else ""),
                }
        except Exception:
            pass
        break
    return overview


def build_manifest(svd_root: Path, site_id: str, smoke_test: bool) -> pd.DataFrame:
    rows: list[dict] = []
    skipped_folders: list[str] = []

    for disease_dir in sorted(svd_root.iterdir()):
        if not disease_dir.is_dir():
            continue

        disease = folder_to_disease(disease_dir.name)
        if disease is None:
            skipped_folders.append(disease_dir.name)
            continue

        overview = _read_overview(disease_dir)

        for speaker_dir in sorted(disease_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            raw_id = speaker_dir.name.strip()
            pid    = f"svd_{raw_id.zfill(4)}"
            meta   = overview.get(raw_id.zfill(4), {"age": np.nan, "sex": ""})

            # Vowel NSPs
            vowel_dir = speaker_dir / "vowels"
            if vowel_dir.exists():
                for nsp in sorted(vowel_dir.glob("*.nsp")):
                    # stem like "1242-a_n" → suffix "a_n"
                    suffix = nsp.stem.split("-", 1)[-1]   # everything after first dash
                    task   = VOWEL_TASK_MAP.get(suffix)
                    if task is None:
                        continue
                    rows.append({
                        "participant_id":  pid,
                        "session_id":      f"{pid}_{suffix}",
                        "task_name":       task,
                        "path":            str(nsp),
                        "tc_dann_disease": disease,
                        "site_id":         site_id,
                        "age":             meta["age"],
                        "sex":             meta["sex"],
                    })

            # Sentence NSP
            sent_dir = speaker_dir / "sentences"
            if sent_dir.exists():
                for nsp in sorted(sent_dir.glob("*-phrase.nsp")):
                    rows.append({
                        "participant_id":  pid,
                        "session_id":      f"{pid}_phrase",
                        "task_name":       "read-speech",
                        "path":            str(nsp),
                        "tc_dann_disease": disease,
                        "site_id":         site_id,
                        "age":             meta["age"],
                        "sex":             meta["sex"],
                    })

    if skipped_folders:
        print(f"  Skipped (no TC-DANN mapping): {', '.join(skipped_folders)}")

    if not rows:
        sys.exit("No NSP files found. Check --svd_root.")

    manifest = pd.DataFrame(rows)
    if smoke_test:
        manifest = manifest.head(20).reset_index(drop=True)
        print("  [smoke_test] limited to 20 recordings")

    print(f"\nManifest built:")
    print(f"  {len(manifest)} recordings | "
          f"{manifest['participant_id'].nunique()} participants | "
          f"{manifest['tc_dann_disease'].nunique()} diseases")
    print("\n  Participants per disease:")
    counts = (manifest.groupby("tc_dann_disease")["participant_id"]
              .nunique().sort_values(ascending=False))
    for d, n in counts.items():
        warn = "  ← WARNING n<10, model will skip" if n < 10 else ""
        print(f"    {d:<45} n={n:>4}{warn}")

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MFCC  →  features/torchaudio_mfcc.parquet
# Schema matches run_tc_dann.py load_mfcc() exactly:
#   participant_id, session_id, task_name, n_frames, mfcc (list[list[float]])
#   mfcc shape: [60, T]
# ─────────────────────────────────────────────────────────────────────────────

def extract_mfcc(manifest: pd.DataFrame, out_dir: Path) -> None:
    xform = torchaudio.transforms.MFCC(
        sample_rate=TARGET_SR,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH,
                   "n_mels": N_MELS, "window_fn": torch.hann_window},
    )
    records: list[dict] = []
    errors = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="MFCC"):
        try:
            mfcc = xform(load_nsp(row["path"]))[0].numpy()   # [60, T]
            if mfcc.shape[0] != N_MFCC:
                continue
            records.append({
                "participant_id": row["participant_id"],
                "session_id":     row["session_id"],
                "task_name":      row["task_name"],
                "n_frames":       int(mfcc.shape[1]),
                "mfcc":           mfcc.tolist(),
            })
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  MFCC error [{Path(row['path']).name}]: {e}")

    out = out_dir / "torchaudio_mfcc.parquet"
    pq.write_table(pa.Table.from_pylist(records), out)
    print(f"  → {out.name}  ({len(records)} rows, {errors} errors)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SPARC / librosa  →  features/sparc_*.parquet
# pitch, loudness, periodicity as 1-D arrays per recording.
# sparc_ema.parquet is always empty — SVD has no articulatory data.
# ─────────────────────────────────────────────────────────────────────────────

def _pitch_lb(w: np.ndarray) -> np.ndarray:
    f0, _, _ = _librosa.pyin(w, fmin=50, fmax=500,
                              sr=TARGET_SR, hop_length=HOP_LENGTH)
    return np.where(np.isnan(f0), 0.0, f0).astype(np.float32)

def _loudness_lb(w: np.ndarray) -> np.ndarray:
    rms = _librosa.feature.rms(y=w, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    return _librosa.amplitude_to_db(rms, ref=np.max).astype(np.float32)

def _periodicity_lb(w: np.ndarray) -> np.ndarray:
    _, vf, _ = _librosa.pyin(w, fmin=50, fmax=500,
                              sr=TARGET_SR, hop_length=HOP_LENGTH)
    return vf.astype(np.float32)

_LB = {"pitch": _pitch_lb, "loudness": _loudness_lb, "periodicity": _periodicity_lb}


def extract_sparc_features(manifest: pd.DataFrame, out_dir: Path) -> None:
    backend = "SPARC" if SPARC_AVAILABLE else "librosa"
    print(f"  Backend: {backend}")

    for feat in ["pitch", "loudness", "periodicity"]:
        records: list[dict] = []
        errors = 0
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=feat):
            try:
                wav_np = load_nsp(row["path"])[0].numpy()
                if SPARC_AVAILABLE:
                    arr = _sparc.extract(wav_np, TARGET_SR, feature=feat)
                    if isinstance(arr, tuple):
                        arr = arr[0]
                    arr = np.asarray(arr, dtype=np.float32)
                else:
                    arr = _LB[feat](wav_np)
                records.append({
                    "participant_id": row["participant_id"],
                    "session_id":     row["session_id"],
                    "task_name":      row["task_name"],
                    feat:             arr.tolist(),
                })
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\n  {feat} error [{Path(row['path']).name}]: {e}")

        out = out_dir / f"sparc_{feat}.parquet"
        pq.write_table(pa.Table.from_pylist(records), out)
        print(f"  → {out.name}  ({len(records)} rows, {errors} errors)")

    # EMA always empty — no articulatory data in SVD
    pq.write_table(pa.Table.from_pylist([]), out_dir / "sparc_ema.parquet")
    print("  → sparc_ema.parquet  (empty — SVD has no EMA data)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — openSMILE static features  →  features/static_features.tsv
# ComParE_2016 functionals: jitter, shimmer, alpha ratio, F0 slopes, etc.
# These are your top SHAP features for Voice+Onco — do not skip opensmile.
# ─────────────────────────────────────────────────────────────────────────────

def extract_static(manifest: pd.DataFrame, out_dir: Path, tmp_wav: Path) -> None:
    if not OPENSMILE_AVAILABLE:
        print("  opensmile unavailable — writing minimal TSV (IDs + demographics only).")
        minimal = manifest[["participant_id", "session_id", "age", "sex", "site_id"]].copy()
        minimal["bmi"] = np.nan
        minimal.drop_duplicates(["participant_id", "session_id"]).to_csv(
            out_dir / "static_features.tsv", sep="\t", index=False)
        print("  → static_features.tsv  (minimal — install opensmile for full features)")
        return

    smile = _opensmile.Smile(
        feature_set=_opensmile.FeatureSet.ComParE_2016,
        feature_level=_opensmile.FeatureLevel.Functionals,
    )
    rows: list[dict] = []
    errors = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="openSMILE"):
        try:
            # openSMILE needs a WAV path — convert NSP and cache in tmp_wav
            nsp = Path(row["path"])
            wav = tmp_wav / (nsp.stem + ".wav")
            if not wav.exists():
                ok = False
                if PRAAT_AVAILABLE:
                    ok = _praat_convert(nsp, wav)
                if not ok and FFMPEG_AVAILABLE:
                    ok = _ffmpeg_convert(nsp, wav)
                if not ok:
                    ok = _raw_nsp_convert(nsp, wav)
                if not ok:
                    raise RuntimeError("NSP→WAV conversion failed")

            d = smile.process_file(str(wav)).iloc[0].to_dict()
            d["participant_id"] = row["participant_id"]
            d["session_id"]     = row["session_id"]
            d["age"]            = row.get("age",  np.nan)
            d["sex"]            = row.get("sex",  "")
            d["bmi"]            = np.nan   # not in SVD — placeholder so DemographicIndex sees the column
            d["site_id"]        = row.get("site_id", "svd")
            rows.append(d)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  openSMILE error [{Path(row['path']).name}]: {e}")

    out = out_dir / "static_features.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"  → {out.name}  ({len(rows)} rows, {errors} errors)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DIAGNOSIS TSVs  →  phenotype/diagnosis/<disease>.tsv
# One TSV per disease, one column: participant_id
# run_tc_dann.py reads these to build binary labels.
# ─────────────────────────────────────────────────────────────────────────────

def write_diagnosis_tsvs(manifest: pd.DataFrame, pheno_dir: Path) -> None:
    pheno_dir.mkdir(parents=True, exist_ok=True)
    pid_counts = (manifest.groupby("tc_dann_disease")["participant_id"].nunique())

    for disease, grp in manifest.groupby("tc_dann_disease"):
        n   = pid_counts[disease]
        out = pheno_dir / f"{disease}.tsv"
        grp[["participant_id"]].drop_duplicates().to_csv(out, sep="\t", index=False)
        warn = "  ← n<10, model will skip this head" if n < 10 else ""
        print(f"  {disease:<45} n={n:>4}{warn}")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL — MERGE WITH BRIDGE2AI
# Concatenates SVD output with Bridge2AI into one merged data_root.
# site_id="svd" vs Bridge2AI site IDs → gradient reversal suppresses
# dataset-origin artefacts, giving you a real cross-dataset site-invariance test.
# ─────────────────────────────────────────────────────────────────────────────

def merge_with_b2ai(svd_out: Path, b2ai_root: Path, merged_root: Path) -> None:
    mf = merged_root / "features"
    mp = merged_root / "phenotype" / "diagnosis"
    mf.mkdir(parents=True, exist_ok=True)
    mp.mkdir(parents=True, exist_ok=True)

    print(f"\nMerging into {merged_root} ...")

    for pf in ["torchaudio_mfcc.parquet", "sparc_pitch.parquet",
               "sparc_loudness.parquet", "sparc_periodicity.parquet",
               "sparc_ema.parquet"]:
        tables = []
        for root in [svd_out, b2ai_root]:
            p = root / "features" / pf
            if p.exists():
                t = pq.read_table(p)
                if t.num_rows > 0:
                    tables.append(t)
        if tables:
            merged = pa.concat_tables(tables, promote_options="default")
            pq.write_table(merged, mf / pf)
            print(f"  → {pf}  ({merged.num_rows} total rows)")

    dfs = []
    for root in [svd_out, b2ai_root]:
        p = root / "features" / "static_features.tsv"
        if p.exists():
            dfs.append(pd.read_csv(p, sep="\t"))
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(
            mf / "static_features.tsv", sep="\t", index=False)
        print("  → static_features.tsv")

    all_diseases: set[str] = set()
    for root in [svd_out, b2ai_root]:
        d = root / "phenotype" / "diagnosis"
        if d.exists():
            all_diseases |= {p.stem for p in d.glob("*.tsv")}

    for disease in sorted(all_diseases):
        frames = []
        for root in [svd_out, b2ai_root]:
            p = root / "phenotype" / "diagnosis" / f"{disease}.tsv"
            if p.exists():
                frames.append(pd.read_csv(p, sep="\t"))
        if frames:
            merged_df = pd.concat(frames, ignore_index=True).drop_duplicates()
            merged_df.to_csv(mp / f"{disease}.tsv", sep="\t", index=False)
            print(f"  → {disease}.tsv  (n={len(merged_df)})")

    print(f"\nMerged data root ready: {merged_root}")
    print(f"  python run_tc_dann.py --data_root {merged_root} --epochs 40")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVD → TC-DANN preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--svd_root",   required=True,
                        help="Path to SVD data directory (contains disease subfolders)")
    parser.add_argument("--out_root",   required=True,
                        help="Output data_root for TC-DANN (will be created)")
    parser.add_argument("--site_id",    default="svd",
                        help="Site label for domain adversary (default: svd)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Process first 20 recordings only — for testing")
    parser.add_argument("--merge_b2ai", default=None,
                        help="Path to Bridge2AI data_root. If given, merges SVD output "
                             "with B2AI into <out_root>_merged/")
    args = parser.parse_args()

    svd_root  = Path(args.svd_root)
    out_root  = Path(args.out_root)
    feat_dir  = out_root / "features"
    pheno_dir = out_root / "phenotype" / "diagnosis"
    tmp_wav   = out_root / "_tmp_wav"

    for d in [feat_dir, pheno_dir, tmp_wav]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SVD → TC-DANN preprocessing pipeline")
    print("=" * 65)
    print(f"  svd_root  : {svd_root}")
    print(f"  out_root  : {out_root}")
    print(f"  site_id   : {args.site_id}")
    print(f"  smoke_test: {args.smoke_test}")
    print(f"  praat     : {'yes' if PRAAT_AVAILABLE else 'no'}")
    print(f"  ffmpeg    : {'yes' if FFMPEG_AVAILABLE else 'no'}")
    print(f"  raw NSP   : yes (always available as fallback)")
    print(f"  SPARC     : {'yes' if SPARC_AVAILABLE else 'no — librosa fallback'}")
    print(f"  opensmile : {'yes' if OPENSMILE_AVAILABLE else 'no — minimal static TSV'}")
    print("=" * 65)

    print("\n[1/5] Building manifest ...")
    manifest = build_manifest(svd_root, args.site_id, args.smoke_test)

    print("\n[2/5] Extracting MFCC ...")
    extract_mfcc(manifest, feat_dir)

    print("\n[3/5] Extracting pitch / loudness / periodicity / EMA ...")
    extract_sparc_features(manifest, feat_dir)

    print("\n[4/5] Extracting static features (openSMILE ComParE_2016) ...")
    extract_static(manifest, feat_dir, tmp_wav)

    print("\n[5/5] Writing diagnosis TSVs ...")
    write_diagnosis_tsvs(manifest, pheno_dir)

    # Clean up temp WAV cache
    shutil.rmtree(tmp_wav,   ignore_errors=True)
    shutil.rmtree(_TMP_DIR,  ignore_errors=True)

    if args.merge_b2ai:
        b2ai_root   = Path(args.merge_b2ai)
        merged_root = out_root.parent / (out_root.name + "_merged")
        merge_with_b2ai(out_root, b2ai_root, merged_root)

    print("\n" + "=" * 65)
    print("Output layout:")
    print(f"  {out_root}/")
    print(f"    features/")
    print(f"      torchaudio_mfcc.parquet")
    print(f"      sparc_pitch.parquet")
    print(f"      sparc_loudness.parquet")
    print(f"      sparc_periodicity.parquet")
    print(f"      sparc_ema.parquet          (empty — SVD has no EMA)")
    print(f"      static_features.tsv")
    print(f"    phenotype/diagnosis/")
    print(f"      amyotrophic_lateral_sclerosis.tsv")
    print(f"      unilateral_vocal_fold_paralysis.tsv")
    print(f"      precancerous_lesions.tsv")
    print(f"      laryngitis.tsv")
    print(f"      benign_lesions.tsv")
    print(f"      parkinsons_disease.tsv")
    print()
    print("Next steps:")
    print(f"  # SVD standalone:")
    print(f"  python run_tc_dann.py --data_root {out_root} --epochs 40")
    print()
    print(f"  # Merged with Bridge2AI (recommended):")
    print(f"  python svd_preprocess.py --svd_root {svd_root} --out_root {out_root} \\")
    print(f"      --merge_b2ai /path/to/bridge2ai")
    print(f"  python run_tc_dann.py --data_root {out_root}_merged --epochs 40")
    print("=" * 65)


if __name__ == "__main__":
    main()
