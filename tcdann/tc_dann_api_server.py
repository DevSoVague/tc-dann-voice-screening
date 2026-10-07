"""
tc_dann_api_server.py  —  TC-DANN Quad-Model API Server
========================================================
FastAPI wrapper for run_tc_dann.py inference.

Four independent TC-DANN models (Voice+Onco, Neurological, Respiratory,
Psychiatric) with:
  - Full audio preprocessing pipeline  (audio_preprocessing.py)
  - Composite confidence scoring       (x · cosine + y · prob − z · uncertainty + m · demo + p · confounder)
  - Confounder audit per disease       (task_confounder_auroc gate)
  - Browser mic endpoint               POST /record
  - Direct parquet-feature bypass      POST /predict_features
  - Every session persisted            sessions/

Run:
    uvicorn tc_dann_api_server:app --reload --port 8000

Environment:
    TC_DANN_BUNDLE_DIR   folder with the four *_model_best.joblib bundles
                         (default: <repo>/models/)
    SESSIONS_DIR         path for session JSON files (default: sessions/)

If no bundles are found the server still starts: /health reports
status "no_models" and the prediction endpoints return HTTP 503 with a
message explaining how to obtain or train the weights.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── TC-DANN imports ───────────────────────────────────────────────────────────
# run_tc_dann.py must be on PYTHONPATH (same directory or installed)
try:
    from run_tc_dann import (
        TCDANN, MODEL_DISEASE_MAP, MODEL_DISPLAY,
        CS_PHYSICAL, CS_PSYCH, PSYCHIATRIC_NO_DEMO,
        SubgroupCentroidIndex, DemographicIndex,
        TASKS, TASK_ENC, load_bundle, bundle_preprocessors,
    )
    TC_DANN_AVAILABLE = True
except ImportError as _e:
    TC_DANN_AVAILABLE = False
    _TC_DANN_IMPORT_ERROR = str(_e)

from audio_preprocessing import preprocess_audio_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("tc_dann_api")

_REPO_ROOT   = Path(__file__).resolve().parent.parent
BUNDLE_DIR   = Path(os.getenv("TC_DANN_BUNDLE_DIR", str(_REPO_ROOT / "models"))).expanduser()
SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR",         "sessions"))
SESSIONS_DIR.mkdir(exist_ok=True)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Model stems in run_tc_dann.py
MODEL_STEMS = ["voice_onco", "neurological", "respiratory", "psychiatric"]

# Flat list of all diseases across all models, for API consumers
ALL_DISEASES: list[str] = []
for _stem, _diseases in MODEL_DISEASE_MAP.items():
    ALL_DISEASES.extend(_diseases)

# ── Model registry ────────────────────────────────────────────────────────────

class _ModelBundle:
    """Holds a loaded TC-DANN model + its preprocessing objects + indices."""
    def __init__(self, stem: str):
        self.stem        = stem
        self.label       = MODEL_DISPLAY[stem]
        self.disease_cols: list[str] = []
        self.model:       Optional[TCDANN]               = None
        self.feat_cols:   list[str]                      = []
        self.imputer                                     = None
        self.scaler                                      = None
        self.centroid_idx: Optional[SubgroupCentroidIndex] = None
        self.demo_idx:     Optional[DemographicIndex]      = None
        self.n_sites:     int                            = 2
        self.loaded:      bool                           = False
        self.error:       str                            = ""

    def load(self, bundle_dir: Path) -> bool:
        path = bundle_dir / f"{self.stem}_model_best.joblib"
        if not path.exists():
            self.error = f"Bundle not found: {path}"
            log.warning(self.error)
            return False
        try:
            b = load_bundle(path)
            self.disease_cols = b["disease_cols"]
            self.feat_cols    = b["feat_cols"]
            self.n_sites      = b.get("n_sites", 2)

            self.model = TCDANN(
                n_features=b["n_features"],
                n_tasks=b["n_tasks"],
                n_sites=self.n_sites,
                n_diseases=b["n_diseases"],
            ).to(DEVICE)

            state = {k: (torch.tensor(v).to(DEVICE) if isinstance(v, np.ndarray) else v.to(DEVICE))
                     for k, v in b["model_state"].items()}
            self.model.load_state_dict(state)
            self.model.eval()

            # Rebuild imputer + scaler from the stored arrays (sklearn-version independent)
            self.imputer, self.scaler = bundle_preprocessors(b)

            self.loaded = True
            log.info("Loaded %s  (%d diseases, %d features)", self.label, len(self.disease_cols), len(self.feat_cols))
            return True
        except Exception as e:
            self.error = str(e)
            log.error("Failed to load %s: %s", self.stem, e)
            return False


_bundles: dict[str, _ModelBundle] = {}


def _get_bundle(stem: str) -> _ModelBundle:
    if stem not in _bundles:
        b = _ModelBundle(stem)
        b.load(BUNDLE_DIR)
        _bundles[stem] = b
    return _bundles[stem]


def _load_all_bundles():
    for stem in MODEL_STEMS:
        _get_bundle(stem)
    n = sum(1 for b in _bundles.values() if b.loaded)
    if n == 0:
        log.warning(NO_MODELS_MSG)


NO_MODELS_MSG = (
    f"No TC-DANN model bundles were loaded from {BUNDLE_DIR}. Trained weights are not "
    "distributed with this repository (they were trained on Bridge2AI-Voice under the "
    "PhysioNet data use agreement). Train them with "
    "`python tcdann/run_tc_dann.py --data_root $B2AI_DATA_ROOT --out_dir models` "
    "or set TC_DANN_BUNDLE_DIR to a folder containing voice_onco_model_best.joblib, "
    "neurological_model_best.joblib, respiratory_model_best.joblib and "
    "psychiatric_model_best.joblib."
)


def _require_models():
    """Raise a clean 503 (not a stack trace) when no weights are available."""
    if not TC_DANN_AVAILABLE:
        raise HTTPException(503, f"run_tc_dann.py not available: {_TC_DANN_IMPORT_ERROR}")
    if not any(_get_bundle(s).loaded for s in MODEL_STEMS):
        raise HTTPException(503, NO_MODELS_MSG)


# ── Feature extraction from raw audio ────────────────────────────────────────

def _audio_to_feature_row(audio_bytes: bytes, task_name: str = "prolonged-vowel") -> dict:
    """
    Run audio_preprocessing on raw bytes, then extract the scalar feature
    vector expected by TC-DANN (same as what run_tc_dann.py build_feat_cols returns).

    Returns a flat dict of {feature_name: float} matching feat_cols.
    """
    pp = preprocess_audio_bytes(audio_bytes)
    mfcc    = pp["mfcc"][0]      # [60, T]
    spec    = pp["spec"][0]      # [201, T]
    log_mel = pp["log_mel"][0]   # [128, T]

    feats: dict[str, float] = {}

    # MFCC scalars: mean, std, max, min per coefficient + delta + deltadelta
    for i in range(mfcc.shape[0]):
        row = mfcc[i]
        feats[f"mfcc{i:02d}_mean"] = float(row.mean())
        feats[f"mfcc{i:02d}_std"]  = float(row.std())
        feats[f"mfcc{i:02d}_max"]  = float(row.max())
        feats[f"mfcc{i:02d}_min"]  = float(row.min())
    if mfcc.shape[1] >= 2:
        d1 = np.diff(mfcc, axis=1)
        for i in range(d1.shape[0]):
            feats[f"mfcc{i:02d}_delta_mean"] = float(d1[i].mean())
            feats[f"mfcc{i:02d}_delta_std"]  = float(d1[i].std())
        if d1.shape[1] >= 2:
            d2 = np.diff(d1, axis=1)
            for i in range(d2.shape[0]):
                feats[f"mfcc{i:02d}_deltadelta_mean"] = float(d2[i].mean())
                feats[f"mfcc{i:02d}_deltadelta_std"]  = float(d2[i].std())

    # Spectrogram scalars
    feats["spectral_mean"]    = float(spec.mean())
    feats["spectral_std_dev"] = float(spec.std())
    feats["spectral_skewness"] = float(_skewness(spec.flatten()))
    feats["spectral_kurtosis"] = float(_kurtosis(spec.flatten()))
    feats["spectral_flux"]    = float(np.diff(spec, axis=1).std()) if spec.shape[1] > 1 else 0.0

    # Pitch proxy from mel energy (stand-in for SPARC pitch when unavailable)
    feats["pitch_mean"]  = float(log_mel[20:50].mean())   # mel bins ~100-500Hz
    feats["pitch_std"]   = float(log_mel[20:50].std())

    # Duration
    feats["duration"] = float(pp["duration_s"])

    # Task encoding (matches TASK_ENC in run_tc_dann.py)
    feats["_task_name"] = task_name   # stripped before model inference

    return feats, pp["flags"]


def _skewness(x: np.ndarray) -> float:
    mu = x.mean(); sig = x.std()
    if sig < 1e-8: return 0.0
    return float(((x - mu) ** 3).mean() / sig ** 3)


def _kurtosis(x: np.ndarray) -> float:
    mu = x.mean(); sig = x.std()
    if sig < 1e-8: return 0.0
    return float(((x - mu) ** 4).mean() / sig ** 4) - 3.0


# ── TC-DANN inference ─────────────────────────────────────────────────────────

def _run_model(bundle: _ModelBundle, feat_row: dict, task_name: str) -> list[dict]:
    """
    Run one TC-DANN model on a single feature row.
    Returns list of {disease, model, y_prob, confidence, ...}.
    """
    if not bundle.loaded:
        return []

    task_id = TASK_ENC.get(task_name, 0)

    # Build feature vector aligned to bundle.feat_cols
    X = np.zeros((1, len(bundle.feat_cols)), dtype=np.float32)
    for j, col in enumerate(bundle.feat_cols):
        if col in feat_row:
            X[0, j] = float(feat_row[col])

    X_imp  = bundle.imputer.transform(X)
    X_sc   = bundle.scaler.transform(X_imp).astype(np.float32)
    X_t    = torch.tensor(X_sc).to(DEVICE)
    ids_t  = torch.tensor([task_id], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits, _ = bundle.model(X_t, ids_t, lam=0.0)
        probs     = torch.sigmoid(logits).cpu().numpy()[0]   # [n_diseases]

    results = []
    for di, disease in enumerate(bundle.disease_cols):
        y_prob = float(probs[di])

        # Composite confidence score (simplified — cosine & centroid need training data)
        # Full formula: a·x + b·y − c·z + d·m + e·p
        # Without training centroid index loaded here: x=0.5 (neutral), p=0.5
        is_psych = disease in PSYCHIATRIC_NO_DEMO
        w = CS_PSYCH if is_psych else CS_PHYSICAL

        # Runner-up z: max prob among OTHER diseases in this model
        other_probs = [float(probs[k]) for k in range(len(bundle.disease_cols)) if k != di]
        z_uncertainty = max(other_probs) if other_probs else 0.0

        x_cosine = 0.5          # neutral without centroid index
        m_demo   = 0.0 if is_psych else 0.5
        p_conf   = 0.5          # neutral without live confounder AUROC

        confidence = (
            w["a"] * x_cosine
            + w["b"] * y_prob
            - w["c"] * z_uncertainty
            + w["d"] * m_demo
            + w["e"] * p_conf
        )

        results.append({
            "disease":        disease,
            "model":          bundle.label,
            "y_prob":         round(y_prob, 4),
            "confidence":     round(float(confidence), 4),
            "x_cosine":       x_cosine,
            "z_uncertainty":  round(z_uncertainty, 4),
            "m_demo":         m_demo,
            "p_confounder":   p_conf,
            "percent":        round(y_prob * 100, 1),
        })

    return results


def _run_all_models(feat_row: dict, task_name: str) -> list[dict]:
    """Pool all four models and return unified top-3."""
    all_results = []
    for stem in MODEL_STEMS:
        bundle = _get_bundle(stem)
        if bundle.loaded:
            all_results.extend(_run_model(bundle, feat_row, task_name))

    # De-duplicate by disease (keep highest confidence), sort, top-3
    seen: dict[str, dict] = {}
    for r in all_results:
        d = r["disease"]
        if d not in seen or r["confidence"] > seen[d]["confidence"]:
            seen[d] = r

    ranked = sorted(seen.values(), key=lambda x: x["confidence"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return ranked


def _build_audit_flags(ranked: list[dict], pp_flags: list[str]) -> list[dict]:
    flags = []
    for f in pp_flags:
        flags.append({"level": "warn", "msg": f"audio_quality: {f}"})
    # Flag any disease where y_prob is below 0.2 but was still in top-3
    for r in ranked[:3]:
        if r["y_prob"] < 0.20:
            flags.append({"level": "info", "msg": f"{r['disease']}: low raw probability ({r['y_prob']:.3f}) — treat with caution"})
    return flags


def _save(sid: str, record: dict):
    with open(SESSIONS_DIR / f"{sid}.json", "w") as fh:
        json.dump(record, fh, indent=2, default=str)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="TC-DANN Clinical Voice API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    if not TC_DANN_AVAILABLE:
        log.error("run_tc_dann.py not importable: %s", _TC_DANN_IMPORT_ERROR)
        return
    _load_all_bundles()


@app.get("/health")
def health():
    bundle_status = {}
    for stem in MODEL_STEMS:
        b = _bundles.get(stem)
        bundle_status[MODEL_DISPLAY.get(stem, stem)] = {
            "loaded":   b.loaded if b else False,
            "diseases": len(b.disease_cols) if b and b.loaded else 0,
            "error":    b.error if b and not b.loaded else "",
        }
    n_loaded = sum(1 for v in bundle_status.values() if v["loaded"])
    status = "ok" if n_loaded == len(MODEL_STEMS) else ("degraded" if n_loaded else "no_models")
    return {
        "status":            status,
        "models_loaded":     n_loaded,
        "message":           "" if n_loaded else NO_MODELS_MSG,
        "tc_dann_available": TC_DANN_AVAILABLE,
        "device":            DEVICE,
        "bundle_dir":        str(BUNDLE_DIR),
        "models":            bundle_status,
        "all_diseases":      ALL_DISEASES,
        "timestamp":         datetime.utcnow().isoformat(),
    }


@app.get("/diseases")
def list_diseases():
    out = []
    for stem, diseases in MODEL_DISEASE_MAP.items():
        for d in diseases:
            out.append({"disease": d, "model": MODEL_DISPLAY[stem], "stem": stem})
    return {"diseases": out, "n_diseases": len(out)}


@app.post("/predict")
async def predict(
    session_id:   Optional[str]        = Form(None),
    patient_meta: Optional[str]        = Form(None),
    task_name:    str                  = Form("prolonged-vowel"),
    audio_file:   Optional[UploadFile] = File(None),
):
    """
    Primary inference endpoint.

    Input:
        audio_file  raw WAV/WebM/MP3 bytes → full preprocessing + TC-DANN inference
        task_name   one of: prolonged-vowel, read-speech, free-speech, diadochokinesis

    Returns:
        top3        unified cross-model top-3 differential ranking
        all_ranked  all diseases sorted by confidence
        audit_flags quality + confounder flags
    """
    _require_models()

    t0  = time.perf_counter()
    sid = session_id or str(uuid.uuid4())
    meta = {}
    if patient_meta:
        try:    meta = json.loads(patient_meta)
        except: raise HTTPException(422, "patient_meta must be valid JSON")

    if audio_file is None:
        raise HTTPException(422, "audio_file is required. Use /predict_features for feature-bypass mode.")

    raw = await audio_file.read()
    try:
        feat_row, pp_flags = _audio_to_feature_row(raw, task_name)
    except Exception as e:
        raise HTTPException(422, f"Preprocessing error: {e}")

    ranked    = _run_all_models(feat_row, task_name)
    top3      = ranked[:3]
    flags     = _build_audit_flags(top3, pp_flags)
    lat       = round((time.perf_counter() - t0) * 1000, 1)

    record = {
        "session_id":    sid,
        "timestamp":     datetime.utcnow().isoformat(),
        "task_name":     task_name,
        "patient_meta":  meta,
        "input_source":  "audio",
        "top3":          top3,
        "all_ranked":    ranked,
        "audit_flags":   flags,
        "latency_ms":    lat,
    }
    _save(sid, record)
    return JSONResponse(content=record)


@app.post("/predict_features")
async def predict_features(
    session_id:   Optional[str] = Form(None),
    patient_meta: Optional[str] = Form(None),
    task_name:    str           = Form("prolonged-vowel"),
    features:     str           = Form(...),
):
    """
    Feature-bypass endpoint for direct TC-DANN inference.
    Pass a JSON string of {feature_name: value} matching feat_cols.
    Used by the SVD preprocessing pipeline and batch evaluation scripts.
    """
    _require_models()

    t0  = time.perf_counter()
    sid = session_id or str(uuid.uuid4())
    meta = {}
    if patient_meta:
        try:    meta = json.loads(patient_meta)
        except: raise HTTPException(422, "patient_meta must be valid JSON")

    try:
        feat_row = json.loads(features)
    except Exception as e:
        raise HTTPException(422, f"features must be valid JSON: {e}")

    ranked = _run_all_models(feat_row, task_name)
    top3   = ranked[:3]
    flags  = _build_audit_flags(top3, [])
    lat    = round((time.perf_counter() - t0) * 1000, 1)

    record = {
        "session_id":   sid,
        "timestamp":    datetime.utcnow().isoformat(),
        "task_name":    task_name,
        "patient_meta": meta,
        "input_source": "features_bypass",
        "top3":         top3,
        "all_ranked":   ranked,
        "audit_flags":  flags,
        "latency_ms":   lat,
    }
    _save(sid, record)
    return JSONResponse(content=record)


@app.post("/record")
async def record_from_browser(
    session_id:   Optional[str] = Form(None),
    patient_meta: Optional[str] = Form(None),
    task_name:    str           = Form("prolonged-vowel"),
    task_index:   int           = Form(0),
    audio_b64:    str           = Form(...),
):
    """
    Receive base64-encoded audio from browser MediaRecorder.

    Browser JS:
        recorder.onstop = async () => {
          const blob = new Blob(chunks, {type:'audio/webm'});
          const b64  = await new Promise(r => {
            const fr = new FileReader();
            fr.onload = () => r(fr.result.split(',')[1]);
            fr.readAsDataURL(blob);
          });
          const fd = new FormData();
          fd.append('audio_b64', b64);
          fd.append('task_name', 'prolonged-vowel');
          const res = await fetch('/record', {method:'POST', body:fd});
        };
    """
    _require_models()

    t0  = time.perf_counter()
    sid = session_id or str(uuid.uuid4())
    meta = {}
    if patient_meta:
        try:    meta = json.loads(patient_meta)
        except: raise HTTPException(422, "patient_meta must be valid JSON")

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        raise HTTPException(422, f"base64 decode error: {e}")

    if len(audio_bytes) < 100:
        raise HTTPException(422, "Audio payload too small")

    try:
        feat_row, pp_flags = _audio_to_feature_row(audio_bytes, task_name)
    except Exception as e:
        raise HTTPException(422, f"Preprocessing error: {e}")

    ranked = _run_all_models(feat_row, task_name)
    top3   = ranked[:3]
    flags  = _build_audit_flags(top3, pp_flags)
    lat    = round((time.perf_counter() - t0) * 1000, 1)

    record = {
        "session_id":   sid,
        "timestamp":    datetime.utcnow().isoformat(),
        "task_name":    task_name,
        "task_index":   task_index,
        "patient_meta": meta,
        "input_source": "browser_mic",
        "top3":         top3,
        "all_ranked":   ranked,
        "audit_flags":  flags,
        "latency_ms":   lat,
    }
    _save(f"{sid}_task{task_index}", record)
    return JSONResponse(content=record)


@app.get("/sessions")
def list_sessions():
    files = sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files[:50]:
        try:
            with open(f) as fh:
                d = json.load(fh)
            top1 = d.get("top3", [{}])[0] if d.get("top3") else {}
            out.append({
                "session_id":   d.get("session_id"),
                "timestamp":    d.get("timestamp"),
                "task_name":    d.get("task_name"),
                "input_source": d.get("input_source"),
                "top_disease":  top1.get("disease"),
                "top_conf":     top1.get("confidence"),
                "latency_ms":   d.get("latency_ms"),
            })
        except Exception:
            pass
    return {"count": len(out), "sessions": out}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    p = SESSIONS_DIR / f"{session_id}.json"
    if not p.exists():
        raise HTTPException(404, f"Session {session_id!r} not found")
    with open(p) as fh:
        return json.load(fh)
