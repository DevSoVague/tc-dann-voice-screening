"""
serve.py

FastAPI inference server for TC-DANN.

Loads:
  - All best ensemble members (filtered by C1 in calibrate_and_save.py)
  - GroupTemperatureScaler (scaler.pt)
  - SplitConformalPredictor (conformal.joblib)
  - Support counts (support_counts.joblib)
  - Confounder LR models (confounder_models.joblib)
  - Metadata (meta.joblib)

Exposes:
  POST /predict   <- main inference endpoint
  GET  /health    <- liveness check
  GET  /diseases  <- list of disease outputs
  GET  /metrics   <- abstain rate tracker

Usage:
    pip install fastapi uvicorn
    python serve.py --ckpt_dir checkpoints/ --artifact_dir artifacts/

    # or with gunicorn for production:
    gunicorn serve:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import TCDANN
from calibration import (
    GroupTemperatureScaler,
    SplitConformalPredictor,
    ensemble_mean_std,
    make_group_id,
)
from confidence import AbstainConfig, AbstainPredictor


# ------------------------------------------------------------------
# Global state (loaded once at startup)
# ------------------------------------------------------------------

_models: List[TCDANN] = []
_scaler: Optional[GroupTemperatureScaler] = None
_conformal: Optional[SplitConformalPredictor] = None
_abstainer: Optional[AbstainPredictor] = None
_confounder_models: Dict = {}
_meta: Dict = {}
_device: str = "cpu"

# Running abstain stats
_abstain_counts = defaultdict(int)
_total_requests = 0


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    # All arrays are flattened lists; the server reshapes them.
    # A single recording per request.

    spec: List[float] = Field(..., description="Flattened spectrogram [C*H*W]")
    spec_shape: List[int] = Field(..., description="[C, H, W]")

    mel: List[float] = Field(..., description="Flattened mel spectrogram [C*H*W]")
    mel_shape: List[int] = Field(..., description="[C, H, W]")

    ema: List[float] = Field(..., description="Flattened EMA features [C*T]")
    ema_shape: List[int] = Field(..., description="[C, T]")
    ema_mask: Optional[List[int]] = Field(None, description="[T] binary mask, 1=valid")

    static: List[float] = Field(..., description="Static feature vector")

    task_id: int = Field(..., description="Integer task ID from task_vocab")

    # Demographic context for calibration + conformal + abstain checks
    site: int = Field(..., description="Site code: 0=USA, 1=Canada, 2=Other")
    age_bucket: int = Field(..., description="Age bucket 0–3 (from dataset.age_to_bucket)")
    sex: int = Field(..., description="Sex code: 0=F, 1=M")

    # Optional: demographics vector for confounder baseline (abstain A2 check)
    demo_features: Optional[List[float]] = Field(
        None, description="Demographics feature vector for confounder LR baseline"
    )


class DiseasePrediction(BaseModel):
    disease: str
    probability: float
    abstain: bool
    abstain_reason: str
    ensemble_std: float


class PredictResponse(BaseModel):
    predictions: List[DiseasePrediction]
    latency_ms: float
    n_ensemble_members: int


# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------

def load_artifacts(ckpt_dir: str, artifact_dir: str):
    global _models, _scaler, _conformal, _abstainer
    global _confounder_models, _meta, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(ckpt_dir)
    art_path = Path(artifact_dir)

    # Load metadata
    _meta = joblib.load(art_path / "meta.joblib")
    disease_cols = _meta["disease_cols"]
    task_vocab   = _meta["task_vocab"]
    n_static     = _meta["n_static"]
    n_groups     = _meta["n_groups"]
    best_ids     = _meta["best_member_ids"]
    D = len(disease_cols)

    print(f"[serve] Loading {len(best_ids)} best ensemble members: {best_ids}")

    # Load only the best ensemble members
    all_ckpts = sorted(ckpt_path.glob("model_*.pt"))
    for i in best_ids:
        if i >= len(all_ckpts):
            print(f"  WARNING: best_id {i} not found in checkpoints, skipping")
            continue
        data = torch.load(all_ckpts[i], map_location=_device, weights_only=False)
        model = TCDANN(
            n_diseases=D,
            n_tasks=len(task_vocab),
            n_sites=_meta["n_sites"],
            n_age_buckets=_meta["n_ages"],
            n_sex=2,
            n_static=n_static,
        ).to(_device)
        model.load_state_dict(data["state_dict"])
        model.eval()
        _models.append(model)
        print(f"  loaded member {i}")

    if not _models:
        raise RuntimeError("No ensemble members loaded. Check ckpt_dir and meta.joblib.")

    # Load GroupTemperatureScaler
    _scaler = GroupTemperatureScaler(n_diseases=D, n_groups=n_groups)
    _scaler.load_state_dict(torch.load(art_path / "scaler.pt", map_location=_device))
    _scaler.eval()
    print(f"[serve] Loaded GroupTemperatureScaler")

    # Load SplitConformalPredictor
    _conformal = joblib.load(art_path / "conformal.joblib")
    print(f"[serve] Loaded SplitConformalPredictor")

    # Load support counts and build AbstainPredictor
    support_counts = joblib.load(art_path / "support_counts.joblib")
    _abstainer = AbstainPredictor(
        config=AbstainConfig(),
        support_counts=support_counts,
    )
    print(f"[serve] Loaded AbstainPredictor")

    # Load confounder LR models
    _confounder_models = joblib.load(art_path / "confounder_models.joblib")
    print(f"[serve] Loaded confounder models for {len(_confounder_models)} diseases")

    print(f"[serve] Ready. Diseases: {disease_cols}")


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------

app = FastAPI(
    title="TC-DANN Inference API",
    description="Task-Conditioned Domain-Adversarial Network for Bridge2AI-Voice",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "n_models": len(_models),
        "diseases": _meta.get("disease_cols", []),
    }


@app.get("/diseases")
def diseases():
    return {"diseases": _meta.get("disease_cols", [])}


@app.get("/metrics")
def metrics():
    total = max(_total_requests, 1)
    return {
        "total_requests": _total_requests,
        "abstain_rates": {
            d: _abstain_counts[d] / total
            for d in _meta.get("disease_cols", [])
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global _total_requests
    t0 = time.perf_counter()

    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    disease_cols = _meta["disease_cols"]
    D = len(disease_cols)

    # ----------------------------------------------------------------
    # Build batch dict (batch size = 1)
    # ----------------------------------------------------------------
    def _t(x, shape=None, dtype=torch.float32):
        arr = torch.tensor(x, dtype=dtype)
        if shape:
            arr = arr.reshape(shape)
        return arr.unsqueeze(0).to(_device)   # add batch dim

    batch = {
        "spec":    _t(req.spec, req.spec_shape),
        "mel":     _t(req.mel, req.mel_shape),
        "ema":     _t(req.ema).reshape(1, *req.ema_shape),   # [1, C, T]
        "static":  _t(req.static),
        "task_id": torch.tensor([req.task_id], dtype=torch.long).to(_device),
        "site":    torch.tensor([req.site], dtype=torch.long).to(_device),
        "age":     torch.tensor([req.age_bucket], dtype=torch.long).to(_device),
        "sex":     torch.tensor([req.sex], dtype=torch.long).to(_device),
    }
    if req.ema_mask is not None:
        batch["ema_mask"] = _t(req.ema_mask, dtype=torch.bool).squeeze(-1)

    # ----------------------------------------------------------------
    # Ensemble forward passes
    # ----------------------------------------------------------------
    all_logits = []
    with torch.no_grad():
        for model in _models:
            out = model(batch)
            all_logits.append(out["disease_logits"].cpu().numpy())   # [1, D]

    p_mean, p_std = ensemble_mean_std(all_logits)   # [1, D] each

    # ----------------------------------------------------------------
    # Temperature scaling
    # ----------------------------------------------------------------
    group_id = make_group_id(
        np.array([req.site]),
        np.array([req.age_bucket]),
        n_sites=_meta["n_sites"],
        n_ages=_meta["n_ages"],
    ).astype(int)

    mean_logits = np.stack(all_logits).mean(axis=0)  # [1, D]
    p_cal = _scaler.transform(
        torch.tensor(mean_logits),
        torch.tensor(group_id).long(),
    ).numpy()   # [1, D]

    # ----------------------------------------------------------------
    # Conformal non-conformity score
    # ----------------------------------------------------------------
    conformal_score = _conformal.score(p_cal, group_id)   # [1, D]

    # ----------------------------------------------------------------
    # Confounder baseline probability (abstain check A2)
    # ----------------------------------------------------------------
    if req.demo_features is not None:
        demo = np.array(req.demo_features, dtype=np.float32).reshape(1, -1)
        p_confounder = np.zeros((1, D))
        for j, d in enumerate(disease_cols):
            if d in _confounder_models:
                p_confounder[0, j] = _confounder_models[d].predict_proba(demo)[0, 1]
    else:
        # If no demo features provided, use 0.5 (neutral) -> A2 won't fire
        p_confounder = np.full((1, D), 0.5)

    # ----------------------------------------------------------------
    # Abstain checks
    # ----------------------------------------------------------------
    result = _abstainer.predict(
        p_mean=p_cal,
        p_std=p_std,
        p_confounder=p_confounder,
        conformal_score=conformal_score,
        group_ids=group_id,
    )

    # ----------------------------------------------------------------
    # Build response
    # ----------------------------------------------------------------
    _total_requests += 1
    predictions = []
    for j, d in enumerate(disease_cols):
        abstain_flag = bool(result["abstain"][0, j])
        if abstain_flag:
            _abstain_counts[d] += 1
        predictions.append(DiseasePrediction(
            disease=d,
            probability=float(p_cal[0, j]),
            abstain=abstain_flag,
            abstain_reason=str(result["reason"][0, j]),
            ensemble_std=float(p_std[0, j]),
        ))

    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        predictions=predictions,
        latency_ms=round(latency_ms, 2),
        n_ensemble_members=len(_models),
    )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--artifact_dir", default="artifacts")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_artifacts(args.ckpt_dir, args.artifact_dir)
    uvicorn.run(app, host=args.host, port=args.port)
