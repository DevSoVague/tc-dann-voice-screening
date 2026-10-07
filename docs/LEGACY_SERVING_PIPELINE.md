# TC-DANN: Training → Calibration → Serving Pipeline

Everything needed to go from raw checkpoints to a running inference API.

---

## New Files Added

| File | Role |
|---|---|
| `calibrate_and_save.py` | Post-training script. Fits calibration, saves all joblibs. |
| `serve.py` | FastAPI inference server. Loads artifacts and serves predictions. |

---

## What Gets Saved as Joblib (and Why)

| File | Type | Why joblib |
|---|---|---|
| `artifacts/conformal.joblib` | `SplitConformalPredictor` | Pure numpy quantile array, no PyTorch needed at inference |
| `artifacts/support_counts.joblib` | `np.ndarray [n_groups, n_diseases]` | Plain int array, tiny |
| `artifacts/confounder_models.joblib` | `Dict[str, LogisticRegression]` | sklearn objects, joblib is the standard |
| `artifacts/meta.joblib` | `dict` | disease_cols, task_vocab, best_member_ids, n_groups |

| File | Type | Why NOT joblib |
|---|---|---|
| `artifacts/scaler.pt` | `GroupTemperatureScaler` | It's a `nn.Module` with `nn.Parameter`, save as PyTorch state dict |
| `checkpoints/model_*.pt` | `TCDANN` | Same reason |

---

## What "Best Model" Means Here

The ensemble has 5 members (seeds 0-4). "Best" is defined by **C1 confounder separation**:
a member is kept only if the majority of its diseases have acoustic AUROC exceeding the
demographics-only LR baseline by ≥ 0.10. This is the same criterion from the audit.

`calibrate_and_save.py` runs this filter automatically and saves `best_member_ids` to
`meta.joblib`. `serve.py` reads that list and only loads those members at startup.
If no member passes (unlikely after 40 epochs), all 5 are used with a printed warning.

---

## Step-by-Step: After Training Completes

### Step 1: Install extra dependencies

```bash
pip install joblib fastapi uvicorn
```

### Step 2: Run calibration (fit and save all artifacts)

```bash
python calibrate_and_save.py \
    --root_dir /data/my_model \
    --ckpt_dir checkpoints/ \
    --artifact_dir artifacts/
```

Expected output:
```
[calibrate] Loading ensemble...
[calibrate] Selecting best ensemble members...
  member 0: 7/9 diseases pass C1  gaps={parkinsons: 0.14, ...}
  member 1: 8/9 diseases pass C1
  ...
Best members (pass C1): [0, 1, 3, 4]
[calibrate] Fitting GroupTemperatureScaler...
  saved scaler.pt  temps range: 0.812 - 1.334
[calibrate] Fitting SplitConformalPredictor...
  saved conformal.joblib  fitted cells: 89/216
[calibrate] Computing support counts...
  saved support_counts.joblib  shape: (12, 9)
[calibrate] Done. Artifacts written to artifacts/
```

### Step 3: Start the inference server

```bash
python serve.py \
    --ckpt_dir checkpoints/ \
    --artifact_dir artifacts/ \
    --host 0.0.0.0 \
    --port 8000
```

### Step 4: Call the API

```bash
# Health check
curl http://localhost:8000/health

# List diseases
curl http://localhost:8000/diseases

# Predict (replace the arrays with real features)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "spec": [0.1, 0.2, ...],
    "spec_shape": [1, 128, 256],
    "mel": [0.1, 0.2, ...],
    "mel_shape": [1, 128, 256],
    "ema": [0.1, 0.2, ...],
    "ema_shape": [12, 100],
    "static": [0.1, 0.2, ...],
    "task_id": 3,
    "site": 0,
    "age_bucket": 2,
    "sex": 1,
    "demo_features": [45.0, 1, 0, 1, 0, 0, ...]
  }'
```

### Example response

```json
{
  "predictions": [
    {
      "disease": "parkinsons",
      "probability": 0.73,
      "abstain": false,
      "abstain_reason": "",
      "ensemble_std": 0.04
    },
    {
      "disease": "depression",
      "probability": 0.51,
      "abstain": true,
      "abstain_reason": "insufficient_subgroup_support",
      "ensemble_std": 0.18
    }
  ],
  "latency_ms": 42.3,
  "n_ensemble_members": 4
}
```

---

## Abstain Reasons

| Reason | Means |
|---|---|
| `ensemble_disagreement` | Ensemble std > 0.15, models disagree |
| `agrees_with_confounder_baseline` | Acoustic model not adding signal over demographics |
| `insufficient_subgroup_support` | < 30 training positives in this patient's subgroup cell |
| `out_of_distribution_for_subgroup` | Conformal score too high, patient is unusual for their subgroup |
| *(empty string)* | Prediction is actionable |

---

## Full Order of Operations

```
AFTER GCP TRAINING
  1. Download checkpoints/ from GCS to local
  2. pip install joblib fastapi uvicorn
  3. python calibrate_and_save.py --root_dir ... --ckpt_dir checkpoints/ --artifact_dir artifacts/
  4. python serve.py --ckpt_dir checkpoints/ --artifact_dir artifacts/
  5. curl http://localhost:8000/health

ON GCP (if serving in the cloud)
  1. Add calibrate_and_save.py + serve.py to your Docker image
  2. Add joblib fastapi uvicorn to requirements.txt
  3. Run calibrate as a Kubernetes Job (one-time, after training job completes)
  4. Run serve as a Kubernetes Deployment with 1 GPU node
```

---

## Add to requirements.txt

```
joblib>=1.3
fastapi>=0.110
uvicorn>=0.29
```
