# models/

Default location the API server and CLI predictor read trained bundles from
(override with `TC_DANN_BUNDLE_DIR`). This folder is empty in the repository.

Trained weights are not distributed: they were trained on Bridge2AI-Voice v3.0.0
under the PhysioNet data use agreement, and each bundle also stores
training-set embeddings used for the subgroup-similarity term of the confidence
score. Credentialed PhysioNet users can produce them with:

```bash
python tcdann/run_tc_dann.py --data_root "$B2AI_DATA_ROOT" --epochs 40 --out_dir models
```

Expected files (exact names):

| File | Used by |
|---|---|
| `voice_onco_model_best.joblib` | API, CLI, Streamlit (Voice+Onco model) |
| `neurological_model_best.joblib` | API, CLI, Streamlit (Neurological model) |
| `respiratory_model_best.joblib` | API, CLI, Streamlit (Respiratory model) |
| `psychiatric_model_best.joblib` | API, CLI, Streamlit (Psychiatric model) |
| `*_model_checkpoint.pt` | optional; raw PyTorch state dicts written alongside, not needed for serving |

Any subset loads: the API serves the models it finds and reports the rest as
missing on `/health`. With none present the API still starts, `/health` returns
`"status": "no_models"`, and prediction endpoints return HTTP 503 with this
explanation.
