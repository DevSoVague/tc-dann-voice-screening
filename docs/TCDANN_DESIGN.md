# Bridge2AI-Voice · TC-DANN v6.0.0

**Quad-model voice pathology classifier with auditable, calibrated confidence scoring.**

---

## What this is

A research pipeline for classifying voice-based disease signals across 20 diagnoses from the Bridge2AI-Voice dataset. It trains four independent neural networks, one per clinical domain, and produces per-patient top-3 predictions ranked by a composite confidence score that is transparent enough to explain to a clinician.

---

## Why four models instead of one

The most important design decision in this codebase is that psychiatric and physical diseases are never trained together.

When a single model tries to predict laryngeal dystonia and depression from the same encoder, something subtle and damaging happens: the spectral features (MFCC, Mel filterbanks, spectrograms) that are highly informative for voice pathology act as noise for psychiatric heads. The shared encoder learns to partially suppress them to reduce psychiatric loss, which simultaneously hurts voice pathology prediction. The model is being pulled in two incompatible directions by every gradient update.

The evidence for this came from ablation: when spectrogram and Mel filterbank modalities were zeroed at inference time, psychiatric AUROC went *up*. A modality improving predictions when removed is a textbook sign of confounding in a shared representation, the model had learned to partially suppress a signal it needed for one task because that signal was hurting another.

Four separate models eliminate this entirely. Each encoder learns exactly the representation its diseases need, with no gradient interference from unrelated clinical domains:

| Model | Diseases | Why together |
|---|---|---|
| **Voice + Oncological** | Laryngeal dystonia, vocal fold paralysis, benign lesions, muscle tension dysphonia, glottic insufficiency, laryngeal cancer, precancerous lesions, control | All manifest directly in laryngeal/vocal tract acoustics. Oncological lesions share acoustic features with structural voice disorders. |
| **Neurological** | Parkinson's disease, cognitive impairment, ALS | All involve motor control degradation that expresses in timing, rhythm, and articulatory precision, patterns that cross-contaminate poorly with resonance-based voice disorders. |
| **Respiratory** | Airway stenosis, COPD/asthma, unexplained chronic cough, laryngitis | Shared mechanism: airflow restriction and turbulence. The acoustic signature is breathiness, reduced loudness, and irregular periodicity, distinct from laryngeal structural pathology. |
| **Psychiatric** | ADHD, PTSD, anxiety, depression, bipolar disorder | No structural acoustic signature. Signal comes from prosody, speech rate, pause patterns, and subtle spectral envelope changes, requires a different representational emphasis. |

`psychiatric_history` was removed entirely. It has no coherent acoustic phenotype and acts as a catch-all label that overlaps with every other psychiatric disease, contaminating the shared representation with conflicting gradients.

---

## Why MFCC and spectrogram features are kept for the psychiatric model

An earlier version of this pipeline excluded MFCC, Mel filterbanks, and spectrograms from the psychiatric model. This was a mistake in attribution.

The improvement seen when those features were zeroed was not because they are uninformative for psychiatric diseases, it was because they were causing interference *in a unified model* that needed them suppressed for psychiatric heads. Once the models were separated, the interference disappeared. The psychiatric model can now learn from spectral features freely, and its AUROC improves as a result.

Only PPG (photoplethysmography) is excluded across all models. PPG contributes zero predictive signal in every domain tested. Including it adds noise to the cosine similarity index and wastes representational capacity.

---

## The confidence score

Most classifiers output a single probability. This is insufficient for clinical use because neural networks are overconfident, and a high probability on an unfamiliar patient is not the same as a high probability on a patient who looks like the training data.

This pipeline computes:

```
confidence = a·x + b·y − c·z + d·m + e·p
```

Each term has a specific clinical meaning:

**x, Subgroup cosine similarity** (weight: 0.25 physical, 0.15 psychiatric)
The input sample is compared to the top-10 nearest training examples from the predicted disease subgroup. High x means the model is in familiar territory. Low x means it is extrapolating. This is distributional shift detection built directly into the output. A model that achieves 0.93 AUROC on familiar patients but is presented with an outlier will now report lower confidence rather than the same number.

Sparse subgroups (fewer than 10 training examples) receive a neutral x of 0.5 rather than a near-zero score driven by noise. Small oncological subgroups like precancerous lesions were falsely penalised by the old approach.

**y, Head probability** (weight: 0.40 physical, 0.50 psychiatric)
The sigmoid output of the disease head. Standard neural network confidence. Given more weight in the psychiatric model because the narrower feature set (no PPG) makes cosine similarity a less reliable signal.

**z, Diagnostic uncertainty penalty** (weight: 0.15 both)
For each predicted disease, the maximum cosine similarity of the top-2 runner-up diseases *within the same model* is computed. If the model nearly predicted something else that also looks plausible in feature space, confidence is penalised. A strong prediction that has no close alternatives is worth more than the same probability with a plausible differential diagnosis lurking nearby. The runner-up comparison is deliberately scoped within each model, a Voice+Oncological runner-up cannot penalise a Psychiatric prediction.

**m, Demographic prior alignment** (weight: 0.10 physical, 0.00 psychiatric)
Cosine similarity between the patient's demographic vector (age, sex, BMI) and the disease subgroup's demographic centroid in training data. A 70-year-old male matching the Parkinson's disease subgroup demographics increases confidence. This weight is hard-zeroed for all psychiatric diseases because the audit confirmed demographics are shortcut features for psychiatric labels, they would inflate confidence for spurious reasons.

**p, Confounder robustness** (weight: 0.10 physical, 0.20 psychiatric)
`p = 1 − task_confounder_auroc`. The confounder AUROC measures how well task type alone (prolonged vowel, read speech, etc.) can predict the disease label. A disease with a high confounder AUROC has an AUROC that is partly explained by which recording protocol was used, not by genuine pathology signal. This term deflates the final confidence score for diseases where we know part of the AUROC is spurious. Given higher weight in the psychiatric model because psychiatric AUROC scores are more confounded by task protocol.

---

## The unified top-3

Each model produces its own top-3 predictions for every patient. These are then pooled and re-ranked globally by confidence score to produce a single unified top-3 that can span models. A patient might have Parkinson's disease as their top prediction (from the Neurological model) and depression as their second (from the Psychiatric model). The unified output reflects this correctly rather than forcing a choice within one domain.

---

## What the audit table tells you

The audit table produced after every run contains, for each disease:

- **tc_dann_auroc**, held-out test AUROC. The primary performance metric.
- **task_confounder_auroc**, AUROC achievable by task type alone. If this is close to tc_dann_auroc, the model has learned recording protocol bias, not pathology.
- **margin**, difference between the two. Must exceed 0.05 to pass audit.
- **worst_task_auroc**, minimum AUROC across task subgroups. NaN when fewer than 2 task subgroups have enough samples to evaluate, rather than reporting a misleading 1.0.
- **n_train_subgroup**, training examples for this disease. Low values explain low cosine similarity scores and NaN worst-task values. They are a sparsity signal, not a model failure.
- **mean_confidence_top1**, average composite confidence when this disease is the top prediction. Calibrates intuition for what a "good" confidence score looks like per disease.

A disease that passes the AUROC audit but shows low mean_confidence_top1 warrants investigation: it may have a small training subgroup, high confounder vulnerability, or a feature distribution that does not cluster tightly enough for the cosine index.

---

## Leakage protections

Every statistical analysis in this pipeline was designed around the principle that the test set must be invisible during training.

Participants are split before any fitting. All recordings from a given participant land exclusively in train or test, never both. Row-level splitting would leak correlated acoustic features across sets because the same person's voice is highly consistent across recordings.

The imputer and scaler are fitted only on training rows. Test rows call `.transform()` only. This is enforced structurally: the fitted objects live on the `VoiceDataset` instance created from training data and are passed explicitly to all evaluation functions.

The confounder baseline (logistic regression on task type) runs on the test split only. In the original codebase it ran on the full dataset, making the confounder AUROC an optimistic estimate that inflated the apparent margin.

---

## Outputs

```
tc_dann_results/
  audit_table.csv              All diseases, all models, one row per disease
  audit_voice_onco.csv         Per-model audit split
  audit_neurological.csv
  audit_respiratory.csv
  audit_psychiatric.csv
  top3_unified.csv             Cross-model top-3 per patient (primary clinical output)
  top3_all_models.csv          Raw per-model top-3 before unified re-ranking
  top3_voice_onco.csv          Per-model top-3 splits
  top3_neurological.csv
  top3_respiratory.csv
  top3_psychiatric.csv
  voice_onco_model_checkpoint.pt        PyTorch state dict
  neurological_model_checkpoint.pt
  respiratory_model_checkpoint.pt
  psychiatric_model_checkpoint.pt
  voice_onco_model_best.joblib          Full inference bundle including
  neurological_model_best.joblib        centroid index + demographic index
  respiratory_model_best.joblib         for confidence scoring at inference
  psychiatric_model_best.joblib
```

The `.joblib` bundles contain everything needed for inference without reloading training data: model weights, fitted imputer, fitted scaler, subgroup centroid index, and demographic centroid index. Load one file, run predictions, get calibrated confidence scores.

---

## Running

```bash
# Full run
python run_tc_dann.py --data_root /path/to/my_model --epochs 40

# Quick smoke test (2000 recordings)
python run_tc_dann.py --data_root /path/to/my_model --epochs 5 --smoke_test

# Custom confidence weights
python run_tc_dann.py --data_root /path/to/my_model \
  --cs_a 0.30 --cs_b 0.35 --cs_e 0.15 \
  --cs_pa 0.10 --cs_pb 0.55 --cs_pe 0.25

# Skip cache (force feature reload)
python run_tc_dann.py --data_root /path/to/my_model --no_cache
```

Confidence weight flags: `--cs_a/b/c/d/e` for physical models, `--cs_pa/pb/pc/pd/pe` for the psychiatric model.
