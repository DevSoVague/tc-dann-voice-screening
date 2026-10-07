# TC-DANN: Confounder-Audited Voice Disease Screening

Task-conditioned, domain-adversarial neural networks that screen for 20 voice-linked diseases on Bridge2AI-Voice v3.0.0, built after an audit showed that demographics and recording protocol alone can "predict" disease.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

![Confounder vulnerability: AUROC from demographics alone, per disease](assets/vulnerability_heatmap.png)

*Confounder audit (aggregate, per disease): a logistic regression on demographics or the recording-task histogram alone, 5-fold participant-stratified, 833 participants.*

> Research code only. Not a medical device, not clinically validated. No data, features, or trained weights are distributed in this repository.

## What it does

- **Confounder audit first.** Shows that demographics alone reach 0.96 AUROC on cognitive impairment and 0.93 on Parkinson's, and that the recording-task histogram alone reaches 0.94 on Parkinson's (`audit/confounder_analysis.py`).
- **Four domain-separated models** (Voice+Oncological, Neurological, Respiratory, Psychiatric) with FiLM task conditioning on every encoder layer and a gradient-reversal site adversary that pushes the 128-d representation to be site-invariant (`tcdann/run_tc_dann.py`).
- **Audit as a gate.** Every disease head is reported against a task-only confounder baseline on the held-out split; in the latest run all 15 reported heads pass (acoustic AUROC above the confounder AUROC), with a mean held-out AUROC of 0.76.
- **Transparent confidence score** per prediction: `conf = a·x + b·y - c·z + d·m + e·p` (subgroup cosine similarity, head probability, runner-up penalty, demographic prior, confounder robustness), with the demographic term forced to 0 for psychiatric heads.
- **External check on the Saarbrücken Voice Database (SVD, German):** 6/6 evaluated diseases pass the confounder audit on held-out patients (ADHD 0.72 AUROC).
- **Serving:** FastAPI inference server (raw audio or features in, unified cross-model top-3 plus audit flags out) and Streamlit front ends, including a 5-stage screening app with a multi-agent clinical-reasoning layer (Claude, Gemini, Tavily, Milvus RAG).

## Architecture

```mermaid
flowchart LR
    A[Bridge2AI-Voice v3<br/>features + phenotype] --> B[Confounder audit<br/>demographics / task-only LR]
    A --> C[Participant-level split<br/>imputer + scaler fit on train]
    C --> D1[Voice+Onco model]
    C --> D2[Neurological model]
    C --> D3[Respiratory model]
    C --> D4[Psychiatric model]
    D1 & D2 & D3 & D4 --> E[Composite confidence<br/>a·x + b·y - c·z + d·m + e·p]
    B --> F[Audit table<br/>AUROC vs confounder AUROC]
    E --> G[Unified top-3 across models]
    G --> H[FastAPI server] --> I[Streamlit apps]
    F --> H
```

Each domain model takes per-modality inputs (MFCC, mel, spectrogram, pitch, loudness, periodicity, EMA, static features; PPG dropped after it showed no signal in ablation), passes them through an FC 512 to 256 to 128 encoder with FiLM conditioning on the recording task, and feeds a gradient-reversal layer into a site classifier so the encoder cannot rely on recording site. The four models were split apart after an ablation showed that physical-voice modalities hurt psychiatric heads inside a single shared model. Predictions from all four models are pooled and re-ranked by the confidence score. See [docs/TCDANN_DESIGN.md](docs/TCDANN_DESIGN.md) for the full design rationale.

## Quickstart

```bash
git clone https://github.com/DevSoVague/tc-dann-voice-screening.git
cd tc-dann-voice-screening
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit paths; export them in your shell
export B2AI_DATA_ROOT=/path/to/b2ai-voice/3.0.0
```

Run the confounder audit, then train and evaluate TC-DANN:

```bash
python audit/confounder_analysis.py                      # writes outputs/audit/
cd tcdann
python run_tc_dann.py --epochs 5 --smoke_test            # quick check
python run_tc_dann.py --epochs 40                        # full run, writes $B2AI_DATA_ROOT/tc_dann_results/
python explain_tc_dann.py --bundle $B2AI_DATA_ROOT/tc_dann_results/psychiatric_model_best.joblib --stem psychiatric
```

Serve the trained bundles:

```bash
cd tcdann
TC_DANN_BUNDLE_DIR=$B2AI_DATA_ROOT/tc_dann_results uvicorn tc_dann_api_server:app --port 8000
streamlit run tc_dann_app.py                             # TC-DANN front end
python tc_dann_predict.py recording.wav --task prolonged-vowel --bundle_dir $B2AI_DATA_ROOT/tc_dann_results
```

`voxclinbench_app.py` is the larger 5-stage screening app with the multi-agent chat and RAG pages. It needs `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `TAVILY_API_KEY` and a running Milvus (`MILVUS_URI`), read from the environment only. It was originally built against a separate model API; it calls `/health` and `/predict` at `VOXCLIN_API` and falls back to mock predictions when no API is reachable.

External validation on SVD:

```bash
python svd/svd_preprocess.py --svd_root /path/to/svd --out_root /path/to/svd_out
SVD_OUT_ROOT=/path/to/svd_out python svd/run_tc_dann_svd.py --epochs 40
```

## Data access

This repository contains **code only**. No recordings, features, phenotype files, splits, predictions, or data-derived weights are included, and none should ever be committed (see `.gitignore`).

1. **Bridge2AI-Voice v3.0.0** is a credentialed dataset on PhysioNet. Complete PhysioNet credentialing (including the required human-subjects training), then request access and sign the Bridge2AI-Voice Data Use Agreement on the dataset page at <https://physionet.org/>.
2. Download it to a local folder you control and point the code at it:
   ```bash
   export B2AI_DATA_ROOT=/path/to/b2ai-voice/3.0.0   # contains features/ and phenotype/
   ```
3. **Saarbrücken Voice Database (SVD)** is used for external validation only. Obtain it from its maintainers under their terms; it is not redistributable. Pass its location with `--svd_root`.

Trained bundles (`*_model_best.joblib`, `*.pt`) are derived from the protected data and are not distributed; train your own after obtaining access.

## Results

Held-out test split (participant-level 80/20), latest local run of `run_tc_dann.py`. `margin` = TC-DANN AUROC minus the AUROC of a task-only confounder baseline on the same split.

| Disease | Model | AUROC | Confounder AUROC | Margin |
|---|---|---|---|---|
| Precancerous lesions | Voice+Onco | 0.98 | 0.22 | 0.76 |
| PTSD | Psychiatric | 0.91 | 0.59 | 0.31 |
| Parkinson's disease | Neurological | 0.90 | 0.64 | 0.26 |
| Airway stenosis | Respiratory | 0.89 | 0.63 | 0.26 |
| ADHD | Psychiatric | 0.84 | 0.48 | 0.36 |
| Bipolar disorder | Psychiatric | 0.81 | 0.34 | 0.47 |
| Benign lesions | Voice+Onco | 0.76 | 0.51 | 0.25 |
| Control | Voice+Onco | 0.76 | 0.46 | 0.29 |
| Cognitive impairment | Neurological | 0.74 | 0.69 | 0.06 |
| Laryngeal dystonia | Voice+Onco | 0.70 | 0.55 | 0.16 |
| Vocal fold paralysis | Voice+Onco | 0.70 | 0.53 | 0.16 |
| Chronic cough | Respiratory | 0.68 | 0.49 | 0.18 |
| Muscle tension dysphonia | Voice+Onco | 0.67 | 0.58 | 0.10 |
| Depression | Psychiatric | 0.57 | 0.49 | 0.09 |
| Anxiety | Psychiatric | 0.55 | 0.47 | 0.07 |

Reading it honestly: psychiatric heads beyond PTSD/ADHD/bipolar sit close to their confounder baseline, and cognitive impairment only clears it by 0.06 because positives in v3 come from a single site. The published MARVEL model (Piao et al., 2025) was used as a comparator; its encoder aligns better with clinical acoustic features (SPARC) than TC-DANN's, so TC-DANN is framed as passing the confounder bar, not as beating MARVEL.

Demographic shortcut audit (`audit/confounder_analysis.py`, 833 participants, 5-fold participant-stratified):

| Disease | Demographics only | Task histogram only |
|---|---|---|
| Cognitive impairment | 0.96 | 0.93 |
| Parkinson's | 0.93 | 0.94 |
| Precancerous | 0.83 | 0.63 |

![Shortcut ablation by feature group](assets/ablation_barplot.png)

## Project structure

```
tc-dann-voice-screening/
├── tcdann/                 final quad-model TC-DANN + serving
│   ├── run_tc_dann.py        training, evaluation, audit table, joblib bundles
│   ├── tc_dann_predict.py    CLI / library inference
│   ├── explain_tc_dann.py    SHAP attributions per disease and task
│   ├── tc_dann_api_server.py FastAPI server
│   ├── tc_dann_app.py        Streamlit front end for TC-DANN
│   ├── voxclinbench_app.py   5-stage screening app + multi-agent chat + RAG pages
│   ├── indexer.py            Milvus PDF indexer (LangGraph)
│   ├── audio_preprocessing.py, confounder_baseline.py, extract_pure_labels.py
├── audit/                  confounder and demographic-stratification audit
├── svd/                    SVD preprocessing + external-validation runner
├── legacy_gcp/             earlier single-model TC-DANN (5-seed ensemble, conformal abstain, GKE)
├── docs/                   design notes, legacy README, GCP deployment guide
├── assets/                 aggregate audit figures
├── requirements.txt
└── .env.example
```

`legacy_gcp/` is an earlier variant (one transformer-fusion model, adversaries on site/age/sex, group temperature scaling and split-conformal abstention) containerized for parallel ensemble training on GKE H100 nodes; see [docs/LEGACY_SINGLE_MODEL.md](docs/LEGACY_SINGLE_MODEL.md) and [docs/GCP_DEPLOYMENT.md](docs/GCP_DEPLOYMENT.md). The final model in `tcdann/` replaced it.

## Team & credits

Built for **42-657 Projects in Biomedical AI**, Carnegie Mellon University (Spring 2026). Team: Devavrath Sandeep, Rayann Ramoutar, Shawn Xiang.

- **Devavrath Sandeep** (this repository): EDA, the confounder and shortcut audit, modality ablation and explainability analysis, the TC-DANN model and confidence score, SVD external validation, and the inference API and front-end apps.
- **Rayann Ramoutar and Shawn Xiang**: the VoxClinBench benchmark (benchmark package and evaluation protocol, cross-lingual transfer and fine-tuning analyses). That work, and the team's Bridge2AI-Voice baseline/MARVEL reproduction code, are not included here.
- `audit/demographic_stratification.py` consumes patient-level predictions produced by the team's benchmark models; those files are not included.
- Data: Bridge2AI-Voice consortium (PhysioNet) and the Saarbrücken Voice Database. Gradient reversal follows Ganin and Lempitsky (2015).

## License

MIT, see [LICENSE](LICENSE). The license covers the code only, not any dataset.
