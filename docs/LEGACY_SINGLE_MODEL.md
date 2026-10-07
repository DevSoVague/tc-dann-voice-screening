# TC-DANN

**Task-Conditioned, Domain-Adversarial Network for Bridge2AI-Voice.**

Constructive follow-up to the Member 2 Confounder & Shortcut Audit (March 23, 2026). Designed to defeat the specific shortcuts identified in that audit and to emit calibrated, subgroup-aware, abstainable confidence scores.

This build is wired to the Bridge2AI-Voice v3.0.0 public-release folder layout. **You do not need to edit any of the Python files.**

---

## 1. What this is, in one paragraph

The audit showed that a non-acoustic logistic regression on demographics plus a task histogram already reaches macro AUROC 0.82 (Tier 1) and 0.87 (Tier 2), and that the same main-model checkpoint drops up to 0.455 AUROC between Protocol A and Protocol B. MARVEL's near-perfect psychiatric AUROC is a cohort-identity artefact. TC-DANN is built to make those shortcuts unusable at training time: task is a conditioning input (not something to infer from which-recordings-exist), site / age / sex are stripped by gradient-reversal adversaries, and the output layer abstains when confidence is not clinically actionable.

## 2. Expected folder layout

The code assumes this directory structure (which matches your `my_model/`):

```
my_model/
├── features/
│   ├── torchaudio_spectrogram.parquet
│   ├── torchaudio_mel_spectrogram.parquet
│   ├── sparc_ema.parquet
│   ├── static_features.tsv
│   └── ... (others are ignored)
├── phenotype/
│   ├── demographics/demographics.tsv
│   ├── diagnosis/*.tsv                (one TSV per disease, plus control.tsv)
│   ├── enrollment/participant.tsv
│   └── task/{recording,session,acoustic_task}.tsv
└── files/                             <-- this folder, where you run commands from
    ├── README.md  (this file)
    ├── data_prep.py
    ├── dataset.py
    ├── model.py
    ├── run.py
    └── ...
```

All parquet and TSV column names are detected automatically. Prefixed schemas (e.g. `mfcc_participant_id`) and unprefixed schemas both work.

## 3. One-time setup

```bash
cd my_model/files
pip install -r requirements.txt
```

## 4. Three commands, in order

**Step A. Verify the data layout is understood correctly.**

```bash
python run.py diagnose
```

You should see: (a) every parquet listed with its column names, (b) row counts per diagnosis TSV, (c) a successful master-manifest build with positive-count-per-disease, and (d) the site and sex distributions. If any step fails, the traceback tells you which file / column was missing.

**Step B. Train the ensemble.**

```bash
python run.py train --epochs 40 --ensemble_size 5
```

This is the full run. On CPU expect several hours per member, on a GPU it should be much faster. To smoke-test first:

```bash
python run.py train --epochs 2 --ensemble_size 1 --batch_size 8
```

Checkpoints land in `checkpoints/` plus the three `split_{train,val,test}.parquet` files for reproducibility.

**Step C. Evaluate against the four audit criteria.**

```bash
python run.py eval --ckpt_dir checkpoints/
```

Writes `eval_out/four_criteria.json` and prints a pass/fail summary per criterion. Criteria C1, C3, C4 are all computed from the checkpoints and the internal test split. Criterion C2 (Protocol A vs Protocol B) will appear as empty unless you plug in your team's existing protocol-split code into the two dicts marked `protA_auroc` / `protB_auroc` in `run.py::cmd_eval`. That is the one remaining integration point because Protocol B requires your group's cohort-construction logic, which is team-specific.

## 5. What each file does

| File | Role |
|---|---|
| `data_prep.py` | Builds the master manifest from your diagnosis TSVs + recording TSV + demographics TSV. Handles disease-name mapping, site / sex normalisation, participant-disjoint split. |
| `dataset.py` | `BridgeVoiceDataset` plus a `FeatureStore` that auto-detects parquet schemas. Subgroup-balanced sampler + cross-site mixup (audit 2.4). |
| `layers.py` | Gradient Reversal Layer (Ganin 2015) + Masked Stats Pooling. |
| `model.py` | TCDANN architecture. Four-branch fusion (spec, mel, EMA, static) + task-ID conditioning + disease-specific attention + three adversarial heads. Static dim auto-adapts to your TSV width. |
| `train.py` | Training loop with adversarial λ schedule, confounder-audit callback, protocol A/B evaluation, worst-subgroup checkpoint selection. |
| `calibration.py` | Group-conditional temperature scaling + split conformal + ensemble helpers. |
| `confidence.py` | Abstain rule combining ensemble agreement, confounder-disagreement, subgroup support, conformal score. |
| `evaluate.py` | Four-criteria validator (C1 confounder separation, C2 protocol stability, C3 subgroup uniformity, C4 external transfer). |
| `run.py` | CLI entry point. Auto-detects `--root_dir`, three commands: `diagnose`, `train`, `eval`. |

## 6. Audit-finding → design-choice mapping

| Audit finding | Design response |
|---|---|
| Task-histogram shortcut (audit 2.2) | Per-recording training + explicit task-ID embedding. Task information is an input, not a signal to infer. |
| Country gap, Parkinson's USA 0.913 vs Canada 0.487 (audit 2.4) | Gradient-reversal head against site, weight 1.0. Cross-site mixup. |
| Age gap, psych_history 71+ = 0.189 (audit 2.4) | Gradient-reversal head against age bucket + subgroup-balanced sampler. |
| Sex gap, PTSD F 0.533 vs M 0.832 (audit 2.4) | Gradient-reversal head against sex + subgroup-balanced sampler. |
| Fusion conflict, ADHD 427313 (update Slide 18) | Four-branch fusion instead of six. No MFCC, no PPG. |
| Psychiatric labels shortcut-dominated (audit 2.1, 2.3) | `DEPLOYABLE_DISEASES` vs `EXPLORATORY_DISEASES` scope split. Psychiatric labels are opt-in via `--include_exploratory`. |
| MARVEL cannot be shown to outperform the confounder baseline on CI / PD (audit 2.1) | The four-criteria evaluator in `evaluate.py` makes confounder separation a first-class pass / fail, not a footnote. |

## 7. The confidence score pipeline

Raw sigmoid → deep-ensemble mean + std → group-conditional temperature scaling → split conformal non-conformity → four-condition abstain rule. A prediction is returned only if all four conditions hold:

- Ensemble std < 0.15
- |p_acoustic − p_confounder_baseline| ≥ 0.10
- ≥ 30 training positives in the (disease × site × age_bucket) cell
- Conformal non-conformity ≤ 1.5 × subgroup calibration quantile

Otherwise the model emits `abstain` with a reason code. This is what makes the score clinically actionable: the model refuses to answer where evidence is insufficient, rather than returning an overconfident wrong answer (which is the exact failure mode the audit caught on 71+ psychiatric patients).

## 8. Expected numbers vs MARVEL

This is important to set honestly so the framing is clear at the defense. TC-DANN is **not** expected to beat MARVEL on Protocol A macro AUROC on confounded diseases. MARVEL's advantage there is cohort-identity exploitation, which TC-DANN is specifically designed not to do.

| Axis | MARVEL | TC-DANN (target) |
|---|---|---|
| Protocol A macro AUROC | ~0.97 | ~0.90-0.93 |
| Protocol B macro AUROC | ~0.95 | ~0.88-0.92 |
| Protocol gap | ~0.02 | ≤ 0.05 |
| Confounder separation on Parkinson's | +0.04 (fails C1) | ≥ +0.10 (passes C1) |
| Worst-subgroup AUROC, 71+ psychiatric | 1.00 (ceiling, shortcut) | abstain flagged |

The story: TC-DANN trades internal benchmark ceiling for confounder-resistance, subgroup uniformity, and honest abstain behaviour. It is the first of the three models (MARVEL, original main model, TC-DANN) designed to pass the four-criteria framework on any disease.

## 9. Known integration points

- **Protocol B AUROC.** `run.py::cmd_eval` has two empty dicts where your team's Protocol A / Protocol B AUROC values should be plugged in. Everything else runs end-to-end without edits.
- **Calibration split.** The conformal predictor and group-conditional temperature scaler want a dedicated calibration split (~10% of training data). The current `eval` command uses the ensemble mean directly, which is fine for the four-criteria check but leaves the abstain layer ungrounded. Fitting it is a ~20-line addition when you are ready.
- **External transfer (C4).** Runs when you pass SVD / NeuroVoz / COUGHVID / MODMA recordings through the trained model with matching feature extraction. Not automated here because it depends on how your team stages those corpora.

## 10. Citations

- Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. ICML.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. (2017). On Calibration of Modern Neural Networks. ICML.
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile Regression. NeurIPS.
- Piao et al. (2025). MARVEL. arXiv 2508.20717.
- Bridge2AI-Voice v3.0.0 dataset documentation (b2aiprep).
- Member 2 Confounder & Shortcut Audit (March 23, 2026).

---

*Built to match the audit, not to beat a number.*
