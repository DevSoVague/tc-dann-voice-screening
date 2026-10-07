"""
explain_tc_dann.py  —  SHAP DeepExplainer attribution for TC-DANN quad models
Bridge2AI-Voice  |  companion to run_tc_dann.py

GradCAM is for spatial CNN feature maps — not applicable here.
DeepExplainer is the correct choice: it backpropagates through the fully-
connected architecture using DeepLIFT-style reference activations, giving
signed per-feature contributions for each disease head.

Outputs (saved to --out_dir):
    shap_summary_bar.png      top-30 features by mean |SHAP| pooled across diseases
    shap_heatmap.png          diseases × top features (row-normalised)
    shap_{disease}.png        per-disease beeswarm (bar fallback if > MAX_BEESWARM_N)
    shap_values.csv           mean |SHAP| and mean SHAP per (disease, feature)

Usage:
    python explain_tc_dann.py \\
        --bundle   tc_dann_results/psychiatric_model_best.joblib \\
        --data_root /path/to/my_model \\
        --stem     psychiatric \\
        [--task    prolonged-vowel]          # FiLM task conditioning
        [--all_tasks]                        # average SHAP over all 4 tasks
        [--n_background 100]                 # random background samples
        [--n_explain    500]                 # test rows to explain
        [--out_dir tc_dann_results/explain_psychiatric]

Notes:
  • run_tc_dann.py must be importable (same directory or on PYTHONPATH).
  • Background uses a random subset of the test set (training data not saved
    in the bundle).  This is a standard fallback and slightly conservative —
    it may underestimate attributions vs. a true out-of-distribution reference.
  • MPS is silently downgraded to CPU for SHAP: MPS autograd hooks are
    incompatible with DeepExplainer's backward pass.
  • BN layers are in eval() mode; Dropout is off.  Both are correct for
    attribution — you want deterministic forward passes.
"""

import argparse
import os
import hashlib
import sys
import warnings
from pathlib import Path
from run_tc_dann import SubgroupCentroidIndex, DemographicIndex

warnings.filterwarnings("ignore")

# ── dependency check ──────────────────────────────────────────────────────────
MISSING = []
for pkg in ["numpy", "pandas", "joblib", "torch", "shap", "matplotlib", "sklearn"]:
    try:
        __import__(pkg)
    except ImportError:
        MISSING.append(pkg)
if MISSING:
    sys.exit(f"Missing packages: pip install {' '.join(MISSING)}")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import architecture and data helpers from the main training script.
# run_tc_dann.py's main() is guarded by __name__ == "__main__" so importing
# only pulls in class/function definitions and module-level constants.
try:
    from run_tc_dann import (
        TCDANN, TASKS, TASK_ENC, DEVICE,
        load_all_features, load_labels_from_diagnosis_tsvs,
        ALL_DISEASES,
    )
except ImportError as exc:
    sys.exit(f"Cannot import from run_tc_dann.py — is it in the same directory?\n{exc}")

# Force CPU for SHAP; MPS does not support the autograd hooks DeepExplainer needs.
SHAP_DEVICE = "cpu" if DEVICE == "mps" else DEVICE

TOP_N          = 30   # features shown in bar / heatmap
MAX_BEESWARM_N = 300  # above this use bar plot instead of beeswarm


# ─────────────────────────────────────────────────────────────────────────────
# Model wrappers
# ─────────────────────────────────────────────────────────────────────────────

class _SingleHeadWrapper(nn.Module):
    """
    Exposes one disease head's sigmoid output as a scalar per sample.
    task_id is fixed — SHAP sees only feature variation, not task variation.
    Dropout is off (eval mode from caller).
    """
    def __init__(self, model: TCDANN, disease_idx: int, task_id: int):
        super().__init__()
        self.model       = model
        self.disease_idx = disease_idx
        self.task_id     = task_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ids = torch.full((x.shape[0],), self.task_id,
                         dtype=torch.long, device=x.device)
        logits, _ = self.model(x, ids, lam=0.0)
        return torch.sigmoid(logits[:, self.disease_idx : self.disease_idx + 1])


# ─────────────────────────────────────────────────────────────────────────────
# Bundle loading
# ─────────────────────────────────────────────────────────────────────────────

def load_bundle(bundle_path: Path):
    """
    Reconstructs model + preprocessing objects from the saved .joblib bundle.
    The bundle stores weight arrays (not live sklearn objects) so we rebuild
    imputer and scaler manually.
    """
    bundle = joblib.load(bundle_path)

    model = TCDANN(
        n_features = bundle["n_features"],
        n_tasks    = bundle["n_tasks"],
        n_sites    = bundle["n_sites"],
        n_diseases = bundle["n_diseases"],
    ).to(SHAP_DEVICE)

    # State dict values are stored as numpy arrays
    state = {}
    for k, v in bundle["model_state"].items():
        state[k] = torch.tensor(v).to(SHAP_DEVICE) if isinstance(v, np.ndarray) \
                   else v.to(SHAP_DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Reconstruct SimpleImputer (only .statistics_ needed for transform)
    imputer = SimpleImputer(strategy="median")
    imputer.statistics_    = bundle["imputer_statistics"].copy()
    imputer.n_features_in_ = len(imputer.statistics_)

    # 🔑 REQUIRED for newer sklearn
    imputer._fit_dtype = np.dtype(np.float32)

    # Reconstruct StandardScaler (mean_ + scale_ sufficient for transform)
    scaler = StandardScaler()
    scaler.mean_             = bundle["scaler_mean"].copy()
    scaler.scale_            = bundle["scaler_std"].copy()
    scaler.var_              = scaler.scale_ ** 2
    scaler.n_features_in_    = len(scaler.mean_)
    scaler.n_samples_seen_   = np.array(1000, dtype=np.int64)  # placeholder

    return model, bundle["feat_cols"], bundle["disease_cols"], imputer, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SHAP computation
# ─────────────────────────────────────────────────────────────────────────────

def _background(X: np.ndarray, n: int, rng) -> torch.Tensor:
    """Random background sample (reference baseline for DeepExplainer)."""
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return torch.tensor(X[idx], dtype=torch.float32).to(SHAP_DEVICE)


def run_shap_single_task(
    model:        TCDANN,
    X_test:       np.ndarray,
    disease_cols: list,
    task_id:      int,
    n_background: int,
    n_explain:    int,
    rng,
) -> tuple[dict, np.ndarray]:
    """
    Runs DeepExplainer for every disease head at a fixed task_id.
    Returns {disease: shap_array (n_explain, n_features)} and the
    subsampled X_explain rows used.
    """
    idx       = rng.choice(len(X_test), size=min(n_explain, len(X_test)), replace=False)
    X_explain = X_test[idx].astype(np.float32)
    bg        = _background(X_explain, n_background, rng)
    X_t       = torch.tensor(X_explain).to(SHAP_DEVICE)

    shap_dict = {}
    for di, disease in enumerate(disease_cols):
        print(f"    [{disease}] ...", end=" ", flush=True)
        wrapper = _SingleHeadWrapper(model, di, task_id).to(SHAP_DEVICE)
        wrapper.eval()
        explainer = shap.DeepExplainer(wrapper, bg)
        sv = explainer.shap_values(X_t, check_additivity=False)

        # Normalise output shape → (n_explain, n_features)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv)
        if sv.ndim == 3 and sv.shape[-1] == 1:
            sv = sv[:, :, 0]
        elif sv.ndim == 3 and sv.shape[0] == 1:
            sv = sv[0]

        shap_dict[disease] = sv.astype(np.float32)
        print(f"done  shape={sv.shape}")

    return shap_dict, X_explain


def run_shap(
    model:        TCDANN,
    X_test:       np.ndarray,
    disease_cols: list,
    task_ids:     list,       # one or all four
    n_background: int,
    n_explain:    int,
    seed:         int,
) -> tuple[dict, np.ndarray]:
    """
    Runs DeepExplainer for each task_id in task_ids, then averages.
    With a single task_id this is just a straight DeepExplainer pass.
    """
    rng = np.random.default_rng(seed)
    accumulated: dict = {}
    X_explain   = None

    for tid in task_ids:
        task_name = TASKS[tid]
        print(f"\n  Task: {task_name} (id={tid})")
        sd, X_exp = run_shap_single_task(
            model, X_test, disease_cols,
            tid, n_background, n_explain, rng)
        X_explain = X_exp          # same subsample each call (same rng seed)
        for d, sv in sd.items():
            accumulated[d] = accumulated.get(d, 0.0) + sv

    if len(task_ids) > 1:
        accumulated = {d: v / len(task_ids) for d, v in accumulated.items()}

    return accumulated, X_explain


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _short(name: str, maxlen: int = 38) -> str:
    return name if len(name) <= maxlen else name[:maxlen - 2] + ".."


def plot_per_disease(shap_dict, feat_cols, X_explain, out_dir, stem):
    feat_arr = np.array(feat_cols)
    for disease, sv in shap_dict.items():
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:TOP_N]
        names    = [_short(feat_arr[i]) for i in top_idx]

        if sv.shape[0] <= MAX_BEESWARM_N:
            # Beeswarm: colour encodes feature value (high=red, low=blue)
            shap.summary_plot(
                sv[:, top_idx],
                X_explain[:, top_idx],
                feature_names=names,
                show=False,
                plot_type="dot",
            )
            plt.gcf().suptitle(f"SHAP beeswarm — {disease}  [{stem}]",
                               fontsize=11, y=1.01)
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(names[::-1], mean_abs[top_idx][::-1], color="#E07B5A")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"SHAP — {disease}  [{stem}]", fontsize=11)

        plt.tight_layout()
        out = out_dir / f"shap_{disease}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  → {out.name}")


def plot_summary_bar(shap_dict, feat_cols, out_dir, stem) -> np.ndarray:
    """Top-N features by mean |SHAP| pooled across all diseases in this model."""
    all_sv   = np.concatenate(list(shap_dict.values()), axis=0)
    mean_abs = np.abs(all_sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:TOP_N]
    feat_arr = np.array(feat_cols)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [_short(feat_arr[i]) for i in reversed(top_idx)],
        mean_abs[top_idx][::-1],
        color="#4C8CBF",
    )
    ax.set_xlabel("Mean |SHAP value|  (pooled across diseases)")
    ax.set_title(f"Top {TOP_N} features  —  {stem} model", fontsize=13)
    plt.tight_layout()
    out = out_dir / "shap_summary_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.name}")
    return top_idx


def plot_heatmap(shap_dict, feat_cols, top_idx, out_dir, stem):
    """
    Diseases × top features.  Values are mean |SHAP|, row-normalised so
    each disease's strongest feature = 1.0.  This makes sparse psychiatric
    diseases visually comparable to data-rich voice/onco diseases.
    """
    feat_arr = np.array(feat_cols)
    diseases = list(shap_dict.keys())
    matrix   = np.array([
        np.abs(shap_dict[d])[:, top_idx].mean(axis=0) for d in diseases
    ])
    row_max  = matrix.max(axis=1, keepdims=True)
    norm     = matrix / np.where(row_max == 0, 1.0, row_max)

    h = max(3.5, len(diseases) * 0.55)
    w = max(12,  len(top_idx) * 0.50)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels([_short(feat_arr[i], 30) for i in top_idx],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(diseases)))
    ax.set_yticklabels(diseases, fontsize=9)
    ax.set_title(f"SHAP feature importance  —  {stem}  (row-normalised)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, label="Relative importance")
    plt.tight_layout()
    out = out_dir / "shap_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.name}")


def save_csv(shap_dict, feat_cols, out_dir):
    feat_arr = np.array(feat_cols)
    rows = []
    for disease, sv in shap_dict.items():
        mean_abs  = np.abs(sv).mean(axis=0)
        mean_shap = sv.mean(axis=0)          # signed: positive = pushes toward diagnosis
        rank      = np.argsort(mean_abs)[::-1]
        for r, fi in enumerate(rank):
            rows.append({
                "disease":     disease,
                "rank":        r + 1,
                "feature":     feat_arr[fi],
                "mean_abs_shap": round(float(mean_abs[fi]), 6),
                "mean_shap":   round(float(mean_shap[fi]), 6),
            })
    df = pd.DataFrame(rows)
    out = out_dir / "shap_values.csv"
    df.to_csv(out, index=False)
    print(f"  → {out.name}  ({len(df):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SHAP DeepExplainer attribution for TC-DANN quad models")
    parser.add_argument("--bundle",       required=True,
                        help="Path to *_model_best.joblib from run_tc_dann.py")
    parser.add_argument("--data_root",    default=os.environ.get("B2AI_DATA_ROOT"), required=os.environ.get("B2AI_DATA_ROOT") is None,
                        help="Same --data_root as used for training")
    parser.add_argument("--stem",         required=True,
                        choices=["voice_onco", "neurological",
                                 "respiratory", "psychiatric"],
                        help="Model stem — must match the bundle")
    parser.add_argument("--task",         default="prolonged-vowel",
                        choices=TASKS,
                        help="Task to condition FiLM on (default: prolonged-vowel)")
    parser.add_argument("--all_tasks",    action="store_true",
                        help="Run SHAP for all 4 tasks and average the values")
    parser.add_argument("--n_background", type=int, default=100,
                        help="Background sample size for DeepExplainer reference")
    parser.add_argument("--n_explain",    type=int, default=500,
                        help="Max test-set rows to explain")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--test_size",    type=float, default=0.2,
                        help="Must match the value used in run_tc_dann.py")
    parser.add_argument("--cache_dir",    type=str, default=None)
    parser.add_argument("--no_cache",     action="store_true")
    parser.add_argument("--smoke_test",   action="store_true")
    parser.add_argument("--out_dir",      type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root   = Path(args.data_root)
    bundle_path = Path(args.bundle)
    out_dir     = (Path(args.out_dir) if args.out_dir
                   else bundle_path.parent / f"explain_{args.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    task_ids = list(TASK_ENC.values()) if args.all_tasks else [TASK_ENC[args.task]]
    task_label = "all tasks (averaged)" if args.all_tasks else args.task

    if SHAP_DEVICE != DEVICE:
        print(f"\n  NOTE: MPS detected — downgrading to CPU for SHAP "
              f"(MPS autograd hooks incompatible with DeepExplainer)")

    # ── 1. Load model bundle ─────────────────────────────────────────────────
    print(f"\n[1/4] Loading bundle: {bundle_path.name}")
    model, feat_cols, disease_cols, imputer, scaler = load_bundle(bundle_path)
    print(f"  diseases  : {disease_cols}")
    print(f"  features  : {len(feat_cols)}")
    print(f"  task cond : {task_label}")
    print(f"  SHAP device: {SHAP_DEVICE}")

    # ── 2. Load data ─────────────────────────────────────────────────────────
    _cache_root   = (Path(args.cache_dir) if args.cache_dir
                     else data_root / "tc_dann_cache")
    _cache_key    = hashlib.md5(
        f"{data_root}|smoke={args.smoke_test}".encode()).hexdigest()[:10]
    _cache_master = _cache_root / f"master_df_{_cache_key}.joblib"
    _cache_labels = _cache_root / f"label_df_{_cache_key}.joblib"
    max_rows      = 2000 if args.smoke_test else None

    print(f"\n[2/4] Loading data...")
    if not args.no_cache and _cache_master.exists() and _cache_labels.exists():
        print(f"  [cache] {_cache_root}")
        master_df = joblib.load(_cache_master)
        label_df  = joblib.load(_cache_labels)
    else:
        master_df = load_all_features(data_root, max_mfcc_rows=max_rows)
        label_df  = load_labels_from_diagnosis_tsvs(data_root)

    disease_label_cols = [d for d in ALL_DISEASES if label_df[d].sum() >= 10]
    merged = master_df.merge(
        label_df[["participant_id"] + disease_label_cols],
        on="participant_id", how="inner")
    if "any_quality_flag" in merged.columns:
        merged = merged[~merged["any_quality_flag"]].reset_index(drop=True)

    # Reproduce the exact train/test split used during training
    all_pids = merged["participant_id"].unique()
    _, test_pids = train_test_split(all_pids, test_size=args.test_size,
                                    random_state=args.seed)
    test_df = merged[merged["participant_id"].isin(test_pids)].reset_index(drop=True)
    print(f"  test rows        : {len(test_df):,}")
    print(f"  test participants: {len(test_pids):,}")

    # ── 3. Preprocess ────────────────────────────────────────────────────────
    print(f"\n[3/4] Preprocessing {len(feat_cols)} features...")
    # Align to feat_cols stored in bundle; fill absent cols with 0 before impute
    X_raw = np.zeros((len(test_df), len(feat_cols)), dtype=np.float32)
    present = [c for c in feat_cols if c in test_df.columns]
    absent  = [c for c in feat_cols if c not in test_df.columns]
    if absent:
        print(f"  WARNING: {len(absent)} feat_cols absent from test_df — "
              f"filled with 0 before impute")
    col_pos = {c: i for i, c in enumerate(feat_cols)}
    for c in present:
        X_raw[:, col_pos[c]] = test_df[c].fillna(np.nan).values.astype(np.float32)

    X_imp  = imputer.transform(X_raw)
    X_test = scaler.transform(X_imp).astype(np.float32)
    print(f"  X_test : {X_test.shape}")

    # Per-disease label counts (informational)
    for d in disease_cols:
        if d in test_df.columns:
            n = int(test_df[d].fillna(0).sum())
            print(f"    {d:<45} n_pos={n}")

    # ── 4. SHAP ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Running SHAP DeepExplainer  "
          f"(n_explain≤{args.n_explain}, n_background={args.n_background}, "
          f"task={task_label})")
    shap_dict, X_explain = run_shap(
        model, X_test, disease_cols,
        task_ids, args.n_background, args.n_explain, args.seed)

    # ── Plots + CSV ───────────────────────────────────────────────────────────
    print(f"\nSaving outputs to: {out_dir}")
    top_idx = plot_summary_bar(shap_dict, feat_cols, out_dir, args.stem)
    plot_heatmap(shap_dict, feat_cols, top_idx, out_dir, args.stem)
    plot_per_disease(shap_dict, feat_cols, X_explain, out_dir, args.stem)
    save_csv(shap_dict, feat_cols, out_dir)

    print(f"\n{'─'*60}")
    print(f"✓ {out_dir}/shap_summary_bar.png")
    print(f"✓ {out_dir}/shap_heatmap.png")
    for d in disease_cols:
        print(f"✓ {out_dir}/shap_{d}.png")
    print(f"✓ {out_dir}/shap_values.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
