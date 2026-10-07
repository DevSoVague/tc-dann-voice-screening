"""
calibrate_and_save.py

Run this AFTER training completes, BEFORE serving.

What it does:
  1. Loads all ensemble checkpoints from --ckpt_dir
  2. Filters to only members that passed C1 (confounder separation >= 0.10)
     — these are the "best" members used for inference
  3. Fits GroupTemperatureScaler on the val split logits (per group x disease cell)
  4. Fits SplitConformalPredictor on the calibrated val probabilities
  5. Computes per-(group, disease) positive support counts from the train split
  6. Saves everything to --artifact_dir:
       artifacts/
         scaler.pt              <- GroupTemperatureScaler state dict (PyTorch)
         conformal.joblib       <- SplitConformalPredictor (numpy quantiles)
         support_counts.joblib  <- [n_groups, n_diseases] int array
         meta.joblib            <- disease_cols, task_vocab, n_static, n_groups, best_member_ids
         confounder_model.joblib <- per-disease sklearn LR for the abstain A2 check

Usage:
    python calibrate_and_save.py \
        --root_dir /data/my_model \
        --ckpt_dir checkpoints/ \
        --artifact_dir artifacts/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from model import TCDANN
from dataset import BridgeVoiceDataset
from train import collect_predictions, per_disease_auroc, run_confounder_audit
from calibration import (
    GroupTemperatureScaler,
    SplitConformalPredictor,
    ensemble_mean_std,
    make_group_id,
)
from data_prep import build_master_manifest, DISEASE_FILE_MAP
from run import _build_demo_matrix, _build_task_vocab


N_SITES = 3
N_AGES = 4
N_GROUPS = N_SITES * N_AGES   # 12 groups total


def _load_ensemble(ckpt_dir: Path, device: str):
    ckpts = sorted(ckpt_dir.glob("model_*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")
    models, histories, meta = [], [], None
    for ck in ckpts:
        data = torch.load(ck, map_location=device, weights_only=False)
        if meta is None:
            meta = {
                "disease_cols": data["disease_cols"],
                "task_vocab":   data["task_vocab"],
                "n_static":     data["n_static"],
            }
        model = TCDANN(
            n_diseases=len(meta["disease_cols"]),
            n_tasks=len(meta["task_vocab"]),
            n_sites=N_SITES,
            n_age_buckets=N_AGES,
            n_sex=2,
            n_static=meta["n_static"],
        ).to(device)
        model.load_state_dict(data["state_dict"])
        model.eval()
        models.append(model)
        histories.append(data.get("history", {}))
    return models, histories, meta


def _select_best_members(
    models,
    histories,
    val_preds_list,
    disease_cols,
    val_demo_df,
    min_separation: float = 0.10,
):
    """
    Return indices of ensemble members that pass C1.
    If none pass, return all (with a warning).
    """
    passing = []
    for i, (preds, history) in enumerate(zip(val_preds_list, histories)):
        baseline = run_confounder_audit(preds, val_demo_df, disease_cols)
        auroc = per_disease_auroc(preds["logits"], preds["y"])
        gaps = {
            d: (float(auroc[j]) - float(baseline[d]))
            for j, d in enumerate(disease_cols)
        }
        n_passing = sum(1 for g in gaps.values() if not np.isnan(g) and g >= min_separation)
        n_total = sum(1 for g in gaps.values() if not np.isnan(g))
        print(f"  member {i}: {n_passing}/{n_total} diseases pass C1  gaps={gaps}")
        if n_passing >= n_total * 0.5:   # majority of diseases pass
            passing.append(i)

    if not passing:
        print("WARNING: no members pass C1 majority threshold — using all members")
        return list(range(len(models)))

    print(f"Best members (pass C1): {passing}")
    return passing


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("[calibrate] Loading ensemble...")
    models, histories, meta = _load_ensemble(ckpt_dir, device)
    disease_cols = meta["disease_cols"]
    task_vocab   = meta["task_vocab"]
    n_static     = meta["n_static"]
    D = len(disease_cols)

    print("[calibrate] Loading splits...")
    val_path   = ckpt_dir / "split_val.parquet"
    train_path = ckpt_dir / "split_train.parquet"
    if not val_path.exists() or not train_path.exists():
        print("  split parquets not found, rebuilding manifest...")
        rec, static, _ = build_master_manifest(Path(args.root_dir), list(DISEASE_FILE_MAP.keys()))
        from data_prep import stratified_participant_split
        splits = stratified_participant_split(rec)
        val_df   = splits["val"]
        train_df = splits["train"]
        _, static, _ = build_master_manifest(Path(args.root_dir), disease_cols)
    else:
        import pandas as pd
        val_df   = pd.read_parquet(val_path)
        train_df = pd.read_parquet(train_path)
        _, static, _ = build_master_manifest(Path(args.root_dir), list(DISEASE_FILE_MAP.keys()))

    features_dir = Path(args.root_dir) / "features"
    val_ds = BridgeVoiceDataset(val_df, disease_cols, task_vocab, static, features_dir)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # ----------------------------------------------------------------
    # 1. Collect val predictions from all members
    # ----------------------------------------------------------------
    print("[calibrate] Collecting val predictions from all members...")
    val_preds_list = []
    for i, model in enumerate(models):
        preds = collect_predictions(model, val_loader, device)
        val_preds_list.append(preds)
        print(f"  member {i}: logits shape {preds['logits'].shape}")

    # ----------------------------------------------------------------
    # 2. Select best members (C1 filter)
    # ----------------------------------------------------------------
    print("[calibrate] Selecting best ensemble members...")
    val_demo = _build_demo_matrix(val_df, task_vocab)
    import pandas as pd
    val_demo_df = pd.DataFrame(val_demo, index=val_df.index)
    best_ids = _select_best_members(
        models, histories, val_preds_list, disease_cols, val_demo_df
    )

    # ----------------------------------------------------------------
    # 3. Ensemble mean logits over best members only
    # ----------------------------------------------------------------
    best_logits = [val_preds_list[i]["logits"] for i in best_ids]
    p_mean, p_std = ensemble_mean_std(best_logits)
    ref_preds = val_preds_list[best_ids[0]]

    group_ids = make_group_id(ref_preds["site"], ref_preds["age"],
                               n_sites=N_SITES, n_ages=N_AGES).astype(int)
    y_val = ref_preds["y"]

    # ----------------------------------------------------------------
    # 4. Fit GroupTemperatureScaler
    # ----------------------------------------------------------------
    print("[calibrate] Fitting GroupTemperatureScaler...")
    # Average logits (pre-sigmoid) across best members for temperature fitting
    mean_logits = np.stack(best_logits).mean(axis=0)
    scaler = GroupTemperatureScaler(n_diseases=D, n_groups=N_GROUPS)
    scaler.fit(
        torch.tensor(mean_logits),
        torch.tensor(y_val).float(),
        torch.tensor(group_ids).long(),
    )
    torch.save(scaler.state_dict(), artifact_dir / "scaler.pt")
    print(f"  saved scaler.pt  temps range: "
          f"{scaler.temperatures().min().item():.3f} – "
          f"{scaler.temperatures().max().item():.3f}")

    # ----------------------------------------------------------------
    # 5. Calibrated probabilities
    # ----------------------------------------------------------------
    p_cal = scaler.transform(
        torch.tensor(mean_logits),
        torch.tensor(group_ids).long(),
    ).numpy()

    # ----------------------------------------------------------------
    # 6. Fit SplitConformalPredictor
    # ----------------------------------------------------------------
    print("[calibrate] Fitting SplitConformalPredictor...")
    conformal = SplitConformalPredictor(n_diseases=D, n_groups=N_GROUPS, alpha=0.1)
    conformal.fit(p_cal, y_val, group_ids)
    joblib.dump(conformal, artifact_dir / "conformal.joblib")
    n_fitted = int((~np.isnan(conformal.q)).sum())
    print(f"  saved conformal.joblib  fitted cells: {n_fitted}/{conformal.q.size}")

    # ----------------------------------------------------------------
    # 7. Support counts from training split
    # ----------------------------------------------------------------
    print("[calibrate] Computing support counts from train split...")
    train_ds = BridgeVoiceDataset(train_df, disease_cols, task_vocab, static, features_dir)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    # Collect site/age/labels without running the full model
    all_site, all_age, all_y = [], [], []
    for batch in train_loader:
        all_site.append(batch["site"].numpy())
        all_age.append(batch["age"].numpy())
        all_y.append(batch["disease"].numpy())
    tr_site = np.concatenate(all_site)
    tr_age  = np.concatenate(all_age)
    tr_y    = np.concatenate(all_y)
    tr_groups = make_group_id(tr_site, tr_age, n_sites=N_SITES, n_ages=N_AGES).astype(int)

    support_counts = np.zeros((N_GROUPS, D), dtype=int)
    for g in range(N_GROUPS):
        sel = tr_groups == g
        support_counts[g] = tr_y[sel].sum(axis=0).astype(int)
    joblib.dump(support_counts, artifact_dir / "support_counts.joblib")
    print(f"  saved support_counts.joblib  shape: {support_counts.shape}")

    # ----------------------------------------------------------------
    # 8. Confounder LR models (one per disease) for abstain check A2
    # ----------------------------------------------------------------
    print("[calibrate] Fitting per-disease confounder LR models...")
    confounder_models = {}
    for j, d in enumerate(disease_cols):
        if len(np.unique(y_val[:, j])) < 2:
            continue
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(val_demo, y_val[:, j])
        confounder_models[d] = lr
    joblib.dump(confounder_models, artifact_dir / "confounder_models.joblib")
    print(f"  saved confounder_models.joblib  ({len(confounder_models)} diseases)")

    # ----------------------------------------------------------------
    # 9. Save metadata
    # ----------------------------------------------------------------
    meta_out = {
        "disease_cols":    disease_cols,
        "task_vocab":      task_vocab,
        "n_static":        n_static,
        "n_groups":        N_GROUPS,
        "n_sites":         N_SITES,
        "n_ages":          N_AGES,
        "best_member_ids": best_ids,
        "n_ensemble":      len(models),
    }
    joblib.dump(meta_out, artifact_dir / "meta.joblib")
    print(f"  saved meta.joblib")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n[calibrate] Done. Artifacts written to {artifact_dir}/")
    print(f"  scaler.pt            GroupTemperatureScaler (PyTorch)")
    print(f"  conformal.joblib     SplitConformalPredictor (numpy)")
    print(f"  support_counts.joblib [n_groups={N_GROUPS}, n_diseases={D}]")
    print(f"  confounder_models.joblib  per-disease sklearn LR")
    print(f"  meta.joblib          disease_cols, task_vocab, best_member_ids")
    print(f"\n  Best member ids: {best_ids} (of {len(models)} total)")
    print(f"  Next step: python serve.py --ckpt_dir {ckpt_dir} --artifact_dir {artifact_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default=os.environ.get("B2AI_DATA_ROOT"), required=os.environ.get("B2AI_DATA_ROOT") is None)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--artifact_dir", default="artifacts")
    main(p.parse_args())
