"""
Entry point for TC-DANN.

Commands:
    python run.py diagnose --root_dir ..
        Scans features/ and phenotype/, prints the detected schema and
        per-disease positive counts. Use this FIRST to confirm the data
        layout is understood correctly.

    python run.py train --root_dir ..
        Builds the manifest, does a participant-disjoint 70/15/15 split,
        trains an ensemble of K models, saves checkpoints to --out_dir.

    python run.py eval --root_dir .. --ckpt_dir checkpoints
        Loads the ensemble, evaluates against the four-criteria framework
        from the audit report, writes eval_out/four_criteria.json.

Auto root detection: if --root_dir is not given, we look at the current
working directory and its parent for the pair of (features/, phenotype/)
directories and use the first match. This means you can just:

    cd my_model/files
    python run.py diagnose
    python run.py train
    python run.py eval --ckpt_dir checkpoints/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import TCDANN
from dataset import BridgeVoiceDataset, build_subgroup_sampler
from train import TrainConfig, fit, collect_predictions
from calibration import ensemble_mean_std, make_group_id
from evaluate import run_four_criteria
from data_prep import (
    DISEASE_FILE_MAP,
    build_master_manifest,
    stratified_participant_split,
    diagnose_schemas,
)


# ------------------------------------------------------------------
# Disease scope (audit-driven, Section 5.5)
# ------------------------------------------------------------------

DEPLOYABLE_DISEASES = [
    "parkinsons",
    "airway_stenosis",
    "laryngeal_dystonia",
    "vf_paralysis",
    "mtd",
    "chronic_cough",
    "benign_lesions",
    "laryngitis",
    "glottic_insufficiency",
]

EXPLORATORY_DISEASES = [
    "cognitive_impairment",
    "depression",
    "ptsd",
    "adhd",
    "psychiatric_history",
    "bipolar",
    "anxiety",
]


# ------------------------------------------------------------------
# Root auto-detection
# ------------------------------------------------------------------

def _auto_root(explicit: str = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    here = Path.cwd()
    for candidate in (here, here.parent, here.parent.parent):
        if (candidate / "features").is_dir() and (candidate / "phenotype").is_dir():
            return candidate.resolve()
    raise RuntimeError(
        "Could not auto-detect --root_dir. Please pass it explicitly. "
        "Expected a directory containing both features/ and phenotype/ subfolders."
    )


def _filter_diseases_to_trainable(manifest: pd.DataFrame,
                                  diseases: List[str],
                                  min_positives: int = 5) -> List[str]:
    """Drop diseases with fewer than min_positives participant positives."""
    ppl = manifest.drop_duplicates("participant_id")
    keep = []
    for d in diseases:
        if d not in ppl.columns:
            print(f"  skipping '{d}': not in manifest")
            continue
        n = int(ppl[d].sum())
        if n < min_positives:
            print(f"  skipping '{d}': only {n} positive participants")
            continue
        keep.append(d)
    return keep


def _build_task_vocab(manifest: pd.DataFrame) -> Dict[str, int]:
    tasks = sorted(set(manifest.task.astype(str).unique()))
    return {t: i for i, t in enumerate(tasks)}


def _build_demo_matrix(df: pd.DataFrame, task_vocab: Dict[str, int]) -> np.ndarray:
    """
    Non-acoustic confounder-baseline feature matrix:
    age, sex one-hot, site one-hot, and a per-row task indicator.
    """
    out = pd.DataFrame(index=df.index)
    out["age"] = pd.to_numeric(df.age, errors="coerce").fillna(df.age.median())
    for s in ("F", "M"):
        out[f"sex_{s}"] = (df.sex == s).astype(int)
    for s in ("USA", "Canada", "Other"):
        out[f"site_{s}"] = (df.site == s).astype(int)
    for t, i in task_vocab.items():
        out[f"task_{i}"] = (df.task == t).astype(int)
    return out.values.astype(np.float32)


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

def cmd_diagnose(args):
    root = _auto_root(args.root_dir)
    print(f"Root: {root}")
    diagnose_schemas(root)


def cmd_train(args):
    root = _auto_root(args.root_dir)
    print(f"[train] root = {root}")

    diseases_all = DEPLOYABLE_DISEASES + (
        EXPLORATORY_DISEASES if args.include_exploratory else []
    )
    print(f"[train] building master manifest for {len(diseases_all)} diseases...")
    rec, static, n_static = build_master_manifest(root, diseases_all)
    print(f"[train] manifest: {len(rec)} recordings, "
          f"{rec.participant_id.nunique()} participants, "
          f"static dim = {n_static}")

    disease_cols = _filter_diseases_to_trainable(
        rec, diseases_all, min_positives=args.min_positives
    )
    if not disease_cols:
        print("[train] ERROR: no diseases with sufficient positives. Aborting.")
        sys.exit(1)
    print(f"[train] trainable diseases ({len(disease_cols)}): {disease_cols}")

    splits = stratified_participant_split(rec, seed=args.seed)
    for name, df in splits.items():
        print(f"[train]   split {name:5s}: {len(df)} recordings, "
              f"{df.participant_id.nunique()} participants")

    task_vocab = _build_task_vocab(rec)
    print(f"[train] task vocab ({len(task_vocab)}): {list(task_vocab.keys())}")

    train_ds = BridgeVoiceDataset(
        splits["train"], disease_cols, task_vocab, static, root / "features",
    )
    val_ds = BridgeVoiceDataset(
        splits["val"], disease_cols, task_vocab, static, root / "features",
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"[train] device = {cfg.device}")

    sampler = build_subgroup_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(cfg.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(cfg.device == "cuda"),
    )

    # pos_weight for class imbalance
    pos_counts = splits["train"][disease_cols].sum().values
    neg_counts = len(splits["train"]) - pos_counts
    pos_weight = torch.tensor(
        neg_counts / np.clip(pos_counts, 1, None)
    ).float().to(cfg.device)

    # Demo matrix for confounder-audit callback
    val_demo = pd.DataFrame(
        _build_demo_matrix(splits["val"], task_vocab),
        index=splits["val"].index,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save splits for reproducibility
    for name, df in splits.items():
        df.to_parquet(out_dir / f"split_{name}.parquet")

    # Train ensemble
    for k in range(args.ensemble_size):
        seed = args.seed + k
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = TCDANN(
            n_diseases=len(disease_cols),
            n_tasks=len(task_vocab),
            n_sites=3, n_age_buckets=4, n_sex=2,
            n_static=n_static,
        )
        print(f"\n=== ensemble member {k} (seed={seed}) ===")
        history = fit(
            model, train_loader, val_loader, cfg, disease_cols,
            val_demo_df=val_demo, pos_weight=pos_weight,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "disease_cols": disease_cols,
                "task_vocab": task_vocab,
                "n_static": n_static,
                "history": history,
            },
            out_dir / f"model_{k}.pt",
        )
        best_worst = max(
            [v for v in history["val_worst"] if v is not None and not np.isnan(v)],
            default=float("nan"),
        )
        print(f"[ensemble {k}] best_worst_subgroup = {best_worst:.3f}")

    print(f"\n[train] Done. Checkpoints written to {out_dir}/")


def cmd_eval(args):
    root = _auto_root(args.root_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(ckpt_dir.glob("model_*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoints in {ckpt_dir}")

    device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
    first = torch.load(ckpts[0], map_location=device, weights_only=False)
    disease_cols = first["disease_cols"]
    task_vocab = first["task_vocab"]
    n_static = first["n_static"]

    # Re-create splits from saved parquets (fast) or rebuild
    test_path = ckpt_dir / "split_test.parquet"
    if test_path.exists():
        test_df = pd.read_parquet(test_path)
        _, static, _ = build_master_manifest(root, list(DISEASE_FILE_MAP.keys()))
    else:
        print("[eval] split_test.parquet not found, rebuilding from scratch")
        rec, static, _ = build_master_manifest(root, list(DISEASE_FILE_MAP.keys()))
        splits = stratified_participant_split(rec, seed=args.seed)
        test_df = splits["test"]

    test_ds = BridgeVoiceDataset(
        test_df, disease_cols, task_vocab, static, root / "features",
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # Collect ensemble predictions
    ensemble_logits = []
    last_preds = None
    for ck in ckpts:
        ck_data = torch.load(ck, map_location=device, weights_only=False)
        model = TCDANN(
            n_diseases=len(disease_cols),
            n_tasks=len(task_vocab),
            n_sites=3, n_age_buckets=4, n_sex=2,
            n_static=n_static,
        ).to(device)
        model.load_state_dict(ck_data["state_dict"])
        preds = collect_predictions(model, test_loader, device)
        ensemble_logits.append(preds["logits"])
        last_preds = preds

    p_mean, p_std = ensemble_mean_std(ensemble_logits)

    # Protocol A/B AUROC placeholders: requires external protocol code.
    # Drop in your own Protocol A / Protocol B split code
    # and fill the two dicts below. Left empty -> C2 section of the report
    # will just be blank.
    protA_auroc: Dict[str, float] = {}
    protB_auroc: Dict[str, float] = {}

    demo_matrix = _build_demo_matrix(test_df, task_vocab)
    result = run_four_criteria(
        p_acoustic_internal=p_mean,
        y_internal=last_preds["y"],
        demo_internal=demo_matrix,
        protA_auroc=protA_auroc,
        protB_auroc=protB_auroc,
        site=last_preds["site"],
        age=last_preds["age"],
        sex=last_preds["sex"],
        disease_cols=disease_cols,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "four_criteria.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"[eval] wrote {out_dir}/four_criteria.json")

    # Summary
    print("\n=== Four-Criteria Summary ===")
    for crit_name, per_disease in result.items():
        if per_disease is None:
            continue
        passed = sum(1 for v in per_disease.values() if v.get("passed"))
        total = len(per_disease)
        print(f"  {crit_name}: {passed}/{total} pass")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pd_cmd = sub.add_parser("diagnose",
                            help="Scan the data layout and print detected schema")
    pd_cmd.add_argument("--root_dir", default=os.environ.get("B2AI_DATA_ROOT"))
    pd_cmd.set_defaults(func=cmd_diagnose)

    pt = sub.add_parser("train", help="Train the TC-DANN ensemble")
    pt.add_argument("--root_dir", default=os.environ.get("B2AI_DATA_ROOT"))
    pt.add_argument("--out_dir", default="checkpoints")
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--batch_size", type=int, default=16)
    pt.add_argument("--num_workers", type=int, default=2)
    pt.add_argument("--ensemble_size", type=int, default=5)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--min_positives", type=int, default=5)
    pt.add_argument("--include_exploratory", action="store_true")
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="Evaluate the ensemble against the four criteria")
    pe.add_argument("--root_dir", default=os.environ.get("B2AI_DATA_ROOT"))
    pe.add_argument("--ckpt_dir", required=True)
    pe.add_argument("--out_dir", default="eval_out")
    pe.add_argument("--batch_size", type=int, default=32)
    pe.add_argument("--num_workers", type=int, default=2)
    pe.add_argument("--seed", type=int, default=42)
    pe.set_defaults(func=cmd_eval)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
