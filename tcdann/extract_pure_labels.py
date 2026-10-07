"""
extract_pure_labels.py
======================
Extracts participants that have EXACTLY ONE diagnosis (no comorbidities)
from the Bridge2AI-Voice v3.0.0 dataset.

Outputs
-------
pure_<disease>.npy          — 1-D string array of participant IDs for each disease
pure_single_label_participants.npz — all diseases in one archive

Usage
-----
    python extract_pure_labels.py [--diag_dir /path/to/phenotype/diagnosis]

Default path matches the EDA_v3 notebook config.
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Default path (matches EDA_v3 notebook) ────────────────────────────────────
# Set B2AI_DATA_ROOT to your local Bridge2AI-Voice v3.0.0 folder (contains phenotype/).
DEFAULT_DIAG_DIR = Path(os.environ.get("B2AI_DATA_ROOT", ".")) / "phenotype" / "diagnosis"

DIAGNOSIS_CATEGORY = {
    "adhd_adult":                       "Psychiatric",
    "ptsd_adult":                       "Psychiatric",
    "control":                          "Control",
    "airway_stenosis":                  "Respiratory",
    "parkinsons_disease":               "Neurological",
    "psychiatric_history":              "Psychiatric",
    "laryngeal_dystonia":               "Voice",
    "cognitive_impairment":             "Neurological",
    "unilateral_vocal_fold_paralysis":  "Voice",
    "benign_lesions":                   "Voice",
    "muscle_tension_dysphonia":         "Voice",
    "depression":                       "Psychiatric",
    "unexplained_chronic_cough":        "Respiratory",
    "anxiety":                          "Psychiatric",
    "precancerous_lesions":             "Voice",
    "bipolar_disorder":                 "Psychiatric",
    "copd_and_asthma":                  "Respiratory",
    "glottic_insufficiency":            "Voice",
    "laryngitis":                       "Voice",
    "laryngeal_cancer":                 "Voice",
    "amyotrophic_lateral_sclerosis":    "Neurological",
}


def load_participant_diagnoses(diag_dir: Path) -> dict:
    """Load all TSVs. Returns {participant_id: [list of conditions]}."""
    participant_diagnoses: dict[str, list[str]] = {}
    tsv_files = sorted(diag_dir.glob("*.tsv"))

    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in {diag_dir}")

    print(f"Found {len(tsv_files)} diagnosis TSV files\n")
    for fpath in tsv_files:
        condition = fpath.stem
        try:
            df = pd.read_csv(fpath, sep="\t", dtype=str)
            df.columns = df.columns.str.strip()
            if "participant_id" not in df.columns:
                print(f"  ⚠  {fpath.name}: no participant_id column — skipped")
                continue
            pids = df["participant_id"].str.strip().dropna().unique()
            for pid in pids:
                participant_diagnoses.setdefault(pid, []).append(condition)
            print(f"  ✓  {condition:<45} {len(pids):>4} participants")
        except Exception as exc:
            print(f"  ✗  {fpath.name}: {exc}")

    return participant_diagnoses


def extract_pure(participant_diagnoses: dict) -> dict[str, np.ndarray]:
    """
    Return {disease: np.array([pid, ...])} for participants with
    EXACTLY ONE diagnosis (no comorbidities).
    """
    single_dx: dict[str, list[str]] = defaultdict(list)
    for pid, diags in participant_diagnoses.items():
        if len(diags) == 1:
            single_dx[diags[0]].append(pid)

    return {dx: np.array(sorted(pids), dtype=str)
            for dx, pids in single_dx.items()}


def main():
    parser = argparse.ArgumentParser(description="Extract pure single-label participant IDs")
    parser.add_argument(
        "--diag_dir", type=Path, default=DEFAULT_DIAG_DIR,
        help="Path to phenotype/diagnosis directory"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("."),
        help="Output directory for .npy / .npz files (default: current dir)"
    )
    args = parser.parse_args()

    if not args.diag_dir.exists():
        raise SystemExit(
            f"ERROR: diagnosis directory not found:\n  {args.diag_dir}\n"
            "Pass --diag_dir /correct/path/to/phenotype/diagnosis"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    participant_diagnoses = load_participant_diagnoses(args.diag_dir)

    total = len(participant_diagnoses)
    multi = sum(1 for d in participant_diagnoses.values() if len(d) > 1)
    single = total - multi
    print(f"\nTotal participants with ≥1 diagnosis : {total:,}")
    print(f"Multi-diagnosis participants          : {multi:,}")
    print(f"Single-diagnosis participants         : {single:,}\n")

    # ── Extract pure labels ────────────────────────────────────────────────────
    pure = extract_pure(participant_diagnoses)

    print(f"{'Disease':<45} {'Category':<14} {'Pure N':>7}  Output")
    print("─" * 90)

    npz_dict: dict[str, np.ndarray] = {}
    for dx, arr in sorted(pure.items(), key=lambda x: -len(x[1])):
        cat = DIAGNOSIS_CATEGORY.get(dx, "Other")
        npy_path = args.out_dir / f"pure_{dx}.npy"
        np.save(npy_path, arr)
        npz_dict[dx] = arr
        print(f"  {dx:<43} {cat:<14} {len(arr):>7}  → {npy_path.name}")

    # ── Combined archive ───────────────────────────────────────────────────────
    npz_path = args.out_dir / "pure_single_label_participants.npz"
    np.savez(npz_path, **npz_dict)

    print(f"\n{'─'*90}")
    print(f"Diseases with pure-label participants : {len(pure)}")
    print(f"Combined archive saved                : {npz_path}")

    # ── Quick verification ────────────────────────────────────────────────────
    print("\n── Verification: reloading .npz ──")
    loaded = np.load(npz_path, allow_pickle=False)
    for key in sorted(loaded.files):
        print(f"  {key:<43}  shape={loaded[key].shape}  dtype={loaded[key].dtype}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
