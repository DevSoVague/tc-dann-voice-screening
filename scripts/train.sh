#!/usr/bin/env bash
# Confounder audit + quad-model TC-DANN training. Writes bundles to models/.
# Requires Bridge2AI-Voice v3.0.0 (PhysioNet credentialed access).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${B2AI_DATA_ROOT:?Set B2AI_DATA_ROOT to your Bridge2AI-Voice 3.0.0 folder (features/ + phenotype/)}"
EPOCHS="${EPOCHS:-40}"
OUT_DIR="${TC_DANN_BUNDLE_DIR:-models}"
python audit/confounder_analysis.py
python tcdann/run_tc_dann.py --data_root "$B2AI_DATA_ROOT" --epochs "$EPOCHS" --out_dir "$OUT_DIR" "$@"
echo "Bundles written to $OUT_DIR"
