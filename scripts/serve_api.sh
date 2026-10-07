#!/usr/bin/env bash
# Start the TC-DANN FastAPI server. Bundles are read from $TC_DANN_BUNDLE_DIR (default: models/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TC_DANN_BUNDLE_DIR="${TC_DANN_BUNDLE_DIR:-$ROOT/models}"
export SESSIONS_DIR="${SESSIONS_DIR:-$ROOT/sessions}"
cd "$ROOT/tcdann"
exec uvicorn tc_dann_api_server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
