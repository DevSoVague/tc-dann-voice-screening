#!/usr/bin/env bash
# Start a Streamlit front end against a running API.
#   scripts/run_app.sh            TC-DANN app (default)
#   scripts/run_app.sh voxclin    5-stage VoxClinBench app with agentic chat + RAG pages
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API="${TC_DANN_API:-http://localhost:${API_PORT:-8000}}"
export TC_DANN_API="$API"
export VOXCLIN_API="${VOXCLIN_API:-$API}"
export SESSIONS_DIR="${SESSIONS_DIR:-$ROOT/sessions}"
cd "$ROOT/tcdann"
case "${1:-tcdann}" in
  tcdann)  APP=tc_dann_app.py ;;
  voxclin) APP=voxclinbench_app.py ;;
  *) echo "usage: $0 [tcdann|voxclin]" >&2; exit 2 ;;
esac
exec streamlit run "$APP" --server.port "${PORT:-8501}" --server.headless "${HEADLESS:-false}"
