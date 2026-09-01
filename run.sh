#!/usr/bin/env bash
# Launch the FastAPI backend + the Streamlit UI together.
# The backend powers "Streaming (live)" voice output AND the agent avatar
# (/avatar + /avatar/token); Streamlit is the UI. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${VOICE_BACKEND_HOST:-localhost}"
PORT="${VOICE_BACKEND_PORT:-8000}"

# Read a value from .env without exporting the whole file (tolerates missing file).
env_val() { [ -f .env ] && sed -n "s/^$1=//p" .env | tail -1 | tr -d '"' || true; }
AVATAR_ENABLED="$(env_val AVATAR_ENABLED)"

echo "Starting backend on ${HOST}:${PORT} …"
.venv/bin/uvicorn server:app --host "$HOST" --port "$PORT" &
BACKEND_PID=$!

cleanup() { echo; echo "Stopping backend…"; kill "$BACKEND_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Wait for the backend to report healthy before launching the UI.
for _ in $(seq 1 30); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "Backend healthy."
    break
  fi
  sleep 1
done

echo
echo "  UI            : http://localhost:8501"
echo "  Voice backend : http://${HOST}:${PORT}"
if [ "${AVATAR_ENABLED,,}" = "true" ]; then
  echo "  Agent avatar  : ENABLED — pick 'Custom' mode → Voice output → 🧑 Avatar"
else
  echo "  Agent avatar  : disabled — set AVATAR_ENABLED=true in .env to show the 🧑 Avatar option"
fi
echo

echo "Starting Streamlit UI…"
.venv/bin/streamlit run app.py
