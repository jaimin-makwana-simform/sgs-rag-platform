#!/usr/bin/env bash
# Launch the FastAPI voice backend + the Streamlit UI together.
# The backend powers the "Streaming (live)" voice output; Streamlit is the UI.
# Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${VOICE_BACKEND_HOST:-localhost}"
PORT="${VOICE_BACKEND_PORT:-8000}"

echo "Starting voice backend on ${HOST}:${PORT} …"
.venv/bin/uvicorn server:app --host "$HOST" --port "$PORT" &
BACKEND_PID=$!

cleanup() { echo; echo "Stopping voice backend…"; kill "$BACKEND_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Wait for the backend to report healthy before launching the UI.
for _ in $(seq 1 30); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "Backend healthy."
    break
  fi
  sleep 1
done

echo "Starting Streamlit UI…"
.venv/bin/streamlit run app.py
