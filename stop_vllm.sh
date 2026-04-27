#!/usr/bin/env bash
set -euo pipefail

echo "[watch] wait experiment start..."
while ! pgrep -f "scripts/run_lightweight_context_cueing_round.sh" >/dev/null \
   && ! pgrep -f "scripts/lightweight_context_cueing_round.py" >/dev/null; do
  sleep 5
done

echo "[watch] experiment running..."
while pgrep -f "scripts/run_lightweight_context_cueing_round.sh" >/dev/null \
   || pgrep -f "scripts/lightweight_context_cueing_round.py" >/dev/null; do
  sleep 20
done

echo "[watch] experiment finished. stopping vllm(:8011)..."
PID=$(lsof -t -iTCP:8011 -sTCP:LISTEN 2>/dev/null || true)
if [ -n "${PID:-}" ]; then
  kill "$PID" 2>/dev/null || true
  sleep 2
  kill -9 "$PID" 2>/dev/null || true
else
  pkill -f "vllm serve.*--port 8011" 2>/dev/null || true
fi

echo "[watch] done"
