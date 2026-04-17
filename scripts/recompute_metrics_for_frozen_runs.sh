#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-outputs/profiling/eval_parity}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/frozen_runs_recomputed_summary.json}"
OUT_MD="${OUT_MD:-${OUT_DIR}/frozen_runs_recomputed_summary.md}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${OUT_DIR}"

(
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" scripts/recompute_metrics_with_parity.py \
    --selection frozen \
    --output-json "${OUT_JSON}" \
    --output-md "${OUT_MD}" \
    --question-being-answered "Parity-adjusted EM/F1 for frozen operating-point runs only"
)

echo "[DONE] Frozen runs recomputed:"
echo "- ${OUT_MD}"
echo "- ${OUT_JSON}"
