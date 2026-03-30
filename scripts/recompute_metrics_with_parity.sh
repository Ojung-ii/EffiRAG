#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
SELECTION="${SELECTION:-all}"  # all | frozen | key_variants
OUT_DIR="${OUT_DIR:-outputs/profiling/eval_parity}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/recomputed_${SELECTION}_summary.json}"
OUT_MD="${OUT_MD:-${OUT_DIR}/recomputed_${SELECTION}_summary.md}"

mkdir -p "${OUT_DIR}"

(
  cd /home/ojungii/EffiRAG
  "${PYTHON_BIN}" scripts/recompute_metrics_with_parity.py \
    --selection "${SELECTION}" \
    --output-json "${OUT_JSON}" \
    --output-md "${OUT_MD}"
)

echo "[DONE] Recomputed metrics with parity:"
echo "- ${OUT_MD}"
echo "- ${OUT_JSON}"
