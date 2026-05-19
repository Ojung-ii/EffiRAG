#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6v_semantic_sufficiency_n200}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
PROCESS_COUNT="${PROCESS_COUNT:-2}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"

# Delegate execution to the existing paired semantic-sufficiency runner.
OUT_ROOT="${OUT_ROOT}" \
SAMPLE_SIZE="${SAMPLE_SIZE}" \
PROCESS_COUNT="${PROCESS_COUNT}" \
GPU_A="${GPU_A}" \
GPU_B="${GPU_B}" \
STRICT_VALIDATE="${STRICT_VALIDATE}" \
PYTHON="${PYTHON}" \
BASELINE_PROFILE="unified_acr_rcedr_v12" \
VARIANT_PROFILE="unified_acr_rcedr_v12_semantic_sufficiency" \
bash scripts/run_phase6v_semantic_sufficiency_abr_2proc.sh

# Normalize summary artifact names for PHASE6V n200 report contract.
if [[ -f "${OUT_ROOT}/PHASE6V_SEMANTIC_SUFFICIENCY_ABR_SUMMARY.md" ]]; then
  cp "${OUT_ROOT}/PHASE6V_SEMANTIC_SUFFICIENCY_ABR_SUMMARY.md" \
     "${OUT_ROOT}/PHASE6V_SEMANTIC_SUFFICIENCY_N200_SUMMARY.md"
fi
if [[ -f "${OUT_ROOT}/phase6v_semantic_sufficiency_abr_summary.json" ]]; then
  cp "${OUT_ROOT}/phase6v_semantic_sufficiency_abr_summary.json" \
     "${OUT_ROOT}/phase6v_semantic_sufficiency_n200_summary.json"
fi

echo "=== done ==="
echo "Summary: ${OUT_ROOT}/PHASE6V_SEMANTIC_SUFFICIENCY_N200_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6v_semantic_sufficiency_n200_summary.json"
