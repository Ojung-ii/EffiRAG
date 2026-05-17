#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
GPU="${GPU:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6p_acr_qa_smoke}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
RETRIEVAL_AUDIT_ROOT="${RETRIEVAL_AUDIT_ROOT:-outputs/phase6p_acr_retrieval_only/audit}"
SUMMARY_OUT_ROOT="${SUMMARY_OUT_ROOT:-${OUT_ROOT}/summary}"

mkdir -p "${OUT_ROOT}/logs"

run_unified_qa () {
  local profile="$1"
  local tag="$2"

  echo "=== QA smoke unified profile: ${profile} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${tag}.log"
}

echo "=== legacy_sota ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_sota_config_4ds.py \
  --mode locked_precomputed \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --output-root "${OUT_ROOT}/qa_runs/legacy_sota" \
  2>&1 | tee "${OUT_ROOT}/logs/qa_legacy_sota.log"

run_unified_qa unified_large unified_large
run_unified_qa unified_gl_rcedr_v1 unified_gl_rcedr_v1
run_unified_qa unified_gl_rcedr_v1_adaptive_support_span_no_bridge adaptive_no_bridge
run_unified_qa unified_acr_v1 unified_acr_v1
run_unified_qa unified_acr_v1_no_answerability unified_acr_v1_no_answerability
run_unified_qa unified_acr_v1_no_structure unified_acr_v1_no_structure
run_unified_qa unified_acr_v1_no_noise_penalty unified_acr_v1_no_noise_penalty
run_unified_qa unified_acr_v1_no_cost unified_acr_v1_no_cost
run_unified_qa unified_acr_v1_no_ordering unified_acr_v1_no_ordering
run_unified_qa unified_acr_v1_beam3 unified_acr_v1_beam3

echo "=== phase6p summary ==="
if [ -f scripts/summarize_phase6p_acr_results.py ]; then
  PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6p_acr_results.py \
    --datasets ${DATASETS} \
    --qa-root "${OUT_ROOT}/qa_runs" \
    --retrieval-audit-root "${RETRIEVAL_AUDIT_ROOT}" \
    --output-dir "${SUMMARY_OUT_ROOT}" \
    --summary-md-name "PHASE6P_ACR_SUMMARY.md" \
    2>&1 | tee "${OUT_ROOT}/logs/summary_phase6p.log"
else
  echo "WARNING: scripts/summarize_phase6p_acr_results.py not found; skipping summary generation."
fi

echo "=== cleanup ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

echo "=== done ==="
echo "Summary: ${SUMMARY_OUT_ROOT}/PHASE6P_ACR_SUMMARY.md"
