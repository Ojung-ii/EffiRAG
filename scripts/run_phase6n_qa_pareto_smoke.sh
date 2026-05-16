#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
GPU="${GPU:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6n_qa_pareto_validation}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6N QA-aware Pareto smoke ==="
echo "GPU=${GPU}"
echo "DATASETS=${DATASETS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"

run_unified_qa () {
  local profile="$1"
  local tag="$2"

  echo "=== QA smoke unified profile: ${profile} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${tag}.log"
}

echo "=== 1. legacy_sota QA reference ==="
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
  scripts/run_sota_config_4ds.py \
  --mode locked_precomputed \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --output-root "${OUT_ROOT}/qa_runs/legacy_sota" \
  2>&1 | tee "${OUT_ROOT}/logs/qa_legacy_sota.log"

run_unified_qa unified_large unified_large
run_unified_qa unified_gl_rcedr_v1 unified_gl_rcedr_v1
run_unified_qa unified_gl_rcedr_v1_support_span_contract unified_gl_rcedr_v1_support_span_contract
run_unified_qa unified_gl_rcedr_v1_support_span_contract_no_cap unified_gl_rcedr_v1_support_span_contract_no_cap
run_unified_qa unified_gl_rcedr_v1_adaptive_support_span unified_gl_rcedr_v1_adaptive_support_span
run_unified_qa unified_gl_rcedr_v1_adaptive_support_span_no_bridge unified_gl_rcedr_v1_adaptive_support_span_no_bridge
run_unified_qa unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty
run_unified_qa unified_gl_rcedr_v1_adaptive_support_span_span40 unified_gl_rcedr_v1_adaptive_support_span_span40

echo "=== QA Pareto report ==="
if [ -f scripts/compare_qa_pareto_profiles.py ]; then
  PYTHONPATH=. "${PYTHON}" scripts/compare_qa_pareto_profiles.py \
    --datasets ${DATASETS} \
    --input-root "${OUT_ROOT}/qa_runs" \
    --retrieval-audit-root "${RETRIEVAL_AUDIT_ROOT:-outputs/phase6m_adaptive_support_span_n100_4way/audit}" \
    --output-dir "${OUT_ROOT}/pareto_report" \
    2>&1 | tee "${OUT_ROOT}/logs/compare_qa_pareto.log"
else
  echo "WARNING: scripts/compare_qa_pareto_profiles.py not found; skipping Pareto report."
fi

echo "=== cleanup ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

echo "=== done ==="
echo "QA runs: ${OUT_ROOT}/qa_runs"
echo "Pareto report: ${OUT_ROOT}/pareto_report"
