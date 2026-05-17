#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6q_acr_v11_qa_4way}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
RETRIEVAL_AUDIT_ROOT="${RETRIEVAL_AUDIT_ROOT:-outputs/phase6m_adaptive_support_span_n100/audit}"

# Required placement: one process on GPU0, remaining three on GPU1.
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-1}"
GPU_D="${GPU_D:-1}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6Q ACR v1.1 QA smoke (4-way) ==="
echo "DATASETS=${DATASETS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GPU_A=${GPU_A} GPU_B=${GPU_B} GPU_C=${GPU_C} GPU_D=${GPU_D}"
echo "RETRIEVAL_AUDIT_ROOT=${RETRIEVAL_AUDIT_ROOT}"

run_unified_qa () {
  local gpu="$1"
  local profile="$2"
  local tag="$3"
  echo "=== [GPU ${gpu}] QA smoke unified profile: ${profile} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${tag}.log"
}

run_legacy_qa () {
  local gpu="$1"
  local tag="$2"
  echo "=== [GPU ${gpu}] QA smoke legacy_sota ==="
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_sota_config_4ds.py \
    --mode locked_precomputed \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${tag}.log"
}

# Process A / GPU0: heavy + medium
group_a () {
  run_legacy_qa "${GPU_A}" legacy_sota
  run_unified_qa "${GPU_A}" unified_acr_v11 unified_acr_v11
}

# Process B / GPU1: light + heavy + light
group_b () {
  run_unified_qa "${GPU_B}" unified_large unified_large
  run_unified_qa "${GPU_B}" unified_acr_v1_beam3 unified_acr_v1_beam3
  run_unified_qa "${GPU_B}" unified_acr_v11_no_cost unified_acr_v11_no_cost
}

# Process C / GPU1: heavy + medium + heavy
group_c () {
  run_unified_qa "${GPU_C}" unified_gl_rcedr_v1_adaptive_support_span_no_bridge adaptive_no_bridge
  run_unified_qa "${GPU_C}" unified_acr_v1 unified_acr_v1
  run_unified_qa "${GPU_C}" unified_acr_v11_beam3 unified_acr_v11_beam3
}

# Process D / GPU1: medium profile bundle
group_d () {
  run_unified_qa "${GPU_D}" unified_acr_v1_no_noise_penalty unified_acr_v1_no_noise_penalty
  run_unified_qa "${GPU_D}" unified_acr_v1_no_ordering unified_acr_v1_no_ordering
  run_unified_qa "${GPU_D}" unified_acr_v11_no_structure unified_acr_v11_no_structure
  run_unified_qa "${GPU_D}" unified_acr_v11_ordered unified_acr_v11_ordered
}

group_a > "${OUT_ROOT}/logs/group_A_gpu0.log" 2>&1 &
PID_A=$!
group_b > "${OUT_ROOT}/logs/group_B_gpu1.log" 2>&1 &
PID_B=$!
group_c > "${OUT_ROOT}/logs/group_C_gpu1.log" 2>&1 &
PID_C=$!
group_d > "${OUT_ROOT}/logs/group_D_gpu1.log" 2>&1 &
PID_D=$!

FAIL=0
for pid in "${PID_A}" "${PID_B}" "${PID_C}" "${PID_D}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] One or more QA workers failed. Check ${OUT_ROOT}/logs/group_*.log" >&2
  exit 1
fi

echo "=== QA Pareto report ==="
if [ -f scripts/compare_qa_pareto_profiles.py ]; then
  PYTHONPATH=. "${PYTHON}" scripts/compare_qa_pareto_profiles.py \
    --datasets ${DATASETS} \
    --profiles \
      legacy_sota \
      unified_large \
      unified_gl_rcedr_v1_adaptive_support_span_no_bridge \
      unified_acr_v1 \
      unified_acr_v1_no_noise_penalty \
      unified_acr_v1_no_ordering \
      unified_acr_v1_beam3 \
      unified_acr_v11 \
      unified_acr_v11_no_structure \
      unified_acr_v11_no_cost \
      unified_acr_v11_ordered \
      unified_acr_v11_beam3 \
    --input-root "${OUT_ROOT}/qa_runs" \
    --retrieval-audit-root "${RETRIEVAL_AUDIT_ROOT}" \
    --output-dir "${OUT_ROOT}/pareto_report" \
    2>&1 | tee "${OUT_ROOT}/logs/compare_qa_pareto.log"
else
  echo "WARNING: scripts/compare_qa_pareto_profiles.py not found; skipping Pareto report."
fi

echo "=== Phase-6Q combined summary ==="
if [ -f scripts/summarize_phase6q_acr_results.py ]; then
  PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6q_acr_results.py \
    --datasets ${DATASETS} \
    --profiles \
      legacy_sota \
      unified_large \
      unified_gl_rcedr_v1_adaptive_support_span_no_bridge \
      unified_acr_v1 \
      unified_acr_v1_no_noise_penalty \
      unified_acr_v1_no_ordering \
      unified_acr_v1_beam3 \
      unified_acr_v11 \
      unified_acr_v11_no_structure \
      unified_acr_v11_no_cost \
      unified_acr_v11_ordered \
      unified_acr_v11_beam3 \
    --qa-root "${OUT_ROOT}/qa_runs" \
    --retrieval-audit-root "${RETRIEVAL_AUDIT_ROOT}" \
    --pareto-root "${OUT_ROOT}/pareto_report" \
    --output "${OUT_ROOT}/PHASE6Q_ACR_V11_SUMMARY.md" \
    2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6q.log"
else
  echo "WARNING: scripts/summarize_phase6q_acr_results.py not found; skipping summary generation."
fi

echo "=== cleanup ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

echo "=== done ==="
echo "QA runs: ${OUT_ROOT}/qa_runs"
echo "Pareto report: ${OUT_ROOT}/pareto_report"
echo "Summary: ${OUT_ROOT}/PHASE6Q_ACR_V11_SUMMARY.md"
