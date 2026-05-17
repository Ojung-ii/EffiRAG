#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6p_acr_qa_smoke_4way}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
RETRIEVAL_AUDIT_ROOT="${RETRIEVAL_AUDIT_ROOT:-outputs/phase6p_acr_retrieval_only_4way/audit}"
SUMMARY_OUT_ROOT="${SUMMARY_OUT_ROOT:-${OUT_ROOT}/summary}"

# Requested placement: one process on GPU0, remaining three on GPU1.
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-1}"
GPU_D="${GPU_D:-1}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6P ACR QA smoke (4-way) ==="
echo "DATASETS=${DATASETS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GPU_A=${GPU_A} GPU_B=${GPU_B} GPU_C=${GPU_C} GPU_D=${GPU_D}"
echo "RETRIEVAL_AUDIT_ROOT=${RETRIEVAL_AUDIT_ROOT}"
echo "SUMMARY_OUT_ROOT=${SUMMARY_OUT_ROOT}"

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

# Balanced profile mixing:
# - Group A (GPU0): legacy + beam-search ablation (typically heavy)
# - Group B/C/D (GPU1): baseline and ACR variants mixed to avoid a long-tail worker
group_a () {
  run_legacy_qa "${GPU_A}" legacy_sota
  run_unified_qa "${GPU_A}" unified_acr_v1_beam3 unified_acr_v1_beam3
}

group_b () {
  run_unified_qa "${GPU_B}" unified_large unified_large
  run_unified_qa "${GPU_B}" unified_acr_v1_no_answerability unified_acr_v1_no_answerability
  run_unified_qa "${GPU_B}" unified_acr_v1_no_cost unified_acr_v1_no_cost
}

group_c () {
  run_unified_qa "${GPU_C}" unified_gl_rcedr_v1 unified_gl_rcedr_v1
  run_unified_qa "${GPU_C}" unified_gl_rcedr_v1_adaptive_support_span_no_bridge adaptive_no_bridge
  run_unified_qa "${GPU_C}" unified_acr_v1_no_ordering unified_acr_v1_no_ordering
}

group_d () {
  run_unified_qa "${GPU_D}" unified_acr_v1 unified_acr_v1
  run_unified_qa "${GPU_D}" unified_acr_v1_no_structure unified_acr_v1_no_structure
  run_unified_qa "${GPU_D}" unified_acr_v1_no_noise_penalty unified_acr_v1_no_noise_penalty
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
echo "QA runs: ${OUT_ROOT}/qa_runs"
echo "Summary: ${SUMMARY_OUT_ROOT}/PHASE6P_ACR_SUMMARY.md"
