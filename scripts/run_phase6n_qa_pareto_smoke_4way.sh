#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6n_qa_pareto_validation_4way}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"

# Requested placement: one process on GPU0, remaining three on GPU1.
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-1}"
GPU_D="${GPU_D:-1}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6N QA-aware Pareto smoke (4-way) ==="
echo "DATASETS=${DATASETS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GPU_A=${GPU_A} GPU_B=${GPU_B} GPU_C=${GPU_C} GPU_D=${GPU_D}"

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

# Balanced mix: legacy + adaptive main on GPU0, and baseline/ablation mixes on GPU1.
group_a () {
  run_legacy_qa "${GPU_A}" legacy_sota
  run_unified_qa "${GPU_A}" unified_gl_rcedr_v1_adaptive_support_span unified_gl_rcedr_v1_adaptive_support_span
}

group_b () {
  run_unified_qa "${GPU_B}" unified_large unified_large
  run_unified_qa "${GPU_B}" unified_gl_rcedr_v1_adaptive_support_span_no_bridge unified_gl_rcedr_v1_adaptive_support_span_no_bridge
}

group_c () {
  run_unified_qa "${GPU_C}" unified_gl_rcedr_v1 unified_gl_rcedr_v1
  run_unified_qa "${GPU_C}" unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty
}

group_d () {
  run_unified_qa "${GPU_D}" unified_gl_rcedr_v1_support_span_contract unified_gl_rcedr_v1_support_span_contract
  run_unified_qa "${GPU_D}" unified_gl_rcedr_v1_support_span_contract_no_cap unified_gl_rcedr_v1_support_span_contract_no_cap
  run_unified_qa "${GPU_D}" unified_gl_rcedr_v1_adaptive_support_span_span40 unified_gl_rcedr_v1_adaptive_support_span_span40
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
