#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6p_acr_retrieval_only_4way}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"

# Requested placement: one process on GPU0, remaining three on GPU1.
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-1}"
GPU_D="${GPU_D:-1}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6P ACR retrieval-only (4-way) ==="
echo "DATASETS=${DATASETS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GPU_A=${GPU_A} GPU_B=${GPU_B} GPU_C=${GPU_C} GPU_D=${GPU_D}"

run_unified () {
  local gpu="$1"
  local profile="$2"
  local tag="$3"
  echo "=== [GPU ${gpu}] retrieval-only unified profile: ${profile} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --retrieval-only \
    --output-root "${OUT_ROOT}/runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_${tag}.log"
}

run_legacy () {
  local gpu="$1"
  local tag="$2"
  echo "=== [GPU ${gpu}] retrieval-only legacy_sota ==="
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_sota_config_4ds.py \
    --mode locked_precomputed \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --retrieval-only \
    --output-root "${OUT_ROOT}/runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_${tag}.log"
}

# Balanced profile mixing:
# - Group A (GPU0): legacy + beam-search ablation (typically heavy)
# - Group B/C/D (GPU1): baseline and ACR variants mixed to avoid a long-tail worker
group_a () {
  run_legacy "${GPU_A}" legacy_sota
  run_unified "${GPU_A}" unified_acr_v1_beam3 unified_acr_v1_beam3
}

group_b () {
  run_unified "${GPU_B}" unified_large unified_large
  run_unified "${GPU_B}" unified_acr_v1_no_answerability unified_acr_v1_no_answerability
  run_unified "${GPU_B}" unified_acr_v1_no_cost unified_acr_v1_no_cost
}

group_c () {
  run_unified "${GPU_C}" unified_gl_rcedr_v1 unified_gl_rcedr_v1
  run_unified "${GPU_C}" unified_gl_rcedr_v1_adaptive_support_span_no_bridge adaptive_no_bridge
  run_unified "${GPU_C}" unified_acr_v1_no_ordering unified_acr_v1_no_ordering
}

group_d () {
  run_unified "${GPU_D}" unified_acr_v1 unified_acr_v1
  run_unified "${GPU_D}" unified_acr_v1_no_structure unified_acr_v1_no_structure
  run_unified "${GPU_D}" unified_acr_v1_no_noise_penalty unified_acr_v1_no_noise_penalty
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
  echo "[ERROR] One or more retrieval workers failed. Check ${OUT_ROOT}/logs/group_*.log" >&2
  exit 1
fi

echo "=== evidence-flow audit ==="
PYTHONPATH=. "${PYTHON}" scripts/audit_evidence_flow.py \
  --datasets ${DATASETS} \
  --profiles \
    legacy_sota \
    unified_large \
    unified_gl_rcedr_v1 \
    unified_gl_rcedr_v1_adaptive_support_span_no_bridge \
    unified_acr_v1 \
    unified_acr_v1_no_answerability \
    unified_acr_v1_no_structure \
    unified_acr_v1_no_noise_penalty \
    unified_acr_v1_no_cost \
    unified_acr_v1_no_ordering \
    unified_acr_v1_beam3 \
  --roots \
    legacy_sota="${OUT_ROOT}/runs/legacy_sota" \
    unified_large="${OUT_ROOT}/runs/unified_large" \
    unified_gl_rcedr_v1="${OUT_ROOT}/runs/unified_gl_rcedr_v1" \
    unified_gl_rcedr_v1_adaptive_support_span_no_bridge="${OUT_ROOT}/runs/adaptive_no_bridge" \
    unified_acr_v1="${OUT_ROOT}/runs/unified_acr_v1" \
    unified_acr_v1_no_answerability="${OUT_ROOT}/runs/unified_acr_v1_no_answerability" \
    unified_acr_v1_no_structure="${OUT_ROOT}/runs/unified_acr_v1_no_structure" \
    unified_acr_v1_no_noise_penalty="${OUT_ROOT}/runs/unified_acr_v1_no_noise_penalty" \
    unified_acr_v1_no_cost="${OUT_ROOT}/runs/unified_acr_v1_no_cost" \
    unified_acr_v1_no_ordering="${OUT_ROOT}/runs/unified_acr_v1_no_ordering" \
    unified_acr_v1_beam3="${OUT_ROOT}/runs/unified_acr_v1_beam3" \
  --output-dir "${OUT_ROOT}/audit" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_evidence_flow.log"

echo "=== cleanup ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

echo "=== done ==="
echo "Runs: ${OUT_ROOT}/runs"
echo "Audit: ${OUT_ROOT}/audit"
