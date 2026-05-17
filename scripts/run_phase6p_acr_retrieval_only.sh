#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
GPU="${GPU:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6p_acr_retrieval_only}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6P ACR retrieval-only smoke ==="
echo "GPU=${GPU}"
echo "DATASETS=${DATASETS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"

run_unified () {
  local profile="$1"
  local tag="$2"

  echo "=== unified profile: ${profile} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --retrieval-only \
    --output-root "${OUT_ROOT}/runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_${tag}.log"
}

echo "=== legacy_sota ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_sota_config_4ds.py \
  --mode locked_precomputed \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/legacy_sota" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_legacy_sota.log"

run_unified unified_large unified_large
run_unified unified_gl_rcedr_v1 unified_gl_rcedr_v1
run_unified unified_gl_rcedr_v1_adaptive_support_span_no_bridge adaptive_no_bridge
run_unified unified_acr_v1 unified_acr_v1
run_unified unified_acr_v1_no_answerability unified_acr_v1_no_answerability
run_unified unified_acr_v1_no_structure unified_acr_v1_no_structure
run_unified unified_acr_v1_no_noise_penalty unified_acr_v1_no_noise_penalty
run_unified unified_acr_v1_no_cost unified_acr_v1_no_cost
run_unified unified_acr_v1_no_ordering unified_acr_v1_no_ordering
run_unified unified_acr_v1_beam3 unified_acr_v1_beam3

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
