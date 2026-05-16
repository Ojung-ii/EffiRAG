#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
GPU="${GPU:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6m_adaptive_support_span}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6M Adaptive Support Span retrieval-only run ==="
echo "GPU=${GPU}"
echo "DATASETS=${DATASETS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"

run_unified () {
  local profile="$1"
  local tag="$2"

  echo "=== unified profile: ${profile} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --retrieval-only \
    --output-root "${OUT_ROOT}/runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_${tag}.log"
}

echo "=== 1. legacy_sota ==="
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
  scripts/run_sota_config_4ds.py \
  --mode locked_precomputed \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/legacy_sota" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_legacy_sota.log"

run_unified unified_large unified_large
run_unified unified_gl_rcedr_v1 unified_gl_rcedr_v1
run_unified unified_gl_rcedr_v1_sentence_contract unified_gl_rcedr_v1_sentence_contract
run_unified unified_gl_rcedr_v1_support_span_contract unified_gl_rcedr_v1_support_span_contract
run_unified unified_gl_rcedr_v1_support_span_contract_no_cap unified_gl_rcedr_v1_support_span_contract_no_cap
run_unified unified_gl_rcedr_v1_adaptive_support_span unified_gl_rcedr_v1_adaptive_support_span
run_unified unified_gl_rcedr_v1_adaptive_support_span_no_bridge unified_gl_rcedr_v1_adaptive_support_span_no_bridge
run_unified unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty
run_unified unified_gl_rcedr_v1_adaptive_support_span_span40 unified_gl_rcedr_v1_adaptive_support_span_span40

echo "=== evidence-flow audit ==="
PYTHONPATH=. "${PYTHON}" scripts/audit_evidence_flow.py \
  --datasets ${DATASETS} \
  --profiles \
    legacy_sota \
    unified_large \
    unified_gl_rcedr_v1 \
    unified_gl_rcedr_v1_sentence_contract \
    unified_gl_rcedr_v1_support_span_contract \
    unified_gl_rcedr_v1_support_span_contract_no_cap \
    unified_gl_rcedr_v1_adaptive_support_span \
    unified_gl_rcedr_v1_adaptive_support_span_no_bridge \
    unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty \
    unified_gl_rcedr_v1_adaptive_support_span_span40 \
  --roots \
    legacy_sota="${OUT_ROOT}/runs/legacy_sota" \
    unified_large="${OUT_ROOT}/runs/unified_large" \
    unified_gl_rcedr_v1="${OUT_ROOT}/runs/unified_gl_rcedr_v1" \
    unified_gl_rcedr_v1_sentence_contract="${OUT_ROOT}/runs/unified_gl_rcedr_v1_sentence_contract" \
    unified_gl_rcedr_v1_support_span_contract="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract" \
    unified_gl_rcedr_v1_support_span_contract_no_cap="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract_no_cap" \
    unified_gl_rcedr_v1_adaptive_support_span="${OUT_ROOT}/runs/unified_gl_rcedr_v1_adaptive_support_span" \
    unified_gl_rcedr_v1_adaptive_support_span_no_bridge="${OUT_ROOT}/runs/unified_gl_rcedr_v1_adaptive_support_span_no_bridge" \
    unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty="${OUT_ROOT}/runs/unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty" \
    unified_gl_rcedr_v1_adaptive_support_span_span40="${OUT_ROOT}/runs/unified_gl_rcedr_v1_adaptive_support_span_span40" \
  --output-dir "${OUT_ROOT}/audit" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_evidence_flow.log"

echo "=== champion bottleneck audit if available ==="
if [ -f scripts/audit_champion_bottlenecks.py ]; then
  PYTHONPATH=. "${PYTHON}" scripts/audit_champion_bottlenecks.py \
    --datasets ${DATASETS} \
    --profiles \
      legacy_sota \
      unified_large \
      unified_gl_rcedr_v1 \
      unified_gl_rcedr_v1_sentence_contract \
      unified_gl_rcedr_v1_support_span_contract \
      unified_gl_rcedr_v1_support_span_contract_no_cap \
      unified_gl_rcedr_v1_adaptive_support_span \
      unified_gl_rcedr_v1_adaptive_support_span_no_bridge \
      unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty \
      unified_gl_rcedr_v1_adaptive_support_span_span40 \
    --roots \
      legacy_sota="${OUT_ROOT}/runs/legacy_sota" \
      unified_large="${OUT_ROOT}/runs/unified_large" \
      unified_gl_rcedr_v1="${OUT_ROOT}/runs/unified_gl_rcedr_v1" \
      unified_gl_rcedr_v1_sentence_contract="${OUT_ROOT}/runs/unified_gl_rcedr_v1_sentence_contract" \
      unified_gl_rcedr_v1_support_span_contract="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract" \
      unified_gl_rcedr_v1_support_span_contract_no_cap="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract_no_cap" \
      unified_gl_rcedr_v1_adaptive_support_span="${OUT_ROOT}/runs/unified_gl_rcedr_v1_adaptive_support_span" \
      unified_gl_rcedr_v1_adaptive_support_span_no_bridge="${OUT_ROOT}/runs/unified_gl_rcedr_v1_adaptive_support_span_no_bridge" \
      unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty="${OUT_ROOT}/runs/unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty" \
      unified_gl_rcedr_v1_adaptive_support_span_span40="${OUT_ROOT}/runs/unified_gl_rcedr_v1_adaptive_support_span_span40" \
    --phase-label "Phase-6M" \
    --audit-title "Adaptive Support Span Contract Audit" \
    --purpose-text "Evaluate adaptive support span rendering with soft length penalty after retrieval-only runs." \
    --summary-md-name "PHASE6M_ADAPTIVE_SUPPORT_SPAN_SUMMARY.md" \
    --output-dir "${OUT_ROOT}/champion_audit" \
    2>&1 | tee "${OUT_ROOT}/logs/audit_champion_bottlenecks.log"
fi

echo "=== cleanup ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

echo "=== done ==="
echo "Evidence-flow audit expected at: ${OUT_ROOT}/audit"
echo "Champion audit expected at: ${OUT_ROOT}/champion_audit"
