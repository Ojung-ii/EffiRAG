#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6o_qa_aware_bottleneck_audit}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"

RETRIEVAL_AUDIT_ROOT="${RETRIEVAL_AUDIT_ROOT:-outputs/phase6m_adaptive_support_span_n100/audit}"
QA_ROOT="${QA_ROOT:-outputs/phase6n_qa_pareto_validation_4way/qa_runs}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6O Method Complexity Audit ==="
PYTHONPATH=. "${PYTHON}" scripts/audit_method_complexity.py \
  --config-root configs/main_config/copy_span_instruction_unified \
  --code-roots effirag scripts \
  --output-dir "${OUT_ROOT}/method_complexity" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_method_complexity.log"

echo "=== Phase-6O QA-aware Bottleneck Audit ==="
PYTHONPATH=. "${PYTHON}" scripts/audit_qa_aware_bottlenecks.py \
  --datasets ${DATASETS} \
  --qa-root "${QA_ROOT}" \
  --retrieval-audit-root "${RETRIEVAL_AUDIT_ROOT}" \
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
  --method-complexity-dir "${OUT_ROOT}/method_complexity" \
  --output-dir "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_qa_aware_bottlenecks.log"

echo "=== cleanup ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

echo "=== done ==="
echo "Report expected at: ${OUT_ROOT}/PHASE6O_QA_AWARE_BOTTLENECK_AUDIT.md"
