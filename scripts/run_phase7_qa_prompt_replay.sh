#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ojungii/EffiRAG"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
LIMIT="${LIMIT:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/phase7_evidence_flow/qa_prompt_diagnosis}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

echo "[qa-prompt-replay] audit"
PYTHONPATH=. "${PYTHON}" scripts/audit_phase7_method.py \
  2>&1 | tee "${OUT_ROOT}/logs/audit_phase7_method.log"

echo "[qa-prompt-replay] replay start"
PYTHONPATH=. "${PYTHON}" scripts/replay_phase7_qa_with_prompt_variants.py \
  --output-root "${OUT_ROOT}" \
  --limit "${LIMIT}" \
  --datasets "hotpotqa,2wikimultihopqa" \
  --prompt-modes "current_phase7,lightrag_short,phase7_short" \
  2>&1 | tee "${OUT_ROOT}/logs/replay_phase7_qa_with_prompt_variants.log"

echo "[qa-prompt-replay] done"
