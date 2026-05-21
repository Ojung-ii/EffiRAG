#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ojungii/EffiRAG"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
LIMIT="${LIMIT:-5}"
OUT_ROOT="${OUT_ROOT:-outputs/phase7_evidence_flow/smoke}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/logs"

echo "[phase7-smoke] compileall"
PYTHONPATH=. "${PYTHON}" -m compileall effirag scripts >/dev/null

echo "[phase7-smoke] audit"
PYTHONPATH=. "${PYTHON}" scripts/audit_phase7_method.py \
  2>&1 | tee "${OUT_ROOT}/logs/audit_phase7_method.log"

run_one() {
  local dataset="$1"
  local config="configs/main_config/phase7_evidence_flow/${dataset}.yaml"
  local out_dir="${OUT_ROOT}/${dataset}"
  mkdir -p "${out_dir}"
  echo "[phase7-smoke] dataset=${dataset} limit=${LIMIT} out=${out_dir}"
  PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
    --config "${config}" \
    --limit "${LIMIT}" \
    --output-dir "${out_dir}" \
    --timestamp-output true \
    --phase7-enable-phase2-refinement false \
    --phase7-objective-mode normalized_equal_weight \
    --phase7-lambda-bridge 1.0 \
    --phase7-mu-redundancy 1.0 \
    --phase7-diagnostics-enabled true \
    --phase7-diagnostics-max-examples-to-dump "${LIMIT}" \
    --phase7-diagnostics-fail-on-unit-mismatch true \
    2>&1 | tee "${OUT_ROOT}/logs/${dataset}.log"
}

run_one hotpotqa
run_one 2wikimultihopqa

PYTHONPATH=. "${PYTHON}" scripts/summarize_phase7_results.py \
  --output-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize.log"

echo "[phase7-smoke] done: ${OUT_ROOT}"
