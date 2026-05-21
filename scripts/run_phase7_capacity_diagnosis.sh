#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ojungii/EffiRAG"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
DATASET="${DATASET:-hotpotqa}"
LIMIT="${LIMIT:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/phase7_evidence_flow/capacity_diagnosis}"
OBJECTIVE_MODE="${OBJECTIVE_MODE:-a_plus_bq_minus_r}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/${DATASET}" "${OUT_ROOT}/logs"

if [[ "${DATASET}" != "hotpotqa" && "${DATASET}" != "2wikimultihopqa" ]]; then
  echo "[capacity] unsupported DATASET=${DATASET}" >&2
  exit 1
fi

CFG="configs/main_config/phase7_evidence_flow/${DATASET}.yaml"
if [[ ! -f "${CFG}" ]]; then
  echo "[capacity] missing config: ${CFG}" >&2
  exit 1
fi

PROFILES=(
  "balanced_384_8:384:8"
  "legacy_512_10:512:10"
  "capacity_512_16:512:16"
  "capacity_768_16:768:16"
)

for row in "${PROFILES[@]}"; do
  IFS=":" read -r PROFILE TOKENS ATOMS <<<"${row}"
  RUN_DIR="${OUT_ROOT}/${DATASET}/${PROFILE}"
  LOG_FILE="${OUT_ROOT}/logs/${DATASET}__${PROFILE}.log"
  mkdir -p "${RUN_DIR}"

  echo "[capacity] dataset=${DATASET} profile=${PROFILE} limit=${LIMIT} out=${RUN_DIR}"
  PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
    --config "${CFG}" \
    --dataset "${DATASET}" \
    --limit "${LIMIT}" \
    --output-dir "${RUN_DIR}" \
    --timestamp-output false \
    --phase7-enable-phase2-refinement false \
    --phase7-objective-mode "${OBJECTIVE_MODE}" \
    --phase7-lambda-bridge 1.0 \
    --phase7-lambda-decay 1.0 \
    --phase7-lambda-bq 1.0 \
    --phase7-mu-redundancy 1.0 \
    --phase7-conditional-redundancy-enabled false \
    --phase7-corridor-enabled true \
    --phase7-anchor-decay-enabled false \
    --phase7-query-intent-enabled false \
    --phase7-intent-phase1-enabled false \
    --phase7-candidate-top-m 96 \
    --phase7-max-context-tokens "${TOKENS}" \
    --phase7-max-selected-atoms "${ATOMS}" \
    --phase7-diagnostics-enabled true \
    --phase7-diagnostics-max-examples-to-dump "${LIMIT}" \
    --phase7-diagnostics-dump-text true \
    --phase7-diagnostics-dump-context true \
    --phase7-diagnostics-dump-scores true \
    --phase7-diagnostics-fail-on-unit-mismatch true \
    >"${LOG_FILE}" 2>&1

  PYTHONPATH=. "${PYTHON}" scripts/run_phase7_oracle_context_qa.py \
    --root "${RUN_DIR}" \
    --out "${RUN_DIR}/oracle_context_report.md" \
    --datasets "${DATASET}" \
    --limit "${LIMIT}" \
    --context-source-roots "${RUN_DIR}" \
    >"${OUT_ROOT}/logs/${DATASET}__${PROFILE}__oracle.log" 2>&1
done

echo "[capacity] done dataset=${DATASET}"
