#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ojungii/EffiRAG"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
DATASET="${DATASET:-hotpotqa}"
PROFILE="${PROFILE:-balanced_384_8}"
LIMIT="${LIMIT:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/phase7_evidence_flow/query_intent_graph_experiment}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

if [[ "${DATASET}" != "hotpotqa" && "${DATASET}" != "2wikimultihopqa" ]]; then
  echo "[intent-graph] unsupported DATASET=${DATASET}" >&2
  exit 1
fi
if [[ "${PROFILE}" != "balanced_384_8" && "${PROFILE}" != "legacy_512_10" ]]; then
  echo "[intent-graph] unsupported PROFILE=${PROFILE}" >&2
  exit 1
fi

TOKENS=384
ATOMS=8
if [[ "${PROFILE}" == "legacy_512_10" ]]; then
  TOKENS=512
  ATOMS=10
fi

CFG="configs/main_config/phase7_evidence_flow/${DATASET}.yaml"
if [[ ! -f "${CFG}" ]]; then
  echo "[intent-graph] missing config: ${CFG}" >&2
  exit 1
fi

VARIANTS=(
  "baseline_bq:a_plus_bq_minus_r:false:false:false"
  "intent_p1:a_plus_bq_minus_r:true:true:false"
  "intent_p1_aq:aq_plus_bq_minus_r:true:true:false"
  "intent_p1_aq_rq:aq_plus_bq_minus_rq:true:true:true"
)

for row in "${VARIANTS[@]}"; do
  IFS=":" read -r VARIANT OBJECTIVE QI_ENABLED QI_P1_ENABLED RQ_ENABLED <<<"${row}"
  RUN_DIR="${OUT_ROOT}/${PROFILE}/${VARIANT}/${DATASET}"
  LOG_FILE="${OUT_ROOT}/logs/${PROFILE}__${VARIANT}__${DATASET}.log"
  mkdir -p "${RUN_DIR}"

  echo "[intent-graph] dataset=${DATASET} profile=${PROFILE} variant=${VARIANT} limit=${LIMIT}"
  PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
    --config "${CFG}" \
    --dataset "${DATASET}" \
    --limit "${LIMIT}" \
    --output-dir "${RUN_DIR}" \
    --timestamp-output false \
    --phase7-enable-phase2-refinement false \
    --phase7-objective-mode "${OBJECTIVE}" \
    --phase7-lambda-bridge 1.0 \
    --phase7-lambda-decay 1.0 \
    --phase7-lambda-bq 1.0 \
    --phase7-mu-redundancy 1.0 \
    --phase7-conditional-redundancy-enabled "${RQ_ENABLED}" \
    --phase7-corridor-enabled true \
    --phase7-anchor-decay-enabled false \
    --phase7-query-intent-enabled "${QI_ENABLED}" \
    --phase7-intent-phase1-enabled "${QI_P1_ENABLED}" \
    --phase7-intent-max-relation-candidates 32 \
    --phase7-intent-max-answer-type-candidates 32 \
    --phase7-intent-max-entity-candidates 32 \
    --phase7-intent-candidate-top-m 128 \
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
    >"${OUT_ROOT}/logs/${PROFILE}__${VARIANT}__${DATASET}__oracle.log" 2>&1

  PYTHONPATH=. "${PYTHON}" scripts/analyze_phase7_oracle_gap.py \
    --root "${RUN_DIR}" \
    --out "${RUN_DIR}/phase7_failure_attribution_report.md" \
    >"${OUT_ROOT}/logs/${PROFILE}__${VARIANT}__${DATASET}__attribution.log" 2>&1
done

echo "[intent-graph] done dataset=${DATASET} profile=${PROFILE}"
