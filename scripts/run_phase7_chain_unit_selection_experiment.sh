#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ojungii/EffiRAG}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
DATASET="${DATASET:-hotpotqa}"
PROFILE="${PROFILE:-legacy_512_10}"
LIMIT="${LIMIT:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/phase7_evidence_flow/chain_unit_selection_experiment}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

if [[ "${DATASET}" != "hotpotqa" && "${DATASET}" != "2wikimultihopqa" ]]; then
  echo "[chain-unit] unsupported DATASET=${DATASET}" >&2
  exit 1
fi
if [[ "${PROFILE}" != "legacy_512_10" && "${PROFILE}" != "balanced_384_8" ]]; then
  echo "[chain-unit] unsupported PROFILE=${PROFILE}" >&2
  exit 1
fi

case "${PROFILE}" in
  legacy_512_10)
    TOKENS=512
    ATOMS=10
    ;;
  balanced_384_8)
    TOKENS=384
    ATOMS=8
    ;;
esac

CFG="configs/main_config/phase7_evidence_flow/${DATASET}.yaml"
if [[ ! -f "${CFG}" ]]; then
  echo "[chain-unit] missing config: ${CFG}" >&2
  exit 1
fi

VARIANTS=(
  "baseline_bq:false:96:0:0:0:0:false:a_plus_bq_minus_r"
  "source_balanced_128:true:128:48:32:32:16:false:a_plus_bq_minus_r"
  "source_balanced_160:true:160:60:40:40:20:false:a_plus_bq_minus_r"
  "chain_unit_selection:true:128:48:32:32:16:true:chain_unit"
)

VARIANTS_FILTER="${VARIANTS_FILTER:-}"
if [[ -n "${VARIANTS_FILTER}" ]]; then
  IFS="," read -r -a WANTED_VARIANTS <<<"${VARIANTS_FILTER}"
  FILTERED_VARIANTS=()
  for row in "${VARIANTS[@]}"; do
    IFS=":" read -r ROW_VARIANT _rest <<<"${row}"
    for wanted in "${WANTED_VARIANTS[@]}"; do
      wanted="$(echo "${wanted}" | xargs)"
      if [[ "${ROW_VARIANT}" == "${wanted}" ]]; then
        FILTERED_VARIANTS+=("${row}")
      fi
    done
  done
  if [[ "${#FILTERED_VARIANTS[@]}" -eq 0 ]]; then
    echo "[chain-unit] no matching variants for VARIANTS_FILTER=${VARIANTS_FILTER}" >&2
    exit 1
  fi
  VARIANTS=("${FILTERED_VARIANTS[@]}")
fi

for row in "${VARIANTS[@]}"; do
  IFS=":" read -r VARIANT SB_ENABLED TOP_M Q_SEM Q_ENTITY Q_FLOW Q_NBR CU_ENABLED OBJECTIVE <<<"${row}"
  RUN_DIR="${OUT_ROOT}/${PROFILE}/${VARIANT}/${DATASET}"
  LOG_FILE="${OUT_ROOT}/logs/${PROFILE}__${VARIANT}__${DATASET}.log"
  mkdir -p "${RUN_DIR}"

  echo "[chain-unit] dataset=${DATASET} profile=${PROFILE} variant=${VARIANT} limit=${LIMIT}"
  PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
    --config "${CFG}" \
    --dataset "${DATASET}" \
    --limit "${LIMIT}" \
    --output-dir "${RUN_DIR}" \
    --timestamp-output false \
    --method phase7_evidence_flow \
    --phase7-enable-phase2-refinement false \
    --phase7-objective-mode "${OBJECTIVE}" \
    --phase7-lambda-bridge 1.0 \
    --phase7-lambda-decay 1.0 \
    --phase7-lambda-bq 1.0 \
    --phase7-mu-redundancy 1.0 \
    --phase7-conditional-redundancy-enabled false \
    --phase7-corridor-enabled true \
    --phase7-anchor-decay-enabled false \
    --phase7-query-intent-enabled false \
    --phase7-intent-phase1-enabled false \
    --phase7-source-balanced-proposal-enabled "${SB_ENABLED}" \
    --phase7-source-balanced-candidate-top-m "${TOP_M}" \
    --phase7-source-balanced-fill-remaining true \
    --phase7-source-quota-semantic "${Q_SEM}" \
    --phase7-source-quota-entity-title "${Q_ENTITY}" \
    --phase7-source-quota-graph-flow "${Q_FLOW}" \
    --phase7-source-quota-anchor-neighborhood "${Q_NBR}" \
    --phase7-chain-unit-enabled "${CU_ENABLED}" \
    --phase7-chain-unit-max-pair-units 256 \
    --phase7-chain-unit-use-same-title true \
    --phase7-chain-unit-use-explicit-transition true \
    --phase7-chain-unit-use-same-carrier true \
    --phase7-chain-unit-use-shared-entity false \
    --phase7-chain-unit-max-unit-size 2 \
    --phase7-candidate-top-m "${TOP_M}" \
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

echo "[chain-unit] done dataset=${DATASET} profile=${PROFILE}"
