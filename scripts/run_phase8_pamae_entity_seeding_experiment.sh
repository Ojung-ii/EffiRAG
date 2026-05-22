#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ojungii/EffiRAG}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
DATASET="${DATASET:-hotpotqa}"
PROFILE="${PROFILE:-legacy_512_10}"
LIMIT="${LIMIT:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/phase8_pamae_entity_seeding}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

if [[ "${DATASET}" != "hotpotqa" && "${DATASET}" != "2wikimultihopqa" ]]; then
  echo "[phase8] unsupported DATASET=${DATASET}" >&2
  exit 1
fi
if [[ "${PROFILE}" != "legacy_512_10" && "${PROFILE}" != "balanced_384_8" ]]; then
  echo "[phase8] unsupported PROFILE=${PROFILE}" >&2
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
  echo "[phase8] missing config: ${CFG}" >&2
  exit 1
fi

VARIANTS=(
  "source_balanced_128:false:false:true:128"
  "pamae_seed_k5:true:false:false:160"
  "pamae_seed_k5_refine:true:true:false:160"
)

for row in "${VARIANTS[@]}"; do
  IFS=":" read -r VARIANT P8_ENABLED REFINE_ENABLED SB_ENABLED TOP_M <<<"${row}"
  RUN_DIR="${OUT_ROOT}/${PROFILE}/${VARIANT}/${DATASET}"
  LOG_FILE="${OUT_ROOT}/logs/${PROFILE}__${VARIANT}__${DATASET}.log"
  mkdir -p "${RUN_DIR}"

  echo "[phase8] dataset=${DATASET} profile=${PROFILE} variant=${VARIANT} limit=${LIMIT}"
  PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
    --config "${CFG}" \
    --dataset "${DATASET}" \
    --limit "${LIMIT}" \
    --output-dir "${RUN_DIR}" \
    --timestamp-output false \
    --method phase7_evidence_flow \
    --phase7-enable-phase2-refinement false \
    --phase7-objective-mode a_plus_bq_minus_r \
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
    --phase7-source-quota-semantic 48 \
    --phase7-source-quota-entity-title 32 \
    --phase7-source-quota-graph-flow 32 \
    --phase7-source-quota-anchor-neighborhood 16 \
    --phase7-chain-unit-enabled false \
    --phase7-candidate-top-m "${TOP_M}" \
    --phase7-max-context-tokens "${TOKENS}" \
    --phase7-max-selected-atoms "${ATOMS}" \
    --phase7-diagnostics-enabled true \
    --phase7-diagnostics-max-examples-to-dump "${LIMIT}" \
    --phase7-diagnostics-dump-text true \
    --phase7-diagnostics-dump-context true \
    --phase7-diagnostics-dump-scores true \
    --phase7-diagnostics-fail-on-unit-mismatch false \
    --phase8-pamae-enabled "${P8_ENABLED}" \
    --phase8-pamae-proposal-mode entity_seed_refine \
    --phase8-entity-universe-top-n 2000 \
    --phase8-entity-universe-min-n 200 \
    --phase8-entity-degree-cap 200 \
    --phase8-entity-source-semantic true \
    --phase8-entity-source-title-lookup true \
    --phase8-entity-source-graph-flow true \
    --phase8-entity-source-balanced true \
    --phase8-pamae-k 5 \
    --phase8-pamae-sample-size-per-k 40 \
    --phase8-pamae-num-samples 5 \
    --phase8-pamae-sampling query_weighted \
    --phase8-pamae-random-seed 42 \
    --phase8-refine-enabled "${REFINE_ENABLED}" \
    --phase8-refine-hops 2 \
    --phase8-refine-max-candidates-per-seed 128 \
    --phase8-refine-degree-cap 100 \
    --phase8-refine-iterations 1 \
    --phase8-seed-top-atoms-per-entity 4 \
    --phase8-seed-top-carriers-per-entity 2 \
    --phase8-seed-path-max-hops 3 \
    --phase8-seed-path-max-pairs 10 \
    --phase8-seed-total-candidate-cap 160 \
    >"${LOG_FILE}" 2>&1
done

echo "[phase8] done dataset=${DATASET} profile=${PROFILE}"
