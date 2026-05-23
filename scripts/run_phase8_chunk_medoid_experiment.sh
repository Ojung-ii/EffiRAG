#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ojungii/EffiRAG}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
DATASET="${DATASET:-hotpotqa}"
PROFILE="${PROFILE:-legacy_512_10}"
LIMIT="${LIMIT:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/phase8_chunk_medoid}"
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-false}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%S_%N)}"
# Optional comma-separated subset:
#   PHASE8_CHUNK_VARIANTS=chunk_pamae_k5
PHASE8_CHUNK_VARIANTS="${PHASE8_CHUNK_VARIANTS:-chunk_pamae_k5}"
PHASE8_CHUNK_SOURCE_BALANCED="${PHASE8_CHUNK_SOURCE_BALANCED:-false}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

if [[ "${DATASET}" != "hotpotqa" && "${DATASET}" != "2wikimultihopqa" ]]; then
  echo "[phase8-chunk] unsupported DATASET=${DATASET}" >&2
  exit 1
fi
if [[ "${PROFILE}" != "legacy_512_10" && "${PROFILE}" != "balanced_384_8" ]]; then
  echo "[phase8-chunk] unsupported PROFILE=${PROFILE}" >&2
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
  echo "[phase8-chunk] missing config: ${CFG}" >&2
  exit 1
fi

VARIANTS=(
  "chunk_pamae_k5:true:false:false:160"
)

MATCHED_VARIANTS=0
for row in "${VARIANTS[@]}"; do
  IFS=":" read -r VARIANT CHUNK_ENABLED BRIDGE_REFINE_ENABLED SB_ENABLED TOP_M <<<"${row}"
  if [[ -n "${PHASE8_CHUNK_VARIANTS}" ]]; then
    if [[ ",${PHASE8_CHUNK_VARIANTS}," != *",${VARIANT},"* ]]; then
      continue
    fi
  fi
  MATCHED_VARIANTS=$((MATCHED_VARIANTS + 1))
  RUN_DIR="${OUT_ROOT}/${DATASET}/${PROFILE}/${VARIANT}/${RUN_ID}"
  LOG_FILE="${OUT_ROOT}/logs/${DATASET}__${PROFILE}__${VARIANT}__${RUN_ID}.log"
  mkdir -p "${RUN_DIR}"

  echo "[phase8-chunk] dataset=${DATASET} profile=${PROFILE} variant=${VARIANT} limit=${LIMIT} run_id=${RUN_ID}"
  PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
    --config "${CFG}" \
    --dataset "${DATASET}" \
    --limit "${LIMIT}" \
    --output-dir "${RUN_DIR}" \
    --timestamp-output false \
    --method phase7_evidence_flow \
    --retrieval-only "${RETRIEVAL_ONLY}" \
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
    --phase8-pamae-enabled false \
    --phase8-chunk-medoid-enabled "${CHUNK_ENABLED}" \
    --phase8-chunk-medoid-proposal-mode carrier_seed_bridge_refine \
    --phase8-chunk-universe-top-n 2000 \
    --phase8-chunk-universe-min-n 200 \
    --phase8-chunk-universe-unit carrier \
    --phase8-chunk-source-semantic true \
    --phase8-chunk-source-source-balanced "${PHASE8_CHUNK_SOURCE_BALANCED}" \
    --phase8-chunk-source-graph-flow true \
    --phase8-chunk-source-title-entity-lookup true \
    --phase8-chunk-medoid-k 5 \
    --phase8-chunk-medoid-sample-size-per-k 40 \
    --phase8-chunk-medoid-num-samples 5 \
    --phase8-chunk-medoid-sampling query_weighted \
    --phase8-chunk-medoid-random-seed 42 \
    --phase8-chunk-bridge-refine-enabled "${BRIDGE_REFINE_ENABLED}" \
    --phase8-chunk-bridge-refine-hops 1 \
    --phase8-chunk-bridge-max-entities-per-seed 8 \
    --phase8-chunk-bridge-entity-degree-cap 100 \
    --phase8-chunk-bridge-max-chunks-per-entity 4 \
    --phase8-chunk-bridge-max-refine-candidates-per-seed 64 \
    --phase8-chunk-bridge-refine-iterations 1 \
    --phase8-chunk-seed-top-sentences-per-carrier 3 \
    --phase8-chunk-seed-top-entities-per-seed 8 \
    --phase8-chunk-seed-top-atoms-per-entity 3 \
    --phase8-chunk-seed-top-carriers-per-entity 2 \
    --phase8-chunk-seed-total-candidate-cap 160 \
    >"${LOG_FILE}" 2>&1
done

if [[ "${MATCHED_VARIANTS}" -eq 0 ]]; then
  echo "[phase8-chunk] no runnable Phase8-only variants matched PHASE8_CHUNK_VARIANTS=${PHASE8_CHUNK_VARIANTS}" >&2
  exit 1
fi

echo "[phase8-chunk] done dataset=${DATASET} profile=${PROFILE}"
