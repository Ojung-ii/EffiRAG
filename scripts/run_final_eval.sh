#!/bin/bash
set -euo pipefail

MODE="${1:-speed_default}"
LIMIT="${LIMIT:-200}"
DATASET="${DATASET:-2wikimultihopqa}"
DATA_PATH="${DATA_PATH:-data/qa/2wikimultihopqa.json}"
GLOBAL_CORPUS_PATH="${GLOBAL_CORPUS_PATH:-data/2wikimultihopqa_corpus.json}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/final_eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-false}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"

BASE_CONFIG=""
case "${MODE}" in
  speed_default)
    BASE_CONFIG="configs/rag_speed_profile.yaml"
    ;;
  quality_variant)
    BASE_CONFIG="configs/rag_quality_profile.yaml"
    ;;
  ablation_off)
    BASE_CONFIG="configs/rag_quality_off_ablation.yaml"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Supported modes: speed_default | quality_variant | ablation_off" >&2
    exit 1
    ;;
esac

RUN_KIND="qa"
if [ "${RETRIEVAL_ONLY}" = "true" ]; then
  RUN_KIND="retrieval_only"
fi
PROFILE_OUTPUT="outputs/profiling/final_eval_${DATASET}_${MODE}_${RUN_KIND}_q${LIMIT}.jsonl"

echo "Running final eval mode=${MODE} retrieval_only=${RETRIEVAL_ONLY} limit=${LIMIT}"
echo "Config: ${BASE_CONFIG}"
echo "Profile output: ${PROFILE_OUTPUT}"

ARGS=(
  --config "${BASE_CONFIG}"
  --dataset "${DATASET}"
  --data-path "${DATA_PATH}"
  --global-corpus-path "${GLOBAL_CORPUS_PATH}"
  --graph-cache-dir "${GRAPH_CACHE_DIR}"
  --force-rebuild-graph-index false
  --embedding-enabled true
  --embedding-model-name nvidia/NV-Embed-v2
  --generator "${GENERATOR}"
  --model-name "${MODEL_NAME}"
  --limit "${LIMIT}"
  --retrieval-only "${RETRIEVAL_ONLY}"
  --output-dir "${OUTPUT_DIR}"
  --profile-stages true
  --profile-output "${PROFILE_OUTPUT}"
)

if [ "${GENERATOR}" = "openai_compat" ]; then
  ARGS+=(
    --llm-base-url "${LLM_BASE_URL}"
    --llm-api-key "${LLM_API_KEY}"
  )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m effirag.run_rag "${ARGS[@]}"
