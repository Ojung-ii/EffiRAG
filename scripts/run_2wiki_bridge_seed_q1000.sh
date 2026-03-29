#!/bin/bash
set -euo pipefail

# 2Wiki q1000 revalidation focused on bridge-seed family.
# Frozen defaults are not touched; this script only runs experiment-specific configs.

LIMIT="${LIMIT:-1000}"
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-false}"
RUN_MILD2="${RUN_MILD2:-false}"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/2wiki_bridge_seed_q1000}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

DATASET_NAME="2wikimultihopqa"
DATA_PATH="data/qa/2wikimultihopqa.json"
CORPUS_PATH="data/2wikimultihopqa_corpus.json"

if [ "${RETRIEVAL_ONLY}" = "true" ]; then
  RUN_KIND="retrieval_only"
  RUN_QA="false"
  CFG_PREFIX="retrieval"
  OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/retrieval_experiments/2wiki_bridge_seed_q1000}"
else
  RUN_KIND="qa"
  RUN_QA="true"
  CFG_PREFIX="rag"
  OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rag_experiments/2wiki_bridge_seed_q1000}"
fi

mkdir -p "${OUTPUT_ROOT}" "${PROFILE_ROOT}"

find_latest_summary() {
  local out_dir="$1"
  local dataset="$2"
  find "${out_dir}/${dataset}" -type f -name rag_summary.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | awk '{print $2}'
}

variants=(
  baseline
  bridge_seed_score_up
  bridge_seed_semantic_mild
)
if [ "${RUN_MILD2}" = "true" ]; then
  variants+=(bridge_seed_semantic_mild2)
fi

run_specs=()

echo "[2wiki_bridge_seed_q1000] mode=${RUN_KIND} limit=${LIMIT} run_mild2=${RUN_MILD2}"

for variant in "${variants[@]}"; do
  CFG="configs/${CFG_PREFIX}_exp_2wiki_bridge_seed_q1000_${variant}.yaml"
  if [ ! -f "${CFG}" ]; then
    echo "[ERROR] Missing config: ${CFG}" >&2
    exit 1
  fi

  OUT_DIR="${OUTPUT_ROOT}/${variant}"
  PROFILE_OUT="${PROFILE_ROOT}/${DATASET_NAME}_${variant}_${RUN_KIND}_q${LIMIT}_${RUN_STAMP}.jsonl"

  echo "  - run variant=${variant} config=${CFG}"

  ARGS=(
    --config "${CFG}"
    --dataset "${DATASET_NAME}"
    --data-path "${DATA_PATH}"
    --global-corpus-path "${CORPUS_PATH}"
    --graph-cache-dir "${GRAPH_CACHE_DIR}"
    --force-rebuild-graph-index false
    --run-qa "${RUN_QA}"
    --retrieval-only "${RETRIEVAL_ONLY}"
    --embedding-enabled true
    --embedding-model-name nvidia/NV-Embed-v2
    --generator "${GENERATOR}"
    --model-name "${MODEL_NAME}"
    --limit "${LIMIT}"
    --output-dir "${OUT_DIR}"
    --profile-stages true
    --profile-output "${PROFILE_OUT}"
    --llm-max-new-tokens 64
  )

  if [ "${GENERATOR}" = "openai_compat" ]; then
    ARGS+=(
      --llm-base-url "${LLM_BASE_URL}"
      --llm-api-key "${LLM_API_KEY}"
    )
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m effirag.run_rag "${ARGS[@]}"

  SUMMARY_PATH="$(find_latest_summary "${OUT_DIR}" "${DATASET_NAME}")"
  if [ -z "${SUMMARY_PATH}" ]; then
    echo "[ERROR] Missing rag_summary.json for ${DATASET_NAME}:${variant}" >&2
    exit 1
  fi
  run_specs+=("${DATASET_NAME}:${variant}=${SUMMARY_PATH}")
done

REPORT_JSON="${PROFILE_ROOT}/2wiki_bridge_seed_${RUN_KIND}_summary_q${LIMIT}_${RUN_STAMP}.json"
REPORT_MD="${PROFILE_ROOT}/2wiki_bridge_seed_${RUN_KIND}_summary_q${LIMIT}_${RUN_STAMP}.md"
ANALYSIS_MD="${PROFILE_ROOT}/2wiki_bridge_seed_${RUN_KIND}_interpretation_q${LIMIT}_${RUN_STAMP}.md"

SUMMARY_ARGS=(
  --schema semantic
  --title "2Wiki Bridge-Seed q1000 Revalidation (${RUN_KIND})"
  --baseline-variant baseline
  --output-json "${REPORT_JSON}"
  --output-md "${REPORT_MD}"
  --analysis-md "${ANALYSIS_MD}"
)
for spec in "${run_specs[@]}"; do
  SUMMARY_ARGS+=(--run "${spec}")
done

"${PYTHON_BIN}" scripts/summarize_experiment_report.py "${SUMMARY_ARGS[@]}"

cp -f "${REPORT_JSON}" "${PROFILE_ROOT}/2wiki_bridge_seed_${RUN_KIND}_summary_latest.json"
cp -f "${REPORT_MD}" "${PROFILE_ROOT}/2wiki_bridge_seed_${RUN_KIND}_summary_latest.md"
cp -f "${ANALYSIS_MD}" "${PROFILE_ROOT}/2wiki_bridge_seed_${RUN_KIND}_interpretation_latest.md"

echo "[2wiki_bridge_seed_q1000] done"
echo "  report_json=${REPORT_JSON}"
echo "  report_md=${REPORT_MD}"
echo "  analysis_md=${ANALYSIS_MD}"
