#!/bin/bash
set -euo pipefail

# HotpotQA 1000-sample revalidation:
# baseline vs semantic_union (QA mode)

LIMIT="${LIMIT:-1000}"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rag_experiments/hotpot_semantic_union_1000}"
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/hotpot_semantic_union_1000}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_ROOT}" "${PROFILE_ROOT}"

DATASET_NAME="hotpotqa"
DATA_PATH="data/qa/hotpotqa.json"
CORPUS_PATH="data/hotpotqa_corpus.json"

find_latest_summary() {
  local out_dir="$1"
  local dataset="$2"
  find "${out_dir}/${dataset}" -type f -name rag_summary.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | awk '{print $2}'
}

variants=(baseline semantic_union)
run_specs=()

echo "[hotpot_semantic_union_1000] dataset=${DATASET_NAME} limit=${LIMIT}"

for variant in "${variants[@]}"; do
  case "${variant}" in
    baseline)
      CFG="configs/rag_exp_hotpot_semantic_union1000_baseline.yaml"
      ;;
    semantic_union)
      CFG="configs/rag_exp_hotpot_semantic_union1000_union.yaml"
      ;;
    *)
      echo "[ERROR] Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac

  OUT_DIR="${OUTPUT_ROOT}/${variant}"
  PROFILE_OUT="${PROFILE_ROOT}/${DATASET_NAME}_${variant}_q${LIMIT}_${RUN_STAMP}.jsonl"

  echo "  - run variant=${variant} config=${CFG}"

  ARGS=(
    --config "${CFG}"
    --dataset "${DATASET_NAME}"
    --data-path "${DATA_PATH}"
    --global-corpus-path "${CORPUS_PATH}"
    --graph-cache-dir "${GRAPH_CACHE_DIR}"
    --force-rebuild-graph-index false
    --run-qa true
    --retrieval-only false
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

REPORT_JSON="${PROFILE_ROOT}/hotpot_semantic_union_qa_summary_q${LIMIT}_${RUN_STAMP}.json"
REPORT_MD="${PROFILE_ROOT}/hotpot_semantic_union_qa_summary_q${LIMIT}_${RUN_STAMP}.md"
ANALYSIS_MD="${PROFILE_ROOT}/hotpot_semantic_union_qa_interpretation_q${LIMIT}_${RUN_STAMP}.md"

SUMMARY_ARGS=(
  --schema semantic
  --title "Hotpot Semantic Union Revalidation (qa, q${LIMIT})"
  --baseline-variant baseline
  --output-json "${REPORT_JSON}"
  --output-md "${REPORT_MD}"
  --analysis-md "${ANALYSIS_MD}"
)
for spec in "${run_specs[@]}"; do
  SUMMARY_ARGS+=(--run "${spec}")
done

"${PYTHON_BIN}" scripts/summarize_experiment_report.py "${SUMMARY_ARGS[@]}"

cp -f "${REPORT_JSON}" "${PROFILE_ROOT}/hotpot_semantic_union_qa_summary_latest.json"
cp -f "${REPORT_MD}" "${PROFILE_ROOT}/hotpot_semantic_union_qa_summary_latest.md"
cp -f "${ANALYSIS_MD}" "${PROFILE_ROOT}/hotpot_semantic_union_qa_interpretation_latest.md"

echo "[hotpot_semantic_union_1000] done"
echo "  report_json=${REPORT_JSON}"
echo "  report_md=${REPORT_MD}"
echo "  analysis_md=${ANALYSIS_MD}"
