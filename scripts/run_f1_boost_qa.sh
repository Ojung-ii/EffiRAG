#!/bin/bash
set -euo pipefail

# Stage-1 experiment: context/render expansion for F1 improvement.
# Adds only experimental runs (no frozen profile mutation).

DATASETS_CSV="${DATASETS:-hotpotqa,2wikimultihopqa}"
LIMIT="${LIMIT:-200}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rag_experiments/f1_boost}"
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/f1_boost}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

# Optional positional override: bash scripts/run_f1_boost_qa.sh hotpotqa,2wikimultihopqa
if [ "${1:-}" != "" ]; then
  DATASETS_CSV="$1"
fi

mkdir -p "${OUTPUT_ROOT}" "${PROFILE_ROOT}"

resolve_paths() {
  local ds="$1"
  case "${ds}" in
    hotpotqa)
      DATA_PATH="data/qa/hotpotqa.json"
      CORPUS_PATH="data/hotpotqa_corpus.json"
      DATASET_NAME="hotpotqa"
      ;;
    2wikimultihopqa|2wiki)
      DATA_PATH="data/qa/2wikimultihopqa.json"
      CORPUS_PATH="data/2wikimultihopqa_corpus.json"
      DATASET_NAME="2wikimultihopqa"
      ;;
    musique)
      DATA_PATH="data/qa/musique.json"
      CORPUS_PATH="data/musique_corpus.json"
      DATASET_NAME="musique"
      ;;
    popqa)
      DATA_PATH="data/qa/popqa.json"
      CORPUS_PATH="data/popqa_corpus.json"
      DATASET_NAME="popqa"
      ;;
    *)
      echo "[ERROR] Unsupported dataset: ${ds}" >&2
      return 1
      ;;
  esac
}

find_latest_summary() {
  local out_dir="$1"
  local dataset="$2"
  find "${out_dir}/${dataset}" -type f -name rag_summary.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | awk '{print $2}'
}

run_specs=()
variants=(baseline variant_a_support variant_b_support_main variant_c_support_corridor)

for ds in ${DATASETS_CSV//,/ }; do
  resolve_paths "${ds}"

  echo "[f1_boost] dataset=${DATASET_NAME} limit=${LIMIT}"

  for variant in "${variants[@]}"; do
    case "${variant}" in
      baseline)
        CFG="configs/rag_exp_f1_baseline.yaml"
        ;;
      variant_a_support)
        CFG="configs/rag_exp_f1_variant_a_support.yaml"
        ;;
      variant_b_support_main)
        CFG="configs/rag_exp_f1_variant_b_support_main.yaml"
        ;;
      variant_c_support_corridor)
        CFG="configs/rag_exp_f1_variant_c_support_corridor.yaml"
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

done

REPORT_JSON="${PROFILE_ROOT}/f1_boost_qa_summary_q${LIMIT}_${RUN_STAMP}.json"
REPORT_MD="${PROFILE_ROOT}/f1_boost_qa_summary_q${LIMIT}_${RUN_STAMP}.md"
ANALYSIS_MD="${PROFILE_ROOT}/f1_boost_qa_interpretation_q${LIMIT}_${RUN_STAMP}.md"

SUMMARY_ARGS=(
  --schema f1
  --title "F1 Boost QA (Context/Render Expansion)"
  --baseline-variant baseline
  --output-json "${REPORT_JSON}"
  --output-md "${REPORT_MD}"
  --analysis-md "${ANALYSIS_MD}"
)
for spec in "${run_specs[@]}"; do
  SUMMARY_ARGS+=(--run "${spec}")
done

"${PYTHON_BIN}" scripts/summarize_experiment_report.py "${SUMMARY_ARGS[@]}"

cp -f "${REPORT_JSON}" "${PROFILE_ROOT}/f1_boost_qa_summary_latest.json"
cp -f "${REPORT_MD}" "${PROFILE_ROOT}/f1_boost_qa_summary_latest.md"
cp -f "${ANALYSIS_MD}" "${PROFILE_ROOT}/f1_boost_qa_interpretation_latest.md"

echo "[f1_boost] done"
echo "  report_json=${REPORT_JSON}"
echo "  report_md=${REPORT_MD}"
echo "  analysis_md=${ANALYSIS_MD}"
