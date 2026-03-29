#!/bin/bash
set -euo pipefail

# Stage-2 experiment: semantic-aware selection ablation.
# baseline vs semantic_run_score vs semantic_seed_score vs semantic_union

DATASETS_CSV="${DATASETS:-hotpotqa,2wikimultihopqa}"
LIMIT="${LIMIT:-200}"
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-false}"
INCLUDE_MUSIQUE_RETRIEVAL="${INCLUDE_MUSIQUE_RETRIEVAL:-true}"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rag_experiments/semantic_selection}"
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/semantic_selection}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

# Optional positional overrides:
#   bash scripts/run_semantic_selection_ablation.sh hotpotqa,2wikimultihopqa
#   bash scripts/run_semantic_selection_ablation.sh hotpotqa true
if [ "${1:-}" != "" ]; then
  DATASETS_CSV="$1"
fi
if [ "${2:-}" != "" ]; then
  RETRIEVAL_ONLY="$2"
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

if [ "${RETRIEVAL_ONLY}" = "true" ] && [ "${INCLUDE_MUSIQUE_RETRIEVAL}" = "true" ]; then
  if [[ ",${DATASETS_CSV}," != *",musique,"* ]]; then
    DATASETS_CSV="${DATASETS_CSV},musique"
  fi
fi

variants=(baseline semantic_run_score semantic_seed_score semantic_union)
run_specs=()

for ds in ${DATASETS_CSV//,/ }; do
  resolve_paths "${ds}"
  echo "[semantic_ablation] dataset=${DATASET_NAME} retrieval_only=${RETRIEVAL_ONLY} limit=${LIMIT}"

  for variant in "${variants[@]}"; do
    case "${variant}" in
      baseline)
        CFG="configs/rag_exp_semantic_ablation_baseline.yaml"
        ;;
      semantic_run_score)
        CFG="configs/rag_exp_semantic_ablation_run_score.yaml"
        ;;
      semantic_seed_score)
        CFG="configs/rag_exp_semantic_ablation_seed_score.yaml"
        ;;
      semantic_union)
        CFG="configs/rag_exp_semantic_ablation_union.yaml"
        ;;
      *)
        echo "[ERROR] Unknown variant: ${variant}" >&2
        exit 1
        ;;
    esac

    OUT_DIR="${OUTPUT_ROOT}/${variant}"
    PROFILE_OUT="${PROFILE_ROOT}/${DATASET_NAME}_${variant}_retrieval${RETRIEVAL_ONLY}_q${LIMIT}_${RUN_STAMP}.jsonl"

    echo "  - run variant=${variant} config=${CFG}"

    ARGS=(
      --config "${CFG}"
      --dataset "${DATASET_NAME}"
      --data-path "${DATA_PATH}"
      --global-corpus-path "${CORPUS_PATH}"
      --graph-cache-dir "${GRAPH_CACHE_DIR}"
      --force-rebuild-graph-index false
      --embedding-enabled true
      --embedding-model-name nvidia/NV-Embed-v2
      --generator "${GENERATOR}"
      --model-name "${MODEL_NAME}"
      --limit "${LIMIT}"
      --retrieval-only "${RETRIEVAL_ONLY}"
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

RUN_KIND="qa"
if [ "${RETRIEVAL_ONLY}" = "true" ]; then
  RUN_KIND="retrieval_only"
fi

REPORT_JSON="${PROFILE_ROOT}/semantic_selection_${RUN_KIND}_summary_q${LIMIT}_${RUN_STAMP}.json"
REPORT_MD="${PROFILE_ROOT}/semantic_selection_${RUN_KIND}_summary_q${LIMIT}_${RUN_STAMP}.md"
ANALYSIS_MD="${PROFILE_ROOT}/semantic_selection_${RUN_KIND}_interpretation_q${LIMIT}_${RUN_STAMP}.md"

SUMMARY_ARGS=(
  --schema semantic
  --title "Semantic Selection Ablation (${RUN_KIND})"
  --baseline-variant baseline
  --output-json "${REPORT_JSON}"
  --output-md "${REPORT_MD}"
  --analysis-md "${ANALYSIS_MD}"
)
for spec in "${run_specs[@]}"; do
  SUMMARY_ARGS+=(--run "${spec}")
done

"${PYTHON_BIN}" scripts/summarize_experiment_report.py "${SUMMARY_ARGS[@]}"

cp -f "${REPORT_JSON}" "${PROFILE_ROOT}/semantic_selection_${RUN_KIND}_summary_latest.json"
cp -f "${REPORT_MD}" "${PROFILE_ROOT}/semantic_selection_${RUN_KIND}_summary_latest.md"
cp -f "${ANALYSIS_MD}" "${PROFILE_ROOT}/semantic_selection_${RUN_KIND}_interpretation_latest.md"

echo "[semantic_ablation] done"
echo "  report_json=${REPORT_JSON}"
echo "  report_md=${REPORT_MD}"
echo "  analysis_md=${ANALYSIS_MD}"
