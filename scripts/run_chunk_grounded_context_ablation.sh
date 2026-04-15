#!/bin/bash
set -euo pipefail

# Next-method-design family: chunk-grounded context ablation.
# Frozen operating configs/scripts are not mutated.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS_CSV="${DATASETS:-${DATASET:-hotpotqa,2wikimultihopqa}}"
LIMIT="${LIMIT:-200}"
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-false}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

if [ "${RETRIEVAL_ONLY}" = "true" ]; then
  RUN_KIND="retrieval_only"
  RUN_QA="false"
  CFG_PREFIX="retrieval"
  OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/retrieval_experiments/next_method_design/chunk_grounded_context}"
else
  RUN_KIND="qa"
  RUN_QA="true"
  CFG_PREFIX="rag"
  OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rag_experiments/next_method_design/chunk_grounded_context}"
fi
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/next_method_design/chunk_grounded_context}"

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

variants=(
  cg_baseline
  cg_sentence_backfill_1
  cg_sentence_backfill_2
  cg_sentence_backfill_window_wide
  cg_corridor_lift_basic
  cg_corridor_lift_grounded
  cg_corridor_lift_grounded_bridge
  cg_package_score_basic
  cg_package_score_grounded
  cg_package_score_grounded_support
)

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

printf '[chunk_grounded] mode=%s limit=%s branch=%s commit=%s\n' "${RUN_KIND}" "${LIMIT}" "${GIT_BRANCH}" "${GIT_COMMIT}"

for ds in ${DATASETS_CSV//,/ }; do
  resolve_paths "${ds}"
  echo "[chunk_grounded] dataset=${DATASET_NAME}"

  run_specs=()
  for variant in "${variants[@]}"; do
    CFG="configs/${CFG_PREFIX}_nextdesign_chunk_grounded_${variant}.yaml"
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

  REPORT_JSON="${PROFILE_ROOT}/chunk_grounded_context_${DATASET_NAME}_${RUN_KIND}_summary_q${LIMIT}_${RUN_STAMP}.json"
  REPORT_MD="${PROFILE_ROOT}/chunk_grounded_context_${DATASET_NAME}_${RUN_KIND}_summary_q${LIMIT}_${RUN_STAMP}.md"
  INTERP_MD="${PROFILE_ROOT}/chunk_grounded_context_${DATASET_NAME}_${RUN_KIND}_interpretation_q${LIMIT}_${RUN_STAMP}.md"

  SUMMARY_ARGS=(
    --title "Chunk-Grounded Context Ablation (${DATASET_NAME}, ${RUN_KIND})"
    --baseline-variant cg_baseline
    --output-json "${REPORT_JSON}"
    --output-md "${REPORT_MD}"
    --interpretation-md "${INTERP_MD}"
    --experiment-family next_method_design
    --question-being-answered "Can chunk-grounded delivery improve EM/F1 without changing retrieval core?"
    --baseline-reference "cg_baseline"
    --frozen-config-reference "configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml"
    --dataset-scope "${DATASET_NAME}"
    --changed-components chunk_grounded_context
    --changed-components evidence_packaging
    --changed-components render_delivery
    --git-branch "${GIT_BRANCH}"
    --git-commit "${GIT_COMMIT}"
    --git-tag "${GIT_TAG}"
  )
  for spec in "${run_specs[@]}"; do
    SUMMARY_ARGS+=(--run "${spec}")
  done

  "${PYTHON_BIN}" scripts/summarize_chunk_grounded_context.py "${SUMMARY_ARGS[@]}"

  cp -f "${REPORT_JSON}" "${PROFILE_ROOT}/chunk_grounded_context_${DATASET_NAME}_${RUN_KIND}_summary_latest.json"
  cp -f "${REPORT_MD}" "${PROFILE_ROOT}/chunk_grounded_context_${DATASET_NAME}_${RUN_KIND}_summary_latest.md"
  cp -f "${INTERP_MD}" "${PROFILE_ROOT}/chunk_grounded_context_${DATASET_NAME}_${RUN_KIND}_interpretation_latest.md"

  echo "[chunk_grounded] dataset=${DATASET_NAME} done"
  echo "  report_json=${REPORT_JSON}"
  echo "  report_md=${REPORT_MD}"
  echo "  interpretation_md=${INTERP_MD}"
done

