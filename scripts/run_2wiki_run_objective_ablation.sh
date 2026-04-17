#!/bin/bash
set -euo pipefail

# Next-method-design family: 2Wiki run-level objective redesign ablation.
# Frozen operating configs/scripts are not mutated.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

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

DATASET_NAME="${DATASET_NAME:-2wikimultihopqa}"
DATA_PATH="${DATA_PATH:-data/qa/2wikimultihopqa.json}"
CORPUS_PATH="${CORPUS_PATH:-data/2wikimultihopqa_corpus.json}"

if [ "${RETRIEVAL_ONLY}" = "true" ]; then
  RUN_KIND="retrieval_only"
  RUN_QA="false"
  CFG_PREFIX="retrieval"
  OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/retrieval_experiments/next_method_design/2wiki_run_objective}"
else
  RUN_KIND="qa"
  RUN_QA="true"
  CFG_PREFIX="rag"
  OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rag_experiments/next_method_design/2wiki_run_objective}"
fi
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/next_method_design/2wiki_run_objective}"

mkdir -p "${OUTPUT_ROOT}" "${PROFILE_ROOT}"

find_latest_summary() {
  local out_dir="$1"
  local dataset="$2"
  find "${out_dir}/${dataset}" -type f -name rag_summary.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | awk '{print $2}'
}

variants=(
  baseline
  bridge_complete
  grounded_bridge
  grounded_bridge_mild_semantic
)
run_specs=()

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

printf '[nextdesign:2wiki_run_objective] mode=%s limit=%s branch=%s commit=%s\n' "${RUN_KIND}" "${LIMIT}" "${GIT_BRANCH}" "${GIT_COMMIT}"

for variant in "${variants[@]}"; do
  CFG="configs/${CFG_PREFIX}_nextdesign_2wiki_run_objective_${variant}.yaml"
  if [ ! -f "${CFG}" ]; then
    echo "[ERROR] Missing config: ${CFG}" >&2
    exit 1
  fi

  OUT_DIR="${OUTPUT_ROOT}/${variant}"
  PROFILE_OUT="${PROFILE_ROOT}/nextdesign_2wiki_run_objective_${variant}_${RUN_KIND}_q${LIMIT}_${RUN_STAMP}.jsonl"

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

REPORT_JSON="${PROFILE_ROOT}/nextdesign_2wiki_run_objective_${RUN_KIND}_summary_q${LIMIT}_${RUN_STAMP}.json"
REPORT_MD="${PROFILE_ROOT}/nextdesign_2wiki_run_objective_${RUN_KIND}_summary_q${LIMIT}_${RUN_STAMP}.md"
ANALYSIS_MD="${PROFILE_ROOT}/nextdesign_2wiki_run_objective_${RUN_KIND}_interpretation_q${LIMIT}_${RUN_STAMP}.md"

SUMMARY_ARGS=(
  --schema semantic
  --title "Next Method Design: 2Wiki Run-Objective Ablation (${RUN_KIND})"
  --baseline-variant baseline
  --output-json "${REPORT_JSON}"
  --output-md "${REPORT_MD}"
  --analysis-md "${ANALYSIS_MD}"
  --experiment-family next_method_design
  --question-being-answered "Does run-level objective redesign improve 2Wiki over seed-only correction?"
  --baseline-reference "2wiki_run_objective_baseline"
  --frozen-config-reference "configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml"
  --dataset-scope "2wikimultihopqa"
  --changed-components run_objective
  --changed-components bridge_completeness
  --changed-components entity_chunk_grounding
  --git-branch "${GIT_BRANCH}"
  --git-commit "${GIT_COMMIT}"
  --git-tag "${GIT_TAG}"
)
for spec in "${run_specs[@]}"; do
  SUMMARY_ARGS+=(--run "${spec}")
done

"${PYTHON_BIN}" scripts/summarize_experiment_report.py "${SUMMARY_ARGS[@]}"

cp -f "${REPORT_JSON}" "${PROFILE_ROOT}/nextdesign_2wiki_run_objective_${RUN_KIND}_summary_latest.json"
cp -f "${REPORT_MD}" "${PROFILE_ROOT}/nextdesign_2wiki_run_objective_${RUN_KIND}_summary_latest.md"
cp -f "${ANALYSIS_MD}" "${PROFILE_ROOT}/nextdesign_2wiki_run_objective_${RUN_KIND}_interpretation_latest.md"

echo "[nextdesign:2wiki_run_objective] done"
echo "  report_json=${REPORT_JSON}"
echo "  report_md=${REPORT_MD}"
echo "  analysis_md=${ANALYSIS_MD}"
