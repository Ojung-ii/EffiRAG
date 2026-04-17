#!/bin/bash
set -euo pipefail

# HotpotQA q1000 revalidation for chunk-grounded corridor candidates.
# Frozen configs/scripts/reports are not mutated.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

LIMIT="${LIMIT:-1000}"
RUN_OPTIONAL_PACKAGE="${RUN_OPTIONAL_PACKAGE:-false}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

DATASET_NAME="hotpotqa"
DATA_PATH="data/qa/hotpotqa.json"
CORPUS_PATH="data/hotpotqa_corpus.json"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rag_experiments/next_method_design/chunk_grounded_context_q1000/hotpotqa}"
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/next_method_design/chunk_grounded_context_q1000}"

mkdir -p "${OUTPUT_ROOT}" "${PROFILE_ROOT}"

find_latest_summary() {
  local out_dir="$1"
  local dataset="$2"
  find "${out_dir}/${dataset}" -type f -name rag_summary.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | awk '{print $2}'
}

variants=(
  baseline
  corridor_lift_grounded_bridge
  corridor_lift_grounded
)
if [ "${RUN_OPTIONAL_PACKAGE}" = "true" ]; then
  variants+=(package_score_grounded)
fi

run_specs=()

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

printf '[hotpot_chunk_grounded_q1000] limit=%s optional_package=%s branch=%s commit=%s\n' "${LIMIT}" "${RUN_OPTIONAL_PACKAGE}" "${GIT_BRANCH}" "${GIT_COMMIT}"

for variant in "${variants[@]}"; do
  CFG="configs/rag_exp_hotpot_chunk_grounded_q1000_${variant}.yaml"
  if [ ! -f "${CFG}" ]; then
    echo "[ERROR] Missing config: ${CFG}" >&2
    exit 1
  fi

  OUT_DIR="${OUTPUT_ROOT}/${variant}"
  PROFILE_OUT="${PROFILE_ROOT}/${DATASET_NAME}_${variant}_qa_q${LIMIT}_${RUN_STAMP}.jsonl"

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
    --evaluator-mode hipporag2_parity
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

REPORT_JSON="${PROFILE_ROOT}/chunk_grounded_context_hotpotqa_q1000_summary_q${LIMIT}_${RUN_STAMP}.json"
REPORT_MD="${PROFILE_ROOT}/chunk_grounded_context_hotpotqa_q1000_summary_q${LIMIT}_${RUN_STAMP}.md"
INTERP_MD="${PROFILE_ROOT}/chunk_grounded_context_hotpotqa_q1000_interpretation_q${LIMIT}_${RUN_STAMP}.md"

SUMMARY_ARGS=(
  --title "Chunk-Grounded Context q1000 Revalidation (hotpotqa, qa)"
  --baseline-variant baseline
  --output-json "${REPORT_JSON}"
  --output-md "${REPORT_MD}"
  --interpretation-md "${INTERP_MD}"
  --experiment-family next_method_design
  --question-being-answered "Does strong chunk-grounded corridor stay robust on Hotpot q1000?"
  --baseline-reference "cg_baseline"
  --frozen-config-reference "configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml"
  --dataset-scope "hotpotqa"
  --changed-components chunk_grounded_context_q1000
  --changed-components corridor_lift_grounded_bridge
  --changed-components evidence_packaging
  --git-branch "${GIT_BRANCH}"
  --git-commit "${GIT_COMMIT}"
  --git-tag "${GIT_TAG}"
)
for spec in "${run_specs[@]}"; do
  SUMMARY_ARGS+=(--run "${spec}")
done

"${PYTHON_BIN}" scripts/summarize_chunk_grounded_context.py "${SUMMARY_ARGS[@]}"

cp -f "${REPORT_JSON}" "${PROFILE_ROOT}/chunk_grounded_context_hotpotqa_q1000_summary_latest.json"
cp -f "${REPORT_MD}" "${PROFILE_ROOT}/chunk_grounded_context_hotpotqa_q1000_summary_latest.md"
cp -f "${INTERP_MD}" "${PROFILE_ROOT}/chunk_grounded_context_hotpotqa_q1000_interpretation_latest.md"

echo "[hotpot_chunk_grounded_q1000] done"
echo "  report_json=${REPORT_JSON}"
echo "  report_md=${REPORT_MD}"
echo "  interpretation_md=${INTERP_MD}"
