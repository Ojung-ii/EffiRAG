#!/bin/bash
set -euo pipefail

# Optional sanity check for MuSiQue in retrieval+delivery fusion track.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
LIMIT="${LIMIT:-100}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

DATASET_NAME="musique"
DATA_PATH="data/qa/musique.json"
CORPUS_PATH="data/musique_corpus.json"

EXPERIMENT_ROOT="outputs/rag_experiments/next_method_design/retrieval_delivery_fusion/musique_sanity"
PROFILE_ROOT="outputs/profiling/next_method_design/retrieval_delivery_fusion/musique_sanity"
mkdir -p "${EXPERIMENT_ROOT}" "${PROFILE_ROOT}"

find_latest_summary() {
  local out_dir="$1"
  local dataset="$2"
  find "${out_dir}/${dataset}" -type f -name rag_summary.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | awk '{print $2}'
}

run_variant() {
  local variant="$1"
  local cfg="$2"

  local out_dir="${EXPERIMENT_ROOT}/${variant}"
  local profile_out="${PROFILE_ROOT}/${DATASET_NAME}_${variant}_qa_q${LIMIT}_${RUN_STAMP}.jsonl"

  local args=(
    --config "${cfg}"
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
    --output-dir "${out_dir}"
    --profile-stages true
    --profile-output "${profile_out}"
    --llm-max-new-tokens 64
  )

  if [ "${GENERATOR}" = "openai_compat" ]; then
    args+=(
      --llm-base-url "${LLM_BASE_URL}"
      --llm-api-key "${LLM_API_KEY}"
    )
  fi

  echo "  - [musique sanity] ${variant}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m effirag.run_rag "${args[@]}"

  find_latest_summary "${out_dir}" "${DATASET_NAME}"
}

run_specs=()
s0="$(run_variant musique_baseline configs/rag_exp_2wiki_fusion_baseline.yaml)"
run_specs+=("${DATASET_NAME}:musique_baseline=${s0}")
s1="$(run_variant musique_chunk_package_basic configs/rag_exp_2wiki_fusion_chunk_package_basic.yaml)"
run_specs+=("${DATASET_NAME}:musique_chunk_package_basic=${s1}")

REPORT_JSON="${PROFILE_ROOT}/musique_chunk_grounded_sanity_q${LIMIT}_${RUN_STAMP}.json"
REPORT_MD="${PROFILE_ROOT}/musique_chunk_grounded_sanity_q${LIMIT}_${RUN_STAMP}.md"
INTERP_MD="${PROFILE_ROOT}/musique_chunk_grounded_sanity_q${LIMIT}_interpretation_${RUN_STAMP}.md"

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

summary_args=(
  --title "MuSiQue Chunk-Grounded Sanity (q${LIMIT}, QA)"
  --baseline-variant musique_baseline
  --output-json "${REPORT_JSON}"
  --output-md "${REPORT_MD}"
  --interpretation-md "${INTERP_MD}"
  --experiment-family next_method_design
  --question-being-answered "Is chunk-grounded package direction at least non-degenerate on MuSiQue?"
  --baseline-reference "frozen baseline"
  --frozen-config-reference "configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml"
  --dataset-scope musique
  --changed-components retrieval_delivery_fusion
  --changed-components chunk_grounded_delivery
  --git-branch "${GIT_BRANCH}"
  --git-commit "${GIT_COMMIT}"
  --git-tag "${GIT_TAG}"
)
for spec in "${run_specs[@]}"; do
  summary_args+=(--run "${spec}")
done
"${PYTHON_BIN}" scripts/summarize_chunk_grounded_context.py "${summary_args[@]}"

cp -f "${REPORT_JSON}" "${PROFILE_ROOT}/musique_chunk_grounded_sanity_q${LIMIT}_latest.json"
cp -f "${REPORT_MD}" "${PROFILE_ROOT}/musique_chunk_grounded_sanity_q${LIMIT}_latest.md"
cp -f "${INTERP_MD}" "${PROFILE_ROOT}/musique_chunk_grounded_sanity_q${LIMIT}_interpretation_latest.md"

echo "[musique_chunk_grounded_sanity] done"
