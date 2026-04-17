#!/bin/bash
set -euo pipefail

# 2Wiki entity-chunk graph ablation: baseline/current-best-fusion/new graph variants.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache_entity_chunk_graph/2wikimultihopqa}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
LIMIT_STAGE1="${LIMIT_STAGE1:-200}"
LIMIT_STAGE2="${LIMIT_STAGE2:-1000}"
MAX_RECALL_DROP_FOR_SHORTLIST="${MAX_RECALL_DROP_FOR_SHORTLIST:-0.01}"
RUN_STAGE2="${RUN_STAGE2:-true}"
RUN_OPTIONAL_MIXED="${RUN_OPTIONAL_MIXED:-true}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

DATASET_NAME="2wikimultihopqa"
DATA_PATH="data/qa/2wikimultihopqa.json"
CORPUS_PATH="data/2wikimultihopqa_corpus.json"

EXPERIMENT_ROOT="outputs/rag_experiments/next_method_design/entity_chunk_graph/2wikimultihopqa"
PROFILE_ROOT="outputs/profiling/next_method_design/entity_chunk_graph/2wikimultihopqa"
mkdir -p "${EXPERIMENT_ROOT}" "${PROFILE_ROOT}"

find_latest_summary() {
  local out_dir="$1"
  local dataset="$2"
  find "${out_dir}/${dataset}" -type f -name rag_summary.json -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}'
}

run_variant() {
  local variant="$1"
  local cfg="$2"
  local limit="$3"
  local stage_label="$4"
  local __retvar="$5"

  if [ ! -f "${cfg}" ]; then
    echo "[ERROR] Missing config: ${cfg}" >&2
    exit 1
  fi

  local out_dir="${EXPERIMENT_ROOT}/${stage_label}/${variant}"
  local profile_out="${PROFILE_ROOT}/${DATASET_NAME}_${variant}_${stage_label}_qa_q${limit}_${RUN_STAMP}.jsonl"

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
    --limit "${limit}"
    --output-dir "${out_dir}"
    --profile-stages true
    --profile-output "${profile_out}"
    --llm-max-new-tokens 64
  )

  if [ "${GENERATOR}" = "openai_compat" ]; then
    args+=(--llm-base-url "${LLM_BASE_URL}" --llm-api-key "${LLM_API_KEY}")
  fi

  echo "  - [${stage_label}] ${variant} (limit=${limit})" >&2
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m effirag.run_rag "${args[@]}" 1>&2

  local summary_path
  summary_path="$(find_latest_summary "${out_dir}" "${DATASET_NAME}")"
  if [ -z "${summary_path}" ] || [ ! -f "${summary_path}" ]; then
    echo "[ERROR] Missing rag_summary.json for ${variant}" >&2
    exit 1
  fi
  printf -v "${__retvar}" '%s' "${summary_path}"
}

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

echo "[2wiki_entity_chunk_graph] stage1 limit=${LIMIT_STAGE1} stage2 limit=${LIMIT_STAGE2}"
echo "[2wiki_entity_chunk_graph] branch=${GIT_BRANCH} commit=${GIT_COMMIT}"

declare -A CFGS
CFGS[2wiki_baseline]="configs/rag_exp_2wiki_entity_chunk_graph_baseline.yaml"
CFGS[2wiki_current_best_fusion]="configs/rag_exp_2wiki_entity_chunk_graph_current_best_fusion.yaml"
CFGS[2wiki_entity_seed_chunk_lift_package_basic]="configs/rag_exp_2wiki_entity_chunk_graph_entity_seed_chunk_lift_package_basic.yaml"
CFGS[2wiki_entity_chunk_mixed_diffusion_package_basic]="configs/rag_exp_2wiki_entity_chunk_graph_entity_chunk_mixed_diffusion_package_basic.yaml"

stage1_variants=(
  2wiki_baseline
  2wiki_current_best_fusion
  2wiki_entity_seed_chunk_lift_package_basic
)
if [ "${RUN_OPTIONAL_MIXED}" = "true" ]; then
  stage1_variants+=(2wiki_entity_chunk_mixed_diffusion_package_basic)
fi

run_specs_stage1=()
for v in "${stage1_variants[@]}"; do
  sp=""
  run_variant "${v}" "${CFGS[$v]}" "${LIMIT_STAGE1}" "stage1" sp
  run_specs_stage1+=("${DATASET_NAME}:${v}=${sp}")
done

STAGE1_SUMMARY_JSON="${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE1}_${RUN_STAMP}.json"
STAGE1_SUMMARY_MD="${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE1}_${RUN_STAMP}.md"
STAGE1_INTERP_MD="${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE1}_interpretation_${RUN_STAMP}.md"

summary_args=(
  --title "2Wiki Entity-Chunk Graph Ablation (q${LIMIT_STAGE1}, QA)"
  --baseline-variant 2wiki_baseline
  --output-json "${STAGE1_SUMMARY_JSON}"
  --output-md "${STAGE1_SUMMARY_MD}"
  --interpretation-md "${STAGE1_INTERP_MD}"
  --experiment-family next_method_design
  --question-being-answered "Does entity-chunk graph retrieval + package delivery outperform current best fusion on 2Wiki?"
  --baseline-reference "frozen baseline (aggressive + top1corr_t1)"
  --frozen-config-reference "configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml"
  --dataset-scope 2wikimultihopqa
  --changed-components entity_chunk_graph
  --changed-components retrieval_delivery_fusion_inheritance
  --changed-components chunk_package_basic
  --git-branch "${GIT_BRANCH}"
  --git-commit "${GIT_COMMIT}"
  --git-tag "${GIT_TAG}"
)
for spec in "${run_specs_stage1[@]}"; do
  summary_args+=(--run "${spec}")
done
"${PYTHON_BIN}" scripts/summarize_chunk_grounded_context.py "${summary_args[@]}"

cp -f "${STAGE1_SUMMARY_JSON}" "${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE1}_latest.json"
cp -f "${STAGE1_SUMMARY_MD}" "${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE1}_latest.md"
cp -f "${STAGE1_INTERP_MD}" "${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE1}_interpretation_latest.md"

SHORTLIST_JSON="${PROFILE_ROOT}/2wiki_entity_chunk_graph_shortlist_q${LIMIT_STAGE1}_${RUN_STAMP}.json"
SHORTLIST_CSV="$(${PYTHON_BIN} scripts/select_fusion_topk.py \
  --summary-json "${STAGE1_SUMMARY_JSON}" \
  --baseline-variant 2wiki_baseline \
  --topk 2 \
  --max-recall-drop "${MAX_RECALL_DROP_FOR_SHORTLIST}" \
  --output-json "${SHORTLIST_JSON}")"

IFS=',' read -r -a shortlist_variants <<< "${SHORTLIST_CSV}"

echo "[2wiki_entity_chunk_graph] shortlist=${SHORTLIST_CSV}"

if [ "${RUN_STAGE2}" != "true" ]; then
  echo "[2wiki_entity_chunk_graph] RUN_STAGE2=false; stop after stage1"
  exit 0
fi

stage2_variants=(2wiki_baseline)
for v in "${shortlist_variants[@]}"; do
  if [ -n "${v}" ] && [ "${v}" != "2wiki_baseline" ]; then
    stage2_variants+=("${v}")
  fi
done

run_specs_stage2=()
for v in "${stage2_variants[@]}"; do
  sp=""
  run_variant "${v}" "${CFGS[$v]}" "${LIMIT_STAGE2}" "stage2" sp
  run_specs_stage2+=("${DATASET_NAME}:${v}=${sp}")
done

STAGE2_SUMMARY_JSON="${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE2}_${RUN_STAMP}.json"
STAGE2_SUMMARY_MD="${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE2}_${RUN_STAMP}.md"
STAGE2_INTERP_MD="${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE2}_interpretation_${RUN_STAMP}.md"

summary2_args=(
  --title "2Wiki Entity-Chunk Graph Ablation (q${LIMIT_STAGE2}, QA)"
  --baseline-variant 2wiki_baseline
  --output-json "${STAGE2_SUMMARY_JSON}"
  --output-md "${STAGE2_SUMMARY_MD}"
  --interpretation-md "${STAGE2_INTERP_MD}"
  --experiment-family next_method_design
  --question-being-answered "q1000 revalidation: does entity-chunk graph beat current best fusion on 2Wiki?"
  --baseline-reference "frozen baseline (aggressive + top1corr_t1)"
  --frozen-config-reference "configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml"
  --dataset-scope 2wikimultihopqa
  --changed-components entity_chunk_graph
  --changed-components retrieval_delivery_fusion_inheritance
  --changed-components chunk_package_basic
  --git-branch "${GIT_BRANCH}"
  --git-commit "${GIT_COMMIT}"
  --git-tag "${GIT_TAG}"
)
for spec in "${run_specs_stage2[@]}"; do
  summary2_args+=(--run "${spec}")
done
"${PYTHON_BIN}" scripts/summarize_chunk_grounded_context.py "${summary2_args[@]}"

cp -f "${STAGE2_SUMMARY_JSON}" "${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE2}_latest.json"
cp -f "${STAGE2_SUMMARY_MD}" "${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE2}_latest.md"
cp -f "${STAGE2_INTERP_MD}" "${PROFILE_ROOT}/2wiki_entity_chunk_graph_q${LIMIT_STAGE2}_interpretation_latest.md"

echo "[2wiki_entity_chunk_graph] done"
