#!/bin/bash
set -euo pipefail

# Build/reuse entity-chunk graph experiment index namespace.
# Frozen cache namespace is not touched.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
OPENIE_MODE="${OPENIE_MODE:-llm}"
OPENIE_MODEL_NAME="${OPENIE_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-nvidia/NV-Embed-v2}"
INDEX_ROOT="${INDEX_ROOT:-outputs/index_cache_entity_chunk_graph}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

DATASETS="${DATASETS:-hotpotqa,2wikimultihopqa}"

resolve_paths() {
  local dataset="$1"
  case "$dataset" in
    hotpotqa)
      echo "data/hotpotqa_corpus.json"
      ;;
    2wikimultihopqa)
      echo "data/2wikimultihopqa_corpus.json"
      ;;
    musique)
      echo "data/musique_corpus.json"
      ;;
    popqa)
      echo "data/popqa_corpus.json"
      ;;
    *)
      echo ""
      ;;
  esac
}

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

PROFILE_ROOT="outputs/profiling/next_method_design/entity_chunk_graph"
mkdir -p "${PROFILE_ROOT}"
REPORT_PATH="${PROFILE_ROOT}/entity_chunk_reindex_${RUN_STAMP}.md"

echo "# Entity-Chunk Graph Reindex Report" > "${REPORT_PATH}"
echo "" >> "${REPORT_PATH}"
echo "- experiment_family: next_method_design" >> "${REPORT_PATH}"
echo "- question_being_answered: Can we create a separate entity-chunk graph cache namespace without touching frozen cache?" >> "${REPORT_PATH}"
echo "- baseline_reference: frozen baseline cache namespace (outputs/index_cache)" >> "${REPORT_PATH}"
echo "- frozen_config_reference: configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml" >> "${REPORT_PATH}"
echo "- dataset_scope: ${DATASETS}" >> "${REPORT_PATH}"
echo "- changed_components: entity_chunk_graph_reindex_path" >> "${REPORT_PATH}"
echo "- git_branch: ${GIT_BRANCH}" >> "${REPORT_PATH}"
echo "- git_commit: ${GIT_COMMIT}" >> "${REPORT_PATH}"
echo "- git_tag: ${GIT_TAG}" >> "${REPORT_PATH}"
echo "" >> "${REPORT_PATH}"

echo "| dataset | corpus_path | cache_dir | force_rebuild | status | summary_path |" >> "${REPORT_PATH}"
echo "| --- | --- | --- | --- | --- | --- |" >> "${REPORT_PATH}"

IFS=',' read -r -a ds_arr <<< "${DATASETS}"
for ds in "${ds_arr[@]}"; do
  dataset="$(echo "$ds" | xargs)"
  [ -z "${dataset}" ] && continue
  corpus_path="$(resolve_paths "${dataset}")"
  if [ -z "${corpus_path}" ] || [ ! -f "${corpus_path}" ]; then
    echo "[WARN] skip ${dataset}: missing corpus path (${corpus_path})"
    echo "| ${dataset} | ${corpus_path:-N/A} | ${INDEX_ROOT}/${dataset} | ${FORCE_REBUILD} | skipped_missing_corpus | N/A |" >> "${REPORT_PATH}"
    continue
  fi

  cache_dir="${INDEX_ROOT}/${dataset}"
  mkdir -p "${cache_dir}"

  echo "[entity_chunk_reindex] dataset=${dataset} cache_dir=${cache_dir} force_rebuild=${FORCE_REBUILD}"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m effirag.run_index \
    --dataset "${dataset}" \
    --corpus-path "${corpus_path}" \
    --cache-dir "${cache_dir}" \
    --force-rebuild "${FORCE_REBUILD}" \
    --embedding-enabled true \
    --embedding-model-name "${EMBEDDING_MODEL_NAME}" \
    --embedding-batch-size 16 \
    --embedding-max-length 192 \
    --embedding-text-max-chars 600 \
    --graph-mode entity_chunk_graph \
    --openie-mode "${OPENIE_MODE}" \
    --openie-model-name "${OPENIE_MODEL_NAME}" \
    --openie-text-max-chars 2200 \
    --openie-max-new-tokens 256 \
    --openie-local-files-only true \
    --openie-parallel-workers 6 \
    --openie-log-every 200 \
    > "${cache_dir}/reindex_${RUN_STAMP}.log" 2>&1
  rc=$?
  set -e

  summary_path="$(find "${cache_dir}" -type f -name index_summary.json -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}')"
  if [ ${rc} -eq 0 ]; then
    echo "| ${dataset} | ${corpus_path} | ${cache_dir} | ${FORCE_REBUILD} | ok | ${summary_path:-N/A} |" >> "${REPORT_PATH}"
  else
    echo "| ${dataset} | ${corpus_path} | ${cache_dir} | ${FORCE_REBUILD} | failed(${rc}) | ${summary_path:-N/A} |" >> "${REPORT_PATH}"
    echo "[ERROR] reindex failed for ${dataset}, see ${cache_dir}/reindex_${RUN_STAMP}.log"
    exit ${rc}
  fi

done

echo "[entity_chunk_reindex] done"
echo "report=${REPORT_PATH}"
