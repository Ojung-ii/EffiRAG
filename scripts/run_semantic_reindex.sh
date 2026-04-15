#!/bin/bash
set -euo pipefail

# Optional helper: build semantic experiment cache in an isolated namespace
# so frozen operating cache is never overwritten.

DATASETS_CSV="${DATASETS:-hotpotqa,2wikimultihopqa,musique}"
CACHE_ROOT="${CACHE_ROOT:-outputs/index_cache_semantic_exp}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EMBED_MODEL="${EMBED_MODEL:-nvidia/NV-Embed-v2}"
OPENIE_MODEL="${OPENIE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OPENIE_API_BASE_URL="${OPENIE_API_BASE_URL:-http://localhost:8011/v1}"

mkdir -p "${CACHE_ROOT}"

resolve_corpus() {
  local ds="$1"
  case "${ds}" in
    hotpotqa)
      DATASET_NAME="hotpotqa"
      CORPUS_PATH="data/hotpotqa_corpus.json"
      ;;
    2wikimultihopqa|2wiki)
      DATASET_NAME="2wikimultihopqa"
      CORPUS_PATH="data/2wikimultihopqa_corpus.json"
      ;;
    musique)
      DATASET_NAME="musique"
      CORPUS_PATH="data/musique_corpus.json"
      ;;
    popqa)
      DATASET_NAME="popqa"
      CORPUS_PATH="data/popqa_corpus.json"
      ;;
    *)
      echo "[ERROR] Unsupported dataset: ${ds}" >&2
      return 1
      ;;
  esac
}

for ds in ${DATASETS_CSV//,/ }; do
  resolve_corpus "${ds}"
  CACHE_DIR="${CACHE_ROOT}/${DATASET_NAME}"

  echo "[semantic_reindex] dataset=${DATASET_NAME}"
  echo "  corpus=${CORPUS_PATH}"
  echo "  cache_dir=${CACHE_DIR}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m effirag.run_index \
    --dataset "${DATASET_NAME}" \
    --corpus-path "${CORPUS_PATH}" \
    --cache-dir "${CACHE_DIR}" \
    --force-rebuild "${FORCE_REBUILD}" \
    --embedding-enabled true \
    --embedding-model-name "${EMBED_MODEL}" \
    --embedding-batch-size 16 \
    --embedding-max-length 192 \
    --embedding-text-max-chars 600 \
    --semantic-scan-batch-size 32768 \
    --openie-mode llm \
    --openie-model-name "${OPENIE_MODEL}" \
    --openie-text-max-chars 2200 \
    --openie-max-new-tokens 256 \
    --openie-api-base-url "${OPENIE_API_BASE_URL}" \
    --openie-parallel-workers 6

done

echo "[semantic_reindex] complete"
echo "  cache_root=${CACHE_ROOT}"
