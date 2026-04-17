#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential index builder for entity_chunk_graph across multiple datasets.
# Usage:
#   bash index_entity_chunk_graph_datasets.sh
#   bash index_entity_chunk_graph_datasets.sh hotpotqa musique
#   bash index_entity_chunk_graph_datasets.sh all
#
# No args => runs all four datasets in order:
#   hotpotqa musique 2wikimultihopqa popqa
#
# Override defaults with environment variables, for example:
#   OPENIE_PARALLEL_WORKERS=6 EMBEDDING_BATCH_SIZE=8 bash index_entity_chunk_graph_datasets.sh hotpotqa

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CORPUS_ROOT="${CORPUS_ROOT:-data}"
CACHE_ROOT="${CACHE_ROOT:-outputs/index_cache_entity_chunk_graph}"
LOG_ROOT="${LOG_ROOT:-outputs/index_logs_entity_chunk_graph}"

GRAPH_MODE="${GRAPH_MODE:-entity_chunk_graph}"
INDEX_CHUNK_UNIT="${INDEX_CHUNK_UNIT:-passage}"
FORCE_REBUILD="${FORCE_REBUILD:-true}"

OPENIE_MODE="${OPENIE_MODE:-llm}"
OPENIE_MODEL_NAME="${OPENIE_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
OPENIE_API_BASE_URL="${OPENIE_API_BASE_URL:-http://localhost:8011/v1}"
OPENIE_PARALLEL_WORKERS="${OPENIE_PARALLEL_WORKERS:-6}"
OPENIE_TEXT_MAX_CHARS="${OPENIE_TEXT_MAX_CHARS:-1800}"
OPENIE_MAX_NEW_TOKENS="${OPENIE_MAX_NEW_TOKENS:-256}"

EMBEDDING_ENABLED="${EMBEDDING_ENABLED:-true}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-nvidia/NV-Embed-v2}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-384}"
EMBEDDING_TEXT_MAX_CHARS="${EMBEDDING_TEXT_MAX_CHARS:-1200}"

DEFAULT_DATASETS=(hotpotqa musique 2wikimultihopqa popqa)

usage() {
  cat <<USAGE
Usage:
  bash $(basename "$0") [all|DATASET ...]

Datasets:
  hotpotqa
  musique
  2wikimultihopqa
  popqa

Examples:
  bash $(basename "$0")
  bash $(basename "$0") hotpotqa musique
  OPENIE_PARALLEL_WORKERS=4 EMBEDDING_BATCH_SIZE=6 bash $(basename "$0") 2wikimultihopqa

Notes:
  - No args means all 4 datasets are processed sequentially.
  - The number of datasets processed is controlled by how many dataset names you pass.
  - vLLM/OpenAI-compatible OpenIE server must already be running at OPENIE_API_BASE_URL.
USAGE
}

is_valid_dataset() {
  case "$1" in
    hotpotqa|musique|2wikimultihopqa|popqa) return 0 ;;
    *) return 1 ;;
  esac
}

corpus_path_for() {
  case "$1" in
    hotpotqa) echo "$CORPUS_ROOT/hotpotqa_corpus.json" ;;
    musique) echo "$CORPUS_ROOT/musique_corpus.json" ;;
    2wikimultihopqa) echo "$CORPUS_ROOT/2wikimultihopqa_corpus.json" ;;
    popqa) echo "$CORPUS_ROOT/popqa_corpus.json" ;;
    *) return 1 ;;
  esac
}

resolve_datasets() {
  local -a requested=()
  if [[ $# -eq 0 ]]; then
    requested=("${DEFAULT_DATASETS[@]}")
  elif [[ $# -eq 1 && "$1" == "all" ]]; then
    requested=("${DEFAULT_DATASETS[@]}")
  else
    requested=("$@")
  fi

  local -a resolved=()
  local ds
  for ds in "${requested[@]}"; do
    if ! is_valid_dataset "$ds"; then
      echo "[ERROR] Unknown dataset: $ds" >&2
      usage >&2
      exit 1
    fi
    resolved+=("$ds")
  done

  printf '%s\n' "${resolved[@]}"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  mapfile -t DATASETS < <(resolve_datasets "$@")

  mkdir -p "$REPO_ROOT/$LOG_ROOT"
  cd "$REPO_ROOT"

  echo "[INFO] Repo root              : $REPO_ROOT"
  echo "[INFO] Corpus root            : $CORPUS_ROOT"
  echo "[INFO] Cache root             : $CACHE_ROOT"
  echo "[INFO] Log root               : $LOG_ROOT"
  echo "[INFO] Datasets               : ${DATASETS[*]}"
  echo "[INFO] Graph mode             : $GRAPH_MODE"
  echo "[INFO] Index chunk unit       : $INDEX_CHUNK_UNIT"
  echo "[INFO] OpenIE API             : $OPENIE_API_BASE_URL"
  echo "[INFO] OpenIE workers         : $OPENIE_PARALLEL_WORKERS"
  echo "[INFO] Embedding batch size   : $EMBEDDING_BATCH_SIZE"
  echo "[INFO] Embedding max length   : $EMBEDDING_MAX_LENGTH"
  echo "[INFO] Embedding text chars   : $EMBEDDING_TEXT_MAX_CHARS"

  local ds corpus_path timestamp log_path
  for ds in "${DATASETS[@]}"; do
    corpus_path="$(corpus_path_for "$ds")"
    if [[ ! -f "$corpus_path" ]]; then
      echo "[ERROR] Corpus file not found for $ds: $corpus_path" >&2
      exit 1
    fi

    timestamp="$(date +%Y%m%d_%H%M%S)"
    log_path="$REPO_ROOT/$LOG_ROOT/${ds}_index_${timestamp}.log"

    echo
    echo "============================================================"
    echo "[INFO] Starting dataset: $ds"
    echo "[INFO] Corpus         : $corpus_path"
    echo "[INFO] Log            : $log_path"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python3 -m effirag.run_index \
      --corpus-path "$corpus_path" \
      --cache-dir "$CACHE_ROOT" \
      --force-rebuild "$FORCE_REBUILD" \
      --graph-mode "$GRAPH_MODE" \
      --index-chunk-unit "$INDEX_CHUNK_UNIT" \
      --openie-mode "$OPENIE_MODE" \
      --openie-model-name "$OPENIE_MODEL_NAME" \
      --openie-api-base-url "$OPENIE_API_BASE_URL" \
      --openie-parallel-workers "$OPENIE_PARALLEL_WORKERS" \
      --openie-text-max-chars "$OPENIE_TEXT_MAX_CHARS" \
      --openie-max-new-tokens "$OPENIE_MAX_NEW_TOKENS" \
      --embedding-enabled "$EMBEDDING_ENABLED" \
      --embedding-model-name "$EMBEDDING_MODEL_NAME" \
      --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
      --embedding-max-length "$EMBEDDING_MAX_LENGTH" \
      --embedding-text-max-chars "$EMBEDDING_TEXT_MAX_CHARS" \
      2>&1 | tee "$log_path"

    echo "[INFO] Finished dataset: $ds"
  done

  echo
  echo "[INFO] All requested datasets finished successfully."
}

main "$@"
