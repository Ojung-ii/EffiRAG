#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH:-.}"

usage() {
  cat <<USAGE
Usage:
  bash scripts/run_entityfirst_pipeline.sh --dataset hotpotqa [options]
  bash scripts/run_entityfirst_pipeline.sh --dataset 2wikimultihopqa [options]
  bash scripts/run_entityfirst_pipeline.sh --dataset both [options]

Required:
  --dataset <hotpotqa|2wikimultihopqa|both>

Options:
  --smoke-only               Run only smoke phase (default: off)
  --retrieval-only           Skip RAG and run indexing+retrieval only
  --no-index                 Reuse existing index cache
  --limit-smoke N            Smoke sample count (default: env LIMIT_SMOKE or 20)
  --limit-full N             Full sample count (default: env LIMIT_FULL or 1000)
  --cuda-devices ID          CUDA_VISIBLE_DEVICES (default: env or 0)
  --llm-base-url URL         vLLM OpenAI-compatible endpoint (default: env or http://localhost:8011/v1)
  --llm-api-key KEY          API key for endpoint (default: env or EMPTY)
  --model-name NAME          Model name passed to run_rag/run_index (default: env or Qwen/Qwen2.5-7B-Instruct)
  --openie-base-url URL      OpenIE endpoint (default: OPENIE_BASE_URL or --llm-base-url)
  --openie-api-key KEY       OpenIE API key (default: OPENIE_API_KEY or --llm-api-key)
  --openie-timeout-sec N     OpenIE request timeout seconds (default: OPENIE_TIMEOUT_SEC or 120)
  --openie-parallel-workers N OpenIE parallel workers (default: OPENIE_PARALLEL_WORKERS or 6)
  --openie-text-max-chars N  OpenIE input max chars (default: OPENIE_TEXT_MAX_CHARS or 2200)
  --openie-max-new-tokens N  OpenIE max new tokens (default: OPENIE_MAX_NEW_TOKENS or 256)
  --openie-local-files-only BOOL Use HF local-only for OpenIE (default: OPENIE_LOCAL_FILES_ONLY or false)
  --openie-log-every N       OpenIE progress heartbeat interval (default: OPENIE_LOG_EVERY or 200)
  -h, --help                 Show this help

Examples:
  bash scripts/run_entityfirst_pipeline.sh --dataset hotpotqa --smoke-only
  bash scripts/run_entityfirst_pipeline.sh --dataset hotpotqa --smoke-only --retrieval-only
  nohup bash scripts/run_entityfirst_pipeline.sh --dataset hotpotqa > hotpot_run.out 2>&1 &
USAGE
}

DATASET=""
SMOKE_ONLY=0
RETRIEVAL_ONLY=0
RUN_INDEX="${RUN_INDEX:-1}"
LIMIT_SMOKE="${LIMIT_SMOKE:-20}"
LIMIT_FULL="${LIMIT_FULL:-1000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
GENERATOR_MODE="${GENERATOR_MODE:-openai_compat}"
OPENIE_BASE_URL="${OPENIE_BASE_URL:-}"
OPENIE_API_KEY="${OPENIE_API_KEY:-}"
OPENIE_TIMEOUT_SEC="${OPENIE_TIMEOUT_SEC:-120}"
OPENIE_PARALLEL_WORKERS="${OPENIE_PARALLEL_WORKERS:-6}"
OPENIE_TEXT_MAX_CHARS="${OPENIE_TEXT_MAX_CHARS:-2200}"
OPENIE_MAX_NEW_TOKENS="${OPENIE_MAX_NEW_TOKENS:-256}"
OPENIE_LOCAL_FILES_ONLY="${OPENIE_LOCAL_FILES_ONLY:-false}"
OPENIE_LOG_EVERY="${OPENIE_LOG_EVERY:-200}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    --smoke-only)
      SMOKE_ONLY=1
      shift
      ;;
    --retrieval-only)
      RETRIEVAL_ONLY=1
      shift
      ;;
    --no-index)
      RUN_INDEX=0
      shift
      ;;
    --limit-smoke)
      LIMIT_SMOKE="${2:-}"
      shift 2
      ;;
    --limit-full)
      LIMIT_FULL="${2:-}"
      shift 2
      ;;
    --cuda-devices)
      CUDA_VISIBLE_DEVICES="${2:-}"
      shift 2
      ;;
    --llm-base-url)
      LLM_BASE_URL="${2:-}"
      shift 2
      ;;
    --llm-api-key)
      LLM_API_KEY="${2:-}"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="${2:-}"
      shift 2
      ;;
    --openie-base-url)
      OPENIE_BASE_URL="${2:-}"
      shift 2
      ;;
    --openie-api-key)
      OPENIE_API_KEY="${2:-}"
      shift 2
      ;;
    --openie-timeout-sec)
      OPENIE_TIMEOUT_SEC="${2:-}"
      shift 2
      ;;
    --openie-parallel-workers)
      OPENIE_PARALLEL_WORKERS="${2:-}"
      shift 2
      ;;
    --openie-text-max-chars)
      OPENIE_TEXT_MAX_CHARS="${2:-}"
      shift 2
      ;;
    --openie-max-new-tokens)
      OPENIE_MAX_NEW_TOKENS="${2:-}"
      shift 2
      ;;
    --openie-local-files-only)
      OPENIE_LOCAL_FILES_ONLY="${2:-}"
      shift 2
      ;;
    --openie-log-every)
      OPENIE_LOG_EVERY="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$OPENIE_BASE_URL" ]]; then
  OPENIE_BASE_URL="$LLM_BASE_URL"
fi
if [[ -z "$OPENIE_API_KEY" ]]; then
  OPENIE_API_KEY="$LLM_API_KEY"
fi

if [[ -z "$DATASET" ]]; then
  echo "--dataset is required." >&2
  usage
  exit 1
fi

case "$DATASET" in
  hotpotqa|2wikimultihopqa|both) ;;
  *)
    echo "Invalid --dataset: $DATASET" >&2
    usage
    exit 1
    ;;
esac

HOTPOT_DATA="${HOTPOT_DATA:-data/qa/hotpotqa.json}"
HOTPOT_CORPUS="${HOTPOT_CORPUS:-data/hotpotqa_corpus.json}"
TWOWIKI_DATA="${TWOWIKI_DATA:-data/qa/2wikimultihopqa.json}"
TWOWIKI_CORPUS="${TWOWIKI_CORPUS:-data/2wikimultihopqa_corpus.json}"

CACHE_ROOT="${CACHE_ROOT:-outputs/index_cache_entityfirst_chunk32}"
RETR_OUT_ROOT="${RETR_OUT_ROOT:-outputs/retrieval_entityfirst_chunk32}"
RAG_OUT_ROOT="${RAG_OUT_ROOT:-outputs/rag_entityfirst_chunk32}"
LOG_ROOT="${LOG_ROOT:-outputs/run_logs_entityfirst_chunk32}"
mkdir -p "$CACHE_ROOT" "$RETR_OUT_ROOT" "$RAG_OUT_ROOT" "$LOG_ROOT"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_ROOT/${DATASET}_pipeline_${RUN_TS}.log"

RETR_CFG="configs/canonical/retrieval_entity_first_chunk_grounded.yaml"
RAG_CFG="configs/canonical/rag_entity_first_chunk_grounded.yaml"

run_python() {
  echo "\n[$(date '+%F %T')] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 $*" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 "$@" 2>&1 | tee -a "$MASTER_LOG"
}

run_audit() {
  run_python -m effirag.config_audit configs/canonical
}

run_index_for_dataset() {
  local dataset="$1"
  local corpus="$2"
  local cache_dir="$3"

  if [[ "$RUN_INDEX" != "1" ]]; then
    echo "[skip] index for $dataset" | tee -a "$MASTER_LOG"
    return 0
  fi

  run_python -m effirag.run_index \
    --dataset "$dataset" \
    --corpus-path "$corpus" \
    --cache-dir "$cache_dir" \
    --force-rebuild true \
    --graph-mode entity_chunk_graph \
    --index-chunk-unit passage \
    --embedding-enabled true \
    --embedding-model-name nvidia/NV-Embed-v2 \
    --embedding-batch-size 16 \
    --embedding-max-length 192 \
    --embedding-text-max-chars 600 \
    --semantic-scan-batch-size 8192 \
    --openie-mode llm \
    --openie-model-name "$MODEL_NAME" \
    --openie-text-max-chars "$OPENIE_TEXT_MAX_CHARS" \
    --openie-max-new-tokens "$OPENIE_MAX_NEW_TOKENS" \
    --openie-local-files-only "$OPENIE_LOCAL_FILES_ONLY" \
    --openie-api-base-url "$OPENIE_BASE_URL" \
    --openie-api-key "$OPENIE_API_KEY" \
    --openie-api-timeout-sec "$OPENIE_TIMEOUT_SEC" \
    --openie-parallel-workers "$OPENIE_PARALLEL_WORKERS" \
    --openie-log-every "$OPENIE_LOG_EVERY"
}

run_retrieval_for_dataset() {
  local dataset="$1"
  local data_path="$2"
  local corpus="$3"
  local cache_dir="$4"
  local limit="$5"

  run_python -m effirag.run_retrieval \
    --config "$RETR_CFG" \
    --dataset "$dataset" \
    --data-path "$data_path" \
    --global-corpus-path "$corpus" \
    --graph-cache-dir "$cache_dir" \
    --output-dir "$RETR_OUT_ROOT" \
    --force-rebuild-graph-index false \
    --limit "$limit"
}

run_rag_for_dataset() {
  local dataset="$1"
  local data_path="$2"
  local corpus="$3"
  local cache_dir="$4"
  local limit="$5"

  run_python -m effirag.run_rag \
    --config "$RAG_CFG" \
    --dataset "$dataset" \
    --data-path "$data_path" \
    --global-corpus-path "$corpus" \
    --graph-cache-dir "$cache_dir" \
    --output-dir "$RAG_OUT_ROOT" \
    --force-rebuild-graph-index false \
    --limit "$limit" \
    --generator "$GENERATOR_MODE" \
    --model-name "$MODEL_NAME" \
    --llm-base-url "$LLM_BASE_URL" \
    --llm-api-key "$LLM_API_KEY"
}

run_for_dataset() {
  local dataset="$1"
  local data_path="$2"
  local corpus="$3"
  local cache_dir="$4"

  run_index_for_dataset "$dataset" "$corpus" "$cache_dir"
  run_retrieval_for_dataset "$dataset" "$data_path" "$corpus" "$cache_dir" "$LIMIT_SMOKE"

  if [[ "$RETRIEVAL_ONLY" == "1" ]]; then
    return 0
  fi

  run_rag_for_dataset "$dataset" "$data_path" "$corpus" "$cache_dir" "$LIMIT_SMOKE"

  if [[ "$SMOKE_ONLY" != "1" ]]; then
    run_retrieval_for_dataset "$dataset" "$data_path" "$corpus" "$cache_dir" "$LIMIT_FULL"
    run_rag_for_dataset "$dataset" "$data_path" "$corpus" "$cache_dir" "$LIMIT_FULL"
  fi
}

echo "Entity-first chunk-grounded pipeline" | tee "$MASTER_LOG"
echo "dataset=$DATASET" | tee -a "$MASTER_LOG"
echo "generator_mode=$GENERATOR_MODE" | tee -a "$MASTER_LOG"
echo "llm_base_url=$LLM_BASE_URL" | tee -a "$MASTER_LOG"
echo "openie_base_url=$OPENIE_BASE_URL" | tee -a "$MASTER_LOG"
echo "model_name=$MODEL_NAME" | tee -a "$MASTER_LOG"
echo "openie_workers=$OPENIE_PARALLEL_WORKERS openie_text_max_chars=$OPENIE_TEXT_MAX_CHARS openie_max_new_tokens=$OPENIE_MAX_NEW_TOKENS" | tee -a "$MASTER_LOG"
echo "limit_smoke=$LIMIT_SMOKE limit_full=$LIMIT_FULL" | tee -a "$MASTER_LOG"
echo "cache_root=$CACHE_ROOT" | tee -a "$MASTER_LOG"

run_audit

if [[ "$DATASET" == "hotpotqa" || "$DATASET" == "both" ]]; then
  run_for_dataset hotpotqa "$HOTPOT_DATA" "$HOTPOT_CORPUS" "$CACHE_ROOT/hotpotqa"
fi

if [[ "$DATASET" == "2wikimultihopqa" || "$DATASET" == "both" ]]; then
  run_for_dataset 2wikimultihopqa "$TWOWIKI_DATA" "$TWOWIKI_CORPUS" "$CACHE_ROOT/2wikimultihopqa"
fi

echo "\nDone. Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
