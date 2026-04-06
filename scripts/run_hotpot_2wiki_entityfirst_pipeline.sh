#!/usr/bin/env bash
set -euo pipefail

# Sequential pipeline for the refactored entity-first, chunk-grounded setup.
# Runs:
#   1) config audit
#   2) index build (hotpotqa, 2wikimultihopqa)
#   3) retrieval smoke/full
#   4) rag smoke/full
#
# Usage examples:
#   bash scripts/run_hotpot_2wiki_entityfirst_pipeline.sh
#   GENERATOR_MODE=openai_compat LLM_BASE_URL=http://localhost:8011/v1 LLM_API_KEY=EMPTY \
#     bash scripts/run_hotpot_2wiki_entityfirst_pipeline.sh
#   LIMIT_FULL=200 RUN_RAG=0 bash scripts/run_hotpot_2wiki_entityfirst_pipeline.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH:-.}"

# ------------------------------
# User-tunable variables
# ------------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIMIT_SMOKE="${LIMIT_SMOKE:-100}"
LIMIT_FULL="${LIMIT_FULL:-1000}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_INDEX="${RUN_INDEX:-1}"
RUN_RETRIEVAL="${RUN_RETRIEVAL:-1}"
RUN_RAG="${RUN_RAG:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"

# heuristic | openai_compat
GENERATOR_MODE="${GENERATOR_MODE:-heuristic}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
OPENIE_BASE_URL="${OPENIE_BASE_URL:-$LLM_BASE_URL}"
OPENIE_API_KEY="${OPENIE_API_KEY:-$LLM_API_KEY}"
OPENIE_TIMEOUT_SEC="${OPENIE_TIMEOUT_SEC:-120}"
OPENIE_PARALLEL_WORKERS="${OPENIE_PARALLEL_WORKERS:-6}"
OPENIE_TEXT_MAX_CHARS="${OPENIE_TEXT_MAX_CHARS:-2200}"
OPENIE_MAX_NEW_TOKENS="${OPENIE_MAX_NEW_TOKENS:-256}"
OPENIE_LOCAL_FILES_ONLY="${OPENIE_LOCAL_FILES_ONLY:-false}"
OPENIE_LOG_EVERY="${OPENIE_LOG_EVERY:-200}"

# Paths
HOTPOT_DATA="${HOTPOT_DATA:-data/qa/hotpotqa.json}"
HOTPOT_CORPUS="${HOTPOT_CORPUS:-/home/ojungii/HippoRAG2/dataset/hotpotqa_corpus.json}"
TWOWIKI_DATA="${TWOWIKI_DATA:-data/qa/2wikimultihopqa.json}"
TWOWIKI_CORPUS="${TWOWIKI_CORPUS:-/home/ojungii/HippoRAG2/dataset/2wikimultihopqa_corpus.json}"

CACHE_ROOT="${CACHE_ROOT:-outputs/index_cache_entityfirst_chunk32}"
RETR_OUT_ROOT="${RETR_OUT_ROOT:-outputs/retrieval_entityfirst_chunk32}"
RAG_OUT_ROOT="${RAG_OUT_ROOT:-outputs/rag_entityfirst_chunk32}"
LOG_ROOT="${LOG_ROOT:-outputs/run_logs_entityfirst_chunk32}"
mkdir -p "$CACHE_ROOT" "$RETR_OUT_ROOT" "$RAG_OUT_ROOT" "$LOG_ROOT"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_ROOT/pipeline_${RUN_TS}.log"

RETR_CFG="configs/canonical/retrieval_entity_first_chunk_grounded.yaml"
RAG_CFG="configs/canonical/rag_entity_first_chunk_grounded.yaml"

run_cmd() {
  echo "\n[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
}

run_python() {
  echo "\n[$(date '+%F %T')] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 $*" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 "$@" 2>&1 | tee -a "$MASTER_LOG"
}

run_audit() {
  if [[ "$RUN_AUDIT" != "1" ]]; then
    return 0
  fi
  run_python -m effirag.config_audit configs/canonical
}

run_index_for_dataset() {
  local dataset="$1"
  local corpus="$2"
  local cache_dir="$3"

  if [[ "$RUN_INDEX" != "1" ]]; then
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

  local extra_args=()
  if [[ "$GENERATOR_MODE" == "openai_compat" ]]; then
    extra_args+=(
      --generator openai_compat
      --model-name "$MODEL_NAME"
      --llm-base-url "$LLM_BASE_URL"
      --llm-api-key "$LLM_API_KEY"
    )
  else
    extra_args+=(--generator heuristic)
  fi

  run_python -m effirag.run_rag \
    --config "$RAG_CFG" \
    --dataset "$dataset" \
    --data-path "$data_path" \
    --global-corpus-path "$corpus" \
    --graph-cache-dir "$cache_dir" \
    --output-dir "$RAG_OUT_ROOT" \
    --force-rebuild-graph-index false \
    --limit "$limit" \
    "${extra_args[@]}"
}

main() {
  echo "Sequential entity-first chunk-grounded pipeline" | tee "$MASTER_LOG"
  echo "root=$ROOT_DIR" | tee -a "$MASTER_LOG"
  echo "generator_mode=$GENERATOR_MODE" | tee -a "$MASTER_LOG"
  echo "llm_base_url=$LLM_BASE_URL" | tee -a "$MASTER_LOG"
  echo "openie_base_url=$OPENIE_BASE_URL" | tee -a "$MASTER_LOG"
  echo "model_name=$MODEL_NAME" | tee -a "$MASTER_LOG"
  echo "openie_workers=$OPENIE_PARALLEL_WORKERS openie_text_max_chars=$OPENIE_TEXT_MAX_CHARS openie_max_new_tokens=$OPENIE_MAX_NEW_TOKENS" | tee -a "$MASTER_LOG"
  echo "limit_smoke=$LIMIT_SMOKE limit_full=$LIMIT_FULL" | tee -a "$MASTER_LOG"
  echo "cache_root=$CACHE_ROOT" | tee -a "$MASTER_LOG"
  echo "retr_output_root=$RETR_OUT_ROOT" | tee -a "$MASTER_LOG"
  echo "rag_output_root=$RAG_OUT_ROOT" | tee -a "$MASTER_LOG"

  run_audit

  run_index_for_dataset hotpotqa "$HOTPOT_CORPUS" "$CACHE_ROOT/hotpotqa"
  run_index_for_dataset 2wikimultihopqa "$TWOWIKI_CORPUS" "$CACHE_ROOT/2wikimultihopqa"

  if [[ "$RUN_RETRIEVAL" == "1" && "$RUN_SMOKE" == "1" ]]; then
    run_retrieval_for_dataset hotpotqa "$HOTPOT_DATA" "$HOTPOT_CORPUS" "$CACHE_ROOT/hotpotqa" "$LIMIT_SMOKE"
    run_retrieval_for_dataset 2wikimultihopqa "$TWOWIKI_DATA" "$TWOWIKI_CORPUS" "$CACHE_ROOT/2wikimultihopqa" "$LIMIT_SMOKE"
  fi

  if [[ "$RUN_RETRIEVAL" == "1" && "$RUN_FULL" == "1" ]]; then
    run_retrieval_for_dataset hotpotqa "$HOTPOT_DATA" "$HOTPOT_CORPUS" "$CACHE_ROOT/hotpotqa" "$LIMIT_FULL"
    run_retrieval_for_dataset 2wikimultihopqa "$TWOWIKI_DATA" "$TWOWIKI_CORPUS" "$CACHE_ROOT/2wikimultihopqa" "$LIMIT_FULL"
  fi

  if [[ "$RUN_RAG" == "1" && "$RUN_SMOKE" == "1" ]]; then
    run_rag_for_dataset hotpotqa "$HOTPOT_DATA" "$HOTPOT_CORPUS" "$CACHE_ROOT/hotpotqa" "$LIMIT_SMOKE"
    run_rag_for_dataset 2wikimultihopqa "$TWOWIKI_DATA" "$TWOWIKI_CORPUS" "$CACHE_ROOT/2wikimultihopqa" "$LIMIT_SMOKE"
  fi

  if [[ "$RUN_RAG" == "1" && "$RUN_FULL" == "1" ]]; then
    run_rag_for_dataset hotpotqa "$HOTPOT_DATA" "$HOTPOT_CORPUS" "$CACHE_ROOT/hotpotqa" "$LIMIT_FULL"
    run_rag_for_dataset 2wikimultihopqa "$TWOWIKI_DATA" "$TWOWIKI_CORPUS" "$CACHE_ROOT/2wikimultihopqa" "$LIMIT_FULL"
  fi

  echo "\nDone. Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
}

main "$@"
