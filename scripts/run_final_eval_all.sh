#!/bin/bash
set -euo pipefail

LIMIT="${LIMIT:-200}"
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-false}"
DATASET="${DATASET:-2wikimultihopqa}"
DATA_PATH="${DATA_PATH:-data/qa/2wikimultihopqa.json}"
GLOBAL_CORPUS_PATH="${GLOBAL_CORPUS_PATH:-/home/ojungii/HippoRAG2/dataset/2wikimultihopqa_corpus.json}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/final_eval}"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"

for MODE in speed_default quality_variant ablation_off; do
  echo "[final_eval_all] mode=${MODE}"
  LIMIT="${LIMIT}" \
  RETRIEVAL_ONLY="${RETRIEVAL_ONLY}" \
  DATASET="${DATASET}" \
  DATA_PATH="${DATA_PATH}" \
  GLOBAL_CORPUS_PATH="${GLOBAL_CORPUS_PATH}" \
  GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  GENERATOR="${GENERATOR}" \
  MODEL_NAME="${MODEL_NAME}" \
  LLM_BASE_URL="${LLM_BASE_URL}" \
  LLM_API_KEY="${LLM_API_KEY}" \
  scripts/run_final_eval.sh "${MODE}"
done
