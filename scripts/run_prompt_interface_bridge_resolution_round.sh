#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "/home/ojungii/miniconda3/envs/effirag/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM="false"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATASETS_CSV="hotpotqa,2wikimultihopqa"
VARIANTS_CSV="light_separator_render,light_separator_bridge_instruction,light_separator_copy_span_instruction,light_separator_title_grounded_format,light_separator_final_answer_oneshot"
N_SAMPLES="1000"
QA_ROOT="data/qa"
CORPUS_ROOT="/home/ojungii/HippoRAG2/dataset"
EFFIRAG_CHAMPION_OUTPUT_ROOT="outputs/light_separator_lockin_round"
OUTPUT_ROOT="outputs/prompt_interface_bridge_resolution_round"
GRAPH_CACHE_DIR="outputs/index_cache"
LLM_BASE_URL="http://localhost:8011/v1"
LLM_API_KEY="EMPTY"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PRINT_TABLES="true"
SKIP_COMPLETED="true"

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_prompt_interface_bridge_resolution_round.sh [options]

Primary:
  --datasets CSV
  --variants CSV
  --n-samples N
  --qa-root PATH
  --corpus-root PATH
  --effirag-champion-output-root PATH
  --output-root PATH
  --graph-cache-dir PATH
  --llm-base-url URL
  --llm-api-key KEY
  --model-name NAME

Optional:
  --print-tables [true|false]
  --skip-completed [true|false]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --variants) VARIANTS_CSV="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --qa-root) QA_ROOT="$2"; shift 2 ;;
    --corpus-root) CORPUS_ROOT="$2"; shift 2 ;;
    --effirag-champion-output-root) EFFIRAG_CHAMPION_OUTPUT_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --graph-cache-dir) GRAPH_CACHE_DIR="$2"; shift 2 ;;
    --llm-base-url) LLM_BASE_URL="$2"; shift 2 ;;
    --llm-api-key) LLM_API_KEY="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --print-tables) PRINT_TABLES="$2"; shift 2 ;;
    --skip-completed) SKIP_COMPLETED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" scripts/prompt_interface_bridge_resolution_round.py \
  --datasets "${DATASETS_CSV}" \
  --variants "${VARIANTS_CSV}" \
  --n-samples "${N_SAMPLES}" \
  --qa-root "${QA_ROOT}" \
  --corpus-root "${CORPUS_ROOT}" \
  --effirag-champion-output-root "${EFFIRAG_CHAMPION_OUTPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --graph-cache-dir "${GRAPH_CACHE_DIR}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --llm-api-key "${LLM_API_KEY}" \
  --model-name "${MODEL_NAME}" \
  --print-tables "${PRINT_TABLES}" \
  --skip-completed "${SKIP_COMPLETED}"
