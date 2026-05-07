#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "/home/ojungii/miniconda3/envs/effirag/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

INPUT_ROUND="${INPUT_ROUND:-/home/ojungii/EffiRAG/outputs/llm_judge_input_round/20260428_083134}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ojungii/EffiRAG/outputs/llm_judge_qwen32b_round}"
BASE_URL="${BASE_URL:-http://localhost:8012/v1}"
MODEL="${MODEL:-qwen2.5-32b-awq-judge}"
API_KEY="${API_KEY:-EMPTY}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_RETRIES="${MAX_RETRIES:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
OVERWRITE="${OVERWRITE:-false}"
USE_GUIDED_JSON="${USE_GUIDED_JSON:-true}"
USE_RESPONSE_FORMAT_JSON="${USE_RESPONSE_FORMAT_JSON:-false}"
MAX_RECORDS="${MAX_RECORDS:-0}"
CONCURRENCY="${CONCURRENCY:-4}"

OVERWRITE_FLAG=()
if [[ "${OVERWRITE}" == "true" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

export PYTHONPATH="${PYTHONPATH:-.}"

"${PYTHON_BIN}" scripts/run_qwen32b_llm_judge.py \
  --input-root "${INPUT_ROUND}/judge_chunks" \
  --prompt-path "${INPUT_ROUND}/prompts/llm_judge_rubric.md" \
  --mapping-root "${INPUT_ROUND}/hidden_mapping" \
  --output-root "${OUTPUT_ROOT}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --model "${MODEL}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-retries "${MAX_RETRIES}" \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --use-guided-json "${USE_GUIDED_JSON}" \
  --use-response-format-json "${USE_RESPONSE_FORMAT_JSON}" \
  --concurrency "${CONCURRENCY}" \
  --max-records "${MAX_RECORDS}" \
  "${OVERWRITE_FLAG[@]}"
