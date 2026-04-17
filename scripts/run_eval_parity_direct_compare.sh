#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-outputs/profiling/eval_parity}"
INPUT_JSONL="${INPUT_JSONL:-}"
OUT_MD="${OUT_MD:-${OUT_DIR}/direct_compare_report.md}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/direct_compare_report.json}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${OUT_DIR}"

if [ -z "${INPUT_JSONL}" ]; then
  INPUT_JSONL="$(find "${REPO_ROOT}/outputs/rag" -maxdepth 4 -name 'rag_query_results.jsonl' | sort | tail -n 1)"
fi

if [ -z "${INPUT_JSONL}" ] || [ ! -f "${INPUT_JSONL}" ]; then
  echo "[ERROR] input jsonl not found: ${INPUT_JSONL}" >&2
  exit 1
fi

(
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" scripts/eval_parity_direct_compare.py \
    --input-jsonl "${INPUT_JSONL}" \
    --output-md "${OUT_MD}" \
    --output-json "${OUT_JSON}"
)

echo "[DONE] Direct compare report generated:"
echo "- ${OUT_MD}"
echo "- ${OUT_JSON}"
