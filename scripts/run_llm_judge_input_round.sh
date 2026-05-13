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

DATASETS_CSV="hotpotqa,2wikimultihopqa"
SYSTEMS_CSV="effirag_light_separator_render,effirag_light_separator_copy_span_instruction,hipporag2"
EFFIRAG_OUTPUT_ROOT="/home/ojungii/EffiRAG/outputs/prompt_interface_bridge_resolution_round/20260428_120903"
HIPPORAG2_OUTPUT_ROOT="/home/ojungii/HippoRAG2/outputs/abgf_calibration"
QA_ROOT="/home/ojungii/HippoRAG2/dataset"
OUTPUT_ROOT="/home/ojungii/EffiRAG/outputs/llm_judge_input_round"
CHUNK_SIZE="50"
SEED="42"
MAX_RECORDS_PER_DATASET="1000"
PRINT_SUMMARY="true"

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_llm_judge_input_round.sh [options]

Primary:
  --datasets CSV
  --systems CSV
  --effirag-output-root PATH
  --hipporag2-output-root PATH
  --qa-root PATH
  --output-root PATH
  --chunk-size N
  --seed N
  --max-records-per-dataset N

Optional:
  --print-summary [true|false]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --systems) SYSTEMS_CSV="$2"; shift 2 ;;
    --effirag-output-root) EFFIRAG_OUTPUT_ROOT="$2"; shift 2 ;;
    --hipporag2-output-root) HIPPORAG2_OUTPUT_ROOT="$2"; shift 2 ;;
    --qa-root) QA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-records-per-dataset) MAX_RECORDS_PER_DATASET="$2"; shift 2 ;;
    --print-summary) PRINT_SUMMARY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" scripts/build_llm_judge_inputs.py \
  --datasets "${DATASETS_CSV}" \
  --systems "${SYSTEMS_CSV}" \
  --effirag-output-root "${EFFIRAG_OUTPUT_ROOT}" \
  --hipporag2-output-root "${HIPPORAG2_OUTPUT_ROOT}" \
  --qa-root "${QA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --chunk-size "${CHUNK_SIZE}" \
  --seed "${SEED}" \
  --max-records-per-dataset "${MAX_RECORDS_PER_DATASET}" \
  --print-summary "${PRINT_SUMMARY}"
