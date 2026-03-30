#!/bin/bash
set -euo pipefail

# Run both q1000 chunk-grounded revalidations and build final interpretation.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

LIMIT="${LIMIT:-1000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/next_method_design/chunk_grounded_context_q1000}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${PROFILE_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" LIMIT="${LIMIT}" PYTHON_BIN="${PYTHON_BIN}" \
  bash scripts/run_hotpot_chunk_grounded_q1000.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" LIMIT="${LIMIT}" PYTHON_BIN="${PYTHON_BIN}" \
  bash scripts/run_2wiki_chunk_grounded_q1000.sh

HOTPOT_SUMMARY="${PROFILE_ROOT}/chunk_grounded_context_hotpotqa_q1000_summary_latest.json"
WIKI_SUMMARY="${PROFILE_ROOT}/chunk_grounded_context_2wikimultihopqa_q1000_summary_latest.json"

if [ ! -f "${HOTPOT_SUMMARY}" ] || [ ! -f "${WIKI_SUMMARY}" ]; then
  echo "[ERROR] Missing latest dataset summaries; cannot build final interpretation." >&2
  exit 1
fi

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

OUT_MD="${PROFILE_ROOT}/chunk_grounded_context_q1000_final_interpretation_${RUN_STAMP}.md"
OUT_JSON="${PROFILE_ROOT}/chunk_grounded_context_q1000_final_interpretation_${RUN_STAMP}.json"

"${PYTHON_BIN}" scripts/summarize_chunk_grounded_q1000_final.py \
  --hotpot-summary "${HOTPOT_SUMMARY}" \
  --wiki-summary "${WIKI_SUMMARY}" \
  --output-md "${OUT_MD}" \
  --output-json "${OUT_JSON}" \
  --git-branch "${GIT_BRANCH}" \
  --git-commit "${GIT_COMMIT}" \
  --git-tag "${GIT_TAG}"

cp -f "${OUT_MD}" "${PROFILE_ROOT}/chunk_grounded_context_q1000_final_interpretation_latest.md"
cp -f "${OUT_JSON}" "${PROFILE_ROOT}/chunk_grounded_context_q1000_final_interpretation_latest.json"

echo "[chunk_grounded_q1000_suite] done"
echo "  final_md=${OUT_MD}"
echo "  final_json=${OUT_JSON}"
