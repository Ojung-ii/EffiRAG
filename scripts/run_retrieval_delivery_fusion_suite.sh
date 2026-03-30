#!/bin/bash
set -euo pipefail

# Run retrieval+delivery fusion experiments for Hotpot/2Wiki and build combined reports.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIMIT_STAGE1="${LIMIT_STAGE1:-200}"
LIMIT_STAGE2="${LIMIT_STAGE2:-1000}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_ROOT="outputs/profiling/next_method_design/retrieval_delivery_fusion"
HIPPORAG_REF="${HIPPORAG_REF:-outputs/profiling/next_method_design/retrieval_delivery_fusion/hipporag2_reference_metrics.json}"

mkdir -p "${PROFILE_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
LIMIT_STAGE1="${LIMIT_STAGE1}" LIMIT_STAGE2="${LIMIT_STAGE2}" PYTHON_BIN="${PYTHON_BIN}" \
  bash scripts/run_hotpot_retrieval_delivery_fusion.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
LIMIT_STAGE1="${LIMIT_STAGE1}" LIMIT_STAGE2="${LIMIT_STAGE2}" PYTHON_BIN="${PYTHON_BIN}" \
  bash scripts/run_2wiki_retrieval_delivery_fusion.sh

HOTPOT_Q1000="${PROFILE_ROOT}/hotpotqa/hotpot_retrieval_delivery_fusion_q${LIMIT_STAGE2}_latest.json"
WIKI_Q1000="${PROFILE_ROOT}/2wikimultihopqa/2wiki_retrieval_delivery_fusion_q${LIMIT_STAGE2}_latest.json"

if [ ! -f "${HOTPOT_Q1000}" ] || [ ! -f "${WIKI_Q1000}" ]; then
  echo "[ERROR] Missing q${LIMIT_STAGE2} summary files." >&2
  echo "  hotpot=${HOTPOT_Q1000}" >&2
  echo "  2wiki=${WIKI_Q1000}" >&2
  exit 1
fi

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

FINAL_MD="${PROFILE_ROOT}/retrieval_delivery_fusion_final_interpretation_${RUN_STAMP}.md"
FINAL_JSON="${PROFILE_ROOT}/retrieval_delivery_fusion_final_interpretation_${RUN_STAMP}.json"

"${PYTHON_BIN}" scripts/summarize_retrieval_delivery_fusion_final.py \
  --hotpot-summary "${HOTPOT_Q1000}" \
  --wiki-summary "${WIKI_Q1000}" \
  --output-md "${FINAL_MD}" \
  --output-json "${FINAL_JSON}" \
  --git-branch "${GIT_BRANCH}" \
  --git-commit "${GIT_COMMIT}" \
  --git-tag "${GIT_TAG}"

GAP_MD="${PROFILE_ROOT}/hipporag2_gap_tracking_${RUN_STAMP}.md"
GAP_JSON="${PROFILE_ROOT}/hipporag2_gap_tracking_${RUN_STAMP}.json"

"${PYTHON_BIN}" scripts/build_hipporag2_gap_tracking.py \
  --hotpot-summary "${HOTPOT_Q1000}" \
  --wiki-summary "${WIKI_Q1000}" \
  --hipporag-ref "${HIPPORAG_REF}" \
  --output-md "${GAP_MD}" \
  --output-json "${GAP_JSON}" \
  --git-branch "${GIT_BRANCH}" \
  --git-commit "${GIT_COMMIT}" \
  --git-tag "${GIT_TAG}"

cp -f "${FINAL_MD}" "${PROFILE_ROOT}/retrieval_delivery_fusion_final_interpretation_latest.md"
cp -f "${FINAL_JSON}" "${PROFILE_ROOT}/retrieval_delivery_fusion_final_interpretation_latest.json"
cp -f "${GAP_MD}" "${PROFILE_ROOT}/hipporag2_gap_tracking_latest.md"
cp -f "${GAP_JSON}" "${PROFILE_ROOT}/hipporag2_gap_tracking_latest.json"

echo "[retrieval_delivery_fusion_suite] done"
echo "  final_interpretation=${FINAL_MD}"
echo "  gap_tracking=${GAP_MD}"
