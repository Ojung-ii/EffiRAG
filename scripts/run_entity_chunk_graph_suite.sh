#!/bin/bash
set -euo pipefail

# Full suite for entity-chunk graph next-method-design track.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

LIMIT_STAGE1="${LIMIT_STAGE1:-200}"
LIMIT_STAGE2="${LIMIT_STAGE2:-1000}"
RUN_REINDEX="${RUN_REINDEX:-false}"
RUN_STAGE2="${RUN_STAGE2:-true}"

if [ "${RUN_REINDEX}" = "true" ]; then
  DATASETS="hotpotqa,2wikimultihopqa" FORCE_REBUILD="${FORCE_REBUILD:-false}" bash scripts/run_entity_chunk_reindex.sh
fi

LIMIT_STAGE1="${LIMIT_STAGE1}" LIMIT_STAGE2="${LIMIT_STAGE2}" RUN_STAGE2="${RUN_STAGE2}" bash scripts/run_hotpot_entity_chunk_graph_ablation.sh
LIMIT_STAGE1="${LIMIT_STAGE1}" LIMIT_STAGE2="${LIMIT_STAGE2}" RUN_STAGE2="${RUN_STAGE2}" bash scripts/run_2wiki_entity_chunk_graph_ablation.sh

# Build HippoRAG2 gap tracking if q1000 summaries exist.
PYTHON_BIN="${PYTHON_BIN:-python3}"
GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

HOT_SUM="outputs/profiling/next_method_design/entity_chunk_graph/hotpotqa/hotpot_entity_chunk_graph_q${LIMIT_STAGE2}_latest.json"
WIKI_SUM="outputs/profiling/next_method_design/entity_chunk_graph/2wikimultihopqa/2wiki_entity_chunk_graph_q${LIMIT_STAGE2}_latest.json"
REF_JSON="outputs/profiling/next_method_design/retrieval_delivery_fusion/hipporag2_reference_metrics.json"
OUT_MD="outputs/profiling/next_method_design/entity_chunk_graph/hipporag2_gap_tracking_latest.md"
OUT_JSON="outputs/profiling/next_method_design/entity_chunk_graph/hipporag2_gap_tracking_latest.json"
FINAL_MD="outputs/profiling/next_method_design/entity_chunk_graph/entity_chunk_graph_final_interpretation_latest.md"
FINAL_JSON="outputs/profiling/next_method_design/entity_chunk_graph/entity_chunk_graph_final_interpretation_latest.json"

if [ -f "${HOT_SUM}" ] && [ -f "${WIKI_SUM}" ]; then
  "${PYTHON_BIN}" scripts/summarize_entity_chunk_graph_final.py \
    --hotpot-summary "${HOT_SUM}" \
    --wiki-summary "${WIKI_SUM}" \
    --output-md "${FINAL_MD}" \
    --output-json "${FINAL_JSON}" \
    --git-branch "${GIT_BRANCH}" \
    --git-commit "${GIT_COMMIT}" \
    --git-tag "${GIT_TAG}"
fi

if [ -f "${HOT_SUM}" ] && [ -f "${WIKI_SUM}" ] && [ -f "${REF_JSON}" ]; then
  "${PYTHON_BIN}" scripts/build_hipporag2_gap_tracking.py \
    --hotpot-summary "${HOT_SUM}" \
    --wiki-summary "${WIKI_SUM}" \
    --hipporag-ref "${REF_JSON}" \
    --output-md "${OUT_MD}" \
    --output-json "${OUT_JSON}" \
    --git-branch "${GIT_BRANCH}" \
    --git-commit "${GIT_COMMIT}" \
    --git-tag "${GIT_TAG}"
fi

echo "[entity_chunk_graph_suite] done"
