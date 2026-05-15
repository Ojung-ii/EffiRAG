#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6j_champion_bottleneck_audit}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"

mkdir -p "${OUT_ROOT}/logs"

# Existing run roots can be overridden by environment variables.
LEGACY_ROOT="${LEGACY_ROOT:-outputs/phase6i_gl_rcedr_v2_n100/runs/legacy_sota}"
UNIFIED_LARGE_ROOT="${UNIFIED_LARGE_ROOT:-outputs/phase6i_gl_rcedr_v2_n100/runs/unified_large}"
CANDIDATE_BOOST_ROOT="${CANDIDATE_BOOST_ROOT:-outputs/phase6i_gl_rcedr_v2_n100/runs/unified_candidate_recall_boost_v1}"
DENSITY_RERANK_ROOT="${DENSITY_RERANK_ROOT:-outputs/phase6i_gl_rcedr_v2_n100/runs/unified_candidate_recall_boost_density_rerank_v1}"
GL_RCEDR_V1_ROOT="${GL_RCEDR_V1_ROOT:-outputs/phase6i_gl_rcedr_v2_n100/runs/unified_gl_rcedr_v1}"
GL_RCEDR_V2_ROOT="${GL_RCEDR_V2_ROOT:-outputs/phase6i_gl_rcedr_v2_n100/runs/unified_gl_rcedr_v2}"

# Optional QA summaries.
HOTPOT_QA_JSON="${HOTPOT_QA_JSON:-}"
TWOWIKI_QA_JSON="${TWOWIKI_QA_JSON:-}"

echo "=== Phase-6J Champion Bottleneck Audit ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "DATASETS=${DATASETS}"

"${PYTHON}" scripts/audit_champion_bottlenecks.py \
  --datasets ${DATASETS} \
  --profiles \
    legacy_sota \
    unified_large \
    unified_candidate_recall_boost_v1 \
    unified_candidate_recall_boost_density_rerank_v1 \
    unified_gl_rcedr_v1 \
    unified_gl_rcedr_v2 \
  --roots \
    legacy_sota="${LEGACY_ROOT}" \
    unified_large="${UNIFIED_LARGE_ROOT}" \
    unified_candidate_recall_boost_v1="${CANDIDATE_BOOST_ROOT}" \
    unified_candidate_recall_boost_density_rerank_v1="${DENSITY_RERANK_ROOT}" \
    unified_gl_rcedr_v1="${GL_RCEDR_V1_ROOT}" \
    unified_gl_rcedr_v2="${GL_RCEDR_V2_ROOT}" \
  --qa-jsons \
    hotpotqa="${HOTPOT_QA_JSON}" \
    2wikimultihopqa="${TWOWIKI_QA_JSON}" \
  --output-dir "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_champion_bottlenecks.log"

echo "=== done ==="
echo "Report expected at: ${OUT_ROOT}/PHASE6J_CHAMPION_BOTTLENECK_AUDIT.md"

