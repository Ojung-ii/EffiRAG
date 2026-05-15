#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
GPU="${GPU:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6i_gl_rcedr_v2}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6I GL-RCEDR v2 retrieval-only run ==="
echo "GPU=${GPU}"
echo "DATASETS=${DATASETS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"

run_unified () {
  local profile="$1"
  local tag="$2"

  echo "=== unified profile: ${profile} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --retrieval-only \
    --output-root "${OUT_ROOT}/runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_${tag}.log"
}

echo "=== 1. legacy_sota ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_sota_config_4ds.py \
  --mode locked_precomputed \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/legacy_sota" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_legacy_sota.log"

run_unified unified_large unified_large
run_unified unified_candidate_recall_boost_v1 unified_candidate_recall_boost_v1
run_unified unified_candidate_recall_boost_density_rerank_v1 unified_candidate_recall_boost_density_rerank_v1
run_unified unified_gl_rcedr_v1 unified_gl_rcedr_v1
run_unified unified_gl_rcedr_v2 unified_gl_rcedr_v2
run_unified unified_gl_rcedr_v2_no_adaptive_bridge unified_gl_rcedr_v2_no_adaptive_bridge
run_unified unified_gl_rcedr_v2_no_cost unified_gl_rcedr_v2_no_cost
run_unified unified_gl_rcedr_v2_no_redundancy unified_gl_rcedr_v2_no_redundancy
run_unified unified_gl_rcedr_v2_density_first unified_gl_rcedr_v2_density_first

echo "=== evidence-flow audit ==="
PYTHONPATH=. "${PYTHON}" scripts/audit_evidence_flow.py \
  --datasets ${DATASETS} \
  --profiles \
    legacy_sota \
    unified_large \
    unified_candidate_recall_boost_v1 \
    unified_candidate_recall_boost_density_rerank_v1 \
    unified_gl_rcedr_v1 \
    unified_gl_rcedr_v2 \
    unified_gl_rcedr_v2_no_adaptive_bridge \
    unified_gl_rcedr_v2_no_cost \
    unified_gl_rcedr_v2_no_redundancy \
    unified_gl_rcedr_v2_density_first \
  --roots \
    legacy_sota="${OUT_ROOT}/runs/legacy_sota" \
    unified_large="${OUT_ROOT}/runs/unified_large" \
    unified_candidate_recall_boost_v1="${OUT_ROOT}/runs/unified_candidate_recall_boost_v1" \
    unified_candidate_recall_boost_density_rerank_v1="${OUT_ROOT}/runs/unified_candidate_recall_boost_density_rerank_v1" \
    unified_gl_rcedr_v1="${OUT_ROOT}/runs/unified_gl_rcedr_v1" \
    unified_gl_rcedr_v2="${OUT_ROOT}/runs/unified_gl_rcedr_v2" \
    unified_gl_rcedr_v2_no_adaptive_bridge="${OUT_ROOT}/runs/unified_gl_rcedr_v2_no_adaptive_bridge" \
    unified_gl_rcedr_v2_no_cost="${OUT_ROOT}/runs/unified_gl_rcedr_v2_no_cost" \
    unified_gl_rcedr_v2_no_redundancy="${OUT_ROOT}/runs/unified_gl_rcedr_v2_no_redundancy" \
    unified_gl_rcedr_v2_density_first="${OUT_ROOT}/runs/unified_gl_rcedr_v2_density_first" \
  --output-dir "${OUT_ROOT}/audit" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_evidence_flow.log"

SUMMARY_PATH="${OUT_ROOT}/PHASE6I_GL_RCEDR_V2_SUMMARY.md"
if [[ ! -f "${SUMMARY_PATH}" ]]; then
  cat > "${SUMMARY_PATH}" <<'EOF'
# Phase-6I GL-RCEDR v2 Retrieval-Only Results

## 1. Purpose

This experiment evaluates GL-RCEDR v2, which simplifies the retrieval pipeline using a unified marginal evidence utility for both global seed selection and local evidence refinement.

## 2. Compared Profiles

| profile | description |
|---|---|
| legacy_sota | heuristic evidence-density teacher/reference |
| unified_large | clean unified baseline |
| unified_candidate_recall_boost_v1 | candidate recall boost baseline |
| unified_candidate_recall_boost_density_rerank_v1 | density-first rerank baseline |
| unified_gl_rcedr_v1 | previous GL-RCEDR baseline |
| unified_gl_rcedr_v2 | main v2 method |
| unified_gl_rcedr_v2_no_adaptive_bridge | ablation |
| unified_gl_rcedr_v2_no_cost | ablation |
| unified_gl_rcedr_v2_no_redundancy | ablation |
| unified_gl_rcedr_v2_density_first | ablation |

## 3. Evidence Flow Results

| dataset | profile | candidate_sf_R | selected_sf_R | rendered_sf_R | rendered_tokens | rendered_sf_F1_per_1k_tokens |
|---|---|---:|---:|---:|---:|---:|

## 4. Failure Taxonomy

| dataset | profile | retrieval_miss_rate | selector_drop_rate | render_drop_rate | low_density_rate | generation_ready_rate |
|---|---|---:|---:|---:|---:|---:|

## 5. Ablation Analysis

| ablation | expected effect | observed |
|---|---|---|
| no_adaptive_bridge | tests query-adaptive bridge utility | TBD |
| no_cost | tests token cost penalty | TBD |
| no_redundancy | tests redundancy pruning | TBD |
| density_first | tests failed density-first hypothesis | TBD |

## 6. Gate Decision

| criterion | pass/fail | note |
|---|---|---|
| candidate recall preserved | TBD | |
| selected_sf_R >= unified_large | TBD | |
| rendered_sf_R >= unified_large | TBD | |
| render_drop near zero | TBD | |
| tokens <= GL-RCEDR v1 | TBD | |
| density > GL-RCEDR v1 | TBD | |
| no dataset-specific heuristic | TBD | |

## 7. Recommendation

- TBD
EOF
fi

echo "=== cleanup ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

echo "=== done ==="
echo "Audit output expected at: ${OUT_ROOT}/audit"
