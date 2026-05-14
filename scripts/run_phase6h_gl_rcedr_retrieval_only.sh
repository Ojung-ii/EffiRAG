#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
GPU="${GPU:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6h_gl_rcedr}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6H GL-RCEDR retrieval-only overnight run ==="
echo "GPU=${GPU}"
echo "DATASETS=${DATASETS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"

echo "=== 1. legacy_sota ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_sota_config_4ds.py \
  --mode locked_precomputed \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/legacy_sota" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_legacy_sota.log"

echo "=== 2. unified_large ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_unified_config_4ds.py \
  --profile unified_large \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/unified_large" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_unified_large.log"

echo "=== 3. candidate_recall_boost_v1 ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_unified_config_4ds.py \
  --profile unified_candidate_recall_boost_v1 \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/unified_candidate_recall_boost_v1" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_candidate_recall_boost_v1.log"

echo "=== 4. density_rerank_v1 ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_unified_config_4ds.py \
  --profile unified_candidate_recall_boost_density_rerank_v1 \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/unified_candidate_recall_boost_density_rerank_v1" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_density_rerank_v1.log"

echo "=== 5. unified_gl_rcedr_v1 ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_unified_config_4ds.py \
  --profile unified_gl_rcedr_v1 \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/unified_gl_rcedr_v1" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_gl_rcedr_v1.log"

echo "=== 6. ablation: no_dynamic_control ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_unified_config_4ds.py \
  --profile unified_gl_rcedr_no_dynamic_control \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/unified_gl_rcedr_no_dynamic_control" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_gl_rcedr_no_dynamic_control.log"

echo "=== 7. ablation: no_stability ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_unified_config_4ds.py \
  --profile unified_gl_rcedr_no_stability \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/unified_gl_rcedr_no_stability" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_gl_rcedr_no_stability.log"

echo "=== 8. ablation: no_bridge_path ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_unified_config_4ds.py \
  --profile unified_gl_rcedr_no_bridge_path \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/unified_gl_rcedr_no_bridge_path" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_gl_rcedr_no_bridge_path.log"

echo "=== 9. ablation: density_first ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  scripts/run_unified_config_4ds.py \
  --profile unified_gl_rcedr_density_first_ablation \
  --datasets ${DATASETS} \
  --sample-size "${SAMPLE_SIZE}" \
  --retrieval-only \
  --output-root "${OUT_ROOT}/runs/unified_gl_rcedr_density_first_ablation" \
  2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_gl_rcedr_density_first_ablation.log"

echo "=== 10. evidence-flow audit ==="
PYTHONPATH=. "${PYTHON}" scripts/audit_evidence_flow.py \
  --datasets ${DATASETS} \
  --profiles \
    legacy_sota \
    unified_large \
    unified_candidate_recall_boost_v1 \
    unified_candidate_recall_boost_density_rerank_v1 \
    unified_gl_rcedr_v1 \
    unified_gl_rcedr_no_dynamic_control \
    unified_gl_rcedr_no_stability \
    unified_gl_rcedr_no_bridge_path \
    unified_gl_rcedr_density_first_ablation \
  --roots \
    legacy_sota="${OUT_ROOT}/runs/legacy_sota" \
    unified_large="${OUT_ROOT}/runs/unified_large" \
    unified_candidate_recall_boost_v1="${OUT_ROOT}/runs/unified_candidate_recall_boost_v1" \
    unified_candidate_recall_boost_density_rerank_v1="${OUT_ROOT}/runs/unified_candidate_recall_boost_density_rerank_v1" \
    unified_gl_rcedr_v1="${OUT_ROOT}/runs/unified_gl_rcedr_v1" \
    unified_gl_rcedr_no_dynamic_control="${OUT_ROOT}/runs/unified_gl_rcedr_no_dynamic_control" \
    unified_gl_rcedr_no_stability="${OUT_ROOT}/runs/unified_gl_rcedr_no_stability" \
    unified_gl_rcedr_no_bridge_path="${OUT_ROOT}/runs/unified_gl_rcedr_no_bridge_path" \
    unified_gl_rcedr_density_first_ablation="${OUT_ROOT}/runs/unified_gl_rcedr_density_first_ablation" \
  --output-dir "${OUT_ROOT}/audit" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_evidence_flow.log"

SUMMARY_PATH="${OUT_ROOT}/PHASE6H_GL_RCEDR_SUMMARY.md"
if [[ ! -f "${SUMMARY_PATH}" ]]; then
  cat > "${SUMMARY_PATH}" <<'EOF'
# Phase-6H GL-RCEDR Retrieval-Only Results

## 1. Purpose

This experiment evaluates GL-RCEDR, a global-to-local recall-constrained evidence-density retrieval method. It uses diverse global stochastic exploration to avoid early support loss, unified seed set selection to choose robust seed sets, and local recall-constrained evidence-density refinement to preserve support-like and bridge/path evidence before optimizing token density.

## 2. Compared Profiles

| profile | description |
|---|---|
| legacy_sota | heuristic evidence-density teacher/reference |
| unified_large | clean unified baseline |
| unified_candidate_recall_boost_v1 | candidate recall boost baseline |
| unified_candidate_recall_boost_density_rerank_v1 | density-first rerank baseline |
| unified_gl_rcedr_v1 | main GL-RCEDR method |
| unified_gl_rcedr_no_dynamic_control | ablation |
| unified_gl_rcedr_no_stability | ablation |
| unified_gl_rcedr_no_bridge_path | ablation |
| unified_gl_rcedr_density_first_ablation | ablation |

## 3. Evidence Flow Results

| dataset | profile | candidate_sf_R | selected_sf_R | rendered_sf_R | rendered_tokens | rendered_sf_F1_per_1k_tokens |
|---|---|---:|---:|---:|---:|---:|

## 4. Failure Taxonomy

| dataset | profile | retrieval_miss_rate | selector_drop_rate | render_drop_rate | low_density_rate | generation_ready_rate |
|---|---|---:|---:|---:|---:|---:|

## 5. Ablation Analysis

| ablation | expected effect | observed |
|---|---|---|
| no_dynamic_control | lower adaptation to query state | TBD |
| no_stability | more noisy seed selection | TBD |
| no_bridge_path | lower multi-hop preservation | TBD |
| density_first | selected recall collapse risk | TBD |

## 6. Gate Decision

| criterion | pass/fail | note |
|---|---|---|
| candidate recall preserved | TBD | |
| selected_sf_R >= unified_large | TBD | |
| rendered_sf_R >= unified_large | TBD | |
| rendered density improved | TBD | |
| render_drop controlled | TBD | |
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
