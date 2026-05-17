#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6r_acr_v11_beam3_4ds_qa_4way}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
RETRIEVAL_AUDIT_ROOT="${RETRIEVAL_AUDIT_ROOT:-}"
BASELINE_TABLE_SOURCE="${BASELINE_TABLE_SOURCE:-native_baseline_table}"
NATIVE_BASELINE_JSON="${NATIVE_BASELINE_JSON:-}"
HIPPORAG2_SUMMARY_ROOT="${HIPPORAG2_SUMMARY_ROOT:-/home/ojungii/HippoRAG2/outputs}"

# Required placement: one process on GPU0, remaining three on GPU1.
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-1}"
GPU_D="${GPU_D:-1}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6R ACR v1.1 beam3 4DS QA smoke (4-way) ==="
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GPU_A=${GPU_A} GPU_B=${GPU_B} GPU_C=${GPU_C} GPU_D=${GPU_D}"
echo "BASELINE_TABLE_SOURCE=${BASELINE_TABLE_SOURCE}"
echo "HIPPORAG2_SUMMARY_ROOT=${HIPPORAG2_SUMMARY_ROOT}"
if [[ -n "${RETRIEVAL_AUDIT_ROOT}" ]]; then
  echo "RETRIEVAL_AUDIT_ROOT=${RETRIEVAL_AUDIT_ROOT}"
fi
if [[ -n "${NATIVE_BASELINE_JSON}" ]]; then
  echo "NATIVE_BASELINE_JSON=${NATIVE_BASELINE_JSON}"
fi

run_unified_qa_one_dataset () {
  local gpu="$1"
  local profile="$2"
  local dataset="$3"
  local tag="$4"

  echo "=== [GPU ${gpu}] QA smoke profile=${profile} dataset=${dataset} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets "${dataset}" \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${tag}.log"
}

run_legacy_qa_one_dataset () {
  local gpu="$1"
  local dataset="$2"
  local tag="$3"

  echo "=== [GPU ${gpu}] legacy_sota dataset=${dataset} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_sota_config_4ds.py \
    --mode locked_precomputed \
    --datasets "${dataset}" \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${tag}.log" || {
      echo "WARNING: legacy_sota failed or unsupported for dataset=${dataset}; continuing." | tee -a "${OUT_ROOT}/logs/qa_${tag}.log"
    }
}

# Process A / GPU0 (heavy + medium)
group_a () {
  run_legacy_qa_one_dataset "${GPU_A}" hotpotqa legacy_sota_hotpotqa
  run_legacy_qa_one_dataset "${GPU_A}" 2wikimultihopqa legacy_sota_2wikimultihopqa
  run_unified_qa_one_dataset "${GPU_A}" unified_acr_v11_beam3 musique acr_v11_beam3_musique
}

# Process B / GPU1 (reference + heavy)
group_b () {
  run_unified_qa_one_dataset "${GPU_B}" unified_large hotpotqa unified_large_hotpotqa
  run_unified_qa_one_dataset "${GPU_B}" unified_large 2wikimultihopqa unified_large_2wikimultihopqa
  run_unified_qa_one_dataset "${GPU_B}" unified_large musique unified_large_musique
  run_unified_qa_one_dataset "${GPU_B}" unified_large popqa unified_large_popqa
  run_unified_qa_one_dataset "${GPU_B}" unified_acr_v11_beam3 hotpotqa acr_v11_beam3_hotpotqa
}

# Process C / GPU1 (ACR v1 sweep + v1.1 partial)
group_c () {
  run_unified_qa_one_dataset "${GPU_C}" unified_acr_v1 hotpotqa acr_v1_hotpotqa
  run_unified_qa_one_dataset "${GPU_C}" unified_acr_v1 2wikimultihopqa acr_v1_2wikimultihopqa
  run_unified_qa_one_dataset "${GPU_C}" unified_acr_v1 musique acr_v1_musique
  run_unified_qa_one_dataset "${GPU_C}" unified_acr_v1 popqa acr_v1_popqa
  run_unified_qa_one_dataset "${GPU_C}" unified_acr_v11 2wikimultihopqa acr_v11_2wikimultihopqa
  run_unified_qa_one_dataset "${GPU_C}" unified_acr_v11 popqa acr_v11_popqa
}

# Process D / GPU1 (adaptive reference + beam3 partial)
group_d () {
  run_unified_qa_one_dataset "${GPU_D}" unified_gl_rcedr_v1_adaptive_support_span_no_bridge hotpotqa adaptive_no_bridge_hotpotqa
  run_unified_qa_one_dataset "${GPU_D}" unified_gl_rcedr_v1_adaptive_support_span_no_bridge 2wikimultihopqa adaptive_no_bridge_2wikimultihopqa
  run_unified_qa_one_dataset "${GPU_D}" unified_gl_rcedr_v1_adaptive_support_span_no_bridge musique adaptive_no_bridge_musique
  run_unified_qa_one_dataset "${GPU_D}" unified_gl_rcedr_v1_adaptive_support_span_no_bridge popqa adaptive_no_bridge_popqa
  run_unified_qa_one_dataset "${GPU_D}" unified_acr_v11_beam3 2wikimultihopqa acr_v11_beam3_2wikimultihopqa
  run_unified_qa_one_dataset "${GPU_D}" unified_acr_v11_beam3 popqa acr_v11_beam3_popqa
}

group_a > "${OUT_ROOT}/logs/group_A_gpu0.log" 2>&1 &
PID_A=$!
group_b > "${OUT_ROOT}/logs/group_B_gpu1.log" 2>&1 &
PID_B=$!
group_c > "${OUT_ROOT}/logs/group_C_gpu1.log" 2>&1 &
PID_C=$!
group_d > "${OUT_ROOT}/logs/group_D_gpu1.log" 2>&1 &
PID_D=$!

FAIL=0
for pid in "${PID_A}" "${PID_B}" "${PID_C}" "${PID_D}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] One or more QA workers failed. Check ${OUT_ROOT}/logs/group_*.log" >&2
  exit 1
fi

echo "=== Create integrated summary ==="
if [ -f scripts/summarize_phase6r_acr_4ds_results.py ]; then
  SUMMARY_CMD=(
    "${PYTHON}" scripts/summarize_phase6r_acr_4ds_results.py
    --input-root "${OUT_ROOT}/qa_runs"
    --baseline-table-source "${BASELINE_TABLE_SOURCE}"
    --hipporag2-summary-root "${HIPPORAG2_SUMMARY_ROOT}"
    --output "${OUT_ROOT}/PHASE6R_ACR_V11_BEAM3_4DS_SUMMARY.md"
  )
  if [[ -n "${RETRIEVAL_AUDIT_ROOT}" ]]; then
    SUMMARY_CMD+=(--retrieval-audit-root "${RETRIEVAL_AUDIT_ROOT}")
  fi
  if [[ -n "${NATIVE_BASELINE_JSON}" ]]; then
    SUMMARY_CMD+=(--native-baseline-json "${NATIVE_BASELINE_JSON}")
  fi
  PYTHONPATH=. "${SUMMARY_CMD[@]}" 2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6r.log"
elif [ -f scripts/compare_qa_pareto_profiles.py ]; then
  PYTHONPATH=. "${PYTHON}" scripts/compare_qa_pareto_profiles.py \
    --datasets hotpotqa 2wikimultihopqa musique popqa \
    --profiles \
      legacy_sota \
      unified_large \
      unified_gl_rcedr_v1_adaptive_support_span_no_bridge \
      unified_acr_v1 \
      unified_acr_v11 \
      unified_acr_v11_beam3 \
    --input-root "${OUT_ROOT}/qa_runs" \
    --retrieval-audit-root "${RETRIEVAL_AUDIT_ROOT}" \
    --output-dir "${OUT_ROOT}/pareto_report" \
    2>&1 | tee "${OUT_ROOT}/logs/compare_qa_pareto.log"
else
  echo "WARNING: no summary script found."
fi

echo "=== cleanup ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

echo "=== done ==="
echo "QA runs: ${OUT_ROOT}/qa_runs"
echo "Summary: ${OUT_ROOT}/PHASE6R_ACR_V11_BEAM3_4DS_SUMMARY.md"
