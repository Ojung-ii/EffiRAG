#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6l_support_span_contract}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
RUN_QA_SMOKE="${RUN_QA_SMOKE:-1}"
QA_GPU="${QA_GPU:-1}"
QA_PROFILE="${QA_PROFILE:-unified_gl_rcedr_v1_support_span_contract}"

# Requested placement: one process on GPU0, remaining three on GPU1.
GPU_W1="${GPU_W1:-0}"
GPU_W2="${GPU_W2:-1}"
GPU_W3="${GPU_W3:-1}"
GPU_W4="${GPU_W4:-1}"

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6L Support Span Contract retrieval-only run (4-way) ==="
echo "DATASETS=${DATASETS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GPU_W1=${GPU_W1} GPU_W2=${GPU_W2} GPU_W3=${GPU_W3} GPU_W4=${GPU_W4}"
echo "RUN_QA_SMOKE=${RUN_QA_SMOKE} QA_GPU=${QA_GPU} QA_PROFILE=${QA_PROFILE}"

run_legacy () {
  local gpu="$1"
  local tag="legacy_sota"
  echo "=== legacy: ${tag} (gpu=${gpu}) ==="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    scripts/run_sota_config_4ds.py \
    --mode locked_precomputed \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --retrieval-only \
    --output-root "${OUT_ROOT}/runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_${tag}.log"
}

run_unified () {
  local gpu="$1"
  local profile="$2"
  local tag="$3"
  echo "=== unified: ${profile} (gpu=${gpu}) ==="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --retrieval-only \
    --output-root "${OUT_ROOT}/runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/retrieval_only_${tag}.log"
}

run_unified_qa () {
  local gpu="$1"
  local profile="$2"
  local tag="$3"
  echo "=== QA smoke: ${profile} (gpu=${gpu}) ==="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets ${DATASETS} \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_smoke/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_smoke_${tag}.log"
}

find_latest_query_json () {
  local root="$1"
  local dataset="$2"
  if [ ! -d "${root}/${dataset}" ]; then
    return 1
  fi
  local latest
  latest="$(find "${root}/${dataset}" -type f -name "rag_query_results.jsonl" | sort | tail -n 1)"
  if [ -z "${latest}" ]; then
    return 1
  fi
  echo "${latest}"
  return 0
}

worker_1 () {
  run_legacy "${GPU_W1}"
  run_unified "${GPU_W1}" unified_large unified_large
  run_unified "${GPU_W1}" unified_gl_rcedr_v1 unified_gl_rcedr_v1
}

worker_2 () {
  run_unified "${GPU_W2}" unified_gl_rcedr_v1_sentence_contract unified_gl_rcedr_v1_sentence_contract
  run_unified "${GPU_W2}" unified_gl_rcedr_v1_sentence_contract_no_item_cap unified_gl_rcedr_v1_sentence_contract_no_item_cap
}

worker_3 () {
  run_unified "${GPU_W3}" unified_gl_rcedr_v1_support_span_contract unified_gl_rcedr_v1_support_span_contract
  run_unified "${GPU_W3}" unified_gl_rcedr_v1_support_span_contract_span40 unified_gl_rcedr_v1_support_span_contract_span40
  run_unified "${GPU_W3}" unified_gl_rcedr_v1_support_span_contract_no_cap unified_gl_rcedr_v1_support_span_contract_no_cap
}

worker_4 () {
  run_unified "${GPU_W4}" unified_gl_rcedr_v1_support_span_contract_no_bridge_signal unified_gl_rcedr_v1_support_span_contract_no_bridge_signal
  run_unified "${GPU_W4}" unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal
}

worker_1 > "${OUT_ROOT}/logs/w1.log" 2>&1 &
PID1=$!
worker_2 > "${OUT_ROOT}/logs/w2.log" 2>&1 &
PID2=$!
worker_3 > "${OUT_ROOT}/logs/w3.log" 2>&1 &
PID3=$!
worker_4 > "${OUT_ROOT}/logs/w4.log" 2>&1 &
PID4=$!

FAIL=0
for pid in "${PID1}" "${PID2}" "${PID3}" "${PID4}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] One or more retrieval workers failed. Check ${OUT_ROOT}/logs/w*.log" >&2
  exit 1
fi

QA_JSON_ARGS=()
if [[ "${RUN_QA_SMOKE}" == "1" || "${RUN_QA_SMOKE}" == "true" || "${RUN_QA_SMOKE}" == "TRUE" ]]; then
  run_unified_qa "${QA_GPU}" "${QA_PROFILE}" "${QA_PROFILE}"
  for dataset in ${DATASETS}; do
    qa_json_path="$(find_latest_query_json "${OUT_ROOT}/qa_smoke/${QA_PROFILE}" "${dataset}" || true)"
    if [ -n "${qa_json_path}" ]; then
      QA_JSON_ARGS+=("${dataset}=${qa_json_path}")
    fi
  done
fi

echo "=== evidence-flow audit ==="
"${PYTHON}" scripts/audit_evidence_flow.py \
  --datasets ${DATASETS} \
  --profiles \
    legacy_sota \
    unified_large \
    unified_gl_rcedr_v1 \
    unified_gl_rcedr_v1_sentence_contract \
    unified_gl_rcedr_v1_sentence_contract_no_item_cap \
    unified_gl_rcedr_v1_support_span_contract \
    unified_gl_rcedr_v1_support_span_contract_span40 \
    unified_gl_rcedr_v1_support_span_contract_no_cap \
    unified_gl_rcedr_v1_support_span_contract_no_bridge_signal \
    unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal \
  --roots \
    legacy_sota="${OUT_ROOT}/runs/legacy_sota" \
    unified_large="${OUT_ROOT}/runs/unified_large" \
    unified_gl_rcedr_v1="${OUT_ROOT}/runs/unified_gl_rcedr_v1" \
    unified_gl_rcedr_v1_sentence_contract="${OUT_ROOT}/runs/unified_gl_rcedr_v1_sentence_contract" \
    unified_gl_rcedr_v1_sentence_contract_no_item_cap="${OUT_ROOT}/runs/unified_gl_rcedr_v1_sentence_contract_no_item_cap" \
    unified_gl_rcedr_v1_support_span_contract="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract" \
    unified_gl_rcedr_v1_support_span_contract_span40="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract_span40" \
    unified_gl_rcedr_v1_support_span_contract_no_cap="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract_no_cap" \
    unified_gl_rcedr_v1_support_span_contract_no_bridge_signal="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract_no_bridge_signal" \
    unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal" \
  --output-dir "${OUT_ROOT}/audit" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_evidence_flow.log"

echo "=== champion bottleneck audit if available ==="
if [ -f scripts/audit_champion_bottlenecks.py ]; then
  "${PYTHON}" scripts/audit_champion_bottlenecks.py \
    --datasets ${DATASETS} \
    --profiles \
      legacy_sota \
      unified_large \
      unified_gl_rcedr_v1 \
      unified_gl_rcedr_v1_sentence_contract \
      unified_gl_rcedr_v1_sentence_contract_no_item_cap \
      unified_gl_rcedr_v1_support_span_contract \
      unified_gl_rcedr_v1_support_span_contract_span40 \
      unified_gl_rcedr_v1_support_span_contract_no_cap \
      unified_gl_rcedr_v1_support_span_contract_no_bridge_signal \
      unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal \
    --qa-jsons "${QA_JSON_ARGS[@]}" \
    --phase-label "Phase-6L" \
    --audit-title "Support-Preserving Span Contract Audit" \
    --purpose-text "Evaluate support-preserving sentence/span contract against Phase-6K and GL-RCEDR v1 baselines, including QA smoke on the main profile." \
    --summary-md-name "PHASE6L_SUPPORT_SPAN_CONTRACT_AUDIT.md" \
    --roots \
      legacy_sota="${OUT_ROOT}/runs/legacy_sota" \
      unified_large="${OUT_ROOT}/runs/unified_large" \
      unified_gl_rcedr_v1="${OUT_ROOT}/runs/unified_gl_rcedr_v1" \
      unified_gl_rcedr_v1_sentence_contract="${OUT_ROOT}/runs/unified_gl_rcedr_v1_sentence_contract" \
      unified_gl_rcedr_v1_sentence_contract_no_item_cap="${OUT_ROOT}/runs/unified_gl_rcedr_v1_sentence_contract_no_item_cap" \
      unified_gl_rcedr_v1_support_span_contract="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract" \
      unified_gl_rcedr_v1_support_span_contract_span40="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract_span40" \
      unified_gl_rcedr_v1_support_span_contract_no_cap="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract_no_cap" \
      unified_gl_rcedr_v1_support_span_contract_no_bridge_signal="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract_no_bridge_signal" \
      unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal="${OUT_ROOT}/runs/unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal" \
    --output-dir "${OUT_ROOT}/champion_audit" \
    2>&1 | tee "${OUT_ROOT}/logs/audit_champion_bottlenecks.log"
fi

echo "=== cleanup ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

echo "=== done ==="
echo "Evidence-flow audit expected at: ${OUT_ROOT}/audit"
echo "Champion audit expected at: ${OUT_ROOT}/champion_audit"
if [[ "${#QA_JSON_ARGS[@]}" -gt 0 ]]; then
  echo "QA smoke merged into champion audit via: ${OUT_ROOT}/qa_smoke/${QA_PROFILE}"
fi
