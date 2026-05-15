#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PYTHONPATH_ENV="${PYTHONPATH_ENV:-.}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6j_champion_bottleneck_audit}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
CUDA_DEVICE_A="${CUDA_DEVICE_A:-0}"
CUDA_DEVICE_B="${CUDA_DEVICE_B:-1}"

mkdir -p "${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}/shards"

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

ROOT_ARGS=(
  legacy_sota="${LEGACY_ROOT}"
  unified_large="${UNIFIED_LARGE_ROOT}"
  unified_candidate_recall_boost_v1="${CANDIDATE_BOOST_ROOT}"
  unified_candidate_recall_boost_density_rerank_v1="${DENSITY_RERANK_ROOT}"
  unified_gl_rcedr_v1="${GL_RCEDR_V1_ROOT}"
  unified_gl_rcedr_v2="${GL_RCEDR_V2_ROOT}"
)

echo "=== Phase-6J Champion Bottleneck Audit (4-way) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "DATASETS=${DATASETS}"

run_worker() {
  local worker_name="$1"
  local worker_out="$2"
  local cuda_device="$3"
  shift 3
  local worker_profiles=("$@")

  mkdir -p "${worker_out}/logs"
  echo "[${worker_name}] cuda=${cuda_device} profiles: ${worker_profiles[*]}"

  CUDA_VISIBLE_DEVICES="${cuda_device}" PYTHONPATH="${PYTHONPATH_ENV}" "${PYTHON}" scripts/audit_champion_bottlenecks.py \
    --datasets ${DATASETS} \
    --profiles "${worker_profiles[@]}" \
    --roots "${ROOT_ARGS[@]}" \
    --qa-jsons hotpotqa="${HOTPOT_QA_JSON}" 2wikimultihopqa="${TWOWIKI_QA_JSON}" \
    --output-dir "${worker_out}" \
    > "${worker_out}/logs/${worker_name}.log" 2>&1
}

W1_OUT="${OUT_ROOT}/shards/w1"
W2_OUT="${OUT_ROOT}/shards/w2"
W3_OUT="${OUT_ROOT}/shards/w3"
W4_OUT="${OUT_ROOT}/shards/w4"

run_worker w1 "${W1_OUT}" "${CUDA_DEVICE_A}" legacy_sota unified_large &
PID1=$!
run_worker w2 "${W2_OUT}" "${CUDA_DEVICE_A}" legacy_sota unified_candidate_recall_boost_v1 unified_candidate_recall_boost_density_rerank_v1 &
PID2=$!
run_worker w3 "${W3_OUT}" "${CUDA_DEVICE_B}" legacy_sota unified_gl_rcedr_v1 &
PID3=$!
run_worker w4 "${W4_OUT}" "${CUDA_DEVICE_B}" legacy_sota unified_gl_rcedr_v2 &
PID4=$!

FAIL=0
for pid in "${PID1}" "${PID2}" "${PID3}" "${PID4}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] One or more shard workers failed." >&2
  exit 1
fi

echo "=== merge shards ==="
PYTHONPATH="${PYTHONPATH_ENV}" "${PYTHON}" scripts/merge_phase6j_champion_bottleneck_shards.py \
  --datasets ${DATASETS} \
  --profiles \
    legacy_sota \
    unified_large \
    unified_candidate_recall_boost_v1 \
    unified_candidate_recall_boost_density_rerank_v1 \
    unified_gl_rcedr_v1 \
    unified_gl_rcedr_v2 \
  --roots "${ROOT_ARGS[@]}" \
  --query-jsonls \
    "${W1_OUT}/query_level_bottlenecks.jsonl" \
    "${W2_OUT}/query_level_bottlenecks.jsonl" \
    "${W3_OUT}/query_level_bottlenecks.jsonl" \
    "${W4_OUT}/query_level_bottlenecks.jsonl" \
  --utility-csvs \
    "${W1_OUT}/utility_component_by_profile.csv" \
    "${W2_OUT}/utility_component_by_profile.csv" \
    "${W3_OUT}/utility_component_by_profile.csv" \
    "${W4_OUT}/utility_component_by_profile.csv" \
  --output-dir "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_champion_bottlenecks_4way_merge.log"

echo "=== done ==="
echo "Report expected at: ${OUT_ROOT}/PHASE6J_CHAMPION_BOTTLENECK_AUDIT.md"
