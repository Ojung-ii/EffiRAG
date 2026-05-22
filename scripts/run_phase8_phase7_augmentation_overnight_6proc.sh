#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ojungii/EffiRAG}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/phase8_phase7_augmentation}"
LIMIT="${LIMIT:-100}"
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-false}"
RUN_BATCH_ID="${RUN_BATCH_ID:-phase8_aug_overnight_$(date -u +%Y%m%dT%H%M%S)}"

# Four-dataset overnight sweep. Override with comma-separated values if needed.
DATASETS_CSV="${DATASETS_CSV:-hotpotqa,2wikimultihopqa,musique,popqa}"
PROFILES_CSV="${PROFILES_CSV:-legacy_512_10,balanced_384_8}"

# Primary augmentation sweep. Chunk augmentation remains opt-in because current
# diagnostics show chunk-universe miss and high latency.
PHASE8_AUG_VARIANTS="${PHASE8_AUG_VARIANTS:-source_balanced_128,sb_plus_entity_refine_k5}"
PHASE8_AUG_EXTRA_CAP="${PHASE8_AUG_EXTRA_CAP:-32}"
PHASE8_AUG_BASE_TOP_M="${PHASE8_AUG_BASE_TOP_M:-128}"

# Requested process layout: GPU 0 has 2 workers; GPU 1 has 4 workers.
GPU_SLOTS_CSV="${GPU_SLOTS_CSV:-0,0,1,1,1,1}"
DRY_RUN="${DRY_RUN:-false}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  out_ref=()
  local old_ifs="${IFS}"
  IFS=","
  read -r -a out_ref <<< "${csv}"
  IFS="${old_ifs}"
}

DATASETS=()
PROFILES=()
GPU_SLOTS=()
split_csv "${DATASETS_CSV}" DATASETS
split_csv "${PROFILES_CSV}" PROFILES
split_csv "${GPU_SLOTS_CSV}" GPU_SLOTS

if [[ "${#GPU_SLOTS[@]}" -eq 0 ]]; then
  echo "[phase8-aug-overnight] no GPU slots configured" >&2
  exit 1
fi

SPECS=()
for dataset in "${DATASETS[@]}"; do
  for profile in "${PROFILES[@]}"; do
    SPECS+=("${dataset}|${profile}")
  done
done

if [[ "${#SPECS[@]}" -eq 0 ]]; then
  echo "[phase8-aug-overnight] no dataset/profile specs configured" >&2
  exit 1
fi

SCHEDULE_JSONL="${OUT_ROOT}/phase8_phase7_augmentation_overnight_schedule_${RUN_BATCH_ID}.jsonl"
: > "${SCHEDULE_JSONL}"

echo "[phase8-aug-overnight] batch=${RUN_BATCH_ID}"
echo "[phase8-aug-overnight] datasets=${DATASETS_CSV}"
echo "[phase8-aug-overnight] profiles=${PROFILES_CSV}"
echo "[phase8-aug-overnight] variants=${PHASE8_AUG_VARIANTS}"
echo "[phase8-aug-overnight] limit=${LIMIT}"
echo "[phase8-aug-overnight] gpu_slots=${GPU_SLOTS_CSV}"
echo "[phase8-aug-overnight] specs=${#SPECS[@]} workers=${#GPU_SLOTS[@]}"

run_one_spec() {
  local worker_idx="$1"
  local gpu="$2"
  local dataset="$3"
  local profile="$4"
  local run_id="${RUN_BATCH_ID}_w${worker_idx}_${dataset}_${profile}"
  local log_file="${OUT_ROOT}/logs/overnight__w${worker_idx}__gpu${gpu}__${dataset}__${profile}__${RUN_BATCH_ID}.log"

  printf '{"batch_id":"%s","worker":"%s","gpu":"%s","dataset":"%s","profile":"%s","variants":"%s","limit":%s,"run_id":"%s","scheduled":true}\n' \
    "${RUN_BATCH_ID}" "${worker_idx}" "${gpu}" "${dataset}" "${profile}" "${PHASE8_AUG_VARIANTS}" "${LIMIT}" "${run_id}" >> "${SCHEDULE_JSONL}"

  echo "[phase8-aug-overnight] START worker=${worker_idx} gpu=${gpu} dataset=${dataset} profile=${profile} run_id=${run_id}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${gpu} DATASET=${dataset} PROFILE=${profile} LIMIT=${LIMIT} RUN_ID=${run_id} PHASE8_AUG_VARIANTS=${PHASE8_AUG_VARIANTS} bash scripts/run_phase8_phase7_augmentation_experiment.sh"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHON="${PYTHON}" \
  DATASET="${dataset}" \
  PROFILE="${profile}" \
  LIMIT="${LIMIT}" \
  OUT_ROOT="${OUT_ROOT}" \
  RETRIEVAL_ONLY="${RETRIEVAL_ONLY}" \
  RUN_ID="${run_id}" \
  PHASE8_AUG_VARIANTS="${PHASE8_AUG_VARIANTS}" \
  PHASE8_AUG_EXTRA_CAP="${PHASE8_AUG_EXTRA_CAP}" \
  PHASE8_AUG_BASE_TOP_M="${PHASE8_AUG_BASE_TOP_M}" \
  bash scripts/run_phase8_phase7_augmentation_experiment.sh > "${log_file}" 2>&1

  echo "[phase8-aug-overnight] DONE worker=${worker_idx} gpu=${gpu} dataset=${dataset} profile=${profile}"
}

run_worker() {
  local worker_idx="$1"
  local gpu="${GPU_SLOTS[$worker_idx]}"
  local n_workers="${#GPU_SLOTS[@]}"
  local i
  for ((i = worker_idx; i < ${#SPECS[@]}; i += n_workers)); do
    IFS="|" read -r dataset profile <<< "${SPECS[$i]}"
    run_one_spec "${worker_idx}" "${gpu}" "${dataset}" "${profile}"
  done
}

PIDS=()
for ((worker_idx = 0; worker_idx < ${#GPU_SLOTS[@]}; worker_idx++)); do
  worker_log="${OUT_ROOT}/logs/overnight_worker_${worker_idx}_gpu${GPU_SLOTS[$worker_idx]}_${RUN_BATCH_ID}.log"
  run_worker "${worker_idx}" > "${worker_log}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${DRY_RUN}" != "true" ]]; then
  "${PYTHON}" scripts/summarize_phase8_phase7_augmentation.py \
    --root "${OUT_ROOT}" \
    --latest-only \
    --expected-limit "${LIMIT}" || status=1
fi

if [[ "${status}" -eq 0 ]]; then
  echo "[phase8-aug-overnight] all workers completed batch=${RUN_BATCH_ID}"
else
  echo "[phase8-aug-overnight] one or more workers failed batch=${RUN_BATCH_ID}" >&2
fi
exit "${status}"
