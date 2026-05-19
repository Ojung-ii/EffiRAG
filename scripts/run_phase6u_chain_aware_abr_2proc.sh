#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6u_chain_aware_abr_n1000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
PROCESS_COUNT="${PROCESS_COUNT:-2}"
GPU="${GPU:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PROFILE="${PROFILE:-unified_acr_rcedr_v12_chain_aware}"
BASELINE_METRICS_PATH="${BASELINE_METRICS_PATH:-configs/SOTA_config/phase6s_v12_n1000_baseline_metrics.json}"

RUN_A_NAME="${RUN_A_NAME:-v12_chain_aware_hotpotqa}"
RUN_B_NAME="${RUN_B_NAME:-v12_chain_aware_2wikimultihopqa}"

if [[ "${PROCESS_COUNT}" != "2" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 2 for this runner, got: ${PROCESS_COUNT}" >&2
  exit 1
fi

if [[ -d "${OUT_ROOT}/qa_runs" ]] && [[ "${ALLOW_REUSE_OUT_ROOT:-0}" != "1" ]]; then
  echo "[ERROR] OUT_ROOT already contains qa_runs: ${OUT_ROOT}" >&2
  echo "Use a fresh OUT_ROOT or set ALLOW_REUSE_OUT_ROOT=1 explicitly." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6U chain-aware ABR run (new variant only) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "PROCESS_COUNT=${PROCESS_COUNT}"
echo "GPU=${GPU}"
echo "PROFILE=${PROFILE}"
echo "BASELINE_METRICS_PATH=${BASELINE_METRICS_PATH}"

audit_one() {
  local dataset="$1"
  PYTHONPATH=. "${PYTHON}" -m effirag.config_audit "configs/main_config/copy_span_instruction_unified/${PROFILE}/${dataset}.yaml" >/dev/null
}

echo "=== config audit gate ==="
audit_one hotpotqa
audit_one 2wikimultihopqa

SCHEDULE_JSONL="${OUT_ROOT}/_phase6u_chain_aware_scheduled_runs.jsonl"
: > "${SCHEDULE_JSONL}"

register_run() {
  local dataset="$1"
  local run_name="$2"
  local group="$3"
  printf '{"phase":"phase6u_chain_aware_abr_n1000","run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"%s","role":"candidate","kind":"unified","scheduled":true}\n' \
    "${run_name}" "${dataset}" "${PROFILE}" "${GPU}" "${group}" >> "${SCHEDULE_JSONL}"
}

TASKS_A=("${PROFILE}|hotpotqa|${RUN_A_NAME}|${GPU}|A")
TASKS_B=("${PROFILE}|2wikimultihopqa|${RUN_B_NAME}|${GPU}|B")
ALL_TASKS=("${TASKS_A[@]}" "${TASKS_B[@]}")

declare -A RUN_NAME_COUNT
for spec in "${ALL_TASKS[@]}"; do
  IFS='|' read -r _profile _dataset _run_name _gpu _group <<< "${spec}"
  RUN_NAME_COUNT["${_run_name}"]=$(( ${RUN_NAME_COUNT["${_run_name}"]:-0} + 1 ))
done
for run_name in "${!RUN_NAME_COUNT[@]}"; do
  if [[ "${RUN_NAME_COUNT[${run_name}]}" -gt 1 ]]; then
    echo "[ERROR] duplicate run_name detected: ${run_name}" >&2
    exit 1
  fi
done

register_run "hotpotqa" "${RUN_A_NAME}" "A"
register_run "2wikimultihopqa" "${RUN_B_NAME}" "B"

export OUT_ROOT SAMPLE_SIZE SCHEDULE_JSONL
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"]).resolve()
jsonl = Path(os.environ["SCHEDULE_JSONL"]).resolve()
runs = []
if jsonl.exists():
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            runs.append(row)

manifest = {
    "phase": "phase6u_chain_aware_abr_n1000",
    "sample_size": int(os.environ.get("SAMPLE_SIZE", "0")),
    "out_root": str(out_root),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": runs,
}

phase6u = out_root / "phase6u_run_manifest.json"
phase6u.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
phase6s = out_root / "phase6s_run_manifest.json"
phase6s.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(str(phase6u))
print(str(phase6s))
PY

run_unified_qa_one_dataset() {
  local profile="$1"
  local dataset="$2"
  local run_name="$3"
  local gpu="$4"
  echo "=== [GPU ${gpu}] profile=${profile} dataset=${dataset} run=${run_name} ==="
  PHASE6T_SCORE_ATTACHMENT_OPT="${PHASE6T_SCORE_ATTACHMENT_OPT:-1}" \
  PHASE6T_OPTIMIZED="${PHASE6T_OPTIMIZED:-1}" \
  PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER="${PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER:-0}" \
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets "${dataset}" \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${run_name}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${run_name}.log"
}

run_task() {
  local spec="$1"
  IFS='|' read -r profile dataset run_name gpu group <<< "${spec}"
  rm -rf "${OUT_ROOT}/qa_runs/${run_name}"
  mkdir -p "${OUT_ROOT}/qa_runs/${run_name}"
  echo "=== [GROUP ${group}] START run=${run_name} profile=${profile} dataset=${dataset} gpu=${gpu} ==="
  if run_unified_qa_one_dataset "${profile}" "${dataset}" "${run_name}" "${gpu}"; then
    echo "=== [GROUP ${group}] DONE run=${run_name} ==="
  else
    local code=$?
    echo "=== [GROUP ${group}] FAIL run=${run_name} exit_code=${code} ===" >&2
    return "${code}"
  fi
}

group_a() { local spec; for spec in "${TASKS_A[@]}"; do run_task "${spec}"; done; }
group_b() { local spec; for spec in "${TASKS_B[@]}"; do run_task "${spec}"; done; }

group_a > "${OUT_ROOT}/logs/group_A_gpu${GPU}.log" 2>&1 &
PID_A=$!
group_b > "${OUT_ROOT}/logs/group_B_gpu${GPU}.log" 2>&1 &
PID_B=$!

FAIL=0
if ! wait "${PID_A}"; then FAIL=1; fi
if ! wait "${PID_B}"; then FAIL=1; fi
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] One or more workers failed. Check ${OUT_ROOT}/logs/group_*.log" >&2
  exit 1
fi

echo "=== summarize Phase-6U chain-aware ABR ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6u_chain_aware_abr_n1000.py \
  --out-root "${OUT_ROOT}" \
  --baseline-metrics "${BASELINE_METRICS_PATH}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6u_chain_aware_abr_n1000.log"

echo "=== artifact consistency validation ==="
VALIDATE_ARGS=()
if [[ "${STRICT_VALIDATE}" == "1" ]]; then
  VALIDATE_ARGS+=(--strict-clean)
fi
PYTHONPATH=. "${PYTHON}" scripts/validate_phase6s_artifacts.py \
  --out-root "${OUT_ROOT}" \
  "${VALIDATE_ARGS[@]}" \
  --json "${OUT_ROOT}/phase6u_artifact_validation.json" \
  2>&1 | tee "${OUT_ROOT}/logs/validate_phase6s_artifacts.log"

echo "=== done ==="
echo "Summary: ${OUT_ROOT}/PHASE6U_CHAIN_AWARE_ABR_N1000_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6u_chain_aware_abr_n1000_summary.json"
