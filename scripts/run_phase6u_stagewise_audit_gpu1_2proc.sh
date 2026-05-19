#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6u_stagewise_audit_n200}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
PROCESS_COUNT="${PROCESS_COUNT:-2}"
GPU="${GPU:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PROFILE="${PROFILE:-unified_acr_rcedr_v12}"

RUN_A_NAME="${RUN_A_NAME:-phase6u_hotpotqa_v12}"
RUN_B_NAME="${RUN_B_NAME:-phase6u_2wikimultihopqa_v12}"

if [[ "${PROCESS_COUNT}" != "2" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 2 for this runner, got: ${PROCESS_COUNT}" >&2
  exit 1
fi
if [[ "${GPU}" != "1" ]]; then
  echo "[ERROR] This runner is GPU1-only. Set GPU=1." >&2
  exit 1
fi

if [[ -d "${OUT_ROOT}/qa_runs" ]] && [[ "${ALLOW_REUSE_OUT_ROOT:-0}" != "1" ]]; then
  echo "[ERROR] OUT_ROOT already contains qa_runs: ${OUT_ROOT}" >&2
  echo "Use a fresh OUT_ROOT or set ALLOW_REUSE_OUT_ROOT=1 explicitly." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs"

echo "=== PHASE6U stagewise evidence audit (GPU1 / 2 proc) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "PROCESS_COUNT=${PROCESS_COUNT}"
echo "GPU=${GPU}"
echo "PROFILE=${PROFILE}"
echo "PHASE6T_SCORE_ATTACHMENT_OPT=${PHASE6T_SCORE_ATTACHMENT_OPT:-1}"
echo "PHASE6T_OPTIMIZED=${PHASE6T_OPTIMIZED:-1}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_ROOT}/logs/nvidia_smi_before.txt" || true
fi

SCHEDULE_JSONL="${OUT_ROOT}/_phase6u_stagewise_audit_scheduled_runs.jsonl"
: > "${SCHEDULE_JSONL}"

register_run() {
  local dataset="$1"
  local run_name="$2"
  local group="$3"
  printf '{"run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"%s","role":"stagewise_audit","kind":"unified","scheduled":true}\n' \
    "${run_name}" "${dataset}" "${PROFILE}" "${GPU}" "${group}" >> "${SCHEDULE_JSONL}"
}

TASKS_A=("${PROFILE}|hotpotqa|${RUN_A_NAME}|A")
TASKS_B=("${PROFILE}|2wikimultihopqa|${RUN_B_NAME}|B")
ALL_TASKS=("${TASKS_A[@]}" "${TASKS_B[@]}")

declare -A RUN_NAME_COUNT
for spec in "${ALL_TASKS[@]}"; do
  IFS='|' read -r _profile _dataset _run_name _group <<< "${spec}"
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
    "phase": "phase6u_stagewise_evidence_audit",
    "sample_size": int(os.environ.get("SAMPLE_SIZE", "0")),
    "out_root": str(out_root),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": runs,
}

phase6u = out_root / "phase6u_run_manifest.json"
phase6s = out_root / "phase6s_run_manifest.json"
phase6u.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
phase6s.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(str(phase6u))
print(str(phase6s))
PY

run_unified_qa_one_dataset() {
  local profile="$1"
  local dataset="$2"
  local run_name="$3"
  echo "=== [GPU ${GPU}] profile=${profile} dataset=${dataset} run=${run_name} ==="
  PHASE6T_SCORE_ATTACHMENT_OPT="${PHASE6T_SCORE_ATTACHMENT_OPT:-1}" \
  PHASE6T_OPTIMIZED="${PHASE6T_OPTIMIZED:-1}" \
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets "${dataset}" \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${run_name}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${run_name}.log"
}

run_task() {
  local spec="$1"
  IFS='|' read -r profile dataset run_name group <<< "${spec}"
  rm -rf "${OUT_ROOT}/qa_runs/${run_name}"
  mkdir -p "${OUT_ROOT}/qa_runs/${run_name}"
  echo "=== [GROUP ${group}] START run=${run_name} profile=${profile} dataset=${dataset} ==="
  if run_unified_qa_one_dataset "${profile}" "${dataset}" "${run_name}"; then
    echo "=== [GROUP ${group}] DONE run=${run_name} ==="
  else
    local code=$?
    echo "=== [GROUP ${group}] FAIL run=${run_name} exit_code=${code} ===" >&2
    return "${code}"
  fi
}

group_a() { local spec; for spec in "${TASKS_A[@]}"; do run_task "${spec}"; done; }
group_b() { local spec; for spec in "${TASKS_B[@]}"; do run_task "${spec}"; done; }

group_a > "${OUT_ROOT}/logs/group_A_gpu1.log" 2>&1 &
PID_A=$!
group_b > "${OUT_ROOT}/logs/group_B_gpu1.log" 2>&1 &
PID_B=$!

FAIL=0
if ! wait "${PID_A}"; then FAIL=1; fi
if ! wait "${PID_B}"; then FAIL=1; fi
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] One or more workers failed. Check ${OUT_ROOT}/logs/group_*.log" >&2
  exit 1
fi

echo "=== summarize PHASE6U stagewise evidence audit ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6u_stagewise_evidence_audit.py \
  --out-root "${OUT_ROOT}" \
  --datasets hotpotqa,2wikimultihopqa \
  --profile "${PROFILE}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6u_stagewise_evidence_audit.log"

echo "=== artifact consistency validation ==="
VALIDATE_ARGS=()
if [[ "${STRICT_VALIDATE}" == "1" ]]; then
  VALIDATE_ARGS+=(--strict-clean)
fi
PYTHONPATH=. "${PYTHON}" scripts/validate_phase6s_artifacts.py \
  --out-root "${OUT_ROOT}" \
  "${VALIDATE_ARGS[@]}" \
  --json "${OUT_ROOT}/phase6u_artifact_validation.json" \
  2>&1 | tee "${OUT_ROOT}/logs/validate_phase6u_artifacts.log"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_ROOT}/logs/nvidia_smi_after.txt" || true
fi

echo "[PHASE6U] scheduled runs:"
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
path = os.path.join(os.environ["OUT_ROOT"], "phase6u_run_manifest.json")
if os.path.exists(path):
    print(len(json.load(open(path)).get("runs", [])))
else:
    print("missing phase6u_run_manifest.json")
PY

echo "[PHASE6U] rag_summary count:"
find "${OUT_ROOT}" -name rag_summary.json | wc -l
echo "[PHASE6U] rag_query_results count:"
find "${OUT_ROOT}" -name rag_query_results.jsonl | wc -l

echo "=== done ==="
echo "Summary: ${OUT_ROOT}/PHASE6U_STAGEWISE_EVIDENCE_AUDIT_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6u_stagewise_evidence_audit_summary.json"
echo "Query-stage JSONL: ${OUT_ROOT}/stagewise_recall_by_query.jsonl"
echo "Dataset-stage CSV: ${OUT_ROOT}/stagewise_recall_by_dataset.csv"
echo "Drop examples: ${OUT_ROOT}/drop_examples_top20.md"

