#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6v_semantic_sufficiency_abr_n100}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
PROCESS_COUNT="${PROCESS_COUNT:-2}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PROFILE_ROOT="${PROFILE_ROOT:-/home/ojungii/EffiRAG/configs/main_config/copy_span_instruction_unified}"

BASELINE_PROFILE="${BASELINE_PROFILE:-unified_acr_rcedr_v12}"
VARIANT_PROFILE="${VARIANT_PROFILE:-unified_acr_rcedr_v12_semantic_sufficiency}"

if [[ "${PROCESS_COUNT}" != "2" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 2 for this runner, got: ${PROCESS_COUNT}" >&2
  exit 1
fi

if [[ -d "${OUT_ROOT}/qa_runs" ]] && [[ "${ALLOW_REUSE_OUT_ROOT:-0}" != "1" ]]; then
  echo "[ERROR] OUT_ROOT already contains qa_runs: ${OUT_ROOT}" >&2
  echo "Use fresh OUT_ROOT or set ALLOW_REUSE_OUT_ROOT=1 explicitly." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/sampled_query_ids"

echo "=== PHASE6V semantic sufficiency ABR (2 proc) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "PROCESS_COUNT=${PROCESS_COUNT}"
echo "GPU_A=${GPU_A} (group A)"
echo "GPU_B=${GPU_B} (group B)"
echo "BASELINE_PROFILE=${BASELINE_PROFILE}"
echo "VARIANT_PROFILE=${VARIANT_PROFILE}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_ROOT}/logs/nvidia_smi_before.txt" || true
fi

SCHEDULE_JSONL="${OUT_ROOT}/_phase6v_semantic_sufficiency_scheduled_runs.jsonl"
: > "${SCHEDULE_JSONL}"

register_run() {
  local run_name="$1"
  local dataset="$2"
  local profile="$3"
  local group="$4"
  local role="$5"
  local gpu="$6"
  printf '{"phase":"phase6v_semantic_sufficiency_abr","run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"%s","role":"%s","kind":"unified","scheduled":true}\n' \
    "${run_name}" "${dataset}" "${profile}" "${gpu}" "${group}" "${role}" >> "${SCHEDULE_JSONL}"
}

TASKS_A=(
  "${BASELINE_PROFILE}|hotpotqa|v12_hotpotqa|A|baseline"
  "${VARIANT_PROFILE}|hotpotqa|v12_semantic_sufficiency_hotpotqa|A|variant_semantic_sufficiency"
)
TASKS_B=(
  "${BASELINE_PROFILE}|2wikimultihopqa|v12_2wikimultihopqa|B|baseline"
  "${VARIANT_PROFILE}|2wikimultihopqa|v12_semantic_sufficiency_2wikimultihopqa|B|variant_semantic_sufficiency"
)
ALL_TASKS=("${TASKS_A[@]}" "${TASKS_B[@]}")

declare -A RUN_NAME_COUNT
for spec in "${ALL_TASKS[@]}"; do
  IFS='|' read -r _profile _dataset _run_name _group _role <<< "${spec}"
  RUN_NAME_COUNT["${_run_name}"]=$(( ${RUN_NAME_COUNT[${_run_name}]:-0} + 1 ))
done
for run_name in "${!RUN_NAME_COUNT[@]}"; do
  if [[ "${RUN_NAME_COUNT[${run_name}]}" -gt 1 ]]; then
    echo "[ERROR] duplicate run_name detected: ${run_name}" >&2
    exit 1
  fi
done

for spec in "${ALL_TASKS[@]}"; do
  IFS='|' read -r _profile _dataset _run_name _group _role <<< "${spec}"
  if [[ "${_group}" == "A" ]]; then
    _gpu="${GPU_A}"
  else
    _gpu="${GPU_B}"
  fi
  register_run "${_run_name}" "${_dataset}" "${_profile}" "${_group}" "${_role}" "${_gpu}"
done

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
    "phase": "phase6v_semantic_sufficiency_abr",
    "sample_size": int(os.environ.get("SAMPLE_SIZE", "0")),
    "out_root": str(out_root),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": runs,
}
(out_root / "phase6v_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(out_root / "phase6s_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(out_root / "phase6v_run_manifest.json")
print(out_root / "phase6s_run_manifest.json")
PY

run_unified_one_dataset() {
  local profile="$1"
  local dataset="$2"
  local run_name="$3"
  local role="$4"
  local gpu="$5"
  local cfg="${PROFILE_ROOT}/${profile}/${dataset}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "[ERROR] missing config: ${cfg}" >&2
    return 2
  fi

  echo "=== [GPU ${gpu}] profile=${profile} dataset=${dataset} run=${run_name} role=${role} ==="
  PYTHONHASHSEED=0 \
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    -m effirag.run_rag \
    --config "${cfg}" \
    --output-dir "${OUT_ROOT}/qa_runs/${run_name}/${dataset}" \
    --timestamp-output true \
    --limit "${SAMPLE_SIZE}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${run_name}.log"
}

write_sample_ids() {
  local run_name="$1"
  local dataset="$2"
  local out_json="${OUT_ROOT}/sampled_query_ids/${dataset}.json"
  local query_path
  query_path=$(find "${OUT_ROOT}/qa_runs/${run_name}" -type f -name 'rag_query_results.jsonl' | sort | tail -n 1)
  if [[ -z "${query_path}" ]]; then
    return 0
  fi
  PYTHONPATH=. "${PYTHON}" - <<PY
import json
from pathlib import Path
query = Path(${query_path@Q})
out = Path(${out_json@Q})
ids = []
for line in query.read_text(encoding='utf-8').splitlines():
    line=line.strip()
    if not line:
        continue
    try:
        obj=json.loads(line)
    except Exception:
        continue
    sid=str(obj.get('sample_id','')).strip()
    if sid:
        ids.append(sid)
out.write_text(json.dumps(ids, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(out)
PY
}

check_sample_ids_match_baseline() {
  local baseline_run="$1"
  local variant_run="$2"
  local dataset="$3"
  local baseline_ids="${OUT_ROOT}/sampled_query_ids/${dataset}.json"
  local variant_query
  variant_query=$(find "${OUT_ROOT}/qa_runs/${variant_run}" -type f -name 'rag_query_results.jsonl' | sort | tail -n 1)
  if [[ ! -f "${baseline_ids}" ]] || [[ -z "${variant_query}" ]]; then
    return 0
  fi
  PYTHONPATH=. "${PYTHON}" - <<PY
import json
from pathlib import Path
base = json.loads(Path(${baseline_ids@Q}).read_text(encoding='utf-8'))
qpath = Path(${variant_query@Q})
vids = []
for line in qpath.read_text(encoding='utf-8').splitlines():
    line=line.strip()
    if not line:
        continue
    obj=json.loads(line)
    sid=str(obj.get('sample_id','')).strip()
    if sid:
        vids.append(sid)
if base != vids:
    print('MISMATCH')
    print('baseline_len', len(base), 'variant_len', len(vids))
    m = min(len(base), len(vids))
    idx = next((i for i in range(m) if base[i] != vids[i]), None)
    print('first_diff_index', idx)
    raise SystemExit(3)
print('MATCH', len(base))
PY
}

run_task() {
  local spec="$1"
  local group_gpu="$2"
  IFS='|' read -r profile dataset run_name group role <<< "${spec}"
  rm -rf "${OUT_ROOT}/qa_runs/${run_name}"
  mkdir -p "${OUT_ROOT}/qa_runs/${run_name}"

  echo "=== [GROUP ${group} GPU ${group_gpu}] START run=${run_name} profile=${profile} dataset=${dataset} ==="
  if run_unified_one_dataset "${profile}" "${dataset}" "${run_name}" "${role}" "${group_gpu}"; then
    if [[ "${role}" == "baseline" ]]; then
      write_sample_ids "${run_name}" "${dataset}" || true
    else
      local baseline_run="v12_${dataset}"
      check_sample_ids_match_baseline "${baseline_run}" "${run_name}" "${dataset}"
    fi
    echo "=== [GROUP ${group}] DONE run=${run_name} ==="
  else
    local code=$?
    echo "=== [GROUP ${group}] FAIL run=${run_name} exit_code=${code} ===" >&2
    return "${code}"
  fi
}

group_a() { local spec; for spec in "${TASKS_A[@]}"; do run_task "${spec}" "${GPU_A}"; done; }
group_b() { local spec; for spec in "${TASKS_B[@]}"; do run_task "${spec}" "${GPU_B}"; done; }

group_a > "${OUT_ROOT}/logs/group_A_gpu${GPU_A}.log" 2>&1 &
PID_A=$!
group_b > "${OUT_ROOT}/logs/group_B_gpu${GPU_B}.log" 2>&1 &
PID_B=$!

FAIL=0
if ! wait "${PID_A}"; then FAIL=1; fi
if ! wait "${PID_B}"; then FAIL=1; fi
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] one or more workers failed. Check logs/group_*.log" >&2
  exit 1
fi

echo "=== artifact consistency validation ==="
VALIDATE_ARGS=()
if [[ "${STRICT_VALIDATE}" == "1" ]]; then
  VALIDATE_ARGS+=(--strict-clean)
fi
PYTHONPATH=. "${PYTHON}" scripts/validate_phase6s_artifacts.py \
  --out-root "${OUT_ROOT}" \
  "${VALIDATE_ARGS[@]}" \
  --json "${OUT_ROOT}/phase6v_artifact_validation.json" \
  2>&1 | tee "${OUT_ROOT}/logs/validate_phase6v_artifacts.log"

echo "=== summarize PHASE6V semantic sufficiency ABR ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6v_semantic_sufficiency_abr.py \
  --out-root "${OUT_ROOT}" \
  --datasets hotpotqa,2wikimultihopqa \
  --baseline-profile "${BASELINE_PROFILE}" \
  --variant-profile "${VARIANT_PROFILE}" \
  --top-k 20 \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6v_semantic_sufficiency_abr.log"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_ROOT}/logs/nvidia_smi_after.txt" || true
fi

echo "=== done ==="
echo "Summary: ${OUT_ROOT}/PHASE6V_SEMANTIC_SUFFICIENCY_ABR_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6v_semantic_sufficiency_abr_summary.json"
