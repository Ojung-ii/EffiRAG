#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6w_unit_transition_atomization_n100}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
PROCESS_COUNT="${PROCESS_COUNT:-4}"
GPU_A="${GPU_A:-}"   # process A
GPU_B="${GPU_B:-}"   # process B
GPU_C="${GPU_C:-}"   # process C
GPU_D="${GPU_D:-}"   # process D
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PROFILE_ROOT="${PROFILE_ROOT:-/home/ojungii/EffiRAG/configs/main_config/copy_span_instruction_unified}"

BASELINE_PROFILE="${BASELINE_PROFILE:-unified_acr_rcedr_v12}"
NO_ATOM_PROFILE="${NO_ATOM_PROFILE:-unified_acr_rcedr_v12_no_atomization}"
ATOM3_PROFILE="${ATOM3_PROFILE:-unified_acr_rcedr_v12_atom_span3}"
RENDER_PACKAGE_PROFILE="${RENDER_PACKAGE_PROFILE:-unified_acr_rcedr_v12_render_package}"

if [[ "${PROCESS_COUNT}" != "4" ]] && [[ "${PROCESS_COUNT}" != "3" ]] && [[ "${PROCESS_COUNT}" != "2" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 2, 3, or 4 for this runner, got: ${PROCESS_COUNT}" >&2
  exit 1
fi

if [[ "${PROCESS_COUNT}" == "2" ]]; then
  GPU_A="${GPU_A:-1}"
  GPU_B="${GPU_B:-1}"
  GPU_C="${GPU_C:-1}"
  GPU_D="${GPU_D:-1}"
else
  GPU_A="${GPU_A:-0}"
  GPU_B="${GPU_B:-1}"
  GPU_C="${GPU_C:-1}"
  GPU_D="${GPU_D:-1}"
fi

if [[ -d "${OUT_ROOT}/qa_runs" ]] && [[ "${ALLOW_REUSE_OUT_ROOT:-0}" != "1" ]]; then
  echo "[ERROR] OUT_ROOT already contains qa_runs: ${OUT_ROOT}" >&2
  echo "Use fresh OUT_ROOT or set ALLOW_REUSE_OUT_ROOT=1 explicitly." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/sampled_query_ids"

echo "=== PHASE6W unit transition & atomization audit (${PROCESS_COUNT} proc) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "GPU_A=${GPU_A} (A), GPU_B=${GPU_B} (B), GPU_C=${GPU_C} (C, used when PROCESS_COUNT>=3), GPU_D=${GPU_D} (D, only when PROCESS_COUNT=4)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_ROOT}/logs/nvidia_smi_before.txt" || true
fi

SCHEDULE_JSONL="${OUT_ROOT}/_phase6w_unit_transition_scheduled_runs.jsonl"
: > "${SCHEDULE_JSONL}"

register_run() {
  local run_name="$1"
  local dataset="$2"
  local profile="$3"
  local group="$4"
  local role="$5"
  local gpu="$6"
  printf '{"phase":"phase6w_unit_transition_atomization","run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"%s","role":"%s","kind":"unified","scheduled":true}\n' \
    "${run_name}" "${dataset}" "${profile}" "${gpu}" "${group}" "${role}" >> "${SCHEDULE_JSONL}"
}

TASKS_A=()
TASKS_B=()
TASKS_C=()
TASKS_D=()

if [[ "${PROCESS_COUNT}" == "2" ]]; then
  # GPU1-only two-process mode:
  # A: hotpotqa (all 4 profiles), B: 2wikimultihopqa (all 4 profiles)
  TASKS_A=(
    "${BASELINE_PROFILE}|hotpotqa|v12_hotpotqa|A|baseline"
    "${NO_ATOM_PROFILE}|hotpotqa|v12_no_atomization_hotpotqa|A|no_atomization"
    "${ATOM3_PROFILE}|hotpotqa|v12_atom_span3_hotpotqa|A|atom_span3"
    "${RENDER_PACKAGE_PROFILE}|hotpotqa|v12_render_package_hotpotqa|A|render_package"
  )
  TASKS_B=(
    "${BASELINE_PROFILE}|2wikimultihopqa|v12_2wikimultihopqa|B|baseline"
    "${NO_ATOM_PROFILE}|2wikimultihopqa|v12_no_atomization_2wikimultihopqa|B|no_atomization"
    "${ATOM3_PROFILE}|2wikimultihopqa|v12_atom_span3_2wikimultihopqa|B|atom_span3"
    "${RENDER_PACKAGE_PROFILE}|2wikimultihopqa|v12_render_package_2wikimultihopqa|B|render_package"
  )
elif [[ "${PROCESS_COUNT}" == "3" ]]; then
  TASKS_A=(
    "${BASELINE_PROFILE}|hotpotqa|v12_hotpotqa|A|baseline"
    "${NO_ATOM_PROFILE}|hotpotqa|v12_no_atomization_hotpotqa|A|no_atomization"
  )
  TASKS_B=(
    "${ATOM3_PROFILE}|hotpotqa|v12_atom_span3_hotpotqa|B|atom_span3"
    "${RENDER_PACKAGE_PROFILE}|hotpotqa|v12_render_package_hotpotqa|B|render_package"
  )
  TASKS_C=(
    "${BASELINE_PROFILE}|2wikimultihopqa|v12_2wikimultihopqa|C|baseline"
    "${NO_ATOM_PROFILE}|2wikimultihopqa|v12_no_atomization_2wikimultihopqa|C|no_atomization"
    "${ATOM3_PROFILE}|2wikimultihopqa|v12_atom_span3_2wikimultihopqa|C|atom_span3"
    "${RENDER_PACKAGE_PROFILE}|2wikimultihopqa|v12_render_package_2wikimultihopqa|C|render_package"
  )
else
  TASKS_A=(
    "${BASELINE_PROFILE}|hotpotqa|v12_hotpotqa|A|baseline"
    "${NO_ATOM_PROFILE}|hotpotqa|v12_no_atomization_hotpotqa|A|no_atomization"
  )
  TASKS_B=(
    "${ATOM3_PROFILE}|hotpotqa|v12_atom_span3_hotpotqa|B|atom_span3"
    "${RENDER_PACKAGE_PROFILE}|hotpotqa|v12_render_package_hotpotqa|B|render_package"
  )
  TASKS_C=(
    "${BASELINE_PROFILE}|2wikimultihopqa|v12_2wikimultihopqa|C|baseline"
    "${NO_ATOM_PROFILE}|2wikimultihopqa|v12_no_atomization_2wikimultihopqa|C|no_atomization"
  )
  TASKS_D=(
    "${ATOM3_PROFILE}|2wikimultihopqa|v12_atom_span3_2wikimultihopqa|D|atom_span3"
    "${RENDER_PACKAGE_PROFILE}|2wikimultihopqa|v12_render_package_2wikimultihopqa|D|render_package"
  )
fi

ALL_TASKS=("${TASKS_A[@]}" "${TASKS_B[@]}" "${TASKS_C[@]}")
if [[ "${PROCESS_COUNT}" == "4" ]]; then
  ALL_TASKS+=("${TASKS_D[@]}")
fi

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

for spec in "${TASKS_A[@]}"; do
  IFS='|' read -r _profile _dataset _run_name _group _role <<< "${spec}"
  register_run "${_run_name}" "${_dataset}" "${_profile}" "${_group}" "${_role}" "${GPU_A}"
done
for spec in "${TASKS_B[@]}"; do
  IFS='|' read -r _profile _dataset _run_name _group _role <<< "${spec}"
  register_run "${_run_name}" "${_dataset}" "${_profile}" "${_group}" "${_role}" "${GPU_B}"
done
for spec in "${TASKS_C[@]}"; do
  IFS='|' read -r _profile _dataset _run_name _group _role <<< "${spec}"
  register_run "${_run_name}" "${_dataset}" "${_profile}" "${_group}" "${_role}" "${GPU_C}"
done
for spec in "${TASKS_D[@]}"; do
  IFS='|' read -r _profile _dataset _run_name _group _role <<< "${spec}"
  register_run "${_run_name}" "${_dataset}" "${_profile}" "${_group}" "${_role}" "${GPU_D}"
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
    "phase": "phase6w_unit_transition_atomization",
    "sample_size": int(os.environ.get("SAMPLE_SIZE", "0")),
    "out_root": str(out_root),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": runs,
}
(out_root / "phase6w_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(out_root / "phase6s_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(out_root / "phase6w_run_manifest.json")
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
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    sid = str(obj.get("sample_id", "")).strip()
    if sid:
        ids.append(sid)
out.write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\\n", encoding='utf-8')
print(out)
PY
}

wait_for_sample_ids() {
  local dataset="$1"
  local max_wait_sec="${2:-3600}"
  local waited=0
  local path="${OUT_ROOT}/sampled_query_ids/${dataset}.json"
  while [[ ! -f "${path}" ]]; do
    if [[ "${waited}" -ge "${max_wait_sec}" ]]; then
      echo "[ERROR] timeout waiting for baseline sample ids: ${path}" >&2
      return 3
    fi
    sleep 5
    waited=$((waited + 5))
  done
  return 0
}

check_sample_ids_match_baseline() {
  local variant_run="$1"
  local dataset="$2"
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
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    sid = str(obj.get('sample_id', '')).strip()
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
  local gpu="$2"
  IFS='|' read -r profile dataset run_name group role <<< "${spec}"
  rm -rf "${OUT_ROOT}/qa_runs/${run_name}"
  mkdir -p "${OUT_ROOT}/qa_runs/${run_name}"

  if [[ "${role}" != "baseline" ]]; then
    wait_for_sample_ids "${dataset}" 3600
  fi

  echo "=== [GROUP ${group}] START run=${run_name} profile=${profile} dataset=${dataset} gpu=${gpu} ==="
  if run_unified_one_dataset "${profile}" "${dataset}" "${run_name}" "${role}" "${gpu}"; then
    if [[ "${role}" == "baseline" ]]; then
      write_sample_ids "${run_name}" "${dataset}" || true
    else
      check_sample_ids_match_baseline "${run_name}" "${dataset}"
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
group_c() { local spec; for spec in "${TASKS_C[@]}"; do run_task "${spec}" "${GPU_C}"; done; }
group_d() { local spec; for spec in "${TASKS_D[@]}"; do run_task "${spec}" "${GPU_D}"; done; }

group_a > "${OUT_ROOT}/logs/group_A_gpu${GPU_A}.log" 2>&1 &
PID_A=$!
group_b > "${OUT_ROOT}/logs/group_B_gpu${GPU_B}.log" 2>&1 &
PID_B=$!
PID_C=""
if [[ "${PROCESS_COUNT}" != "2" ]]; then
  group_c > "${OUT_ROOT}/logs/group_C_gpu${GPU_C}.log" 2>&1 &
  PID_C=$!
fi
PID_D=""
if [[ "${PROCESS_COUNT}" == "4" ]]; then
  group_d > "${OUT_ROOT}/logs/group_D_gpu${GPU_D}.log" 2>&1 &
  PID_D=$!
fi

FAIL=0
if ! wait "${PID_A}"; then FAIL=1; fi
if ! wait "${PID_B}"; then FAIL=1; fi
if [[ -n "${PID_C}" ]]; then
  if ! wait "${PID_C}"; then FAIL=1; fi
fi
if [[ -n "${PID_D}" ]]; then
  if ! wait "${PID_D}"; then FAIL=1; fi
fi
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] one or more workers failed. Check logs/group_*.log" >&2
  exit 1
fi

echo "=== summarize PHASE6W unit transition atomization ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6w_unit_transition_atomization.py \
  --out-root "${OUT_ROOT}" \
  --datasets hotpotqa,2wikimultihopqa \
  --baseline-profile "${BASELINE_PROFILE}" \
  --variant-profiles "${NO_ATOM_PROFILE},${ATOM3_PROFILE},${RENDER_PACKAGE_PROFILE}" \
  --top-k 20 \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6w_unit_transition_atomization.log"

echo "=== artifact consistency validation ==="
VALIDATE_ARGS=()
if [[ "${STRICT_VALIDATE}" == "1" ]]; then
  VALIDATE_ARGS+=(--strict-clean)
fi
PYTHONPATH=. "${PYTHON}" scripts/validate_phase6s_artifacts.py \
  --out-root "${OUT_ROOT}" \
  "${VALIDATE_ARGS[@]}" \
  --json "${OUT_ROOT}/phase6w_artifact_validation.json" \
  2>&1 | tee "${OUT_ROOT}/logs/validate_phase6w_artifacts.log"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_ROOT}/logs/nvidia_smi_after.txt" || true
fi

echo "=== done ==="
echo "Summary: ${OUT_ROOT}/PHASE6W_UNIT_TRANSITION_ATOMIZATION_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6w_unit_transition_atomization_summary.json"
