#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6t_rerank20_fast_n1000_gpu1_2proc}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
PROCESS_COUNT="${PROCESS_COUNT:-2}"
GPU="${GPU:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"

if [[ "${PROCESS_COUNT}" != "2" && "${PROCESS_COUNT}" != "3" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 2 or 3, got: ${PROCESS_COUNT}" >&2
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

echo "=== Phase-6T rerank20/fast n1000 (GPU1-only) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "PROCESS_COUNT=${PROCESS_COUNT}"
echo "GPU=${GPU}"

echo "=== Phase-6T candidate audit gate ==="
PYTHONPATH=. "${PYTHON}" scripts/audit_phase6t_latency_triage.py \
  --profiles \
    unified_acr_rcedr_v12_sota_contract_rerank20 \
    unified_acr_rcedr_v12_sota_contract_fast \
  --datasets hotpotqa 2wikimultihopqa \
  --check-configs configs/main_config/copy_span_instruction_unified \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_phase6t_rerank20_fast.log"

SCHEDULE_JSONL="${OUT_ROOT}/_phase6t_rerank20_fast_scheduled_runs.jsonl"
: > "${SCHEDULE_JSONL}"

register_run () {
  local profile="$1"
  local dataset="$2"
  local run_name="$3"
  local group="$4"
  printf '{"run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"%s","role":"candidate","kind":"unified","scheduled":true}\n' \
    "${run_name}" "${dataset}" "${profile}" "${GPU}" "${group}" >> "${SCHEDULE_JSONL}"
}

run_unified_qa_one_dataset () {
  local profile="$1"
  local dataset="$2"
  local run_name="$3"
  echo "=== [GPU ${GPU}] profile=${profile} dataset=${dataset} run=${run_name} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets "${dataset}" \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${run_name}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${run_name}.log"
}

run_task () {
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

TASKS_A=()
TASKS_B=()
TASKS_C=()

if [[ "${PROCESS_COUNT}" == "3" ]]; then
  TASKS_A=(
    "unified_acr_rcedr_v12_sota_contract_rerank20|hotpotqa|rerank20_hotpotqa|A"
  )
  TASKS_B=(
    "unified_acr_rcedr_v12_sota_contract_fast|hotpotqa|fast_hotpotqa|B"
  )
  TASKS_C=(
    "unified_acr_rcedr_v12_sota_contract_rerank20|2wikimultihopqa|rerank20_2wiki|C"
    "unified_acr_rcedr_v12_sota_contract_fast|2wikimultihopqa|fast_2wiki|C"
  )
else
  TASKS_A=(
    "unified_acr_rcedr_v12_sota_contract_rerank20|hotpotqa|rerank20_hotpotqa|A"
    "unified_acr_rcedr_v12_sota_contract_fast|2wikimultihopqa|fast_2wiki|A"
  )
  TASKS_B=(
    "unified_acr_rcedr_v12_sota_contract_fast|hotpotqa|fast_hotpotqa|B"
    "unified_acr_rcedr_v12_sota_contract_rerank20|2wikimultihopqa|rerank20_2wiki|B"
  )
fi

ALL_TASKS=("${TASKS_A[@]}" "${TASKS_B[@]}" "${TASKS_C[@]}")

declare -A RUN_NAME_COUNT
for spec in "${ALL_TASKS[@]}"; do
  [[ -z "${spec}" ]] && continue
  IFS='|' read -r _profile _dataset _run_name _group <<< "${spec}"
  RUN_NAME_COUNT["${_run_name}"]=$(( ${RUN_NAME_COUNT["${_run_name}"]:-0} + 1 ))
done
for run_name in "${!RUN_NAME_COUNT[@]}"; do
  if [[ "${RUN_NAME_COUNT[${run_name}]}" -gt 1 ]]; then
    echo "[ERROR] duplicate run_name detected: ${run_name}" >&2
    exit 1
  fi
done

for spec in "${ALL_TASKS[@]}"; do
  [[ -z "${spec}" ]] && continue
  IFS='|' read -r profile dataset run_name group <<< "${spec}"
  register_run "${profile}" "${dataset}" "${run_name}" "${group}"
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
    "phase": "phase6t_rerank20_fast_n1000",
    "sample_size": int(os.environ.get("SAMPLE_SIZE", "0")),
    "out_root": str(out_root),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": runs,
}

phase6t = out_root / "phase6t_run_manifest.json"
phase6s_alias = out_root / "phase6s_run_manifest.json"
phase6t.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
phase6s_alias.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(str(phase6t))
print(str(phase6s_alias))
PY

group_a () { local spec; for spec in "${TASKS_A[@]}"; do run_task "${spec}"; done; }
group_b () { local spec; for spec in "${TASKS_B[@]}"; do run_task "${spec}"; done; }
group_c () { local spec; for spec in "${TASKS_C[@]}"; do run_task "${spec}"; done; }

group_a > "${OUT_ROOT}/logs/group_A_gpu1.log" 2>&1 &
PID_A=$!
group_b > "${OUT_ROOT}/logs/group_B_gpu1.log" 2>&1 &
PID_B=$!
if [[ "${PROCESS_COUNT}" == "3" ]]; then
  group_c > "${OUT_ROOT}/logs/group_C_gpu1.log" 2>&1 &
  PID_C=$!
fi

FAIL=0
if ! wait "${PID_A}"; then FAIL=1; fi
if ! wait "${PID_B}"; then FAIL=1; fi
if [[ "${PROCESS_COUNT}" == "3" ]]; then
  if ! wait "${PID_C}"; then FAIL=1; fi
fi
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] One or more workers failed. Check ${OUT_ROOT}/logs/group_*.log" >&2
  exit 1
fi

echo "=== summarize Phase-6T rerank20/fast ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6t_rerank20_fast_n1000.py \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6t_rerank20_fast.log"

echo "=== artifact consistency validation ==="
VALIDATE_ARGS=()
if [[ "${STRICT_VALIDATE}" == "1" ]]; then
  VALIDATE_ARGS+=(--strict-clean)
fi
PYTHONPATH=. "${PYTHON}" scripts/validate_phase6s_artifacts.py \
  --out-root "${OUT_ROOT}" \
  "${VALIDATE_ARGS[@]}" \
  --json "${OUT_ROOT}/phase6t_artifact_validation.json" \
  2>&1 | tee "${OUT_ROOT}/logs/validate_phase6t_artifacts.log"

echo "[Phase-6T] scheduled runs:"
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
path = os.path.join(os.environ["OUT_ROOT"], "phase6t_run_manifest.json")
if os.path.exists(path):
    print(len(json.load(open(path)).get("runs", [])))
else:
    print("missing phase6t_run_manifest.json")
PY

echo "[Phase-6T] rag_summary count:"
find "${OUT_ROOT}" -name rag_summary.json | wc -l

echo "[Phase-6T] rag_query_results count:"
find "${OUT_ROOT}" -name rag_query_results.jsonl | wc -l

echo "=== done ==="
echo "Summary: ${OUT_ROOT}/PHASE6T_RERANK20_FAST_N1000_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6t_rerank20_fast_n1000_summary.json"
