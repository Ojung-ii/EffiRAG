#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6t_candidate_score_parity_hotpotqa_n20}"
SAMPLE_SIZE="${SAMPLE_SIZE:-20}"
PROCESS_COUNT="${PROCESS_COUNT:-1}"
DATASET="${DATASET:-hotpotqa}"
GPU="${GPU:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PROFILE="${PROFILE:-unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank}"

RUN_BASELINE="score_parity_baseline_hotpotqa"
RUN_OPTIMIZED="score_parity_optimized_hotpotqa"

if [[ "${PROCESS_COUNT}" != "1" && "${PROCESS_COUNT}" != "2" && "${PROCESS_COUNT}" != "3" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 1, 2, or 3 for candidate-level score parity debug." >&2
  exit 1
fi
if [[ "${GPU}" != "1" ]]; then
  echo "[ERROR] This runner is GPU1-only. Set GPU=1." >&2
  exit 1
fi
if [[ "${DATASET}" != "hotpotqa" ]]; then
  echo "[ERROR] This runner currently supports DATASET=hotpotqa only." >&2
  exit 1
fi
if [[ -d "${OUT_ROOT}/qa_runs" ]] && [[ "${ALLOW_REUSE_OUT_ROOT:-0}" != "1" ]]; then
  echo "[ERROR] OUT_ROOT already contains qa_runs: ${OUT_ROOT}" >&2
  echo "Use a fresh OUT_ROOT or set ALLOW_REUSE_OUT_ROOT=1 explicitly." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/score_parity"

BASE_CAND_FILE="${OUT_ROOT}/score_parity/baseline_candidates.jsonl"
OPT_CAND_FILE="${OUT_ROOT}/score_parity/optimized_candidates.jsonl"
: > "${BASE_CAND_FILE}"
: > "${OPT_CAND_FILE}"

echo "=== Phase-6T candidate-level score parity debug (GPU1 serial) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "PROCESS_COUNT=${PROCESS_COUNT}"
echo "GPU=${GPU}"
echo "DATASET=${DATASET}"
echo "PROFILE=${PROFILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
echo "PYTHONHASHSEED=${PYTHONHASHSEED}"

action_nvidia_smi() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$1" || true
  fi
}
action_nvidia_smi "${OUT_ROOT}/logs/nvidia_smi_before.txt"

SCHEDULE_JSONL="${OUT_ROOT}/_phase6t_candidate_score_parity_runs.jsonl"
: > "${SCHEDULE_JSONL}"
printf '{"run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"A","role":"parity_debug","kind":"unified","scheduled":true,"score_attachment_opt":false}\n' \
  "${RUN_BASELINE}" "${DATASET}" "${PROFILE}" "${GPU}" >> "${SCHEDULE_JSONL}"
printf '{"run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"A","role":"parity_debug","kind":"unified","scheduled":true,"score_attachment_opt":true}\n' \
  "${RUN_OPTIMIZED}" "${DATASET}" "${PROFILE}" "${GPU}" >> "${SCHEDULE_JSONL}"

export OUT_ROOT SAMPLE_SIZE SCHEDULE_JSONL
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"]).resolve()
jsonl = Path(os.environ["SCHEDULE_JSONL"]).resolve()
runs = []
for line in jsonl.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    if isinstance(obj, dict):
        runs.append(obj)

seen = set()
for row in runs:
    rn = str(row.get("run_name", "") or "").strip()
    if not rn:
        raise SystemExit("[ERROR] empty run_name in schedule")
    if rn in seen:
        raise SystemExit(f"[ERROR] duplicate run_name detected: {rn}")
    seen.add(rn)

manifest = {
    "phase": "phase6t_candidate_score_parity",
    "sample_size": int(os.environ.get("SAMPLE_SIZE", "0")),
    "out_root": str(out_root),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": runs,
}
(out_root / "phase6t_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(out_root / "phase6s_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(str(out_root / "phase6t_run_manifest.json"))
PY

run_one() {
  local run_name="$1"
  local opt_flag="$2"
  local mode_name="$3"
  local dump_file="$4"

  rm -rf "${OUT_ROOT}/qa_runs/${run_name}"
  mkdir -p "${OUT_ROOT}/qa_runs/${run_name}"

  echo "=== [GPU ${GPU}] run=${run_name} opt=${opt_flag} mode=${mode_name} sample_size=${SAMPLE_SIZE} ==="
  PHASE6T_SCORE_ATTACHMENT_OPT="${opt_flag}" \
  PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER="0" \
  PHASE6T_SCORE_ATTACHMENT_PARITY_DUMP="1" \
  PHASE6T_SCORE_PARITY_MODE="${mode_name}" \
  PHASE6T_SCORE_PARITY_DUMP_PATH="${dump_file}" \
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${PROFILE}" \
    --datasets "${DATASET}" \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${run_name}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${run_name}.log"
}

if [[ "${PROCESS_COUNT}" == "1" ]]; then
  run_one "${RUN_BASELINE}" "0" "baseline" "${BASE_CAND_FILE}"
  run_one "${RUN_OPTIMIZED}" "1" "optimized" "${OPT_CAND_FILE}"
else
  if [[ "${PROCESS_COUNT}" == "3" ]]; then
    echo "[WARN] PROCESS_COUNT=3 requested, but only 2 runs exist. Using 2 parallel workers."
  fi
  run_one "${RUN_BASELINE}" "0" "baseline" "${BASE_CAND_FILE}" &
  PID_BASE=$!
  run_one "${RUN_OPTIMIZED}" "1" "optimized" "${OPT_CAND_FILE}" &
  PID_OPT=$!

  FAIL=0
  if ! wait "${PID_BASE}"; then FAIL=1; fi
  if ! wait "${PID_OPT}"; then FAIL=1; fi
  if [[ "${FAIL}" != "0" ]]; then
    echo "[ERROR] One or more parallel runs failed. Check ${OUT_ROOT}/logs/qa_*.log" >&2
    exit 1
  fi
fi

echo "=== candidate-level score parity compare ==="
PYTHONPATH=. "${PYTHON}" scripts/compare_phase6t_candidate_score_parity.py \
  --baseline-candidates "${BASE_CAND_FILE}" \
  --optimized-candidates "${OPT_CAND_FILE}" \
  --baseline-root "${OUT_ROOT}/qa_runs/${RUN_BASELINE}" \
  --optimized-root "${OUT_ROOT}/qa_runs/${RUN_OPTIMIZED}" \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/compare_phase6t_candidate_score_parity.log"

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

action_nvidia_smi "${OUT_ROOT}/logs/nvidia_smi_after.txt"

echo "[Phase-6T] scheduled runs:"
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
p = os.path.join(os.environ["OUT_ROOT"], "phase6t_run_manifest.json")
if os.path.exists(p):
    print(len(json.load(open(p)).get("runs", [])))
else:
    print("missing phase6t_run_manifest.json")
PY

echo "[Phase-6T] rag_summary count:"
find "${OUT_ROOT}" -name rag_summary.json | wc -l

echo "[Phase-6T] rag_query_results count:"
find "${OUT_ROOT}" -name rag_query_results.jsonl | wc -l

echo "=== done ==="
echo "Summary: ${OUT_ROOT}/PHASE6T_SCORE_ATTACHMENT_PARITY_REPAIR_SUMMARY.md"
echo "JSON: ${OUT_ROOT}/phase6t_score_attachment_parity_repair_summary.json"
echo "Top diff examples: ${OUT_ROOT}/score_parity/top_score_diff_examples.md"
