#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6t_score_attachment_stability_hotpotqa_n50}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
PROCESS_COUNT="${PROCESS_COUNT:-1}"
DATASET="${DATASET:-hotpotqa}"
GPU="${GPU:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PROFILE="${PROFILE:-unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank}"

RUN_BASELINE="score_attach_baseline_hotpotqa"
RUN_OPTIMIZED="score_attach_optimized_hotpotqa"

if [[ "${PROCESS_COUNT}" != "1" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 1 for stability check." >&2
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

mkdir -p "${OUT_ROOT}/logs"

# Enable score-attachment trace for per-run latency internals unless user turns it off.
export PHASE6T_SCORE_ATTACH_TRACE="${PHASE6T_SCORE_ATTACH_TRACE:-1}"
export PHASE6T_SCORE_ATTACH_TRACE_LIMIT="${PHASE6T_SCORE_ATTACH_TRACE_LIMIT:-${SAMPLE_SIZE}}"
export PHASE6T_SCORE_ATTACH_TRACE_SAMPLE_IDS="${PHASE6T_SCORE_ATTACH_TRACE_SAMPLE_IDS:-}"

echo "=== Phase-6T score-attachment stability (GPU1 serial) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "PROCESS_COUNT=${PROCESS_COUNT}"
echo "GPU=${GPU}"
echo "DATASET=${DATASET}"
echo "PROFILE=${PROFILE}"
echo "PHASE6T_SCORE_ATTACH_TRACE=${PHASE6T_SCORE_ATTACH_TRACE}"
echo "PHASE6T_SCORE_ATTACH_TRACE_LIMIT=${PHASE6T_SCORE_ATTACH_TRACE_LIMIT}"
echo "PHASE6T_SCORE_ATTACH_TRACE_SAMPLE_IDS=${PHASE6T_SCORE_ATTACH_TRACE_SAMPLE_IDS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

action_nvidia_smi() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$1" || true
  fi
}
action_nvidia_smi "${OUT_ROOT}/logs/nvidia_smi_before.txt"

SCHEDULE_JSONL="${OUT_ROOT}/_phase6t_score_attachment_stability_runs.jsonl"
: > "${SCHEDULE_JSONL}"
printf '{"run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"A","role":"baseline","kind":"unified","scheduled":true}\n' \
  "${RUN_BASELINE}" "${DATASET}" "${PROFILE}" "${GPU}" >> "${SCHEDULE_JSONL}"
printf '{"run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"A","role":"optimized","kind":"unified","scheduled":true}\n' \
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
    "phase": "phase6t_score_attachment_stability",
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
  local trace_log="${OUT_ROOT}/logs/score_attachment_trace_${run_name}.log"

  rm -rf "${OUT_ROOT}/qa_runs/${run_name}"
  mkdir -p "${OUT_ROOT}/qa_runs/${run_name}"
  : > "${trace_log}"

  echo "=== [GPU ${GPU}] run=${run_name} opt=${opt_flag} sample_size=${SAMPLE_SIZE} ==="
  PHASE6T_SCORE_ATTACHMENT_OPT="${opt_flag}" \
  PHASE6T_SCORE_ATTACH_TRACE_LOG="${trace_log}" \
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${PROFILE}" \
    --datasets "${DATASET}" \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${run_name}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${run_name}.log"

  # Generate per-run score-attachment trace summary when trace is enabled.
  if [[ "${PHASE6T_SCORE_ATTACH_TRACE}" == "1" ]]; then
    PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6t_score_attachment_trace.py \
      --trace-log "${trace_log}" \
      --out-root "${OUT_ROOT}/qa_runs/${run_name}" \
      2>&1 | tee "${OUT_ROOT}/logs/summarize_score_attach_trace_${run_name}.log"
  fi
}

run_one "${RUN_BASELINE}" "0"
run_one "${RUN_OPTIMIZED}" "1"

echo "=== compare baseline vs optimized stability ==="
PYTHONPATH=. "${PYTHON}" scripts/compare_phase6t_score_attachment_stability.py \
  --baseline-root "${OUT_ROOT}/qa_runs/${RUN_BASELINE}" \
  --optimized-root "${OUT_ROOT}/qa_runs/${RUN_OPTIMIZED}" \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/compare_phase6t_score_attachment_stability.log"

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
echo "Summary: ${OUT_ROOT}/PHASE6T_SCORE_ATTACHMENT_STABILITY_SUMMARY.md"
echo "JSON: ${OUT_ROOT}/phase6t_score_attachment_stability_summary.json"
echo "Changed examples: ${OUT_ROOT}/changed_query_examples.md"
