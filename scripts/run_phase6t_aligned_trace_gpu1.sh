#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6t_aligned_trace_hotpotqa_n5}"
SAMPLE_SIZE="${SAMPLE_SIZE:-5}"
PROCESS_COUNT="${PROCESS_COUNT:-1}"
DATASET="${DATASET:-hotpotqa}"
GPU="${GPU:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PHASE6T_CPROFILE="${PHASE6T_CPROFILE:-0}"
PHASE6T_CPROFILE_LIMIT="${PHASE6T_CPROFILE_LIMIT:-3}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PROFILE="${PROFILE:-unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank}"
RUN_NAME="${RUN_NAME:-fast_no_sentence_rerank_hotpotqa}"

if [[ "${PROCESS_COUNT}" != "1" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 1 for aligned trace profiling." >&2
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

export PHASE6T_PROP_TRACE="${PHASE6T_PROP_TRACE:-1}"
export PHASE6T_PROP_TRACE_LIMIT="${PHASE6T_PROP_TRACE_LIMIT:-5}"
export PHASE6T_PROP_TRACE_SAMPLE_IDS="${PHASE6T_PROP_TRACE_SAMPLE_IDS:-}"
export PHASE6T_PROP_TRACE_LOG="${OUT_ROOT}/logs/proposal_trace.log"
: > "${PHASE6T_PROP_TRACE_LOG}"

export PHASE6T_CAND_UNION_TRACE="${PHASE6T_CAND_UNION_TRACE:-1}"
export PHASE6T_CAND_UNION_TRACE_LIMIT="${PHASE6T_CAND_UNION_TRACE_LIMIT:-5}"
export PHASE6T_CAND_UNION_TRACE_SAMPLE_IDS="${PHASE6T_CAND_UNION_TRACE_SAMPLE_IDS:-}"
export PHASE6T_CAND_UNION_TRACE_LOG="${OUT_ROOT}/logs/candidate_union_trace.log"
: > "${PHASE6T_CAND_UNION_TRACE_LOG}"

export PHASE6T_ALIGNED_TRACE="${PHASE6T_ALIGNED_TRACE:-1}"
export PHASE6T_ALIGNED_TRACE_LIMIT="${PHASE6T_ALIGNED_TRACE_LIMIT:-5}"
export PHASE6T_ALIGNED_TRACE_SAMPLE_IDS="${PHASE6T_ALIGNED_TRACE_SAMPLE_IDS:-}"
export PHASE6T_ALIGNED_TRACE_LOG="${OUT_ROOT}/logs/proposal_aligned_trace.log"
: > "${PHASE6T_ALIGNED_TRACE_LOG}"

append_cprofile_marker () {
  local step="$1"
  local extra_json="$2"
  LOG_PATH="${PHASE6T_ALIGNED_TRACE_LOG}" STEP_NAME="${step}" EXTRA_JSON="${extra_json}" PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
import time

log_path = str(os.environ.get("LOG_PATH", "") or "")
step = str(os.environ.get("STEP_NAME", "") or "")
extra_raw = str(os.environ.get("EXTRA_JSON", "{}") or "{}")
try:
    extra = json.loads(extra_raw)
except Exception:
    extra = {"raw": extra_raw}
t = float(time.perf_counter())
line = (
    f"[PROP_ALIGNED_TRACE] qid=__runner__ trace_id=__runner__ scope=cprofile_marker "
    f"step={step} t={t:.6f} dt_prev_ms=0.000 dt_total_ms=0.000 "
    f"extra={json.dumps(extra, ensure_ascii=False, sort_keys=True)}\\n"
)
if log_path:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
PY
}

echo "=== Phase-6T aligned proposal trace (GPU1-only) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "PROCESS_COUNT=${PROCESS_COUNT}"
echo "GPU=${GPU}"
echo "DATASET=${DATASET}"
echo "PROFILE=${PROFILE}"
echo "PHASE6T_PROP_TRACE=${PHASE6T_PROP_TRACE}"
echo "PHASE6T_CAND_UNION_TRACE=${PHASE6T_CAND_UNION_TRACE}"
echo "PHASE6T_ALIGNED_TRACE=${PHASE6T_ALIGNED_TRACE}"
echo "PHASE6T_CPROFILE=${PHASE6T_CPROFILE}"
echo "PHASE6T_CPROFILE_LIMIT=${PHASE6T_CPROFILE_LIMIT}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "OPENAI_API_BASE=${OPENAI_API_BASE:-unset}"
echo "VLLM_BASE_URL=${VLLM_BASE_URL:-unset}"
echo "LLM_BASE_URL=${LLM_BASE_URL:-unset}"
echo "EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL:-unset}"
echo "EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME:-unset}"
echo "EMBEDDING_LOCAL_MODE=${EMBEDDING_LOCAL_MODE:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_ROOT}/logs/nvidia_smi_before.txt" || true
fi

SCHEDULE_JSONL="${OUT_ROOT}/_phase6t_aligned_trace_scheduled_runs.jsonl"
: > "${SCHEDULE_JSONL}"
printf '{"run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"A","role":"aligned_trace_profile","kind":"unified","scheduled":true}\n' \
  "${RUN_NAME}" "${DATASET}" "${PROFILE}" "${GPU}" >> "${SCHEDULE_JSONL}"

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
        obj = json.loads(line)
        if isinstance(obj, dict):
            runs.append(obj)

manifest = {
    "phase": "phase6t_aligned_trace",
    "sample_size": int(os.environ.get("SAMPLE_SIZE", "0")),
    "out_root": str(out_root),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": runs,
}
(out_root / "phase6t_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(out_root / "phase6s_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(str(out_root / "phase6t_run_manifest.json"))
PY

rm -rf "${OUT_ROOT}/qa_runs/${RUN_NAME}"
mkdir -p "${OUT_ROOT}/qa_runs/${RUN_NAME}"

RUN_SAMPLE_SIZE="${SAMPLE_SIZE}"
if [[ "${PHASE6T_CPROFILE}" == "1" ]]; then
  if [[ "${PHASE6T_CPROFILE_LIMIT}" =~ ^[0-9]+$ ]] && [[ "${PHASE6T_CPROFILE_LIMIT}" -gt 0 ]] && [[ "${PHASE6T_CPROFILE_LIMIT}" -lt "${SAMPLE_SIZE}" ]]; then
    RUN_SAMPLE_SIZE="${PHASE6T_CPROFILE_LIMIT}"
  fi
fi

echo "=== [GPU ${GPU}] profile=${PROFILE} dataset=${DATASET} run=${RUN_NAME} sample_size=${RUN_SAMPLE_SIZE} ==="
if [[ "${PHASE6T_CPROFILE}" == "1" ]]; then
  append_cprofile_marker "cprofile_start" "{\"sample_size\": ${RUN_SAMPLE_SIZE}}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" -m cProfile \
    -o "${OUT_ROOT}/phase6t_aligned_trace.cprof" \
    scripts/run_unified_config_4ds.py \
    --profile "${PROFILE}" \
    --datasets "${DATASET}" \
    --sample-size "${RUN_SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${RUN_NAME}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${RUN_NAME}.log"

  PYTHONPATH=. "${PYTHON}" scripts/summarize_cprofile.py \
    --profile "${OUT_ROOT}/phase6t_aligned_trace.cprof" \
    --out "${OUT_ROOT}/PHASE6T_CPROFILE_TOP_FUNCTIONS.txt" \
    2>&1 | tee "${OUT_ROOT}/logs/summarize_cprofile.log"
  append_cprofile_marker "cprofile_end" "{\"sample_size\": ${RUN_SAMPLE_SIZE}}"
else
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${PROFILE}" \
    --datasets "${DATASET}" \
    --sample-size "${RUN_SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${RUN_NAME}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${RUN_NAME}.log"
fi

echo "=== summarize aligned proposal trace ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6t_aligned_trace.py \
  --trace-log "${OUT_ROOT}/logs/proposal_aligned_trace.log" \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6t_aligned_trace.log"

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

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_ROOT}/logs/nvidia_smi_after.txt" || true
fi

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
echo "Aligned trace log: ${OUT_ROOT}/logs/proposal_aligned_trace.log"
echo "Summary: ${OUT_ROOT}/PHASE6T_ALIGNED_PROPOSAL_TRACE_SUMMARY.md"
echo "JSON: ${OUT_ROOT}/proposal_aligned_trace_summary.json"
