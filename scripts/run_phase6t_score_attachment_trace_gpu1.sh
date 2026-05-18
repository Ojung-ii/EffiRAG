#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6t_score_attachment_trace_hotpotqa_n5}"
SAMPLE_SIZE="${SAMPLE_SIZE:-5}"
PROCESS_COUNT="${PROCESS_COUNT:-1}"
DATASET="${DATASET:-hotpotqa}"
GPU="${GPU:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PHASE6T_SCORE_ATTACH_CPROFILE="${PHASE6T_SCORE_ATTACH_CPROFILE:-0}"
PHASE6T_SCORE_ATTACH_CPROFILE_LIMIT="${PHASE6T_SCORE_ATTACH_CPROFILE_LIMIT:-3}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PROFILE="${PROFILE:-unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank}"
RUN_NAME="${RUN_NAME:-score_attach_trace_hotpotqa}"

if [[ "${PROCESS_COUNT}" != "1" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 1 for score-attachment trace profiling." >&2
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

export PHASE6T_SCORE_ATTACH_TRACE="${PHASE6T_SCORE_ATTACH_TRACE:-1}"
export PHASE6T_SCORE_ATTACH_TRACE_LIMIT="${PHASE6T_SCORE_ATTACH_TRACE_LIMIT:-5}"
export PHASE6T_SCORE_ATTACH_TRACE_SAMPLE_IDS="${PHASE6T_SCORE_ATTACH_TRACE_SAMPLE_IDS:-}"
export PHASE6T_SCORE_ATTACH_TRACE_LOG="${OUT_ROOT}/logs/score_attachment_trace.log"
: > "${PHASE6T_SCORE_ATTACH_TRACE_LOG}"

echo "=== Phase-6T score-attachment trace (GPU1-only) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "PROCESS_COUNT=${PROCESS_COUNT}"
echo "GPU=${GPU}"
echo "DATASET=${DATASET}"
echo "PROFILE=${PROFILE}"
echo "PHASE6T_SCORE_ATTACH_TRACE=${PHASE6T_SCORE_ATTACH_TRACE}"
echo "PHASE6T_SCORE_ATTACH_TRACE_LIMIT=${PHASE6T_SCORE_ATTACH_TRACE_LIMIT}"
echo "PHASE6T_SCORE_ATTACH_TRACE_SAMPLE_IDS=${PHASE6T_SCORE_ATTACH_TRACE_SAMPLE_IDS}"
echo "PHASE6T_SCORE_ATTACH_TRACE_LOG=${PHASE6T_SCORE_ATTACH_TRACE_LOG}"
echo "PHASE6T_SCORE_ATTACH_CPROFILE=${PHASE6T_SCORE_ATTACH_CPROFILE}"
echo "PHASE6T_SCORE_ATTACH_CPROFILE_LIMIT=${PHASE6T_SCORE_ATTACH_CPROFILE_LIMIT}"

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

SCHEDULE_JSONL="${OUT_ROOT}/_phase6t_score_attachment_trace_scheduled_runs.jsonl"
: > "${SCHEDULE_JSONL}"
printf '{"run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"A","role":"score_attachment_trace_profile","kind":"unified","scheduled":true}\n' \
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

seen = set()
for row in runs:
    name = str(row.get("run_name", "") or "").strip()
    if not name:
        raise SystemExit("[ERROR] empty run_name in schedule")
    if name in seen:
        raise SystemExit(f"[ERROR] duplicate run_name detected: {name}")
    seen.add(name)

manifest = {
    "phase": "phase6t_score_attachment_trace",
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
if [[ "${PHASE6T_SCORE_ATTACH_CPROFILE}" == "1" ]]; then
  if [[ "${PHASE6T_SCORE_ATTACH_CPROFILE_LIMIT}" =~ ^[0-9]+$ ]] && [[ "${PHASE6T_SCORE_ATTACH_CPROFILE_LIMIT}" -gt 0 ]] && [[ "${PHASE6T_SCORE_ATTACH_CPROFILE_LIMIT}" -lt "${SAMPLE_SIZE}" ]]; then
    RUN_SAMPLE_SIZE="${PHASE6T_SCORE_ATTACH_CPROFILE_LIMIT}"
  fi
fi

echo "=== [GPU ${GPU}] profile=${PROFILE} dataset=${DATASET} run=${RUN_NAME} sample_size=${RUN_SAMPLE_SIZE} ==="
if [[ "${PHASE6T_SCORE_ATTACH_CPROFILE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" -m cProfile \
    -o "${OUT_ROOT}/score_attachment.cprof" \
    scripts/run_unified_config_4ds.py \
    --profile "${PROFILE}" \
    --datasets "${DATASET}" \
    --sample-size "${RUN_SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${RUN_NAME}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${RUN_NAME}.log"

  PYTHONPATH=. "${PYTHON}" scripts/summarize_cprofile.py \
    --profile "${OUT_ROOT}/score_attachment.cprof" \
    --out "${OUT_ROOT}/PHASE6T_SCORE_ATTACHMENT_CPROFILE_TOP_FUNCTIONS.txt" \
    2>&1 | tee "${OUT_ROOT}/logs/summarize_cprofile.log"
else
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${PROFILE}" \
    --datasets "${DATASET}" \
    --sample-size "${RUN_SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${RUN_NAME}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${RUN_NAME}.log"
fi

echo "=== summarize score-attachment trace ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6t_score_attachment_trace.py \
  --trace-log "${OUT_ROOT}/logs/score_attachment_trace.log" \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6t_score_attachment_trace.log"

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
echo "Trace log: ${OUT_ROOT}/logs/score_attachment_trace.log"
echo "Summary: ${OUT_ROOT}/PHASE6T_SCORE_ATTACHMENT_TRACE_SUMMARY.md"
echo "JSON: ${OUT_ROOT}/score_attachment_trace_summary.json"
