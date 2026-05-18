#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6t_proposal_prepost_hotpotqa_n20}"
SAMPLE_SIZE="${SAMPLE_SIZE:-20}"
PROCESS_COUNT="${PROCESS_COUNT:-1}"
DATASET="${DATASET:-hotpotqa}"
GPU="${GPU:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
PROFILE="${PROFILE:-unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank}"
RUN_NAME="${RUN_NAME:-fast_no_sentence_rerank_hotpotqa}"

if [[ "${PROCESS_COUNT}" != "1" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 1 for clean proposal pre/post profiling." >&2
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

echo "=== Phase-6T proposal pre/post profile (GPU1-only) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "PROCESS_COUNT=${PROCESS_COUNT}"
echo "GPU=${GPU}"
echo "DATASET=${DATASET}"
echo "PROFILE=${PROFILE}"
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

echo "=== Phase-6T audit gate ==="
PYTHONPATH=. "${PYTHON}" scripts/audit_phase6t_latency_triage.py \
  --profiles "${PROFILE}" \
  --datasets "${DATASET}" \
  --check-configs configs/main_config/copy_span_instruction_unified \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_phase6t_proposal_prepost.log"

SCHEDULE_JSONL="${OUT_ROOT}/_phase6t_proposal_prepost_scheduled_runs.jsonl"
: > "${SCHEDULE_JSONL}"
printf '{"run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"A","role":"proposal_prepost_profile","kind":"unified","scheduled":true}\n' \
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
    "phase": "phase6t_proposal_prepost_profile",
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

echo "=== [GPU ${GPU}] profile=${PROFILE} dataset=${DATASET} run=${RUN_NAME} ==="
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON}" \
  scripts/run_unified_config_4ds.py \
  --profile "${PROFILE}" \
  --datasets "${DATASET}" \
  --sample-size "${SAMPLE_SIZE}" \
  --output-root "${OUT_ROOT}/qa_runs/${RUN_NAME}" \
  2>&1 | tee "${OUT_ROOT}/logs/qa_${RUN_NAME}.log"

echo "=== summarize Phase-6T proposal pre/post profile ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6t_proposal_prepost_profile.py \
  --out-root "${OUT_ROOT}" \
  --dataset "${DATASET}" \
  --profile "${PROFILE}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6t_proposal_prepost.log"

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
echo "Summary: ${OUT_ROOT}/PHASE6T_PROPOSAL_PRE_POST_PROFILE.md"
echo "JSON: ${OUT_ROOT}/proposal_pre_post_profile_summary.json"
