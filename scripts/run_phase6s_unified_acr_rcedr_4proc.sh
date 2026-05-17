#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6s_unified_acr_rcedr_4proc}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-1}"
GPU_D="${GPU_D:-1}"
EXTRA_POPQA="${EXTRA_POPQA:-0}"
SMOKE_CORE_ONLY="${SMOKE_CORE_ONLY:-auto}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"

mkdir -p "${OUT_ROOT}/logs"
SCHEDULE_JSONL="${OUT_ROOT}/_phase6s_scheduled_runs.jsonl"
: > "${SCHEDULE_JSONL}"
LOCK_DIR="${OUT_ROOT}/.phase6s_run_lock"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: another Phase-6S run appears active for OUT_ROOT=${OUT_ROOT} (lock exists: ${LOCK_DIR})" >&2
  echo "If no run is active, remove the lock directory manually and retry." >&2
  exit 1
fi
cleanup_lock () {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup_lock EXIT

echo "=== Phase-6S Unified ACR-RCEDR 4-process QA run ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "GPU_A=${GPU_A} GPU_B=${GPU_B} GPU_C=${GPU_C} GPU_D=${GPU_D}"
echo "EXTRA_POPQA=${EXTRA_POPQA}"
echo "SMOKE_CORE_ONLY=${SMOKE_CORE_ONLY}"
echo "STRICT_VALIDATE=${STRICT_VALIDATE}"

if [[ "${SMOKE_CORE_ONLY}" == "auto" ]]; then
  if [[ "${SAMPLE_SIZE}" -le 5 ]]; then
    SMOKE_CORE_ONLY="1"
  else
    SMOKE_CORE_ONLY="0"
  fi
fi
echo "SMOKE_CORE_ONLY(resolved)=${SMOKE_CORE_ONLY}"

register_run () {
  local kind="$1"
  local profile="$2"
  local dataset="$3"
  local tag="$4"
  local gpu="$5"
  local group="$6"
  local role="$7"
  printf '{"run_name":"%s","dataset":"%s","profile":"%s","gpu":"%s","process_group":"%s","role":"%s","kind":"%s","scheduled":true}\n' \
    "${tag}" "${dataset}" "${profile}" "${gpu}" "${group}" "${role}" "${kind}" >> "${SCHEDULE_JSONL}"
}

run_unified_qa_one_dataset () {
  local gpu="$1"
  local profile="$2"
  local dataset="$3"
  local tag="$4"
  echo "=== [GPU ${gpu}] profile=${profile} dataset=${dataset} tag=${tag} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets "${dataset}" \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${tag}.log"
}

run_legacy_qa_one_dataset () {
  local gpu="$1"
  local dataset="$2"
  local tag="$3"
  echo "=== [GPU ${gpu}] legacy_sota dataset=${dataset} tag=${tag} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_sota_config_4ds.py \
    --mode locked_precomputed \
    --datasets "${dataset}" \
    --sample-size "${SAMPLE_SIZE}" \
    --output-root "${OUT_ROOT}/qa_runs/${tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${tag}.log" || {
      echo "WARNING: legacy_sota failed or unsupported for dataset=${dataset}; continuing." | tee -a "${OUT_ROOT}/logs/qa_${tag}.log"
    }
}

normalize_run_attempt_artifacts () {
  local run_name="$1"
  local run_dir="${OUT_ROOT}/qa_runs/${run_name}"
  PYTHONPATH=. "${PYTHON}" - "${run_dir}" "${run_name}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
run_name = sys.argv[2]
if not run_dir.exists():
    raise SystemExit(0)

attempt_map = {}
for summary_path in run_dir.rglob("rag_summary.json"):
    attempt_dir = summary_path.parent
    row = attempt_map.setdefault(attempt_dir, {"summary": None, "query": None})
    row["summary"] = summary_path
for query_path in run_dir.rglob("rag_query_results.jsonl"):
    attempt_dir = query_path.parent
    row = attempt_map.setdefault(attempt_dir, {"summary": None, "query": None})
    row["query"] = query_path

if not attempt_map:
    raise SystemExit(0)

complete_attempts = sorted([p for p, row in attempt_map.items() if row.get("summary") and row.get("query")], key=lambda p: str(p))
if not complete_attempts:
    # No complete attempt yet; keep artifacts for diagnostics.
    raise SystemExit(0)

keep_attempt = complete_attempts[-1]
for attempt_dir in sorted(attempt_map.keys(), key=lambda p: str(p)):
    if attempt_dir == keep_attempt:
        continue
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir, ignore_errors=True)

# Keep run_manifest summary_path aligned with kept attempt.
manifest_path = run_dir / "run_manifest.json"
if manifest_path.exists():
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        records = payload.get("records", [])
        if isinstance(records, list):
            for row in records:
                if isinstance(row, dict):
                    row["summary_path"] = str((keep_attempt / "rag_summary.json").resolve())
            payload["records"] = records
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"[Phase-6S] normalized attempts for {run_name}: kept={keep_attempt}")
PY
}

run_task () {
  local spec="$1"
  IFS='|' read -r kind profile dataset tag gpu group role <<< "${spec}"
  if [[ "${kind}" == "legacy" ]]; then
    run_legacy_qa_one_dataset "${gpu}" "${dataset}" "${tag}"
  else
    run_unified_qa_one_dataset "${gpu}" "${profile}" "${dataset}" "${tag}"
  fi
  normalize_run_attempt_artifacts "${tag}"
}

# Balanced core matrix for HotpotQA/2Wiki + lighter PopQA/MuSiQue coverage.
TASKS_A=(
  "legacy|legacy_sota|hotpotqa|legacy_sota_hotpotqa|${GPU_A}|A|reference"
  "legacy|legacy_sota|2wikimultihopqa|legacy_sota_2wikimultihopqa|${GPU_A}|A|reference"
  "unified|unified_large|hotpotqa|unified_large_hotpotqa|${GPU_A}|A|baseline"
  "unified|unified_acr_rcedr_v12|hotpotqa|v12_hotpotqa|${GPU_A}|A|main"
  "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|hotpotqa|v12_beam3_no_sentence_rerank_hotpotqa|${GPU_A}|A|ablation"
  "unified|unified_acr_rcedr_v12|musique|v12_musique|${GPU_A}|A|stress"
)

TASKS_B=(
  "unified|unified_large|2wikimultihopqa|unified_large_2wiki|${GPU_B}|B|baseline"
  "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|hotpotqa|gl_ref_hotpotqa|${GPU_B}|B|reference"
  "unified|unified_acr_v11|hotpotqa|acr_v11_hotpotqa|${GPU_B}|B|cascade_ref"
  "unified|unified_acr_rcedr_v12|2wikimultihopqa|v12_2wiki|${GPU_B}|B|main"
  "unified|unified_acr_rcedr_v12_no_bridge|hotpotqa|v12_no_bridge_hotpotqa|${GPU_B}|B|ablation"
  "unified|unified_acr_rcedr_v12_beam3|popqa|v12_beam3_popqa|${GPU_B}|B|robustness"
  "unified|unified_acr_rcedr_v12_beam3|musique|v12_beam3_musique|${GPU_B}|B|stress"
)

TASKS_C=(
  "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|2wikimultihopqa|gl_ref_2wiki|${GPU_C}|C|reference"
  "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|popqa|gl_ref_popqa|${GPU_C}|C|reference"
  "unified|unified_large|popqa|unified_large_popqa|${GPU_C}|C|baseline"
  "unified|unified_acr_v11|2wikimultihopqa|acr_v11_2wiki|${GPU_C}|C|cascade_ref"
  "unified|unified_acr_rcedr_v12_answerability_only|2wikimultihopqa|v12_answerability_only_2wiki|${GPU_C}|C|ablation"
  "unified|unified_acr_rcedr_v12_no_sentence_rerank|hotpotqa|v12_no_sentence_rerank_hotpotqa|${GPU_C}|C|ablation"
  "unified|unified_acr_rcedr_v12_no_redundancy|2wikimultihopqa|v12_no_redundancy_2wiki|${GPU_C}|C|ablation"
  "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|2wikimultihopqa|v12_beam3_no_sentence_rerank_2wiki|${GPU_C}|C|ablation"
)

TASKS_D=(
  "unified|unified_large|musique|unified_large_musique|${GPU_D}|D|stress"
  "unified|unified_acr_v11_beam3|hotpotqa|acr_v11_beam3_hotpotqa|${GPU_D}|D|cascade_ref"
  "unified|unified_acr_v11_beam3|2wikimultihopqa|acr_v11_beam3_2wiki|${GPU_D}|D|cascade_ref"
  "unified|unified_acr_rcedr_v12_answerability_only|hotpotqa|v12_answerability_only_hotpotqa|${GPU_D}|D|ablation"
  "unified|unified_acr_rcedr_v12_no_redundancy|hotpotqa|v12_no_redundancy_hotpotqa|${GPU_D}|D|ablation"
  "unified|unified_acr_rcedr_v12_no_bridge|2wikimultihopqa|v12_no_bridge_2wiki|${GPU_D}|D|ablation"
  "unified|unified_acr_rcedr_v12_no_sentence_rerank|2wikimultihopqa|v12_no_sentence_rerank_2wiki|${GPU_D}|D|ablation"
  "unified|unified_acr_rcedr_v12_beam3|hotpotqa|v12_beam3_hotpotqa|${GPU_D}|D|ablation"
  "unified|unified_acr_rcedr_v12_beam3|2wikimultihopqa|v12_beam3_2wiki|${GPU_D}|D|ablation"
  "unified|unified_acr_rcedr_v12|popqa|v12_popqa|${GPU_D}|D|robustness"
  "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|popqa|v12_beam3_no_sentence_rerank_popqa|${GPU_D}|D|robustness"
  "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|musique|v12_beam3_no_sentence_rerank_musique|${GPU_D}|D|stress"
)

if [[ "${SMOKE_CORE_ONLY}" == "1" ]]; then
  # Keep all HotpotQA/2Wiki core rows for smoke reliability; trim optional PopQA/MuSiQue coverage.
  TASKS_A=(
    "legacy|legacy_sota|hotpotqa|legacy_sota_hotpotqa|${GPU_A}|A|reference"
    "legacy|legacy_sota|2wikimultihopqa|legacy_sota_2wikimultihopqa|${GPU_A}|A|reference"
    "unified|unified_large|hotpotqa|unified_large_hotpotqa|${GPU_A}|A|baseline"
    "unified|unified_acr_rcedr_v12|hotpotqa|v12_hotpotqa|${GPU_A}|A|main"
    "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|hotpotqa|v12_beam3_no_sentence_rerank_hotpotqa|${GPU_A}|A|ablation"
  )
  TASKS_B=(
    "unified|unified_large|2wikimultihopqa|unified_large_2wiki|${GPU_B}|B|baseline"
    "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|hotpotqa|gl_ref_hotpotqa|${GPU_B}|B|reference"
    "unified|unified_acr_v11|hotpotqa|acr_v11_hotpotqa|${GPU_B}|B|cascade_ref"
    "unified|unified_acr_rcedr_v12|2wikimultihopqa|v12_2wiki|${GPU_B}|B|main"
    "unified|unified_acr_rcedr_v12_no_bridge|hotpotqa|v12_no_bridge_hotpotqa|${GPU_B}|B|ablation"
  )
  TASKS_C=(
    "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|2wikimultihopqa|gl_ref_2wiki|${GPU_C}|C|reference"
    "unified|unified_acr_v11|2wikimultihopqa|acr_v11_2wiki|${GPU_C}|C|cascade_ref"
    "unified|unified_acr_rcedr_v12_answerability_only|2wikimultihopqa|v12_answerability_only_2wiki|${GPU_C}|C|ablation"
    "unified|unified_acr_rcedr_v12_no_sentence_rerank|hotpotqa|v12_no_sentence_rerank_hotpotqa|${GPU_C}|C|ablation"
    "unified|unified_acr_rcedr_v12_no_redundancy|2wikimultihopqa|v12_no_redundancy_2wiki|${GPU_C}|C|ablation"
    "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|2wikimultihopqa|v12_beam3_no_sentence_rerank_2wiki|${GPU_C}|C|ablation"
  )
  TASKS_D=(
    "unified|unified_acr_v11_beam3|hotpotqa|acr_v11_beam3_hotpotqa|${GPU_D}|D|cascade_ref"
    "unified|unified_acr_v11_beam3|2wikimultihopqa|acr_v11_beam3_2wiki|${GPU_D}|D|cascade_ref"
    "unified|unified_acr_rcedr_v12_answerability_only|hotpotqa|v12_answerability_only_hotpotqa|${GPU_D}|D|ablation"
    "unified|unified_acr_rcedr_v12_no_redundancy|hotpotqa|v12_no_redundancy_hotpotqa|${GPU_D}|D|ablation"
    "unified|unified_acr_rcedr_v12_no_bridge|2wikimultihopqa|v12_no_bridge_2wiki|${GPU_D}|D|ablation"
    "unified|unified_acr_rcedr_v12_no_sentence_rerank|2wikimultihopqa|v12_no_sentence_rerank_2wiki|${GPU_D}|D|ablation"
    "unified|unified_acr_rcedr_v12_beam3|hotpotqa|v12_beam3_hotpotqa|${GPU_D}|D|ablation"
    "unified|unified_acr_rcedr_v12_beam3|2wikimultihopqa|v12_beam3_2wiki|${GPU_D}|D|ablation"
  )
fi

if [[ "${EXTRA_POPQA}" == "1" ]]; then
  TASKS_B+=("unified|unified_acr_rcedr_v12_answerability_only|popqa|v12_answerability_only_popqa|${GPU_B}|B|optional")
  TASKS_C+=("unified|unified_acr_rcedr_v12_no_bridge|popqa|v12_no_bridge_popqa|${GPU_C}|C|optional")
  TASKS_D+=("unified|unified_acr_rcedr_v12_no_redundancy|popqa|v12_no_redundancy_popqa|${GPU_D}|D|optional")
fi

ALL_TASKS=("${TASKS_A[@]}" "${TASKS_B[@]}" "${TASKS_C[@]}" "${TASKS_D[@]}")

# Fail-fast duplicate run_name detection before any run directory cleanup.
declare -A RUN_NAME_COUNT
for spec in "${ALL_TASKS[@]}"; do
  IFS='|' read -r _kind _profile _dataset _tag _gpu _group _role <<< "${spec}"
  RUN_NAME_COUNT["${_tag}"]=$(( ${RUN_NAME_COUNT["${_tag}"]:-0} + 1 ))
done
for run_name in "${!RUN_NAME_COUNT[@]}"; do
  if [[ "${RUN_NAME_COUNT[${run_name}]}" -gt 1 ]]; then
    echo "ERROR: duplicate run_name detected: ${run_name}" >&2
    exit 1
  fi
done

# Ensure per-run directory hygiene to avoid multi-attempt residue in smoke/debug/full runs.
for spec in "${ALL_TASKS[@]}"; do
  IFS='|' read -r _kind _profile _dataset _tag _gpu _group _role <<< "${spec}"
  rm -rf "${OUT_ROOT}/qa_runs/${_tag}"
done

for spec in "${ALL_TASKS[@]}"; do
  IFS='|' read -r kind profile dataset tag gpu group role <<< "${spec}"
  register_run "${kind}" "${profile}" "${dataset}" "${tag}" "${gpu}" "${group}" "${role}"
done

export OUT_ROOT SAMPLE_SIZE SCHEDULE_JSONL
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"]).resolve()
sample_size = int(os.environ["SAMPLE_SIZE"])
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
    "phase": "phase6s_unified_acr_rcedr",
    "sample_size": sample_size,
    "out_root": str(out_root),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": runs,
}
manifest_path = out_root / "phase6s_run_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(str(manifest_path))
PY

group_a () { local spec; for spec in "${TASKS_A[@]}"; do run_task "${spec}"; done; }
group_b () { local spec; for spec in "${TASKS_B[@]}"; do run_task "${spec}"; done; }
group_c () { local spec; for spec in "${TASKS_C[@]}"; do run_task "${spec}"; done; }
group_d () { local spec; for spec in "${TASKS_D[@]}"; do run_task "${spec}"; done; }

group_a > "${OUT_ROOT}/logs/group_A_gpu0.log" 2>&1 &
PID_A=$!
group_b > "${OUT_ROOT}/logs/group_B_gpu1.log" 2>&1 &
PID_B=$!
group_c > "${OUT_ROOT}/logs/group_C_gpu1.log" 2>&1 &
PID_C=$!
group_d > "${OUT_ROOT}/logs/group_D_gpu1.log" 2>&1 &
PID_D=$!

FAIL=0
for pid in "${PID_A}" "${PID_B}" "${PID_C}" "${PID_D}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] One or more workers failed. Check ${OUT_ROOT}/logs/group_*.log" >&2
  exit 1
fi

echo "=== Build unified summary ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6s_unified_acr_rcedr_results.py \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6s.log"

echo "[Phase-6S] rag_summary count:"
find "${OUT_ROOT}" -name rag_summary.json | wc -l

echo "[Phase-6S] rag_query_results count:"
find "${OUT_ROOT}" -name rag_query_results.jsonl | wc -l

echo "[Phase-6S] root manifest:"
ls -lh "${OUT_ROOT}/phase6s_run_manifest.json" || true

echo "[Phase-6S] scheduled runs:"
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
path = os.environ.get("OUT_ROOT", "outputs/phase6s_unified_acr_rcedr_4proc") + "/phase6s_run_manifest.json"
if os.path.exists(path):
    data = json.load(open(path))
    print(len(data.get("runs", [])))
else:
    print("missing root manifest")
PY

echo "[Phase-6S] artifact consistency report (non-strict):"
PYTHONPATH=. "${PYTHON}" scripts/validate_phase6s_artifacts.py \
  --out-root "${OUT_ROOT}" \
  --json "${OUT_ROOT}/phase6s_artifact_validation.json" \
  2>&1 | tee "${OUT_ROOT}/logs/validate_phase6s_artifacts.log"

echo "[Phase-6S] artifact consistency details:"
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os

path = os.path.join(os.environ["OUT_ROOT"], "phase6s_artifact_validation.json")
if not os.path.exists(path):
    print("artifact_validation_json_missing")
    raise SystemExit(0)

data = json.load(open(path))
scheduled = sorted({str(r.get("run_name", "")).strip() for r in data.get("scheduled_runs", []) if str(r.get("run_name", "")).strip()})
completed = sorted({str(r.get("run_name", "")).strip() for r in data.get("completed_runs", []) if str(r.get("run_name", "")).strip()})
run_attempts = dict(data.get("run_attempts", {}) or {})
run_names_with_attempts = sorted(run_attempts.keys())
incomplete = sorted([name for name, info in run_attempts.items() if int(info.get("incomplete_attempt_count", 0)) > 0])
multi_attempt = sorted([name for name, info in run_attempts.items() if int(info.get("attempt_count", 0)) > 1])
extra = sorted(
    {
        str(row.get("run_name", "")).strip()
        for row in data.get("statuses", [])
        if str(row.get("status", "")).strip() == "extra_not_in_manifest"
    }
)
scheduled_not_completed = sorted(
    {
        str(row.get("run_name", "")).strip()
        for row in data.get("statuses", [])
        if str(row.get("status", "")).strip() in {"scheduled_but_missing", "scheduled_but_incomplete", "parse_error"}
    }
)

print("scheduled_run_names:", scheduled)
print("run_names_with_complete_attempts:", completed)
print("run_names_with_incomplete_attempts:", incomplete)
print("extra_run_names_not_in_manifest:", extra)
print("scheduled_run_names_with_no_complete_attempt:", scheduled_not_completed)
print("multi_attempt_run_names:", multi_attempt)
PY

if [[ "${STRICT_VALIDATE}" == "1" ]]; then
  echo "[Phase-6S] artifact consistency strict validation:"
  PYTHONPATH=. "${PYTHON}" scripts/validate_phase6s_artifacts.py \
    --out-root "${OUT_ROOT}" \
    --strict \
    2>&1 | tee "${OUT_ROOT}/logs/validate_phase6s_artifacts_strict.log"
fi

echo "=== cleanup ==="
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache

echo "=== done ==="
echo "QA runs: ${OUT_ROOT}/qa_runs"
echo "Summary: ${OUT_ROOT}/PHASE6S_UNIFIED_ACR_RCEDR_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6s_unified_acr_rcedr_summary.json"
