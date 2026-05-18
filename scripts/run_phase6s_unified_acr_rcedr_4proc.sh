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
COST_BASED_ASSIGN="${COST_BASED_ASSIGN:-1}"
COST_SOURCE_ROOT="${COST_SOURCE_ROOT:-outputs/phase6s_debug20}"
COST_SOURCE_SUMMARY_JSON="${COST_SOURCE_SUMMARY_JSON:-}"
COST_DEFAULT_RETRIEVAL_MS="${COST_DEFAULT_RETRIEVAL_MS:-12000}"
PLAN_ONLY="${PLAN_ONLY:-0}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"

if [[ -d "${OUT_ROOT}/qa_runs" ]] && [[ "${ALLOW_REUSE_OUT_ROOT:-0}" != "1" ]]; then
  echo "[ERROR] OUT_ROOT already contains qa_runs: ${OUT_ROOT}" >&2
  echo "Use a fresh OUT_ROOT or set ALLOW_REUSE_OUT_ROOT=1 explicitly." >&2
  exit 1
fi

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
echo "COST_BASED_ASSIGN=${COST_BASED_ASSIGN}"
echo "COST_SOURCE_ROOT=${COST_SOURCE_ROOT}"
echo "COST_SOURCE_SUMMARY_JSON=${COST_SOURCE_SUMMARY_JSON:-<auto>}"
echo "COST_DEFAULT_RETRIEVAL_MS=${COST_DEFAULT_RETRIEVAL_MS}"
echo "PLAN_ONLY=${PLAN_ONLY}"

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
  local run_dir="${OUT_ROOT}/qa_runs/${tag}"
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}"
  if [[ "${kind}" == "legacy" ]]; then
    run_legacy_qa_one_dataset "${gpu}" "${dataset}" "${tag}"
  else
    run_unified_qa_one_dataset "${gpu}" "${profile}" "${dataset}" "${tag}"
  fi
  normalize_run_attempt_artifacts "${tag}"
}

# Base task set: kind|profile|dataset|run_name|role
BASE_TASKS=(
  "legacy|legacy_sota|hotpotqa|legacy_sota_hotpotqa|reference"
  "legacy|legacy_sota|2wikimultihopqa|legacy_sota_2wikimultihopqa|reference"
  "unified|unified_large|hotpotqa|unified_large_hotpotqa|baseline"
  "unified|unified_large|2wikimultihopqa|unified_large_2wiki|baseline"
  "unified|unified_large|popqa|unified_large_popqa|baseline"
  "unified|unified_large|musique|unified_large_musique|stress"
  "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|hotpotqa|gl_ref_hotpotqa|reference"
  "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|2wikimultihopqa|gl_ref_2wiki|reference"
  "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|popqa|gl_ref_popqa|reference"
  "unified|unified_acr_v11|hotpotqa|acr_v11_hotpotqa|cascade_ref"
  "unified|unified_acr_v11|2wikimultihopqa|acr_v11_2wiki|cascade_ref"
  "unified|unified_acr_v11_beam3|hotpotqa|acr_v11_beam3_hotpotqa|cascade_ref"
  "unified|unified_acr_v11_beam3|2wikimultihopqa|acr_v11_beam3_2wiki|cascade_ref"
  "unified|unified_acr_rcedr_v12|hotpotqa|v12_hotpotqa|main"
  "unified|unified_acr_rcedr_v12|2wikimultihopqa|v12_2wiki|main"
  "unified|unified_acr_rcedr_v12|popqa|v12_popqa|robustness"
  "unified|unified_acr_rcedr_v12|musique|v12_musique|stress"
  "unified|unified_acr_rcedr_v12_answerability_only|hotpotqa|v12_answerability_only_hotpotqa|ablation"
  "unified|unified_acr_rcedr_v12_answerability_only|2wikimultihopqa|v12_answerability_only_2wiki|ablation"
  "unified|unified_acr_rcedr_v12_no_bridge|hotpotqa|v12_no_bridge_hotpotqa|ablation"
  "unified|unified_acr_rcedr_v12_no_bridge|2wikimultihopqa|v12_no_bridge_2wiki|ablation"
  "unified|unified_acr_rcedr_v12_no_redundancy|hotpotqa|v12_no_redundancy_hotpotqa|ablation"
  "unified|unified_acr_rcedr_v12_no_redundancy|2wikimultihopqa|v12_no_redundancy_2wiki|ablation"
  "unified|unified_acr_rcedr_v12_no_sentence_rerank|hotpotqa|v12_no_sentence_rerank_hotpotqa|ablation"
  "unified|unified_acr_rcedr_v12_no_sentence_rerank|2wikimultihopqa|v12_no_sentence_rerank_2wiki|ablation"
  "unified|unified_acr_rcedr_v12_beam3|hotpotqa|v12_beam3_hotpotqa|ablation"
  "unified|unified_acr_rcedr_v12_beam3|2wikimultihopqa|v12_beam3_2wiki|ablation"
  "unified|unified_acr_rcedr_v12_beam3|popqa|v12_beam3_popqa|robustness"
  "unified|unified_acr_rcedr_v12_beam3|musique|v12_beam3_musique|stress"
  "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|hotpotqa|v12_beam3_no_sentence_rerank_hotpotqa|ablation"
  "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|2wikimultihopqa|v12_beam3_no_sentence_rerank_2wiki|ablation"
  "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|popqa|v12_beam3_no_sentence_rerank_popqa|robustness"
  "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|musique|v12_beam3_no_sentence_rerank_musique|stress"
)

if [[ "${SMOKE_CORE_ONLY}" == "1" ]]; then
  # Keep all HotpotQA/2Wiki core rows for smoke reliability; trim optional PopQA/MuSiQue coverage.
  BASE_TASKS=(
    "legacy|legacy_sota|hotpotqa|legacy_sota_hotpotqa|reference"
    "legacy|legacy_sota|2wikimultihopqa|legacy_sota_2wikimultihopqa|reference"
    "unified|unified_large|hotpotqa|unified_large_hotpotqa|baseline"
    "unified|unified_large|2wikimultihopqa|unified_large_2wiki|baseline"
    "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|hotpotqa|gl_ref_hotpotqa|reference"
    "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|2wikimultihopqa|gl_ref_2wiki|reference"
    "unified|unified_acr_v11|hotpotqa|acr_v11_hotpotqa|cascade_ref"
    "unified|unified_acr_v11|2wikimultihopqa|acr_v11_2wiki|cascade_ref"
    "unified|unified_acr_v11_beam3|hotpotqa|acr_v11_beam3_hotpotqa|cascade_ref"
    "unified|unified_acr_v11_beam3|2wikimultihopqa|acr_v11_beam3_2wiki|cascade_ref"
    "unified|unified_acr_rcedr_v12|hotpotqa|v12_hotpotqa|main"
    "unified|unified_acr_rcedr_v12|2wikimultihopqa|v12_2wiki|main"
    "unified|unified_acr_rcedr_v12_answerability_only|hotpotqa|v12_answerability_only_hotpotqa|ablation"
    "unified|unified_acr_rcedr_v12_answerability_only|2wikimultihopqa|v12_answerability_only_2wiki|ablation"
    "unified|unified_acr_rcedr_v12_no_bridge|hotpotqa|v12_no_bridge_hotpotqa|ablation"
    "unified|unified_acr_rcedr_v12_no_bridge|2wikimultihopqa|v12_no_bridge_2wiki|ablation"
    "unified|unified_acr_rcedr_v12_no_redundancy|hotpotqa|v12_no_redundancy_hotpotqa|ablation"
    "unified|unified_acr_rcedr_v12_no_redundancy|2wikimultihopqa|v12_no_redundancy_2wiki|ablation"
    "unified|unified_acr_rcedr_v12_no_sentence_rerank|hotpotqa|v12_no_sentence_rerank_hotpotqa|ablation"
    "unified|unified_acr_rcedr_v12_no_sentence_rerank|2wikimultihopqa|v12_no_sentence_rerank_2wiki|ablation"
    "unified|unified_acr_rcedr_v12_beam3|hotpotqa|v12_beam3_hotpotqa|ablation"
    "unified|unified_acr_rcedr_v12_beam3|2wikimultihopqa|v12_beam3_2wiki|ablation"
    "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|hotpotqa|v12_beam3_no_sentence_rerank_hotpotqa|ablation"
    "unified|unified_acr_rcedr_v12_beam3_no_sentence_rerank|2wikimultihopqa|v12_beam3_no_sentence_rerank_2wiki|ablation"
  )
fi

if [[ "${EXTRA_POPQA}" == "1" ]]; then
  BASE_TASKS+=("unified|unified_acr_rcedr_v12_answerability_only|popqa|v12_answerability_only_popqa|optional")
  BASE_TASKS+=("unified|unified_acr_rcedr_v12_no_bridge|popqa|v12_no_bridge_popqa|optional")
  BASE_TASKS+=("unified|unified_acr_rcedr_v12_no_redundancy|popqa|v12_no_redundancy_popqa|optional")
fi

ALL_TASKS=("${BASE_TASKS[@]}")

# Fail-fast duplicate run_name detection before any run directory cleanup.
declare -A RUN_NAME_COUNT
for spec in "${ALL_TASKS[@]}"; do
  IFS='|' read -r _kind _profile _dataset _tag _role <<< "${spec}"
  RUN_NAME_COUNT["${_tag}"]=$(( ${RUN_NAME_COUNT["${_tag}"]:-0} + 1 ))
done
for run_name in "${!RUN_NAME_COUNT[@]}"; do
  if [[ "${RUN_NAME_COUNT[${run_name}]}" -gt 1 ]]; then
    echo "ERROR: duplicate run_name detected: ${run_name}" >&2
    exit 1
  fi
done

TASKS_FILE="${OUT_ROOT}/_phase6s_tasks.txt"
ASSIGNED_JSONL="${OUT_ROOT}/_phase6s_assigned_runs.jsonl"
GROUP_A_FILE="${OUT_ROOT}/_phase6s_tasks_A.txt"
GROUP_B_FILE="${OUT_ROOT}/_phase6s_tasks_B.txt"
GROUP_C_FILE="${OUT_ROOT}/_phase6s_tasks_C.txt"
GROUP_D_FILE="${OUT_ROOT}/_phase6s_tasks_D.txt"
PLAN_JSON="${OUT_ROOT}/phase6s_cost_allocation_plan.json"

: > "${TASKS_FILE}"
: > "${SCHEDULE_JSONL}"
: > "${ASSIGNED_JSONL}"
: > "${GROUP_A_FILE}"
: > "${GROUP_B_FILE}"
: > "${GROUP_C_FILE}"
: > "${GROUP_D_FILE}"

for spec in "${ALL_TASKS[@]}"; do
  echo "${spec}" >> "${TASKS_FILE}"
done

export OUT_ROOT SAMPLE_SIZE SCHEDULE_JSONL ASSIGNED_JSONL TASKS_FILE
export GROUP_A_FILE GROUP_B_FILE GROUP_C_FILE GROUP_D_FILE PLAN_JSON
export COST_BASED_ASSIGN COST_SOURCE_ROOT COST_SOURCE_SUMMARY_JSON COST_DEFAULT_RETRIEVAL_MS
export GPU_A GPU_B GPU_C GPU_D
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

def _safe_text(value: Any) -> str:
    return str(value or "").strip()

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return float(default)
    if x < 0:
        return float(default)
    return x

def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}

out_root = Path(os.environ["OUT_ROOT"]).resolve()
tasks_file = Path(os.environ["TASKS_FILE"]).resolve()
schedule_jsonl = Path(os.environ["SCHEDULE_JSONL"]).resolve()
assigned_jsonl = Path(os.environ["ASSIGNED_JSONL"]).resolve()
group_files = {
    "A": Path(os.environ["GROUP_A_FILE"]).resolve(),
    "B": Path(os.environ["GROUP_B_FILE"]).resolve(),
    "C": Path(os.environ["GROUP_C_FILE"]).resolve(),
    "D": Path(os.environ["GROUP_D_FILE"]).resolve(),
}
plan_json = Path(os.environ["PLAN_JSON"]).resolve()

gpu_by_group = {
    "A": _safe_text(os.environ.get("GPU_A", "0")) or "0",
    "B": _safe_text(os.environ.get("GPU_B", "1")) or "1",
    "C": _safe_text(os.environ.get("GPU_C", "1")) or "1",
    "D": _safe_text(os.environ.get("GPU_D", "1")) or "1",
}
cost_based_assign = _safe_text(os.environ.get("COST_BASED_ASSIGN", "1")) not in {"0", "false", "False"}
cost_source_root = Path(_safe_text(os.environ.get("COST_SOURCE_ROOT", "outputs/phase6s_debug20")) or "outputs/phase6s_debug20")
summary_json_env = _safe_text(os.environ.get("COST_SOURCE_SUMMARY_JSON", ""))
if summary_json_env:
    summary_path = Path(summary_json_env).resolve()
else:
    summary_path = (cost_source_root / "phase6s_unified_acr_rcedr_summary.json").resolve()
default_cost = _safe_float(os.environ.get("COST_DEFAULT_RETRIEVAL_MS", "12000"), 12000.0)

tasks: List[Dict[str, Any]] = []
for line in tasks_file.read_text(encoding="utf-8").splitlines():
    line = _safe_text(line)
    if not line:
        continue
    parts = line.split("|")
    if len(parts) != 5:
        raise SystemExit(f"invalid_task_spec:{line}")
    kind, profile, dataset, run_name, role = [p.strip() for p in parts]
    tasks.append(
        {
            "kind": kind,
            "profile": profile,
            "dataset": dataset,
            "run_name": run_name,
            "role": role,
        }
    )

# Cost priors from n=20 summary.
cost_by_run: Dict[str, float] = {}
cost_by_pair: Dict[Tuple[str, str], List[float]] = {}
cost_by_profile: Dict[str, List[float]] = {}
cost_by_dataset: Dict[str, List[float]] = {}
all_costs: List[float] = []

if summary_path.exists():
    payload = _load_json(summary_path)
    rows = payload.get("completed_rows", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            run_name = _safe_text(row.get("run_name"))
            profile = _safe_text(row.get("profile"))
            dataset = _safe_text(row.get("dataset"))
            c = _safe_float(row.get("retrieval_ms"), 0.0)
            if c <= 0:
                continue
            if run_name:
                cost_by_run[run_name] = c
            if profile and dataset:
                cost_by_pair.setdefault((profile, dataset), []).append(c)
            if profile:
                cost_by_profile.setdefault(profile, []).append(c)
            if dataset:
                cost_by_dataset.setdefault(dataset, []).append(c)
            all_costs.append(c)

def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

pair_mean = {k: _mean(v) for k, v in cost_by_pair.items() if v}
profile_mean = {k: _mean(v) for k, v in cost_by_profile.items() if v}
dataset_mean = {k: _mean(v) for k, v in cost_by_dataset.items() if v}
global_mean = _mean(all_costs) if all_costs else default_cost

def estimate_cost(task: Dict[str, Any]) -> Tuple[float, str]:
    run_name = _safe_text(task.get("run_name"))
    profile = _safe_text(task.get("profile"))
    dataset = _safe_text(task.get("dataset"))
    if run_name in cost_by_run:
        return cost_by_run[run_name], "run_name"
    key = (profile, dataset)
    if key in pair_mean:
        return pair_mean[key], "profile_dataset_mean"
    if profile in profile_mean:
        return profile_mean[profile], "profile_mean"
    if dataset in dataset_mean:
        return dataset_mean[dataset], "dataset_mean"
    return (global_mean if global_mean > 0 else default_cost), "default"

scored_tasks: List[Dict[str, Any]] = []
for task in tasks:
    c, source = estimate_cost(task)
    row = dict(task)
    row["estimated_retrieval_ms"] = float(c)
    row["cost_source"] = source
    scored_tasks.append(row)

if cost_based_assign:
    ordered = sorted(scored_tasks, key=lambda x: (-float(x["estimated_retrieval_ms"]), _safe_text(x.get("run_name"))))
else:
    ordered = list(scored_tasks)

bins: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": [], "C": [], "D": []}
totals: Dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

def _pick_group() -> str:
    return sorted(totals.items(), key=lambda kv: (kv[1], kv[0]))[0][0]

for row in ordered:
    group = _pick_group()
    row["process_group"] = group
    row["gpu"] = gpu_by_group[group]
    bins[group].append(row)
    totals[group] += float(row["estimated_retrieval_ms"])

for path in group_files.values():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
for group, rows in bins.items():
    out_lines = []
    for row in rows:
        out_lines.append(
            "|".join(
                [
                    _safe_text(row.get("kind")),
                    _safe_text(row.get("profile")),
                    _safe_text(row.get("dataset")),
                    _safe_text(row.get("run_name")),
                    _safe_text(row.get("gpu")),
                    _safe_text(row.get("process_group")),
                    _safe_text(row.get("role")),
                ]
            )
        )
    if out_lines:
        group_files[group].write_text("\n".join(out_lines) + "\n", encoding="utf-8")

schedule_rows: List[Dict[str, Any]] = []
for group in ["A", "B", "C", "D"]:
    for row in bins[group]:
        schedule_rows.append(
            {
                "run_name": _safe_text(row.get("run_name")),
                "dataset": _safe_text(row.get("dataset")),
                "profile": _safe_text(row.get("profile")),
                "gpu": _safe_text(row.get("gpu")),
                "process_group": _safe_text(row.get("process_group")),
                "role": _safe_text(row.get("role")),
                "kind": _safe_text(row.get("kind")),
                "scheduled": True,
                "estimated_retrieval_ms": float(row.get("estimated_retrieval_ms", 0.0)),
                "cost_source": _safe_text(row.get("cost_source")),
            }
        )

schedule_jsonl.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in schedule_rows) + ("\n" if schedule_rows else ""),
    encoding="utf-8",
)
assigned_jsonl.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in schedule_rows) + ("\n" if schedule_rows else ""),
    encoding="utf-8",
)
plan_json.write_text(
    json.dumps(
        {
            "cost_based_assign": cost_based_assign,
            "cost_source_root": str(cost_source_root.resolve()) if cost_source_root.exists() else str(cost_source_root),
            "cost_source_summary_json": str(summary_path),
            "default_cost_ms": float(default_cost),
            "global_mean_cost_ms": float(global_mean),
            "group_totals_ms": totals,
            "runs": schedule_rows,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(f"[Phase-6S planner] tasks={len(schedule_rows)} cost_based_assign={cost_based_assign}")
print(f"[Phase-6S planner] cost_source_summary_json={summary_path}")
for group in ["A", "B", "C", "D"]:
    print(f"[Phase-6S planner] group={group} gpu={gpu_by_group[group]} runs={len(bins[group])} est_total_ms={totals[group]:.2f}")
PY

mapfile -t TASKS_A < "${GROUP_A_FILE}"
mapfile -t TASKS_B < "${GROUP_B_FILE}"
mapfile -t TASKS_C < "${GROUP_C_FILE}"
mapfile -t TASKS_D < "${GROUP_D_FILE}"

# Ensure per-run directory hygiene to avoid multi-attempt residue in smoke/debug/full runs.
for spec in "${TASKS_A[@]}" "${TASKS_B[@]}" "${TASKS_C[@]}" "${TASKS_D[@]}"; do
  IFS='|' read -r _kind _profile _dataset _tag _gpu _group _role <<< "${spec}"
  rm -rf "${OUT_ROOT}/qa_runs/${_tag}"
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

if [[ "${PLAN_ONLY}" == "1" ]]; then
  echo "[Phase-6S] PLAN_ONLY=1 -> wrote schedule/manifest without launching workers."
  echo "[Phase-6S] manifest: ${OUT_ROOT}/phase6s_run_manifest.json"
  echo "[Phase-6S] allocation plan: ${PLAN_JSON}"
  exit 0
fi

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
    --strict-clean \
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
