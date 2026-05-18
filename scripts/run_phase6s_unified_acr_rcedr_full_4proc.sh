#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6s_unified_acr_rcedr_full_n1000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
FULL_RUN="${FULL_RUN:-0}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-1}"
GPU_D="${GPU_D:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
COST_BASED_ASSIGN="${COST_BASED_ASSIGN:-1}"
COST_SOURCE_SUMMARY_JSON="${COST_SOURCE_SUMMARY_JSON:-outputs/phase6s_unified_acr_rcedr_n100/phase6s_unified_acr_rcedr_summary.json}"
COST_DEFAULT_RETRIEVAL_MS="${COST_DEFAULT_RETRIEVAL_MS:-12000}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"

if [[ -d "${OUT_ROOT}/qa_runs" ]] && [[ "${ALLOW_REUSE_OUT_ROOT:-0}" != "1" ]]; then
  echo "[ERROR] OUT_ROOT already contains qa_runs: ${OUT_ROOT}" >&2
  echo "Use a fresh OUT_ROOT or set ALLOW_REUSE_OUT_ROOT=1 explicitly." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs"

echo "=== Phase-6S full-run (fixed v12 main) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "FULL_RUN=${FULL_RUN}"
echo "GPU_A=${GPU_A} GPU_B=${GPU_B} GPU_C=${GPU_C} GPU_D=${GPU_D}"
echo "COST_BASED_ASSIGN=${COST_BASED_ASSIGN}"
echo "COST_SOURCE_SUMMARY_JSON=${COST_SOURCE_SUMMARY_JSON}"

echo "=== methodology audit gate ==="
PYTHONPATH=. "${PYTHON}" scripts/audit_phase6s_methodology.py \
  --main-profile unified_acr_rcedr_v12 \
  --check-configs configs/main_config/copy_span_instruction_unified \
  --datasets hotpotqa 2wikimultihopqa popqa musique \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/audit_phase6s_methodology.log"

TASKS_FILE="${OUT_ROOT}/_phase6s_full_tasks.txt"
SCHEDULE_JSONL="${OUT_ROOT}/_phase6s_full_scheduled_runs.jsonl"
ASSIGNED_JSONL="${OUT_ROOT}/_phase6s_full_assigned_runs.jsonl"
GROUP_A_FILE="${OUT_ROOT}/_phase6s_full_tasks_A.txt"
GROUP_B_FILE="${OUT_ROOT}/_phase6s_full_tasks_B.txt"
GROUP_C_FILE="${OUT_ROOT}/_phase6s_full_tasks_C.txt"
GROUP_D_FILE="${OUT_ROOT}/_phase6s_full_tasks_D.txt"
PLAN_JSON="${OUT_ROOT}/phase6s_full_cost_allocation_plan.json"

: > "${TASKS_FILE}"
: > "${SCHEDULE_JSONL}"
: > "${ASSIGNED_JSONL}"
: > "${GROUP_A_FILE}"
: > "${GROUP_B_FILE}"
: > "${GROUP_C_FILE}"
: > "${GROUP_D_FILE}"

# kind|profile|dataset|run_name|role
FULL_TASKS=(
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
  "unified|unified_large|popqa|unified_large_popqa|robustness"
  "unified|unified_gl_rcedr_v1_adaptive_support_span_no_bridge|popqa|gl_ref_popqa|robustness"
  "unified|unified_acr_rcedr_v12|popqa|v12_popqa|robustness"
  "unified|unified_large|musique|unified_large_musique|stress"
  "unified|unified_acr_rcedr_v12|musique|v12_musique|stress"
)

for spec in "${FULL_TASKS[@]}"; do
  echo "${spec}" >> "${TASKS_FILE}"
done

# Fail-fast duplicate run_name.
declare -A RUN_NAME_COUNT
for spec in "${FULL_TASKS[@]}"; do
  IFS='|' read -r _kind _profile _dataset _run_name _role <<< "${spec}"
  RUN_NAME_COUNT["${_run_name}"]=$(( ${RUN_NAME_COUNT["${_run_name}"]:-0} + 1 ))
done
for run_name in "${!RUN_NAME_COUNT[@]}"; do
  if [[ "${RUN_NAME_COUNT[${run_name}]}" -gt 1 ]]; then
    echo "[ERROR] duplicate run_name detected: ${run_name}" >&2
    exit 1
  fi
done

export OUT_ROOT TASKS_FILE SCHEDULE_JSONL ASSIGNED_JSONL
export GROUP_A_FILE GROUP_B_FILE GROUP_C_FILE GROUP_D_FILE PLAN_JSON
export COST_BASED_ASSIGN COST_SOURCE_SUMMARY_JSON COST_DEFAULT_RETRIEVAL_MS
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


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


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
cost_based_assign = _safe_text(os.environ.get("COST_BASED_ASSIGN", "1")) not in {"0", "false", "False"}
summary_json = Path(_safe_text(os.environ.get("COST_SOURCE_SUMMARY_JSON", ""))).resolve()
default_cost = _safe_float(os.environ.get("COST_DEFAULT_RETRIEVAL_MS", "12000"), 12000.0)
gpu_by_group = {
    "A": _safe_text(os.environ.get("GPU_A", "0")) or "0",
    "B": _safe_text(os.environ.get("GPU_B", "1")) or "1",
    "C": _safe_text(os.environ.get("GPU_C", "1")) or "1",
    "D": _safe_text(os.environ.get("GPU_D", "1")) or "1",
}

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

rows = []
if summary_json.exists():
    payload = _load_json(summary_json)
    maybe_rows = payload.get("completed_rows", [])
    if isinstance(maybe_rows, list):
        rows = [r for r in maybe_rows if isinstance(r, dict)]

cost_by_run: Dict[str, float] = {}
cost_by_pair: Dict[Tuple[str, str], List[float]] = {}
cost_by_profile: Dict[str, List[float]] = {}
cost_by_dataset: Dict[str, List[float]] = {}
all_costs: List[float] = []
for row in rows:
    run_name = _safe_text(row.get("run_name"))
    profile = _safe_text(row.get("profile"))
    dataset = _safe_text(row.get("dataset"))
    retrieval_ms = _safe_float(row.get("retrieval_ms"), 0.0)
    if retrieval_ms <= 0:
        continue
    if run_name:
        cost_by_run[run_name] = retrieval_ms
    if profile and dataset:
        cost_by_pair.setdefault((profile, dataset), []).append(retrieval_ms)
    if profile:
        cost_by_profile.setdefault(profile, []).append(retrieval_ms)
    if dataset:
        cost_by_dataset.setdefault(dataset, []).append(retrieval_ms)
    all_costs.append(retrieval_ms)

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
    if (profile, dataset) in pair_mean:
        return pair_mean[(profile, dataset)], "profile_dataset_mean"
    if profile in profile_mean:
        return profile_mean[profile], "profile_mean"
    if dataset in dataset_mean:
        return dataset_mean[dataset], "dataset_mean"
    return (global_mean if global_mean > 0 else default_cost), "default"


static_group_map = {
    "v12_no_sentence_rerank_hotpotqa": "A",
    "gl_ref_hotpotqa": "A",
    "v12_answerability_only_2wiki": "A",
    "acr_v11_beam3_2wiki": "A",
    "v12_2wiki": "A",
    "unified_large_hotpotqa": "A",
    "v12_no_redundancy_hotpotqa": "B",
    "acr_v11_beam3_hotpotqa": "B",
    "v12_popqa": "B",
    "v12_no_redundancy_2wiki": "B",
    "acr_v11_2wiki": "B",
    "legacy_sota_2wikimultihopqa": "B",
    "legacy_sota_hotpotqa": "B",
    "v12_answerability_only_hotpotqa": "C",
    "v12_hotpotqa": "C",
    "gl_ref_popqa": "C",
    "v12_no_bridge_2wiki": "C",
    "v12_no_sentence_rerank_2wiki": "C",
    "unified_large_2wiki": "C",
    "acr_v11_hotpotqa": "D",
    "v12_no_bridge_hotpotqa": "D",
    "v12_musique": "D",
    "gl_ref_2wiki": "D",
    "unified_large_musique": "D",
    "unified_large_popqa": "D",
}

scored_tasks: List[Dict[str, Any]] = []
for task in tasks:
    c, source = estimate_cost(task)
    row = dict(task)
    row["estimated_retrieval_ms"] = float(c)
    row["cost_source"] = source
    scored_tasks.append(row)

bins: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": [], "C": [], "D": []}
totals: Dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

if cost_based_assign:
    ordered = sorted(scored_tasks, key=lambda x: (-float(x["estimated_retrieval_ms"]), _safe_text(x.get("run_name"))))
    for row in ordered:
        group = sorted(totals.items(), key=lambda kv: (kv[1], kv[0]))[0][0]
        row["process_group"] = group
        row["gpu"] = gpu_by_group[group]
        bins[group].append(row)
        totals[group] += float(row["estimated_retrieval_ms"])
else:
    fallback_sorted = sorted(scored_tasks, key=lambda x: _safe_text(x.get("run_name")))
    for row in fallback_sorted:
        run_name = _safe_text(row.get("run_name"))
        group = static_group_map.get(run_name)
        if group not in {"A", "B", "C", "D"}:
            group = sorted(totals.items(), key=lambda kv: (kv[1], kv[0]))[0][0]
        row["process_group"] = group
        row["gpu"] = gpu_by_group[group]
        bins[group].append(row)
        totals[group] += float(row["estimated_retrieval_ms"])

schedule_rows: List[Dict[str, Any]] = []
for group in ["A", "B", "C", "D"]:
    group_rows = bins[group]
    lines = []
    for row in group_rows:
        lines.append(
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
    group_files[group].write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

schedule_jsonl.write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in schedule_rows) + ("\n" if schedule_rows else ""),
    encoding="utf-8",
)
assigned_jsonl.write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in schedule_rows) + ("\n" if schedule_rows else ""),
    encoding="utf-8",
)
plan_json.write_text(
    json.dumps(
        {
            "cost_based_assign": cost_based_assign,
            "cost_source_summary_json": str(summary_json),
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

print(f"[Phase-6S full planner] tasks={len(schedule_rows)}")
print(f"[Phase-6S full planner] cost_source_summary_json={summary_json}")
for g in ["A", "B", "C", "D"]:
    print(f"[Phase-6S full planner] group={g} gpu={gpu_by_group[g]} runs={len(bins[g])} est_total_ms={totals[g]:.2f}")
PY

mapfile -t TASKS_A < "${GROUP_A_FILE}"
mapfile -t TASKS_B < "${GROUP_B_FILE}"
mapfile -t TASKS_C < "${GROUP_C_FILE}"
mapfile -t TASKS_D < "${GROUP_D_FILE}"

# Root manifest is the source-of-truth for this run.
export OUT_ROOT SAMPLE_SIZE SCHEDULE_JSONL FULL_RUN
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"]).resolve()
sample_size = int(os.environ.get("SAMPLE_SIZE", "0"))
full_run = str(os.environ.get("FULL_RUN", "0")).strip() in {"1", "true", "True"}
jsonl = Path(os.environ["SCHEDULE_JSONL"]).resolve()
runs = []
if jsonl.exists():
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            runs.append(payload)

manifest = {
    "phase": "phase6s_unified_acr_rcedr_full",
    "sample_size": sample_size,
    "full_run": full_run,
    "out_root": str(out_root),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": runs,
}
path = out_root / "phase6s_run_manifest.json"
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(str(path))
PY

run_unified_qa_one_dataset () {
  local gpu="$1"
  local profile="$2"
  local dataset="$3"
  local run_name="$4"
  echo "=== [GPU ${gpu}] profile=${profile} dataset=${dataset} run=${run_name} ==="
  rm -rf "${OUT_ROOT}/qa_runs/${run_name}"
  mkdir -p "${OUT_ROOT}/qa_runs/${run_name}"

  local sample_args=()
  if [[ "${FULL_RUN}" != "1" ]]; then
    sample_args=(--sample-size "${SAMPLE_SIZE}")
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_unified_config_4ds.py \
    --profile "${profile}" \
    --datasets "${dataset}" \
    "${sample_args[@]}" \
    --output-root "${OUT_ROOT}/qa_runs/${run_name}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${run_name}.log"
}

run_legacy_qa_one_dataset () {
  local gpu="$1"
  local dataset="$2"
  local run_name="$3"
  echo "=== [GPU ${gpu}] legacy_sota dataset=${dataset} run=${run_name} ==="
  rm -rf "${OUT_ROOT}/qa_runs/${run_name}"
  mkdir -p "${OUT_ROOT}/qa_runs/${run_name}"

  local sample_args=()
  if [[ "${FULL_RUN}" != "1" ]]; then
    sample_args=(--sample-size "${SAMPLE_SIZE}")
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" \
    scripts/run_sota_config_4ds.py \
    --mode locked_precomputed \
    --datasets "${dataset}" \
    "${sample_args[@]}" \
    --output-root "${OUT_ROOT}/qa_runs/${run_name}" \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${run_name}.log"
}

run_task () {
  local spec="$1"
  IFS='|' read -r kind profile dataset run_name gpu group role <<< "${spec}"
  if [[ "${kind}" == "legacy" ]]; then
    run_legacy_qa_one_dataset "${gpu}" "${dataset}" "${run_name}"
  else
    run_unified_qa_one_dataset "${gpu}" "${profile}" "${dataset}" "${run_name}"
  fi
}

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

echo "=== summarize ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6s_unified_acr_rcedr_results.py \
  --out-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6s.log"

echo "=== strict artifact validation ==="
PYTHONPATH=. "${PYTHON}" scripts/validate_phase6s_artifacts.py \
  --out-root "${OUT_ROOT}" \
  --strict-clean \
  --json "${OUT_ROOT}/phase6s_artifact_validation.json" \
  2>&1 | tee "${OUT_ROOT}/logs/validate_phase6s_artifacts_strict.log"

echo "[Phase-6S full] scheduled runs:"
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["OUT_ROOT"]) / "phase6s_run_manifest.json"
if path.exists():
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(len(payload.get("runs", [])))
else:
    print("missing root manifest")
PY

echo "[Phase-6S full] rag_summary count:"
find "${OUT_ROOT}" -name rag_summary.json | wc -l

echo "[Phase-6S full] rag_query_results count:"
find "${OUT_ROOT}" -name rag_query_results.jsonl | wc -l

echo "=== done ==="
echo "Methodology audit: ${OUT_ROOT}/PHASE6S_METHODOLOGY_AUDIT.md"
echo "Summary: ${OUT_ROOT}/PHASE6S_UNIFIED_ACR_RCEDR_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6s_unified_acr_rcedr_summary.json"
