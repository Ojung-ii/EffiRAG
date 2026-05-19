#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6x_early_collapse_recovery_n100}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
PROCESS_COUNT="${PROCESS_COUNT:-2}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"

BASELINE_PROFILE="${BASELINE_PROFILE:-unified_acr_rcedr_v12}"
VARIANT_PROFILE="${VARIANT_PROFILE:-unified_acr_rcedr_v12_early_collapse_recovery}"
PROFILE_ROOT="${PROFILE_ROOT:-configs/main_config/copy_span_instruction_unified}"

if [[ "${PROCESS_COUNT}" != "2" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 2, got: ${PROCESS_COUNT}" >&2
  exit 1
fi

if [[ -d "${OUT_ROOT}/qa_runs" ]] && [[ "${ALLOW_REUSE_OUT_ROOT:-0}" != "1" ]]; then
  echo "[ERROR] OUT_ROOT already contains qa_runs: ${OUT_ROOT}" >&2
  echo "Use a fresh OUT_ROOT or set ALLOW_REUSE_OUT_ROOT=1 explicitly." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/qa_runs" "${OUT_ROOT}/configs" "${OUT_ROOT}/shared_sample_ids"

BASE_HOTPOT_CFG="${PROFILE_ROOT}/${BASELINE_PROFILE}/hotpotqa.yaml"
BASE_2WIKI_CFG="${PROFILE_ROOT}/${BASELINE_PROFILE}/2wikimultihopqa.yaml"
VAR_HOTPOT_CFG="${PROFILE_ROOT}/${VARIANT_PROFILE}/hotpotqa.yaml"
VAR_2WIKI_CFG="${PROFILE_ROOT}/${VARIANT_PROFILE}/2wikimultihopqa.yaml"

for p in "${BASE_HOTPOT_CFG}" "${BASE_2WIKI_CFG}" "${VAR_HOTPOT_CFG}" "${VAR_2WIKI_CFG}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] missing config: ${p}" >&2
    exit 1
  fi
done

export OUT_ROOT SAMPLE_SIZE BASE_HOTPOT_CFG BASE_2WIKI_CFG VAR_HOTPOT_CFG VAR_2WIKI_CFG PYTHON
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

out_root = Path(Path(__import__("os").environ["OUT_ROOT"]).resolve())
sample_size = int(__import__("os").environ["SAMPLE_SIZE"])

cfg_paths = {
    "baseline_hotpotqa": Path(__import__("os").environ["BASE_HOTPOT_CFG"]).resolve(),
    "baseline_2wikimultihopqa": Path(__import__("os").environ["BASE_2WIKI_CFG"]).resolve(),
    "variant_hotpotqa": Path(__import__("os").environ["VAR_HOTPOT_CFG"]).resolve(),
    "variant_2wikimultihopqa": Path(__import__("os").environ["VAR_2WIKI_CFG"]).resolve(),
}

def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def write_yaml(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

def sample_id(record):
    return str(record.get("_id") or record.get("id") or record.get("qid") or "").strip()

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

base_hotpot = load_yaml(cfg_paths["baseline_hotpotqa"])
base_2wiki = load_yaml(cfg_paths["baseline_2wikimultihopqa"])
var_hotpot = load_yaml(cfg_paths["variant_hotpotqa"])
var_2wiki = load_yaml(cfg_paths["variant_2wikimultihopqa"])

datasets = {
    "hotpotqa": base_hotpot.get("data_path", ""),
    "2wikimultihopqa": base_2wiki.get("data_path", ""),
}

subset_root = out_root / "shared_sample_ids" / "subset_data"
subset_root.mkdir(parents=True, exist_ok=True)
shared_ids = {}
subset_paths = {}
for dataset, data_path in datasets.items():
    src = Path(str(data_path)).expanduser().resolve()
    records = json.loads(src.read_text(encoding="utf-8"))
    sampled = []
    ids = []
    seen = set()
    for row in records:
        if not isinstance(row, dict):
            continue
        sid = sample_id(row)
        if not sid or sid in seen:
            continue
        seen.add(sid)
        sampled.append(row)
        ids.append(sid)
        if len(ids) >= sample_size:
            break
    if len(ids) < sample_size:
        raise RuntimeError(f"insufficient sample size for {dataset}: requested={sample_size}, got={len(ids)}")
    id_path = out_root / "shared_sample_ids" / f"{dataset}.json"
    subset_path = subset_root / f"{dataset}.json"
    write_json(id_path, ids)
    write_json(subset_path, sampled)
    shared_ids[dataset] = str(id_path.resolve())
    subset_paths[dataset] = str(subset_path.resolve())

def patch_cfg(cfg, dataset):
    out = dict(cfg or {})
    out["data_path"] = subset_paths[dataset]
    out["limit"] = int(sample_size)
    out["timestamp_output"] = True
    return out

patched = {
    "phase6x_hotpotqa_v12": patch_cfg(base_hotpot, "hotpotqa"),
    "phase6x_hotpotqa_v12_early_collapse_recovery": patch_cfg(var_hotpot, "hotpotqa"),
    "phase6x_2wikimultihopqa_v12": patch_cfg(base_2wiki, "2wikimultihopqa"),
    "phase6x_2wikimultihopqa_v12_early_collapse_recovery": patch_cfg(var_2wiki, "2wikimultihopqa"),
}

patched_paths = {}
for run_name, cfg in patched.items():
    p = out_root / "configs" / f"{run_name}.yaml"
    write_yaml(p, cfg)
    patched_paths[run_name] = str(p.resolve())

manifest = {
    "phase": "phase6x_early_collapse_recovery",
    "sample_size": int(sample_size),
    "out_root": str(out_root),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "shared_sample_ids": shared_ids,
    "runs": [
        {
            "run_name": "phase6x_hotpotqa_v12",
            "dataset": "hotpotqa",
            "profile": "unified_acr_rcedr_v12",
            "role": "baseline",
            "gpu": "0",
            "process_group": "A",
            "patched_config_path": patched_paths["phase6x_hotpotqa_v12"],
            "scheduled": True,
        },
        {
            "run_name": "phase6x_hotpotqa_v12_early_collapse_recovery",
            "dataset": "hotpotqa",
            "profile": "unified_acr_rcedr_v12_early_collapse_recovery",
            "role": "variant",
            "gpu": "0",
            "process_group": "A",
            "patched_config_path": patched_paths["phase6x_hotpotqa_v12_early_collapse_recovery"],
            "scheduled": True,
        },
        {
            "run_name": "phase6x_2wikimultihopqa_v12",
            "dataset": "2wikimultihopqa",
            "profile": "unified_acr_rcedr_v12",
            "role": "baseline",
            "gpu": "1",
            "process_group": "B",
            "patched_config_path": patched_paths["phase6x_2wikimultihopqa_v12"],
            "scheduled": True,
        },
        {
            "run_name": "phase6x_2wikimultihopqa_v12_early_collapse_recovery",
            "dataset": "2wikimultihopqa",
            "profile": "unified_acr_rcedr_v12_early_collapse_recovery",
            "role": "variant",
            "gpu": "1",
            "process_group": "B",
            "patched_config_path": patched_paths["phase6x_2wikimultihopqa_v12_early_collapse_recovery"],
            "scheduled": True,
        },
    ],
}
for name in ("phase6x_run_manifest.json", "phase6s_run_manifest.json"):
    write_json(out_root / name, manifest)
PY

run_one() {
  local gpu="$1"
  local run_name="$2"
  local cfg_path="$3"
  local run_root="${OUT_ROOT}/qa_runs/${run_name}"
  rm -rf "${run_root}"
  mkdir -p "${run_root}"
  echo "=== [GPU ${gpu}] START ${run_name} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
    --config "${cfg_path}" \
    --output-dir "${run_root}" \
    --timestamp-output true \
    2>&1 | tee "${OUT_ROOT}/logs/qa_${run_name}.log"
  echo "=== [GPU ${gpu}] DONE ${run_name} ==="
}

group_a() {
  run_one "${GPU_A}" "phase6x_hotpotqa_v12" "${OUT_ROOT}/configs/phase6x_hotpotqa_v12.yaml"
  run_one "${GPU_A}" "phase6x_hotpotqa_v12_early_collapse_recovery" "${OUT_ROOT}/configs/phase6x_hotpotqa_v12_early_collapse_recovery.yaml"
}

group_b() {
  run_one "${GPU_B}" "phase6x_2wikimultihopqa_v12" "${OUT_ROOT}/configs/phase6x_2wikimultihopqa_v12.yaml"
  run_one "${GPU_B}" "phase6x_2wikimultihopqa_v12_early_collapse_recovery" "${OUT_ROOT}/configs/phase6x_2wikimultihopqa_v12_early_collapse_recovery.yaml"
}

group_a > "${OUT_ROOT}/logs/group_A_gpu${GPU_A}.log" 2>&1 &
PID_A=$!
group_b > "${OUT_ROOT}/logs/group_B_gpu${GPU_B}.log" 2>&1 &
PID_B=$!

FAIL=0
if ! wait "${PID_A}"; then FAIL=1; fi
if ! wait "${PID_B}"; then FAIL=1; fi
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] one or more process groups failed." >&2
  exit 1
fi

echo "=== summarize PHASE6X early-collapse recovery ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6x_early_collapse_recovery.py \
  --out-root "${OUT_ROOT}" \
  --datasets hotpotqa,2wikimultihopqa \
  --baseline-profile "${BASELINE_PROFILE}" \
  --variant-profile "${VARIANT_PROFILE}" \
  --top-k 20 \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6x_early_collapse_recovery.log"

VALIDATE_ARGS=()
if [[ "${STRICT_VALIDATE}" == "1" ]]; then
  VALIDATE_ARGS+=(--strict-clean)
fi

echo "=== validate artifacts ==="
PYTHONPATH=. "${PYTHON}" scripts/validate_phase6s_artifacts.py \
  --out-root "${OUT_ROOT}" \
  "${VALIDATE_ARGS[@]}" \
  --json "${OUT_ROOT}/phase6x_artifact_validation.json" \
  2>&1 | tee "${OUT_ROOT}/logs/validate_phase6x_artifacts.log"

echo "=== done ==="
echo "Summary: ${OUT_ROOT}/PHASE6X_EARLY_COLLAPSE_RECOVERY_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6x_early_collapse_recovery_summary.json"
