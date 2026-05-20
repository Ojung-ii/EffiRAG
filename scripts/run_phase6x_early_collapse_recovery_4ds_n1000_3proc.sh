#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

OUT_ROOT="${OUT_ROOT:-outputs/phase6x_early_collapse_recovery_4ds_n1000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
PROCESS_COUNT="${PROCESS_COUNT:-3}"
GPU_A="${GPU_A:-0}"  # hotpotqa + 2wikimultihopqa
GPU_B="${GPU_B:-1}"  # musique
GPU_C="${GPU_C:-1}"  # popqa
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"

BASELINE_PROFILE="${BASELINE_PROFILE:-unified_acr_rcedr_v12}"
VARIANT_PROFILE="${VARIANT_PROFILE:-unified_acr_rcedr_v12_early_collapse_recovery}"
PROFILE_ROOT="${PROFILE_ROOT:-configs/main_config/copy_span_instruction_unified}"

if [[ "${PROCESS_COUNT}" != "3" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 3, got: ${PROCESS_COUNT}" >&2
  exit 1
fi

if [[ -d "${OUT_ROOT}/qa_runs" ]] && [[ "${ALLOW_REUSE_OUT_ROOT:-0}" != "1" ]]; then
  echo "[ERROR] OUT_ROOT already contains qa_runs: ${OUT_ROOT}" >&2
  echo "Use a fresh OUT_ROOT or set ALLOW_REUSE_OUT_ROOT=1 explicitly." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/qa_runs" "${OUT_ROOT}/configs" "${OUT_ROOT}/sampled_query_ids"

BASE_HOTPOT_CFG="${PROFILE_ROOT}/${BASELINE_PROFILE}/hotpotqa.yaml"
BASE_2WIKI_CFG="${PROFILE_ROOT}/${BASELINE_PROFILE}/2wikimultihopqa.yaml"
BASE_MUSIQUE_CFG="${PROFILE_ROOT}/${BASELINE_PROFILE}/musique.yaml"
BASE_POPQA_CFG="${PROFILE_ROOT}/${BASELINE_PROFILE}/popqa.yaml"
VAR_HOTPOT_CFG="${PROFILE_ROOT}/${VARIANT_PROFILE}/hotpotqa.yaml"
VAR_2WIKI_CFG="${PROFILE_ROOT}/${VARIANT_PROFILE}/2wikimultihopqa.yaml"
VAR_MUSIQUE_CFG="${PROFILE_ROOT}/${VARIANT_PROFILE}/musique.yaml"
VAR_POPQA_CFG="${PROFILE_ROOT}/${VARIANT_PROFILE}/popqa.yaml"

for p in \
  "${BASE_HOTPOT_CFG}" "${BASE_2WIKI_CFG}" "${BASE_MUSIQUE_CFG}" "${BASE_POPQA_CFG}" \
  "${VAR_HOTPOT_CFG}" "${VAR_2WIKI_CFG}" "${VAR_MUSIQUE_CFG}" "${VAR_POPQA_CFG}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] missing config: ${p}" >&2
    exit 1
  fi
done

export OUT_ROOT SAMPLE_SIZE \
  BASE_HOTPOT_CFG BASE_2WIKI_CFG BASE_MUSIQUE_CFG BASE_POPQA_CFG \
  VAR_HOTPOT_CFG VAR_2WIKI_CFG VAR_MUSIQUE_CFG VAR_POPQA_CFG
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

env = __import__("os").environ
out_root = Path(env["OUT_ROOT"]).resolve()
sample_size = int(env["SAMPLE_SIZE"])

cfg_paths = {
    "baseline_hotpotqa": Path(env["BASE_HOTPOT_CFG"]).resolve(),
    "baseline_2wikimultihopqa": Path(env["BASE_2WIKI_CFG"]).resolve(),
    "baseline_musique": Path(env["BASE_MUSIQUE_CFG"]).resolve(),
    "baseline_popqa": Path(env["BASE_POPQA_CFG"]).resolve(),
    "variant_hotpotqa": Path(env["VAR_HOTPOT_CFG"]).resolve(),
    "variant_2wikimultihopqa": Path(env["VAR_2WIKI_CFG"]).resolve(),
    "variant_musique": Path(env["VAR_MUSIQUE_CFG"]).resolve(),
    "variant_popqa": Path(env["VAR_POPQA_CFG"]).resolve(),
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


baseline_cfg = {
    "hotpotqa": load_yaml(cfg_paths["baseline_hotpotqa"]),
    "2wikimultihopqa": load_yaml(cfg_paths["baseline_2wikimultihopqa"]),
    "musique": load_yaml(cfg_paths["baseline_musique"]),
    "popqa": load_yaml(cfg_paths["baseline_popqa"]),
}
variant_cfg = {
    "hotpotqa": load_yaml(cfg_paths["variant_hotpotqa"]),
    "2wikimultihopqa": load_yaml(cfg_paths["variant_2wikimultihopqa"]),
    "musique": load_yaml(cfg_paths["variant_musique"]),
    "popqa": load_yaml(cfg_paths["variant_popqa"]),
}

subset_root = out_root / "sampled_query_ids" / "subset_data"
subset_root.mkdir(parents=True, exist_ok=True)
shared_ids = {}
subset_paths = {}

for dataset, cfg in baseline_cfg.items():
    src = Path(str(cfg.get("data_path", ""))).expanduser().resolve()
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
    id_path = out_root / "sampled_query_ids" / f"{dataset}.json"
    subset_path = subset_root / f"{dataset}.json"
    write_json(id_path, ids)
    write_json(subset_path, sampled)
    shared_ids[dataset] = str(id_path.resolve())
    subset_paths[dataset] = str(subset_path.resolve())

legacy_shared = out_root / "shared_sample_ids"
legacy_shared.mkdir(parents=True, exist_ok=True)
for dataset, id_path in shared_ids.items():
    write_json(legacy_shared / f"{dataset}.json", json.loads(Path(id_path).read_text(encoding="utf-8")))


def patch_cfg(cfg, dataset):
    out = dict(cfg or {})
    out["data_path"] = subset_paths[dataset]
    out["limit"] = int(sample_size)
    out["timestamp_output"] = True
    return out


patched = {}
for dataset in ("hotpotqa", "2wikimultihopqa", "musique", "popqa"):
    patched[f"phase6x_{dataset}_v12"] = patch_cfg(baseline_cfg[dataset], dataset)
    patched[f"phase6x_{dataset}_v12_early_collapse_recovery"] = patch_cfg(variant_cfg[dataset], dataset)

patched_paths = {}
for run_name, cfg in patched.items():
    p = out_root / "configs" / f"{run_name}.yaml"
    write_yaml(p, cfg)
    patched_paths[run_name] = str(p.resolve())

manifest = {
    "phase": "phase6x_early_collapse_recovery_4ds_n1000",
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
            "gpu": "0",
            "process_group": "A",
            "patched_config_path": patched_paths["phase6x_2wikimultihopqa_v12"],
            "scheduled": True,
        },
        {
            "run_name": "phase6x_2wikimultihopqa_v12_early_collapse_recovery",
            "dataset": "2wikimultihopqa",
            "profile": "unified_acr_rcedr_v12_early_collapse_recovery",
            "role": "variant",
            "gpu": "0",
            "process_group": "A",
            "patched_config_path": patched_paths["phase6x_2wikimultihopqa_v12_early_collapse_recovery"],
            "scheduled": True,
        },
        {
            "run_name": "phase6x_musique_v12",
            "dataset": "musique",
            "profile": "unified_acr_rcedr_v12",
            "role": "baseline",
            "gpu": "1",
            "process_group": "B",
            "patched_config_path": patched_paths["phase6x_musique_v12"],
            "scheduled": True,
        },
        {
            "run_name": "phase6x_musique_v12_early_collapse_recovery",
            "dataset": "musique",
            "profile": "unified_acr_rcedr_v12_early_collapse_recovery",
            "role": "variant",
            "gpu": "1",
            "process_group": "B",
            "patched_config_path": patched_paths["phase6x_musique_v12_early_collapse_recovery"],
            "scheduled": True,
        },
        {
            "run_name": "phase6x_popqa_v12",
            "dataset": "popqa",
            "profile": "unified_acr_rcedr_v12",
            "role": "baseline",
            "gpu": "1",
            "process_group": "C",
            "patched_config_path": patched_paths["phase6x_popqa_v12"],
            "scheduled": True,
        },
        {
            "run_name": "phase6x_popqa_v12_early_collapse_recovery",
            "dataset": "popqa",
            "profile": "unified_acr_rcedr_v12_early_collapse_recovery",
            "role": "variant",
            "gpu": "1",
            "process_group": "C",
            "patched_config_path": patched_paths["phase6x_popqa_v12_early_collapse_recovery"],
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
  run_one "${GPU_A}" "phase6x_2wikimultihopqa_v12" "${OUT_ROOT}/configs/phase6x_2wikimultihopqa_v12.yaml"
  run_one "${GPU_A}" "phase6x_2wikimultihopqa_v12_early_collapse_recovery" "${OUT_ROOT}/configs/phase6x_2wikimultihopqa_v12_early_collapse_recovery.yaml"
}

group_b() {
  run_one "${GPU_B}" "phase6x_musique_v12" "${OUT_ROOT}/configs/phase6x_musique_v12.yaml"
  run_one "${GPU_B}" "phase6x_musique_v12_early_collapse_recovery" "${OUT_ROOT}/configs/phase6x_musique_v12_early_collapse_recovery.yaml"
}

group_c() {
  run_one "${GPU_C}" "phase6x_popqa_v12" "${OUT_ROOT}/configs/phase6x_popqa_v12.yaml"
  run_one "${GPU_C}" "phase6x_popqa_v12_early_collapse_recovery" "${OUT_ROOT}/configs/phase6x_popqa_v12_early_collapse_recovery.yaml"
}

group_a > "${OUT_ROOT}/logs/group_A_gpu${GPU_A}.log" 2>&1 &
PID_A=$!
group_b > "${OUT_ROOT}/logs/group_B_gpu${GPU_B}.log" 2>&1 &
PID_B=$!
group_c > "${OUT_ROOT}/logs/group_C_gpu${GPU_C}.log" 2>&1 &
PID_C=$!

FAIL=0
if ! wait "${PID_A}"; then FAIL=1; fi
if ! wait "${PID_B}"; then FAIL=1; fi
if ! wait "${PID_C}"; then FAIL=1; fi
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] one or more process groups failed." >&2
  exit 1
fi

echo "=== summarize PHASE6X early-collapse recovery 4DS n1000 ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6x_early_collapse_recovery.py \
  --out-root "${OUT_ROOT}" \
  --datasets hotpotqa,2wikimultihopqa,musique,popqa \
  --baseline-profile "${BASELINE_PROFILE}" \
  --variant-profile "${VARIANT_PROFILE}" \
  --top-k 20 \
  --summary-title "PHASE6X Early-Collapse Recovery 4DS N1000 Summary" \
  --summary-md-name "PHASE6X_EARLY_COLLAPSE_RECOVERY_4DS_N1000_SUMMARY.md" \
  --summary-json-name "phase6x_early_collapse_recovery_4ds_n1000_summary.json" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6x_early_collapse_recovery_4ds_n1000.log"

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
echo "Summary: ${OUT_ROOT}/PHASE6X_EARLY_COLLAPSE_RECOVERY_4DS_N1000_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6x_early_collapse_recovery_4ds_n1000_summary.json"
