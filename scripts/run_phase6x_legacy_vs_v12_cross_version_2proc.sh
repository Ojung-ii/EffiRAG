#!/usr/bin/env bash
set -euo pipefail

cd /home/ojungii/EffiRAG

LEGACY_ROOT="${LEGACY_ROOT:-/home/ojungii/EffiRAG_legacy_56779c0}"
LATEST_ROOT="${LATEST_ROOT:-/home/ojungii/EffiRAG_latest_audit}"
OUT_ROOT="${OUT_ROOT:-outputs/phase6x_legacy_vs_v12_cross_version_n100}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
PROCESS_COUNT="${PROCESS_COUNT:-2}"
STRICT_VALIDATE="${STRICT_VALIDATE:-1}"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"

LEGACY_PROFILE_MODE="${LEGACY_PROFILE_MODE:-on_the_fly}"
LEGACY_PROFILE_ROOT="${LEGACY_PROFILE_ROOT:-configs/SOTA_config/copy_span_instruction_4ds}"
LATEST_PROFILE_ROOT="${LATEST_PROFILE_ROOT:-configs/main_config/copy_span_instruction_unified/unified_acr_rcedr_v12}"

OUT_ROOT="$("${PYTHON}" -c "from pathlib import Path; print(Path(r'''${OUT_ROOT}''').resolve())")"

if [[ "${PROCESS_COUNT}" != "2" ]]; then
  echo "[ERROR] PROCESS_COUNT must be 2 for this runner, got: ${PROCESS_COUNT}" >&2
  exit 1
fi

if [[ ! -d "${LEGACY_ROOT}" ]]; then
  echo "[ERROR] LEGACY_ROOT missing: ${LEGACY_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${LATEST_ROOT}" ]]; then
  echo "[ERROR] LATEST_ROOT missing: ${LATEST_ROOT}" >&2
  exit 1
fi

if [[ -d "${OUT_ROOT}/raw" ]] && [[ "${ALLOW_REUSE_OUT_ROOT:-0}" != "1" ]]; then
  echo "[ERROR] OUT_ROOT already contains raw runs: ${OUT_ROOT}" >&2
  echo "Use fresh OUT_ROOT or set ALLOW_REUSE_OUT_ROOT=1 explicitly." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/raw" "${OUT_ROOT}/normalized" "${OUT_ROOT}/examples" "${OUT_ROOT}/shared_sample_ids" "${OUT_ROOT}/configs"

echo "=== PHASE6X cross-version legacy-vs-v12 stagewise audit ==="
echo "LEGACY_ROOT=${LEGACY_ROOT}"
echo "LATEST_ROOT=${LATEST_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "LEGACY_PROFILE_MODE=${LEGACY_PROFILE_MODE}"

LEGACY_COMMIT="$(git -C "${LEGACY_ROOT}" rev-parse HEAD)"
LATEST_COMMIT="$(git -C "${LATEST_ROOT}" rev-parse HEAD)"
echo "legacy_commit=${LEGACY_COMMIT}"
echo "latest_commit=${LATEST_COMMIT}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_ROOT}/logs/nvidia_smi_before.txt" || true
fi

LEGACY_HOTPOT_CFG_SRC="${LEGACY_ROOT}/${LEGACY_PROFILE_ROOT}/${LEGACY_PROFILE_MODE}/hotpotqa.yaml"
LEGACY_2WIKI_CFG_SRC="${LEGACY_ROOT}/${LEGACY_PROFILE_ROOT}/${LEGACY_PROFILE_MODE}/2wikimultihopqa.yaml"
LATEST_HOTPOT_CFG_SRC="${LATEST_ROOT}/${LATEST_PROFILE_ROOT}/hotpotqa.yaml"
LATEST_2WIKI_CFG_SRC="${LATEST_ROOT}/${LATEST_PROFILE_ROOT}/2wikimultihopqa.yaml"

for cfg in "${LEGACY_HOTPOT_CFG_SRC}" "${LEGACY_2WIKI_CFG_SRC}" "${LATEST_HOTPOT_CFG_SRC}" "${LATEST_2WIKI_CFG_SRC}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "[ERROR] missing config: ${cfg}" >&2
    exit 1
  fi
done

export LEGACY_ROOT LATEST_ROOT OUT_ROOT SAMPLE_SIZE \
  LEGACY_HOTPOT_CFG_SRC LEGACY_2WIKI_CFG_SRC LATEST_HOTPOT_CFG_SRC LATEST_2WIKI_CFG_SRC \
  LEGACY_COMMIT LATEST_COMMIT
PYTHONPATH=. "${PYTHON}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sample_id_of(record):
    return str(record.get("_id") or record.get("id") or record.get("qid") or "").strip()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


out_root = Path(os.environ["OUT_ROOT"]).resolve()
sample_size = int(os.environ["SAMPLE_SIZE"])

legacy_hotpot_src = Path(os.environ["LEGACY_HOTPOT_CFG_SRC"]).resolve()
legacy_2wiki_src = Path(os.environ["LEGACY_2WIKI_CFG_SRC"]).resolve()
latest_hotpot_src = Path(os.environ["LATEST_HOTPOT_CFG_SRC"]).resolve()
latest_2wiki_src = Path(os.environ["LATEST_2WIKI_CFG_SRC"]).resolve()

legacy_hotpot_cfg = load_yaml(legacy_hotpot_src)
legacy_2wiki_cfg = load_yaml(legacy_2wiki_src)
latest_hotpot_cfg = load_yaml(latest_hotpot_src)
latest_2wiki_cfg = load_yaml(latest_2wiki_src)

# Prefer latest v12 paths as canonical source; fallback to legacy if missing.
data_candidates = {
    "hotpotqa": [
        Path(str(latest_hotpot_cfg.get("data_path", "") or "")).expanduser(),
        Path(str(legacy_hotpot_cfg.get("data_path", "") or "")).expanduser(),
    ],
    "2wikimultihopqa": [
        Path(str(latest_2wiki_cfg.get("data_path", "") or "")).expanduser(),
        Path(str(legacy_2wiki_cfg.get("data_path", "") or "")).expanduser(),
    ],
}

shared_ids_root = out_root / "shared_sample_ids"
subset_root = shared_ids_root / "subset_data"
subset_root.mkdir(parents=True, exist_ok=True)

subset_paths = {}
id_paths = {}

for dataset, candidates in data_candidates.items():
    source_path = None
    for cand in candidates:
        if str(cand) and cand.exists():
            source_path = cand
            break
    if source_path is None:
        raise FileNotFoundError(f"cannot locate source dataset file for {dataset}: {candidates}")

    records = load_json(source_path)
    if not isinstance(records, list):
        raise ValueError(f"dataset file must be JSON list: {source_path}")

    sampled_ids = []
    sampled_records = []
    seen = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        sid = sample_id_of(rec)
        if not sid or sid in seen:
            continue
        seen.add(sid)
        sampled_ids.append(sid)
        sampled_records.append(rec)
        if len(sampled_ids) >= sample_size:
            break

    if len(sampled_ids) < sample_size:
        raise ValueError(f"insufficient samples for {dataset}: requested={sample_size}, got={len(sampled_ids)}")

    id_path = shared_ids_root / f"{dataset}.json"
    subset_path = subset_root / f"{dataset}.json"
    write_json(id_path, sampled_ids)
    write_json(subset_path, sampled_records)
    id_paths[dataset] = str(id_path.resolve())
    subset_paths[dataset] = str(subset_path.resolve())


def patch_cfg(base_cfg: dict, *, dataset: str, method_label: str) -> dict:
    cfg = dict(base_cfg or {})
    cfg["data_path"] = subset_paths[dataset]
    cfg["limit"] = int(sample_size)
    cfg["timestamp_output"] = True
    # Keep behavior identical otherwise; run-specific output_dir will be supplied via CLI.
    return cfg


patched = {
    "legacy_hotpotqa": patch_cfg(legacy_hotpot_cfg, dataset="hotpotqa", method_label="legacy"),
    "legacy_2wikimultihopqa": patch_cfg(legacy_2wiki_cfg, dataset="2wikimultihopqa", method_label="legacy"),
    "v12_hotpotqa": patch_cfg(latest_hotpot_cfg, dataset="hotpotqa", method_label="v12"),
    "v12_2wikimultihopqa": patch_cfg(latest_2wiki_cfg, dataset="2wikimultihopqa", method_label="v12"),
}

patched_paths = {}
for run_name, cfg in patched.items():
    p = out_root / "configs" / f"{run_name}.yaml"
    write_yaml(p, cfg)
    patched_paths[run_name] = str(p.resolve())


def key_flags(cfg: dict, method_label: str) -> dict:
    keys = {
        "retrieval_objective_mode": cfg.get("retrieval_objective_mode"),
        "answer_support_pinning_enabled": cfg.get("answer_support_pinning_enabled"),
        "final_top_slice_reorder_enabled": cfg.get("final_top_slice_reorder_enabled"),
        "top_corridors": cfg.get("top_corridors"),
        "max_corridors_in_context": cfg.get("max_corridors_in_context"),
        "render_mode": cfg.get("render_mode"),
        "chunk_package_enabled": cfg.get("chunk_package_enabled"),
        "max_excerpt_sentences_per_package": cfg.get("max_excerpt_sentences_per_package"),
        "max_context_sentences_per_selected": cfg.get("max_context_sentences_per_selected"),
        "trim_on": cfg.get("trim_on"),
        "trim_rho": cfg.get("trim_rho"),
    }
    if method_label == "v12":
        keys.update(
            {
                "unified_acr_rcedr_enabled": cfg.get("unified_acr_rcedr_enabled"),
                "unified_acr_rcedr_max_atoms": cfg.get("unified_acr_rcedr_max_atoms"),
                "unified_acr_rcedr_max_tokens": cfg.get("unified_acr_rcedr_max_tokens"),
            }
        )
    return keys


runs = [
    {
        "run_name": "legacy_hotpotqa",
        "method_label": "legacy",
        "dataset": "hotpotqa",
        "gpu": "0",
        "process_group": "A",
        "profile": "legacy_sota_on_the_fly",
        "root": str(Path(os.environ["LEGACY_ROOT"]).resolve()),
        "config_path": str(legacy_hotpot_src),
        "patched_config_path": patched_paths["legacy_hotpotqa"],
        "run_root": str((out_root / "raw" / "legacy_hotpotqa").resolve()),
        "scheduled": True,
        "key_flags": key_flags(legacy_hotpot_cfg, "legacy"),
    },
    {
        "run_name": "v12_hotpotqa",
        "method_label": "v12",
        "dataset": "hotpotqa",
        "gpu": "0",
        "process_group": "A",
        "profile": "unified_acr_rcedr_v12",
        "root": str(Path(os.environ["LATEST_ROOT"]).resolve()),
        "config_path": str(latest_hotpot_src),
        "patched_config_path": patched_paths["v12_hotpotqa"],
        "run_root": str((out_root / "raw" / "v12_hotpotqa").resolve()),
        "scheduled": True,
        "key_flags": key_flags(latest_hotpot_cfg, "v12"),
    },
    {
        "run_name": "legacy_2wikimultihopqa",
        "method_label": "legacy",
        "dataset": "2wikimultihopqa",
        "gpu": "1",
        "process_group": "B",
        "profile": "legacy_sota_on_the_fly",
        "root": str(Path(os.environ["LEGACY_ROOT"]).resolve()),
        "config_path": str(legacy_2wiki_src),
        "patched_config_path": patched_paths["legacy_2wikimultihopqa"],
        "run_root": str((out_root / "raw" / "legacy_2wikimultihopqa").resolve()),
        "scheduled": True,
        "key_flags": key_flags(legacy_2wiki_cfg, "legacy"),
    },
    {
        "run_name": "v12_2wikimultihopqa",
        "method_label": "v12",
        "dataset": "2wikimultihopqa",
        "gpu": "1",
        "process_group": "B",
        "profile": "unified_acr_rcedr_v12",
        "root": str(Path(os.environ["LATEST_ROOT"]).resolve()),
        "config_path": str(latest_2wiki_src),
        "patched_config_path": patched_paths["v12_2wikimultihopqa"],
        "run_root": str((out_root / "raw" / "v12_2wikimultihopqa").resolve()),
        "scheduled": True,
        "key_flags": key_flags(latest_2wiki_cfg, "v12"),
    },
]

manifest = {
    "phase": "phase6x_legacy_vs_v12_cross_version_audit",
    "sample_size": sample_size,
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "out_root": str(out_root),
    "legacy_root": str(Path(os.environ["LEGACY_ROOT"]).resolve()),
    "latest_root": str(Path(os.environ["LATEST_ROOT"]).resolve()),
    "legacy_commit": str(os.environ["LEGACY_COMMIT"]),
    "latest_commit": str(os.environ["LATEST_COMMIT"]),
    "legacy_hotpotqa_config_path": str(legacy_hotpot_src),
    "legacy_2wiki_config_path": str(legacy_2wiki_src),
    "latest_hotpotqa_config_path": str(latest_hotpot_src),
    "latest_2wiki_config_path": str(latest_2wiki_src),
    "shared_sample_ids": dict(id_paths),
    "subset_data_paths": dict(subset_paths),
    "runs": runs,
}
write_json(out_root / "phase6x_run_manifest.json", manifest)
print(out_root / "phase6x_run_manifest.json")
PY

run_one() {
  local run_name="$1"
  local root="$2"
  local cfg="$3"
  local dataset="$4"
  local gpu="$5"
  local run_root="$6"
  local log_path="${OUT_ROOT}/logs/qa_${run_name}.log"

  rm -rf "${run_root}"
  mkdir -p "${run_root}"

  echo "=== [${run_name}] START dataset=${dataset} gpu=${gpu} ==="
  (
    cd "${root}"
    PYTHONHASHSEED=0 \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH=. \
    "${PYTHON}" -m effirag.run_rag \
      --config "${cfg}" \
      --output-dir "${run_root}" \
      --timestamp-output true \
      --limit "${SAMPLE_SIZE}"
  ) 2>&1 | tee "${log_path}"
  local code=${PIPESTATUS[0]}
  if [[ "${code}" -ne 0 ]]; then
    echo "[ERROR] run failed: ${run_name} (exit=${code})" >&2
    return "${code}"
  fi

  # Enforce shared sample IDs.
  PYTHONPATH=. "${PYTHON}" - <<PY
import json
from pathlib import Path
run_root = Path(${run_root@Q})
dataset = ${dataset@Q}
ids_path = Path(${OUT_ROOT@Q}) / "shared_sample_ids" / f"{dataset}.json"
paths = sorted(run_root.rglob("rag_query_results.jsonl"))
if not paths:
    raise SystemExit("missing rag_query_results.jsonl")
qpath = paths[-1]
expected = json.loads(ids_path.read_text(encoding="utf-8"))
observed = []
for line in qpath.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    sid = str(obj.get("sample_id", "")).strip()
    if sid:
        observed.append(sid)
if expected != observed:
    print("MISMATCH", dataset)
    print("expected_len", len(expected), "observed_len", len(observed))
    m = min(len(expected), len(observed))
    idx = next((i for i in range(m) if expected[i] != observed[i]), None)
    print("first_diff_index", idx)
    raise SystemExit(3)
print("MATCH", dataset, len(expected))
PY
  echo "=== [${run_name}] DONE ==="
}

group_hotpot() {
  local legacy_cfg="${OUT_ROOT}/configs/legacy_hotpotqa.yaml"
  local v12_cfg="${OUT_ROOT}/configs/v12_hotpotqa.yaml"
  run_one "legacy_hotpotqa" "${LEGACY_ROOT}" "${legacy_cfg}" "hotpotqa" "0" "${OUT_ROOT}/raw/legacy_hotpotqa"
  run_one "v12_hotpotqa" "${LATEST_ROOT}" "${v12_cfg}" "hotpotqa" "0" "${OUT_ROOT}/raw/v12_hotpotqa"
}

group_2wiki() {
  local legacy_cfg="${OUT_ROOT}/configs/legacy_2wikimultihopqa.yaml"
  local v12_cfg="${OUT_ROOT}/configs/v12_2wikimultihopqa.yaml"
  run_one "legacy_2wikimultihopqa" "${LEGACY_ROOT}" "${legacy_cfg}" "2wikimultihopqa" "1" "${OUT_ROOT}/raw/legacy_2wikimultihopqa"
  run_one "v12_2wikimultihopqa" "${LATEST_ROOT}" "${v12_cfg}" "2wikimultihopqa" "1" "${OUT_ROOT}/raw/v12_2wikimultihopqa"
}

group_hotpot > "${OUT_ROOT}/logs/group_A_gpu0.log" 2>&1 &
PID_A=$!
group_2wiki > "${OUT_ROOT}/logs/group_B_gpu1.log" 2>&1 &
PID_B=$!

FAIL=0
if ! wait "${PID_A}"; then FAIL=1; fi
if ! wait "${PID_B}"; then FAIL=1; fi
if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] one or more workers failed. Check ${OUT_ROOT}/logs/group_*.log" >&2
  exit 1
fi

echo "=== summarize PHASE6X legacy-vs-v12 cross-version audit ==="
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase6x_legacy_vs_v12_cross_version_audit.py \
  --out-root "${OUT_ROOT}" \
  --datasets hotpotqa,2wikimultihopqa \
  --top-k 20 \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase6x_legacy_vs_v12_cross_version_audit.log"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_ROOT}/logs/nvidia_smi_after.txt" || true
fi

echo "=== done ==="
echo "Summary: ${OUT_ROOT}/PHASE6X_LEGACY_VS_V12_CROSS_VERSION_AUDIT_SUMMARY.md"
echo "Summary JSON: ${OUT_ROOT}/phase6x_legacy_vs_v12_cross_version_audit_summary.json"
echo "Validation JSON: ${OUT_ROOT}/phase6x_artifact_validation.json"
