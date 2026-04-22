#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "/home/ojungii/miniconda3/envs/effirag/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM="false"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATASETS_CSV="hotpotqa,2wikimultihopqa"
VARIANTS_CSV="champion_config,baseline_mid_reconfirm,ref_reconfirm"
N_SAMPLES="1000"
QA_ROOT="data/qa"
CORPUS_ROOT="/home/ojungii/HippoRAG2/dataset"
OUTPUT_ROOT="outputs/retrieval_metric_hotpot_2wiki_round"
GRAPH_CACHE_DIR="outputs/index_cache"
LLM_BASE_URL="http://localhost:8011/v1"
LLM_API_KEY="EMPTY"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
SKIP_COMPLETED="true"
STOP_ON_FAILURE="false"
SEQUENTIAL="true"

DATA_PATH_HOTPOTQA=""
DATA_PATH_2WIKIMULTIHOPQA=""
CORPUS_PATH_HOTPOTQA=""
CORPUS_PATH_2WIKIMULTIHOPQA=""

RAG_BASE_CFG="configs/canonical/rag_entity_first_chunk_grounded.yaml"
UNIVERSAL_EMBED_MAX_LEN=256
UNIVERSAL_EMBED_MAX_CHARS=800

log_msg() {
  local now
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${now}] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

parse_bool() {
  local raw="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
    1|true|yes|y|on) echo "true" ;;
    0|false|no|n|off) echo "false" ;;
    *) echo "" ;;
  esac
}

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  out_ref=()
  IFS=',' read -r -a _items <<< "${raw}"
  for item in "${_items[@]}"; do
    item="$(echo "${item}" | sed -e 's/^ *//' -e 's/ *$//')"
    [[ -n "${item}" ]] && out_ref+=("${item}")
  done
}

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_retrieval_metric_hotpot_2wiki.sh [options]

Required/primary:
  --datasets CSV
  --variants CSV
  --n-samples N
  --qa-root PATH
  --corpus-root PATH
  --output-root PATH
  --graph-cache-dir PATH
  --llm-base-url URL
  --llm-api-key KEY
  --model-name NAME

Optional:
  --skip-completed [true|false]
  --stop-on-failure [true|false]
  --sequential [true|false]
  --data-path-hotpotqa PATH
  --data-path-2wikimultihopqa PATH
  --corpus-path-hotpotqa PATH
  --corpus-path-2wikimultihopqa PATH
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --variants) VARIANTS_CSV="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --qa-root) QA_ROOT="$2"; shift 2 ;;
    --corpus-root) CORPUS_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --graph-cache-dir) GRAPH_CACHE_DIR="$2"; shift 2 ;;
    --llm-base-url) LLM_BASE_URL="$2"; shift 2 ;;
    --llm-api-key) LLM_API_KEY="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --skip-completed) SKIP_COMPLETED="$(parse_bool "$2")"; shift 2 ;;
    --stop-on-failure) STOP_ON_FAILURE="$(parse_bool "$2")"; shift 2 ;;
    --sequential) SEQUENTIAL="$(parse_bool "$2")"; shift 2 ;;
    --data-path-hotpotqa) DATA_PATH_HOTPOTQA="$2"; shift 2 ;;
    --data-path-2wikimultihopqa) DATA_PATH_2WIKIMULTIHOPQA="$2"; shift 2 ;;
    --corpus-path-hotpotqa) CORPUS_PATH_HOTPOTQA="$2"; shift 2 ;;
    --corpus-path-2wikimultihopqa) CORPUS_PATH_2WIKIMULTIHOPQA="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

SKIP_COMPLETED="$(parse_bool "${SKIP_COMPLETED}")"
STOP_ON_FAILURE="$(parse_bool "${STOP_ON_FAILURE}")"
SEQUENTIAL="$(parse_bool "${SEQUENTIAL}")"
[[ -n "${SKIP_COMPLETED}" ]] || die "Invalid --skip-completed"
[[ -n "${STOP_ON_FAILURE}" ]] || die "Invalid --stop-on-failure"
[[ -n "${SEQUENTIAL}" ]] || die "Invalid --sequential"
if [[ "${SEQUENTIAL}" != "true" ]]; then
  log_msg "[WARN] sequential-only runner; forcing --sequential true"
  SEQUENTIAL="true"
fi

split_csv "${DATASETS_CSV}" SELECTED_DATASETS
split_csv "${VARIANTS_CSV}" SELECTED_VARIANTS
[[ ${#SELECTED_DATASETS[@]} -gt 0 ]] || die "--datasets resolved to empty"
[[ ${#SELECTED_VARIANTS[@]} -gt 0 ]] || die "--variants resolved to empty"

VALID_DATASETS=("hotpotqa" "2wikimultihopqa")
VALID_VARIANTS=("champion_config" "baseline_mid_reconfirm" "ref_reconfirm")

contains_item() {
  local needle="$1"
  shift
  for x in "$@"; do
    if [[ "${x}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

for ds in "${SELECTED_DATASETS[@]}"; do
  contains_item "${ds}" "${VALID_DATASETS[@]}" || die "Unsupported dataset '${ds}'"
done
for v in "${SELECTED_VARIANTS[@]}"; do
  contains_item "${v}" "${VALID_VARIANTS[@]}" || die "Unsupported variant '${v}'"
done

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
ROUND_ROOT="${OUTPUT_ROOT}/${RUN_STAMP}"
LOG_ROOT="logs/retrieval_metric_hotpot_2wiki_round/${RUN_STAMP}"
CFG_ROOT="${ROUND_ROOT}/configs"
RUN_ROOT="${ROUND_ROOT}/runs"
mkdir -p "${ROUND_ROOT}" "${LOG_ROOT}" "${CFG_ROOT}" "${RUN_ROOT}"

RUN_RECORDS_TSV="${ROUND_ROOT}/run_records.tsv"
FAILURES_JSON="${ROUND_ROOT}/failures.json"
ENV_SNAPSHOT_JSON="${ROUND_ROOT}/environment_snapshot.json"
CHAMPION_MANIFEST_JSON="${ROUND_ROOT}/champion_manifest.json"
CHAMPION_REPRO_TABLE_MD="${ROUND_ROOT}/champion_repro_table.md"
CHAMPION_MATRIX_TSV="${ROUND_ROOT}/champion_matrix.tsv"
MANIFEST_MATRIX_TSV="${ROUND_ROOT}/manifest_matrix.tsv"

METRIC_JSON="${ROUND_ROOT}/metric_round_metrics.json"
METRIC_MD="${ROUND_ROOT}/metric_round_metrics.md"
METRIC_DELTA_MD="${ROUND_ROOT}/metric_delta_table.md"
METRIC_EFF_MD="${ROUND_ROOT}/metric_efficiency_table.md"
METRIC_DIAG_MD="${ROUND_ROOT}/metric_diagnostic_table.md"
METRIC_DEFS_MD="${ROUND_ROOT}/metric_definition_notes.md"

printf "stage\tvariant\tdataset\tprofile\tkind\tqa_path\tcorpus_path\tsummary_path\tconfig_path\tlog_path\tstatus\tstart_ts\tend_ts\telapsed_s\tn_samples\terror_message\n" > "${RUN_RECORDS_TSV}"
printf "stage\tvariant\tdataset\tprofile\tsource\tqa_path\tcorpus_path\tn_samples\tchampion_summary\tchampion_sidecar_config\n" > "${MANIFEST_MATRIX_TSV}"

declare -A QA_PATH_BY_DATASET
declare -A CORPUS_PATH_BY_DATASET
declare -A CHAMPION_SUMMARY_BY_DATASET
declare -A CHAMPION_SIDECAR_CFG_BY_DATASET

resolve_dataset_paths() {
  local dataset="$1"
  local qa_path=""
  local corpus_path=""
  case "${dataset}" in
    hotpotqa)
      qa_path="${DATA_PATH_HOTPOTQA:-${QA_ROOT}/hotpotqa.json}"
      corpus_path="${CORPUS_PATH_HOTPOTQA:-${CORPUS_ROOT}/hotpotqa_corpus.json}"
      ;;
    2wikimultihopqa)
      qa_path="${DATA_PATH_2WIKIMULTIHOPQA:-${QA_ROOT}/2wikimultihopqa.json}"
      corpus_path="${CORPUS_PATH_2WIKIMULTIHOPQA:-${CORPUS_ROOT}/2wikimultihopqa_corpus.json}"
      ;;
    *)
      die "Unsupported dataset in resolve_dataset_paths: ${dataset}"
      ;;
  esac
  [[ -f "${qa_path}" ]] || die "Missing QA path for ${dataset}: ${qa_path}"
  [[ -f "${corpus_path}" ]] || die "Missing corpus path for ${dataset}: ${corpus_path}"
  QA_PATH_BY_DATASET["${dataset}"]="${qa_path}"
  CORPUS_PATH_BY_DATASET["${dataset}"]="${corpus_path}"
}

write_environment_snapshot() {
  "${PYTHON_BIN}" - "${ENV_SNAPSHOT_JSON}" \
    "${LLM_BASE_URL}" "${LLM_API_KEY}" "${MODEL_NAME}" "${DATASETS_CSV}" "${VARIANTS_CSV}" "${N_SAMPLES}" <<'PY'
import json
import subprocess
import sys

out_path, llm_base_url, llm_api_key, model_name, datasets_csv, variants_csv, n_samples = sys.argv[1:8]

def _cmd(args):
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return ""

payload = {
    "python_version": sys.version,
    "git_commit_hash": _cmd(["git", "rev-parse", "HEAD"]),
    "active_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    "model_name": model_name,
    "llm_base_url": llm_base_url,
    "llm_api_key": llm_api_key if llm_api_key == "EMPTY" else "***",
    "selected_datasets": [x for x in datasets_csv.split(",") if x],
    "selected_variants": [x for x in variants_csv.split(",") if x],
    "n_samples": int(float(n_samples)),
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(out_path)
PY
}

upsert_record() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local profile="$4"
  local kind="$5"
  local qa_path="$6"
  local corpus_path="$7"
  local summary_path="$8"
  local config_path="$9"
  local log_path="${10}"
  local status="${11}"
  local start_ts="${12}"
  local end_ts="${13}"
  local elapsed_s="${14}"
  local n_samples="${15}"
  local error_message="${16}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${stage}" "${variant}" "${dataset}" "${profile}" "${kind}" \
    "${qa_path}" "${corpus_path}" "${summary_path}" "${config_path}" "${log_path}" \
    "${status}" "${start_ts}" "${end_ts}" "${elapsed_s}" "${n_samples}" "${error_message}" >> "${RUN_RECORDS_TSV}"
}

find_latest_summary() {
  local out_dir="$1"
  local dataset="$2"
  local path
  path="$(find "${out_dir}/${dataset}" -type f -name "rag_summary.json" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}' || true)"
  echo "${path}"
}

is_complete_summary() {
  local summary_path="$1"
  local expected_limit="$2"
  "${PYTHON_BIN}" - "${summary_path}" "${expected_limit}" <<'PY'
import json
import sys
from pathlib import Path
sp = Path(sys.argv[1])
limit = int(float(sys.argv[2]))
if not sp.exists():
    raise SystemExit(1)
_ = json.loads(sp.read_text(encoding="utf-8"))
qp = sp.with_name("rag_query_results.jsonl")
if not qp.exists():
    raise SystemExit(1)
count = 0
with qp.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            count += 1
if count < limit:
    raise SystemExit(1)
print("ok")
PY
}

discover_champions() {
  log_msg "[Stage 0] champion discovery"
  "${PYTHON_BIN}" - \
    "${CHAMPION_MANIFEST_JSON}" \
    "${CHAMPION_REPRO_TABLE_MD}" \
    "${CHAMPION_MATRIX_TSV}" \
    "${DATASETS_CSV}" <<'PY'
import json
import re
import sys
from pathlib import Path

out_json, out_md, out_tsv, datasets_csv = sys.argv[1:5]
datasets = [x.strip() for x in datasets_csv.split(",") if x.strip()]
drop_pat = re.compile(r"(?:/D2/|oracle|hippo)", re.IGNORECASE)

def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)

def _i(v, d=0):
    try:
        return int(v)
    except Exception:
        return int(d)

candidates = {d: [] for d in datasets}
for sp in Path("outputs").rglob("rag_summary.json"):
    spt = str(sp)
    if drop_pat.search(spt):
        continue
    try:
        summary = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        continue
    ds = str(summary.get("dataset", "") or "")
    if ds not in candidates:
        continue
    if bool(summary.get("retrieval_only", False)):
        continue
    if bool(summary.get("oracle_support_injection_enabled", False)):
        continue
    rp = dict(summary.get("retrieval_params", {}) or {})
    if _i(rp.get("embedding_max_length", 0), 0) != 256:
        continue
    if _i(rp.get("embedding_text_max_chars", 0), 0) != 800:
        continue
    r5 = _f(summary.get("recall_at_5", summary.get("supporting_fact_recall_at_5", 0.0)), 0.0)
    if r5 <= 0.0:
        ratk = summary.get("supporting_fact_recall_at_k", {}) or {}
        if isinstance(ratk, dict):
            r5 = _f(ratk.get("5", ratk.get(5, 0.0)), 0.0)
    candidates[ds].append(
        {
            "summary_path": str(sp.resolve()),
            "f1": _f(summary.get("f1", summary.get("F1", 0.0)), 0.0),
            "em": _f(summary.get("em", summary.get("EM", 0.0)), 0.0),
            "r5": r5,
            "retrieval_ms": _f(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
            "total_ms": _f(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
            "fallback_rate": _f(summary.get("fallback_rate", 0.0), 0.0),
            "run_timestamp": str(summary.get("run_timestamp", "") or ""),
            "run_timestamp_utc": str(summary.get("run_timestamp_utc", "") or ""),
            "retrieval_mode": str(rp.get("retrieval_objective_mode", "baseline") or "baseline"),
            "canonical_variant_name": str(rp.get("canonical_variant_name", "") or ""),
            "order_strategy": str((summary.get("render_params", {}) or {}).get("order_strategy", "score") or "score"),
        }
    )

champions = {}
for ds in datasets:
    arr = candidates.get(ds, [])
    if not arr:
        continue
    arr.sort(key=lambda x: (x["f1"], x["em"], x["r5"], -x["total_ms"]), reverse=True)
    best = dict(arr[0])
    sp = Path(best["summary_path"])
    run_dir = sp.parent
    logs_dir = run_dir / "logs"
    sidecar_cfg = None
    ts = str(best.get("run_timestamp", "") or "")
    if ts:
        cand = logs_dir / f"config_{ts}.json"
        if cand.exists():
            sidecar_cfg = cand
    if sidecar_cfg is None:
        cfgs = sorted(logs_dir.glob("config_*.json"))
        if cfgs:
            sidecar_cfg = cfgs[-1]
    sidecar_result = None
    if ts:
        cand = logs_dir / f"result_{ts}.json"
        if cand.exists():
            sidecar_result = cand
    if sidecar_result is None:
        rs = sorted(logs_dir.glob("result_*.json"))
        if rs:
            sidecar_result = rs[-1]

    round_root = None
    token = "/runs/"
    if token in str(sp):
        round_root = str(sp).split(token, 1)[0]
    entry = dict(best)
    entry["run_dir"] = str(run_dir.resolve())
    entry["logs_dir"] = str(logs_dir.resolve())
    entry["rag_query_results_path"] = str((run_dir / "rag_query_results.jsonl").resolve())
    entry["sidecar_config_path"] = str(sidecar_cfg.resolve()) if sidecar_cfg else ""
    entry["sidecar_result_path"] = str(sidecar_result.resolve()) if sidecar_result else ""
    entry["config_history_path"] = str((logs_dir / "config_history.jsonl").resolve())
    entry["result_history_path"] = str((logs_dir / "result_history.jsonl").resolve())
    entry["round_root"] = str(round_root or "")
    if round_root:
        rr = Path(round_root)
        entry["round_metrics_json"] = str((rr / "round_metrics.json").resolve())
        entry["round_metrics_md"] = str((rr / "round_metrics.md").resolve())
        entry["run_records_tsv"] = str((rr / "run_records.tsv").resolve())
    champions[ds] = entry

for must in ("hotpotqa", "2wikimultihopqa"):
    if must in datasets and must not in champions:
        raise SystemExit(f"missing champion for dataset={must}")

payload = {
    "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "datasets": datasets,
    "champions": champions,
}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

headers = ["dataset", "canonical_variant_name", "retrieval_mode", "R@5", "EM", "F1", "retrieval_ms", "total_ms", "summary_path", "sidecar_config_path"]
md = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
with Path(out_tsv).open("w", encoding="utf-8") as f:
    f.write("dataset\tsummary_path\tsidecar_config_path\tretrieval_mode\tcanonical_variant_name\tr5\tem\tf1\tretrieval_ms\ttotal_ms\tfallback_rate\n")
    for ds in datasets:
        row = champions.get(ds)
        if not row:
            continue
        md.append(
            "| " + " | ".join(
                [
                    ds,
                    str(row.get("canonical_variant_name", "")),
                    str(row.get("retrieval_mode", "")),
                    f"{_f(row.get('r5', 0.0)):.4f}",
                    f"{_f(row.get('em', 0.0)):.4f}",
                    f"{_f(row.get('f1', 0.0)):.4f}",
                    f"{_f(row.get('retrieval_ms', 0.0)):.2f}",
                    f"{_f(row.get('total_ms', 0.0)):.2f}",
                    str(row.get("summary_path", "")),
                    str(row.get("sidecar_config_path", "")),
                ]
            )
            + " |"
        )
        f.write(
            "\t".join(
                [
                    ds,
                    str(row.get("summary_path", "")),
                    str(row.get("sidecar_config_path", "")),
                    str(row.get("retrieval_mode", "")),
                    str(row.get("canonical_variant_name", "")),
                    str(_f(row.get("r5", 0.0))),
                    str(_f(row.get("em", 0.0))),
                    str(_f(row.get("f1", 0.0))),
                    str(_f(row.get("retrieval_ms", 0.0))),
                    str(_f(row.get("total_ms", 0.0))),
                    str(_f(row.get("fallback_rate", 0.0))),
                ]
            )
            + "\n"
        )
Path(out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
print(out_json)
PY

  while IFS=$'\t' read -r ds summary_path sidecar_cfg mode cano r5 em f1 retrieval_ms total_ms fallback; do
    if [[ "${ds}" == "dataset" ]]; then
      continue
    fi
    CHAMPION_SUMMARY_BY_DATASET["${ds}"]="${summary_path}"
    CHAMPION_SIDECAR_CFG_BY_DATASET["${ds}"]="${sidecar_cfg}"
  done < "${CHAMPION_MATRIX_TSV}"
}

variant_spec_fields() {
  local dataset="$1"
  local variant="$2"
  case "${variant}" in
    baseline_mid_reconfirm)
      echo "baseline_mid|baseline|{}|score"
      ;;
    ref_reconfirm)
      if [[ "${dataset}" == "hotpotqa" ]]; then
        echo "strong_ref_hotpot|r2_plus_path_preserve|{\"candidate_top_t\":30,\"semantic_topn_chunk\":25,\"graph_reserve_topn\":25}|score"
      elif [[ "${dataset}" == "2wikimultihopqa" ]]; then
        echo "strong_ref_2wiki|r2_plus_path_preserve|{\"semantic_topn_chunk\":25}|score"
      else
        die "Unsupported dataset for ref_reconfirm: ${dataset}"
      fi
      ;;
    champion_config)
      echo "champion|champion|{}|score"
      ;;
    *)
      die "Unsupported variant: ${variant}"
      ;;
  esac
}

create_cfg_from_sidecar() {
  local out_cfg="$1"
  local dataset="$2"
  local variant="$3"
  local sidecar_config_json="$4"

  [[ -f "${sidecar_config_json}" ]] || die "Missing champion sidecar config: ${sidecar_config_json}"
  mkdir -p "$(dirname "${out_cfg}")"
  "${PYTHON_BIN}" - "${out_cfg}" "${dataset}" "${variant}" "${sidecar_config_json}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

out_cfg, dataset, variant, sidecar = sys.argv[1:5]
blob = json.loads(Path(sidecar).read_text(encoding="utf-8"))
cfg = dict(blob.get("config", {}) or {})
if not cfg:
    raise SystemExit(f"invalid_sidecar_config:{sidecar}")

# Strict replay for champion runs:
# preserve historical sidecar config as-is to avoid condition drift.
if "dataset" not in cfg:
    cfg["dataset"] = str(dataset)
if "canonical_variant_name" not in cfg:
    cfg["canonical_variant_name"] = str(variant)

Path(out_cfg).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
PY
}

validate_champion_cfg_replay() {
  local generated_cfg="$1"
  local sidecar_config_json="$2"
  "${PYTHON_BIN}" - "${generated_cfg}" "${sidecar_config_json}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

generated_cfg, sidecar_json = sys.argv[1:3]
g = yaml.safe_load(Path(generated_cfg).read_text(encoding="utf-8")) or {}
s_blob = json.loads(Path(sidecar_json).read_text(encoding="utf-8"))
s = dict(s_blob.get("config", {}) or {})
if not s:
    raise SystemExit(f"invalid_sidecar_config:{sidecar_json}")

# Allow only backfill when missing from historical sidecar.
if "dataset" not in s and "dataset" in g:
    s["dataset"] = g["dataset"]
if "canonical_variant_name" not in s and "canonical_variant_name" in g:
    s["canonical_variant_name"] = g["canonical_variant_name"]

if g != s:
    bad = []
    keys = sorted(set(g.keys()) | set(s.keys()))
    for k in keys:
        if g.get(k) != s.get(k):
            bad.append({"key": k, "generated": g.get(k), "sidecar": s.get(k)})
    print("champion_config_drift_detected")
    print(json.dumps(bad[:40], ensure_ascii=False, indent=2))
    raise SystemExit(2)
print("champion_config_replay_ok")
PY
}

create_cfg_from_variant() {
  local out_cfg="$1"
  local dataset="$2"
  local variant="$3"
  local retrieval_mode="$4"
  local knob_json="$5"
  local order_strategy="$6"
  local corpus_path="$7"
  mkdir -p "$(dirname "${out_cfg}")"
  "${PYTHON_BIN}" - \
    "${RAG_BASE_CFG}" \
    "${out_cfg}" \
    "${dataset}" \
    "${variant}" \
    "${retrieval_mode}" \
    "${knob_json}" \
    "${order_strategy}" \
    "${MODEL_NAME}" \
    "${UNIVERSAL_EMBED_MAX_LEN}" \
    "${UNIVERSAL_EMBED_MAX_CHARS}" \
    "${corpus_path}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

base_cfg, out_cfg, dataset, variant, retrieval_mode, knob_json, order_strategy, model_name, embed_len, embed_chars, corpus_path = sys.argv[1:12]
cfg = yaml.safe_load(Path(base_cfg).read_text(encoding="utf-8")) or {}
knobs = json.loads(knob_json or "{}")

cfg["dataset"] = str(dataset)
cfg["canonical_variant_name"] = str(variant)
cfg["retrieval_objective_mode"] = str(retrieval_mode)
cfg["order_strategy"] = str(order_strategy)
cfg["embedding_max_length"] = int(embed_len)
cfg["embedding_text_max_chars"] = int(embed_chars)
cfg["global_corpus_path"] = str(corpus_path)

cfg["shared_budget_profile"] = "off"
cfg["shared_budget_proposal_step"] = 0
cfg["shared_budget_exploration_step"] = 0
cfg["proposal_union_experiment_mode"] = "off"
cfg["chunk_node_enabled_in_diffusion"] = False
cfg["run_qa"] = True
cfg["retrieval_only"] = False
cfg["generator"] = "vllm"
cfg["model_name"] = str(model_name)
cfg["openie_mode"] = "llm"
cfg["openie_model_name"] = str(model_name)
cfg["openie_local_files_only"] = True
cfg["evaluator_mode"] = "hipporag2_parity"
cfg["num_workers"] = 1
cfg["force_rebuild_graph_index"] = False
cfg["stagewise_loss_funnel_enabled"] = True
cfg["oracle_support_injection_enabled"] = False

for k, v in (knobs or {}).items():
    cfg[k] = v

Path(out_cfg).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
PY
}

run_failed_count=0

run_combo() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local profile="$4"

  local qa_path="${QA_PATH_BY_DATASET[$dataset]}"
  local corpus_path="${CORPUS_PATH_BY_DATASET[$dataset]}"
  local champion_summary="${CHAMPION_SUMMARY_BY_DATASET[$dataset]:-}"
  local champion_sidecar="${CHAMPION_SIDECAR_CFG_BY_DATASET[$dataset]:-}"

  local cfg_dir="${CFG_ROOT}/${stage}/${variant}/${dataset}"
  local cfg_path="${cfg_dir}/rag.yaml"
  local out_dir="${RUN_ROOT}/${stage}/${variant}/rag"
  local log_path="${LOG_ROOT}/${stage}__${variant}__${dataset}.log"
  mkdir -p "${cfg_dir}" "${out_dir}" "$(dirname "${log_path}")"

  if [[ "${variant}" == "champion_config" ]]; then
    create_cfg_from_sidecar "${cfg_path}" "${dataset}" "${variant}" "${champion_sidecar}"
    validate_champion_cfg_replay "${cfg_path}" "${champion_sidecar}" >/dev/null
  else
    local spec mode knobs order
    spec="$(variant_spec_fields "${dataset}" "${variant}")"
    IFS='|' read -r _profile mode knobs order <<< "${spec}"
    create_cfg_from_variant "${cfg_path}" "${dataset}" "${variant}" "${mode}" "${knobs}" "${order}" "${corpus_path}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${stage}" "${variant}" "${dataset}" "${profile}" \
    "$([[ "${variant}" == "champion_config" ]] && echo "champion_sidecar" || echo "canonical_variant")" \
    "${qa_path}" "${corpus_path}" "${N_SAMPLES}" "${champion_summary}" "${champion_sidecar}" >> "${MANIFEST_MATRIX_TSV}"

  if [[ "${SKIP_COMPLETED}" == "true" ]]; then
    local existing_summary
    existing_summary="$(find_latest_summary "${out_dir}" "${dataset}")"
    if [[ -n "${existing_summary}" ]] && is_complete_summary "${existing_summary}" "${N_SAMPLES}" >/dev/null 2>&1; then
      log_msg "[SKIP] stage=${stage} variant=${variant} dataset=${dataset}"
      upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
        "${qa_path}" "${corpus_path}" "${existing_summary}" "${cfg_path}" "${log_path}" \
        "skipped" "" "" "0" "${N_SAMPLES}" ""
      return 0
    fi
  fi

  local start_epoch end_epoch elapsed start_ts end_ts
  start_epoch="$(date +%s)"
  start_ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  local -a cmd
  if [[ "${variant}" == "champion_config" ]]; then
    # Strict champion replay: do not force CLI overrides that can alter historical behavior.
    cmd=(
      "${PYTHON_BIN}" -m effirag.run_rag
      --config "${cfg_path}"
      --dataset "${dataset}"
      --data-path "${qa_path}"
      --limit "${N_SAMPLES}"
      --output-dir "${out_dir}"
      --timestamp-output true
    )
  else
    cmd=(
      "${PYTHON_BIN}" -m effirag.run_rag
      --config "${cfg_path}"
      --dataset "${dataset}"
      --data-path "${qa_path}"
      --graph-cache-dir "${GRAPH_CACHE_DIR}"
      --force-rebuild-graph-index false
      --num-workers 1
      --openie-mode llm
      --openie-model-name "${MODEL_NAME}"
      --openie-local-files-only true
      --openie-api-base-url "${LLM_BASE_URL}"
      --openie-api-key "${LLM_API_KEY}"
      --generator vllm
      --model-name "${MODEL_NAME}"
      --llm-base-url "${LLM_BASE_URL}"
      --llm-api-key "${LLM_API_KEY}"
      --llm-max-new-tokens 64
      --run-qa true
      --retrieval-only false
      --evaluator-mode hipporag2_parity
      --limit "${N_SAMPLES}"
      --output-dir "${out_dir}"
      --timestamp-output true
    )
  fi

  log_msg "[RUN] stage=${stage} variant=${variant} dataset=${dataset}"
  {
    echo "[COMMAND] ${cmd[*]}"
    echo "[QA_PATH] ${qa_path}"
    echo "[CORPUS_PATH] ${corpus_path}"
    echo "[CHAMPION_SUMMARY] ${champion_summary}"
    echo "[CHAMPION_SIDECAR] ${champion_sidecar}"
  } >> "${log_path}"

  set +e
  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
  local rc=${PIPESTATUS[0]}
  set -e

  end_epoch="$(date +%s)"
  end_ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  elapsed="$((end_epoch - start_epoch))"

  if [[ ${rc} -ne 0 ]]; then
    log_msg "[FAIL] stage=${stage} variant=${variant} dataset=${dataset} rc=${rc}"
    upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
      "${qa_path}" "${corpus_path}" "" "${cfg_path}" "${log_path}" \
      "failed" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" "exit_code_${rc}"
    run_failed_count=$((run_failed_count + 1))
    if [[ "${STOP_ON_FAILURE}" == "true" ]]; then
      exit ${rc}
    fi
    return ${rc}
  fi

  local summary_path
  summary_path="$(find_latest_summary "${out_dir}" "${dataset}")"
  if [[ -z "${summary_path}" ]]; then
    upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
      "${qa_path}" "${corpus_path}" "" "${cfg_path}" "${log_path}" \
      "failed" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" "missing_summary"
    run_failed_count=$((run_failed_count + 1))
    if [[ "${STOP_ON_FAILURE}" == "true" ]]; then
      exit 2
    fi
    return 2
  fi

  upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
    "${qa_path}" "${corpus_path}" "${summary_path}" "${cfg_path}" "${log_path}" \
    "ok" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" ""
  return 0
}

metric_roundtrip_sanity() {
  log_msg "[Stage 2] metric roundtrip sanity"
  "${PYTHON_BIN}" - "${RUN_RECORDS_TSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
required = [
    "recall_at_5",
    "hit_at_5",
    "mrr_at_10",
    "ndcg_at_10",
    "supporting_fact_f1",
    "context_precision",
    "faithfulness",
]
rows = []
with path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row.get("status") in {"ok", "skipped"} and row.get("summary_path"):
            rows.append(row)
if not rows:
    raise SystemExit("no_success_rows")
for row in rows:
    sp = Path(row["summary_path"])
    if not sp.exists():
        raise SystemExit(f"missing_summary:{sp}")
    summary = json.loads(sp.read_text(encoding="utf-8"))
    for key in required:
        if key not in summary:
            raise SystemExit(f"missing_metric:{key}:{sp}")
print("metric_roundtrip_ok")
PY
}

collect_failures() {
  "${PYTHON_BIN}" - "${RUN_RECORDS_TSV}" "${FAILURES_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

records_path, out_path = sys.argv[1:3]
failures = []
with open(records_path, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        status = str(row.get("status", "") or "")
        if status not in {"ok", "skipped"}:
            failures.append(row)
Path(out_path).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
print(out_path)
PY
}

log_msg "[Stage 0] preflight"
"${PYTHON_BIN}" -m py_compile effirag/*.py
PYTHONPATH=. "${PYTHON_BIN}" -m effirag.config_audit configs/canonical
for ds in "${SELECTED_DATASETS[@]}"; do
  resolve_dataset_paths "${ds}"
  log_msg "[PATH] dataset=${ds} qa=${QA_PATH_BY_DATASET[$ds]} corpus=${CORPUS_PATH_BY_DATASET[$ds]}"
done
write_environment_snapshot >/dev/null
discover_champions

log_msg "[Stage 1] sequential runs"
for ds in "${SELECTED_DATASETS[@]}"; do
  for variant in "${SELECTED_VARIANTS[@]}"; do
    run_combo "s1" "${variant}" "${ds}" "${variant}" || true
  done
done

collect_failures >/dev/null
metric_roundtrip_sanity

log_msg "[Stage 2] aggregate tables"
"${PYTHON_BIN}" scripts/aggregate_metric_round.py --round-root "${ROUND_ROOT}" --print-tables true

[[ -f "${METRIC_JSON}" ]] || die "missing metric_round_metrics.json"
[[ -f "${METRIC_MD}" ]] || die "missing metric_round_metrics.md"
[[ -f "${METRIC_DELTA_MD}" ]] || die "missing metric_delta_table.md"
[[ -f "${METRIC_EFF_MD}" ]] || die "missing metric_efficiency_table.md"
[[ -f "${METRIC_DIAG_MD}" ]] || die "missing metric_diagnostic_table.md"
[[ -f "${METRIC_DEFS_MD}" ]] || die "missing metric_definition_notes.md"

log_msg "Round complete: ${ROUND_ROOT}"
log_msg "champion_manifest.json: ${CHAMPION_MANIFEST_JSON}"
log_msg "metric_round_metrics.json: ${METRIC_JSON}"
log_msg "run_records.tsv: ${RUN_RECORDS_TSV}"
log_msg "failures.json: ${FAILURES_JSON}"
