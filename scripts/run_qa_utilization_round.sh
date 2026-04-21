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

DATASETS="hotpotqa,2wikimultihopqa,musique,popqa"
VARIANTS="answer_normalization_light,answer_verification_light,answer_type_aware_extraction"
N_SAMPLES="1000"
QA_ROOT="data/qa"
CORPUS_ROOT="/home/ojungii/HippoRAG2/dataset"
OUTPUT_ROOT="outputs/qa_utilization_round"
GRAPH_CACHE_DIR="outputs/index_cache"
LLM_BASE_URL="http://localhost:8011/v1"
LLM_API_KEY="EMPTY"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
SEQUENTIAL="true"
SKIP_COMPLETED="true"
STOP_ON_FAILURE="false"
RUN_PREFLIGHT="true"
MANIFEST_ONLY="false"
RESUME_ROUND_ROOT=""

# Dataset path overrides.
DATA_PATH_HOTPOTQA=""
DATA_PATH_2WIKIMULTIHOPQA=""
DATA_PATH_MUSIQUE=""
DATA_PATH_POPQA=""
CORPUS_PATH_HOTPOTQA=""
CORPUS_PATH_2WIKIMULTIHOPQA=""
CORPUS_PATH_MUSIQUE=""
CORPUS_PATH_POPQA=""

UNIVERSAL_EMBED_MAX_LEN=256
UNIVERSAL_EMBED_MAX_CHARS=800
RAG_BASE_CFG="configs/canonical/rag_entity_first_chunk_grounded.yaml"

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
Usage: bash scripts/run_qa_utilization_round.sh [options]

Main options:
  --datasets CSV                               (default: hotpotqa,2wikimultihopqa,musique,popqa)
  --variants CSV                               (default: answer_normalization_light,answer_verification_light,answer_type_aware_extraction)
  --n-samples N                                (default: 1000)
  --qa-root PATH                               (default: data/qa)
  --corpus-root PATH                           (default: /home/ojungii/HippoRAG2/dataset)
  --output-root PATH                           (default: outputs/qa_utilization_round)
  --graph-cache-dir PATH                       (default: outputs/index_cache)
  --llm-base-url URL                           (default: http://localhost:8011/v1)
  --llm-api-key KEY                            (default: EMPTY)
  --model-name NAME                            (default: Qwen/Qwen2.5-7B-Instruct)
  --sequential [true|false]
  --skip-completed [true|false]
  --stop-on-failure [true|false]
  --run-preflight [true|false]
  --manifest-only
  --resume-round-root PATH

Dataset overrides:
  --data-path-hotpotqa PATH
  --data-path-2wikimultihopqa PATH
  --data-path-musique PATH
  --data-path-popqa PATH
  --corpus-path-hotpotqa PATH
  --corpus-path-2wikimultihopqa PATH
  --corpus-path-musique PATH
  --corpus-path-popqa PATH
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --qa-root) QA_ROOT="$2"; shift 2 ;;
    --corpus-root) CORPUS_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --graph-cache-dir) GRAPH_CACHE_DIR="$2"; shift 2 ;;
    --llm-base-url) LLM_BASE_URL="$2"; shift 2 ;;
    --llm-api-key) LLM_API_KEY="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --sequential)
      if [[ $# -gt 1 && "${2:-}" != --* ]]; then
        SEQUENTIAL="$(parse_bool "$2")"; shift 2
      else
        SEQUENTIAL="true"; shift 1
      fi
      ;;
    --skip-completed)
      if [[ $# -gt 1 && "${2:-}" != --* ]]; then
        SKIP_COMPLETED="$(parse_bool "$2")"; shift 2
      else
        SKIP_COMPLETED="true"; shift 1
      fi
      ;;
    --stop-on-failure)
      if [[ $# -gt 1 && "${2:-}" != --* ]]; then
        STOP_ON_FAILURE="$(parse_bool "$2")"; shift 2
      else
        STOP_ON_FAILURE="true"; shift 1
      fi
      ;;
    --run-preflight)
      if [[ $# -gt 1 && "${2:-}" != --* ]]; then
        RUN_PREFLIGHT="$(parse_bool "$2")"; shift 2
      else
        RUN_PREFLIGHT="true"; shift 1
      fi
      ;;
    --manifest-only|--dry-run) MANIFEST_ONLY="true"; shift 1 ;;
    --resume-round-root) RESUME_ROUND_ROOT="$2"; shift 2 ;;

    --data-path-hotpotqa) DATA_PATH_HOTPOTQA="$2"; shift 2 ;;
    --data-path-2wikimultihopqa) DATA_PATH_2WIKIMULTIHOPQA="$2"; shift 2 ;;
    --data-path-musique) DATA_PATH_MUSIQUE="$2"; shift 2 ;;
    --data-path-popqa) DATA_PATH_POPQA="$2"; shift 2 ;;
    --corpus-path-hotpotqa) CORPUS_PATH_HOTPOTQA="$2"; shift 2 ;;
    --corpus-path-2wikimultihopqa) CORPUS_PATH_2WIKIMULTIHOPQA="$2"; shift 2 ;;
    --corpus-path-musique) CORPUS_PATH_MUSIQUE="$2"; shift 2 ;;
    --corpus-path-popqa) CORPUS_PATH_POPQA="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

SEQUENTIAL="$(parse_bool "${SEQUENTIAL}")"
SKIP_COMPLETED="$(parse_bool "${SKIP_COMPLETED}")"
STOP_ON_FAILURE="$(parse_bool "${STOP_ON_FAILURE}")"
RUN_PREFLIGHT="$(parse_bool "${RUN_PREFLIGHT}")"

[[ -n "${SEQUENTIAL}" ]] || die "Invalid --sequential"
[[ -n "${SKIP_COMPLETED}" ]] || die "Invalid --skip-completed"
[[ -n "${STOP_ON_FAILURE}" ]] || die "Invalid --stop-on-failure"
[[ -n "${RUN_PREFLIGHT}" ]] || die "Invalid --run-preflight"

if [[ "${SEQUENTIAL}" != "true" ]]; then
  log_msg "[WARN] sequential-only runner; forcing --sequential true"
  SEQUENTIAL="true"
fi

split_csv "${DATASETS}" SELECTED_DATASETS
split_csv "${VARIANTS}" SELECTED_VARIANTS
[[ ${#SELECTED_DATASETS[@]} -gt 0 ]] || die "--datasets resolved to empty"
[[ ${#SELECTED_VARIANTS[@]} -gt 0 ]] || die "--variants resolved to empty"

VALID_DATASETS=("hotpotqa" "2wikimultihopqa" "musique" "popqa")
VALID_VARIANTS=("answer_normalization_light" "answer_verification_light" "answer_type_aware_extraction")
PRIMARY_DATASETS=("hotpotqa" "2wikimultihopqa")
HOLDOUT_DATASETS=("musique" "popqa")

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
  contains_item "${ds}" "${VALID_DATASETS[@]}" || die "Unsupported dataset '${ds}'. Supported: ${VALID_DATASETS[*]}"
done
for v in "${SELECTED_VARIANTS[@]}"; do
  contains_item "${v}" "${VALID_VARIANTS[@]}" || die "Unsupported variant '${v}'. Supported: ${VALID_VARIANTS[*]}"
done

for ds in "${PRIMARY_DATASETS[@]}"; do
  contains_item "${ds}" "${SELECTED_DATASETS[@]}" || die "Primary dataset '${ds}' must be included in --datasets"
done

if [[ -n "${RESUME_ROUND_ROOT}" ]]; then
  ROUND_ROOT="${RESUME_ROUND_ROOT}"
  ROUND_SOURCE="resume"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="${OUTPUT_ROOT}/${RUN_STAMP}"
  ROUND_SOURCE="new"
fi

LOG_ROOT="logs/qa_utilization_round/$(basename "${ROUND_ROOT}")"
CFG_ROOT="${ROUND_ROOT}/configs"
RUN_ROOT="${ROUND_ROOT}/runs"
mkdir -p "${ROUND_ROOT}" "${LOG_ROOT}" "${CFG_ROOT}" "${RUN_ROOT}"

RUN_MANIFEST_JSON="${ROUND_ROOT}/run_manifest.json"
RUN_RECORDS_TSV="${ROUND_ROOT}/run_records.tsv"
MANIFEST_MATRIX_TSV="${ROUND_ROOT}/manifest_matrix.tsv"
ENV_SNAPSHOT_JSON="${ROUND_ROOT}/environment_snapshot.json"
FAILURES_JSON="${ROUND_ROOT}/failures.json"

CHAMPION_MANIFEST_JSON="${ROUND_ROOT}/champion_manifest.json"
CHAMPION_REPRO_TABLE_MD="${ROUND_ROOT}/champion_repro_table.md"
CHAMPION_MATRIX_TSV="${ROUND_ROOT}/champion_matrix.tsv"
WINNER_SELECTION_JSON="${ROUND_ROOT}/winner_selection.json"

EVIDENCE_METRICS_JSON="${ROUND_ROOT}/evidence_round_metrics.json"
EVIDENCE_METRICS_MD="${ROUND_ROOT}/evidence_round_metrics.md"
METRIC_INTEGRITY_JSON="${ROUND_ROOT}/metric_integrity_report.json"
QUERY_DIAG_JSONL="${ROUND_ROOT}/qa_utilization_query_diagnostics_all.jsonl"
QUERY_DIAG_CSV="${ROUND_ROOT}/qa_utilization_query_diagnostics_all.csv"

ROUND_METRICS_JSON="${ROUND_ROOT}/qa_utilization_round_metrics.json"
ROUND_METRICS_MD="${ROUND_ROOT}/qa_utilization_round_metrics.md"
DELTA_TABLE_MD="${ROUND_ROOT}/qa_utilization_delta_table.md"
ANSWER_REALIZATION_DIAG_JSON="${ROUND_ROOT}/answer_realization_diagnostics.json"

if [[ ! -f "${RUN_RECORDS_TSV}" ]]; then
  printf "stage\tvariant\tdataset\tprofile\tkind\tqa_path\tcorpus_path\tsummary_path\tconfig_path\tlog_path\tstatus\tstart_ts\tend_ts\telapsed_s\tn_samples\terror_message\n" > "${RUN_RECORDS_TSV}"
fi

printf "stage\tvariant\tdataset\tprofile\tretrieval_mode\tknobs_json\torder_strategy\tqa_path\tqa_source\tcorpus_path\tcorpus_source\tn_samples\tsource_summary\n" > "${MANIFEST_MATRIX_TSV}"

declare -A QA_PATH_BY_DATASET
declare -A QA_SOURCE_BY_DATASET
declare -A CORPUS_PATH_BY_DATASET
declare -A CORPUS_SOURCE_BY_DATASET

declare -A CHAMPION_SUMMARY_BY_DATASET
declare -A CHAMPION_VARIANT_BY_DATASET
declare -A CHAMPION_MODE_BY_DATASET
declare -A CHAMPION_ORDER_BY_DATASET
declare -A CHAMPION_F1_BY_DATASET
declare -A CHAMPION_EM_BY_DATASET
declare -A CHAMPION_R5_BY_DATASET
declare -A CHAMPION_TOTAL_BY_DATASET
declare -A CHAMPION_FALLBACK_BY_DATASET

declare -A FLAG_BY_VARIANT
FLAG_BY_VARIANT["answer_normalization_light"]="qa_answer_normalization_light"
FLAG_BY_VARIANT["answer_verification_light"]="qa_answer_verification_light"
FLAG_BY_VARIANT["answer_type_aware_extraction"]="qa_answer_type_aware_extraction"

get_data_override() {
  local dataset="$1"
  case "${dataset}" in
    hotpotqa) echo "${DATA_PATH_HOTPOTQA}" ;;
    2wikimultihopqa) echo "${DATA_PATH_2WIKIMULTIHOPQA}" ;;
    musique) echo "${DATA_PATH_MUSIQUE}" ;;
    popqa) echo "${DATA_PATH_POPQA}" ;;
    *) echo "" ;;
  esac
}

get_corpus_override() {
  local dataset="$1"
  case "${dataset}" in
    hotpotqa) echo "${CORPUS_PATH_HOTPOTQA}" ;;
    2wikimultihopqa) echo "${CORPUS_PATH_2WIKIMULTIHOPQA}" ;;
    musique) echo "${CORPUS_PATH_MUSIQUE}" ;;
    popqa) echo "${CORPUS_PATH_POPQA}" ;;
    *) echo "" ;;
  esac
}

resolve_dataset_paths() {
  local dataset="$1"
  local qa_override corpus_override qa_path qa_source corpus_path corpus_source
  qa_override="$(get_data_override "${dataset}")"
  corpus_override="$(get_corpus_override "${dataset}")"

  if [[ -n "${qa_override}" ]]; then
    [[ -f "${qa_override}" ]] || die "QA override not found for ${dataset}: ${qa_override}"
    qa_path="${qa_override}"
    qa_source="override"
  else
    local qa_auto="${QA_ROOT}/${dataset}.json"
    local qa_fallback="${REPO_ROOT}/data/${dataset}.json"
    if [[ -f "${qa_auto}" ]]; then
      qa_path="${qa_auto}"; qa_source="auto_qa_root"
    elif [[ -f "${qa_fallback}" ]]; then
      qa_path="${qa_fallback}"; qa_source="auto_repo_data"
    else
      die "QA path not found for ${dataset}. Tried '${qa_auto}' and '${qa_fallback}'. Use override."
    fi
  fi

  if [[ -n "${corpus_override}" ]]; then
    [[ -f "${corpus_override}" ]] || die "Corpus override not found for ${dataset}: ${corpus_override}"
    corpus_path="${corpus_override}"
    corpus_source="override"
  else
    local corpus_auto="${CORPUS_ROOT}/${dataset}_corpus.json"
    local corpus_fallback="${REPO_ROOT}/data/${dataset}_corpus.json"
    if [[ -f "${corpus_auto}" ]]; then
      corpus_path="${corpus_auto}"; corpus_source="auto_corpus_root"
    elif [[ -f "${corpus_fallback}" ]]; then
      corpus_path="${corpus_fallback}"; corpus_source="auto_repo_data"
    else
      die "Corpus path not found for ${dataset}. Tried '${corpus_auto}' and '${corpus_fallback}'. Use override."
    fi
  fi

  QA_PATH_BY_DATASET["${dataset}"]="${qa_path}"
  QA_SOURCE_BY_DATASET["${dataset}"]="${qa_source}"
  CORPUS_PATH_BY_DATASET["${dataset}"]="${corpus_path}"
  CORPUS_SOURCE_BY_DATASET["${dataset}"]="${corpus_source}"
}

assert_cfg_race_safe() {
  local cfg_path="$1"
  "${PYTHON_BIN}" - "${cfg_path}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
if int(cfg.get('num_workers', 1) or 1) > 1:
    raise SystemExit('num_workers must stay 1 for cache/meta race safety')
if bool(cfg.get('force_rebuild_graph_index', False)):
    raise SystemExit('force_rebuild_graph_index must stay false for cache/meta race safety')
print('safe')
PY
}

find_latest_summary() {
  local out_dir="$1"
  local dataset="$2"
  local summary_name="$3"
  local search_dir="${out_dir}/${dataset}"
  if [[ ! -d "${search_dir}" ]]; then
    return 0
  fi
  find "${search_dir}" -type f -name "${summary_name}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}' || true
}

is_complete_summary() {
  local summary_path="$1"
  local expected_limit="$2"
  local query_name="$3"
  "${PYTHON_BIN}" - "${summary_path}" "${expected_limit}" "${query_name}" <<'PY'
import json
import sys
from pathlib import Path
sp = Path(sys.argv[1])
expected = int(float(sys.argv[2]))
qname = str(sys.argv[3])
if not sp.exists():
    raise SystemExit(1)
_ = json.loads(sp.read_text(encoding='utf-8'))
qp = sp.with_name(qname)
if not qp.exists():
    raise SystemExit(1)
count = 0
with qp.open('r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            count += 1
if expected > 0 and count < expected:
    raise SystemExit(1)
if expected <= 0 and count <= 0:
    raise SystemExit(1)
print(sp)
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

  "${PYTHON_BIN}" - "${RUN_RECORDS_TSV}" \
    "${stage}" "${variant}" "${dataset}" "${profile}" "${kind}" \
    "${qa_path}" "${corpus_path}" "${summary_path}" "${config_path}" "${log_path}" \
    "${status}" "${start_ts}" "${end_ts}" "${elapsed_s}" "${n_samples}" "${error_message}" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
vals = sys.argv[2:18]
(
    stage, variant, dataset, profile, kind,
    qa_path, corpus_path, summary_path, config_path, log_path,
    status, start_ts, end_ts, elapsed_s, n_samples, error_message,
) = vals
header = [
    'stage','variant','dataset','profile','kind','qa_path','corpus_path','summary_path',
    'config_path','log_path','status','start_ts','end_ts','elapsed_s','n_samples','error_message'
]
rows = []
if path.exists():
    with path.open('r', encoding='utf-8') as f:
        _ = f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < len(header):
                p.extend([''] * (len(header) - len(p)))
            rows.append(p[:len(header)])

key = (stage, variant, dataset, kind)
updated = False
for r in rows:
    if (r[0], r[1], r[2], r[4]) == key:
        r[:] = [
            stage, variant, dataset, profile, kind, qa_path, corpus_path, summary_path,
            config_path, log_path, status, start_ts, end_ts, elapsed_s, n_samples, error_message,
        ]
        updated = True
        break
if not updated:
    rows.append([
        stage, variant, dataset, profile, kind, qa_path, corpus_path, summary_path,
        config_path, log_path, status, start_ts, end_ts, elapsed_s, n_samples, error_message,
    ])

path.parent.mkdir(parents=True, exist_ok=True)
with path.open('w', encoding='utf-8') as f:
    f.write('\t'.join(header) + '\n')
    for r in rows:
        f.write('\t'.join(r) + '\n')
PY
}

append_manifest_row() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local profile="$4"
  local retrieval_mode="$5"
  local knobs_json="$6"
  local order_strategy="$7"
  local qa_path="$8"
  local qa_source="$9"
  local corpus_path="${10}"
  local corpus_source="${11}"
  local n_samples="${12}"
  local source_summary="${13}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${stage}" "${variant}" "${dataset}" "${profile}" "${retrieval_mode}" "${knobs_json}" "${order_strategy}" \
    "${qa_path}" "${qa_source}" "${corpus_path}" "${corpus_source}" "${n_samples}" "${source_summary}" >> "${MANIFEST_MATRIX_TSV}"
}

write_environment_snapshot() {
  "${PYTHON_BIN}" - "${ENV_SNAPSHOT_JSON}" \
    "${LLM_BASE_URL}" "${LLM_API_KEY}" "${MODEL_NAME}" \
    "${DATASETS}" "${VARIANTS}" "${N_SAMPLES}" <<'PY'
import json
import subprocess
import sys

out_path, llm_base_url, llm_api_key, model_name, datasets_csv, variants_csv, n_samples = sys.argv[1:8]

def _cmd(args):
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return ''

payload = {
    'python_version': sys.version,
    'git_commit_hash': _cmd(['git', 'rev-parse', 'HEAD']),
    'active_branch': _cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
    'model_name': model_name,
    'llm_base_url': llm_base_url,
    'llm_api_key': llm_api_key if llm_api_key == 'EMPTY' else '***',
    'selected_datasets': [x for x in datasets_csv.split(',') if x],
    'selected_variants': [x for x in variants_csv.split(',') if x],
    'n_samples': int(float(n_samples)),
}
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(out_path)
PY
}

write_manifest() {
  "${PYTHON_BIN}" - \
    "${RUN_MANIFEST_JSON}" \
    "${MANIFEST_MATRIX_TSV}" \
    "${ROUND_ROOT}" \
    "${ROUND_SOURCE}" \
    "${QA_ROOT}" \
    "${CORPUS_ROOT}" \
    "${OUTPUT_ROOT}" \
    "${GRAPH_CACHE_DIR}" \
    "${DATASETS}" \
    "${VARIANTS}" \
    "${N_SAMPLES}" \
    "${SEQUENTIAL}" \
    "${LLM_BASE_URL}" \
    "${MODEL_NAME}" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
(
    out_json,
    matrix_tsv,
    round_root,
    round_source,
    qa_root,
    corpus_root,
    output_root,
    graph_cache_dir,
    datasets_csv,
    variants_csv,
    n_samples,
    sequential,
    llm_base_url,
    model_name,
) = sys.argv[1:15]

rows = []
with open(matrix_tsv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        if not row:
            continue
        row['n_samples'] = int(float(row.get('n_samples') or 0))
        rows.append(row)

datasets = [x.strip() for x in datasets_csv.split(',') if x.strip()]
variants = [x.strip() for x in variants_csv.split(',') if x.strip()]

dataset_paths = {}
for row in rows:
    ds = str(row.get('dataset', ''))
    dataset_paths[ds] = {
        'qa_path': row.get('qa_path', ''),
        'qa_source': row.get('qa_source', ''),
        'corpus_path': row.get('corpus_path', ''),
        'corpus_source': row.get('corpus_source', ''),
    }

payload = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'round_root': str(Path(round_root).resolve()),
    'round_source': str(round_source),
    'selection': {
        'datasets': datasets,
        'variants': variants,
        'n_samples': int(float(n_samples)),
        'sequential': bool(str(sequential).lower() == 'true'),
    },
    'paths': {
        'qa_root': qa_root,
        'corpus_root': corpus_root,
        'output_root': output_root,
        'graph_cache_dir': graph_cache_dir,
    },
    'llm': {
        'base_url': llm_base_url,
        'model_name': model_name,
    },
    'dataset_paths': dataset_paths,
    'matrix': rows,
}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(out_json)
PY
}

prompt_sanity_check() {
  log_msg "[Stage 0] preflight: prompt sanity"
  "${PYTHON_BIN}" - <<'PY'
from effirag.generator import _build_qa_prompt
from effirag.types import RenderedContext, Sample

sample = Sample(qid='sanity', question='Who wrote Hamlet?', answer='William Shakespeare', contexts=[])
rendered = RenderedContext(
    sample_id='sanity',
    method='effirag',
    text='[1] Hamlet was written by William Shakespeare.',
    sentences=['Hamlet was written by William Shakespeare.'],
    sentence_ids=['sid1'],
    truncated=False,
    metadata={'strategy_flags': ['qa_answer_normalization_light']},
)
prompt = _build_qa_prompt(sample=sample, rendered=rendered)
if 'Final answer:' not in prompt:
    raise SystemExit('prompt_sanity_failed')
print('prompt_sanity_ok')
PY
}

run_preflight_checks() {
  log_msg "[Stage 0] preflight: python compile"
  "${PYTHON_BIN}" -m py_compile effirag/*.py

  log_msg "[Stage 0] preflight: config audit"
  PYTHONPATH=. "${PYTHON_BIN}" -m effirag.config_audit configs/canonical

  log_msg "[Stage 0] preflight: dataset path resolve"
  for ds in "${SELECTED_DATASETS[@]}"; do
    resolve_dataset_paths "${ds}"
    log_msg "[PATH] dataset=${ds} qa=${QA_PATH_BY_DATASET[$ds]} (${QA_SOURCE_BY_DATASET[$ds]}) corpus=${CORPUS_PATH_BY_DATASET[$ds]} (${CORPUS_SOURCE_BY_DATASET[$ds]})"
  done

  prompt_sanity_check
}

run_preflight_without_heavy_checks() {
  for ds in "${SELECTED_DATASETS[@]}"; do
    resolve_dataset_paths "${ds}"
  done
}

discover_champions() {
  log_msg "[Stage 0] champion discovery"
  "${PYTHON_BIN}" - \
    "${CHAMPION_MANIFEST_JSON}" \
    "${CHAMPION_REPRO_TABLE_MD}" \
    "${CHAMPION_MATRIX_TSV}" \
    "${DATASETS}" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest_json, table_md, matrix_tsv, datasets_csv = sys.argv[1:5]
sel = [x.strip() for x in datasets_csv.split(',') if x.strip()]
pat_drop = re.compile(r'(?:/D2/|upper_bound|oracle|hippo)', re.IGNORECASE)

def _safe_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)

def _extract(summary):
    rp = dict(summary.get('retrieval_params', {}) or {})
    rend = dict(summary.get('render_params', {}) or {})
    r5 = _safe_float(summary.get('supporting_fact_recall_at_5', 0.0), 0.0)
    if r5 <= 0.0:
        bag = summary.get('supporting_fact_recall_at_k', {}) or {}
        r5 = _safe_float(bag.get('5', bag.get(5, 0.0)), 0.0)
    return {
        'f1': _safe_float(summary.get('f1', summary.get('F1', 0.0)), 0.0),
        'em': _safe_float(summary.get('em', summary.get('EM', 0.0)), 0.0),
        'r5': r5,
        'total_ms': _safe_float(summary.get('total_latency_ms', summary.get('total_ms', 0.0)), 0.0),
        'fallback_rate': _safe_float(summary.get('fallback_rate', 0.0), 0.0),
        'variant': str(rp.get('canonical_variant_name', '') or ''),
        'retrieval_mode': str(rp.get('retrieval_objective_mode', 'baseline') or 'baseline'),
        'order_strategy': str(rend.get('order_strategy', summary.get('order_strategy', 'score')) or 'score'),
        'embedding_max_length': int(rp.get('embedding_max_length', 0) or 0),
        'embedding_text_max_chars': int(rp.get('embedding_text_max_chars', 0) or 0),
        'retrieval_only': bool(summary.get('retrieval_only', False)),
        'oracle_support_injection_enabled': bool(summary.get('oracle_support_injection_enabled', False)),
        'run_timestamp': str(summary.get('run_timestamp', '') or ''),
        'run_timestamp_utc': str(summary.get('run_timestamp_utc', '') or ''),
    }

cands = {ds: [] for ds in sel}
for sp in Path('outputs').rglob('rag_summary.json'):
    ptxt = str(sp)
    if pat_drop.search(ptxt):
        continue
    try:
        summary = json.loads(sp.read_text(encoding='utf-8'))
    except Exception:
        continue
    dataset = str(summary.get('dataset', '') or '')
    if dataset not in cands:
        continue
    ext = _extract(summary)
    if ext['retrieval_only']:
        continue
    if ext['oracle_support_injection_enabled']:
        continue
    if ext['embedding_max_length'] != 256 or ext['embedding_text_max_chars'] != 800:
        continue
    cands[dataset].append({
        'summary_path': str(sp.resolve()),
        **ext,
    })

champions = {}
for ds in sel:
    arr = cands.get(ds, [])
    if not arr:
        continue
    arr.sort(key=lambda x: (x['f1'], x['em'], x['r5'], -x['total_ms']), reverse=True)
    champions[ds] = arr[0]

if 'hotpotqa' not in champions or '2wikimultihopqa' not in champions:
    missing = [x for x in ('hotpotqa', '2wikimultihopqa') if x not in champions]
    raise SystemExit('missing_primary_champions:' + ','.join(missing))

payload = {
    'generated_at_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
    'selection_datasets': sel,
    'champions': champions,
    'discovery_note': 'best non-oracle QA runs under universal mid regime (256/800).',
}
Path(manifest_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

headers = ['dataset','variant','retrieval_mode','order_strategy','R@5','EM','F1','total_ms','fallback_rate','summary_path']
lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
for ds in sel:
    row = champions.get(ds)
    if not row:
        continue
    lines.append('| ' + ' | '.join([
        ds,
        str(row.get('variant', '')),
        str(row.get('retrieval_mode', '')),
        str(row.get('order_strategy', '')),
        f"{_safe_float(row.get('r5', 0.0)):.4f}",
        f"{_safe_float(row.get('em', 0.0)):.4f}",
        f"{_safe_float(row.get('f1', 0.0)):.4f}",
        f"{_safe_float(row.get('total_ms', 0.0)):.2f}",
        f"{_safe_float(row.get('fallback_rate', 0.0)):.4f}",
        str(row.get('summary_path', '')),
    ]) + ' |')
Path(table_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')

with open(matrix_tsv, 'w', encoding='utf-8') as f:
    f.write('dataset\tsummary_path\tvariant\tretrieval_mode\torder_strategy\tr5\tem\tf1\ttotal_ms\tfallback_rate\n')
    for ds in sel:
        row = champions.get(ds)
        if not row:
            continue
        f.write('\t'.join([
            ds,
            str(row.get('summary_path', '')),
            str(row.get('variant', '')),
            str(row.get('retrieval_mode', '')),
            str(row.get('order_strategy', 'score')),
            str(row.get('r5', 0.0)),
            str(row.get('em', 0.0)),
            str(row.get('f1', 0.0)),
            str(row.get('total_ms', 0.0)),
            str(row.get('fallback_rate', 0.0)),
        ]) + '\n')

print(manifest_json)
PY

  while IFS=$'\t' read -r ds summary_path variant mode order r5 em f1 total_ms fallback; do
    if [[ "${ds}" == "dataset" ]]; then
      continue
    fi
    CHAMPION_SUMMARY_BY_DATASET["${ds}"]="${summary_path}"
    CHAMPION_VARIANT_BY_DATASET["${ds}"]="${variant}"
    CHAMPION_MODE_BY_DATASET["${ds}"]="${mode}"
    CHAMPION_ORDER_BY_DATASET["${ds}"]="${order}"
    CHAMPION_R5_BY_DATASET["${ds}"]="${r5}"
    CHAMPION_EM_BY_DATASET["${ds}"]="${em}"
    CHAMPION_F1_BY_DATASET["${ds}"]="${f1}"
    CHAMPION_TOTAL_BY_DATASET["${ds}"]="${total_ms}"
    CHAMPION_FALLBACK_BY_DATASET["${ds}"]="${fallback}"
  done < "${CHAMPION_MATRIX_TSV}"
}

create_variant_cfg_from_summary() {
  local out_cfg="$1"
  local dataset="$2"
  local canonical_variant="$3"
  local summary_path="$4"
  local extra_flag="$5"

  mkdir -p "$(dirname "${out_cfg}")"
  "${PYTHON_BIN}" - \
    "${RAG_BASE_CFG}" \
    "${out_cfg}" \
    "${dataset}" \
    "${canonical_variant}" \
    "${summary_path}" \
    "${extra_flag}" \
    "${MODEL_NAME}" \
    "${UNIVERSAL_EMBED_MAX_LEN}" \
    "${UNIVERSAL_EMBED_MAX_CHARS}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

base_cfg, out_cfg, dataset, canonical_variant, summary_path, extra_flag, model_name, embed_len, embed_chars = sys.argv[1:10]
cfg = yaml.safe_load(Path(base_cfg).read_text(encoding='utf-8')) or {}

summary = {}
if str(summary_path or '').strip() and Path(summary_path).exists():
    summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))

for section_name in ('retrieval_params', 'render_params', 'generation_params'):
    section = dict(summary.get(section_name, {}) or {})
    for k, v in section.items():
        if k in cfg:
            cfg[k] = v

cfg['dataset'] = str(dataset)
cfg['canonical_variant_name'] = str(canonical_variant)
cfg['embedding_max_length'] = int(embed_len)
cfg['embedding_text_max_chars'] = int(embed_chars)

cfg['shared_budget_profile'] = 'off'
cfg['shared_budget_proposal_step'] = 0
cfg['shared_budget_exploration_step'] = 0
cfg['proposal_union_experiment_mode'] = 'off'
cfg['chunk_node_enabled_in_diffusion'] = False
cfg['run_qa'] = True
cfg['retrieval_only'] = False
cfg['generator'] = 'vllm'
cfg['model_name'] = str(model_name)
cfg['openie_mode'] = 'llm'
cfg['openie_model_name'] = str(model_name)
cfg['openie_local_files_only'] = True
cfg['evaluator_mode'] = 'hipporag2_parity'
cfg['num_workers'] = 1
cfg['force_rebuild_graph_index'] = False
cfg['stagewise_loss_funnel_enabled'] = True
cfg['oracle_support_injection_enabled'] = False

base_order = str(cfg.get('order_strategy', 'score') or 'score').strip()
flag = str(extra_flag or '').strip()
if flag:
    pieces = [x for x in base_order.split('+') if x]
    if flag not in pieces:
        pieces.append(flag)
    cfg['order_strategy'] = '+'.join(pieces)

Path(out_cfg).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding='utf-8')
PY

  assert_cfg_race_safe "${out_cfg}" >/dev/null
}

create_variant_cfg_fallback_ref() {
  local out_cfg="$1"
  local dataset="$2"
  local canonical_variant="$3"
  local extra_flag="$4"

  mkdir -p "$(dirname "${out_cfg}")"
  "${PYTHON_BIN}" - \
    "${RAG_BASE_CFG}" \
    "${out_cfg}" \
    "${dataset}" \
    "${canonical_variant}" \
    "${extra_flag}" \
    "${MODEL_NAME}" \
    "${UNIVERSAL_EMBED_MAX_LEN}" \
    "${UNIVERSAL_EMBED_MAX_CHARS}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

base_cfg, out_cfg, dataset, canonical_variant, extra_flag, model_name, embed_len, embed_chars = sys.argv[1:9]
cfg = yaml.safe_load(Path(base_cfg).read_text(encoding='utf-8')) or {}

cfg['dataset'] = str(dataset)
cfg['canonical_variant_name'] = str(canonical_variant)
cfg['retrieval_objective_mode'] = 'r2_plus_path_preserve'
cfg['order_strategy'] = 'score'
cfg['embedding_max_length'] = int(embed_len)
cfg['embedding_text_max_chars'] = int(embed_chars)

if dataset == 'hotpotqa':
    cfg['candidate_top_t'] = 30
    cfg['semantic_topn_chunk'] = 25
    cfg['graph_reserve_topn'] = 25
elif dataset == '2wikimultihopqa':
    cfg['semantic_topn_chunk'] = 25

cfg['shared_budget_profile'] = 'off'
cfg['shared_budget_proposal_step'] = 0
cfg['shared_budget_exploration_step'] = 0
cfg['proposal_union_experiment_mode'] = 'off'
cfg['chunk_node_enabled_in_diffusion'] = False
cfg['run_qa'] = True
cfg['retrieval_only'] = False
cfg['generator'] = 'vllm'
cfg['model_name'] = str(model_name)
cfg['openie_mode'] = 'llm'
cfg['openie_model_name'] = str(model_name)
cfg['openie_local_files_only'] = True
cfg['evaluator_mode'] = 'hipporag2_parity'
cfg['num_workers'] = 1
cfg['force_rebuild_graph_index'] = False
cfg['stagewise_loss_funnel_enabled'] = True
cfg['oracle_support_injection_enabled'] = False

flag = str(extra_flag or '').strip()
if flag:
    pieces = [x for x in str(cfg.get('order_strategy', 'score') or 'score').split('+') if x]
    if flag not in pieces:
        pieces.append(flag)
    cfg['order_strategy'] = '+'.join(pieces)

Path(out_cfg).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding='utf-8')
PY

  assert_cfg_race_safe "${out_cfg}" >/dev/null
}

run_failed_count=0

run_combo_with_base() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local profile="$4"
  local source_summary="$5"
  local extra_flag="$6"

  local qa_path="${QA_PATH_BY_DATASET[$dataset]}"
  local corpus_path="${CORPUS_PATH_BY_DATASET[$dataset]}"

  local cfg_dir="${CFG_ROOT}/${stage}/${variant}/${dataset}"
  local cfg_path="${cfg_dir}/rag.yaml"
  local out_dir="${RUN_ROOT}/${stage}/${variant}/rag"
  local log_path="${LOG_ROOT}/${stage}__${variant}__${dataset}.log"
  mkdir -p "${cfg_dir}" "${out_dir}" "$(dirname "${log_path}")"

  if [[ -n "${source_summary}" && -f "${source_summary}" ]]; then
    create_variant_cfg_from_summary "${cfg_path}" "${dataset}" "${variant}" "${source_summary}" "${extra_flag}"
  else
    create_variant_cfg_fallback_ref "${cfg_path}" "${dataset}" "${variant}" "${extra_flag}"
  fi

  local retrieval_mode knobs_json order_strategy
  retrieval_mode="$("${PYTHON_BIN}" - <<'PY' "${cfg_path}"
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8')) or {}
print(str(cfg.get('retrieval_objective_mode','baseline')))
PY
)"
  knobs_json="$("${PYTHON_BIN}" - <<'PY' "${cfg_path}"
import json,sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8')) or {}
keys=['candidate_top_t','semantic_topn_chunk','graph_reserve_topn']
out={k:cfg.get(k) for k in keys if k in cfg}
print(json.dumps(out, ensure_ascii=False))
PY
)"
  order_strategy="$("${PYTHON_BIN}" - <<'PY' "${cfg_path}"
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8')) or {}
print(str(cfg.get('order_strategy','score')))
PY
)"

  append_manifest_row "${stage}" "${variant}" "${dataset}" "${profile}" "${retrieval_mode}" "${knobs_json}" "${order_strategy}" \
    "${qa_path}" "${QA_SOURCE_BY_DATASET[$dataset]}" "${corpus_path}" "${CORPUS_SOURCE_BY_DATASET[$dataset]}" "${N_SAMPLES}" "${source_summary}"

  if [[ "${SKIP_COMPLETED}" == "true" ]]; then
    local existing_summary
    existing_summary="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
    if [[ -n "${existing_summary}" ]]; then
      if is_complete_summary "${existing_summary}" "${N_SAMPLES}" "rag_query_results.jsonl" >/dev/null 2>&1; then
        log_msg "[SKIP] stage=${stage} variant=${variant} dataset=${dataset}"
        upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
          "${qa_path}" "${corpus_path}" "${existing_summary}" "${cfg_path}" "${log_path}" \
          "skipped" "" "" "0" "${N_SAMPLES}" ""
        return 0
      fi
    fi
  fi

  local start_epoch end_epoch elapsed
  local start_ts end_ts
  start_epoch="$(date +%s)"
  start_ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  local -a cmd
  cmd=(
    "${PYTHON_BIN}" -m effirag.run_rag
    --config "${cfg_path}"
    --dataset "${dataset}"
    --data-path "${qa_path}"
    --global-corpus-path "${corpus_path}"
    --graph-cache-dir "${GRAPH_CACHE_DIR}"
    --force-rebuild-graph-index false
    --num-workers 1
    --openie-mode llm
    --openie-model-name "${MODEL_NAME}"
    --openie-local-files-only true
    --openie-api-base-url "${LLM_BASE_URL}"
    --openie-api-key "${LLM_API_KEY}"
    --embedding-enabled true
    --embedding-max-length "${UNIVERSAL_EMBED_MAX_LEN}"
    --embedding-text-max-chars "${UNIVERSAL_EMBED_MAX_CHARS}"
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

  log_msg "[RUN] stage=${stage} variant=${variant} dataset=${dataset} profile=${profile}"
  {
    echo "[COMMAND] ${cmd[*]}"
    echo "[SOURCE_SUMMARY] ${source_summary}"
    echo "[EXTRA_FLAG] ${extra_flag}"
    echo "[QA_PATH] ${qa_path}"
    echo "[CORPUS_PATH] ${corpus_path}"
  } >> "${log_path}"

  set +e
  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
  local rc=${PIPESTATUS[0]}
  set -e

  end_epoch="$(date +%s)"
  end_ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  elapsed="$((end_epoch - start_epoch))"

  if [[ ${rc} -ne 0 ]]; then
    log_msg "[FAIL] stage=${stage} variant=${variant} dataset=${dataset} exit_code=${rc}"
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
  summary_path="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
  if [[ -z "${summary_path}" ]]; then
    log_msg "[FAIL] stage=${stage} variant=${variant} dataset=${dataset} missing rag_summary.json"
    upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
      "${qa_path}" "${corpus_path}" "" "${cfg_path}" "${log_path}" \
      "failed" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" "missing_summary"
    run_failed_count=$((run_failed_count + 1))
    if [[ "${STOP_ON_FAILURE}" == "true" ]]; then
      exit 1
    fi
    return 1
  fi

  upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
    "${qa_path}" "${corpus_path}" "${summary_path}" "${cfg_path}" "${log_path}" \
    "ok" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" ""
  log_msg "[DONE] stage=${stage} variant=${variant} dataset=${dataset}"
  return 0
}

run_stage0_repro() {
  log_msg "[Stage 0] champion reproduction"
  local ds
  for ds in "${PRIMARY_DATASETS[@]}"; do
    local src
    src="${CHAMPION_SUMMARY_BY_DATASET[$ds]:-}"
    [[ -n "${src}" ]] || die "Missing champion summary for ${ds}"
    run_combo_with_base "s0" "champion_reconfirm" "${ds}" "champion_reconfirm" "${src}" "" || true
  done
}

run_stage1_variants() {
  log_msg "[Stage 1] qa utilization variants on champion runs"
  local ds v flag src
  for ds in "${PRIMARY_DATASETS[@]}"; do
    src="${CHAMPION_SUMMARY_BY_DATASET[$ds]:-}"
    [[ -n "${src}" ]] || die "Missing champion summary for ${ds}"
    for v in "${SELECTED_VARIANTS[@]}"; do
      flag="${FLAG_BY_VARIANT[$v]:-}"
      [[ -n "${flag}" ]] || continue
      run_combo_with_base "s1" "${v}" "${ds}" "qa_utilization" "${src}" "${flag}" || true
    done
  done
}

select_winner_variant() {
  "${PYTHON_BIN}" - "${RUN_RECORDS_TSV}" "${WINNER_SELECTION_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

records_path, out_json = sys.argv[1:3]
rows = []
with open(records_path, 'r', encoding='utf-8') as f:
    rr = csv.DictReader(f, delimiter='\t')
    for row in rr:
        if row.get('kind') != 'rag':
            continue
        if row.get('status') not in {'ok', 'skipped'}:
            continue
        sp = str(row.get('summary_path') or '').strip()
        if not sp or not Path(sp).exists():
            continue
        try:
            summary = json.loads(Path(sp).read_text(encoding='utf-8'))
        except Exception:
            continue
        rows.append({
            'stage': str(row.get('stage', '')),
            'variant': str(row.get('variant', '')),
            'dataset': str(row.get('dataset', '')),
            'summary_path': sp,
            'f1': float(summary.get('f1', summary.get('F1', 0.0)) or 0.0),
            'total_ms': float(summary.get('total_latency_ms', summary.get('total_ms', 0.0)) or 0.0),
        })

ref = {(r['dataset']): r for r in rows if r['stage'] == 's0' and r['variant'] == 'champion_reconfirm'}
cands = [r for r in rows if r['stage'] == 's1' and r['variant'] in {
    'answer_normalization_light','answer_verification_light','answer_type_aware_extraction'
}]

by_variant = {}
for r in cands:
    by_variant.setdefault(r['variant'], {})[r['dataset']] = r

scores = []
for v, ds_map in by_variant.items():
    if '2wikimultihopqa' not in ds_map:
        continue
    d2 = ds_map.get('2wikimultihopqa')
    h = ds_map.get('hotpotqa')
    ref2 = ref.get('2wikimultihopqa')
    refh = ref.get('hotpotqa')
    if not ref2:
        continue
    d2_f1 = float(d2['f1'] - ref2['f1'])
    d2_t = float(d2['total_ms'] - ref2['total_ms'])
    h_f1 = float(h['f1'] - refh['f1']) if (h and refh) else 0.0
    h_t = float(h['total_ms'] - refh['total_ms']) if (h and refh) else 0.0
    score = float(d2_f1 + 0.30 * h_f1 - 0.0002 * max(0.0, d2_t) - 0.0001 * max(0.0, h_t))
    scores.append({
        'variant': v,
        'score': score,
        'dF1_2wiki': d2_f1,
        'dF1_hotpot': h_f1,
        'dTotal_2wiki': d2_t,
        'dTotal_hotpot': h_t,
    })

scores.sort(key=lambda x: (x['score'], x['dF1_2wiki'], x['dF1_hotpot']), reverse=True)
best = scores[0]['variant'] if scores else ''

payload = {
    'winner_variant': best,
    'scores': scores,
}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(best)
PY
}

run_stage2_holdout() {
  log_msg "[Stage 2] holdout check on winner variant"
  local winner
  winner="$(select_winner_variant)"
  if [[ -z "${winner}" ]]; then
    log_msg "[Stage 2] skipped: no winner from Stage 1"
    return 0
  fi

  local flag
  flag="${FLAG_BY_VARIANT[$winner]:-}"
  if [[ -z "${flag}" ]]; then
    log_msg "[Stage 2] skipped: missing flag for winner ${winner}"
    return 0
  fi

  local ds src
  for ds in "${HOLDOUT_DATASETS[@]}"; do
    if ! contains_item "${ds}" "${SELECTED_DATASETS[@]}"; then
      continue
    fi
    src="${CHAMPION_SUMMARY_BY_DATASET[$ds]:-}"
    run_combo_with_base "s2" "${winner}" "${ds}" "holdout_winner" "${src}" "${flag}" || true
  done
}

run_failed_summary() {
  "${PYTHON_BIN}" - "${RUN_RECORDS_TSV}" "${FAILURES_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path
rows = []
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    rr = csv.DictReader(f, delimiter='\t')
    for row in rr:
        if str(row.get('status', '')) == 'failed':
            rows.append({
                'stage': str(row.get('stage', '')),
                'variant': str(row.get('variant', '')),
                'dataset': str(row.get('dataset', '')),
                'error_message': str(row.get('error_message', '')),
                'log_path': str(row.get('log_path', '')),
            })
Path(sys.argv[2]).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

aggregate_round_outputs() {
  log_msg "[Aggregate] evidence diagnostics"
  "${PYTHON_BIN}" scripts/aggregate_evidence_sufficiency.py \
    --round-root "${ROUND_ROOT}" \
    --manifest "${RUN_MANIFEST_JSON}" \
    --run-records "${RUN_RECORDS_TSV}" \
    --output-json "${EVIDENCE_METRICS_JSON}" \
    --output-md "${EVIDENCE_METRICS_MD}" \
    --metric-integrity-report "${METRIC_INTEGRITY_JSON}" \
    --query-diag-jsonl "${QUERY_DIAG_JSONL}" \
    --query-diag-csv "${QUERY_DIAG_CSV}" \
    --base-variants "champion_reconfirm"

  run_failed_summary

  log_msg "[Aggregate] qa utilization summary"
  "${PYTHON_BIN}" - \
    "${EVIDENCE_METRICS_JSON}" \
    "${CHAMPION_MANIFEST_JSON}" \
    "${WINNER_SELECTION_JSON}" \
    "${ROUND_METRICS_JSON}" \
    "${ROUND_METRICS_MD}" \
    "${DELTA_TABLE_MD}" \
    "${ANSWER_REALIZATION_DIAG_JSON}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(evi_json, champion_json, winner_json, out_json, out_md, delta_md, diag_json) = sys.argv[1:8]

evi = json.loads(Path(evi_json).read_text(encoding='utf-8'))
ch = json.loads(Path(champion_json).read_text(encoding='utf-8'))
winner = {}
if Path(winner_json).exists():
    winner = json.loads(Path(winner_json).read_text(encoding='utf-8'))

perf = list(evi.get('main_performance', []) or [])
evi_tbl = list(evi.get('evidence_sufficiency_table', []) or [])
perf_idx = {(str(r.get('dataset','')), str(r.get('variant',''))): r for r in perf}
evi_idx = {(str(r.get('dataset','')), str(r.get('variant',''))): r for r in evi_tbl}

rows = []
for key, p in perf_idx.items():
    d, v = key
    e = evi_idx.get(key, {})
    rows.append({
        'dataset': d,
        'variant': v,
        'R@5': float(p.get('R@5', 0.0) or 0.0),
        'EM': float(p.get('EM', 0.0) or 0.0),
        'F1': float(p.get('F1', 0.0) or 0.0),
        'retrieval_ms': float(p.get('retrieval_ms', 0.0) or 0.0),
        'generation_ms': float(p.get('generation_ms', 0.0) or 0.0),
        'total_ms': float(p.get('total_ms', 0.0) or 0.0),
        'fallback_rate': float(p.get('fallback_rate', 0.0) or 0.0),
        'answer_present_but_generation_fail': float(e.get('answer_present_but_generation_fail_rate', 0.0) or 0.0),
        'output_overlap_answer_bearing': float(e.get('output_overlap_answer_bearing_rate', 0.0) or 0.0),
        'exact_match_surface_correction_rate': float(p.get('exact_match_surface_correction_rate', 0.0) or 0.0),
        'evidence_supported_answer_rate': float(p.get('evidence_supported_answer_rate', 0.0) or 0.0),
        'answer_type_match_rate': float(p.get('answer_type_match_rate', 0.0) or 0.0),
    })

# Stable ordering.
rank = {('hotpotqa','champion_reconfirm'):0,('2wikimultihopqa','champion_reconfirm'):1}
def _ord(r):
    base = rank.get((r['dataset'], r['variant']), 1000)
    return (base, r['dataset'], r['variant'])
rows.sort(key=_ord)

champions = dict(ch.get('champions', {}) or {})

def _champion_row(ds):
    return next((r for r in rows if r['dataset'] == ds and r['variant'] == 'champion_reconfirm'), None)

delta_rows = []
for ds in ('hotpotqa', '2wikimultihopqa'):
    ref = _champion_row(ds)
    if ref is None:
        continue
    for r in rows:
        if r['dataset'] != ds or r['variant'] == 'champion_reconfirm':
            continue
        delta_rows.append({
            'dataset': ds,
            'variant': r['variant'],
            'dEM': float(r['EM'] - ref['EM']),
            'dF1': float(r['F1'] - ref['F1']),
            'dABGF': float(r['answer_present_but_generation_fail'] - ref['answer_present_but_generation_fail']),
            'dOverlap': float(r['output_overlap_answer_bearing'] - ref['output_overlap_answer_bearing']),
            'dTotalMs': float(r['total_ms'] - ref['total_ms']),
        })

diag_rows = []
for r in rows:
    diag_rows.append({
        'dataset': r['dataset'],
        'variant': r['variant'],
        'answer_present_but_generation_fail': r['answer_present_but_generation_fail'],
        'output_overlap_answer_bearing': r['output_overlap_answer_bearing'],
        'exact_match_surface_correction_rate': r['exact_match_surface_correction_rate'],
        'evidence_supported_answer_rate': r['evidence_supported_answer_rate'],
        'answer_type_match_rate': r['answer_type_match_rate'],
    })

out = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'round_root': str(Path(out_json).resolve().parent),
    'champion_manifest': champions,
    'winner_selection': winner,
    'main_performance': rows,
    'delta_vs_champion': delta_rows,
    'answer_realization_diagnostics': diag_rows,
    'source_evidence_metrics_json': str(Path(evi_json).resolve()),
}
Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
Path(diag_json).write_text(json.dumps({'rows': diag_rows, 'winner_selection': winner}, ensure_ascii=False, indent=2), encoding='utf-8')


def fmt(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return "0.0000"

def table(headers, body):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

lines = []
lines.append('# QA Utilization Round')
lines.append('')
lines.append('## 1. Champion Discovery')
lines.append('')
if champions:
    c_rows = []
    for ds in ('hotpotqa','2wikimultihopqa','musique','popqa'):
        c = champions.get(ds)
        if not c:
            continue
        c_rows.append([
            ds,
            str(c.get('variant','')),
            str(c.get('retrieval_mode','')),
            fmt(c.get('r5',0.0)),
            fmt(c.get('em',0.0)),
            fmt(c.get('f1',0.0)),
            fmt(c.get('total_ms',0.0),2),
        ])
    lines.append(table(['dataset','variant','retrieval_mode','R@5','EM','F1','total_ms'], c_rows))
else:
    lines.append('- champion manifest missing')
lines.append('')
lines.append('## 2. Hotpot / 2Wiki Main Results')
lines.append('')
main_rows = [r for r in rows if r['dataset'] in {'hotpotqa','2wikimultihopqa'}]
lines.append(table(
    ['dataset','variant','R@5','EM','F1','retrieval_ms','generation_ms','total_ms','fallback_rate'],
    [[r['dataset'], r['variant'], fmt(r['R@5']), fmt(r['EM']), fmt(r['F1']), fmt(r['retrieval_ms'],2), fmt(r['generation_ms'],2), fmt(r['total_ms'],2), fmt(r['fallback_rate'])] for r in main_rows]
))
lines.append('')
lines.append('## 3. Champion Delta Table')
lines.append('')
lines.append(table(
    ['dataset','variant','dEM','dF1','dABGF','dOverlap','dTotalMs'],
    [[r['dataset'], r['variant'], fmt(r['dEM']), fmt(r['dF1']), fmt(r['dABGF']), fmt(r['dOverlap']), fmt(r['dTotalMs'],2)] for r in delta_rows]
))
lines.append('')
lines.append('## 4. Answer Realization Diagnostics')
lines.append('')
lines.append(table(
    ['dataset','variant','surface_correction','evidence_supported','type_match','ABGF','overlap'],
    [[r['dataset'], r['variant'], fmt(r['exact_match_surface_correction_rate']), fmt(r['evidence_supported_answer_rate']), fmt(r['answer_type_match_rate']), fmt(r['answer_present_but_generation_fail']), fmt(r['output_overlap_answer_bearing'])] for r in diag_rows]
))
lines.append('')
lines.append('## 5. Winner Holdout Check (musique/popqa)')
lines.append('')
holdout_rows = [r for r in rows if r['dataset'] in {'musique','popqa'}]
if holdout_rows:
    lines.append(table(
        ['dataset','variant','R@5','EM','F1','retrieval_ms','generation_ms','total_ms'],
        [[r['dataset'], r['variant'], fmt(r['R@5']), fmt(r['EM']), fmt(r['F1']), fmt(r['retrieval_ms'],2), fmt(r['generation_ms'],2), fmt(r['total_ms'],2)] for r in holdout_rows]
    ))
else:
    lines.append('- holdout runs not executed')
lines.append('')
lines.append('## 6. Interpretation')
lines.append('')
lines.append('- answer_normalization_light: surface-form mismatch correction strength is reflected in surface_correction and evidence_supported rates.')
lines.append('- answer_verification_light: useful when ABGF drops without large latency increase.')
lines.append('- answer_type_aware_extraction: useful when answer_type_match rises with stable or improved F1.')

Path(out_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
Path(delta_md).write_text(table(
    ['dataset','variant','dEM','dF1','dABGF','dOverlap','dTotalMs'],
    [[r['dataset'], r['variant'], fmt(r['dEM']), fmt(r['dF1']), fmt(r['dABGF']), fmt(r['dOverlap']), fmt(r['dTotalMs'],2)] for r in delta_rows]
) + '\n', encoding='utf-8')

print(out_json)
PY
}

check_cache_meta_race() {
  log_msg "[CHECK] cache/meta race guard"
  "${PYTHON_BIN}" - "${CFG_ROOT}" <<'PY'
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1])
unsafe = []
for p in root.rglob('*.yaml'):
    cfg = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    if int(cfg.get('num_workers', 1) or 1) > 1:
        unsafe.append(str(p))
    if bool(cfg.get('force_rebuild_graph_index', False)):
        unsafe.append(str(p))
if unsafe:
    raise SystemExit('unsafe cache/meta race configs:\n' + '\n'.join(sorted(set(unsafe))))
print('safe')
PY
}

main() {
  log_msg "Round root: ${ROUND_ROOT} (source=${ROUND_SOURCE})"
  log_msg "Datasets: ${DATASETS}"
  log_msg "Variants: ${VARIANTS}"

  if [[ "${RUN_PREFLIGHT}" == "true" ]]; then
    run_preflight_checks
  else
    run_preflight_without_heavy_checks
  fi

  write_environment_snapshot >/dev/null

  discover_champions

  if [[ "${MANIFEST_ONLY}" == "true" ]]; then
    log_msg "Manifest-only mode enabled; writing manifest and exiting."
    write_manifest >/dev/null
    log_msg "run_manifest.json: ${RUN_MANIFEST_JSON}"
    log_msg "champion_manifest.json: ${CHAMPION_MANIFEST_JSON}"
    exit 0
  fi

  run_stage0_repro
  run_stage1_variants
  run_stage2_holdout

  check_cache_meta_race
  write_manifest >/dev/null
  aggregate_round_outputs

  log_msg "Completed with failed_runs=${run_failed_count}"
  log_msg "champion_manifest.json: ${CHAMPION_MANIFEST_JSON}"
  log_msg "champion_repro_table.md: ${CHAMPION_REPRO_TABLE_MD}"
  log_msg "qa_utilization_round_metrics.json: ${ROUND_METRICS_JSON}"
  log_msg "qa_utilization_round_metrics.md: ${ROUND_METRICS_MD}"
  log_msg "qa_utilization_delta_table.md: ${DELTA_TABLE_MD}"
  log_msg "answer_realization_diagnostics.json: ${ANSWER_REALIZATION_DIAG_JSON}"
  log_msg "run_records.tsv: ${RUN_RECORDS_TSV}"
  log_msg "failures.json: ${FAILURES_JSON}"
  log_msg "environment_snapshot.json: ${ENV_SNAPSHOT_JSON}"
}

main "$@"
