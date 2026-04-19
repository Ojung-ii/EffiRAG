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

DATASETS_CSV="hotpotqa,2wikimultihopqa,musique,popqa"
VARIANTS_CSV="baseline_mid_reconfirm,ref_reconfirm"
N_SAMPLES=1000
QA_ROOT="data/qa"
CORPUS_ROOT="/home/ojungii/HippoRAG2/dataset"
OUTPUT_ROOT="outputs/multidataset_generalization_round"
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
  cat <<'EOF'
Usage: bash scripts/run_multidataset_generalization.sh [options]

Options:
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
  --sequential [true|false]
  --skip-completed [true|false]
  --stop-on-failure [true|false]
  --manifest-only
  --dry-run
  --resume-round-root PATH

  --data-path-hotpotqa PATH
  --data-path-2wikimultihopqa PATH
  --data-path-musique PATH
  --data-path-popqa PATH
  --corpus-path-hotpotqa PATH
  --corpus-path-2wikimultihopqa PATH
  --corpus-path-musique PATH
  --corpus-path-popqa PATH
EOF
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

[[ -n "${SEQUENTIAL}" ]] || die "Invalid --sequential value"
[[ -n "${SKIP_COMPLETED}" ]] || die "Invalid --skip-completed value"
[[ -n "${STOP_ON_FAILURE}" ]] || die "Invalid --stop-on-failure value"
[[ -n "${RUN_PREFLIGHT}" ]] || die "Invalid --run-preflight value"

if [[ "${SEQUENTIAL}" != "true" ]]; then
  log_msg "[WARN] --sequential=false requested, but this runner is sequential-only; forcing sequential=true"
  SEQUENTIAL="true"
fi

[[ -n "${LLM_BASE_URL}" ]] || die "--llm-base-url must be non-empty"
[[ -n "${MODEL_NAME}" ]] || die "--model-name must be non-empty"

split_csv "${DATASETS_CSV}" SELECTED_DATASETS
split_csv "${VARIANTS_CSV}" SELECTED_VARIANTS
[[ ${#SELECTED_DATASETS[@]} -gt 0 ]] || die "--datasets resolved to empty list"
[[ ${#SELECTED_VARIANTS[@]} -gt 0 ]] || die "--variants resolved to empty list"

VALID_DATASETS=("hotpotqa" "2wikimultihopqa" "musique" "popqa")
VALID_VARIANTS=("baseline_mid_reconfirm" "ref_reconfirm")

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

if [[ -n "${RESUME_ROUND_ROOT}" ]]; then
  ROUND_ROOT="${RESUME_ROUND_ROOT}"
  ROUND_SOURCE="resume"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="${OUTPUT_ROOT}/${RUN_STAMP}"
  ROUND_SOURCE="new"
fi

LOG_ROOT="logs/multidataset_generalization/$(basename "${ROUND_ROOT}")"
CFG_ROOT="${ROUND_ROOT}/configs"
RUN_ROOT="${ROUND_ROOT}/runs"
mkdir -p "${ROUND_ROOT}" "${LOG_ROOT}" "${CFG_ROOT}" "${RUN_ROOT}"

RUN_MANIFEST_JSON="${ROUND_ROOT}/run_manifest.json"
RUN_RECORDS_TSV="${ROUND_ROOT}/run_records.tsv"
ROUND_METRICS_JSON="${ROUND_ROOT}/round_metrics.json"
ROUND_METRICS_MD="${ROUND_ROOT}/round_metrics.md"
FAILURES_JSON="${ROUND_ROOT}/failures.json"
ENV_SNAPSHOT_JSON="${ROUND_ROOT}/environment_snapshot.json"
MANIFEST_MATRIX_TSV="${ROUND_ROOT}/manifest_matrix.tsv"

if [[ ! -f "${RUN_RECORDS_TSV}" ]]; then
  printf "stage\tvariant\tdataset\tprofile\tkind\tqa_path\tcorpus_path\tsummary_path\tconfig_path\tlog_path\tstatus\tstart_ts\tend_ts\telapsed_s\tn_samples\terror_message\n" > "${RUN_RECORDS_TSV}"
fi

printf "stage\tvariant\tdataset\tprofile\tretrieval_mode\tknobs_json\tqa_path\tqa_source\tcorpus_path\tcorpus_source\tn_samples\n" > "${MANIFEST_MATRIX_TSV}"

declare -A QA_PATH_BY_DATASET
declare -A QA_SOURCE_BY_DATASET
declare -A CORPUS_PATH_BY_DATASET
declare -A CORPUS_SOURCE_BY_DATASET

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

  local qa_override
  local corpus_override
  qa_override="$(get_data_override "${dataset}")"
  corpus_override="$(get_corpus_override "${dataset}")"

  local qa_path=""
  local qa_source=""
  local corpus_path=""
  local corpus_source=""

  if [[ -n "${qa_override}" ]]; then
    [[ -f "${qa_override}" ]] || die "QA override not found for ${dataset}: ${qa_override}"
    qa_path="${qa_override}"
    qa_source="override"
  else
    local qa_auto="${QA_ROOT}/${dataset}.json"
    local qa_fallback="${REPO_ROOT}/data/${dataset}.json"
    if [[ -f "${qa_auto}" ]]; then
      qa_path="${qa_auto}"
      qa_source="auto_qa_root"
    elif [[ -f "${qa_fallback}" ]]; then
      qa_path="${qa_fallback}"
      qa_source="auto_repo_data"
    else
      die "QA path not found for ${dataset}. Tried '${qa_auto}' and '${qa_fallback}'. Use --data-path-${dataset} override."
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
      corpus_path="${corpus_auto}"
      corpus_source="auto_corpus_root"
    elif [[ -f "${corpus_fallback}" ]]; then
      corpus_path="${corpus_fallback}"
      corpus_source="auto_repo_data"
    else
      die "Corpus path not found for ${dataset}. Tried '${corpus_auto}' and '${corpus_fallback}'. Use --corpus-path-${dataset} override."
    fi
  fi

  QA_PATH_BY_DATASET["${dataset}"]="${qa_path}"
  QA_SOURCE_BY_DATASET["${dataset}"]="${qa_source}"
  CORPUS_PATH_BY_DATASET["${dataset}"]="${corpus_path}"
  CORPUS_SOURCE_BY_DATASET["${dataset}"]="${corpus_source}"
}

VAR_PROFILE=""
VAR_MODE=""
VAR_KNOBS=""
resolve_variant_spec() {
  local dataset="$1"
  local variant="$2"

  case "${variant}" in
    baseline_mid_reconfirm)
      VAR_PROFILE="baseline_mid"
      VAR_MODE="baseline"
      VAR_KNOBS='{}'
      ;;
    ref_reconfirm)
      case "${dataset}" in
        hotpotqa)
          VAR_PROFILE="strong_ref_hotpot"
          VAR_MODE="r2_plus_path_preserve"
          VAR_KNOBS='{"candidate_top_t":30,"semantic_topn_chunk":25,"graph_reserve_topn":25}'
          ;;
        2wikimultihopqa)
          VAR_PROFILE="strong_ref_2wiki"
          VAR_MODE="r2_plus_path_preserve"
          VAR_KNOBS='{"semantic_topn_chunk":25}'
          ;;
        musique)
          VAR_PROFILE="strong_ref_musique"
          VAR_MODE="r2_plus_path_preserve"
          VAR_KNOBS='{}'
          ;;
        popqa)
          VAR_PROFILE="strong_ref_popqa"
          VAR_MODE="r2_plus_path_preserve"
          VAR_KNOBS='{}'
          ;;
        *)
          die "Unsupported dataset for ref_reconfirm: ${dataset}"
          ;;
      esac
      ;;
    *)
      die "Unsupported variant: ${variant}"
      ;;
  esac
}

assert_cfg_race_safe() {
  local cfg_path="$1"
  "${PYTHON_BIN}" - "${cfg_path}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
if int(cfg.get('num_workers', 1) or 1) > 1:
    raise SystemExit('num_workers must stay 1 for sequential cache safety')
if bool(cfg.get('force_rebuild_graph_index', False)):
    raise SystemExit('force_rebuild_graph_index must stay false in this runner')
print('safe')
PY
}

create_variant_cfg() {
  local out_cfg="$1"
  local dataset="$2"
  local variant="$3"
  local retrieval_mode="$4"
  local knob_json="$5"

  mkdir -p "$(dirname "${out_cfg}")"
  "${PYTHON_BIN}" - \
    "${RAG_BASE_CFG}" \
    "${out_cfg}" \
    "${dataset}" \
    "${variant}" \
    "${retrieval_mode}" \
    "${knob_json}" \
    "${MODEL_NAME}" \
    "${UNIVERSAL_EMBED_MAX_LEN}" \
    "${UNIVERSAL_EMBED_MAX_CHARS}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

base_cfg, out_cfg, dataset, variant, retrieval_mode, knob_json, model_name, embed_len, embed_chars = sys.argv[1:10]
cfg = yaml.safe_load(Path(base_cfg).read_text(encoding='utf-8')) or {}
knobs = json.loads(knob_json or "{}")

cfg['dataset'] = str(dataset)
cfg['canonical_variant_name'] = str(variant)
cfg['embedding_max_length'] = int(embed_len)
cfg['embedding_text_max_chars'] = int(embed_chars)
cfg['retrieval_objective_mode'] = str(retrieval_mode)

cfg['run_qa'] = True
cfg['retrieval_only'] = False
cfg['generator'] = 'vllm'
cfg['model_name'] = str(model_name)
cfg['openie_mode'] = 'llm'
cfg['openie_model_name'] = str(model_name)
cfg['openie_local_files_only'] = True
cfg['evaluator_mode'] = 'hipporag2_parity'

cfg['shared_budget_profile'] = 'off'
cfg['shared_budget_proposal_step'] = 0
cfg['shared_budget_exploration_step'] = 0
cfg['proposal_union_experiment_mode'] = 'off'
cfg['chunk_node_enabled_in_diffusion'] = False
cfg['num_workers'] = 1
cfg['force_rebuild_graph_index'] = False
cfg['stagewise_loss_funnel_enabled'] = True

for k, v in (knobs or {}).items():
    cfg[k] = v

Path(out_cfg).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding='utf-8')
PY

  assert_cfg_race_safe "${out_cfg}" >/dev/null
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
  local query_results_name="$3"
  "${PYTHON_BIN}" - "${summary_path}" "${expected_limit}" "${query_results_name}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_limit = int(float(sys.argv[2]))
query_name = str(sys.argv[3])
if not summary_path.exists():
    raise SystemExit(1)
_ = json.loads(summary_path.read_text(encoding='utf-8'))
qpath = summary_path.with_name(query_name)
if not qpath.exists():
    raise SystemExit(1)
cnt = 0
with qpath.open('r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            cnt += 1
if expected_limit > 0 and cnt < expected_limit:
    raise SystemExit(1)
if expected_limit <= 0 and cnt <= 0:
    raise SystemExit(1)
print(summary_path)
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

tsv = Path(sys.argv[1])
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
if tsv.exists():
    with tsv.open('r', encoding='utf-8') as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < len(header):
                parts.extend([''] * (len(header) - len(parts)))
            rows.append(parts[:len(header)])

key = (stage, variant, dataset, kind)
updated = False
for row in rows:
    rkey = (row[0], row[1], row[2], row[4])
    if rkey == key:
        row[:] = [
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

with tsv.open('w', encoding='utf-8') as f:
    f.write('\t'.join(header) + '\n')
    for row in rows:
        f.write('\t'.join(row) + '\n')
PY
}

write_environment_snapshot() {
  "${PYTHON_BIN}" - "${ENV_SNAPSHOT_JSON}" \
    "${LLM_BASE_URL}" "${LLM_API_KEY}" "${MODEL_NAME}" \
    "${DATASETS_CSV}" "${VARIANTS_CSV}" "${N_SAMPLES}" <<'PY'
import json
import os
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
    'tokenizers_parallelism': os.environ.get('TOKENIZERS_PARALLELISM', ''),
    'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE', ''),
    'transformers_offline': os.environ.get('TRANSFORMERS_OFFLINE', ''),
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
    "${DATASETS_CSV}" \
    "${VARIANTS_CSV}" \
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
variant_mapping = {}
for row in rows:
    ds = str(row.get('dataset', ''))
    v = str(row.get('variant', ''))
    dataset_paths[ds] = {
        'qa_path': row.get('qa_path', ''),
        'qa_source': row.get('qa_source', ''),
        'corpus_path': row.get('corpus_path', ''),
        'corpus_source': row.get('corpus_source', ''),
    }
    variant_mapping.setdefault(v, {})[ds] = {
        'profile': row.get('profile', ''),
        'retrieval_mode': row.get('retrieval_mode', ''),
        'knobs_json': row.get('knobs_json', ''),
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
    'variant_mapping': variant_mapping,
    'matrix': rows,
}

Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(out_json)
PY
}

run_preflight_checks() {
  log_msg "[Stage 0] preflight: python compile"
  "${PYTHON_BIN}" -m py_compile effirag/*.py

  log_msg "[Stage 0] preflight: config audit"
  PYTHONPATH=. "${PYTHON_BIN}" -m effirag.config_audit configs/canonical

  log_msg "[Stage 0] preflight: path auto-resolve sanity"
  for ds in "${SELECTED_DATASETS[@]}"; do
    resolve_dataset_paths "${ds}"
    log_msg "[PATH] dataset=${ds} qa=${QA_PATH_BY_DATASET[$ds]} (${QA_SOURCE_BY_DATASET[$ds]}) corpus=${CORPUS_PATH_BY_DATASET[$ds]} (${CORPUS_SOURCE_BY_DATASET[$ds]})"
  done

  log_msg "[Stage 0] preflight: override sanity"
  for ds in "${SELECTED_DATASETS[@]}"; do
    local qpath="${QA_PATH_BY_DATASET[$ds]}"
    local cpath="${CORPUS_PATH_BY_DATASET[$ds]}"
    [[ -f "${qpath}" ]] || die "Resolved QA path missing for ${ds}: ${qpath}"
    [[ -f "${cpath}" ]] || die "Resolved corpus path missing for ${ds}: ${cpath}"
  done

  if [[ "${SEQUENTIAL}" != "true" ]]; then
    die "sequential must remain true"
  fi

  log_msg "[Stage 0] preflight: fallback env check"
  [[ -n "${LLM_BASE_URL}" ]] || die "LLM_BASE_URL is empty"
  [[ -n "${MODEL_NAME}" ]] || die "MODEL_NAME is empty"
}

run_preflight_without_heavy_checks() {
  for ds in "${SELECTED_DATASETS[@]}"; do
    resolve_dataset_paths "${ds}"
  done
}

build_manifest_matrix() {
  for ds in "${SELECTED_DATASETS[@]}"; do
    for v in "${SELECTED_VARIANTS[@]}"; do
      resolve_variant_spec "${ds}" "${v}"
      printf "s1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${v}" "${ds}" "${VAR_PROFILE}" "${VAR_MODE}" "${VAR_KNOBS}" \
        "${QA_PATH_BY_DATASET[$ds]}" "${QA_SOURCE_BY_DATASET[$ds]}" \
        "${CORPUS_PATH_BY_DATASET[$ds]}" "${CORPUS_SOURCE_BY_DATASET[$ds]}" \
        "${N_SAMPLES}" >> "${MANIFEST_MATRIX_TSV}"
    done
  done
}

run_failed_count=0

run_combo() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"

  resolve_variant_spec "${dataset}" "${variant}"

  local qa_path="${QA_PATH_BY_DATASET[$dataset]}"
  local corpus_path="${CORPUS_PATH_BY_DATASET[$dataset]}"

  local cfg_dir="${CFG_ROOT}/${stage}/${variant}/${dataset}"
  local cfg_path="${cfg_dir}/rag.yaml"
  local out_dir="${RUN_ROOT}/${stage}/${variant}/rag"
  local log_path="${LOG_ROOT}/${stage}__${variant}__${dataset}.log"
  mkdir -p "${cfg_dir}" "${out_dir}" "$(dirname "${log_path}")"

  create_variant_cfg "${cfg_path}" "${dataset}" "${variant}" "${VAR_MODE}" "${VAR_KNOBS}"

  if [[ "${SKIP_COMPLETED}" == "true" ]]; then
    local existing_summary
    existing_summary="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
    if [[ -n "${existing_summary}" ]]; then
      if is_complete_summary "${existing_summary}" "${N_SAMPLES}" "rag_query_results.jsonl" >/dev/null 2>&1; then
        log_msg "[SKIP] stage=${stage} variant=${variant} dataset=${dataset}"
        upsert_record "${stage}" "${variant}" "${dataset}" "${VAR_PROFILE}" "rag" \
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

  log_msg "[RUN] stage=${stage} variant=${variant} dataset=${dataset} profile=${VAR_PROFILE} mode=${VAR_MODE}"
  {
    echo "[COMMAND] ${cmd[*]}"
    echo "[KNOBS] ${VAR_KNOBS}"
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
    upsert_record "${stage}" "${variant}" "${dataset}" "${VAR_PROFILE}" "rag" \
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
    upsert_record "${stage}" "${variant}" "${dataset}" "${VAR_PROFILE}" "rag" \
      "${qa_path}" "${corpus_path}" "" "${cfg_path}" "${log_path}" \
      "failed" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" "missing_summary"
    run_failed_count=$((run_failed_count + 1))
    if [[ "${STOP_ON_FAILURE}" == "true" ]]; then
      exit 1
    fi
    return 1
  fi

  if ! is_complete_summary "${summary_path}" "${N_SAMPLES}" "rag_query_results.jsonl" >/dev/null 2>&1; then
    log_msg "[WARN] stage=${stage} variant=${variant} dataset=${dataset} summary exists but appears incomplete"
  fi

  upsert_record "${stage}" "${variant}" "${dataset}" "${VAR_PROFILE}" "rag" \
    "${qa_path}" "${corpus_path}" "${summary_path}" "${cfg_path}" "${log_path}" \
    "ok" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" ""
  log_msg "[DONE] stage=${stage} variant=${variant} dataset=${dataset}"
}

run_stage1() {
  log_msg "[Stage 1] sequential dataset x variant runs"
  for ds in "${SELECTED_DATASETS[@]}"; do
    for v in "${SELECTED_VARIANTS[@]}"; do
      run_combo "s1" "${v}" "${ds}" || true
    done
  done
}

run_aggregation() {
  log_msg "[Stage 2] aggregate results"
  "${PYTHON_BIN}" scripts/aggregate_multidataset_results.py \
    --round-root "${ROUND_ROOT}" \
    --manifest "${RUN_MANIFEST_JSON}" \
    --run-records "${RUN_RECORDS_TSV}" \
    --output-json "${ROUND_METRICS_JSON}" \
    --output-md "${ROUND_METRICS_MD}" \
    --failures-json "${FAILURES_JSON}" \
    --fallback-warning-threshold 0.10
}

log_msg "Round root: ${ROUND_ROOT}"
log_msg "Log root: ${LOG_ROOT}"
log_msg "Selected datasets: ${DATASETS_CSV}"
log_msg "Selected variants: ${VARIANTS_CSV}"
log_msg "n_samples: ${N_SAMPLES}"
log_msg "LLM endpoint: ${LLM_BASE_URL}"
log_msg "Model: ${MODEL_NAME}"

if [[ "${RUN_PREFLIGHT}" == "true" ]]; then
  run_preflight_checks
else
  run_preflight_without_heavy_checks
fi

build_manifest_matrix
write_manifest
write_environment_snapshot >/dev/null

log_msg "Manifest: ${RUN_MANIFEST_JSON}"
log_msg "Environment snapshot: ${ENV_SNAPSHOT_JSON}"

if [[ "${MANIFEST_ONLY}" == "true" ]]; then
  log_msg "manifest-only mode: stopping before execution"
  exit 0
fi

run_stage1
run_aggregation

if [[ ${run_failed_count} -gt 0 ]]; then
  log_msg "[SUMMARY] Completed with failures: ${run_failed_count}. See ${FAILURES_JSON}"
else
  log_msg "[SUMMARY] Completed without failures"
fi

log_msg "round_metrics.json: ${ROUND_METRICS_JSON}"
log_msg "round_metrics.md: ${ROUND_METRICS_MD}"
