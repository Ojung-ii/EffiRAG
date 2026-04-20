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
OUTPUT_ROOT="outputs/final_lockin_round"
GRAPH_CACHE_DIR="outputs/index_cache"
LLM_BASE_URL="http://localhost:8011/v1"
LLM_API_KEY="EMPTY"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
WITH_APPENDIX_REFERENCE="false"
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

# Prior bridge table source.
EVIDENCE_BRIDGE_ROUND_JSON="outputs/evidence_sufficiency_audit/20260420_114526/round_metrics.json"

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
Usage: bash scripts/run_final_lockin_round.sh [options]

Required/primary options:
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
  --with-appendix-reference [true|false]

Optional:
  --sequential [true|false]
  --skip-completed [true|false]
  --stop-on-failure [true|false]
  --run-preflight [true|false]
  --manifest-only
  --resume-round-root PATH
  --evidence-bridge-round-json PATH

Dataset path overrides:
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
    --with-appendix-reference)
      if [[ $# -gt 1 && "${2:-}" != --* ]]; then
        WITH_APPENDIX_REFERENCE="$(parse_bool "$2")"; shift 2
      else
        WITH_APPENDIX_REFERENCE="true"; shift 1
      fi
      ;;
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
    --evidence-bridge-round-json) EVIDENCE_BRIDGE_ROUND_JSON="$2"; shift 2 ;;

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
WITH_APPENDIX_REFERENCE="$(parse_bool "${WITH_APPENDIX_REFERENCE}")"

[[ -n "${SEQUENTIAL}" ]] || die "Invalid --sequential"
[[ -n "${SKIP_COMPLETED}" ]] || die "Invalid --skip-completed"
[[ -n "${STOP_ON_FAILURE}" ]] || die "Invalid --stop-on-failure"
[[ -n "${RUN_PREFLIGHT}" ]] || die "Invalid --run-preflight"
[[ -n "${WITH_APPENDIX_REFERENCE}" ]] || die "Invalid --with-appendix-reference"

if [[ "${SEQUENTIAL}" != "true" ]]; then
  log_msg "[WARN] sequential-only runner; forcing --sequential true"
  SEQUENTIAL="true"
fi

split_csv "${DATASETS_CSV}" SELECTED_DATASETS
split_csv "${VARIANTS_CSV}" SELECTED_VARIANTS
[[ ${#SELECTED_DATASETS[@]} -gt 0 ]] || die "--datasets resolved to empty list"
[[ ${#SELECTED_VARIANTS[@]} -gt 0 ]] || die "--variants resolved to empty list"

VALID_DATASETS=("hotpotqa" "2wikimultihopqa" "musique" "popqa")
VALID_MAIN_VARIANTS=("baseline_mid_reconfirm" "ref_reconfirm")
APPENDIX_VARIANT="raw_focus_front_scaffold_light"

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
  contains_item "${v}" "${VALID_MAIN_VARIANTS[@]}" || die "Unsupported variant '${v}'. Supported main variants: ${VALID_MAIN_VARIANTS[*]}"
done

if [[ -n "${RESUME_ROUND_ROOT}" ]]; then
  ROUND_ROOT="${RESUME_ROUND_ROOT}"
  ROUND_SOURCE="resume"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="${OUTPUT_ROOT}/${RUN_STAMP}"
  ROUND_SOURCE="new"
fi

LOG_ROOT="logs/final_lockin_round/$(basename "${ROUND_ROOT}")"
CFG_ROOT="${ROUND_ROOT}/configs"
RUN_ROOT="${ROUND_ROOT}/runs"
mkdir -p "${ROUND_ROOT}" "${LOG_ROOT}" "${CFG_ROOT}" "${RUN_ROOT}"

RUN_MANIFEST_JSON="${ROUND_ROOT}/run_manifest.json"
RUN_RECORDS_TSV="${ROUND_ROOT}/run_records.tsv"
MANIFEST_MATRIX_TSV="${ROUND_ROOT}/manifest_matrix.tsv"
ENV_SNAPSHOT_JSON="${ROUND_ROOT}/environment_snapshot.json"
SELECTED_VARIANTS_JSON="${ROUND_ROOT}/selected_variants.json"
REFERENCE_INTEGRITY_JSON="${ROUND_ROOT}/reference_integrity.json"
FAILURES_JSON="${ROUND_ROOT}/failures.json"
ROUND_METRICS_JSON="${ROUND_ROOT}/round_metrics.json"
ROUND_METRICS_MD="${ROUND_ROOT}/round_metrics.md"

PAPER_MAIN_TABLE_MD="${ROUND_ROOT}/paper_main_table.md"
PAPER_DELTA_TABLE_MD="${ROUND_ROOT}/paper_delta_table.md"
PAPER_VARIANT_MEAN_TABLE_MD="${ROUND_ROOT}/paper_variant_mean_table.md"
PAPER_EVIDENCE_BRIDGE_TABLE_MD="${ROUND_ROOT}/paper_evidence_sufficiency_bridge_table.md"
APPENDIX_REFERENCE_TABLE_MD="${ROUND_ROOT}/appendix_reference_table.md"

if [[ ! -f "${RUN_RECORDS_TSV}" ]]; then
  printf "stage\tvariant\tdataset\tprofile\tkind\tqa_path\tcorpus_path\tsummary_path\tconfig_path\tlog_path\tstatus\tstart_ts\tend_ts\telapsed_s\tn_samples\terror_message\n" > "${RUN_RECORDS_TSV}"
fi

printf "stage\tvariant\tdataset\tprofile\tretrieval_mode\tknobs_json\torder_strategy\tqa_path\tqa_source\tcorpus_path\tcorpus_source\tn_samples\n" > "${MANIFEST_MATRIX_TSV}"

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

variant_spec_fields() {
  local dataset="$1"
  local variant="$2"
  case "${variant}" in
    baseline_mid_reconfirm)
      echo "baseline_mid|baseline|{}|score"
      ;;
    ref_reconfirm)
      case "${dataset}" in
        hotpotqa)
          echo "strong_ref_hotpot|r2_plus_path_preserve|{\"candidate_top_t\":30,\"semantic_topn_chunk\":25,\"graph_reserve_topn\":25}|score"
          ;;
        2wikimultihopqa)
          echo "strong_ref_2wiki|r2_plus_path_preserve|{\"semantic_topn_chunk\":25}|score"
          ;;
        musique)
          echo "strong_ref_musique|r2_plus_path_preserve|{}|score"
          ;;
        popqa)
          echo "strong_ref_popqa|r2_plus_path_preserve|{}|score"
          ;;
        *)
          die "Unsupported dataset for ref_reconfirm: ${dataset}"
          ;;
      esac
      ;;
    raw_focus_front_scaffold_light)
      if [[ "${dataset}" != "hotpotqa" ]]; then
        die "${variant} is appendix-only and supported only for hotpotqa"
      fi
      echo "appendix_hotpot_ref|r2_plus_path_preserve|{\"candidate_top_t\":30,\"semantic_topn_chunk\":25,\"graph_reserve_topn\":25}|score+raw_focus_front+raw_focus_scaffold_light"
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
from pathlib import Path
import yaml

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
  local order_strategy="$6"

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
    "${UNIVERSAL_EMBED_MAX_CHARS}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

base_cfg, out_cfg, dataset, variant, retrieval_mode, knob_json, order_strategy, model_name, embed_len, embed_chars = sys.argv[1:11]
cfg = yaml.safe_load(Path(base_cfg).read_text(encoding='utf-8')) or {}
knobs = json.loads(knob_json or "{}")

cfg['dataset'] = str(dataset)
cfg['canonical_variant_name'] = str(variant)
cfg['embedding_max_length'] = int(embed_len)
cfg['embedding_text_max_chars'] = int(embed_chars)
cfg['retrieval_objective_mode'] = str(retrieval_mode)

# Round invariants
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

if str(order_strategy or '').strip():
    cfg['order_strategy'] = str(order_strategy).strip()

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

with path.open('w', encoding='utf-8') as f:
    f.write('\t'.join(header) + '\n')
    for row in rows:
        f.write('\t'.join(row) + '\n')
PY
}

write_environment_snapshot() {
  "${PYTHON_BIN}" - "${ENV_SNAPSHOT_JSON}" \
    "${LLM_BASE_URL}" "${LLM_API_KEY}" "${MODEL_NAME}" \
    "${DATASETS_CSV}" "${VARIANTS_CSV}" "${N_SAMPLES}" "${WITH_APPENDIX_REFERENCE}" <<'PY'
import json
import os
import subprocess
import sys

out_path, llm_base_url, llm_api_key, model_name, datasets_csv, variants_csv, n_samples, with_appendix = sys.argv[1:9]

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
    'with_appendix_reference': bool(str(with_appendix).lower() == 'true'),
    'tokenizers_parallelism': os.environ.get('TOKENIZERS_PARALLELISM', ''),
    'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE', ''),
    'transformers_offline': os.environ.get('TRANSFORMERS_OFFLINE', ''),
}
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(out_path)
PY
}

write_selected_variants_json() {
  "${PYTHON_BIN}" - "${SELECTED_VARIANTS_JSON}" "${VARIANTS_CSV}" "${WITH_APPENDIX_REFERENCE}" "${APPENDIX_VARIANT}" <<'PY'
import json
import sys

out_path, variants_csv, with_appendix, appendix_variant = sys.argv[1:5]
payload = {
    'main_variants': [x.strip() for x in variants_csv.split(',') if x.strip()],
    'appendix_variant': str(appendix_variant),
    'with_appendix_reference': bool(str(with_appendix).lower() == 'true'),
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
    "${MODEL_NAME}" \
    "${WITH_APPENDIX_REFERENCE}" \
    "${APPENDIX_VARIANT}" <<'PY'
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
    with_appendix,
    appendix_variant,
) = sys.argv[1:17]

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
        'appendix_variant': str(appendix_variant),
        'with_appendix_reference': bool(str(with_appendix).lower() == 'true'),
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

run_preflight_checks() {
  log_msg "[Stage 0] preflight: python compile"
  "${PYTHON_BIN}" -m py_compile effirag/*.py

  log_msg "[Stage 0] preflight: config audit"
  PYTHONPATH=. "${PYTHON_BIN}" -m effirag.config_audit configs/canonical

  log_msg "[Stage 0] preflight: dataset path sanity"
  for ds in "${SELECTED_DATASETS[@]}"; do
    resolve_dataset_paths "${ds}"
    log_msg "[PATH] dataset=${ds} qa=${QA_PATH_BY_DATASET[$ds]} (${QA_SOURCE_BY_DATASET[$ds]}) corpus=${CORPUS_PATH_BY_DATASET[$ds]} (${CORPUS_SOURCE_BY_DATASET[$ds]})"
  done

  [[ -n "${LLM_BASE_URL}" ]] || die "LLM_BASE_URL is empty"
  [[ -n "${MODEL_NAME}" ]] || die "MODEL_NAME is empty"
}

run_preflight_without_heavy_checks() {
  for ds in "${SELECTED_DATASETS[@]}"; do
    resolve_dataset_paths "${ds}"
  done
}

build_manifest_matrix() {
  local ds v spec profile mode knobs order
  for ds in "${SELECTED_DATASETS[@]}"; do
    for v in "${SELECTED_VARIANTS[@]}"; do
      spec="$(variant_spec_fields "${ds}" "${v}")"
      IFS='|' read -r profile mode knobs order <<< "${spec}"
      printf "s1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${v}" "${ds}" "${profile}" "${mode}" "${knobs}" "${order}" \
        "${QA_PATH_BY_DATASET[$ds]}" "${QA_SOURCE_BY_DATASET[$ds]}" \
        "${CORPUS_PATH_BY_DATASET[$ds]}" "${CORPUS_SOURCE_BY_DATASET[$ds]}" "${N_SAMPLES}" >> "${MANIFEST_MATRIX_TSV}"
    done
  done

  if [[ "${WITH_APPENDIX_REFERENCE}" == "true" ]]; then
    if contains_item "hotpotqa" "${SELECTED_DATASETS[@]}"; then
      spec="$(variant_spec_fields "hotpotqa" "${APPENDIX_VARIANT}")"
      IFS='|' read -r profile mode knobs order <<< "${spec}"
      printf "s2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${APPENDIX_VARIANT}" "hotpotqa" "${profile}" "${mode}" "${knobs}" "${order}" \
        "${QA_PATH_BY_DATASET[hotpotqa]}" "${QA_SOURCE_BY_DATASET[hotpotqa]}" \
        "${CORPUS_PATH_BY_DATASET[hotpotqa]}" "${CORPUS_SOURCE_BY_DATASET[hotpotqa]}" "${N_SAMPLES}" >> "${MANIFEST_MATRIX_TSV}"
    else
      log_msg "[WARN] --with-appendix-reference=true but hotpotqa is not in --datasets; appendix stage will be skipped"
    fi
  fi
}

run_failed_count=0

run_combo() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local profile="$4"
  local retrieval_mode="$5"
  local knobs_json="$6"
  local order_strategy="$7"

  local qa_path="${QA_PATH_BY_DATASET[$dataset]}"
  local corpus_path="${CORPUS_PATH_BY_DATASET[$dataset]}"

  local cfg_dir="${CFG_ROOT}/${stage}/${variant}/${dataset}"
  local cfg_path="${cfg_dir}/rag.yaml"
  local out_dir="${RUN_ROOT}/${stage}/${variant}/rag"
  local log_path="${LOG_ROOT}/${stage}__${variant}__${dataset}.log"
  mkdir -p "${cfg_dir}" "${out_dir}" "$(dirname "${log_path}")"

  create_variant_cfg "${cfg_path}" "${dataset}" "${variant}" "${retrieval_mode}" "${knobs_json}" "${order_strategy}"

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

  log_msg "[RUN] stage=${stage} variant=${variant} dataset=${dataset} profile=${profile} mode=${retrieval_mode}"
  {
    echo "[COMMAND] ${cmd[*]}"
    echo "[KNOBS] ${knobs_json}"
    echo "[ORDER_STRATEGY] ${order_strategy}"
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

  if ! is_complete_summary "${summary_path}" "${N_SAMPLES}" "rag_query_results.jsonl" >/dev/null 2>&1; then
    log_msg "[WARN] stage=${stage} variant=${variant} dataset=${dataset} summary exists but appears incomplete"
  fi

  upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
    "${qa_path}" "${corpus_path}" "${summary_path}" "${cfg_path}" "${log_path}" \
    "ok" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" ""
  log_msg "[DONE] stage=${stage} variant=${variant} dataset=${dataset}"
}

run_stage() {
  local stage_name="$1"
  log_msg "[${stage_name}] sequential run"
  while IFS=$'\t' read -r stage variant dataset profile retrieval_mode knobs_json order_strategy qa_path qa_source corpus_path corpus_source n_samples; do
    [[ -z "${stage:-}" ]] && continue
    if [[ "${stage}" != "${stage_name}" ]]; then
      continue
    fi
    run_combo "${stage}" "${variant}" "${dataset}" "${profile}" "${retrieval_mode}" "${knobs_json}" "${order_strategy}" || true
  done < <(tail -n +2 "${MANIFEST_MATRIX_TSV}")
}

check_reference_integrity() {
  "${PYTHON_BIN}" - "${RUN_RECORDS_TSV}" "${REFERENCE_INTEGRITY_JSON}" "${DATASETS_CSV}" "${VARIANTS_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

records_tsv, out_json, datasets_csv, variants_csv = sys.argv[1:5]
datasets = [x.strip() for x in datasets_csv.split(',') if x.strip()]
variants = [x.strip() for x in variants_csv.split(',') if x.strip()]
need = {("s1", v, d) for d in datasets for v in variants}
seen = {}

with open(records_tsv, 'r', encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter='\t')
    for row in r:
        if row.get('kind') != 'rag':
            continue
        key = (str(row.get('stage','')), str(row.get('variant','')), str(row.get('dataset','')))
        if key not in need:
            continue
        seen[key] = dict(row)

missing = []
ok = []
for key in sorted(need):
    row = seen.get(key)
    if not row:
        missing.append({'stage': key[0], 'variant': key[1], 'dataset': key[2], 'reason': 'missing_record'})
        continue
    status = str(row.get('status',''))
    if status not in {'ok', 'skipped'}:
        missing.append({'stage': key[0], 'variant': key[1], 'dataset': key[2], 'reason': f'bad_status:{status}'})
        continue
    ok.append({'stage': key[0], 'variant': key[1], 'dataset': key[2], 'status': status, 'summary_path': row.get('summary_path','')})

payload = {
    'status': 'ok' if not missing else 'partial',
    'required_pairs': len(need),
    'ok_pairs': len(ok),
    'missing_pairs': missing,
    'ok_rows': ok,
}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(out_json)
PY
}

build_round_reports() {
  "${PYTHON_BIN}" - \
    "${RUN_MANIFEST_JSON}" \
    "${RUN_RECORDS_TSV}" \
    "${REFERENCE_INTEGRITY_JSON}" \
    "${EVIDENCE_BRIDGE_ROUND_JSON}" \
    "${ROUND_METRICS_JSON}" \
    "${ROUND_METRICS_MD}" \
    "${FAILURES_JSON}" \
    "${PAPER_MAIN_TABLE_MD}" \
    "${PAPER_DELTA_TABLE_MD}" \
    "${PAPER_VARIANT_MEAN_TABLE_MD}" \
    "${PAPER_EVIDENCE_BRIDGE_TABLE_MD}" \
    "${APPENDIX_REFERENCE_TABLE_MD}" \
    "${WITH_APPENDIX_REFERENCE}" <<'PY'
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path,
    records_path,
    reference_integrity_path,
    evidence_bridge_json,
    out_json,
    out_md,
    failures_json,
    paper_main_md,
    paper_delta_md,
    paper_variant_mean_md,
    paper_bridge_md,
    appendix_md,
    with_appendix,
) = sys.argv[1:14]

manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
ref_integrity = json.loads(Path(reference_integrity_path).read_text(encoding='utf-8')) if Path(reference_integrity_path).exists() else {}
with_appendix = str(with_appendix).strip().lower() == 'true'

selection = manifest.get('selection', {}) or {}
selected_datasets = [str(x) for x in (selection.get('datasets', []) or [])]
main_variants = [str(x) for x in (selection.get('variants', []) or [])]
appendix_variant = str(selection.get('appendix_variant', 'raw_focus_front_scaffold_light') or 'raw_focus_front_scaffold_light')


def _safe_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)


def _fmt(v, nd=4):
    return f"{_safe_float(v, 0.0):.{nd}f}"


def _r_at(summary, k):
    direct = _safe_float(summary.get(f'supporting_fact_recall_at_{k}', 0.0), 0.0)
    if direct > 0.0:
        return direct
    bag = summary.get('supporting_fact_recall_at_k', {}) or {}
    if isinstance(bag, dict):
        if str(k) in bag:
            return _safe_float(bag.get(str(k), 0.0), 0.0)
        if k in bag:
            return _safe_float(bag.get(k, 0.0), 0.0)
    bag2 = summary.get('recall_at_k', {}) or {}
    if isinstance(bag2, dict):
        if str(k) in bag2:
            return _safe_float(bag2.get(str(k), 0.0), 0.0)
        if k in bag2:
            return _safe_float(bag2.get(k, 0.0), 0.0)
    return _safe_float(summary.get(f'Recall@{k}', 0.0), 0.0)


def _markdown_table(headers, rows):
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join([str(x) for x in row]) + ' |')
    return '\n'.join(lines)


run_records = []
with Path(records_path).open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        run_records.append({str(k): str(v or '') for k, v in row.items()})

# Parse metrics from summary files.
records = []
failures = []
for row in run_records:
    if row.get('kind') != 'rag':
        continue
    stage = row.get('stage', '')
    variant = row.get('variant', '')
    dataset = row.get('dataset', '')
    profile = row.get('profile', '')
    status = row.get('status', '')
    summary_path = Path(row.get('summary_path', '')).resolve() if row.get('summary_path', '') else None

    payload = {
        'stage': stage,
        'variant': variant,
        'dataset': dataset,
        'profile': profile,
        'status': status,
        'summary_path': str(summary_path) if summary_path else '',
        'log_path': row.get('log_path', ''),
    }

    if status in {'ok', 'skipped'} and summary_path and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            payload.update({
                'R@5': float(_r_at(summary, 5)),
                'EM': float(_safe_float(summary.get('em', summary.get('EM', 0.0)), 0.0)),
                'F1': float(_safe_float(summary.get('f1', summary.get('F1', 0.0)), 0.0)),
                'retrieval_ms': float(_safe_float(summary.get('retrieval_latency_ms', summary.get('retrieval_ms', 0.0)), 0.0)),
                'generation_ms': float(_safe_float(summary.get('generation_latency_ms', summary.get('generation_ms', 0.0)), 0.0)),
                'total_ms': float(_safe_float(summary.get('total_latency_ms', summary.get('total_ms', 0.0)), 0.0)),
                'fallback_rate': float(_safe_float(summary.get('fallback_rate', summary.get('stagewise_fallback_rate', 0.0)), 0.0)),
            })
        except Exception as exc:
            payload['status'] = 'failed'
            payload['error_message'] = f'summary_parse_failed:{exc}'
            failures.append({
                'stage': stage,
                'variant': variant,
                'dataset': dataset,
                'reason': 'summary_parse_failed',
                'detail': str(exc),
                'summary_path': str(summary_path),
            })
    else:
        if status != 'skipped':
            failures.append({
                'stage': stage,
                'variant': variant,
                'dataset': dataset,
                'reason': 'run_failed_or_missing',
                'detail': row.get('error_message', ''),
                'summary_path': row.get('summary_path', ''),
            })

    records.append(payload)

Path(failures_json).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')

# Main paper rows: Stage1 + baseline/ref only.
main_rows = []
for dataset in selected_datasets:
    for variant in main_variants:
        rec = next((r for r in records if r.get('stage') == 's1' and r.get('dataset') == dataset and r.get('variant') == variant), None)
        if rec and rec.get('status') in {'ok', 'skipped'} and 'R@5' in rec:
            main_rows.append(rec)
        else:
            main_rows.append({
                'stage': 's1',
                'dataset': dataset,
                'variant': variant,
                'status': 'missing',
                'R@5': 0.0,
                'EM': 0.0,
                'F1': 0.0,
                'retrieval_ms': 0.0,
                'generation_ms': 0.0,
                'total_ms': 0.0,
                'fallback_rate': 0.0,
            })

# Delta rows baseline -> ref.
delta_rows = []
for dataset in selected_datasets:
    base = next((r for r in main_rows if r.get('dataset') == dataset and r.get('variant') == 'baseline_mid_reconfirm'), None)
    ref = next((r for r in main_rows if r.get('dataset') == dataset and r.get('variant') == 'ref_reconfirm'), None)
    if not base or not ref:
        continue
    delta_rows.append({
        'dataset': dataset,
        'ΔR@5': float(_safe_float(ref.get('R@5', 0.0), 0.0) - _safe_float(base.get('R@5', 0.0), 0.0)),
        'ΔEM': float(_safe_float(ref.get('EM', 0.0), 0.0) - _safe_float(base.get('EM', 0.0), 0.0)),
        'ΔF1': float(_safe_float(ref.get('F1', 0.0), 0.0) - _safe_float(base.get('F1', 0.0), 0.0)),
        'Δretrieval_ms': float(_safe_float(ref.get('retrieval_ms', 0.0), 0.0) - _safe_float(base.get('retrieval_ms', 0.0), 0.0)),
        'Δgeneration_ms': float(_safe_float(ref.get('generation_ms', 0.0), 0.0) - _safe_float(base.get('generation_ms', 0.0), 0.0)),
        'Δtotal_ms': float(_safe_float(ref.get('total_ms', 0.0), 0.0) - _safe_float(base.get('total_ms', 0.0), 0.0)),
    })

# Variant mean summary.
variant_mean_rows = []
for variant in main_variants:
    subset = [r for r in main_rows if r.get('variant') == variant and r.get('status') in {'ok', 'skipped'} and 'R@5' in r]
    completed = sorted({str(r.get('dataset', '')) for r in subset if str(r.get('dataset', ''))})
    if subset:
        mean_em = float(statistics.mean([_safe_float(r.get('EM', 0.0), 0.0) for r in subset]))
        mean_f1 = float(statistics.mean([_safe_float(r.get('F1', 0.0), 0.0) for r in subset]))
        mean_retr = float(statistics.mean([_safe_float(r.get('retrieval_ms', 0.0), 0.0) for r in subset]))
        mean_total = float(statistics.mean([_safe_float(r.get('total_ms', 0.0), 0.0) for r in subset]))
    else:
        mean_em = mean_f1 = mean_retr = mean_total = 0.0
    variant_mean_rows.append({
        'variant': variant,
        'mean_EM': mean_em,
        'mean_F1': mean_f1,
        'mean_retrieval_ms': mean_retr,
        'mean_total_ms': mean_total,
        'completed_datasets': completed,
    })

# Prior evidence bridge table.
bridge_rows = []
bridge_source_status = 'missing'
bridge_source_variant = 'ref_reconfirm_or_baseline_mid_reconfirm'
if Path(evidence_bridge_json).exists():
    try:
        prior = json.loads(Path(evidence_bridge_json).read_text(encoding='utf-8'))
        evid = list(prior.get('evidence_sufficiency_table', []) or [])
        evid_idx = {(str(r.get('dataset', '')), str(r.get('variant', ''))): r for r in evid}
        for dataset in selected_datasets:
            row = evid_idx.get((dataset, 'ref_reconfirm')) or evid_idx.get((dataset, 'baseline_mid_reconfirm'))
            if not row:
                continue
            bridge_rows.append({
                'dataset': dataset,
                'strict_support_recall': float(_safe_float(row.get('strict_support_recall', 0.0), 0.0)),
                'equivalent_evidence_coverage': float(_safe_float(row.get('equivalent_evidence_coverage', 0.0), 0.0)),
                'minimal_support_subset_coverage': float(_safe_float(row.get('minimal_support_subset_coverage', 0.0), 0.0)),
                'answer_present_but_generation_fail': float(_safe_float(row.get('answer_present_but_generation_fail_rate', 0.0), 0.0)),
            })
        bridge_source_status = 'ok'
    except Exception:
        bridge_source_status = 'broken'

# Optional appendix table.
appendix_rows = [
    r for r in records
    if r.get('stage') == 's2' and r.get('variant') == appendix_variant and r.get('dataset') == 'hotpotqa' and r.get('status') in {'ok', 'skipped'} and 'R@5' in r
]

fallback_warn_rows = [
    {
        'dataset': r.get('dataset', ''),
        'variant': r.get('variant', ''),
        'fallback_rate': float(_safe_float(r.get('fallback_rate', 0.0), 0.0)),
    }
    for r in main_rows
    if _safe_float(r.get('fallback_rate', 0.0), 0.0) > 0.0
]

# Write paper markdown table files.
main_headers = ['dataset','variant','R@5','EM','F1','retrieval_ms','generation_ms','total_ms','fallback_rate']
main_table_rows = []
for r in main_rows:
    main_table_rows.append([
        r.get('dataset', ''), r.get('variant', ''),
        _fmt(r.get('R@5', 0.0)), _fmt(r.get('EM', 0.0)), _fmt(r.get('F1', 0.0)),
        _fmt(r.get('retrieval_ms', 0.0), 2), _fmt(r.get('generation_ms', 0.0), 2), _fmt(r.get('total_ms', 0.0), 2),
        _fmt(r.get('fallback_rate', 0.0)),
    ])
Path(paper_main_md).write_text(_markdown_table(main_headers, main_table_rows) + '\n', encoding='utf-8')

delta_headers = ['dataset','ΔR@5','ΔEM','ΔF1','Δretrieval_ms','Δgeneration_ms','Δtotal_ms']
delta_table_rows = []
for r in delta_rows:
    delta_table_rows.append([
        r.get('dataset', ''),
        _fmt(r.get('ΔR@5', 0.0)), _fmt(r.get('ΔEM', 0.0)), _fmt(r.get('ΔF1', 0.0)),
        _fmt(r.get('Δretrieval_ms', 0.0), 2), _fmt(r.get('Δgeneration_ms', 0.0), 2), _fmt(r.get('Δtotal_ms', 0.0), 2),
    ])
Path(paper_delta_md).write_text(_markdown_table(delta_headers, delta_table_rows) + '\n', encoding='utf-8')

mean_headers = ['variant','mean_EM','mean_F1','mean_retrieval_ms','mean_total_ms','completed_datasets']
mean_table_rows = []
for r in variant_mean_rows:
    mean_table_rows.append([
        r.get('variant', ''),
        _fmt(r.get('mean_EM', 0.0)), _fmt(r.get('mean_F1', 0.0)),
        _fmt(r.get('mean_retrieval_ms', 0.0), 2), _fmt(r.get('mean_total_ms', 0.0), 2),
        ','.join(list(r.get('completed_datasets', []) or [])),
    ])
Path(paper_variant_mean_md).write_text(_markdown_table(mean_headers, mean_table_rows) + '\n', encoding='utf-8')

bridge_headers = ['dataset','strict_support_recall','equivalent_evidence_coverage','minimal_support_subset_coverage','answer_present_but_generation_fail']
bridge_table_rows = []
for r in bridge_rows:
    bridge_table_rows.append([
        r.get('dataset', ''),
        _fmt(r.get('strict_support_recall', 0.0)),
        _fmt(r.get('equivalent_evidence_coverage', 0.0)),
        _fmt(r.get('minimal_support_subset_coverage', 0.0)),
        _fmt(r.get('answer_present_but_generation_fail', 0.0)),
    ])
bridge_text = _markdown_table(bridge_headers, bridge_table_rows) if bridge_table_rows else '- missing prior evidence bridge rows'
bridge_text += '\n\n- from prior evidence sufficiency audit\n'
Path(paper_bridge_md).write_text(bridge_text, encoding='utf-8')

if with_appendix and appendix_rows:
    app_headers = ['dataset','variant','R@5','EM','F1','retrieval_ms','generation_ms','total_ms']
    app_rows = []
    for r in appendix_rows:
        app_rows.append([
            r.get('dataset', ''), r.get('variant', ''),
            _fmt(r.get('R@5', 0.0)), _fmt(r.get('EM', 0.0)), _fmt(r.get('F1', 0.0)),
            _fmt(r.get('retrieval_ms', 0.0), 2), _fmt(r.get('generation_ms', 0.0), 2), _fmt(r.get('total_ms', 0.0), 2),
        ])
    Path(appendix_md).write_text(_markdown_table(app_headers, app_rows) + '\n', encoding='utf-8')
else:
    p = Path(appendix_md)
    if p.exists():
        p.unlink()

# Build round_metrics.json
round_payload = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'round_root': str(Path(out_json).resolve().parent),
    'methodology_note': 'Final lock-in round for paper tables (no retrieval-core modifications).',
    'selection': {
        'datasets': selected_datasets,
        'main_variants': main_variants,
        'appendix_variant': appendix_variant,
        'with_appendix_reference': with_appendix,
    },
    'reference_integrity': ref_integrity,
    'main_performance': main_rows,
    'delta_table': delta_rows,
    'variant_mean_summary': variant_mean_rows,
    'evidence_sufficiency_bridge': {
        'source': str(Path(evidence_bridge_json).resolve()) if Path(evidence_bridge_json).exists() else str(evidence_bridge_json),
        'status': bridge_source_status,
        'source_variant_policy': bridge_source_variant,
        'rows': bridge_rows,
        'note': 'from prior evidence sufficiency audit',
    },
    'appendix_reference_table': appendix_rows,
    'fallback_warnings': fallback_warn_rows,
    'failures': failures,
    'artifacts': {
        'paper_main_table_md': str(Path(paper_main_md).resolve()),
        'paper_delta_table_md': str(Path(paper_delta_md).resolve()),
        'paper_variant_mean_table_md': str(Path(paper_variant_mean_md).resolve()),
        'paper_evidence_sufficiency_bridge_table_md': str(Path(paper_bridge_md).resolve()),
        'appendix_reference_table_md': str(Path(appendix_md).resolve()) if Path(appendix_md).exists() else '',
    },
}
Path(out_json).write_text(json.dumps(round_payload, ensure_ascii=False, indent=2), encoding='utf-8')

# Build round_metrics.md
md_lines = []
md_lines.append('# Final Lock-in Round')
md_lines.append('')
md_lines.append('- This round is for stable paper-table reproduction, not new method exploration.')
md_lines.append('- Retrieval core, budget, and universal regime remain fixed.')
md_lines.append('')
md_lines.append('## 1. Run Manifest Summary')
md_lines.append('')
md_lines.append(f"- datasets: {','.join(selected_datasets)}")
md_lines.append(f"- main_variants: {','.join(main_variants)}")
md_lines.append(f"- appendix_variant_enabled: {with_appendix}")
md_lines.append(f"- reference_integrity_status: {ref_integrity.get('status','unknown')}")
md_lines.append(f"- failure_count: {len(failures)}")
if fallback_warn_rows:
    md_lines.append(f"- fallback_warning_count: {len(fallback_warn_rows)}")
else:
    md_lines.append('- fallback_warning_count: 0')
md_lines.append('')

md_lines.append('## 2. Main Paper Table')
md_lines.append('')
md_lines.append(_markdown_table(main_headers, main_table_rows) if main_table_rows else '- no rows')
md_lines.append('')

md_lines.append('## 3. Delta Table')
md_lines.append('')
md_lines.append(_markdown_table(delta_headers, delta_table_rows) if delta_table_rows else '- no rows')
md_lines.append('')

md_lines.append('## 4. Variant Mean Summary')
md_lines.append('')
md_lines.append(_markdown_table(mean_headers, mean_table_rows) if mean_table_rows else '- no rows')
md_lines.append('')

md_lines.append('## 5. Evidence Sufficiency Bridge (Prior Audit)')
md_lines.append('')
md_lines.append(bridge_text.strip())
md_lines.append('')

md_lines.append('## 6. Optional Appendix Table')
md_lines.append('')
if with_appendix and appendix_rows:
    md_lines.append(_markdown_table(['dataset','variant','R@5','EM','F1','retrieval_ms','generation_ms','total_ms'], [
        [r.get('dataset',''), r.get('variant',''), _fmt(r.get('R@5',0.0)), _fmt(r.get('EM',0.0)), _fmt(r.get('F1',0.0)), _fmt(r.get('retrieval_ms',0.0),2), _fmt(r.get('generation_ms',0.0),2), _fmt(r.get('total_ms',0.0),2)]
        for r in appendix_rows
    ]))
else:
    md_lines.append('- appendix reference disabled or missing')
md_lines.append('')

md_lines.append('## 7. Interpretation')
md_lines.append('')
md_lines.append('- ref_reconfirm mainline candidacy should be judged by consistency of ΔF1/ΔEM across four datasets and time-cost trade-offs.')
md_lines.append('- strict recall alone may understate utility when equivalent/minimal support coverage remains higher (see bridge table from prior audit).')
md_lines.append('- family view: hotpotqa/2wiki are conversion-sensitive multi-hop, musique is harder compositional multi-hop ceiling, popqa is prior-heavy with equivalent-evidence effects.')
md_lines.append('- avoid unsupported causal claims; use observed deltas and bridge metrics together.')

Path(out_md).write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

print(out_json)
print(out_md)
PY
}

check_cache_meta_race() {
  log_msg "[CHECK] cache/meta race guard across generated configs"
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
    raise SystemExit('unsafe cache/meta race configs:\n' + '\n'.join(unsafe))
print('safe')
PY
}

log_msg "Round root: ${ROUND_ROOT}"
log_msg "Log root: ${LOG_ROOT}"
log_msg "Selected datasets: ${DATASETS_CSV}"
log_msg "Selected variants: ${VARIANTS_CSV}"
log_msg "n_samples: ${N_SAMPLES}"
log_msg "with_appendix_reference: ${WITH_APPENDIX_REFERENCE}"
log_msg "LLM endpoint: ${LLM_BASE_URL}"
log_msg "Model: ${MODEL_NAME}"

if [[ "${RUN_PREFLIGHT}" == "true" ]]; then
  run_preflight_checks
else
  run_preflight_without_heavy_checks
fi

build_manifest_matrix
write_manifest >/dev/null
write_environment_snapshot >/dev/null
write_selected_variants_json >/dev/null

log_msg "Manifest: ${RUN_MANIFEST_JSON}"
log_msg "Environment snapshot: ${ENV_SNAPSHOT_JSON}"
log_msg "Selected variants json: ${SELECTED_VARIANTS_JSON}"

if [[ "${MANIFEST_ONLY}" == "true" ]]; then
  log_msg "manifest-only mode: stopping before execution"
  exit 0
fi

run_stage "s1"
if [[ "${WITH_APPENDIX_REFERENCE}" == "true" ]]; then
  run_stage "s2"
fi

check_reference_integrity >/dev/null
check_cache_meta_race
build_round_reports

if [[ ${run_failed_count} -gt 0 ]]; then
  log_msg "[SUMMARY] Completed with failures: ${run_failed_count}. See ${FAILURES_JSON}"
else
  log_msg "[SUMMARY] Completed without failures"
fi

log_msg "run_manifest.json: ${RUN_MANIFEST_JSON}"
log_msg "run_records.tsv: ${RUN_RECORDS_TSV}"
log_msg "round_metrics.json: ${ROUND_METRICS_JSON}"
log_msg "round_metrics.md: ${ROUND_METRICS_MD}"
log_msg "paper_main_table.md: ${PAPER_MAIN_TABLE_MD}"
log_msg "paper_delta_table.md: ${PAPER_DELTA_TABLE_MD}"
log_msg "paper_variant_mean_table.md: ${PAPER_VARIANT_MEAN_TABLE_MD}"
log_msg "paper_evidence_sufficiency_bridge_table.md: ${PAPER_EVIDENCE_BRIDGE_TABLE_MD}"
if [[ -f "${APPENDIX_REFERENCE_TABLE_MD}" ]]; then
  log_msg "appendix_reference_table.md: ${APPENDIX_REFERENCE_TABLE_MD}"
fi
