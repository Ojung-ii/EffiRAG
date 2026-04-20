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

DATASETS="2wikimultihopqa,hotpotqa"
VARIANTS="ref_reconfirm,quote_then_answer_light,grounded_answer_light,evidence_focus_light"
N_SAMPLES="1000"
QA_ROOT="data/qa"
CORPUS_ROOT="/home/ojungii/HippoRAG2/dataset"
OUTPUT_ROOT="outputs/generation_intervention_round"
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
RUN_STAGE3="auto"   # auto|true|false

# Dataset path overrides.
DATA_PATH_HOTPOTQA=""
DATA_PATH_2WIKIMULTIHOPQA=""
CORPUS_PATH_HOTPOTQA=""
CORPUS_PATH_2WIKIMULTIHOPQA=""

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
Usage: bash scripts/run_generation_intervention_round.sh [options]

Main options:
  --datasets CSV                               (default: 2wikimultihopqa,hotpotqa)
  --variants CSV                               (default: ref_reconfirm,quote_then_answer_light,grounded_answer_light,evidence_focus_light)
  --n-samples N                                (default: 1000)
  --qa-root PATH                               (default: data/qa)
  --corpus-root PATH                           (default: /home/ojungii/HippoRAG2/dataset)
  --output-root PATH                           (default: outputs/generation_intervention_round)
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
  --run-stage3 auto|true|false

Dataset overrides:
  --data-path-hotpotqa PATH
  --data-path-2wikimultihopqa PATH
  --corpus-path-hotpotqa PATH
  --corpus-path-2wikimultihopqa PATH
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
    --run-stage3) RUN_STAGE3="$(echo "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;

    --data-path-hotpotqa) DATA_PATH_HOTPOTQA="$2"; shift 2 ;;
    --data-path-2wikimultihopqa) DATA_PATH_2WIKIMULTIHOPQA="$2"; shift 2 ;;
    --corpus-path-hotpotqa) CORPUS_PATH_HOTPOTQA="$2"; shift 2 ;;
    --corpus-path-2wikimultihopqa) CORPUS_PATH_2WIKIMULTIHOPQA="$2"; shift 2 ;;

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
case "${RUN_STAGE3}" in
  auto|true|false) ;;
  *) die "Invalid --run-stage3. Use auto|true|false" ;;
esac

if [[ "${SEQUENTIAL}" != "true" ]]; then
  log_msg "[WARN] sequential-only runner; forcing --sequential true"
  SEQUENTIAL="true"
fi

split_csv "${DATASETS}" SELECTED_DATASETS
split_csv "${VARIANTS}" SELECTED_VARIANTS
[[ ${#SELECTED_DATASETS[@]} -gt 0 ]] || die "--datasets resolved to empty"
[[ ${#SELECTED_VARIANTS[@]} -gt 0 ]] || die "--variants resolved to empty"

VALID_DATASETS=("2wikimultihopqa" "hotpotqa")
VALID_VARIANTS=("ref_reconfirm" "quote_then_answer_light" "grounded_answer_light" "evidence_focus_light")

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

LOG_ROOT="logs/generation_intervention_round/$(basename "${ROUND_ROOT}")"
CFG_ROOT="${ROUND_ROOT}/configs"
RUN_ROOT="${ROUND_ROOT}/runs"
mkdir -p "${ROUND_ROOT}" "${LOG_ROOT}" "${CFG_ROOT}" "${RUN_ROOT}"

RUN_MANIFEST_JSON="${ROUND_ROOT}/run_manifest.json"
RUN_RECORDS_TSV="${ROUND_ROOT}/run_records.tsv"
MANIFEST_MATRIX_TSV="${ROUND_ROOT}/manifest_matrix.tsv"
ENV_SNAPSHOT_JSON="${ROUND_ROOT}/environment_snapshot.json"
FAILURES_JSON="${ROUND_ROOT}/failures.json"
REFERENCE_INTEGRITY_JSON="${ROUND_ROOT}/reference_integrity.json"
ROUND_METRICS_JSON="${ROUND_ROOT}/round_metrics.json"
ROUND_METRICS_MD="${ROUND_ROOT}/round_metrics.md"

EVIDENCE_METRICS_JSON="${ROUND_ROOT}/evidence_round_metrics.json"
EVIDENCE_METRICS_MD="${ROUND_ROOT}/evidence_round_metrics.md"
METRIC_INTEGRITY_JSON="${ROUND_ROOT}/metric_integrity_report.json"
QUERY_DIAG_JSONL="${ROUND_ROOT}/generation_intervention_query_diagnostics_all.jsonl"
QUERY_DIAG_CSV="${ROUND_ROOT}/generation_intervention_query_diagnostics_all.csv"

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
    *) echo "" ;;
  esac
}

get_corpus_override() {
  local dataset="$1"
  case "${dataset}" in
    hotpotqa) echo "${CORPUS_PATH_HOTPOTQA}" ;;
    2wikimultihopqa) echo "${CORPUS_PATH_2WIKIMULTIHOPQA}" ;;
    *) echo "" ;;
  esac
}

resolve_dataset_paths() {
  local dataset="$1"
  local qa_override corpus_override
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
      corpus_path="${corpus_auto}"
      corpus_source="auto_corpus_root"
    elif [[ -f "${corpus_fallback}" ]]; then
      corpus_path="${corpus_fallback}"
      corpus_source="auto_repo_data"
    else
      die "Corpus path not found for ${dataset}. Tried '${corpus_auto}' and '${corpus_fallback}'. Use override."
    fi
  fi

  QA_PATH_BY_DATASET["${dataset}"]="${qa_path}"
  QA_SOURCE_BY_DATASET["${dataset}"]="${qa_source}"
  CORPUS_PATH_BY_DATASET["${dataset}"]="${corpus_path}"
  CORPUS_SOURCE_BY_DATASET["${dataset}"]="${corpus_source}"
}

dataset_ref_knobs_json() {
  local dataset="$1"
  case "${dataset}" in
    hotpotqa) echo '{"candidate_top_t":30,"semantic_topn_chunk":25,"graph_reserve_topn":25}' ;;
    2wikimultihopqa) echo '{"semantic_topn_chunk":25}' ;;
    *) echo '{}' ;;
  esac
}

dataset_reference_order_strategy() {
  local dataset="$1"
  case "${dataset}" in
    hotpotqa) echo 'score+raw_focus_front+raw_focus_scaffold_light' ;;
    2wikimultihopqa) echo 'score' ;;
    *) echo 'score' ;;
  esac
}

intervention_flag_token() {
  local variant="$1"
  case "${variant}" in
    quote_then_answer_light) echo 'gen_quote_then_answer_light' ;;
    grounded_answer_light) echo 'gen_grounded_answer_light' ;;
    evidence_focus_light) echo 'gen_evidence_focus_light' ;;
    *) echo '' ;;
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
    raise SystemExit('num_workers must stay 1 for cache/meta race safety')
if bool(cfg.get('force_rebuild_graph_index', False)):
    raise SystemExit('force_rebuild_graph_index must stay false for cache/meta race safety')
print('safe')
PY
}

create_variant_cfg() {
  local out_cfg="$1"
  local dataset="$2"
  local canonical_variant="$3"
  local retrieval_mode="$4"
  local knob_json="$5"
  local order_strategy="$6"

  mkdir -p "$(dirname "${out_cfg}")"
  "${PYTHON_BIN}" - \
    "${RAG_BASE_CFG}" \
    "${out_cfg}" \
    "${dataset}" \
    "${canonical_variant}" \
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

# Round invariants: retrieval core fixed, no budget/proposal/exploration toggles.
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

# Keep retrieval core untouched; no additional retrieval heuristics.
cfg['hybrid_anchor_recall_enabled'] = False
cfg['bridge_candidate_induction_enabled'] = False
cfg['role_aware_chunk_scoring_enabled'] = False
cfg['coverage_selection_enabled'] = False
cfg['corridor_compact_shaping_enabled'] = False
cfg['corridor_answer_preserve_enabled'] = False
cfg['corridor_bridge_purity_shaping_enabled'] = False
cfg['corridor_path_preserve_enabled'] = False
cfg['corridor_path_preserve_compact_enabled'] = False
cfg['corridor_path_preserve_guarded_enabled'] = False
cfg['corridor_path_preserve_compact_lite_enabled'] = False

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

write_environment_snapshot() {
  "${PYTHON_BIN}" - "${ENV_SNAPSHOT_JSON}" \
    "${LLM_BASE_URL}" "${LLM_API_KEY}" "${MODEL_NAME}" \
    "${DATASETS}" "${VARIANTS}" "${N_SAMPLES}" "${RUN_STAGE3}" <<'PY'
import json
import subprocess
import sys

out_path, llm_base_url, llm_api_key, model_name, datasets_csv, variants_csv, n_samples, run_stage3 = sys.argv[1:9]

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
    'run_stage3': run_stage3,
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

sample = Sample(qid="sanity", question="Who wrote Hamlet?", answer="William Shakespeare", contexts=[])
base = RenderedContext(
    sample_id="sanity",
    method="effirag",
    text="[1] Hamlet was written by William Shakespeare.",
    sentences=["Hamlet was written by William Shakespeare."],
    sentence_ids=["sid1"],
    truncated=False,
    metadata={},
)

cases = [
    {
        "name": "quote_then_answer_light",
        "meta": {
            "generation_intervention_enabled": True,
            "generation_intervention_text": "First identify one short supporting phrase from the context, then output only the final answer span.",
        },
        "must_include": "supporting phrase",
    },
    {
        "name": "grounded_answer_light",
        "meta": {
            "generation_intervention_enabled": True,
            "generation_intervention_text": "Answer using the wording most directly supported by the earliest answer-bearing evidence.",
        },
        "must_include": "wording most directly supported",
    },
    {
        "name": "evidence_focus_light",
        "meta": {
            "generation_intervention_enabled": True,
            "generation_intervention_text": "Prioritize evidence lines [1], [2] when deciding the answer; output only the final answer span.",
        },
        "must_include": "evidence lines [1], [2]",
    },
]

for case in cases:
    rendered = RenderedContext(**{**base.__dict__, "metadata": dict(case["meta"])})
    prompt = _build_qa_prompt(sample=sample, rendered=rendered)
    if case["must_include"] not in prompt:
        raise SystemExit(f"prompt_sanity_failed:{case['name']}")

print("prompt_sanity_ok")
PY
}

build_manifest_matrix() {
  # Stage 1 references (dataset-specific)
  for ds in "${SELECTED_DATASETS[@]}"; do
    if [[ "${ds}" == "2wikimultihopqa" ]]; then
      printf "s1\tref_reconfirm\t%s\tstrong_ref_2wiki\tr2_plus_path_preserve\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${ds}" "$(dataset_ref_knobs_json "${ds}")" "$(dataset_reference_order_strategy "${ds}")" \
        "${QA_PATH_BY_DATASET[$ds]}" "${QA_SOURCE_BY_DATASET[$ds]}" "${CORPUS_PATH_BY_DATASET[$ds]}" "${CORPUS_SOURCE_BY_DATASET[$ds]}" "${N_SAMPLES}" >> "${MANIFEST_MATRIX_TSV}"
    elif [[ "${ds}" == "hotpotqa" ]]; then
      printf "s1\traw_focus_front_scaffold_light\t%s\tstrong_ref_hotpot\tr2_plus_path_preserve\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${ds}" "$(dataset_ref_knobs_json "${ds}")" "$(dataset_reference_order_strategy "${ds}")" \
        "${QA_PATH_BY_DATASET[$ds]}" "${QA_SOURCE_BY_DATASET[$ds]}" "${CORPUS_PATH_BY_DATASET[$ds]}" "${CORPUS_SOURCE_BY_DATASET[$ds]}" "${N_SAMPLES}" >> "${MANIFEST_MATRIX_TSV}"
    fi
  done

  # Stage 2 intervention runs.
  for ds in "${SELECTED_DATASETS[@]}"; do
    for v in "${SELECTED_VARIANTS[@]}"; do
      local flag
      flag="$(intervention_flag_token "${v}")"
      if [[ -z "${flag}" ]]; then
        continue
      fi
      local order_base
      order_base="$(dataset_reference_order_strategy "${ds}")"
      printf "s2\t%s\t%s\tgeneration_intervention\tr2_plus_path_preserve\t%s\t%s+%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${v}" "${ds}" "$(dataset_ref_knobs_json "${ds}")" "${order_base}" "${flag}" \
        "${QA_PATH_BY_DATASET[$ds]}" "${QA_SOURCE_BY_DATASET[$ds]}" "${CORPUS_PATH_BY_DATASET[$ds]}" "${CORPUS_SOURCE_BY_DATASET[$ds]}" "${N_SAMPLES}" >> "${MANIFEST_MATRIX_TSV}"
    done
  done
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

run_failed_count=0

run_combo_custom() {
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

  log_msg "[RUN] stage=${stage} variant=${variant} dataset=${dataset} profile=${profile}"
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

  upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
    "${qa_path}" "${corpus_path}" "${summary_path}" "${cfg_path}" "${log_path}" \
    "ok" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" ""
  log_msg "[DONE] stage=${stage} variant=${variant} dataset=${dataset}"
  return 0
}

run_stage1_references() {
  log_msg "[Stage 1] reference integrity check"

  if contains_item "2wikimultihopqa" "${SELECTED_DATASETS[@]}"; then
    run_combo_custom "s1" "ref_reconfirm" "2wikimultihopqa" "strong_ref_2wiki" \
      "r2_plus_path_preserve" "$(dataset_ref_knobs_json 2wikimultihopqa)" \
      "$(dataset_reference_order_strategy 2wikimultihopqa)" || true
  fi

  if contains_item "hotpotqa" "${SELECTED_DATASETS[@]}"; then
    run_combo_custom "s1" "raw_focus_front_scaffold_light" "hotpotqa" "strong_ref_hotpot" \
      "r2_plus_path_preserve" "$(dataset_ref_knobs_json hotpotqa)" \
      "$(dataset_reference_order_strategy hotpotqa)" || true
  fi
}

run_stage2_interventions() {
  log_msg "[Stage 2] minimal generation-side intervention variants"
  local ds
  for ds in 2wikimultihopqa hotpotqa; do
    if ! contains_item "${ds}" "${SELECTED_DATASETS[@]}"; then
      continue
    fi
    local order_base
    order_base="$(dataset_reference_order_strategy "${ds}")"

    local v
    for v in "${SELECTED_VARIANTS[@]}"; do
      local tok
      tok="$(intervention_flag_token "${v}")"
      if [[ -z "${tok}" ]]; then
        continue
      fi
      run_combo_custom "s2" "${v}" "${ds}" "generation_intervention" \
        "r2_plus_path_preserve" "$(dataset_ref_knobs_json "${ds}")" "${order_base}+${tok}" || true
    done
  done
}

compute_stage3_plan() {
  "${PYTHON_BIN}" - "${RUN_RECORDS_TSV}" <<'PY'
import csv
import json
from pathlib import Path

records = []
with open(Path(__import__('sys').argv[1]), 'r', encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter='\t')
    for row in r:
        if row.get('kind') != 'rag':
            continue
        if row.get('status') not in {'ok', 'skipped'}:
            continue
        sp = str(row.get('summary_path') or '').strip()
        if not sp or not Path(sp).exists():
            continue
        records.append(row)

def load_f1(summary_path):
    try:
        s = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    except Exception:
        return None
    for k in ('f1', 'F1'):
        if k in s:
            try:
                return float(s[k])
            except Exception:
                pass
    return None

ref_f1 = None
for row in records:
    if row.get('stage') == 's1' and row.get('dataset') == '2wikimultihopqa' and row.get('variant') == 'ref_reconfirm':
        ref_f1 = load_f1(row.get('summary_path', ''))
        break

cands = []
for row in records:
    if row.get('stage') != 's2':
        continue
    if row.get('dataset') != '2wikimultihopqa':
        continue
    variant = str(row.get('variant') or '')
    if variant not in {'quote_then_answer_light', 'grounded_answer_light', 'evidence_focus_light'}:
        continue
    f1 = load_f1(row.get('summary_path', ''))
    if f1 is None:
        continue
    cands.append((variant, f1))

best_variant = ''
best_f1 = None
if cands:
    cands.sort(key=lambda x: x[1], reverse=True)
    best_variant, best_f1 = cands[0]

should_run = False
if ref_f1 is not None and best_f1 is not None and best_f1 >= (ref_f1 - 1.0e-6):
    should_run = True

print(json.dumps({
    'ref_f1': ref_f1,
    'best_variant': best_variant,
    'best_f1': best_f1,
    'should_run': should_run,
}, ensure_ascii=False))
PY
}

run_stage3_optional_ab() {
  log_msg "[Stage 3] optional 2Wiki wording A/B"

  if ! contains_item "2wikimultihopqa" "${SELECTED_DATASETS[@]}"; then
    log_msg "[Stage 3] skipped: 2wikimultihopqa not selected"
    return 0
  fi

  local plan_json
  plan_json="$(compute_stage3_plan)"
  local should_run best_variant
  should_run="$("${PYTHON_BIN}" - <<'PY' "${plan_json}"
import json,sys
x=json.loads(sys.argv[1])
print('true' if bool(x.get('should_run', False)) else 'false')
PY
)"
  best_variant="$("${PYTHON_BIN}" - <<'PY' "${plan_json}"
import json,sys
x=json.loads(sys.argv[1])
print(str(x.get('best_variant','')))
PY
)"

  if [[ "${RUN_STAGE3}" == "false" ]]; then
    log_msg "[Stage 3] skipped: run_stage3=false"
    return 0
  fi
  if [[ "${RUN_STAGE3}" == "auto" && "${should_run}" != "true" ]]; then
    log_msg "[Stage 3] skipped: auto condition not met"
    return 0
  fi

  if [[ -z "${best_variant}" ]]; then
    log_msg "[Stage 3] skipped: no eligible Stage 2 variant"
    return 0
  fi

  local base_order tok
  base_order="$(dataset_reference_order_strategy 2wikimultihopqa)"
  tok="$(intervention_flag_token "${best_variant}")"
  if [[ -z "${tok}" ]]; then
    log_msg "[Stage 3] skipped: best variant has no intervention token"
    return 0
  fi

  run_combo_custom "s3" "${best_variant}_wording_ab1" "2wikimultihopqa" "generation_intervention_ab" \
    "r2_plus_path_preserve" "$(dataset_ref_knobs_json 2wikimultihopqa)" "${base_order}+${tok}+gen_light_ab_chain" || true

  run_combo_custom "s3" "${best_variant}_wording_ab2" "2wikimultihopqa" "generation_intervention_ab" \
    "r2_plus_path_preserve" "$(dataset_ref_knobs_json 2wikimultihopqa)" "${base_order}+${tok}+gen_light_ab_wording" || true
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
    --base-variants "ref_reconfirm,raw_focus_front_scaffold_light"

  log_msg "[Aggregate] round summary tables"
  "${PYTHON_BIN}" - \
    "${EVIDENCE_METRICS_JSON}" \
    "${RUN_RECORDS_TSV}" \
    "${REFERENCE_INTEGRITY_JSON}" \
    "${ROUND_METRICS_JSON}" \
    "${ROUND_METRICS_MD}" \
    "${FAILURES_JSON}" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

in_json, records_tsv, ref_json_path, out_json, out_md, failures_json = sys.argv[1:7]
payload = json.loads(Path(in_json).read_text(encoding='utf-8'))

perf = list(payload.get('main_performance', []) or [])
ev = list(payload.get('evidence_sufficiency_table', []) or [])
metric_integrity_status = str(payload.get('metric_integrity_status', 'unknown'))

perf_idx = {(str(r.get('dataset','')), str(r.get('variant',''))): r for r in perf}
ev_idx = {(str(r.get('dataset','')), str(r.get('variant',''))): r for r in ev}

# Keep appearance order from run_records for readable tables.
order = {}
with open(records_tsv, 'r', encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter='\t')
    i = 0
    for row in r:
        if row.get('kind') != 'rag':
            continue
        if row.get('status') not in {'ok', 'skipped'}:
            continue
        key = (str(row.get('dataset','')), str(row.get('variant','')))
        if key not in order:
            order[key] = i
            i += 1

rows = []
for key, p in perf_idx.items():
    d, v = key
    e = ev_idx.get(key, {})
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
        'answer_bearing_chunk_present_rate': float(e.get('answer_bearing_chunk_present_rate', 0.0) or 0.0),
        'minimal_support_subset_coverage': float(e.get('minimal_support_subset_coverage', 0.0) or 0.0),
        'equivalent_evidence_coverage': float(e.get('equivalent_evidence_coverage', 0.0) or 0.0),
        'answer_present_but_generation_fail_rate': float(e.get('answer_present_but_generation_fail_rate', 0.0) or 0.0),
        'output_overlap_answer_bearing_rate': float(e.get('output_overlap_answer_bearing_rate', 0.0) or 0.0),
        'avg_first_answer_chunk_rank': float(e.get('avg_first_answer_chunk_rank', 0.0) or 0.0),
        'avg_first_answer_bundle_rank': float(e.get('avg_first_answer_bundle_rank', 0.0) or 0.0),
        'avg_evidence_dispersion_score': float(e.get('avg_evidence_dispersion_score', 0.0) or 0.0),
        'avg_evidence_duplication_score': float(e.get('avg_evidence_duplication_score', 0.0) or 0.0),
    })

rows.sort(key=lambda r: order.get((r['dataset'], r['variant']), 10**9))

# Reference integrity for this round.
reference_targets = [
    ('2wikimultihopqa', 'ref_reconfirm'),
    ('hotpotqa', 'raw_focus_front_scaffold_light'),
]
reference_rows = []
missing = []
for ds, vv in reference_targets:
    rec = next((r for r in rows if r['dataset'] == ds and r['variant'] == vv), None)
    if rec is None:
        missing.append({'dataset': ds, 'variant': vv})
        continue
    reference_rows.append({
        'dataset': ds,
        'variant': vv,
        'R@5': rec['R@5'],
        'EM': rec['EM'],
        'F1': rec['F1'],
        'retrieval_ms': rec['retrieval_ms'],
        'generation_ms': rec['generation_ms'],
        'total_ms': rec['total_ms'],
        'fallback_rate': rec['fallback_rate'],
        'answer_present_but_generation_fail_rate': rec['answer_present_but_generation_fail_rate'],
        'output_overlap_answer_bearing_rate': rec['output_overlap_answer_bearing_rate'],
    })

reference_integrity = {
    'status': 'ok' if not missing else 'partial',
    'metric_integrity_status': metric_integrity_status,
    'references': reference_rows,
    'missing': missing,
}
Path(ref_json_path).write_text(json.dumps(reference_integrity, ensure_ascii=False, indent=2), encoding='utf-8')

# 2Wiki deltas vs ref_reconfirm.
wiki_ref = next((r for r in rows if r['dataset'] == '2wikimultihopqa' and r['variant'] == 'ref_reconfirm'), None)
wiki_delta = []
if wiki_ref is not None:
    for r in rows:
        if r['dataset'] != '2wikimultihopqa':
            continue
        if r['variant'] == 'ref_reconfirm':
            continue
        wiki_delta.append({
            'dataset': r['dataset'],
            'variant': r['variant'],
            'dEM_vs_ref': float(r['EM'] - wiki_ref['EM']),
            'dF1_vs_ref': float(r['F1'] - wiki_ref['F1']),
            'd_generation_ms_vs_ref': float(r['generation_ms'] - wiki_ref['generation_ms']),
            'd_total_ms_vs_ref': float(r['total_ms'] - wiki_ref['total_ms']),
            'd_answer_present_but_generation_fail_vs_ref': float(r['answer_present_but_generation_fail_rate'] - wiki_ref['answer_present_but_generation_fail_rate']),
            'd_output_overlap_answer_bearing_vs_ref': float(r['output_overlap_answer_bearing_rate'] - wiki_ref['output_overlap_answer_bearing_rate']),
        })

# Hotpot regression check vs raw_focus_front_scaffold_light.
hotpot_ref = next((r for r in rows if r['dataset'] == 'hotpotqa' and r['variant'] == 'raw_focus_front_scaffold_light'), None)
hotpot_reg = []
if hotpot_ref is not None:
    for r in rows:
        if r['dataset'] != 'hotpotqa':
            continue
        if r['variant'] == 'raw_focus_front_scaffold_light':
            continue
        hotpot_reg.append({
            'dataset': r['dataset'],
            'variant': r['variant'],
            'dEM_vs_ref': float(r['EM'] - hotpot_ref['EM']),
            'dF1_vs_ref': float(r['F1'] - hotpot_ref['F1']),
            'd_generation_ms_vs_ref': float(r['generation_ms'] - hotpot_ref['generation_ms']),
            'd_total_ms_vs_ref': float(r['total_ms'] - hotpot_ref['total_ms']),
        })

# Failures summary.
failures = []
with open(records_tsv, 'r', encoding='utf-8') as f:
    rr = csv.DictReader(f, delimiter='\t')
    for row in rr:
        if str(row.get('status', '')) == 'failed':
            failures.append({
                'stage': str(row.get('stage', '')),
                'variant': str(row.get('variant', '')),
                'dataset': str(row.get('dataset', '')),
                'error_message': str(row.get('error_message', '')),
                'log_path': str(row.get('log_path', '')),
            })
Path(failures_json).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')

# Diagnostic table (requested fields only).
diagnostic_rows = []
for r in rows:
    diagnostic_rows.append({
        'dataset': r['dataset'],
        'variant': r['variant'],
        'answer_bearing_chunk_present_rate': r['answer_bearing_chunk_present_rate'],
        'minimal_support_subset_coverage': r['minimal_support_subset_coverage'],
        'equivalent_evidence_coverage': r['equivalent_evidence_coverage'],
        'answer_present_but_generation_fail_rate': r['answer_present_but_generation_fail_rate'],
        'output_overlap_answer_bearing_rate': r['output_overlap_answer_bearing_rate'],
        'avg_first_answer_chunk_rank': r['avg_first_answer_chunk_rank'],
        'avg_first_answer_bundle_rank': r['avg_first_answer_bundle_rank'],
        'avg_evidence_dispersion_score': r['avg_evidence_dispersion_score'],
        'avg_evidence_duplication_score': r['avg_evidence_duplication_score'],
    })

# Heuristic interpretation by variant family.
def pick(ds, variant):
    return next((x for x in rows if x['dataset'] == ds and x['variant'] == variant), None)

wiki_interp = {}
if wiki_ref is not None:
    for v in ('quote_then_answer_light', 'grounded_answer_light', 'evidence_focus_light'):
        rr = pick('2wikimultihopqa', v)
        if rr is None:
            continue
        wiki_interp[v] = {
            'dF1': float(rr['F1'] - wiki_ref['F1']),
            'dABGF': float(rr['answer_present_but_generation_fail_rate'] - wiki_ref['answer_present_but_generation_fail_rate']),
            'dOverlap': float(rr['output_overlap_answer_bearing_rate'] - wiki_ref['output_overlap_answer_bearing_rate']),
        }

payload_out = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'round_root': str(Path(out_json).resolve().parent),
    'methodology_note': 'Generation-side minimal intervention round (retrieval core fixed).',
    'reference_integrity': reference_integrity,
    'main_performance': rows,
    'wiki_delta_vs_ref': wiki_delta,
    'hotpot_regression_vs_ref': hotpot_reg,
    'generation_diagnostics': diagnostic_rows,
    'variant_interpretation_2wiki': wiki_interp,
    'metric_integrity_status': metric_integrity_status,
    'failures': failures,
    'source_evidence_metrics_json': str(Path(in_json).resolve()),
}
Path(out_json).write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding='utf-8')


def fmt(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return "0.0000"


def md_table(headers, data_rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in data_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

lines = []
lines.append('# Generation-side Minimal Intervention Round')
lines.append('')
lines.append('- Retrieval core is fixed (`r2_plus_path_preserve` core, mid regime).')
lines.append('- This round tests only generation-side minimal interventions to reduce conversion failures.')
lines.append('')
lines.append('## 1. Reference Integrity')
lines.append('')
lines.append(f"- reference_status: {reference_integrity['status']}")
lines.append(f"- metric_integrity_status: {metric_integrity_status}")
lines.append(f"- failure_count: {len(failures)}")
if missing:
    lines.append(f"- missing_references: {missing}")
lines.append('')

ref_headers = ['dataset','variant','R@5','EM','F1','retrieval_ms','generation_ms','total_ms','fallback_rate','answer_present_but_generation_fail_rate','output_overlap_answer_bearing_rate']
ref_rows = []
for r in reference_rows:
    ref_rows.append([
        r['dataset'], r['variant'],
        fmt(r['R@5']), fmt(r['EM']), fmt(r['F1']),
        fmt(r['retrieval_ms'],2), fmt(r['generation_ms'],2), fmt(r['total_ms'],2),
        fmt(r['fallback_rate']), fmt(r['answer_present_but_generation_fail_rate']), fmt(r['output_overlap_answer_bearing_rate']),
    ])
if ref_rows:
    lines.append(md_table(ref_headers, ref_rows))
else:
    lines.append('- no reference rows found')
lines.append('')

lines.append('## 2. Main Performance Table')
lines.append('')
main_headers = ['dataset','variant','R@5','EM','F1','retrieval_ms','generation_ms','total_ms','fallback_rate']
main_rows = []
for r in rows:
    main_rows.append([
        r['dataset'], r['variant'],
        fmt(r['R@5']), fmt(r['EM']), fmt(r['F1']),
        fmt(r['retrieval_ms'],2), fmt(r['generation_ms'],2), fmt(r['total_ms'],2), fmt(r['fallback_rate']),
    ])
lines.append(md_table(main_headers, main_rows) if main_rows else '- no rows')
lines.append('')

lines.append('## 3. 2Wiki Intervention Delta Table (vs ref_reconfirm)')
lines.append('')
w_headers = ['variant','dEM','dF1','d_generation_ms','d_total_ms','d_answer_present_but_generation_fail','d_output_overlap_answer_bearing']
w_rows = []
for r in wiki_delta:
    w_rows.append([
        r['variant'], fmt(r['dEM_vs_ref']), fmt(r['dF1_vs_ref']),
        fmt(r['d_generation_ms_vs_ref'],2), fmt(r['d_total_ms_vs_ref'],2),
        fmt(r['d_answer_present_but_generation_fail_vs_ref']), fmt(r['d_output_overlap_answer_bearing_vs_ref']),
    ])
lines.append(md_table(w_headers, w_rows) if w_rows else '- no 2Wiki deltas available')
lines.append('')

lines.append('## 4. Hotpot Regression Check (vs raw_focus_front_scaffold_light)')
lines.append('')
h_headers = ['variant','dEM','dF1','d_generation_ms','d_total_ms']
h_rows = []
for r in hotpot_reg:
    h_rows.append([
        r['variant'], fmt(r['dEM_vs_ref']), fmt(r['dF1_vs_ref']),
        fmt(r['d_generation_ms_vs_ref'],2), fmt(r['d_total_ms_vs_ref'],2),
    ])
lines.append(md_table(h_headers, h_rows) if h_rows else '- no Hotpot deltas available')
lines.append('')

lines.append('## 5. Generation-side Diagnostic Table')
lines.append('')
d_headers = [
    'dataset','variant','answer_bearing_chunk_present_rate','minimal_support_subset_coverage','equivalent_evidence_coverage',
    'answer_present_but_generation_fail_rate','output_overlap_answer_bearing_rate','avg_first_answer_chunk_rank',
    'avg_first_answer_bundle_rank','avg_evidence_dispersion_score','avg_evidence_duplication_score'
]
d_rows = []
for r in diagnostic_rows:
    d_rows.append([
        r['dataset'], r['variant'],
        fmt(r['answer_bearing_chunk_present_rate']),
        fmt(r['minimal_support_subset_coverage']),
        fmt(r['equivalent_evidence_coverage']),
        fmt(r['answer_present_but_generation_fail_rate']),
        fmt(r['output_overlap_answer_bearing_rate']),
        fmt(r['avg_first_answer_chunk_rank'],3),
        fmt(r['avg_first_answer_bundle_rank'],3),
        fmt(r['avg_evidence_dispersion_score']),
        fmt(r['avg_evidence_duplication_score']),
    ])
lines.append(md_table(d_headers, d_rows) if d_rows else '- no diagnostics available')
lines.append('')

lines.append('## 6. Interpretation')
lines.append('')
if wiki_interp:
    for k in ('quote_then_answer_light','grounded_answer_light','evidence_focus_light'):
        if k in wiki_interp:
            x = wiki_interp[k]
            lines.append(f"- {k}: dF1={fmt(x.get('dF1',0.0))}, dABGF={fmt(x.get('dABGF',0.0))}, dOverlap={fmt(x.get('dOverlap',0.0))}")
else:
    lines.append('- 2Wiki intervention interpretation unavailable (missing reference or runs).')
lines.append('- Positive signal: dF1 >= 0 with lower answer_present_but_generation_fail or higher overlap.')
lines.append('- Stop signal: persistent dF1 drop on both 2Wiki and Hotpot with no diagnostic gain.')
lines.append('')

lines.append('## 7. Artifacts')
lines.append('')
lines.append(f"- reference_integrity.json: {ref_json_path}")
lines.append(f"- metric_integrity_report.json: {payload.get('metric_integrity_report_path','')}")
lines.append(f"- query diagnostics: {payload.get('query_diagnostics_jsonl','')} / {payload.get('query_diagnostics_csv','')}")
lines.append(f"- failures.json: {failures_json}")

Path(out_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out_json)
print(out_md)
PY
}

fallback_rate_check() {
  log_msg "[CHECK] fallback_rate sanity"
  "${PYTHON_BIN}" - "${ROUND_METRICS_JSON}" <<'PY'
import json,sys
p=json.loads(open(sys.argv[1], encoding='utf-8').read())
rows=list(p.get('main_performance',[]) or [])
if not rows:
    print('no_rows')
    raise SystemExit(0)
mx=max(float(r.get('fallback_rate',0.0) or 0.0) for r in rows)
print(f'fallback_rate_max={mx:.4f}')
if mx > 0.20:
    print('warning: fallback_rate is high')
PY
}

log_msg "Round root: ${ROUND_ROOT}"
log_msg "Log root: ${LOG_ROOT}"
log_msg "Datasets: ${DATASETS}"
log_msg "Variants: ${VARIANTS}"
log_msg "n_samples: ${N_SAMPLES}"
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

log_msg "Manifest: ${RUN_MANIFEST_JSON}"
log_msg "Environment snapshot: ${ENV_SNAPSHOT_JSON}"

if [[ "${MANIFEST_ONLY}" == "true" ]]; then
  log_msg "manifest-only mode: stopping before execution"
  exit 0
fi

run_stage1_references
run_stage2_interventions
run_stage3_optional_ab

check_cache_meta_race
aggregate_round_outputs
fallback_rate_check

if [[ ${run_failed_count} -gt 0 ]]; then
  log_msg "[SUMMARY] Completed with failures: ${run_failed_count}. See ${FAILURES_JSON}"
else
  log_msg "[SUMMARY] Completed without failures"
fi

log_msg "reference_integrity.json: ${REFERENCE_INTEGRITY_JSON}"
log_msg "round_metrics.json: ${ROUND_METRICS_JSON}"
log_msg "round_metrics.md: ${ROUND_METRICS_MD}"
log_msg "query diagnostics: ${QUERY_DIAG_JSONL} / ${QUERY_DIAG_CSV}"
