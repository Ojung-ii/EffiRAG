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
VARIANTS="baseline_mid_reconfirm,ref_reconfirm"
N_SAMPLES="1000"
QA_ROOT="data/qa"
CORPUS_ROOT="/home/ojungii/HippoRAG2/dataset"
OUTPUT_ROOT="outputs/evidence_sufficiency_audit"
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
RUN_STAGE4="auto"   # auto|true|false
RUN_STAGE4_AB3="false"

# Pass-through path overrides.
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

usage() {
  cat <<'EOF'
Usage: bash scripts/run_evidence_sufficiency_audit.sh [options]

Main options:
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
  --run-preflight [true|false]
  --manifest-only
  --resume-round-root PATH

Stage4:
  --run-stage4 auto|true|false
  --run-stage4-ab3 [true|false]

Dataset overrides:
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

    --run-stage4) RUN_STAGE4="$2"; shift 2 ;;
    --run-stage4-ab3)
      if [[ $# -gt 1 && "${2:-}" != --* ]]; then
        RUN_STAGE4_AB3="$(parse_bool "$2")"; shift 2
      else
        RUN_STAGE4_AB3="true"; shift 1
      fi
      ;;

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
RUN_STAGE4_AB3="$(parse_bool "${RUN_STAGE4_AB3}")"

[[ -n "${SEQUENTIAL}" ]] || die "Invalid --sequential"
[[ -n "${SKIP_COMPLETED}" ]] || die "Invalid --skip-completed"
[[ -n "${STOP_ON_FAILURE}" ]] || die "Invalid --stop-on-failure"
[[ -n "${RUN_PREFLIGHT}" ]] || die "Invalid --run-preflight"
[[ -n "${RUN_STAGE4_AB3}" ]] || die "Invalid --run-stage4-ab3"

RUN_STAGE4="$(echo "${RUN_STAGE4}" | tr '[:upper:]' '[:lower:]')"
case "${RUN_STAGE4}" in
  auto|true|false) ;;
  *) die "Invalid --run-stage4. Use auto|true|false" ;;
esac

if [[ "${SEQUENTIAL}" != "true" ]]; then
  log_msg "[WARN] sequential-only runner; forcing --sequential true"
  SEQUENTIAL="true"
fi

if [[ -n "${RESUME_ROUND_ROOT}" ]]; then
  ROUND_ROOT="${RESUME_ROUND_ROOT}"
  ROUND_SOURCE="resume"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="${OUTPUT_ROOT}/${RUN_STAMP}"
  ROUND_SOURCE="new"
fi

LOG_ROOT="logs/evidence_sufficiency_audit/$(basename "${ROUND_ROOT}")"
mkdir -p "${ROUND_ROOT}" "${LOG_ROOT}"

MANIFEST_PATH="${ROUND_ROOT}/run_manifest.json"
RUN_RECORDS_PATH="${ROUND_ROOT}/run_records.tsv"
ROUND_METRICS_JSON="${ROUND_ROOT}/round_metrics.json"
ROUND_METRICS_MD="${ROUND_ROOT}/round_metrics.md"
INTEGRITY_JSON="${ROUND_ROOT}/metric_integrity_report.json"
QUERY_DIAG_JSONL="${ROUND_ROOT}/evidence_sufficiency_query_diagnostics_all.jsonl"
QUERY_DIAG_CSV="${ROUND_ROOT}/evidence_sufficiency_query_diagnostics_all.csv"

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

  "${PYTHON_BIN}" - "${RUN_RECORDS_PATH}" \
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

create_stage4_cfg() {
  local out_cfg="$1"
  local variant="$2"
  local order_strategy="$3"
  mkdir -p "$(dirname "${out_cfg}")"
  "${PYTHON_BIN}" - "${RAG_BASE_CFG}" "${out_cfg}" "${variant}" "${order_strategy}" "${MODEL_NAME}" "${UNIVERSAL_EMBED_MAX_LEN}" "${UNIVERSAL_EMBED_MAX_CHARS}" <<'PY'
import sys
from pathlib import Path
import yaml

base_cfg, out_cfg, variant, order_strategy, model_name, embed_len, embed_chars = sys.argv[1:8]
cfg = yaml.safe_load(Path(base_cfg).read_text(encoding='utf-8')) or {}

cfg['dataset'] = '2wikimultihopqa'
cfg['canonical_variant_name'] = str(variant)
cfg['retrieval_objective_mode'] = 'r2_plus_path_preserve'
cfg['semantic_topn_chunk'] = 25
cfg['run_qa'] = True
cfg['retrieval_only'] = False
cfg['generator'] = 'vllm'
cfg['model_name'] = str(model_name)
cfg['openie_mode'] = 'llm'
cfg['openie_model_name'] = str(model_name)
cfg['openie_local_files_only'] = True
cfg['evaluator_mode'] = 'hipporag2_parity'
cfg['embedding_max_length'] = int(embed_len)
cfg['embedding_text_max_chars'] = int(embed_chars)
cfg['graph_cache_dir'] = cfg.get('graph_cache_dir', 'outputs/index_cache')
cfg['force_rebuild_graph_index'] = False
cfg['num_workers'] = 1
cfg['stagewise_loss_funnel_enabled'] = True
cfg['order_strategy'] = str(order_strategy)

Path(out_cfg).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding='utf-8')
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

stage4_run_variant() {
  local variant="$1"
  local order_strategy="$2"
  local qa_path="$3"
  local corpus_path="$4"

  local stage="s4"
  local dataset="2wikimultihopqa"
  local profile="strong_ref_2wiki"
  local cfg_dir="${ROUND_ROOT}/configs/${stage}/${variant}/${dataset}"
  local cfg_path="${cfg_dir}/rag.yaml"
  local out_dir="${ROUND_ROOT}/runs/${stage}/${variant}/rag"
  local log_path="${LOG_ROOT}/${stage}__${variant}__${dataset}.log"
  mkdir -p "${cfg_dir}" "${out_dir}" "$(dirname "${log_path}")"

  create_stage4_cfg "${cfg_path}" "${variant}" "${order_strategy}"

  if [[ "${SKIP_COMPLETED}" == "true" ]]; then
    local existing
    existing="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
    if [[ -n "${existing}" ]] && is_complete_summary "${existing}" "${N_SAMPLES}" "rag_query_results.jsonl" >/dev/null 2>&1; then
      log_msg "[SKIP] stage=${stage} variant=${variant} dataset=${dataset}"
      upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
        "${qa_path}" "${corpus_path}" "${existing}" "${cfg_path}" "${log_path}" \
        "skipped" "" "" "0" "${N_SAMPLES}" ""
      return 0
    fi
  fi

  local start_ts end_ts start_epoch end_epoch elapsed
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

  log_msg "[RUN] stage=${stage} variant=${variant} dataset=${dataset} order_strategy=${order_strategy}"
  {
    echo "[COMMAND] ${cmd[*]}"
    echo "[ORDER_STRATEGY] ${order_strategy}"
  } >> "${log_path}"

  set +e
  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
  local rc=${PIPESTATUS[0]}
  set -e

  end_epoch="$(date +%s)"
  end_ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  elapsed="$((end_epoch - start_epoch))"

  if [[ ${rc} -ne 0 ]]; then
    upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
      "${qa_path}" "${corpus_path}" "" "${cfg_path}" "${log_path}" \
      "failed" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" "exit_code_${rc}"
    log_msg "[FAIL] stage=${stage} variant=${variant} dataset=${dataset}"
    return ${rc}
  fi

  local summary_path
  summary_path="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
  if [[ -z "${summary_path}" ]]; then
    upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
      "${qa_path}" "${corpus_path}" "" "${cfg_path}" "${log_path}" \
      "failed" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" "missing_summary"
    log_msg "[FAIL] stage=${stage} variant=${variant} dataset=${dataset} missing summary"
    return 1
  fi

  upsert_record "${stage}" "${variant}" "${dataset}" "${profile}" "rag" \
    "${qa_path}" "${corpus_path}" "${summary_path}" "${cfg_path}" "${log_path}" \
    "ok" "${start_ts}" "${end_ts}" "${elapsed}" "${N_SAMPLES}" ""
  log_msg "[DONE] stage=${stage} variant=${variant} dataset=${dataset}"
  return 0
}

run_stage1_base() {
  log_msg "[Stage 1] Baseline/ref sequential run + base aggregation"
  local -a cmd
  cmd=(
    bash scripts/run_multidataset_generalization.sh
    --datasets "${DATASETS}"
    --variants "${VARIANTS}"
    --n-samples "${N_SAMPLES}"
    --qa-root "${QA_ROOT}"
    --corpus-root "${CORPUS_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --graph-cache-dir "${GRAPH_CACHE_DIR}"
    --llm-base-url "${LLM_BASE_URL}"
    --llm-api-key "${LLM_API_KEY}"
    --model-name "${MODEL_NAME}"
    --sequential "${SEQUENTIAL}"
    --skip-completed "${SKIP_COMPLETED}"
    --stop-on-failure "${STOP_ON_FAILURE}"
    --run-preflight "${RUN_PREFLIGHT}"
    --resume-round-root "${ROUND_ROOT}"
  )

  if [[ "${MANIFEST_ONLY}" == "true" ]]; then
    cmd+=(--manifest-only)
  fi
  if [[ -n "${DATA_PATH_HOTPOTQA}" ]]; then
    cmd+=(--data-path-hotpotqa "${DATA_PATH_HOTPOTQA}")
  fi
  if [[ -n "${DATA_PATH_2WIKIMULTIHOPQA}" ]]; then
    cmd+=(--data-path-2wikimultihopqa "${DATA_PATH_2WIKIMULTIHOPQA}")
  fi
  if [[ -n "${DATA_PATH_MUSIQUE}" ]]; then
    cmd+=(--data-path-musique "${DATA_PATH_MUSIQUE}")
  fi
  if [[ -n "${DATA_PATH_POPQA}" ]]; then
    cmd+=(--data-path-popqa "${DATA_PATH_POPQA}")
  fi
  if [[ -n "${CORPUS_PATH_HOTPOTQA}" ]]; then
    cmd+=(--corpus-path-hotpotqa "${CORPUS_PATH_HOTPOTQA}")
  fi
  if [[ -n "${CORPUS_PATH_2WIKIMULTIHOPQA}" ]]; then
    cmd+=(--corpus-path-2wikimultihopqa "${CORPUS_PATH_2WIKIMULTIHOPQA}")
  fi
  if [[ -n "${CORPUS_PATH_MUSIQUE}" ]]; then
    cmd+=(--corpus-path-musique "${CORPUS_PATH_MUSIQUE}")
  fi
  if [[ -n "${CORPUS_PATH_POPQA}" ]]; then
    cmd+=(--corpus-path-popqa "${CORPUS_PATH_POPQA}")
  fi

  "${cmd[@]}"
}

run_stage2_and_3_aggregate() {
  log_msg "[Stage 2/3] Metric integrity + evidence sufficiency + family summary aggregation"
  "${PYTHON_BIN}" scripts/aggregate_evidence_sufficiency.py \
    --round-root "${ROUND_ROOT}" \
    --manifest "${MANIFEST_PATH}" \
    --run-records "${RUN_RECORDS_PATH}" \
    --output-json "${ROUND_METRICS_JSON}" \
    --output-md "${ROUND_METRICS_MD}" \
    --metric-integrity-report "${INTEGRITY_JSON}" \
    --query-diag-jsonl "${QUERY_DIAG_JSONL}" \
    --query-diag-csv "${QUERY_DIAG_CSV}" \
    --base-variants "baseline_mid_reconfirm,ref_reconfirm"
}

run_stage4_optional() {
  log_msg "[Stage 4] Optional tiny 2Wiki scaffold A/B"

  local cond
  cond="$(${PYTHON_BIN} - <<'PY' "${ROUND_METRICS_JSON}"
import json,sys
p=json.load(open(sys.argv[1]))
print('true' if bool(p.get('stage4_condition_met', False)) else 'false')
PY
)"

  local should_run="false"
  if [[ "${RUN_STAGE4}" == "true" ]]; then
    should_run="true"
  elif [[ "${RUN_STAGE4}" == "auto" && "${cond}" == "true" ]]; then
    should_run="true"
  fi

  if [[ "${should_run}" != "true" ]]; then
    log_msg "[Stage 4] skipped (run_stage4=${RUN_STAGE4}, condition=${cond})"
    return 0
  fi

  local has_2wiki
  has_2wiki="$(${PYTHON_BIN} - <<'PY' "${MANIFEST_PATH}"
import json,sys
m=json.load(open(sys.argv[1]))
ds=[str(x) for x in ((m.get('selection',{}) or {}).get('datasets',[]) or [])]
print('true' if '2wikimultihopqa' in ds else 'false')
PY
)"
  if [[ "${has_2wiki}" != "true" ]]; then
    log_msg "[Stage 4] skipped: 2wikimultihopqa not selected"
    return 0
  fi

  local wiki_qa_path wiki_corpus_path ref_summary
  wiki_qa_path="$(${PYTHON_BIN} - <<'PY' "${MANIFEST_PATH}"
import json,sys
m=json.load(open(sys.argv[1]))
print(((m.get('dataset_paths',{}) or {}).get('2wikimultihopqa',{}) or {}).get('qa_path',''))
PY
)"
  wiki_corpus_path="$(${PYTHON_BIN} - <<'PY' "${MANIFEST_PATH}"
import json,sys
m=json.load(open(sys.argv[1]))
print(((m.get('dataset_paths',{}) or {}).get('2wikimultihopqa',{}) or {}).get('corpus_path',''))
PY
)"

  [[ -n "${wiki_qa_path}" ]] || die "Stage4: missing 2Wiki qa_path"
  [[ -n "${wiki_corpus_path}" ]] || die "Stage4: missing 2Wiki corpus_path"

  # Alias wiki_ref_reconfirm to existing ref_reconfirm summary if present.
  ref_summary="$(${PYTHON_BIN} - <<'PY' "${RUN_RECORDS_PATH}"
import csv,sys
sp=''
with open(sys.argv[1], encoding='utf-8') as f:
    r=csv.DictReader(f, delimiter='\t')
    for row in r:
        if row.get('stage')=='s1' and row.get('variant')=='ref_reconfirm' and row.get('dataset')=='2wikimultihopqa' and row.get('status') in {'ok','skipped'}:
            sp=row.get('summary_path','') or ''
print(sp)
PY
)"
  if [[ -n "${ref_summary}" && -f "${ref_summary}" ]]; then
    upsert_record "s4" "wiki_ref_reconfirm" "2wikimultihopqa" "strong_ref_2wiki" "rag" \
      "${wiki_qa_path}" "${wiki_corpus_path}" "${ref_summary}" "" "" "skipped" "" "" "0" "${N_SAMPLES}" "aliased_from_ref_reconfirm"
  fi

  local failed=0
  stage4_run_variant "wiki_scaffold_ab_1" "score+raw_focus_scaffold_ab1" "${wiki_qa_path}" "${wiki_corpus_path}" || failed=$((failed+1))
  stage4_run_variant "wiki_scaffold_ab_2" "score+raw_focus_scaffold_ab2" "${wiki_qa_path}" "${wiki_corpus_path}" || failed=$((failed+1))
  if [[ "${RUN_STAGE4_AB3}" == "true" ]]; then
    stage4_run_variant "wiki_scaffold_ab_3" "score+raw_focus_scaffold_ab3" "${wiki_qa_path}" "${wiki_corpus_path}" || failed=$((failed+1))
  fi

  # Re-aggregate to include stage4 runs.
  run_stage2_and_3_aggregate

  if [[ ${failed} -gt 0 ]]; then
    log_msg "[Stage 4] finished with ${failed} failed variant(s)"
  else
    log_msg "[Stage 4] completed"
  fi
}

log_msg "Round root: ${ROUND_ROOT}"
log_msg "Log root: ${LOG_ROOT}"
log_msg "Datasets: ${DATASETS}"
log_msg "Variants: ${VARIANTS}"
log_msg "n_samples: ${N_SAMPLES}"
log_msg "LLM endpoint: ${LLM_BASE_URL}"
log_msg "Model: ${MODEL_NAME}"

# Stage 1 run (includes preflight checks through base runner).
run_stage1_base

if [[ "${MANIFEST_ONLY}" == "true" ]]; then
  log_msg "manifest-only mode: skip Stage2/3/4"
  exit 0
fi

# Stage 2/3 aggregation.
run_stage2_and_3_aggregate

# Stage 4 conditional tiny A/B.
run_stage4_optional

log_msg "[DONE] round_metrics.json=${ROUND_METRICS_JSON}"
log_msg "[DONE] round_metrics.md=${ROUND_METRICS_MD}"
log_msg "[DONE] metric_integrity_report.json=${INTEGRITY_JSON}"
log_msg "[DONE] query diagnostics: ${QUERY_DIAG_JSONL} / ${QUERY_DIAG_CSV}"
