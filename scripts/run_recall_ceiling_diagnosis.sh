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
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LIMIT_RAG="${LIMIT_RAG:-1000}"
PROMPT_SANITY_LIMIT="${PROMPT_SANITY_LIMIT:-200}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-true}"
RUN_PREFLIGHT_CHECKS="${RUN_PREFLIGHT_CHECKS:-true}"
AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-true}"
FORCE_NEW_ROUND="${FORCE_NEW_ROUND:-false}"
RUN_SWEEP_HIGH="${RUN_SWEEP_HIGH:-false}"

RUN_STAGE2_CANDIDATE_AUDIT="${RUN_STAGE2_CANDIDATE_AUDIT:-auto}"
RUN_STAGE2_ANCHOR_AUDIT="${RUN_STAGE2_ANCHOR_AUDIT:-auto}"
RUN_STAGE2_CORRIDOR_AUDIT="${RUN_STAGE2_CORRIDOR_AUDIT:-auto}"

UNIVERSAL_EMBED_MAX_LEN="${UNIVERSAL_EMBED_MAX_LEN:-256}"
UNIVERSAL_EMBED_MAX_CHARS="${UNIVERSAL_EMBED_MAX_CHARS:-800}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
RAG_BASE_CFG="configs/canonical/rag_entity_first_chunk_grounded.yaml"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-0}"

HOTPOT_D1_REF_SUMMARY="${HOTPOT_D1_REF_SUMMARY:-outputs/exp_retrieval_gap_recheck/hotpotqa/D1/rag/hotpotqa/20260406_053010_574786/rag_summary.json}"
HOTPOT_F2_REF_SUMMARY="${HOTPOT_F2_REF_SUMMARY:-outputs/exp_retrieval_gap/hotpotqa/F2/rag/hotpotqa/20260405_141352_352678/rag_summary.json}"
WIKI_D1_REF_SUMMARY="${WIKI_D1_REF_SUMMARY:-outputs/exp_retrieval_gap_recheck/2wikimultihopqa/D1/rag/2wikimultihopqa/20260406_054816_597964/rag_summary.json}"
WIKI_E1_REF_SUMMARY="${WIKI_E1_REF_SUMMARY:-outputs/exp_retrieval_gap/2wikimultihopqa/E1/rag/2wikimultihopqa/20260405_144256_613687/rag_summary.json}"
WIKI_W3_REF_SUMMARY="${WIKI_W3_REF_SUMMARY:-}"
D2_HOTPOT_REF_SUMMARY="${D2_HOTPOT_REF_SUMMARY:-outputs/exp_retrieval_gap_recheck/hotpotqa/D2/rag/hotpotqa/20260406_053942_373258/rag_summary.json}"
D2_WIKI_REF_SUMMARY="${D2_WIKI_REF_SUMMARY:-outputs/exp_retrieval_gap/2wikimultihopqa/D2/rag/2wikimultihopqa/20260405_143345_219316/rag_summary.json}"

if [[ -n "${RESUME_ROUND_ROOT:-}" ]]; then
  ROUND_ROOT="${RESUME_ROUND_ROOT}"
  ROUND_ROOT_SOURCE="resume_env"
elif [[ "${FORCE_NEW_ROUND}" == "true" ]]; then
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="outputs/recall_ceiling_diagnosis/${RUN_STAMP}"
  ROUND_ROOT_SOURCE="force_new"
elif [[ "${AUTO_RESUME_LATEST}" == "true" ]]; then
  LATEST_ROUND_ROOT="$(find outputs/recall_ceiling_diagnosis -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${LATEST_ROUND_ROOT}" ]]; then
    ROUND_ROOT="${LATEST_ROUND_ROOT}"
    ROUND_ROOT_SOURCE="auto_resume_latest"
  else
    RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
    ROUND_ROOT="outputs/recall_ceiling_diagnosis/${RUN_STAMP}"
    ROUND_ROOT_SOURCE="new_round"
  fi
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="outputs/recall_ceiling_diagnosis/${RUN_STAMP}"
  ROUND_ROOT_SOURCE="new_round"
fi

LOG_ROOT="logs/recall_ceiling_diagnosis/$(basename "${ROUND_ROOT}")"
CFG_ROOT="${ROUND_ROOT}/configs"
RUN_ROOT="${ROUND_ROOT}/runs"
mkdir -p "${ROUND_ROOT}" "${LOG_ROOT}" "${CFG_ROOT}" "${RUN_ROOT}"

RECORD_TSV="${ROUND_ROOT}/run_records.tsv"
if [[ ! -f "${RECORD_TSV}" ]]; then
  printf "stage\tvariant\tdataset\tprofile\tkind\tsummary_path\tlog_path\tstatus\n" > "${RECORD_TSV}"
fi

REF_BUNDLE_JSON="${ROUND_ROOT}/reference_bundle.json"
AUDIT_PLAN_JSON="${ROUND_ROOT}/stage2_audit_plan.json"
PREFLIGHT_HOTPOT_JSON="${ROUND_ROOT}/preflight_hotpot_prompt_sanity.json"
PREFLIGHT_HOTPOT_MD="${ROUND_ROOT}/preflight_hotpot_prompt_sanity.md"
PREFLIGHT_WIKI_JSON="${ROUND_ROOT}/preflight_wiki_prompt_sanity.json"
PREFLIGHT_WIKI_MD="${ROUND_ROOT}/preflight_wiki_prompt_sanity.md"
ROUND_METRICS_JSON="${ROUND_ROOT}/round_metrics.json"
ROUND_METRICS_MD="${ROUND_ROOT}/round_metrics.md"
QUERY_DIAG_JSONL="${ROUND_ROOT}/diagnosis_query_funnel_all.jsonl"
QUERY_DIAG_CSV="${ROUND_ROOT}/diagnosis_query_funnel_all.csv"

log_msg() {
  local now
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${now}] $*"
}

resolve_vllm_max_model_len() {
  local raw="${VLLM_MAX_MODEL_LEN}"
  if [[ -n "${raw}" && "${raw}" != "0" ]]; then
    VLLM_MAX_MODEL_LEN_RESOLVED="${raw}"
    VLLM_MAX_MODEL_LEN_SOURCE="env"
    return 0
  fi

  local ps_line
  ps_line="$(ps -ef | rg 'vllm serve' | rg -v 'rg|grep' | head -n 1 || true)"
  if [[ -n "${ps_line}" ]]; then
    local parsed
    parsed="$(python3 - <<'PY' "${ps_line}"
import re,sys
line=sys.argv[1]
patterns=[r'--max-model-len\s+(\d+)', r'--max_model_len\s+(\d+)', r'--max-model-len=(\d+)', r'--max_model_len=(\d+)']
for p in patterns:
    m=re.search(p,line)
    if m:
        print(m.group(1))
        raise SystemExit(0)
print('')
PY
)"
    if [[ -n "${parsed}" ]]; then
      VLLM_MAX_MODEL_LEN_RESOLVED="${parsed}"
      VLLM_MAX_MODEL_LEN_SOURCE="ps"
      return 0
    fi
  fi

  VLLM_MAX_MODEL_LEN_RESOLVED="4096"
  VLLM_MAX_MODEL_LEN_SOURCE="default_assumed"
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

try:
    _ = json.loads(summary_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

qpath = summary_path.with_name(query_name)
if not qpath.exists():
    raise SystemExit(1)

cnt = 0
with qpath.open("r", encoding="utf-8") as f:
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
  local summary_path="$6"
  local log_path="$7"
  local status="$8"
  "${PYTHON_BIN}" - "${RECORD_TSV}" "${stage}" "${variant}" "${dataset}" "${profile}" "${kind}" "${summary_path}" "${log_path}" "${status}" <<'PY'
import sys
from pathlib import Path

tsv = Path(sys.argv[1])
stage,variant,dataset,profile,kind,summary_path,log_path,status = sys.argv[2:10]
header = "stage\tvariant\tdataset\tprofile\tkind\tsummary_path\tlog_path\tstatus\n"
rows = []
seen = set()
if tsv.exists():
    with tsv.open("r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 8:
                continue
            key = tuple(p[:5])
            if key in seen:
                continue
            rows.append(p)
            seen.add(key)

target = (stage, variant, dataset, profile, kind)
updated = False
for r in rows:
    if tuple(r[:5]) == target:
        r[5] = summary_path
        r[6] = log_path
        r[7] = status
        updated = True
        break
if not updated:
    rows.append([stage, variant, dataset, profile, kind, summary_path, log_path, status])

tsv.parent.mkdir(parents=True, exist_ok=True)
with tsv.open("w", encoding="utf-8") as f:
    f.write(header)
    for r in rows:
        f.write("\t".join(r) + "\n")
PY
}

assert_cfg_race_safe() {
  local cfg_path="$1"
  "${PYTHON_BIN}" - "${cfg_path}" <<'PY'
import sys,yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
if int(cfg.get('num_workers',1) or 1) > 1 and bool(cfg.get('force_rebuild_graph_index', False)):
    raise SystemExit('unsafe cache/meta race config')
print('safe')
PY
}

build_reference_bundle() {
  "${PYTHON_BIN}" - \
    "${REF_BUNDLE_JSON}" \
    "${HOTPOT_D1_REF_SUMMARY}" \
    "${HOTPOT_F2_REF_SUMMARY}" \
    "${WIKI_D1_REF_SUMMARY}" \
    "${WIKI_E1_REF_SUMMARY}" \
    "${WIKI_W3_REF_SUMMARY}" \
    "${D2_HOTPOT_REF_SUMMARY}" \
    "${D2_WIKI_REF_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

out_json, hotpot_d1, hotpot_f2, wiki_d1, wiki_e1, wiki_w3, d2_hotpot, d2_wiki = sys.argv[1:9]


def _safe_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)


def _load(path):
    p = Path(path)
    if not str(path or '').strip() or not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def _metric(summary, *keys):
    if not isinstance(summary, dict):
        return 0.0
    for key in keys:
        if key in summary:
            return _safe_float(summary.get(key), 0.0)
    return 0.0


def _r_at(summary, k):
    if not isinstance(summary, dict):
        return 0.0
    d = _safe_float(summary.get(f'supporting_fact_recall_at_{k}', 0.0), 0.0)
    if d > 0.0:
        return d
    s = summary.get('supporting_fact_recall_at_k', {}) or {}
    if str(k) in s:
        return _safe_float(s[str(k)], 0.0)
    if k in s:
        return _safe_float(s[k], 0.0)
    s2 = summary.get('recall_at_k', {}) or {}
    if str(k) in s2:
        return _safe_float(s2[str(k)], 0.0)
    if k in s2:
        return _safe_float(s2[k], 0.0)
    return 0.0


def _item(label, path, summary):
    return {
        'label': str(label),
        'path': str(Path(path).resolve()) if path and Path(path).exists() else '',
        'r1': _r_at(summary, 1),
        'r5': _r_at(summary, 5),
        'r20': _r_at(summary, 20),
        'em': _metric(summary, 'em', 'EM'),
        'f1': _metric(summary, 'f1', 'F1'),
        'retrieval_ms': _safe_float(summary.get('retrieval_latency_ms', summary.get('retrieval_ms', 0.0)), 0.0),
        'generation_ms': _safe_float(summary.get('generation_latency_ms', summary.get('generation_ms', 0.0)), 0.0),
        'total_ms': _safe_float(summary.get('total_latency_ms', summary.get('total_ms', 0.0)), 0.0),
    }

hotpot_d1_s = _load(hotpot_d1) or {}
hotpot_f2_s = _load(hotpot_f2) or {}
wiki_d1_s = _load(wiki_d1) or {}
wiki_e1_s = _load(wiki_e1) or {}

wiki_w3_resolved = ''
if str(wiki_w3 or '').strip() and Path(wiki_w3).exists():
    wiki_w3_resolved = str(Path(wiki_w3).resolve())
else:
    cands = sorted(Path('outputs').glob('**/2wikimultihopqa/W3/rag/2wikimultihopqa/*/rag_summary.json'))
    if cands:
        wiki_w3_resolved = str(cands[-1].resolve())
wiki_w3_s = _load(wiki_w3_resolved) if wiki_w3_resolved else None

wiki_best_s = wiki_w3_s if isinstance(wiki_w3_s, dict) else wiki_e1_s
wiki_best_path = wiki_w3_resolved if isinstance(wiki_w3_s, dict) else wiki_e1
wiki_best_label = 'w3' if isinstance(wiki_w3_s, dict) else 'e1'

d2_hotpot_s = _load(d2_hotpot) or {}
d2_wiki_s = _load(d2_wiki) or {}

# Fixed HippoRAG2 table provided by user (2026-03-23/24 KST)
hippo = {
    'hotpotqa': {
        'label': 'HippoRAG2',
        'path': 'fixed_table://hipporag2/hotpotqa/2026-03-23',
        'r1': 0.4380,
        'r5': 0.9500,
        'r20': 0.9890,
        'em': 0.3350,
        'f1': 0.5154,
        'retrieval_ms': 207.7214 * 1000.0,
        'generation_ms': 0.3412 * 1000.0,
        'total_ms': 208.0626 * 1000.0,
    },
    '2wikimultihopqa': {
        'label': 'HippoRAG2',
        'path': 'fixed_table://hipporag2/2wikimultihopqa/2026-03-24',
        'r1': 0.4088,
        'r5': 0.8955,
        'r20': 0.9560,
        'em': 0.2670,
        'f1': 0.4499,
        'retrieval_ms': 873.6446 * 1000.0,
        'generation_ms': 2003.5872 * 1000.0,
        'total_ms': 2877.2318 * 1000.0,
    },
}

payload = {
    'wiki_best_label': wiki_best_label,
    'hotpotqa': {
        'd1': _item('D1', hotpot_d1, hotpot_d1_s),
        'best_non_oracle': _item('F2', hotpot_f2, hotpot_f2_s),
        'd2': _item('D2', d2_hotpot, d2_hotpot_s),
        'hipporag2': hippo['hotpotqa'],
    },
    '2wikimultihopqa': {
        'd1': _item('D1', wiki_d1, wiki_d1_s),
        'best_non_oracle': _item('WIKI_BEST', wiki_best_path, wiki_best_s),
        'd2': _item('D2', d2_wiki, d2_wiki_s),
        'hipporag2': hippo['2wikimultihopqa'],
    },
    'prompt_sanity_reference': {
        'hotpotqa': str(Path(hotpot_f2).resolve()) if hotpot_f2 and Path(hotpot_f2).exists() else '',
        '2wikimultihopqa': str(Path(wiki_best_path).resolve()) if wiki_best_path and Path(wiki_best_path).exists() else '',
    },
}

Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(out_json)
PY
}

create_variant_cfg() {
  local base_cfg="$1"
  local out_cfg="$2"
  local dataset="$3"
  local variant_tag="$4"
  local retrieval_mode="$5"
  local knob_json="$6"

  mkdir -p "$(dirname "${out_cfg}")"
  "${PYTHON_BIN}" - \
    "${base_cfg}" \
    "${out_cfg}" \
    "${dataset}" \
    "${variant_tag}" \
    "${retrieval_mode}" \
    "${knob_json}" \
    "${UNIVERSAL_EMBED_MAX_LEN}" \
    "${UNIVERSAL_EMBED_MAX_CHARS}" \
    "${MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

base_cfg, out_cfg, dataset, variant_tag, retrieval_mode, knob_json, embed_len, embed_chars, model_name = sys.argv[1:10]
cfg = yaml.safe_load(Path(base_cfg).read_text(encoding='utf-8')) or {}
knobs = json.loads(knob_json)

cfg['dataset'] = str(dataset)
cfg['canonical_variant_name'] = str(variant_tag)

# Round invariants
cfg['embedding_max_length'] = int(embed_len)
cfg['embedding_text_max_chars'] = int(embed_chars)
cfg['retrieval_objective_mode'] = str(retrieval_mode)
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

# Keep diagnostics on
cfg['stagewise_loss_funnel_enabled'] = True

# Prevent stale objective toggles from leaking across rounds
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

# Keep shortlist constraints coherent
shortlist = int(cfg.get('phase1_run_shortlist_topk', 2) or 2)
preshort = int(cfg.get('phase1_run_preshortlist_topm', shortlist) or shortlist)
full_topk = int(cfg.get('phase1_full_run_score_topk', preshort) or preshort)
cfg['phase1_run_shortlist_topk'] = max(1, shortlist)
cfg['phase1_run_preshortlist_topm'] = max(cfg['phase1_run_shortlist_topk'], preshort)
cfg['phase1_full_run_score_topk'] = max(cfg['phase1_run_shortlist_topk'], cfg['phase1_run_preshortlist_topm'], full_topk)

Path(out_cfg).parent.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(cfg, open(out_cfg, 'w', encoding='utf-8'), sort_keys=False, allow_unicode=False)
PY

  assert_cfg_race_safe "${out_cfg}" >/dev/null
}

annotate_summary_with_refs() {
  local summary_path="$1"
  local dataset="$2"
  "${PYTHON_BIN}" - "${summary_path}" "${dataset}" "${REF_BUNDLE_JSON}" <<'PY'
import json
import sys
from pathlib import Path
from effirag.utils import compute_rag_derived_metrics

summary_path = Path(sys.argv[1])
dataset = str(sys.argv[2])
refs = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
summary = json.loads(summary_path.read_text(encoding='utf-8'))

d2_f1 = float((((refs.get(dataset, {}) or {}).get('d2', {}) or {}).get('f1', 0.0) or 0.0))
hippo_f1 = float((((refs.get(dataset, {}) or {}).get('hipporag2', {}) or {}).get('f1', 0.0) or 0.0))
hippo_r5 = float((((refs.get(dataset, {}) or {}).get('hipporag2', {}) or {}).get('r5', 0.0) or 0.0))

summary['oracle_ceiling_f1'] = d2_f1
summary['hipporag2_reference_f1'] = hippo_f1
summary['hipporag2_reference_recall_at_5'] = hippo_r5
summary.update(compute_rag_derived_metrics(summary))
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

export_query_funnel_diagnostics() {
  local summary_path="$1"
  local stage="$2"
  local variant="$3"
  local dataset="$4"

  "${PYTHON_BIN}" - "${summary_path}" "${stage}" "${variant}" "${dataset}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
stage = str(sys.argv[2])
variant = str(sys.argv[3])
dataset = str(sys.argv[4])
qpath = summary_path.with_name('rag_query_results.jsonl')
if not qpath.exists():
    raise SystemExit(0)

rows = []
with qpath.open('r', encoding='utf-8') as f:
    for line in f:
        line = str(line or '').strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

out_jsonl = summary_path.with_name('diagnosis_query_funnel.jsonl')
out_csv = summary_path.with_name('diagnosis_query_funnel.csv')
out_summary = summary_path.with_name('diagnosis_funnel_summary.json')

fields = [
    'stage',
    'variant',
    'dataset',
    'query_id',
    'answer_correct',
    'anchor_hit',
    'proposal_hit',
    'phase1_hit',
    'final_chunk_hit',
    'rendered_hit',
    'fallback',
    'oracle_answer_support_exists',
    'gold_evidence_drop_stage',
]

written = []
for row in rows:
    retrieval = (row.get('retrieval', {}) or {})
    diagnostics = (retrieval.get('diagnostics', {}) or {})
    funnel = (diagnostics.get('stagewise_loss_funnel', {}) or {})

    def _f(v, d=0.0):
        try:
            return float(v)
        except Exception:
            return float(d)

    anchor_hit = 1 if _f(funnel.get('anchor_hit_rate', diagnostics.get('anchor_hit_rate', 0.0))) > 0.0 else 0
    proposal_hit = 1 if _f(funnel.get('proposal_hit_rate', diagnostics.get('proposal_hit_rate', 0.0))) > 0.0 else 0
    phase1_hit = 1 if _f(funnel.get('phase1_hit_rate', diagnostics.get('phase1_hit_rate', 0.0))) > 0.0 else 0
    final_chunk_hit = 1 if _f(funnel.get('final_chunk_hit_rate', diagnostics.get('final_chunk_hit_rate', 0.0))) > 0.0 else 0
    rendered_hit = 1 if _f(funnel.get('rendered_hit_rate', diagnostics.get('rendered_hit_rate', 0.0))) > 0.0 else 0

    metrics = (row.get('metrics', {}) or {})
    em = _f(metrics.get('em', metrics.get('EM', 0.0)))
    f1 = _f(metrics.get('f1', metrics.get('F1', 0.0)))
    answer_correct = 1 if (em >= 0.5 or f1 >= 0.5) else 0
    fallback = 1 if bool(row.get('generation_fallback', False)) else 0
    oracle_support_exists = 1 if int(_f(funnel.get('gold_text_unit_total', 0.0))) > 0 else 0

    drop_stage = 'none'
    if anchor_hit <= 0:
        drop_stage = 'anchor'
    elif proposal_hit <= 0:
        drop_stage = 'proposal'
    elif phase1_hit <= 0:
        drop_stage = 'phase1'
    elif final_chunk_hit <= 0:
        drop_stage = 'final_chunk'
    elif rendered_hit <= 0:
        drop_stage = 'rendered'
    elif answer_correct <= 0:
        drop_stage = 'generation'

    item = {
        'stage': stage,
        'variant': variant,
        'dataset': dataset,
        'query_id': str(row.get('sample_id', '') or row.get('sample_index', '')),
        'answer_correct': int(answer_correct),
        'anchor_hit': int(anchor_hit),
        'proposal_hit': int(proposal_hit),
        'phase1_hit': int(phase1_hit),
        'final_chunk_hit': int(final_chunk_hit),
        'rendered_hit': int(rendered_hit),
        'fallback': int(fallback),
        'oracle_answer_support_exists': int(oracle_support_exists),
        'gold_evidence_drop_stage': str(drop_stage),
    }
    written.append(item)

out_jsonl.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in written) + ('\n' if written else ''), encoding='utf-8')

with out_csv.open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in written:
        w.writerow(r)

if not written:
    out_summary.write_text(json.dumps({'query_count': 0}, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0)

n = float(len(written))
ret = {
    'query_count': int(len(written)),
    'answer_correct_rate': sum(r['answer_correct'] for r in written) / n,
    'anchor_hit_rate': sum(r['anchor_hit'] for r in written) / n,
    'proposal_hit_rate': sum(r['proposal_hit'] for r in written) / n,
    'phase1_hit_rate': sum(r['phase1_hit'] for r in written) / n,
    'final_chunk_hit_rate': sum(r['final_chunk_hit'] for r in written) / n,
    'rendered_hit_rate': sum(r['rendered_hit'] for r in written) / n,
    'fallback_rate': sum(r['fallback'] for r in written) / n,
    'oracle_answer_support_exists_rate': sum(r['oracle_answer_support_exists'] for r in written) / n,
    'drop_stage_counts': {},
}
for r in written:
    k = r['gold_evidence_drop_stage']
    ret['drop_stage_counts'][k] = int(ret['drop_stage_counts'].get(k, 0) + 1)

out_summary.write_text(json.dumps(ret, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

run_prompt_sanity_from_reference() {
  local dataset="$1"
  local out_json="$2"
  local out_md="$3"
  local max_model_len="$4"
  local sample_limit="$5"

  if [[ "${SKIP_COMPLETED}" == "true" && -f "${out_json}" ]]; then
    if "${PYTHON_BIN}" - "${out_json}" "${sample_limit}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
limit=max(1,int(float(sys.argv[2])))
obj=json.loads(p.read_text(encoding='utf-8'))
ok=bool(obj.get('pass', False))
sample_count=int(obj.get('sample_count', 0) or 0)
if ok and sample_count >= limit:
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
      log_msg "[SKIP] prompt_sanity dataset=${dataset}"
      return 0
    fi
  fi

  "${PYTHON_BIN}" - "${dataset}" "${REF_BUNDLE_JSON}" "${out_json}" "${out_md}" "${max_model_len}" "${sample_limit}" <<'PY'
import json
import sys
from pathlib import Path

dataset, ref_path, out_json, out_md, max_model_len, sample_limit = sys.argv[1:7]
max_model_len = int(max_model_len)
sample_limit = int(sample_limit)

refs = json.loads(Path(ref_path).read_text(encoding='utf-8'))
summary_path = str(((refs.get('prompt_sanity_reference', {}) or {}).get(dataset, '') or ''))
if not summary_path or not Path(summary_path).exists():
    payload = {
        'dataset': dataset,
        'sample_count': 0,
        'prompt_over_limit_rate': 1.0,
        'pass': False,
        'reason': 'reference_summary_missing',
    }
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(out_md).write_text(f"# Prompt Sanity ({dataset})\\n\\n- pass: false\\n- reason: reference_summary_missing\\n", encoding='utf-8')
    raise SystemExit(5)

summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
qpath = Path(summary_path).with_name('rag_query_results.jsonl')
if not qpath.exists():
    payload = {
        'dataset': dataset,
        'sample_count': 0,
        'prompt_over_limit_rate': 1.0,
        'pass': False,
        'reason': 'reference_query_results_missing',
    }
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(out_md).write_text(f"# Prompt Sanity ({dataset})\\n\\n- pass: false\\n- reason: reference_query_results_missing\\n", encoding='utf-8')
    raise SystemExit(5)

max_new = int(float((summary.get('generation_params', {}) or {}).get('llm_max_new_tokens', 64) or 64))
prompt_tokens = []
over = []
for line in qpath.open('r', encoding='utf-8'):
    if len(prompt_tokens) >= sample_limit:
        break
    line = str(line or '').strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except Exception:
        continue
    gmeta = (((row.get('generation', {}) or {}).get('metadata', {}) or {}))
    gd = (row.get('generation_diagnostics', {}) or {})
    ptok = gmeta.get('prompt_tokens', gd.get('prompt_tokens', 0))
    try:
        ptok = int(ptok)
    except Exception:
        ptok = 0
    if ptok <= 0:
        pred = str(row.get('prediction', '') or '')
        ptok = max(1, int(len(pred) / 4))
    prompt_tokens.append(ptok)
    total = int(ptok) + int(max_new)
    if total > max_model_len:
        over.append({
            'sample_id': str(row.get('sample_id', '')),
            'prompt_tokens': int(ptok),
            'max_new_tokens': int(max_new),
            'max_model_len': int(max_model_len),
            'over_by': int(total - max_model_len),
        })

n = len(prompt_tokens)
over_rate = (len(over) / float(n)) if n > 0 else 1.0
p95 = 0
p99 = 0
mx = 0
if n > 0:
    vals = sorted(prompt_tokens)
    p95 = vals[min(n - 1, int(0.95 * (n - 1)))]
    p99 = vals[min(n - 1, int(0.99 * (n - 1)))]
    mx = vals[-1]

payload = {
    'dataset': dataset,
    'reference_summary_path': str(Path(summary_path).resolve()),
    'reference_query_results_path': str(qpath.resolve()),
    'sample_count': int(n),
    'max_new_tokens': int(max_new),
    'max_model_len': int(max_model_len),
    'prompt_token_p95': int(p95),
    'prompt_token_p99': int(p99),
    'prompt_token_max': int(mx),
    'prompt_over_limit_rate': float(over_rate),
    'over_limit_examples': over[:20],
    'pass': bool(n > 0 and over_rate <= 1.0e-12),
}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
md = [
    f"# Prompt Sanity ({dataset})",
    '',
    f"- sample_count: {n}",
    f"- prompt_over_limit_rate: {over_rate:.6f}",
    f"- p99 + max_new_tokens: {p99} + {max_new} = {p99 + max_new}",
    f"- max_model_len: {max_model_len}",
    f"- pass: {payload['pass']}",
]
Path(out_md).write_text('\n'.join(md) + '\n', encoding='utf-8')
print('PASS' if payload['pass'] else 'FAIL')
raise SystemExit(0 if payload['pass'] else 5)
PY
}

run_rag_variant() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local profile_key="$4"
  local retrieval_mode="$5"
  local knob_json="$6"
  local limit="$7"

  local cfg_dir="${CFG_ROOT}/${stage}/${variant}/${dataset}"
  local rag_cfg="${cfg_dir}/rag.yaml"
  local out_dir="${RUN_ROOT}/${stage}/${variant}/rag"
  local log_file="${LOG_ROOT}/${stage}__${variant}__${dataset}.log"
  mkdir -p "${cfg_dir}" "${out_dir}" "$(dirname "${log_file}")"

  create_variant_cfg "${RAG_BASE_CFG}" "${rag_cfg}" "${dataset}" "${variant}" "${retrieval_mode}" "${knob_json}"

  if [[ "${SKIP_COMPLETED}" == "true" ]]; then
    local existing_summary
    existing_summary="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
    if [[ -n "${existing_summary}" ]]; then
      if is_complete_summary "${existing_summary}" "${limit}" "rag_query_results.jsonl" >/dev/null 2>&1; then
        annotate_summary_with_refs "${existing_summary}" "${dataset}"
        export_query_funnel_diagnostics "${existing_summary}" "${stage}" "${variant}" "${dataset}"
        log_msg "[SKIP] stage=${stage} variant=${variant} dataset=${dataset}"
        upsert_record "${stage}" "${variant}" "${dataset}" "${profile_key}" "rag" "${existing_summary}" "${log_file}" "skipped"
        return 0
      fi
    fi
  fi

  local -a cmd
  cmd=(
    "${PYTHON_BIN}" -m effirag.run_rag
    --config "${rag_cfg}"
    --dataset "${dataset}"
    --data-path "data/qa/${dataset}.json"
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
    --limit "${limit}"
    --output-dir "${out_dir}"
    --timestamp-output true
  )

  log_msg "[RUN] stage=${stage} variant=${variant} dataset=${dataset} mode=${retrieval_mode} limit=${limit}"
  {
    echo "[COMMAND] ${cmd[*]}"
    echo "[KNOBS] ${knob_json}"
  } >> "${log_file}"

  set +e
  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -ne 0 ]]; then
    log_msg "[FAIL] stage=${stage} variant=${variant} dataset=${dataset} exit_code=${rc}"
    upsert_record "${stage}" "${variant}" "${dataset}" "${profile_key}" "rag" "" "${log_file}" "failed"
    if [[ "${STOP_ON_FAILURE}" == "true" ]]; then
      exit ${rc}
    fi
    return ${rc}
  fi

  local summary_path
  summary_path="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
  if [[ -z "${summary_path}" ]]; then
    log_msg "[FAIL] stage=${stage} variant=${variant} dataset=${dataset} missing rag_summary.json"
    upsert_record "${stage}" "${variant}" "${dataset}" "${profile_key}" "rag" "" "${log_file}" "failed"
    if [[ "${STOP_ON_FAILURE}" == "true" ]]; then
      exit 1
    fi
    return 1
  fi

  annotate_summary_with_refs "${summary_path}" "${dataset}"
  export_query_funnel_diagnostics "${summary_path}" "${stage}" "${variant}" "${dataset}"
  upsert_record "${stage}" "${variant}" "${dataset}" "${profile_key}" "rag" "${summary_path}" "${log_file}" "ok"
  log_msg "[DONE] stage=${stage} variant=${variant} dataset=${dataset}"
}

build_stage2_audit_plan() {
  "${PYTHON_BIN}" - "${RECORD_TSV}" "${AUDIT_PLAN_JSON}" <<'PY'
import json
import sys
from pathlib import Path

record_tsv, out_json = sys.argv[1:3]


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)


def _load_summary(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


rows = []
if Path(record_tsv).exists():
    with Path(record_tsv).open('r', encoding='utf-8') as f:
        _ = f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) != 8:
                continue
            stage, variant, dataset, profile, kind, summary_path, log_path, status = p
            if kind != 'rag' or status not in {'ok', 'skipped'}:
                continue
            rows.append({
                'stage': stage,
                'variant': variant,
                'dataset': dataset,
                'summary_path': summary_path,
            })


def _stagewise(summary, key, default=0.0):
    if key in summary:
        return _f(summary.get(key, default), default)
    return _f(summary.get(f'stagewise_{key}', default), default)


def _pick(ds, variant):
    for r in rows:
        if r['dataset'] == ds and r['variant'] == variant:
            s = _load_summary(r['summary_path'])
            if s:
                return s
    return {}

result = {
    'run_candidate_audit': False,
    'run_anchor_audit': False,
    'run_corridor_audit': False,
    'reasons': [],
}

for ds in ('hotpotqa', '2wikimultihopqa'):
    s = _pick(ds, 'r2_plus_path_preserve_reconfirm')
    if not s:
        continue
    anchor = _stagewise(s, 'anchor_hit_rate')
    proposal = _stagewise(s, 'proposal_hit_rate')
    phase1 = _stagewise(s, 'phase1_hit_rate')
    final_chunk = _stagewise(s, 'final_chunk_hit_rate')
    rendered = _stagewise(s, 'rendered_hit_rate')
    rendered_ret = _stagewise(s, 'rendered_retention')

    if proposal < 0.72 or (proposal - phase1) > 0.12:
        result['run_candidate_audit'] = True
        result['reasons'].append(f'{ds}:candidate_ceiling_signal(proposal={proposal:.3f},phase1={phase1:.3f})')
    if anchor < 0.60 or (anchor - proposal) > 0.15:
        result['run_anchor_audit'] = True
        result['reasons'].append(f'{ds}:anchor_ceiling_signal(anchor={anchor:.3f},proposal={proposal:.3f})')
    if rendered_ret < 0.85 or (final_chunk - rendered) > 0.08:
        result['run_corridor_audit'] = True
        result['reasons'].append(
            f'{ds}:corridor_ceiling_signal(final={final_chunk:.3f},rendered={rendered:.3f},ret={rendered_ret:.3f})'
        )

Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(out_json)
PY
}

should_run_audit() {
  local kind="$1"
  local mode="${2:-auto}"
  if [[ "${mode}" == "true" ]]; then
    echo "1"
    return 0
  fi
  if [[ "${mode}" == "false" ]]; then
    echo "0"
    return 0
  fi
  "${PYTHON_BIN}" - "${AUDIT_PLAN_JSON}" "${kind}" <<'PY'
import json
import sys
from pathlib import Path
p=Path(sys.argv[1])
kind=str(sys.argv[2])
if not p.exists():
    print('0')
    raise SystemExit(0)
obj=json.loads(p.read_text(encoding='utf-8'))
key={
    'candidate': 'run_candidate_audit',
    'anchor': 'run_anchor_audit',
    'corridor': 'run_corridor_audit',
}.get(kind, '')
print('1' if bool(obj.get(key, False)) else '0')
PY
}

build_round_metrics() {
  "${PYTHON_BIN}" - "${RECORD_TSV}" "${REF_BUNDLE_JSON}" "${ROUND_METRICS_JSON}" "${ROUND_METRICS_MD}" "${QUERY_DIAG_JSONL}" "${QUERY_DIAG_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

record_tsv, ref_json, out_json, out_md, out_q_jsonl, out_q_csv = sys.argv[1:7]
refs = json.loads(Path(ref_json).read_text(encoding='utf-8')) if Path(ref_json).exists() else {}


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)


def _r_at(summary, k):
    d = _f(summary.get(f'supporting_fact_recall_at_{k}', 0.0))
    if d > 0.0:
        return d
    s = summary.get('supporting_fact_recall_at_k', {}) or {}
    if str(k) in s:
        return _f(s[str(k)])
    if k in s:
        return _f(s[k])
    s2 = summary.get('recall_at_k', {}) or {}
    if str(k) in s2:
        return _f(s2[str(k)])
    if k in s2:
        return _f(s2[k])
    return 0.0


def _g(summary, key, default=0.0):
    if key in summary:
        return _f(summary.get(key, default), default)
    return _f(summary.get(f'stagewise_{key}', default), default)


def _load_query_summary(summary_path):
    p = Path(summary_path).with_name('diagnosis_funnel_summary.json')
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _knob_diff_label(summary, base_summary):
    keys = [
        'phase1_run_preshortlist_topm',
        'phase1_run_shortlist_topk',
        'phase1_full_run_score_topk',
        'candidate_top_t',
        'semantic_topn_chunk',
        'graph_reserve_topn',
        'max_anchors',
        'samples_per_anchor',
        'corridor_top_bc',
    ]
    rp = dict(summary.get('retrieval_params', {}) or {})
    rb = dict(base_summary.get('retrieval_params', {}) or {})
    diffs = []
    for k in keys:
        v = rp.get(k, None)
        b = rb.get(k, None)
        if v is None or b is None:
            continue
        if str(v) != str(b):
            diffs.append(f'{k}:{b}->{v}')
    if not diffs:
        return 'baseline', ''
    if len(diffs) == 1:
        d = diffs[0]
        knob, vv = d.split(':', 1)
        return knob, vv
    return 'multi', ', '.join(diffs)


rows = []
all_query_rows = []
with Path(record_tsv).open('r', encoding='utf-8') as f:
    _ = f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) != 8:
            continue
        stage, variant, dataset, profile, kind, summary_path, log_path, status = p
        if kind != 'rag' or status not in {'ok', 'skipped'}:
            continue
        sp = Path(summary_path)
        if not sp.exists():
            continue
        try:
            summary = json.loads(sp.read_text(encoding='utf-8'))
        except Exception:
            continue

        qsum = _load_query_summary(summary_path)
        for qline in sp.with_name('diagnosis_query_funnel.jsonl').read_text(encoding='utf-8').splitlines() if sp.with_name('diagnosis_query_funnel.jsonl').exists() else []:
            try:
                all_query_rows.append(json.loads(qline))
            except Exception:
                pass

        r5 = _r_at(summary, 5)
        em = _f(summary.get('em', summary.get('EM', 0.0)))
        f1 = _f(summary.get('f1', summary.get('F1', 0.0)))
        retrieval_ms = _f(summary.get('retrieval_latency_ms', summary.get('retrieval_ms', 0.0)))
        total_ms = _f(summary.get('total_latency_ms', summary.get('total_ms', 0.0)))
        fallback_rate = _f(summary.get('fallback_rate', 0.0))

        hippo = ((refs.get(dataset, {}) or {}).get('hipporag2', {}) or {})
        d2 = ((refs.get(dataset, {}) or {}).get('d2', {}) or {})
        gap_to_hippo_r5 = (_f(hippo.get('r5', 0.0)) - r5) if _f(hippo.get('r5', 0.0)) > 0.0 else 0.0
        oracle_gap = (_f(d2.get('f1', 0.0)) - f1) if _f(d2.get('f1', 0.0)) > 0.0 else 0.0

        rows.append({
            'stage': stage,
            'variant': variant,
            'dataset': dataset,
            'summary_path': str(sp),
            'R@5': r5,
            'EM': em,
            'F1': f1,
            'retrieval_ms': retrieval_ms,
            'total_ms': total_ms,
            'fallback_rate': fallback_rate,
            'gap_to_hippo_r5': gap_to_hippo_r5,
            'oracle_gap': oracle_gap,
            'anchor_hit_rate': _g(summary, 'anchor_hit_rate', _f(qsum.get('anchor_hit_rate', 0.0))),
            'proposal_hit_rate': _g(summary, 'proposal_hit_rate', _f(qsum.get('proposal_hit_rate', 0.0))),
            'phase1_hit_rate': _g(summary, 'phase1_hit_rate', _f(qsum.get('phase1_hit_rate', 0.0))),
            'final_chunk_hit_rate': _g(summary, 'final_chunk_hit_rate', _f(qsum.get('final_chunk_hit_rate', 0.0))),
            'rendered_hit_rate': _g(summary, 'rendered_hit_rate', _f(qsum.get('rendered_hit_rate', 0.0))),
            'rendered_retention': _g(summary, 'rendered_retention', 0.0),
            'answer_correct': _g(summary, 'answer_correct', _f(qsum.get('answer_correct_rate', 0.0))),
            'legacy_path_complete_rate': _g(summary, 'path_complete_rate', 0.0),
            'strict_path_complete_rate': _g(summary, 'strict_path_complete_rate', 0.0),
            'answer_bearing_path_hit': _g(summary, 'answer_bearing_path_hit', 0.0),
            'false_path_rate': _g(summary, 'false_path_rate', 0.0),
            'equivalent_evidence_coverage': _g(summary, 'equivalent_evidence_coverage', 0.0),
            'path_complete_rate': _g(summary, 'path_complete_rate', 0.0),
            'bridge_answer_pair_retention': _g(summary, 'bridge_answer_pair_retention', 0.0),
            'incomplete_path_rate': _g(summary, 'incomplete_path_rate', 0.0),
            'retrieval_params': dict(summary.get('retrieval_params', {}) or {}),
            'drop_stage_counts': dict(qsum.get('drop_stage_counts', {}) or {}),
        })

# Add references as stage=reference rows
for ds in ('hotpotqa', '2wikimultihopqa'):
    for key, name in (
        ('d1', 'frozen_D1'),
        ('best_non_oracle', 'frozen_best_non_oracle'),
        ('d2', 'D2_upper_bound'),
        ('hipporag2', 'HippoRAG2'),
    ):
        ref = ((refs.get(ds, {}) or {}).get(key, {}) or {})
        if not ref:
            continue
        rows.append({
            'stage': 'reference',
            'variant': f"{name}({ref.get('label','')})",
            'dataset': ds,
            'summary_path': str(ref.get('path', '')),
            'R@5': _f(ref.get('r5', 0.0)),
            'EM': _f(ref.get('em', 0.0)),
            'F1': _f(ref.get('f1', 0.0)),
            'retrieval_ms': _f(ref.get('retrieval_ms', 0.0)),
            'total_ms': _f(ref.get('total_ms', 0.0)),
            'fallback_rate': 0.0,
            'gap_to_hippo_r5': 0.0,
            'oracle_gap': 0.0,
            'anchor_hit_rate': 0.0,
            'proposal_hit_rate': 0.0,
            'phase1_hit_rate': 0.0,
            'final_chunk_hit_rate': 0.0,
            'rendered_hit_rate': 0.0,
            'rendered_retention': 0.0,
            'answer_correct': 0.0,
            'legacy_path_complete_rate': 0.0,
            'strict_path_complete_rate': 0.0,
            'answer_bearing_path_hit': 0.0,
            'false_path_rate': 0.0,
            'equivalent_evidence_coverage': 0.0,
            'path_complete_rate': 0.0,
            'bridge_answer_pair_retention': 0.0,
            'incomplete_path_rate': 0.0,
            'retrieval_params': {},
            'drop_stage_counts': {},
        })

# Baselines for knob diff and delta
baseline_by_ds = {}
r2_base_by_ds = {}
for r in rows:
    if r['stage'] == 'reference':
        continue
    if r['variant'] == 'baseline_mid_reconfirm' and r['dataset'] not in baseline_by_ds:
        baseline_by_ds[r['dataset']] = r
    if r['variant'] == 'r2_plus_path_preserve_reconfirm' and r['dataset'] not in r2_base_by_ds:
        r2_base_by_ds[r['dataset']] = r

for r in rows:
    if r['stage'] == 'reference':
        continue
    b = baseline_by_ds.get(r['dataset'])
    if b is None:
        r['delta_R5_vs_baseline'] = 0.0
        r['delta_EM_vs_baseline'] = 0.0
        r['delta_F1_vs_baseline'] = 0.0
        r['delta_retrieval_ms_vs_baseline'] = 0.0
        r['delta_total_ms_vs_baseline'] = 0.0
    else:
        r['delta_R5_vs_baseline'] = _f(r['R@5']) - _f(b['R@5'])
        r['delta_EM_vs_baseline'] = _f(r['EM']) - _f(b['EM'])
        r['delta_F1_vs_baseline'] = _f(r['F1']) - _f(b['F1'])
        r['delta_retrieval_ms_vs_baseline'] = _f(r['retrieval_ms']) - _f(b['retrieval_ms'])
        r['delta_total_ms_vs_baseline'] = _f(r['total_ms']) - _f(b['total_ms'])

for r in rows:
    if r['stage'] == 'reference':
        r['changed_knob'] = ''
        r['knob_value'] = ''
        continue
    rb = r2_base_by_ds.get(r['dataset'])
    if rb is None:
        r['changed_knob'] = 'unknown'
        r['knob_value'] = ''
        continue
    k, v = _knob_diff_label({'retrieval_params': r.get('retrieval_params', {})}, {'retrieval_params': rb.get('retrieval_params', {})})
    r['changed_knob'] = k
    r['knob_value'] = v

# Write aggregated query diagnostics
Path(out_q_jsonl).write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in all_query_rows) + ('\n' if all_query_rows else ''), encoding='utf-8')
if all_query_rows:
    fields = [
        'stage','variant','dataset','query_id','answer_correct','anchor_hit','proposal_hit','phase1_hit','final_chunk_hit','rendered_hit','fallback','oracle_answer_support_exists','gold_evidence_drop_stage'
    ]
    with Path(out_q_csv).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for q in all_query_rows:
            w.writerow({k: q.get(k, '') for k in fields})
else:
    with Path(out_q_csv).open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stage','variant','dataset','query_id','answer_correct','anchor_hit','proposal_hit','phase1_hit','final_chunk_hit','rendered_hit','fallback','oracle_answer_support_exists','gold_evidence_drop_stage'])

payload = {
    'records': rows,
    'query_diagnostics_jsonl': str(Path(out_q_jsonl).resolve()),
    'query_diagnostics_csv': str(Path(out_q_csv).resolve()),
}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

lines = [
    '# Recall Ceiling Diagnosis Metrics (RAG full)',
    '',
    '## Main Diagnosis Table',
    '',
    '| stage | variant | dataset | R@5 | EM | F1 | retrieval_ms | total_ms | fallback_rate | gap_to_hippo_r5 | oracle_gap |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    lines.append(
        f"| {r['stage']} | {r['variant']} | {r['dataset']} | {r['R@5']:.4f} | {r['EM']:.4f} | {r['F1']:.4f} | {r['retrieval_ms']:.2f} | {r['total_ms']:.2f} | {r['fallback_rate']:.4f} | {r['gap_to_hippo_r5']:+.4f} | {r['oracle_gap']:+.4f} |"
    )

lines += [
    '',
    '## Stage-wise Funnel Table',
    '',
    '| stage | variant | dataset | anchor_hit_rate | proposal_hit_rate | phase1_hit_rate | final_chunk_hit_rate | rendered_hit_rate | rendered_retention | answer_correct | fallback_rate |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    if r['stage'] == 'reference':
        continue
    lines.append(
        f"| {r['stage']} | {r['variant']} | {r['dataset']} | {r['anchor_hit_rate']:.4f} | {r['proposal_hit_rate']:.4f} | {r['phase1_hit_rate']:.4f} | {r['final_chunk_hit_rate']:.4f} | {r['rendered_hit_rate']:.4f} | {r['rendered_retention']:.4f} | {r['answer_correct']:.4f} | {r['fallback_rate']:.4f} |"
    )

lines += [
    '',
    '## Budget Sweep Table',
    '',
    '| stage | variant | dataset | changed_knob | knob_value | proposal_hit_rate | phase1_hit_rate | final_chunk_hit_rate | rendered_hit_rate | R@5 | EM | F1 | retrieval_ms | total_ms |',
    '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    if r['stage'] == 'reference':
        continue
    if not str(r['stage']).startswith('s2') and str(r['variant']) != 'r2_plus_path_preserve_reconfirm':
        continue
    lines.append(
        f"| {r['stage']} | {r['variant']} | {r['dataset']} | {r['changed_knob']} | {r['knob_value']} | {r['proposal_hit_rate']:.4f} | {r['phase1_hit_rate']:.4f} | {r['final_chunk_hit_rate']:.4f} | {r['rendered_hit_rate']:.4f} | {r['R@5']:.4f} | {r['EM']:.4f} | {r['F1']:.4f} | {r['retrieval_ms']:.2f} | {r['total_ms']:.2f} |"
    )

lines += [
    '',
    '## Metric Validity Table',
    '',
    '| stage | variant | dataset | legacy_path_complete_rate | strict_path_complete_rate | answer_bearing_path_hit | false_path_rate | equivalent_evidence_coverage |',
    '|---|---|---|---:|---:|---:|---:|---:|',
]
for r in rows:
    if r['stage'] == 'reference':
        continue
    lines.append(
        f"| {r['stage']} | {r['variant']} | {r['dataset']} | {r['legacy_path_complete_rate']:.4f} | {r['strict_path_complete_rate']:.4f} | {r['answer_bearing_path_hit']:.4f} | {r['false_path_rate']:.4f} | {r['equivalent_evidence_coverage']:.4f} |"
    )

Path(out_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
}

run_stage1() {
  log_msg "[Stage 1] baseline integrity + funnel audit"
  run_rag_variant "s1" "baseline_mid_reconfirm" "hotpotqa" "canonical_mid" "baseline" "{}" "${LIMIT_RAG}"
  run_rag_variant "s1" "baseline_mid_reconfirm" "2wikimultihopqa" "canonical_mid" "baseline" "{}" "${LIMIT_RAG}"

  run_rag_variant "s1" "r2_plus_path_preserve_reconfirm" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" "{}" "${LIMIT_RAG}"
  run_rag_variant "s1" "r2_plus_path_preserve_reconfirm" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" "{}" "${LIMIT_RAG}"
}

run_stage2_run_shortlist() {
  log_msg "[Stage 2A] run shortlist ceiling audit (mandatory)"
  run_rag_variant "s2a" "s2a_preshortlist_topm4" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"phase1_run_preshortlist_topm":4}' "${LIMIT_RAG}"
  run_rag_variant "s2a" "s2a_preshortlist_topm4" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"phase1_run_preshortlist_topm":4}' "${LIMIT_RAG}"

  run_rag_variant "s2a" "s2a_shortlist_topk4" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"phase1_run_shortlist_topk":4}' "${LIMIT_RAG}"
  run_rag_variant "s2a" "s2a_shortlist_topk4" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"phase1_run_shortlist_topk":4}' "${LIMIT_RAG}"

  run_rag_variant "s2a" "s2a_full_run_score_topk4" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"phase1_full_run_score_topk":4}' "${LIMIT_RAG}"
  run_rag_variant "s2a" "s2a_full_run_score_topk4" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"phase1_full_run_score_topk":4}' "${LIMIT_RAG}"

  run_rag_variant "s2a" "s2a_balanced4" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"phase1_run_preshortlist_topm":4,"phase1_run_shortlist_topk":4,"phase1_full_run_score_topk":4}' "${LIMIT_RAG}"
  run_rag_variant "s2a" "s2a_balanced4" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"phase1_run_preshortlist_topm":4,"phase1_run_shortlist_topk":4,"phase1_full_run_score_topk":4}' "${LIMIT_RAG}"

  if [[ "${RUN_SWEEP_HIGH}" == "true" ]]; then
    run_rag_variant "s2a" "s2a_balanced6" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"phase1_run_preshortlist_topm":6,"phase1_run_shortlist_topk":6,"phase1_full_run_score_topk":6}' "${LIMIT_RAG}"
    run_rag_variant "s2a" "s2a_balanced6" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"phase1_run_preshortlist_topm":6,"phase1_run_shortlist_topk":6,"phase1_full_run_score_topk":6}' "${LIMIT_RAG}"
  fi
}

run_stage2_candidate_audit() {
  log_msg "[Stage 2B] candidate universe audit (conditional)"
  run_rag_variant "s2b" "s2b_candidate_top_t30" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"candidate_top_t":30}' "${LIMIT_RAG}"
  run_rag_variant "s2b" "s2b_candidate_top_t30" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"candidate_top_t":30}' "${LIMIT_RAG}"

  run_rag_variant "s2b" "s2b_semantic_topn_chunk25" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"semantic_topn_chunk":25}' "${LIMIT_RAG}"
  run_rag_variant "s2b" "s2b_semantic_topn_chunk25" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"semantic_topn_chunk":25}' "${LIMIT_RAG}"

  run_rag_variant "s2b" "s2b_graph_reserve_topn25" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"graph_reserve_topn":25}' "${LIMIT_RAG}"
  run_rag_variant "s2b" "s2b_graph_reserve_topn25" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"graph_reserve_topn":25}' "${LIMIT_RAG}"

  run_rag_variant "s2b" "s2b_balanced_mid" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"candidate_top_t":30,"semantic_topn_chunk":25,"graph_reserve_topn":25}' "${LIMIT_RAG}"
  run_rag_variant "s2b" "s2b_balanced_mid" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"candidate_top_t":30,"semantic_topn_chunk":25,"graph_reserve_topn":25}' "${LIMIT_RAG}"
}

run_stage2_anchor_audit() {
  log_msg "[Stage 2C] anchor ceiling audit (conditional)"
  run_rag_variant "s2c" "s2c_max_anchors4" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"max_anchors":4}' "${LIMIT_RAG}"
  run_rag_variant "s2c" "s2c_max_anchors4" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"max_anchors":4}' "${LIMIT_RAG}"

  run_rag_variant "s2c" "s2c_max_anchors5" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"max_anchors":5}' "${LIMIT_RAG}"
  run_rag_variant "s2c" "s2c_max_anchors5" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"max_anchors":5}' "${LIMIT_RAG}"
}

run_stage2_corridor_audit() {
  log_msg "[Stage 2D] corridor ceiling audit (conditional)"
  run_rag_variant "s2d" "s2d_corridor_top_bc30" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"corridor_top_bc":30}' "${LIMIT_RAG}"
  run_rag_variant "s2d" "s2d_corridor_top_bc30" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"corridor_top_bc":30}' "${LIMIT_RAG}"

  run_rag_variant "s2d" "s2d_corridor_top_bc40" "hotpotqa" "canonical_mid" "r2_plus_path_preserve" '{"corridor_top_bc":40}' "${LIMIT_RAG}"
  run_rag_variant "s2d" "s2d_corridor_top_bc40" "2wikimultihopqa" "canonical_mid" "r2_plus_path_preserve" '{"corridor_top_bc":40}' "${LIMIT_RAG}"
}

run_sanity_checks_after_stage1() {
  "${PYTHON_BIN}" - "${RECORD_TSV}" <<'PY'
import json
import sys
from pathlib import Path

record_tsv = Path(sys.argv[1])
rows = []
with record_tsv.open('r', encoding='utf-8') as f:
    _ = f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) != 8:
            continue
        stage, variant, dataset, profile, kind, summary_path, log_path, status = p
        if stage != 's1' or kind != 'rag' or status not in {'ok','skipped'}:
            continue
        rows.append((variant, dataset, summary_path))

if not rows:
    raise SystemExit('stage1 runs missing')

for variant, dataset, spath in rows:
    p = Path(spath)
    if not p.exists():
        raise SystemExit(f'missing summary: {variant} {dataset}')
    summary = json.loads(p.read_text(encoding='utf-8'))
    fallback = float(summary.get('fallback_rate', 0.0) or 0.0)
    anchor = float(summary.get('stagewise_anchor_hit_rate', summary.get('anchor_hit_rate', 0.0)) or 0.0)
    proposal = float(summary.get('stagewise_proposal_hit_rate', summary.get('proposal_hit_rate', 0.0)) or 0.0)
    phase1 = float(summary.get('stagewise_phase1_hit_rate', summary.get('phase1_hit_rate', 0.0)) or 0.0)
    final_chunk = float(summary.get('stagewise_final_chunk_hit_rate', summary.get('final_chunk_hit_rate', 0.0)) or 0.0)
    rendered = float(summary.get('stagewise_rendered_hit_rate', summary.get('rendered_hit_rate', 0.0)) or 0.0)
    if min(anchor, proposal, phase1, final_chunk) < 0.0:
        raise SystemExit(f'invalid stagewise values: {variant} {dataset}')
    print(f'[SANITY] {variant} {dataset} fallback_rate={fallback:.4f} anchor={anchor:.4f} proposal={proposal:.4f} phase1={phase1:.4f} final={final_chunk:.4f} rendered={rendered:.4f}')
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
    if int(cfg.get('num_workers', 1) or 1) > 1 and bool(cfg.get('force_rebuild_graph_index', False)):
        unsafe.append(str(p))
if unsafe:
    raise SystemExit('unsafe cache/meta race configs:\n' + '\n'.join(unsafe))
print('safe')
PY
}

# -----------------------------
# Execution starts here
# -----------------------------

log_msg "[INFO] round_root=${ROUND_ROOT} (source=${ROUND_ROOT_SOURCE})"

if [[ "${RUN_PREFLIGHT_CHECKS}" == "true" ]]; then
  log_msg "[CHECK] python -m py_compile effirag/*.py"
  "${PYTHON_BIN}" -m py_compile effirag/*.py

  log_msg "[CHECK] PYTHONPATH=. python3 -m effirag.config_audit configs/canonical"
  PYTHONPATH=. "${PYTHON_BIN}" -m effirag.config_audit configs/canonical
fi

build_reference_bundle
resolve_vllm_max_model_len
log_msg "[INFO] vLLM max_model_len=${VLLM_MAX_MODEL_LEN_RESOLVED} source=${VLLM_MAX_MODEL_LEN_SOURCE}"

log_msg "[Preflight] hotpot prompt sanity"
run_prompt_sanity_from_reference "hotpotqa" "${PREFLIGHT_HOTPOT_JSON}" "${PREFLIGHT_HOTPOT_MD}" "${VLLM_MAX_MODEL_LEN_RESOLVED}" "${PROMPT_SANITY_LIMIT}"

log_msg "[Preflight] 2wiki prompt sanity"
run_prompt_sanity_from_reference "2wikimultihopqa" "${PREFLIGHT_WIKI_JSON}" "${PREFLIGHT_WIKI_MD}" "${VLLM_MAX_MODEL_LEN_RESOLVED}" "${PROMPT_SANITY_LIMIT}"

run_stage1
run_sanity_checks_after_stage1

run_stage2_run_shortlist

build_stage2_audit_plan
log_msg "[INFO] stage2 audit plan: ${AUDIT_PLAN_JSON}"
if [[ -f "${AUDIT_PLAN_JSON}" ]]; then
  cat "${AUDIT_PLAN_JSON}"
fi

if [[ "$(should_run_audit candidate "${RUN_STAGE2_CANDIDATE_AUDIT}")" == "1" ]]; then
  run_stage2_candidate_audit
else
  log_msg "[SKIP] Stage 2B candidate audit"
fi

if [[ "$(should_run_audit anchor "${RUN_STAGE2_ANCHOR_AUDIT}")" == "1" ]]; then
  run_stage2_anchor_audit
else
  log_msg "[SKIP] Stage 2C anchor audit"
fi

if [[ "$(should_run_audit corridor "${RUN_STAGE2_CORRIDOR_AUDIT}")" == "1" ]]; then
  run_stage2_corridor_audit
else
  log_msg "[SKIP] Stage 2D corridor audit"
fi

check_cache_meta_race

build_round_metrics

log_msg "[DONE] outputs: ${ROUND_ROOT}"
log_msg "[DONE] logs: ${LOG_ROOT}"
log_msg "[DONE] run records: ${RECORD_TSV}"
log_msg "[DONE] query diagnostics: ${QUERY_DIAG_JSONL}"
log_msg "[DONE] round metrics: ${ROUND_METRICS_JSON}"
