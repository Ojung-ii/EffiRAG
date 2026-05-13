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
RUN_OPTIONAL_TOP_FOCUS="${RUN_OPTIONAL_TOP_FOCUS:-false}"

UNIVERSAL_EMBED_MAX_LEN="${UNIVERSAL_EMBED_MAX_LEN:-256}"
UNIVERSAL_EMBED_MAX_CHARS="${UNIVERSAL_EMBED_MAX_CHARS:-800}"

GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
RAG_BASE_CFG="configs/canonical/rag_entity_first_chunk_grounded.yaml"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-0}"

# Frozen references for mandatory comparison rows.
HOTPOT_F2_REF_SUMMARY="${HOTPOT_F2_REF_SUMMARY:-outputs/exp_retrieval_gap/hotpotqa/F2/rag/hotpotqa/20260405_141352_352678/rag_summary.json}"
WIKI_BEST_REF_SUMMARY="${WIKI_BEST_REF_SUMMARY:-outputs/exp_retrieval_gap/2wikimultihopqa/W3/rag/2wikimultihopqa/20260409_091427_631634/rag_summary.json}"
WIKI_E1_REF_SUMMARY="${WIKI_E1_REF_SUMMARY:-outputs/exp_retrieval_gap/2wikimultihopqa/E1/rag/2wikimultihopqa/20260405_144256_613687/rag_summary.json}"
D2_HOTPOT_REF_SUMMARY="${D2_HOTPOT_REF_SUMMARY:-outputs/exp_retrieval_gap_recheck/hotpotqa/D2/rag/hotpotqa/20260406_053942_373258/rag_summary.json}"
D2_WIKI_REF_SUMMARY="${D2_WIKI_REF_SUMMARY:-outputs/exp_retrieval_gap/2wikimultihopqa/D2/rag/2wikimultihopqa/20260405_143345_219316/rag_summary.json}"

# Dataset-specific strong references from recall ceiling diagnosis round.
HOTPOT_STRONG_REF_SUMMARY="${HOTPOT_STRONG_REF_SUMMARY:-outputs/recall_ceiling_diagnosis/20260417_135741/runs/s2b/s2b_balanced_mid/rag/hotpotqa/20260417_100054_632161/rag_summary.json}"
WIKI_STRONG_REF_SUMMARY="${WIKI_STRONG_REF_SUMMARY:-outputs/recall_ceiling_diagnosis/20260417_135741/runs/s2b/s2b_semantic_topn_chunk25/rag/2wikimultihopqa/20260417_093458_173659/rag_summary.json}"

if [[ -n "${RESUME_ROUND_ROOT:-}" ]]; then
  ROUND_ROOT="${RESUME_ROUND_ROOT}"
  ROUND_ROOT_SOURCE="resume_env"
elif [[ "${FORCE_NEW_ROUND}" == "true" ]]; then
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="outputs/conversion_bottleneck_followup/${RUN_STAMP}"
  ROUND_ROOT_SOURCE="force_new"
elif [[ "${AUTO_RESUME_LATEST}" == "true" ]]; then
  LATEST_ROUND_ROOT="$(find outputs/conversion_bottleneck_followup -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${LATEST_ROUND_ROOT}" ]]; then
    ROUND_ROOT="${LATEST_ROUND_ROOT}"
    ROUND_ROOT_SOURCE="auto_resume_latest"
  else
    RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
    ROUND_ROOT="outputs/conversion_bottleneck_followup/${RUN_STAMP}"
    ROUND_ROOT_SOURCE="new_round"
  fi
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="outputs/conversion_bottleneck_followup/${RUN_STAMP}"
  ROUND_ROOT_SOURCE="new_round"
fi

LOG_ROOT="logs/conversion_bottleneck_followup/$(basename "${ROUND_ROOT}")"
CFG_ROOT="${ROUND_ROOT}/configs"
RUN_ROOT="${ROUND_ROOT}/runs"
mkdir -p "${ROUND_ROOT}" "${LOG_ROOT}" "${CFG_ROOT}" "${RUN_ROOT}"

RECORD_TSV="${ROUND_ROOT}/run_records.tsv"
if [[ ! -f "${RECORD_TSV}" ]]; then
  printf "stage\tvariant\tdataset\tprofile\tkind\tsummary_path\tlog_path\tstatus\n" > "${RECORD_TSV}"
fi

REF_BUNDLE_JSON="${ROUND_ROOT}/reference_bundle.json"
PREFLIGHT_HOTPOT_JSON="${ROUND_ROOT}/preflight_hotpot_prompt_sanity.json"
PREFLIGHT_HOTPOT_MD="${ROUND_ROOT}/preflight_hotpot_prompt_sanity.md"
PREFLIGHT_WIKI_JSON="${ROUND_ROOT}/preflight_wiki_prompt_sanity.json"
PREFLIGHT_WIKI_MD="${ROUND_ROOT}/preflight_wiki_prompt_sanity.md"
ROUND_METRICS_JSON="${ROUND_ROOT}/round_metrics.json"
ROUND_METRICS_MD="${ROUND_ROOT}/round_metrics.md"
QUERY_DIAG_JSONL="${ROUND_ROOT}/conversion_query_diagnostics_all.jsonl"
QUERY_DIAG_CSV="${ROUND_ROOT}/conversion_query_diagnostics_all.csv"

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
    "${HOTPOT_F2_REF_SUMMARY}" \
    "${WIKI_BEST_REF_SUMMARY}" \
    "${WIKI_E1_REF_SUMMARY}" \
    "${D2_HOTPOT_REF_SUMMARY}" \
    "${D2_WIKI_REF_SUMMARY}" \
    "${HOTPOT_STRONG_REF_SUMMARY}" \
    "${WIKI_STRONG_REF_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

out_json, hotpot_f2, wiki_best, wiki_e1, d2_hotpot, d2_wiki, hotpot_ref, wiki_ref = sys.argv[1:9]


def _load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _r_at(summary, k):
    direct = _safe_float(summary.get(f"supporting_fact_recall_at_{k}", 0.0), 0.0)
    if direct > 0.0:
        return direct
    bag = summary.get("supporting_fact_recall_at_k", {}) or {}
    if str(k) in bag:
        return _safe_float(bag[str(k)], 0.0)
    if k in bag:
        return _safe_float(bag[k], 0.0)
    bag2 = summary.get("recall_at_k", {}) or {}
    if str(k) in bag2:
        return _safe_float(bag2[str(k)], 0.0)
    if k in bag2:
        return _safe_float(bag2[k], 0.0)
    return _safe_float(summary.get("Recall@%d" % int(k), 0.0), 0.0)


def _item(label, path):
    summary = _load_json(path)
    return {
        "label": str(label),
        "path": str(Path(path).resolve()),
        "r5": _r_at(summary, 5),
        "em": _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0),
        "f1": _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0),
        "retrieval_ms": _safe_float(summary.get("retrieval_latency_ms", summary.get("retrieval_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_latency_ms", summary.get("generation_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_latency_ms", summary.get("total_ms", 0.0)), 0.0),
    }


wiki_best_path = str(wiki_best or "").strip()
if not wiki_best_path or not Path(wiki_best_path).exists():
    if Path(str(wiki_e1 or "")).exists():
        wiki_best_path = str(wiki_e1)
    else:
        cands = sorted(Path("outputs").glob("**/2wikimultihopqa/W3/rag/2wikimultihopqa/*/rag_summary.json"))
        if cands:
            wiki_best_path = str(cands[-1])
if not wiki_best_path or not Path(wiki_best_path).exists():
    raise FileNotFoundError("wiki_best_reference_missing")

payload = {
    "hotpotqa": {
        "best_non_oracle": _item("F2", hotpot_f2),
        "d2": _item("D2", d2_hotpot),
        "hipporag2": {
            "label": "HippoRAG2",
            "path": "fixed_table://hipporag2/hotpotqa/2026-03-23",
            "r5": 0.9500,
            "em": 0.3350,
            "f1": 0.5154,
            "retrieval_ms": 207.7214 * 1000.0,
            "generation_ms": 0.3412 * 1000.0,
            "total_ms": 208.0626 * 1000.0,
        },
    },
    "2wikimultihopqa": {
        "best_non_oracle": _item("WIKI_BEST", wiki_best_path),
        "d2": _item("D2", d2_wiki),
        "hipporag2": {
            "label": "HippoRAG2",
            "path": "fixed_table://hipporag2/2wikimultihopqa/2026-03-24",
            "r5": 0.8955,
            "em": 0.2670,
            "f1": 0.4499,
            "retrieval_ms": 873.6446 * 1000.0,
            "generation_ms": 2003.5872 * 1000.0,
            "total_ms": 2877.2318 * 1000.0,
        },
    },
    "prompt_sanity_reference": {
        "hotpotqa": str(Path(hotpot_ref).resolve()) if Path(hotpot_ref).exists() else "",
        "2wikimultihopqa": str(Path(wiki_ref).resolve()) if Path(wiki_ref).exists() else "",
    },
    "strong_reference": {
        "hotpotqa": str(Path(hotpot_ref).resolve()) if Path(hotpot_ref).exists() else "",
        "2wikimultihopqa": str(Path(wiki_ref).resolve()) if Path(wiki_ref).exists() else "",
    },
}

Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
  local order_strategy="$7"

  mkdir -p "$(dirname "${out_cfg}")"
  "${PYTHON_BIN}" - \
    "${base_cfg}" \
    "${out_cfg}" \
    "${dataset}" \
    "${variant_tag}" \
    "${retrieval_mode}" \
    "${knob_json}" \
    "${order_strategy}" \
    "${UNIVERSAL_EMBED_MAX_LEN}" \
    "${UNIVERSAL_EMBED_MAX_CHARS}" \
    "${MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

base_cfg, out_cfg, dataset, variant_tag, retrieval_mode, knob_json, order_strategy, embed_len, embed_chars, model_name = sys.argv[1:11]
cfg = yaml.safe_load(Path(base_cfg).read_text(encoding='utf-8')) or {}
knobs = json.loads(knob_json or "{}")

cfg['dataset'] = str(dataset)
cfg['canonical_variant_name'] = str(variant_tag)

# Round invariants.
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
cfg['stagewise_loss_funnel_enabled'] = True

# Keep retrieval core fixed; only render/interface knobs may vary.
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

if str(order_strategy or "").strip():
    cfg['order_strategy'] = str(order_strategy).strip()

if not str(cfg.get('render_mode', '') or '').strip():
    cfg['render_mode'] = 'corridor_aware_flat'

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

export_query_conversion_diagnostics() {
  local summary_path="$1"
  local stage="$2"
  local variant="$3"
  local dataset="$4"

  "${PYTHON_BIN}" - "${summary_path}" "${stage}" "${variant}" "${dataset}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

from effirag.utils import content_tokens

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

out_jsonl = summary_path.with_name('conversion_query_diagnostics.jsonl')
out_csv = summary_path.with_name('conversion_query_diagnostics.csv')
out_summary = summary_path.with_name('conversion_diagnostics_summary.json')

fields = [
    'stage',
    'variant',
    'dataset',
    'query_id',
    'answer_correct',
    'fallback',
    'rendered_hit',
    'answer_bearing_path_hit',
    'answer_bearing_chunk_present',
    'answer_bearing_bundle_present',
    'first_answer_bearing_chunk_rank',
    'first_answer_bearing_bundle_rank',
    'answer_bearing_token_start_ratio',
    'answer_bearing_token_end_ratio',
    'num_answer_bearing_chunks',
    'num_answer_bearing_bundles',
    'evidence_dispersion_score',
    'evidence_duplication_score',
    'evidence_truncation_flag',
    'answer_present_but_generation_fail',
    'model_output_length',
    'output_overlaps_answer_bearing_evidence',
    'strict_path_complete',
]


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


def _tokens(text):
    return set(content_tokens(str(text or "")))


def _safe_ratio(n, d):
    n = _f(n, 0.0)
    d = _f(d, 0.0)
    if d <= 0.0:
        return 0.0
    return float(n / d)


def _jacc(a, b):
    if not a or not b:
        return 0.0
    u = a.union(b)
    if not u:
        return 0.0
    return float(len(a.intersection(b)) / len(u))


def _corridor_sentence_ids(c):
    out = []
    seen = set()
    for key in ('main_path_unit_ids', 'main_path_sentence_ids', 'support_unit_ids', 'support_sentence_ids', 'connector_adjacent_unit_ids', 'connector_adjacent_sentence_ids'):
        for sid in list((c or {}).get(key, []) or []):
            sid = str(sid or '')
            if not sid or sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
    return out


def _answer_bearing_score(c):
    comp = dict((c or {}).get('final_score_components', {}) or {})
    path_complete = _f(comp.get('path_complete_score', comp.get('path_complete', 0.0)), 0.0)
    pair_retention = _f(comp.get('bridge_answer_pair_retention', 0.0), 0.0)
    answer_density = _f(comp.get('answer_side_density', 0.0), 0.0)
    bridge_purity = _f(comp.get('bridge_purity', 0.0), 0.0)
    preserve = 1.0 if bool(comp.get('path_preserve_applied', False)) else 0.0
    score = (
        0.42 * max(0.0, min(1.0, path_complete))
        + 0.33 * max(0.0, min(1.0, pair_retention))
        + 0.20 * max(0.0, min(1.0, answer_density))
        + 0.05 * preserve
    )
    is_bearing = bool(
        score >= 0.58
        and pair_retention >= 0.40
        and answer_density >= 0.45
        and (path_complete >= 0.45 or bridge_purity >= 0.35)
    )
    return float(max(0.0, min(1.0, score))), bool(is_bearing)


def _build_answer_bearing_maps(retrieval):
    corridors = sorted(list((retrieval or {}).get('corridors', []) or []), key=lambda c: _f((c or {}).get('corridor_score', 0.0), 0.0), reverse=True)
    bearing_corridors = {}
    bearing_corridor_rank = {}
    bearing_sentences = {}
    for rank, corridor in enumerate(corridors, start=1):
        cid = str((corridor or {}).get('corridor_id', f'c{rank:02d}') or f'c{rank:02d}')
        score, ok = _answer_bearing_score(corridor)
        if not ok:
            continue
        bearing_corridors[cid] = float(score)
        bearing_corridor_rank[cid] = int(rank)
        for sid in _corridor_sentence_ids(corridor):
            prev = _f(bearing_sentences.get(sid, -1.0), -1.0)
            if score > prev:
                bearing_sentences[sid] = float(score)
    ordered = sorted(bearing_corridors.keys(), key=lambda cid: (_f(bearing_corridors.get(cid, 0.0), 0.0), -_i(bearing_corridor_rank.get(cid, 10**9), 10**9)), reverse=True)
    return bearing_corridors, bearing_corridor_rank, bearing_sentences, ordered


def _subset_stats(items):
    items = list(items or [])
    n = len(items)
    if n <= 0:
        return {
            'count': 0,
            'answer_bearing_chunk_present_rate': 0.0,
            'answer_bearing_bundle_present_rate': 0.0,
            'answer_present_but_generation_fail_rate': 0.0,
            'avg_first_answer_bearing_chunk_rank': 0.0,
            'avg_first_answer_bearing_bundle_rank': 0.0,
            'avg_answer_bearing_token_start_ratio': 0.0,
            'avg_evidence_dispersion_score': 0.0,
            'avg_evidence_duplication_score': 0.0,
            'output_overlap_answer_bearing_rate': 0.0,
            'fallback_rate': 0.0,
        }

    def _mean(key, pred=None):
        vals = []
        for it in items:
            if pred is not None and not pred(it):
                continue
            vals.append(_f(it.get(key, 0.0), 0.0))
        if not vals:
            return 0.0
        return float(sum(vals) / float(len(vals)))

    present_chunk = [it for it in items if _i(it.get('first_answer_bearing_chunk_rank', 0), 0) > 0]
    present_bundle = [it for it in items if _i(it.get('first_answer_bearing_bundle_rank', 0), 0) > 0]
    return {
        'count': int(n),
        'answer_bearing_chunk_present_rate': _mean('answer_bearing_chunk_present'),
        'answer_bearing_bundle_present_rate': _mean('answer_bearing_bundle_present'),
        'answer_present_but_generation_fail_rate': _mean('answer_present_but_generation_fail'),
        'avg_first_answer_bearing_chunk_rank': (
            float(sum(_f(it.get('first_answer_bearing_chunk_rank', 0), 0.0) for it in present_chunk) / float(len(present_chunk)))
            if present_chunk
            else 0.0
        ),
        'avg_first_answer_bearing_bundle_rank': (
            float(sum(_f(it.get('first_answer_bearing_bundle_rank', 0), 0.0) for it in present_bundle) / float(len(present_bundle)))
            if present_bundle
            else 0.0
        ),
        'avg_answer_bearing_token_start_ratio': _mean('answer_bearing_token_start_ratio', pred=lambda it: _i(it.get('num_answer_bearing_chunks', 0), 0) > 0),
        'avg_evidence_dispersion_score': _mean('evidence_dispersion_score'),
        'avg_evidence_duplication_score': _mean('evidence_duplication_score'),
        'output_overlap_answer_bearing_rate': _mean('output_overlaps_answer_bearing_evidence'),
        'fallback_rate': _mean('fallback'),
    }


written = []
for row in rows:
    retrieval = (row.get('retrieval', {}) or {})
    diag = (retrieval.get('diagnostics', {}) or {})
    metrics = (row.get('metrics', {}) or {})
    rendering = (row.get('rendering', {}) or {})
    rendered = (row.get('rendered', {}) or {})
    rendered_meta = (rendered.get('metadata', {}) or {})
    gen_diag = (row.get('generation_diagnostics', {}) or {})

    em = _f(metrics.get('em', metrics.get('EM', 0.0)), 0.0)
    f1 = _f(metrics.get('f1', metrics.get('F1', 0.0)), 0.0)
    answer_correct = 1 if (em >= 1.0 or f1 >= 0.5) else 0

    fallback = 1 if bool(row.get('generation_fallback', False)) else 0
    if fallback <= 0:
        finish_reason = str(gen_diag.get('finish_reason', '') or '').strip().lower()
        if 'fallback' in finish_reason:
            fallback = 1

    rendered_hit = 1 if _f(metrics.get('rendered_supporting_fact_recall', metrics.get('rendered_sf_recall', 0.0)), 0.0) > 0.0 else 0
    if rendered_hit <= 0:
        funnel = (diag.get('stagewise_loss_funnel', {}) or {})
        rendered_hit = 1 if _f(funnel.get('rendered_hit_rate', 0.0), 0.0) > 0.0 else 0

    strict_path_complete = _f(diag.get('strict_path_complete_rate', ((diag.get('connector_metrics', {}) or {}).get('strict_path_complete_rate', 0.0))), 0.0)
    answer_bearing_path_hit = _f(diag.get('answer_bearing_path_hit', ((diag.get('connector_metrics', {}) or {}).get('answer_bearing_path_hit', 0.0)),), 0.0)

    bearing_corridors, _bearing_corridor_rank, bearing_sentences, _bearing_order = _build_answer_bearing_maps(retrieval)

    rendered_sentence_ids = list(row.get('rendered_sentence_ids', rendered.get('sentence_ids', [])) or [])
    rendered_sentences = list(rendered.get('sentences', []) or [])
    rendered_corridor_ids = list(row.get('rendered_corridor_ids', rendered.get('rendered_corridor_ids', [])) or [])
    sentence_text_by_id = {}
    for sid, sent in zip(rendered_sentence_ids, rendered_sentences):
        sid = str(sid or '')
        txt = str(sent or '').strip()
        if sid and txt:
            sentence_text_by_id[sid] = txt

    answer_bearing_chunk_ids = [sid for sid in rendered_sentence_ids if str(sid or '') in bearing_sentences]
    answer_bearing_bundle_ids = [cid for cid in rendered_corridor_ids if str(cid or '') in bearing_corridors]

    answer_bearing_chunk_present = 1 if answer_bearing_chunk_ids else 0
    answer_bearing_bundle_present = 1 if answer_bearing_bundle_ids else 0

    first_answer_bearing_chunk_rank = 0
    for idx, sid in enumerate(rendered_sentence_ids, start=1):
        if str(sid or '') in bearing_sentences:
            first_answer_bearing_chunk_rank = int(idx)
            break

    first_answer_bearing_bundle_rank = 0
    for idx, cid in enumerate(rendered_corridor_ids, start=1):
        if str(cid or '') in bearing_corridors:
            first_answer_bearing_bundle_rank = int(idx)
            break

    sent_token_lens = [max(1, len(content_tokens(str(sentence_text_by_id.get(str(sid or ''), '') or '')))) for sid in rendered_sentence_ids]
    total_rendered_tokens = max(1, int(sum(sent_token_lens)))
    answer_positions = [idx for idx, sid in enumerate(rendered_sentence_ids) if str(sid or '') in bearing_sentences]
    if answer_positions:
        start_pos = int(sum(sent_token_lens[: min(answer_positions)]))
        end_pos = int(sum(sent_token_lens[: max(answer_positions) + 1]))
        answer_bearing_token_start_ratio = _safe_ratio(start_pos, total_rendered_tokens)
        answer_bearing_token_end_ratio = _safe_ratio(end_pos, total_rendered_tokens)
    else:
        answer_bearing_token_start_ratio = 0.0
        answer_bearing_token_end_ratio = 0.0

    answer_bearing_chunk_set = set(str(sid or '') for sid in answer_bearing_chunk_ids if str(sid or ''))
    answer_bearing_bundle_set = set(str(cid or '') for cid in answer_bearing_bundle_ids if str(cid or ''))
    num_answer_bearing_chunks = int(len(answer_bearing_chunk_set))
    num_answer_bearing_bundles = int(len(answer_bearing_bundle_set))

    evidence_dispersion_score = 0.0
    if len(answer_positions) >= 2 and len(rendered_sentence_ids) >= 2:
        span = int(max(answer_positions) - min(answer_positions) + 1)
        packed = int(len(answer_positions))
        evidence_dispersion_score = _safe_ratio(max(0, span - packed), max(1, len(rendered_sentence_ids) - 1))

    evidence_duplication_score = 0.0
    ab_texts = [str(sentence_text_by_id.get(str(sid or ''), '') or '') for sid in answer_bearing_chunk_ids]
    ab_sets = [_tokens(t) for t in ab_texts if t]
    if len(ab_sets) >= 2:
        sims = []
        for i in range(len(ab_sets)):
            for j in range(i + 1, len(ab_sets)):
                sims.append(_jacc(ab_sets[i], ab_sets[j]))
        if sims:
            evidence_duplication_score = float(sum(sims) / float(len(sims)))

    evidence_truncation_flag = 1 if (
        bool(rendered.get('truncated', False))
        or _i(rendering.get('truncated_sentences', 0), 0) > 0
        or _i(rendering.get('truncated_corridors', 0), 0) > 0
        or _i(gen_diag.get('truncated_sentence_count', 0), 0) > 0
        or _i(gen_diag.get('truncated_corridor_count', 0), 0) > 0
    ) else 0

    answer_present_but_generation_fail = 1 if (answer_bearing_chunk_present > 0 and answer_correct <= 0) else 0
    prediction = str(row.get('prediction', '') or '')
    model_output_length = int(len(content_tokens(prediction)))
    pred_tokens = _tokens(prediction)
    evidence_tokens = set()
    for sid in answer_bearing_chunk_ids:
        evidence_tokens.update(_tokens(sentence_text_by_id.get(str(sid or ''), '')))
    output_overlap = 1 if (pred_tokens and evidence_tokens and pred_tokens.intersection(evidence_tokens)) else 0

    item = {
        'stage': stage,
        'variant': variant,
        'dataset': dataset,
        'query_id': str(row.get('sample_id', '') or row.get('sample_index', '')),
        'answer_correct': int(answer_correct),
        'fallback': int(fallback),
        'rendered_hit': int(rendered_hit),
        'answer_bearing_path_hit': float(max(0.0, min(1.0, answer_bearing_path_hit))),
        'answer_bearing_chunk_present': int(answer_bearing_chunk_present),
        'answer_bearing_bundle_present': int(answer_bearing_bundle_present),
        'first_answer_bearing_chunk_rank': int(first_answer_bearing_chunk_rank),
        'first_answer_bearing_bundle_rank': int(first_answer_bearing_bundle_rank),
        'answer_bearing_token_start_ratio': float(max(0.0, min(1.0, answer_bearing_token_start_ratio))),
        'answer_bearing_token_end_ratio': float(max(0.0, min(1.0, answer_bearing_token_end_ratio))),
        'num_answer_bearing_chunks': int(num_answer_bearing_chunks),
        'num_answer_bearing_bundles': int(num_answer_bearing_bundles),
        'evidence_dispersion_score': float(max(0.0, min(1.0, evidence_dispersion_score))),
        'evidence_duplication_score': float(max(0.0, min(1.0, evidence_duplication_score))),
        'evidence_truncation_flag': int(evidence_truncation_flag),
        'answer_present_but_generation_fail': int(answer_present_but_generation_fail),
        'model_output_length': int(model_output_length),
        'output_overlaps_answer_bearing_evidence': int(output_overlap),
        'strict_path_complete': float(max(0.0, min(1.0, strict_path_complete))),
    }
    written.append(item)

out_jsonl.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in written) + ('\n' if written else ''), encoding='utf-8')
with out_csv.open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in written:
        w.writerow({k: r.get(k, '') for k in fields})

if not written:
    out_summary.write_text(json.dumps({'query_count': 0}, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0)

n = float(len(written))
present_chunk = [r for r in written if int(r.get('num_answer_bearing_chunks', 0)) > 0]
present_bundle = [r for r in written if int(r.get('num_answer_bearing_bundles', 0)) > 0]

ret = {
    'query_count': int(len(written)),
    'answer_bearing_chunk_present_rate': float(sum(int(r.get('answer_bearing_chunk_present', 0)) for r in written) / n),
    'answer_bearing_bundle_present_rate': float(sum(int(r.get('answer_bearing_bundle_present', 0)) for r in written) / n),
    'answer_present_but_generation_fail_rate': float(sum(int(r.get('answer_present_but_generation_fail', 0)) for r in written) / n),
    'avg_first_answer_bearing_chunk_rank': float(
        sum(_f(r.get('first_answer_bearing_chunk_rank', 0), 0.0) for r in present_chunk) / float(len(present_chunk))
    ) if present_chunk else 0.0,
    'avg_first_answer_bearing_bundle_rank': float(
        sum(_f(r.get('first_answer_bearing_bundle_rank', 0), 0.0) for r in present_bundle) / float(len(present_bundle))
    ) if present_bundle else 0.0,
    'avg_answer_bearing_token_start_ratio': float(
        sum(_f(r.get('answer_bearing_token_start_ratio', 0.0), 0.0) for r in present_chunk) / float(len(present_chunk))
    ) if present_chunk else 0.0,
    'avg_evidence_dispersion_score': float(sum(_f(r.get('evidence_dispersion_score', 0.0), 0.0) for r in written) / n),
    'avg_evidence_duplication_score': float(sum(_f(r.get('evidence_duplication_score', 0.0), 0.0) for r in written) / n),
    'evidence_truncation_rate': float(sum(int(r.get('evidence_truncation_flag', 0)) for r in written) / n),
    'fallback_rate': float(sum(int(r.get('fallback', 0)) for r in written) / n),
    'strict_path_complete': float(sum(_f(r.get('strict_path_complete', 0.0), 0.0) for r in written) / n),
    'answer_bearing_path_hit': float(sum(_f(r.get('answer_bearing_path_hit', 0.0), 0.0) for r in written) / n),
    'output_overlap_answer_bearing_rate': float(sum(int(r.get('output_overlaps_answer_bearing_evidence', 0)) for r in written) / n),
}

correct = [r for r in written if int(r.get('answer_correct', 0)) > 0]
incorrect = [r for r in written if int(r.get('answer_correct', 0)) <= 0]
gen_fail = [r for r in written if int(r.get('answer_present_but_generation_fail', 0)) > 0]
non_fail = [r for r in written if int(r.get('answer_present_but_generation_fail', 0)) <= 0]

ret['outcome_conditioned'] = {
    'correct_subset': _subset_stats(correct),
    'incorrect_subset': _subset_stats(incorrect),
    'generation_fail_subset': _subset_stats(gen_fail),
    'non_fail_subset': _subset_stats(non_fail),
}

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
    payload = {'dataset': dataset, 'sample_count': 0, 'prompt_over_limit_rate': 1.0, 'pass': False, 'reason': 'reference_summary_missing'}
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(out_md).write_text(f"# Prompt Sanity ({dataset})\\n\\n- pass: false\\n- reason: reference_summary_missing\\n", encoding='utf-8')
    raise SystemExit(5)

summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
qpath = Path(summary_path).with_name('rag_query_results.jsonl')
if not qpath.exists():
    payload = {'dataset': dataset, 'sample_count': 0, 'prompt_over_limit_rate': 1.0, 'pass': False, 'reason': 'reference_query_results_missing'}
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
        over.append({'sample_id': str(row.get('sample_id', '')), 'prompt_tokens': int(ptok), 'max_new_tokens': int(max_new), 'max_model_len': int(max_model_len), 'over_by': int(total - max_model_len)})

n = len(prompt_tokens)
over_rate = (len(over) / float(n)) if n > 0 else 1.0
payload = {
    'dataset': dataset,
    'reference_summary_path': str(Path(summary_path).resolve()),
    'reference_query_results_path': str(qpath.resolve()),
    'sample_count': int(n),
    'prompt_over_limit_rate': float(over_rate),
    'max_new_tokens': int(max_new),
    'max_model_len': int(max_model_len),
    'pass': bool(n > 0 and over_rate <= 1.0e-12),
}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
Path(out_md).write_text(
    "\\n".join([
        f"# Prompt Sanity ({dataset})",
        "",
        f"- sample_count: {n}",
        f"- prompt_over_limit_rate: {over_rate:.6f}",
        f"- max_new_tokens: {max_new}",
        f"- max_model_len: {max_model_len}",
        f"- pass: {payload['pass']}",
    ]) + "\\n",
    encoding='utf-8',
)
print('PASS' if payload['pass'] else 'FAIL')
raise SystemExit(0 if payload['pass'] else 5)
PY
}

dataset_ref_knobs_json() {
  local dataset="$1"
  if [[ "${dataset}" == "hotpotqa" ]]; then
    echo '{"candidate_top_t":30,"semantic_topn_chunk":25,"graph_reserve_topn":25}'
  else
    echo '{"semantic_topn_chunk":25}'
  fi
}

run_rag_variant() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local profile_key="$4"
  local retrieval_mode="$5"
  local knob_json="$6"
  local order_strategy="$7"
  local limit="$8"

  local cfg_dir="${CFG_ROOT}/${stage}/${variant}/${dataset}"
  local rag_cfg="${cfg_dir}/rag.yaml"
  local out_dir="${RUN_ROOT}/${stage}/${variant}/rag"
  local log_file="${LOG_ROOT}/${stage}__${variant}__${dataset}.log"
  mkdir -p "${cfg_dir}" "${out_dir}" "$(dirname "${log_file}")"

  create_variant_cfg "${RAG_BASE_CFG}" "${rag_cfg}" "${dataset}" "${variant}" "${retrieval_mode}" "${knob_json}" "${order_strategy}"

  if [[ "${SKIP_COMPLETED}" == "true" ]]; then
    local existing_summary
    existing_summary="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
    if [[ -n "${existing_summary}" ]]; then
      if is_complete_summary "${existing_summary}" "${limit}" "rag_query_results.jsonl" >/dev/null 2>&1; then
        annotate_summary_with_refs "${existing_summary}" "${dataset}"
        export_query_conversion_diagnostics "${existing_summary}" "${stage}" "${variant}" "${dataset}"
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
    echo "[ORDER_STRATEGY] ${order_strategy}"
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
  export_query_conversion_diagnostics "${summary_path}" "${stage}" "${variant}" "${dataset}"
  upsert_record "${stage}" "${variant}" "${dataset}" "${profile_key}" "rag" "${summary_path}" "${log_file}" "ok"
  log_msg "[DONE] stage=${stage} variant=${variant} dataset=${dataset}"
}

run_stage1() {
  log_msg "[Stage 1] baseline integrity + reference reconfirm"
  run_rag_variant "s1" "baseline_mid_reconfirm" "hotpotqa" "baseline_mid" "baseline" "{}" "score" "${LIMIT_RAG}"
  run_rag_variant "s1" "baseline_mid_reconfirm" "2wikimultihopqa" "baseline_mid" "baseline" "{}" "score" "${LIMIT_RAG}"

  run_rag_variant "s1" "ref_reconfirm" "hotpotqa" "strong_ref_hotpot" "r2_plus_path_preserve" "$(dataset_ref_knobs_json hotpotqa)" "score" "${LIMIT_RAG}"
  run_rag_variant "s1" "ref_reconfirm" "2wikimultihopqa" "strong_ref_2wiki" "r2_plus_path_preserve" "$(dataset_ref_knobs_json 2wikimultihopqa)" "score" "${LIMIT_RAG}"
}

run_stage2() {
  log_msg "[Stage 2] minimal interface variants"
  run_rag_variant "s2" "raw_focus_front" "hotpotqa" "strong_ref_hotpot" "r2_plus_path_preserve" "$(dataset_ref_knobs_json hotpotqa)" "score+raw_focus_front" "${LIMIT_RAG}"
  run_rag_variant "s2" "raw_focus_front" "2wikimultihopqa" "strong_ref_2wiki" "r2_plus_path_preserve" "$(dataset_ref_knobs_json 2wikimultihopqa)" "score+raw_focus_front" "${LIMIT_RAG}"

  run_rag_variant "s2" "raw_focus_front_dedup" "hotpotqa" "strong_ref_hotpot" "r2_plus_path_preserve" "$(dataset_ref_knobs_json hotpotqa)" "score+raw_focus_front+raw_focus_dedup" "${LIMIT_RAG}"
  run_rag_variant "s2" "raw_focus_front_dedup" "2wikimultihopqa" "strong_ref_2wiki" "r2_plus_path_preserve" "$(dataset_ref_knobs_json 2wikimultihopqa)" "score+raw_focus_front+raw_focus_dedup" "${LIMIT_RAG}"

  run_rag_variant "s2" "raw_focus_front_scaffold_light" "hotpotqa" "strong_ref_hotpot" "r2_plus_path_preserve" "$(dataset_ref_knobs_json hotpotqa)" "score+raw_focus_front+raw_focus_scaffold_light" "${LIMIT_RAG}"
  run_rag_variant "s2" "raw_focus_front_scaffold_light" "2wikimultihopqa" "strong_ref_2wiki" "r2_plus_path_preserve" "$(dataset_ref_knobs_json 2wikimultihopqa)" "score+raw_focus_front+raw_focus_scaffold_light" "${LIMIT_RAG}"
}

run_optional_stage2_top_focus() {
  log_msg "[Stage 2-opt] top1 answer-bearing bundle only"
  run_rag_variant "s2_opt" "top1_answer_bearing_bundle_only" "hotpotqa" "strong_ref_hotpot" "r2_plus_path_preserve" "$(dataset_ref_knobs_json hotpotqa)" "score+raw_focus_front+raw_focus_dedup+raw_focus_top1_bundle_only" "${LIMIT_RAG}"
  run_rag_variant "s2_opt" "top1_answer_bearing_bundle_only" "2wikimultihopqa" "strong_ref_2wiki" "r2_plus_path_preserve" "$(dataset_ref_knobs_json 2wikimultihopqa)" "score+raw_focus_front+raw_focus_dedup+raw_focus_top1_bundle_only" "${LIMIT_RAG}"
}

check_reference_integrity() {
  "${PYTHON_BIN}" - "${RECORD_TSV}" <<'PY'
import json
import sys
from pathlib import Path

record_tsv = Path(sys.argv[1])
need = {
    ("s1", "baseline_mid_reconfirm", "hotpotqa"),
    ("s1", "baseline_mid_reconfirm", "2wikimultihopqa"),
    ("s1", "ref_reconfirm", "hotpotqa"),
    ("s1", "ref_reconfirm", "2wikimultihopqa"),
}
seen = set()
summaries = []
with record_tsv.open("r", encoding="utf-8") as f:
    _ = f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) != 8:
            continue
        stage, variant, dataset, _profile, kind, summary_path, _log_path, status = p
        if kind != "rag" or status not in {"ok", "skipped"}:
            continue
        key = (stage, variant, dataset)
        if key in need:
            seen.add(key)
            summaries.append((variant, dataset, summary_path))

missing = sorted(list(need - seen))
if missing:
    raise SystemExit(f"missing_reference_runs:{missing}")

for variant, dataset, sp in summaries:
    p = Path(sp)
    if not p.exists():
        raise SystemExit(f"missing_summary:{variant}:{dataset}")
    summary = json.loads(p.read_text(encoding="utf-8"))
    fb = float(summary.get("fallback_rate", 0.0) or 0.0)
    if fb < 0.0 or fb > 1.0:
        raise SystemExit(f"invalid_fallback_rate:{variant}:{dataset}:{fb}")
    print(f"[REF] {variant} {dataset} fallback_rate={fb:.4f}")
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


def _r5(summary):
    direct = _f(summary.get('supporting_fact_recall_at_5', 0.0), 0.0)
    if direct > 0.0:
        return direct
    bag = summary.get('supporting_fact_recall_at_k', {}) or {}
    if '5' in bag:
        return _f(bag['5'], 0.0)
    if 5 in bag:
        return _f(bag[5], 0.0)
    return _f(summary.get('recall_at_5', 0.0), 0.0)


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

        conv_summary = {}
        conv_path = sp.with_name('conversion_diagnostics_summary.json')
        if conv_path.exists():
            try:
                conv_summary = json.loads(conv_path.read_text(encoding='utf-8'))
            except Exception:
                conv_summary = {}
        qpath = sp.with_name('conversion_query_diagnostics.jsonl')
        if qpath.exists():
            for qline in qpath.read_text(encoding='utf-8').splitlines():
                try:
                    all_query_rows.append(json.loads(qline))
                except Exception:
                    pass

        r5 = _r5(summary)
        em = _f(summary.get('em', summary.get('EM', 0.0)), 0.0)
        f1 = _f(summary.get('f1', summary.get('F1', 0.0)), 0.0)
        retrieval_ms = _f(summary.get('retrieval_latency_ms', summary.get('retrieval_ms', 0.0)), 0.0)
        generation_ms = _f(summary.get('generation_latency_ms', summary.get('generation_ms', 0.0)), 0.0)
        total_ms = _f(summary.get('total_latency_ms', summary.get('total_ms', 0.0)), 0.0)
        fallback_rate = _f(summary.get('fallback_rate', 0.0), 0.0)

        strict_path_complete = _f(summary.get('strict_path_complete_rate', 0.0), 0.0)
        answer_bearing_path_hit = _f(summary.get('answer_bearing_path_hit', 0.0), 0.0)

        hippo = ((refs.get(dataset, {}) or {}).get('hipporag2', {}) or {})
        d2 = ((refs.get(dataset, {}) or {}).get('d2', {}) or {})
        rows.append({
            'stage': stage,
            'variant': variant,
            'dataset': dataset,
            'profile': profile,
            'summary_path': str(sp),
            'R@5': r5,
            'EM': em,
            'F1': f1,
            'retrieval_ms': retrieval_ms,
            'generation_ms': generation_ms,
            'total_ms': total_ms,
            'fallback_rate': fallback_rate,
            'strict_path_complete': strict_path_complete,
            'answer_bearing_path_hit': answer_bearing_path_hit,
            'answer_bearing_chunk_present_rate': _f(conv_summary.get('answer_bearing_chunk_present_rate', 0.0), 0.0),
            'answer_bearing_bundle_present_rate': _f(conv_summary.get('answer_bearing_bundle_present_rate', 0.0), 0.0),
            'answer_present_but_generation_fail_rate': _f(conv_summary.get('answer_present_but_generation_fail_rate', 0.0), 0.0),
            'avg_first_answer_bearing_chunk_rank': _f(conv_summary.get('avg_first_answer_bearing_chunk_rank', 0.0), 0.0),
            'avg_first_answer_bearing_bundle_rank': _f(conv_summary.get('avg_first_answer_bearing_bundle_rank', 0.0), 0.0),
            'avg_answer_bearing_token_start_ratio': _f(conv_summary.get('avg_answer_bearing_token_start_ratio', 0.0), 0.0),
            'avg_evidence_dispersion_score': _f(conv_summary.get('avg_evidence_dispersion_score', 0.0), 0.0),
            'avg_evidence_duplication_score': _f(conv_summary.get('avg_evidence_duplication_score', 0.0), 0.0),
            'evidence_truncation_rate': _f(conv_summary.get('evidence_truncation_rate', 0.0), 0.0),
            'output_overlap_answer_bearing_rate': _f(conv_summary.get('output_overlap_answer_bearing_rate', 0.0), 0.0),
            'conv_outcomes': dict(conv_summary.get('outcome_conditioned', {}) or {}),
            'gap_to_hippo_r5': (_f(hippo.get('r5', 0.0), 0.0) - r5) if _f(hippo.get('r5', 0.0), 0.0) > 0.0 else 0.0,
            'oracle_gap': (_f(d2.get('f1', 0.0), 0.0) - f1) if _f(d2.get('f1', 0.0), 0.0) > 0.0 else 0.0,
        })

# Add external references.
for ds in ('hotpotqa', '2wikimultihopqa'):
    ds_ref = refs.get(ds, {}) or {}
    for key, name in (
        ('best_non_oracle', 'frozen_best_non_oracle'),
        ('d2', 'D2_upper_bound'),
        ('hipporag2', 'HippoRAG2'),
    ):
        ref = ds_ref.get(key, {}) or {}
        if not ref:
            continue
        rows.append({
            'stage': 'reference',
            'variant': f"{name}({ref.get('label','')})",
            'dataset': ds,
            'profile': 'reference',
            'summary_path': str(ref.get('path', '')),
            'R@5': _f(ref.get('r5', 0.0), 0.0),
            'EM': _f(ref.get('em', 0.0), 0.0),
            'F1': _f(ref.get('f1', 0.0), 0.0),
            'retrieval_ms': _f(ref.get('retrieval_ms', 0.0), 0.0),
            'generation_ms': _f(ref.get('generation_ms', 0.0), 0.0),
            'total_ms': _f(ref.get('total_ms', 0.0), 0.0),
            'fallback_rate': 0.0,
            'strict_path_complete': 0.0,
            'answer_bearing_path_hit': 0.0,
            'answer_bearing_chunk_present_rate': 0.0,
            'answer_bearing_bundle_present_rate': 0.0,
            'answer_present_but_generation_fail_rate': 0.0,
            'avg_first_answer_bearing_chunk_rank': 0.0,
            'avg_first_answer_bearing_bundle_rank': 0.0,
            'avg_answer_bearing_token_start_ratio': 0.0,
            'avg_evidence_dispersion_score': 0.0,
            'avg_evidence_duplication_score': 0.0,
            'evidence_truncation_rate': 0.0,
            'output_overlap_answer_bearing_rate': 0.0,
            'conv_outcomes': {},
            'gap_to_hippo_r5': 0.0,
            'oracle_gap': 0.0,
        })

# Delta vs dataset-specific strong reference (ref_reconfirm).
ref_by_ds = {}
for r in rows:
    if r.get('stage') == 'reference':
        continue
    if r.get('variant') == 'ref_reconfirm' and r.get('dataset') not in ref_by_ds:
        ref_by_ds[r.get('dataset')] = r

for r in rows:
    if r.get('stage') == 'reference':
        r['delta_R5_vs_reference'] = 0.0
        r['delta_EM_vs_reference'] = 0.0
        r['delta_F1_vs_reference'] = 0.0
        r['delta_retrieval_ms_vs_reference'] = 0.0
        r['delta_generation_ms_vs_reference'] = 0.0
        r['delta_total_ms_vs_reference'] = 0.0
        r['f1_per_100ms'] = 0.0
        continue
    ref = ref_by_ds.get(r.get('dataset'))
    if ref is None:
        r['delta_R5_vs_reference'] = 0.0
        r['delta_EM_vs_reference'] = 0.0
        r['delta_F1_vs_reference'] = 0.0
        r['delta_retrieval_ms_vs_reference'] = 0.0
        r['delta_generation_ms_vs_reference'] = 0.0
        r['delta_total_ms_vs_reference'] = 0.0
    else:
        r['delta_R5_vs_reference'] = _f(r['R@5']) - _f(ref['R@5'])
        r['delta_EM_vs_reference'] = _f(r['EM']) - _f(ref['EM'])
        r['delta_F1_vs_reference'] = _f(r['F1']) - _f(ref['F1'])
        r['delta_retrieval_ms_vs_reference'] = _f(r['retrieval_ms']) - _f(ref['retrieval_ms'])
        r['delta_generation_ms_vs_reference'] = _f(r['generation_ms']) - _f(ref['generation_ms'])
        r['delta_total_ms_vs_reference'] = _f(r['total_ms']) - _f(ref['total_ms'])
    r['f1_per_100ms'] = (_f(r['F1']) * 100.0 / _f(r['total_ms'])) if _f(r['total_ms']) > 0 else 0.0

payload = {
    'records': rows,
    'query_diagnostics_jsonl': str(Path(out_q_jsonl).resolve()),
    'query_diagnostics_csv': str(Path(out_q_csv).resolve()),
}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

Path(out_q_jsonl).write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in all_query_rows) + ('\n' if all_query_rows else ''), encoding='utf-8')
if all_query_rows:
    q_fields = [
        'stage','variant','dataset','query_id','answer_correct','fallback','rendered_hit','answer_bearing_path_hit',
        'answer_bearing_chunk_present','answer_bearing_bundle_present','first_answer_bearing_chunk_rank','first_answer_bearing_bundle_rank',
        'answer_bearing_token_start_ratio','answer_bearing_token_end_ratio','num_answer_bearing_chunks','num_answer_bearing_bundles',
        'evidence_dispersion_score','evidence_duplication_score','evidence_truncation_flag','answer_present_but_generation_fail',
        'model_output_length','output_overlaps_answer_bearing_evidence','strict_path_complete'
    ]
    with Path(out_q_csv).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=q_fields)
        w.writeheader()
        for q in all_query_rows:
            w.writerow({k: q.get(k, '') for k in q_fields})
else:
    with Path(out_q_csv).open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stage','variant','dataset','query_id','answer_correct','fallback','rendered_hit','answer_bearing_path_hit','answer_bearing_chunk_present','answer_bearing_bundle_present','first_answer_bearing_chunk_rank','first_answer_bearing_bundle_rank','answer_bearing_token_start_ratio','answer_bearing_token_end_ratio','num_answer_bearing_chunks','num_answer_bearing_bundles','evidence_dispersion_score','evidence_duplication_score','evidence_truncation_flag','answer_present_but_generation_fail','model_output_length','output_overlaps_answer_bearing_evidence','strict_path_complete'])

lines = [
    '# Conversion Bottleneck Followup Metrics (RAG full)',
    '',
    '## Main Performance Table',
    '',
    '| stage | variant | dataset | R@5 | EM | F1 | retrieval_ms | generation_ms | total_ms | fallback_rate | gap_to_hippo_r5 | oracle_gap |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    lines.append(
        f"| {r['stage']} | {r['variant']} | {r['dataset']} | {r['R@5']:.4f} | {r['EM']:.4f} | {r['F1']:.4f} | {r['retrieval_ms']:.2f} | {r['generation_ms']:.2f} | {r['total_ms']:.2f} | {r['fallback_rate']:.4f} | {r['gap_to_hippo_r5']:+.4f} | {r['oracle_gap']:+.4f} |"
    )

lines += [
    '',
    '## Conversion Diagnostic Table',
    '',
    '| stage | variant | dataset | answer_bearing_chunk_present_rate | answer_bearing_bundle_present_rate | answer_present_but_generation_fail_rate | avg_first_answer_bearing_chunk_rank | avg_first_answer_bearing_bundle_rank | avg_answer_bearing_token_start_ratio | avg_evidence_dispersion_score | avg_evidence_duplication_score | evidence_truncation_rate | strict_path_complete | answer_bearing_path_hit |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    if r['stage'] == 'reference':
        continue
    lines.append(
        f"| {r['stage']} | {r['variant']} | {r['dataset']} | {r['answer_bearing_chunk_present_rate']:.4f} | {r['answer_bearing_bundle_present_rate']:.4f} | {r['answer_present_but_generation_fail_rate']:.4f} | {r['avg_first_answer_bearing_chunk_rank']:.3f} | {r['avg_first_answer_bearing_bundle_rank']:.3f} | {r['avg_answer_bearing_token_start_ratio']:.4f} | {r['avg_evidence_dispersion_score']:.4f} | {r['avg_evidence_duplication_score']:.4f} | {r['evidence_truncation_rate']:.4f} | {r['strict_path_complete']:.4f} | {r['answer_bearing_path_hit']:.4f} |"
    )

lines += [
    '',
    '## Efficiency Table (vs dataset reference: ref_reconfirm)',
    '',
    '| stage | variant | dataset | ΔR@5 | ΔEM | ΔF1 | Δretrieval_ms | Δgeneration_ms | Δtotal_ms | F1_per_100ms |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    if r['stage'] == 'reference':
        continue
    lines.append(
        f"| {r['stage']} | {r['variant']} | {r['dataset']} | {r['delta_R5_vs_reference']:+.4f} | {r['delta_EM_vs_reference']:+.4f} | {r['delta_F1_vs_reference']:+.4f} | {r['delta_retrieval_ms_vs_reference']:+.2f} | {r['delta_generation_ms_vs_reference']:+.2f} | {r['delta_total_ms_vs_reference']:+.2f} | {r['f1_per_100ms']:.4f} |"
    )

lines += [
    '',
    '## Outcome-conditioned (Correct vs Incorrect)',
    '',
    '| variant | dataset | subset | count | answer_bearing_chunk_present_rate | avg_first_answer_bearing_chunk_rank | avg_evidence_dispersion_score | avg_evidence_duplication_score | output_overlap_answer_bearing_rate | fallback_rate |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    if r['stage'] == 'reference':
        continue
    outcomes = r.get('conv_outcomes', {}) or {}
    for subset_key, subset_name in (('correct_subset', 'correct'), ('incorrect_subset', 'incorrect')):
        s = outcomes.get(subset_key, {}) or {}
        lines.append(
            f"| {r['variant']} | {r['dataset']} | {subset_name} | {int(_f(s.get('count', 0), 0))} | {_f(s.get('answer_bearing_chunk_present_rate', 0.0), 0.0):.4f} | {_f(s.get('avg_first_answer_bearing_chunk_rank', 0.0), 0.0):.3f} | {_f(s.get('avg_evidence_dispersion_score', 0.0), 0.0):.4f} | {_f(s.get('avg_evidence_duplication_score', 0.0), 0.0):.4f} | {_f(s.get('output_overlap_answer_bearing_rate', 0.0), 0.0):.4f} | {_f(s.get('fallback_rate', 0.0), 0.0):.4f} |"
        )

lines += [
    '',
    '## Outcome-conditioned (Generation Fail vs Non-fail)',
    '',
    '| variant | dataset | subset | count | answer_bearing_chunk_present_rate | avg_first_answer_bearing_chunk_rank | avg_evidence_dispersion_score | avg_evidence_duplication_score | output_overlap_answer_bearing_rate | fallback_rate |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    if r['stage'] == 'reference':
        continue
    outcomes = r.get('conv_outcomes', {}) or {}
    for subset_key, subset_name in (('generation_fail_subset', 'generation_fail'), ('non_fail_subset', 'non_fail')):
        s = outcomes.get(subset_key, {}) or {}
        lines.append(
            f"| {r['variant']} | {r['dataset']} | {subset_name} | {int(_f(s.get('count', 0), 0))} | {_f(s.get('answer_bearing_chunk_present_rate', 0.0), 0.0):.4f} | {_f(s.get('avg_first_answer_bearing_chunk_rank', 0.0), 0.0):.3f} | {_f(s.get('avg_evidence_dispersion_score', 0.0), 0.0):.4f} | {_f(s.get('avg_evidence_duplication_score', 0.0), 0.0):.4f} | {_f(s.get('output_overlap_answer_bearing_rate', 0.0), 0.0):.4f} | {_f(s.get('fallback_rate', 0.0), 0.0):.4f} |"
        )

Path(out_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
}

check_output_artifacts() {
  "${PYTHON_BIN}" - "${RECORD_TSV}" "${ROUND_METRICS_JSON}" "${ROUND_METRICS_MD}" <<'PY'
import json
import sys
from pathlib import Path

record_tsv = Path(sys.argv[1])
round_json = Path(sys.argv[2])
round_md = Path(sys.argv[3])
if not record_tsv.exists():
    raise SystemExit("run_records_missing")
if not round_json.exists():
    raise SystemExit("round_metrics_json_missing")
if not round_md.exists():
    raise SystemExit("round_metrics_md_missing")
payload = json.loads(round_json.read_text(encoding="utf-8"))
records = list(payload.get("records", []) or [])
if not records:
    raise SystemExit("round_metrics_empty")

need = [
    ("s1", "baseline_mid_reconfirm", "hotpotqa"),
    ("s1", "baseline_mid_reconfirm", "2wikimultihopqa"),
    ("s1", "ref_reconfirm", "hotpotqa"),
    ("s1", "ref_reconfirm", "2wikimultihopqa"),
    ("s2", "raw_focus_front", "hotpotqa"),
    ("s2", "raw_focus_front", "2wikimultihopqa"),
]
seen = {(r.get("stage"), r.get("variant"), r.get("dataset")) for r in records if r.get("stage") != "reference"}
missing = [x for x in need if x not in seen]
if missing:
    raise SystemExit(f"missing_required_rows:{missing}")
print("ok")
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
check_reference_integrity

run_stage2
if [[ "${RUN_OPTIONAL_TOP_FOCUS}" == "true" ]]; then
  run_optional_stage2_top_focus
fi

check_cache_meta_race
build_round_metrics
check_output_artifacts

log_msg "[DONE] outputs: ${ROUND_ROOT}"
log_msg "[DONE] logs: ${LOG_ROOT}"
log_msg "[DONE] run records: ${RECORD_TSV}"
log_msg "[DONE] query diagnostics: ${QUERY_DIAG_JSONL}"
log_msg "[DONE] round metrics: ${ROUND_METRICS_JSON}"
