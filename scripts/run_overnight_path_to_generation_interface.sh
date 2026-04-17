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

# Execution knobs
LIMIT_RAG="${LIMIT_RAG:-1000}"
PROMPT_SANITY_LIMIT="${PROMPT_SANITY_LIMIT:-200}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-true}"
RUN_PREFLIGHT_CHECKS="${RUN_PREFLIGHT_CHECKS:-true}"
RUN_SMOKE="${RUN_SMOKE:-false}"
AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-true}"
FORCE_NEW_ROUND="${FORCE_NEW_ROUND:-false}"
# Universal embedding regime is frozen for this round.
UNIVERSAL_REGIME="${UNIVERSAL_REGIME:-mid}"
UNIVERSAL_EMBED_MAX_LEN="${UNIVERSAL_EMBED_MAX_LEN:-256}"
UNIVERSAL_EMBED_MAX_CHARS="${UNIVERSAL_EMBED_MAX_CHARS:-800}"

GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
RAG_BASE_CFG="configs/canonical/rag_entity_first_chunk_grounded.yaml"
RETR_BASE_CFG="configs/canonical/retrieval_entity_first_chunk_grounded.yaml"

# 0 => resolve from running vLLM process or fallback(4096)
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-0}"

# Frozen references
HOTPOT_D1_REF_SUMMARY="${HOTPOT_D1_REF_SUMMARY:-outputs/exp_retrieval_gap_recheck/hotpotqa/D1/rag/hotpotqa/20260406_053010_574786/rag_summary.json}"
HOTPOT_F2_REF_SUMMARY="${HOTPOT_F2_REF_SUMMARY:-outputs/exp_retrieval_gap/hotpotqa/F2/rag/hotpotqa/20260405_141352_352678/rag_summary.json}"
WIKI_D1_REF_SUMMARY="${WIKI_D1_REF_SUMMARY:-outputs/exp_retrieval_gap_recheck/2wikimultihopqa/D1/rag/2wikimultihopqa/20260406_054816_597964/rag_summary.json}"
WIKI_E1_REF_SUMMARY="${WIKI_E1_REF_SUMMARY:-outputs/exp_retrieval_gap/2wikimultihopqa/E1/rag/2wikimultihopqa/20260405_144256_613687/rag_summary.json}"
WIKI_W3_REF_SUMMARY="${WIKI_W3_REF_SUMMARY:-}"
D2_HOTPOT_REF_SUMMARY="${D2_HOTPOT_REF_SUMMARY:-outputs/exp_retrieval_gap_recheck/hotpotqa/D2/rag/hotpotqa/20260406_053942_373258/rag_summary.json}"
D2_WIKI_REF_SUMMARY="${D2_WIKI_REF_SUMMARY:-outputs/exp_retrieval_gap/2wikimultihopqa/D2/rag/2wikimultihopqa/20260405_143345_219316/rag_summary.json}"
HIPPORAG2_HOTPOT_SUMMARY="${HIPPORAG2_HOTPOT_SUMMARY:-/home/ojungii/HippoRAG2/outputs/hotpotqa/hotpotqa_eval_summary_results.json}"
HIPPORAG2_WIKI_SUMMARY="${HIPPORAG2_WIKI_SUMMARY:-/home/ojungii/HippoRAG2/outputs/2wikimultihopqa/2wikimultihopqa_eval_summary_results.json}"
# Use fixed HippoRAG2 reference table (2026-03-23/24) for consistent gap tracking.
HIPPORAG2_USE_FIXED_TABLE="${HIPPORAG2_USE_FIXED_TABLE:-true}"

# Optional artifact sources for traceability
ROUND_METRICS_REF="${ROUND_METRICS_REF:-$(find outputs -type f -name round_metrics.json | sort | tail -n 1)}"
RUN_RECORDS_REF="${RUN_RECORDS_REF:-$(find outputs -type f -name run_records.tsv | sort | tail -n 1)}"
CURRENT_PROPOSAL_ROUND_METRICS="${CURRENT_PROPOSAL_ROUND_METRICS:-$(find outputs/overnight_proposal_round -type f -name round_metrics.json 2>/dev/null | sort | tail -n 1)}"

ROUND_ROOT_SOURCE="new_round"
if [[ -n "${RESUME_ROUND_ROOT:-}" ]]; then
  ROUND_ROOT="${RESUME_ROUND_ROOT}"
  ROUND_ROOT_SOURCE="resume_env"
elif [[ "${FORCE_NEW_ROUND}" == "true" ]]; then
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="outputs/overnight_path_to_generation_interface/${RUN_STAMP}"
  ROUND_ROOT_SOURCE="force_new"
elif [[ "${AUTO_RESUME_LATEST}" == "true" ]]; then
  LATEST_ROUND_ROOT="$(find outputs/overnight_path_to_generation_interface -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${LATEST_ROUND_ROOT}" ]]; then
    ROUND_ROOT="${LATEST_ROUND_ROOT}"
    ROUND_ROOT_SOURCE="auto_resume_latest"
  else
    RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
    ROUND_ROOT="outputs/overnight_path_to_generation_interface/${RUN_STAMP}"
    ROUND_ROOT_SOURCE="new_round"
  fi
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="outputs/overnight_path_to_generation_interface/${RUN_STAMP}"
  ROUND_ROOT_SOURCE="new_round"
fi

LOG_ROOT="logs/overnight_path_to_generation_interface/$(basename "${ROUND_ROOT}")"
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
UNIVERSAL_SELECTION_JSON="${ROUND_ROOT}/universal_regime_selection.json"

log_msg() {
  local now
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${now}] $*"
}

maybe_snapshot_git_state() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log_msg "[WARN] git repository not detected in ${REPO_ROOT}; skip snapshot/branch step."
    return 0
  fi
  if [[ "${ENABLE_GIT_SNAPSHOT:-false}" != "true" ]]; then
    log_msg "[INFO] Git snapshot step is disabled. Enable with ENABLE_GIT_SNAPSHOT=true"
    log_msg "[INFO] Planned commands:"
    log_msg "[INFO]   git add -A"
    log_msg "[INFO]   git commit -m 'chore: snapshot before path-to-generation interface refinement'"
    return 0
  fi

  git add -A
  if ! git diff --cached --quiet; then
    git commit -m "chore: snapshot before path-to-generation interface refinement"
  else
    log_msg "[INFO] No staged changes to commit before redesign branch."
  fi
  log_msg "[INFO] Snapshot commit done on branch: $(git rev-parse --abbrev-ref HEAD)"
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

build_reference_bundle() {
  "${PYTHON_BIN}" - \
    "${REF_BUNDLE_JSON}" \
    "${RAG_BASE_CFG}" \
    "${RETR_BASE_CFG}" \
    "${HOTPOT_D1_REF_SUMMARY}" \
    "${HOTPOT_F2_REF_SUMMARY}" \
    "${WIKI_D1_REF_SUMMARY}" \
    "${WIKI_E1_REF_SUMMARY}" \
    "${WIKI_W3_REF_SUMMARY}" \
    "${D2_HOTPOT_REF_SUMMARY}" \
    "${D2_WIKI_REF_SUMMARY}" \
    "${HIPPORAG2_HOTPOT_SUMMARY}" \
    "${HIPPORAG2_WIKI_SUMMARY}" \
    "${ROUND_METRICS_REF}" \
    "${RUN_RECORDS_REF}" \
    "${CURRENT_PROPOSAL_ROUND_METRICS}" \
    "${HIPPORAG2_USE_FIXED_TABLE}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

(
    out_json,
    rag_cfg_path,
    retr_cfg_path,
    hotpot_d1_path,
    hotpot_f2_path,
    wiki_d1_path,
    wiki_e1_path,
    wiki_w3_path,
    d2_hotpot_path,
    d2_wiki_path,
    hippo_hotpot_path,
    hippo_wiki_path,
    round_metrics_ref,
    run_records_ref,
    current_proposal_round_metrics,
    hippo_use_fixed_table,
) = sys.argv[1:17]


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


def _as_bool(v):
    return str(v or "").strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _metric(summary, *keys):
    for key in keys:
        if key in summary:
            val = _safe_float(summary.get(key), 0.0)
            if val != 0.0:
                return val
    return 0.0


def _r_at(summary, k):
    direct = _metric(summary, f"supporting_fact_recall_at_{k}", f"Recall@{k}", f"recall_at_{k}", f"R@{k}")
    if direct > 0.0:
        return direct
    s = summary.get("supporting_fact_recall_at_k", {}) or {}
    if str(k) in s:
        return _safe_float(s[str(k)], 0.0)
    if k in s:
        return _safe_float(s[k], 0.0)
    s2 = summary.get("recall_at_k", {}) or {}
    if str(k) in s2:
        return _safe_float(s2[str(k)], 0.0)
    if k in s2:
        return _safe_float(s2[k], 0.0)
    return 0.0


def _read_sidecar(summary_path):
    run_dir = Path(summary_path).resolve().parent
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return {}, ""
    cands = sorted(logs_dir.glob("config_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        try:
            return json.loads(p.read_text(encoding="utf-8")), str(p.resolve())
        except Exception:
            continue
    return {}, ""


def _extract_retrieval_knobs(summary, sidecar, retr_cfg):
    rp = dict(summary.get("retrieval_params", {}) or {})
    if not rp:
        rp = dict(sidecar.get("retrieval_params", {}) or {})
    out = {}
    keys = [
        "semantic_topn_entity",
        "semantic_topn_chunk",
        "graph_reserve_topn",
        "max_anchors",
        "samples_per_anchor",
        "phase1_run_preshortlist_topm",
        "phase1_full_run_score_topk",
        "phase1_run_shortlist_topk",
        "final_top_slice_reorder_enabled",
        "final_top_slice_reorder_topk",
        "answer_support_pinning_enabled",
        "answer_support_pinning_min",
        "embedding_text_max_chars",
        "embedding_max_length",
        "openie_text_max_chars",
        "openie_max_new_tokens",
        "graph_mode",
        "index_chunk_unit",
        "passage_chunking_strategy",
        "passage_chunk_size_sentences",
        "passage_chunk_stride_sentences",
        "passage_chunk_min_sentences",
        "passage_chunk_max_chars",
    ]
    for k in keys:
        if k in rp:
            out[k] = rp[k]
        elif k in retr_cfg:
            out[k] = retr_cfg[k]
    return out


def _extract_render_profile(summary, sidecar, rag_cfg):
    src = dict(summary.get("render_params", {}) or {})
    if not src:
        src = dict(sidecar.get("render_params", {}) or {})
    keys = [
        "render_mode",
        "max_context_sentences",
        "max_corridors_in_context",
        "max_main_sentences_per_corridor",
        "max_support_per_corridor",
        "max_total_sentences",
        "max_sentences",
        "alpha",
        "beta",
        "gamma_main",
        "delta_support",
        "eta_connector",
        "zeta_query",
        "xi_locality",
        "lambda_redundancy",
        "top_corridors",
        "reserve_top_corridor",
        "order_strategy",
        "chunk_grounding_enabled",
        "chunk_grounding_mode",
        "chunk_excerpt_max_per_corridor",
        "chunk_excerpt_window_sentences_before",
        "chunk_excerpt_window_sentences_after",
        "chunk_excerpt_max_total_sentences",
        "chunk_excerpt_dedup_enabled",
        "chunk_grounding_top_corridor_chunks",
        "chunk_grounding_top_k_packages",
        "package_score_answer_weight",
        "package_score_bridge_weight",
        "package_score_support_weight",
        "package_score_chunk_grounding_weight",
        "package_score_redundancy_weight",
    ]
    out = {}
    for k in keys:
        if k in src:
            out[k] = src[k]
        elif k in rag_cfg:
            out[k] = rag_cfg[k]
    return out


def _extract_run_profile(summary, sidecar):
    rp = dict(summary.get("retrieval_params", {}) or {})
    if not rp:
        rp = dict(sidecar.get("retrieval_params", {}) or {})

    gp = dict(summary.get("generation_params", {}) or {})
    if not gp:
        gp = dict(sidecar.get("generation_params", {}) or {})

    return {
        "generator": str(summary.get("generator", sidecar.get("generator", "vllm")) or "vllm"),
        "evaluator_mode": str(summary.get("evaluator_mode", sidecar.get("evaluator_mode", "hipporag2_parity")) or "hipporag2_parity"),
        "global_corpus_path": str(rp.get("global_corpus_path", "") or ""),
        "openie_mode": str(rp.get("openie_mode", "llm") or "llm"),
        "openie_model_name": str(rp.get("openie_model_name", "") or ""),
        "openie_local_files_only": bool(rp.get("openie_local_files_only", True)),
        "llm_max_new_tokens": int(_safe_float(gp.get("llm_max_new_tokens", 64), 64)),
        "model_name": str(gp.get("model_name", "") or ""),
    }


def _ref_item(label, path):
    summary = _load_json(path)
    sidecar, sidecar_path = _read_sidecar(path)
    return {
        "label": str(label),
        "path": str(Path(path).resolve()),
        "run_dir": str(Path(path).resolve().parent),
        "sidecar_config_path": sidecar_path,
        "f1": _metric(summary, "f1", "F1"),
        "em": _metric(summary, "em", "EM"),
        "r1": _r_at(summary, 1),
        "r5": _r_at(summary, 5),
        "r20": _r_at(summary, 20),
        "retrieval_ms": _safe_float(summary.get("retrieval_latency_ms", summary.get("retrieval_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_latency_ms", summary.get("generation_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_latency_ms", summary.get("total_ms", 0.0)), 0.0),
        "summary": summary,
        "sidecar": sidecar,
    }


def _fixed_hipporag2_item(dataset):
    dataset = str(dataset or "").strip().lower()
    table = {
        "hotpotqa": {
            "r1": 0.4380,
            "r5": 0.9500,
            "r20": 0.9890,
            "em": 0.3350,
            "f1": 0.5154,
            "retrieval_s": 207.7214,
            "generation_s": 0.3412,
            "total_s": 208.0626,
            "timestamp_kst": "2026-03-23T21:37:21+09:00",
        },
        "2wikimultihopqa": {
            "r1": 0.4088,
            "r5": 0.8955,
            "r20": 0.9560,
            "em": 0.2670,
            "f1": 0.4499,
            "retrieval_s": 873.6446,
            "generation_s": 2003.5872,
            "total_s": 2877.2318,
            "timestamp_kst": "2026-03-24T00:39:02+09:00",
        },
    }
    item = table.get(dataset)
    if item is None:
        raise RuntimeError(f"unsupported_fixed_hipporag2_dataset:{dataset}")
    return {
        "label": "HippoRAG2",
        "path": f"fixed_table://hipporag2/{dataset}/2026-03-24",
        "run_dir": "",
        "sidecar_config_path": "",
        "f1": _safe_float(item.get("f1", 0.0), 0.0),
        "em": _safe_float(item.get("em", 0.0), 0.0),
        "r1": _safe_float(item.get("r1", 0.0), 0.0),
        "r5": _safe_float(item.get("r5", 0.0), 0.0),
        "r20": _safe_float(item.get("r20", 0.0), 0.0),
        "retrieval_ms": _safe_float(item.get("retrieval_s", 0.0), 0.0) * 1000.0,
        "generation_ms": _safe_float(item.get("generation_s", 0.0), 0.0) * 1000.0,
        "total_ms": _safe_float(item.get("total_s", 0.0), 0.0) * 1000.0,
        "timestamp_kst": str(item.get("timestamp_kst", "")),
    }


rag_cfg = yaml.safe_load(Path(rag_cfg_path).read_text(encoding="utf-8")) or {}
retr_cfg = yaml.safe_load(Path(retr_cfg_path).read_text(encoding="utf-8")) or {}

hotpot_d1 = _ref_item("D1", hotpot_d1_path)
hotpot_f2 = _ref_item("F2", hotpot_f2_path)
wiki_d1 = _ref_item("D1", wiki_d1_path)
wiki_e1 = _ref_item("E1", wiki_e1_path)

wiki_w3_path_resolved = ""
if str(wiki_w3_path or "").strip() and Path(wiki_w3_path).exists():
    wiki_w3_path_resolved = str(Path(wiki_w3_path).resolve())
else:
    cands = sorted(Path("outputs").glob("**/2wikimultihopqa/W3/rag/2wikimultihopqa/*/rag_summary.json"))
    if cands:
        wiki_w3_path_resolved = str(cands[-1].resolve())

wiki_w3 = _ref_item("W3", wiki_w3_path_resolved) if wiki_w3_path_resolved else None
wiki_best = wiki_w3 if wiki_w3 is not None else wiki_e1
wiki_best_label = "w3" if wiki_w3 is not None else "e1"

d2_hotpot = _ref_item("D2", d2_hotpot_path)
d2_wiki = _ref_item("D2", d2_wiki_path)
if _as_bool(hippo_use_fixed_table):
    hippo_hotpot = _fixed_hipporag2_item("hotpotqa")
    hippo_wiki = _fixed_hipporag2_item("2wikimultihopqa")
else:
    hippo_hotpot = _ref_item("HippoRAG2", hippo_hotpot_path)
    hippo_wiki = _ref_item("HippoRAG2", hippo_wiki_path)

payload = {
    "hotpotqa": {
        "d1": {k: v for k, v in hotpot_d1.items() if k not in {"summary", "sidecar"}},
        "best_non_oracle": {k: v for k, v in hotpot_f2.items() if k not in {"summary", "sidecar"}},
        "d2": {k: v for k, v in d2_hotpot.items() if k not in {"summary", "sidecar"}},
        "hipporag2": {k: v for k, v in hippo_hotpot.items() if k not in {"summary", "sidecar"}},
    },
    "2wikimultihopqa": {
        "d1": {k: v for k, v in wiki_d1.items() if k not in {"summary", "sidecar"}},
        "best_non_oracle": {k: v for k, v in wiki_best.items() if k not in {"summary", "sidecar"}},
        "d2": {k: v for k, v in d2_wiki.items() if k not in {"summary", "sidecar"}},
        "hipporag2": {k: v for k, v in hippo_wiki.items() if k not in {"summary", "sidecar"}},
    },
    "wiki_best_label": wiki_best_label,
    "frozen_profiles": {
        "hotpot_d1": {
            "summary_path": hotpot_d1["path"],
            "retrieval_knobs": _extract_retrieval_knobs(hotpot_d1["summary"], hotpot_d1["sidecar"], retr_cfg),
            "render_profile": _extract_render_profile(hotpot_d1["summary"], hotpot_d1["sidecar"], rag_cfg),
            "run_profile": _extract_run_profile(hotpot_d1["summary"], hotpot_d1["sidecar"]),
        },
        "hotpot_f2": {
            "summary_path": hotpot_f2["path"],
            "retrieval_knobs": _extract_retrieval_knobs(hotpot_f2["summary"], hotpot_f2["sidecar"], retr_cfg),
            "render_profile": _extract_render_profile(hotpot_f2["summary"], hotpot_f2["sidecar"], rag_cfg),
            "run_profile": _extract_run_profile(hotpot_f2["summary"], hotpot_f2["sidecar"]),
        },
        "wiki_d1": {
            "summary_path": wiki_d1["path"],
            "retrieval_knobs": _extract_retrieval_knobs(wiki_d1["summary"], wiki_d1["sidecar"], retr_cfg),
            "render_profile": _extract_render_profile(wiki_d1["summary"], wiki_d1["sidecar"], rag_cfg),
            "run_profile": _extract_run_profile(wiki_d1["summary"], wiki_d1["sidecar"]),
        },
        "wiki_e1": {
            "summary_path": wiki_e1["path"],
            "retrieval_knobs": _extract_retrieval_knobs(wiki_e1["summary"], wiki_e1["sidecar"], retr_cfg),
            "render_profile": _extract_render_profile(wiki_e1["summary"], wiki_e1["sidecar"], rag_cfg),
            "run_profile": _extract_run_profile(wiki_e1["summary"], wiki_e1["sidecar"]),
        },
        "wiki_best": {
            "summary_path": wiki_best["path"],
            "retrieval_knobs": _extract_retrieval_knobs(wiki_best["summary"], wiki_best["sidecar"], retr_cfg),
            "render_profile": _extract_render_profile(wiki_best["summary"], wiki_best["sidecar"], rag_cfg),
            "run_profile": _extract_run_profile(wiki_best["summary"], wiki_best["sidecar"]),
        },
    },
    "artifact_trace": {
        "round_metrics_ref": str(Path(round_metrics_ref).resolve()) if str(round_metrics_ref or "").strip() and Path(round_metrics_ref).exists() else "",
        "run_records_ref": str(Path(run_records_ref).resolve()) if str(run_records_ref or "").strip() and Path(run_records_ref).exists() else "",
        "current_proposal_round_metrics": (
            str(Path(current_proposal_round_metrics).resolve())
            if str(current_proposal_round_metrics or "").strip() and Path(current_proposal_round_metrics).exists()
            else ""
        ),
        "round_metrics_ref_record_count": 0,
        "run_records_ref_row_count": 0,
    },
    "execution_defaults": {
        "embedding_max_length": int(retr_cfg.get("embedding_max_length", 320) or 320),
        "embedding_text_max_chars": int(retr_cfg.get("embedding_text_max_chars", 900) or 900),
    },
    "embedding_regimes": {
        "old": {
            "embedding_max_length": int(
                (_extract_retrieval_knobs(hotpot_f2["summary"], hotpot_f2["sidecar"], retr_cfg).get("embedding_max_length", 192))
                or 192
            ),
            "embedding_text_max_chars": int(
                (_extract_retrieval_knobs(hotpot_f2["summary"], hotpot_f2["sidecar"], retr_cfg).get("embedding_text_max_chars", 600))
                or 600
            ),
        },
        "new": {
            "embedding_max_length": int(retr_cfg.get("embedding_max_length", 320) or 320),
            "embedding_text_max_chars": int(retr_cfg.get("embedding_text_max_chars", 900) or 900),
        },
    },
    "calibration_defaults": {
        "delta": 2,
        "reserve_delta": 2,
    },
    "current_proposal_round_best": {},
}

if wiki_w3 is not None:
    payload["frozen_profiles"]["wiki_w3"] = {
        "summary_path": wiki_w3["path"],
        "retrieval_knobs": _extract_retrieval_knobs(wiki_w3["summary"], wiki_w3["sidecar"], retr_cfg),
        "render_profile": _extract_render_profile(wiki_w3["summary"], wiki_w3["sidecar"], rag_cfg),
        "run_profile": _extract_run_profile(wiki_w3["summary"], wiki_w3["sidecar"]),
    }

rm_path = payload["artifact_trace"]["round_metrics_ref"]
if rm_path:
    try:
        rm = json.loads(Path(rm_path).read_text(encoding="utf-8"))
        payload["artifact_trace"]["round_metrics_ref_record_count"] = int(len(rm.get("records", []) or []))
    except Exception:
        pass

rr_path = payload["artifact_trace"]["run_records_ref"]
if rr_path:
    try:
        with Path(rr_path).open("r", encoding="utf-8") as f:
            cnt = sum(1 for _ in f)
        payload["artifact_trace"]["run_records_ref_row_count"] = int(max(0, cnt - 1))
    except Exception:
        pass

pr_path = payload["artifact_trace"]["current_proposal_round_metrics"]
if pr_path:
    try:
        prm = json.loads(Path(pr_path).read_text(encoding="utf-8"))
        recs = [r for r in (prm.get("records", []) or []) if str(r.get("stage", "")) != "reference"]
        for ds in ("hotpotqa", "2wikimultihopqa"):
            ds_rows = [r for r in recs if str(r.get("dataset", "")) == ds]
            if not ds_rows:
                continue
            best = sorted(
                ds_rows,
                key=lambda x: (_safe_float(x.get("F1", x.get("f1", 0.0)), 0.0), _safe_float(x.get("Recall@5", 0.0), 0.0)),
                reverse=True,
            )[0]
            payload["current_proposal_round_best"][ds] = {
                "stage": str(best.get("stage", "")),
                "variant": str(best.get("variant", "")),
                "summary_path": str(best.get("summary_path", "")),
                "f1": _safe_float(best.get("F1", best.get("f1", 0.0)), 0.0),
                "em": _safe_float(best.get("EM", best.get("em", 0.0)), 0.0),
                "r1": _safe_float(best.get("Recall@1", 0.0), 0.0),
                "r5": _safe_float(best.get("Recall@5", 0.0), 0.0),
                "r20": _safe_float(best.get("Recall@20", 0.0), 0.0),
            }
    except Exception:
        pass

out = Path(out_json)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(str(out))
PY
}

bundle_has_profile() {
  local profile_key="$1"
  "${PYTHON_BIN}" - "${REF_BUNDLE_JSON}" "${profile_key}" <<'PY'
import json,sys
obj=json.loads(open(sys.argv[1], 'r', encoding='utf-8').read())
print('1' if sys.argv[2] in (obj.get('frozen_profiles', {}) or {}) else '0')
PY
}

get_profile_field() {
  local profile_key="$1"
  local group="$2"
  local field="$3"
  "${PYTHON_BIN}" - "${REF_BUNDLE_JSON}" "${profile_key}" "${group}" "${field}" <<'PY'
import json,sys
obj=json.loads(open(sys.argv[1], 'r', encoding='utf-8').read())
pk,group,field=sys.argv[2:5]
val=(((obj.get('frozen_profiles', {}) or {}).get(pk, {}) or {}).get(group, {}) or {}).get(field, "")
if val is None:
    val=""
print(str(val))
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
        first = f.readline()
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

annotate_summary_with_refs() {
  local summary_path="$1"
  local dataset="$2"
  "${PYTHON_BIN}" - "${summary_path}" "${dataset}" "${REF_BUNDLE_JSON}" <<'PY'
import json
import sys
from pathlib import Path
from effirag.utils import compute_rag_derived_metrics

summary_path = Path(sys.argv[1])
dataset = sys.argv[2]
refs = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))

d2_f1 = float((((refs.get(dataset, {}) or {}).get("d2", {}) or {}).get("f1", 0.0) or 0.0))
hippo_f1 = float((((refs.get(dataset, {}) or {}).get("hipporag2", {}) or {}).get("f1", 0.0) or 0.0))
hippo_r5 = float((((refs.get(dataset, {}) or {}).get("hipporag2", {}) or {}).get("r5", 0.0) or 0.0))

summary["oracle_ceiling_f1"] = d2_f1
summary["hipporag2_reference_f1"] = hippo_f1
summary["hipporag2_reference_recall_at_5"] = hippo_r5
summary.update(compute_rag_derived_metrics(summary))
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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

create_variant_cfg() {
  local base_cfg="$1"
  local out_cfg="$2"
  local dataset="$3"
  local variant_tag="$4"
  local profile_key="$5"
  local embedding_regime="${6:-mid}"
  local retrieval_mode="${7:-baseline}"
  local downstream_mode="${8:-inherit}"
  local reorder_topk="${9:-3}"
  local pinning_min="${10:-1}"

  mkdir -p "$(dirname "${out_cfg}")"
  "${PYTHON_BIN}" - \
    "${base_cfg}" \
    "${out_cfg}" \
    "${dataset}" \
    "${variant_tag}" \
    "${profile_key}" \
    "${embedding_regime}" \
    "${retrieval_mode}" \
    "${downstream_mode}" \
    "${reorder_topk}" \
    "${pinning_min}" \
    "${REF_BUNDLE_JSON}" \
    "${UNIVERSAL_EMBED_MAX_LEN}" \
    "${UNIVERSAL_EMBED_MAX_CHARS}" \
    "${UNIVERSAL_EMBED_MAX_LEN}" \
    "${UNIVERSAL_EMBED_MAX_CHARS}" \
    "${UNIVERSAL_EMBED_MAX_LEN}" \
    "${UNIVERSAL_EMBED_MAX_CHARS}" <<'PY'
import json
import sys
import yaml

(
  base_cfg,
  out_cfg,
  dataset,
  variant_tag,
  profile_key,
  embedding_regime,
  retrieval_mode,
  downstream_mode,
  reorder_topk,
  pinning_min,
  bundle_json,
  old_len,
  old_chars,
  mid_len,
  mid_chars,
  new_len,
  new_chars,
) = sys.argv[1:18]

cfg = yaml.safe_load(open(base_cfg, 'r', encoding='utf-8')) or {}
bundle = json.loads(open(bundle_json, 'r', encoding='utf-8').read())
profiles = bundle.get('frozen_profiles', {}) or {}
if profile_key not in profiles:
    raise RuntimeError(f'profile_not_found:{profile_key}')

prof = profiles[profile_key]
knobs = dict(prof.get('retrieval_knobs', {}) or {})
render = dict(prof.get('render_profile', {}) or {})

for k,v in knobs.items():
    cfg[k] = v
for k,v in render.items():
    cfg[k] = v

regime = str(embedding_regime or 'mid').strip().lower()
if regime == 'old':
    cfg['embedding_text_max_chars'] = int(old_chars)
    cfg['embedding_max_length'] = int(old_len)
elif regime == 'mid':
    cfg['embedding_text_max_chars'] = int(mid_chars)
    cfg['embedding_max_length'] = int(mid_len)
elif regime == 'new':
    cfg['embedding_text_max_chars'] = int(new_chars)
    cfg['embedding_max_length'] = int(new_len)
else:
    raise RuntimeError(f'unsupported_embedding_regime:{embedding_regime}')

cfg['dataset'] = str(dataset)
cfg['canonical_variant_name'] = str(variant_tag)
# Keep prior proposal/exploration experiments fully disabled in this round.
cfg['shared_budget_profile'] = 'off'
cfg['shared_budget_proposal_step'] = 0
cfg['shared_budget_exploration_step'] = 0
cfg['proposal_union_experiment_mode'] = 'off'

mode = str(retrieval_mode or 'baseline').strip().lower()
allowed_modes = {
    'baseline',
    'r2_bridge_only',
    'r2_connector_core',
    'r2_plus_path_preserve',
}
if mode not in allowed_modes:
    raise RuntimeError(f'unsupported_retrieval_mode:{retrieval_mode}')

cfg['retrieval_objective_mode'] = mode
cfg['hybrid_anchor_recall_enabled'] = False
cfg['bridge_candidate_induction_enabled'] = False
cfg['role_aware_chunk_scoring_enabled'] = False
cfg['coverage_selection_enabled'] = False

# Keep redesign round isolated from older run-level objective tuning knobs.
cfg['run_score_pair_coverage_weight'] = 0.0
cfg['run_score_bridge_completeness_weight'] = 0.0
cfg['run_score_entity_chunk_grounding_weight'] = 0.0
cfg['run_score_anchor_dispersion_penalty'] = 0.0

tag = str(variant_tag or '').strip().lower()
if tag in {'r2_bridge_only', 'bridge_candidate_induction'}:
    cfg['bridge_candidate_induction_enabled'] = True
    cfg['role_aware_chunk_scoring_enabled'] = False
    cfg['run_score_semantic_weight'] = 0.24
    cfg['run_score_anchor_weight'] = 0.20
    cfg['run_score_structure_weight'] = 0.24
    cfg['run_score_bridge_weight'] = 0.18
    cfg['run_score_redundancy_weight'] = 0.10
    cfg['run_score_pair_coverage_weight'] = 0.02
    cfg['run_score_bridge_completeness_weight'] = 0.02
    cfg['seed_objective_bridge_weight'] = 0.24
    cfg['seed_objective_anchor_coverage_weight'] = 0.14
elif tag == 'r2_plus_r3_anchor_light':
    cfg['bridge_candidate_induction_enabled'] = True
    cfg['role_aware_chunk_scoring_enabled'] = True
    cfg['run_score_semantic_weight'] = 0.24
    cfg['run_score_anchor_weight'] = 0.20
    cfg['run_score_structure_weight'] = 0.24
    cfg['run_score_bridge_weight'] = 0.18
    cfg['run_score_redundancy_weight'] = 0.10
    cfg['run_score_pair_coverage_weight'] = 0.02
    cfg['run_score_bridge_completeness_weight'] = 0.02
    cfg['seed_objective_bridge_weight'] = 0.24
    cfg['seed_objective_anchor_coverage_weight'] = 0.14
    cfg['role_chunk_weight_anchor_match'] = 0.40
    cfg['role_chunk_weight_bridge_support'] = 0.30
    cfg['role_chunk_weight_answer_likelihood'] = 0.18
    cfg['role_chunk_weight_path_consistency'] = 0.12
elif tag == 'r2_plus_r3_answer_light':
    cfg['bridge_candidate_induction_enabled'] = True
    cfg['role_aware_chunk_scoring_enabled'] = True
    cfg['run_score_semantic_weight'] = 0.24
    cfg['run_score_anchor_weight'] = 0.20
    cfg['run_score_structure_weight'] = 0.24
    cfg['run_score_bridge_weight'] = 0.18
    cfg['run_score_redundancy_weight'] = 0.10
    cfg['run_score_pair_coverage_weight'] = 0.02
    cfg['run_score_bridge_completeness_weight'] = 0.02
    cfg['seed_objective_bridge_weight'] = 0.24
    cfg['seed_objective_anchor_coverage_weight'] = 0.14
    cfg['role_chunk_weight_anchor_match'] = 0.24
    cfg['role_chunk_weight_bridge_support'] = 0.28
    cfg['role_chunk_weight_answer_likelihood'] = 0.34
    cfg['role_chunk_weight_path_consistency'] = 0.14
elif tag in {
    'r2_plus_path_preserve_reconfirm',
    'r2_plus_path_preserve_chain_summary',
    'r2_plus_path_preserve_derivation_prompt',
    'r2_plus_path_preserve_chain_summary_plus_derivation',
    'r2_plus_path_preserve_top1_path_focus',
}:
    cfg['retrieval_objective_mode'] = 'r2_plus_path_preserve'
    if tag == 'r2_plus_path_preserve_reconfirm':
        pass
    elif tag == 'r2_plus_path_preserve_chain_summary':
        cfg['render_mode'] = 'path_bundle'
        cfg['order_strategy'] = 'path_bundle_ordered+path_bundle_chain_summary'
        cfg['chunk_grounding_enabled'] = False
    elif tag == 'r2_plus_path_preserve_derivation_prompt':
        cfg['render_mode'] = 'path_bundle'
        cfg['order_strategy'] = 'path_bundle_ordered+path_bundle_derivation_prompt'
        cfg['chunk_grounding_enabled'] = False
    elif tag == 'r2_plus_path_preserve_chain_summary_plus_derivation':
        cfg['render_mode'] = 'path_bundle'
        cfg['order_strategy'] = 'path_bundle_ordered+path_bundle_chain_summary+path_bundle_derivation_prompt'
        cfg['chunk_grounding_enabled'] = False
    elif tag == 'r2_plus_path_preserve_top1_path_focus':
        cfg['render_mode'] = 'path_bundle'
        cfg['order_strategy'] = (
            'path_bundle_ordered+path_bundle_chain_summary+path_bundle_derivation_prompt+path_bundle_top1_focus'
        )
        cfg['chunk_grounding_enabled'] = False

mode = str(downstream_mode or 'inherit').strip().lower()
if mode in {'reorder_light', 'wiki_fixed'}:
    cfg['final_top_slice_reorder_enabled'] = True
    cfg['final_top_slice_reorder_topk'] = max(1, int(reorder_topk))
    cfg['answer_support_pinning_enabled'] = False
elif mode in {'pin_stable', 'hotpot_fixed'}:
    cfg['answer_support_pinning_enabled'] = True
    cfg['answer_support_pinning_min'] = max(1, int(pinning_min))
    cfg['final_top_slice_reorder_enabled'] = False
elif mode == 'off':
    cfg['final_top_slice_reorder_enabled'] = False
    cfg['answer_support_pinning_enabled'] = False
# inherit => keep frozen profile downstream settings as-is

yaml.safe_dump(cfg, open(out_cfg, 'w', encoding='utf-8'), sort_keys=False, allow_unicode=False)
PY

  assert_cfg_race_safe "${out_cfg}" >/dev/null
}

run_prompt_sanity() {
  local dataset="$1"
  local profile_key="$2"
  local out_json="$3"
  local out_md="$4"
  local max_model_len="$5"
  local sample_limit="$6"

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
      log_msg "[SKIP] prompt_sanity dataset=${dataset} (complete preflight found)"
      return 0
    fi
  fi

  "${PYTHON_BIN}" - \
    "${dataset}" \
    "${profile_key}" \
    "${REF_BUNDLE_JSON}" \
    "${out_json}" \
    "${out_md}" \
    "${max_model_len}" \
    "${sample_limit}" \
    "${MODEL_NAME}" <<'PY'
import inspect
import json
import math
import sys
from pathlib import Path

from effirag.registry import get_dataset_loader, register_defaults
from effirag.render import render_context
from effirag.run_rag import _reconstruct_retrieval


def _safe_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)


def _safe_int(v, d=0):
    try:
        return int(v)
    except Exception:
        return int(d)


def _percentile(values, q):
    vals = sorted([float(v) for v in values])
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _build_prompt(q, c):
    return (
        "You are a QA assistant.\n"
        "Use only the provided context.\n"
        "Return only the final answer span.\n"
        "Do not output reasoning, explanations, or <think> tags.\n"
        "If the question is yes/no, output exactly yes or no.\n"
        f"Question: {q}\n"
        f"Context:\n{c}\n"
        "Final answer:"
    )


def _render_from_params(sample, retrieval, render_params, render_mode):
    requested = {
        "sample": sample,
        "retrieval_result": retrieval,
        "max_context_sentences": _safe_int(render_params.get("max_context_sentences", 8), 8),
        "render_mode": str(render_mode or "flat"),
        "max_corridors_in_context": _safe_int(render_params.get("max_corridors_in_context", 3), 3),
        "max_main_sentences_per_corridor": _safe_int(render_params.get("max_main_sentences_per_corridor", 2), 2),
        "max_support_per_corridor": _safe_int(render_params.get("max_support_per_corridor", 1), 1),
        "max_total_sentences": render_params.get("max_total_sentences", 8),
        "alpha": _safe_float(render_params.get("alpha", 0.28), 0.28),
        "beta": _safe_float(render_params.get("beta", 0.22), 0.22),
        "gamma_main": _safe_float(render_params.get("gamma_main", 0.26), 0.26),
        "delta_support": _safe_float(render_params.get("delta_support", 0.14), 0.14),
        "eta_connector": _safe_float(render_params.get("eta_connector", 0.12), 0.12),
        "zeta_query": _safe_float(render_params.get("zeta_query", 0.12), 0.12),
        "xi_locality": _safe_float(render_params.get("xi_locality", 0.08), 0.08),
        "lambda_redundancy": _safe_float(render_params.get("lambda_redundancy", 0.25), 0.25),
        "top_corridors": _safe_int(render_params.get("top_corridors", 3), 3),
        "max_sentences": _safe_int(render_params.get("max_sentences", 10), 10),
        "reserve_top_corridor": bool(render_params.get("reserve_top_corridor", False)),
        "order_strategy": str(render_params.get("order_strategy", "score") or "score"),
        "chunk_grounding_enabled": bool(render_params.get("chunk_grounding_enabled", False)),
        "chunk_grounding_mode": str(render_params.get("chunk_grounding_mode", "sentence_backfill") or "sentence_backfill"),
        "chunk_excerpt_max_per_corridor": _safe_int(render_params.get("chunk_excerpt_max_per_corridor", 1), 1),
        "chunk_excerpt_window_sentences_before": _safe_int(render_params.get("chunk_excerpt_window_sentences_before", 1), 1),
        "chunk_excerpt_window_sentences_after": _safe_int(render_params.get("chunk_excerpt_window_sentences_after", 1), 1),
        "chunk_excerpt_max_total_sentences": _safe_int(render_params.get("chunk_excerpt_max_total_sentences", 4), 4),
        "chunk_excerpt_dedup_enabled": bool(render_params.get("chunk_excerpt_dedup_enabled", True)),
        "chunk_grounding_top_corridor_chunks": _safe_int(render_params.get("chunk_grounding_top_corridor_chunks", 2), 2),
        "chunk_grounding_top_k_packages": _safe_int(render_params.get("chunk_grounding_top_k_packages", 4), 4),
        "package_score_answer_weight": _safe_float(render_params.get("package_score_answer_weight", 0.5), 0.5),
        "package_score_bridge_weight": _safe_float(render_params.get("package_score_bridge_weight", 0.2), 0.2),
        "package_score_support_weight": _safe_float(render_params.get("package_score_support_weight", 0.15), 0.15),
        "package_score_chunk_grounding_weight": _safe_float(render_params.get("package_score_chunk_grounding_weight", 0.15), 0.15),
        "package_score_redundancy_weight": _safe_float(render_params.get("package_score_redundancy_weight", 0.10), 0.10),
    }
    sig = inspect.signature(render_context)
    allowed = set(sig.parameters.keys())
    kwargs = {k: v for k, v in requested.items() if k in allowed}
    return render_context(**kwargs)


def _tok_count(tokenizer, text):
    if tokenizer is None:
        return max(1, int(len(text) / 4))
    try:
        enc = tokenizer(text, add_special_tokens=True, truncation=False, return_attention_mask=False)
        ids = []
        if isinstance(enc, dict):
            ids = enc.get("input_ids", []) or []
        elif hasattr(enc, "input_ids"):
            ids = getattr(enc, "input_ids", []) or []
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return int(len(ids[0]))
        return int(len(ids)) if ids else max(1, int(len(text) / 4))
    except Exception:
        return max(1, int(len(text) / 4))


dataset, profile_key, ref_bundle_path, out_json, out_md, max_model_len, sample_limit, model_name = sys.argv[1:9]
max_model_len = int(max_model_len)
sample_limit = int(sample_limit)

bundle = json.loads(Path(ref_bundle_path).read_text(encoding='utf-8'))
summary_path = str((((bundle.get('frozen_profiles', {}) or {}).get(profile_key, {}) or {}).get('summary_path', '')))
if not summary_path:
    raise RuntimeError(f'profile_summary_missing:{profile_key}')

summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
query_path = Path(summary_path).with_name('rag_query_results.jsonl')
if not query_path.exists():
    raise RuntimeError(f'missing_query_results:{query_path}')

rows = []
with query_path.open('r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

register_defaults()
loader = get_dataset_loader(str(dataset))
samples = loader(split='validation', limit=None, data_path=f'data/qa/{dataset}.json')
by_qid = {str(s.qid): s for s in samples}

render_params = dict(summary.get('render_params', {}) or {})
render_mode = str(summary.get('render_mode_resolved') or summary.get('render_mode_requested') or render_params.get('render_mode', 'flat'))
max_new_tokens = int(_safe_float((summary.get('generation_params', {}) or {}).get('llm_max_new_tokens', 64), 64))

# tokenizer (best effort local)
tokenizer = None
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_name), trust_remote_code=True, local_files_only=True)
except Exception:
    tokenizer = None

prompt_lens = []
over = []
seen = 0
render_fail = 0
for row in rows:
    if seen >= sample_limit:
        break
    qid = str(row.get('sample_id', '') or '')
    sample = by_qid.get(qid)
    if sample is None:
        continue
    retrieval_payload = dict(row.get('retrieval', {}) or {})
    if not retrieval_payload:
        continue
    try:
        retrieval = _reconstruct_retrieval(retrieval_payload)
        rendered = _render_from_params(sample=sample, retrieval=retrieval, render_params=render_params, render_mode=render_mode)
    except Exception:
        render_fail += 1
        continue

    prompt = _build_prompt(sample.question, rendered.text)
    plen = _tok_count(tokenizer, prompt)
    prompt_lens.append(int(plen))
    if int(plen + max_new_tokens) > max_model_len:
        over.append({
            'sample_id': qid,
            'prompt_tokens': int(plen),
            'max_new_tokens': int(max_new_tokens),
            'max_model_len': int(max_model_len),
            'over_by': int(plen + max_new_tokens - max_model_len),
        })
    seen += 1

p95 = _percentile(prompt_lens, 0.95)
p99 = _percentile(prompt_lens, 0.99)
pmax = max(prompt_lens) if prompt_lens else 0
over_rate = float(len(over) / seen) if seen > 0 else 0.0

payload = {
    'dataset': str(dataset),
    'profile_key': str(profile_key),
    'reference_summary_path': summary_path,
    'reference_query_results_path': str(query_path),
    'sample_limit': int(sample_limit),
    'sample_count': int(seen),
    'render_fail_count': int(render_fail),
    'max_model_len': int(max_model_len),
    'max_new_tokens': int(max_new_tokens),
    'prompt_token_p95': float(p95),
    'prompt_token_p99': float(p99),
    'prompt_token_max': float(pmax),
    'prompt_over_limit_rate': float(over_rate),
    'over_limit_examples': over[:20],
    'pass': bool(seen > 0 and over_rate <= 1e-12),
}

Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
lines = [
    f"# Prompt Sanity ({dataset})",
    '',
    f"- sample_count: {seen}",
    f"- render_fail_count: {render_fail}",
    f"- prompt_over_limit_rate: {over_rate:.6f}",
    f"- prompt_p99 + max_new_tokens: {p99:.2f} + {max_new_tokens} = {p99 + max_new_tokens:.2f}",
    f"- max_model_len: {max_model_len}",
    f"- pass: {payload['pass']}",
]
Path(out_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('PASS' if payload['pass'] else 'FAIL')
raise SystemExit(0 if payload['pass'] else 5)
PY
}

run_rag_variant() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local profile_key="$4"
  local cfg_path="$5"
  local limit="$6"

  local out_dir="${RUN_ROOT}/${stage}/${variant}/rag"
  local log_file="${LOG_ROOT}/${stage}__${variant}__${dataset}.log"
  mkdir -p "${out_dir}" "$(dirname "${log_file}")"

  if [[ "${SKIP_COMPLETED}" == "true" ]]; then
    local existing_summary
    existing_summary="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
    if [[ -n "${existing_summary}" ]]; then
      if is_complete_summary "${existing_summary}" "${limit}" "rag_query_results.jsonl" >/dev/null 2>&1; then
        annotate_summary_with_refs "${existing_summary}" "${dataset}"
        log_msg "[SKIP] stage=${stage} variant=${variant} dataset=${dataset} (complete summary found)"
        upsert_record "${stage}" "${variant}" "${dataset}" "${profile_key}" "rag" "${existing_summary}" "${log_file}" "skipped"
        return 0
      fi
    fi
  fi

  local run_generator run_evaluator run_global run_openie_mode run_openie_model run_openie_local run_llm_new
  run_generator="$(get_profile_field "${profile_key}" "run_profile" "generator")"
  run_evaluator="$(get_profile_field "${profile_key}" "run_profile" "evaluator_mode")"
  run_global="$(get_profile_field "${profile_key}" "run_profile" "global_corpus_path")"
  run_openie_mode="$(get_profile_field "${profile_key}" "run_profile" "openie_mode")"
  run_openie_model="$(get_profile_field "${profile_key}" "run_profile" "openie_model_name")"
  run_openie_local="$(get_profile_field "${profile_key}" "run_profile" "openie_local_files_only")"
  run_llm_new="$(get_profile_field "${profile_key}" "run_profile" "llm_max_new_tokens")"

  local -a cmd
  cmd=(
    "${PYTHON_BIN}" -m effirag.run_rag
    --config "${cfg_path}"
    --dataset "${dataset}"
    --data-path "data/qa/${dataset}.json"
    --graph-cache-dir "${GRAPH_CACHE_DIR}"
    --force-rebuild-graph-index false
    --num-workers 1
    --openie-mode "${run_openie_mode:-llm}"
    --openie-model-name "${run_openie_model:-${MODEL_NAME}}"
    --openie-local-files-only "${run_openie_local:-true}"
    --openie-api-base-url "${LLM_BASE_URL}"
    --openie-api-key "${LLM_API_KEY}"
    --embedding-enabled true
    --generator "${run_generator:-vllm}"
    --model-name "${MODEL_NAME}"
    --llm-base-url "${LLM_BASE_URL}"
    --llm-api-key "${LLM_API_KEY}"
    --llm-max-new-tokens "${run_llm_new:-64}"
    --run-qa true
    --retrieval-only false
    --evaluator-mode "${run_evaluator:-hipporag2_parity}"
    --limit "${limit}"
    --output-dir "${out_dir}"
    --timestamp-output true
  )
  if [[ -n "${run_global}" ]]; then
    cmd+=(--global-corpus-path "${run_global}")
  fi

  log_msg "[RUN] stage=${stage} variant=${variant} dataset=${dataset} profile=${profile_key} limit=${limit}"
  {
    echo "[COMMAND] ${cmd[*]}"
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
  upsert_record "${stage}" "${variant}" "${dataset}" "${profile_key}" "rag" "${summary_path}" "${log_file}" "ok"
  log_msg "[DONE] stage=${stage} variant=${variant} dataset=${dataset}"
}

build_round_metrics() {
  "${PYTHON_BIN}" - "${RECORD_TSV}" "${REF_BUNDLE_JSON}" "${ROUND_METRICS_JSON}" "${ROUND_METRICS_MD}" <<'PY'
import json
import sys
from pathlib import Path

record_tsv, ref_json, out_json, out_md = sys.argv[1:5]
refs = json.loads(Path(ref_json).read_text(encoding='utf-8'))


def _f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


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


def _row(stage, variant, dataset, summary_path, summary):
    def _safe_div(num, den):
        den = _f(den)
        if den <= 0.0:
            return 0.0
        return _f(num) / den

    def _g(key, default=0.0):
        if key in summary:
            return _f(summary.get(key, default))
        return _f(summary.get(f"stagewise_{key}", default))

    sf = _f(summary.get('supporting_fact_recall', 0.0))
    rsf = _f(summary.get('rendered_supporting_fact_recall', summary.get('rendered_sf_recall', 0.0)))
    em = _f(summary.get('em', summary.get('EM', 0.0)))
    f1 = _f(summary.get('f1', summary.get('F1', 0.0)))
    r1 = _r_at(summary, 1)
    r5 = _r_at(summary, 5)
    r20 = _r_at(summary, 20)

    d1 = ((refs.get(dataset, {}) or {}).get('d1', {}) or {})
    best = ((refs.get(dataset, {}) or {}).get('best_non_oracle', {}) or {})
    d2 = ((refs.get(dataset, {}) or {}).get('d2', {}) or {})
    hippo = ((refs.get(dataset, {}) or {}).get('hipporag2', {}) or {})

    d2_f1 = _f(d2.get('f1', 0.0))
    hippo_f1 = _f(hippo.get('f1', 0.0))
    hippo_r5 = _f(hippo.get('r5', 0.0))
    retrieval_ms = _f(summary.get('retrieval_latency_ms', summary.get('retrieval_ms', 0.0)))
    total_ms = _f(summary.get('total_latency_ms', summary.get('total_ms', 0.0)))
    oracle_gap = (d2_f1 - f1) if d2_f1 > 0 else 0.0

    return {
        'stage': stage,
        'variant': variant,
        'dataset': dataset,
        'summary_path': summary_path,
        'supporting_fact_recall': sf,
        'Recall@1': r1,
        'Recall@5': r5,
        'Recall@20': r20,
        'rendered_supporting_fact_recall': rsf,
        'EM': em,
        'F1': f1,
        'retrieval_ms': retrieval_ms,
        'generation_ms': _f(summary.get('generation_latency_ms', summary.get('generation_ms', 0.0))),
        'total_ms': total_ms,
        'f1_per_100ms': _f(summary.get('f1_per_100ms', _safe_div(f1 * 100.0, total_ms))),
        'r5_per_100ms': _f(summary.get('r5_per_100ms', _safe_div(r5 * 100.0, retrieval_ms))),
        'seed_anchor_recall': _g('seed_anchor_recall'),
        'seed_bridge_recall': _g('seed_bridge_recall'),
        'seed_answer_recall': _g('seed_answer_recall'),
        'run_bridge_coverage': _g('run_bridge_coverage'),
        'corridor_role_coverage': _g('corridor_role_coverage'),
        'bridge_purity': _g('bridge_purity'),
        'answer_preserve_activation_rate': _g('answer_preserve_activation_rate'),
        'preserved_answer_usefulness': _g('preserved_answer_usefulness'),
        'path_complete_rate': _g('path_complete_rate'),
        'bridge_answer_pair_retention': _g('bridge_answer_pair_retention'),
        'incomplete_path_rate': _g('incomplete_path_rate'),
        'chain_compactness': _g('chain_compactness'),
        'path_preserve_activation_rate': _g('path_preserve_activation_rate'),
        'path_bundle_count': _g('path_bundle_count'),
        'avg_chunks_per_bundle': _g('avg_chunks_per_bundle'),
        'bridge_answer_adjacency_rate': _g('bridge_answer_adjacency_rate'),
        'first_complete_path_rank': _g('first_complete_path_rank'),
        'bundle_dedup_ratio': _g('bundle_dedup_ratio'),
        'answer_path_coverage': _g('answer_path_coverage'),
        'summary_token_count': _g('summary_token_count'),
        'path_focus_count': _g('path_focus_count'),
        'derivation_prompt_activation': _g('derivation_prompt_activation'),
        'answer_chain_readability_score': _g('answer_chain_readability_score'),
        'anchor_bridge_balance': _g('anchor_bridge_balance'),
        'answer_side_density': _g('answer_side_density'),
        'support_set_compactness': _g('support_set_compactness'),
        'bridge_noise_ratio': _g('bridge_noise_ratio'),
        'useful_bridge_rate': _g('useful_bridge_rate'),
        'bridge_to_answer_path_hit': _g('bridge_to_answer_path_hit'),
        'conversion_after_bridge': _g('conversion_after_bridge'),
        'conversion_after_preserve': _g('conversion_after_preserve'),
        'conversion_after_path_preserve': _g('conversion_after_path_preserve'),
        'conversion_after_path_bundle': _g('conversion_after_path_bundle'),
        'seed_diversity': _g('seed_diversity'),
        'seed_redundancy': _g('seed_redundancy'),
        'redundancy_rate': _g('redundancy_rate'),
        'hotpot_overpreserve_rate': _g('hotpot_overpreserve_rate'),
        'rendered_retention': (rsf / sf) if sf > 0 else 0.0,
        'answer_conversion': _f(summary.get('answer_conversion', (f1 / rsf) if rsf > 0 else 0.0)),
        'em_conversion': _g('em_conversion'),
        'f1_per_r5': _g('f1_per_r5', (f1 / r5) if r5 > 0 else 0.0),
        'error_bucket_bridge_missing': _g('error_bucket_bridge_missing'),
        'error_bucket_answer_side_missing': _g('error_bucket_answer_side_missing'),
        'error_bucket_bridge_present_noisy': _g('error_bucket_bridge_present_noisy'),
        'error_bucket_answer_present_generation_fail': _g('error_bucket_answer_present_generation_fail'),
        'answer_present_but_generation_fail': _g(
            'answer_present_but_generation_fail',
            _g('error_bucket_answer_present_generation_fail'),
        ),
        'oracle_gap': oracle_gap,
        'oracle_gap_per_ms': _f(summary.get('oracle_gap_per_ms', _safe_div(oracle_gap, total_ms))),
        'gap_to_hippo_r5': (hippo_r5 - r5) if hippo_r5 > 0 else 0.0,
        'gap_to_hippo_f1': (hippo_f1 - f1) if hippo_f1 > 0 else 0.0,
        'delta_R5_vs_D1': r5 - _f(d1.get('r5', 0.0)),
        'delta_EM_vs_D1': em - _f(d1.get('em', 0.0)),
        'delta_F1_vs_D1': f1 - _f(d1.get('f1', 0.0)),
        'delta_R5_vs_best_non_oracle': r5 - _f(best.get('r5', 0.0)),
        'delta_EM_vs_best_non_oracle': em - _f(best.get('em', 0.0)),
        'delta_F1_vs_best_non_oracle': f1 - _f(best.get('f1', 0.0)),
        'delta_R5_vs_baseline': 0.0,
        'delta_EM_vs_baseline': 0.0,
        'delta_F1_vs_baseline': 0.0,
        'delta_retrieval_ms_vs_baseline': 0.0,
        'delta_generation_ms_vs_baseline': 0.0,
        'delta_total_ms_vs_baseline': 0.0,
    }

rows = []
with Path(record_tsv).open('r', encoding='utf-8') as f:
    next(f, None)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) != 8:
            continue
        stage, variant, dataset, profile, kind, summary_path, log_path, status = p
        if kind != 'rag':
            continue
        if status not in {'ok', 'skipped'}:
            continue
        sp = Path(summary_path)
        if not sp.exists():
            continue
        try:
            summary = json.loads(sp.read_text(encoding='utf-8'))
        except Exception:
            continue
        rows.append(_row(stage, variant, dataset, str(sp), summary))

# add references
for ds in ('hotpotqa', '2wikimultihopqa'):
    for key, label in (
        ('d1', 'frozen_D1'),
        ('best_non_oracle', 'frozen_best_non_oracle'),
        ('d2', 'D2_upper_bound'),
        ('hipporag2', 'HippoRAG2'),
    ):
        ref = ((refs.get(ds, {}) or {}).get(key, {}) or {})
        fake = {
            'supporting_fact_recall': 0.0,
            'supporting_fact_recall_at_1': _f(ref.get('r1', 0.0)),
            'supporting_fact_recall_at_5': _f(ref.get('r5', 0.0)),
            'supporting_fact_recall_at_20': _f(ref.get('r20', 0.0)),
            'rendered_supporting_fact_recall': 0.0,
            'em': _f(ref.get('em', 0.0)),
            'f1': _f(ref.get('f1', 0.0)),
            'retrieval_latency_ms': _f(ref.get('retrieval_ms', 0.0)),
            'generation_latency_ms': _f(ref.get('generation_ms', 0.0)),
            'total_latency_ms': _f(ref.get('total_ms', 0.0)),
        }
        rows.append(_row('reference', f"{label}({ref.get('label','')})", ds, str(ref.get('path', '')), fake))

baselines = {}
for ds in ('hotpotqa', '2wikimultihopqa'):
    ds_main = [r for r in rows if r.get('dataset') == ds and str(r.get('stage')) != 'reference']
    cand = [r for r in ds_main if str(r.get('variant')) == 'baseline_mid_reconfirm']
    if cand:
        baselines[ds] = cand[0]

for r in rows:
    if str(r.get('stage')) == 'reference':
        continue
    base = baselines.get(r.get('dataset'))
    if not base:
        continue
    r['delta_R5_vs_baseline'] = _f(r.get('Recall@5')) - _f(base.get('Recall@5'))
    r['delta_EM_vs_baseline'] = _f(r.get('EM')) - _f(base.get('EM'))
    r['delta_F1_vs_baseline'] = _f(r.get('F1')) - _f(base.get('F1'))
    r['delta_retrieval_ms_vs_baseline'] = _f(r.get('retrieval_ms')) - _f(base.get('retrieval_ms'))
    r['delta_generation_ms_vs_baseline'] = _f(r.get('generation_ms')) - _f(base.get('generation_ms'))
    r['delta_total_ms_vs_baseline'] = _f(r.get('total_ms')) - _f(base.get('total_ms'))

pareto_summary = {}
for ds in ('hotpotqa', '2wikimultihopqa'):
    ds_main = [r for r in rows if r.get('dataset') == ds and str(r.get('stage')) != 'reference']
    if not ds_main:
        continue
    base = baselines.get(ds, ds_main[0])
    accuracy_winner = sorted(
        ds_main,
        key=lambda x: (_f(x.get('F1')), _f(x.get('Recall@5')), -_f(x.get('total_ms'))),
        reverse=True,
    )[0]
    efficiency_winner = sorted(
        ds_main,
        key=lambda x: (_f(x.get('f1_per_100ms')), _f(x.get('r5_per_100ms')), -_f(x.get('total_ms'))),
        reverse=True,
    )[0]
    feasible = [
        r
        for r in ds_main
        if _f(r.get('F1')) >= _f(base.get('F1')) and _f(r.get('Recall@5')) >= _f(base.get('Recall@5'))
    ]
    if feasible:
        balanced_winner = sorted(
            feasible,
            key=lambda x: (_f(x.get('f1_per_100ms')), _f(x.get('F1')), _f(x.get('Recall@5')), -_f(x.get('total_ms'))),
            reverse=True,
        )[0]
    else:
        balanced_winner = sorted(
            ds_main,
            key=lambda x: (
                _f(x.get('f1_per_100ms')) + 0.5 * _f(x.get('F1')) + 0.3 * _f(x.get('Recall@5')),
                -_f(x.get('total_ms')),
            ),
            reverse=True,
        )[0]
    pareto_summary[ds] = {
        'accuracy_winner': str(accuracy_winner.get('variant', '')),
        'efficiency_winner': str(efficiency_winner.get('variant', '')),
        'balanced_winner': str(balanced_winner.get('variant', '')),
    }

payload = {
    'wiki_best_label': refs.get('wiki_best_label', 'e1'),
    'records': rows,
    'pareto_summary': pareto_summary,
}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

lines = [
    '# Overnight Path-to-Generation Interface Metrics (RAG full)',
    '',
    '## Main Performance Table',
    '',
    '| stage | variant | dataset | R@5 | EM | F1 | retrieval_ms | generation_ms | total_ms | gap_to_hippo_r5 | oracle_gap |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    lines.append(
        f"| {r['stage']} | {r['variant']} | {r['dataset']} | {r['Recall@5']:.4f} | {r['EM']:.4f} | {r['F1']:.4f} | {r['retrieval_ms']:.2f} | {r['generation_ms']:.2f} | {r['total_ms']:.2f} | {r['gap_to_hippo_r5']:+.4f} | {r['oracle_gap']:+.4f} |"
    )

lines += [
    '',
    '## Efficiency Table',
    '',
    '| stage | variant | dataset | ΔR@5_vs_baseline | ΔEM_vs_baseline | ΔF1_vs_baseline | Δretrieval_ms_vs_baseline | Δgeneration_ms_vs_baseline | Δtotal_ms_vs_baseline | F1_per_100ms | R5_per_100ms | oracle_gap_per_ms |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    if str(r.get('stage')) == 'reference':
        continue
    lines.append(
        f"| {r['stage']} | {r['variant']} | {r['dataset']} | {r['delta_R5_vs_baseline']:+.4f} | {r['delta_EM_vs_baseline']:+.4f} | {r['delta_F1_vs_baseline']:+.4f} | {r['delta_retrieval_ms_vs_baseline']:+.2f} | {r['delta_generation_ms_vs_baseline']:+.2f} | {r['delta_total_ms_vs_baseline']:+.2f} | {r['f1_per_100ms']:.4f} | {r['r5_per_100ms']:.4f} | {r['oracle_gap_per_ms']:.6f} |"
    )

lines += [
    '',
    '## Path-to-Generation Diagnostic Table',
    '',
    '| stage | variant | dataset | path_bundle_count | avg_chunks_per_bundle | bridge_answer_adjacency_rate | first_complete_path_rank | summary_token_count | path_focus_count | derivation_prompt_activation | answer_chain_readability_score | answer_present_but_generation_fail | conversion_after_path_bundle |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    if str(r.get('stage')) == 'reference':
        continue
    lines.append(
        f"| {r['stage']} | {r['variant']} | {r['dataset']} | {r['path_bundle_count']:.4f} | {r['avg_chunks_per_bundle']:.4f} | {r['bridge_answer_adjacency_rate']:.4f} | {r['first_complete_path_rank']:.4f} | {r['summary_token_count']:.4f} | {r['path_focus_count']:.4f} | {r['derivation_prompt_activation']:.4f} | {r['answer_chain_readability_score']:.4f} | {r['answer_present_but_generation_fail']:.4f} | {r['conversion_after_path_bundle']:.4f} |"
    )

lines += [
    '',
    '## Pareto Winners',
    '',
    '| dataset | accuracy_winner | efficiency_winner | balanced_winner |',
    '|---|---|---|---|',
]
for ds in ('hotpotqa', '2wikimultihopqa'):
    item = pareto_summary.get(ds, {}) or {}
    lines.append(
        f"| {ds} | {item.get('accuracy_winner','')} | {item.get('efficiency_winner','')} | {item.get('balanced_winner','')} |"
    )

Path(out_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
}

run_variant_for_dataset() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local profile_key="$4"
  local embedding_regime="$5"
  local retrieval_mode="$6"
  local downstream_mode="$7"
  local reorder_topk="$8"
  local pinning_min="$9"
  local limit="${10}"

  local cfg_dir="${CFG_ROOT}/${stage}/${variant}/${dataset}"
  local rag_cfg="${cfg_dir}/rag.yaml"

  create_variant_cfg \
    "${RAG_BASE_CFG}" \
    "${rag_cfg}" \
    "${dataset}" \
    "${variant}" \
    "${profile_key}" \
    "${embedding_regime}" \
    "${retrieval_mode}" \
    "${downstream_mode}" \
    "${reorder_topk}" \
    "${pinning_min}"

  run_rag_variant "${stage}" "${variant}" "${dataset}" "${profile_key}" "${rag_cfg}" "${limit}"
}

# -----------------------------
# Execution starts here
# -----------------------------
maybe_snapshot_git_state

log_msg "[INFO] round_root=${ROUND_ROOT} (source=${ROUND_ROOT_SOURCE})"

if [[ "${RUN_PREFLIGHT_CHECKS}" == "true" ]]; then
  log_msg "[CHECK] python -m py_compile effirag/*.py"
  "${PYTHON_BIN}" -m py_compile effirag/*.py

  log_msg "[CHECK] PYTHONPATH=. python3 -m effirag.config_audit configs/canonical"
  PYTHONPATH=. "${PYTHON_BIN}" -m effirag.config_audit configs/canonical

  if [[ "${RUN_SMOKE}" == "true" ]]; then
    log_msg "[CHECK] PYTHONPATH=. python3 tests/smoke_entity_chunk.py"
    PYTHONPATH=. "${PYTHON_BIN}" tests/smoke_entity_chunk.py
  fi
fi

build_reference_bundle
resolve_vllm_max_model_len
log_msg "[INFO] vLLM max_model_len=${VLLM_MAX_MODEL_LEN_RESOLVED} source=${VLLM_MAX_MODEL_LEN_SOURCE}"

WIKI_W3_AVAILABLE="$(bundle_has_profile "wiki_w3")"
WIKI_BEST_LABEL="$(${PYTHON_BIN} - "${REF_BUNDLE_JSON}" <<'PY'
import json,sys
print((json.loads(open(sys.argv[1], 'r', encoding='utf-8').read()) or {}).get('wiki_best_label','e1'))
PY
)"

if [[ "${UNIVERSAL_REGIME}" != "mid" ]]; then
  log_msg "[WARN] UNIVERSAL_REGIME=${UNIVERSAL_REGIME} is not allowed in this round. Forcing mid."
  UNIVERSAL_REGIME="mid"
fi
log_msg "[INFO] fixed_universal_regime=${UNIVERSAL_REGIME}, embedding_max_length=${UNIVERSAL_EMBED_MAX_LEN}, embedding_text_max_chars=${UNIVERSAL_EMBED_MAX_CHARS}"

"${PYTHON_BIN}" - "${UNIVERSAL_SELECTION_JSON}" "${UNIVERSAL_REGIME}" "${UNIVERSAL_EMBED_MAX_LEN}" "${UNIVERSAL_EMBED_MAX_CHARS}" <<'PY'
import json,sys
out_path, regime, max_len, max_chars = sys.argv[1:5]
payload = {
    "winner_regime": str(regime),
    "fixed_by": "manual_freeze_for_path_to_generation_interface_round",
    "embedding_max_length": int(max_len),
    "embedding_text_max_chars": int(max_chars),
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(out_path)
PY

log_msg "[Preflight] hotpot_prompt_sanity"
run_prompt_sanity "hotpotqa" "hotpot_f2" "${PREFLIGHT_HOTPOT_JSON}" "${PREFLIGHT_HOTPOT_MD}" "${VLLM_MAX_MODEL_LEN_RESOLVED}" "${PROMPT_SANITY_LIMIT}"

log_msg "[Preflight] wiki_prompt_sanity (profile=${WIKI_BEST_LABEL})"
run_prompt_sanity "2wikimultihopqa" "wiki_best" "${PREFLIGHT_WIKI_JSON}" "${PREFLIGHT_WIKI_MD}" "${VLLM_MAX_MODEL_LEN_RESOLVED}" "${PROMPT_SANITY_LIMIT}"

WIKI_CORE_PROFILE="wiki_best"
if [[ "${WIKI_W3_AVAILABLE}" != "1" ]]; then
  log_msg "[WARN] W3 reference not found. wiki profile will use ${WIKI_CORE_PROFILE} (${WIKI_BEST_LABEL})."
fi

log_msg "[Stage P0] baseline_mid_reconfirm"
run_variant_for_dataset "p0" "baseline_mid_reconfirm" "hotpotqa" "hotpot_f2" "${UNIVERSAL_REGIME}" "baseline" "inherit" 0 0 "${LIMIT_RAG}"
run_variant_for_dataset "p0" "baseline_mid_reconfirm" "2wikimultihopqa" "${WIKI_CORE_PROFILE}" "${UNIVERSAL_REGIME}" "baseline" "inherit" 0 0 "${LIMIT_RAG}"

log_msg "[Stage P1] r2_plus_path_preserve_reconfirm"
run_variant_for_dataset "p1" "r2_plus_path_preserve_reconfirm" "hotpotqa" "hotpot_f2" "${UNIVERSAL_REGIME}" "r2_plus_path_preserve" "inherit" 0 0 "${LIMIT_RAG}"
run_variant_for_dataset "p1" "r2_plus_path_preserve_reconfirm" "2wikimultihopqa" "${WIKI_CORE_PROFILE}" "${UNIVERSAL_REGIME}" "r2_plus_path_preserve" "inherit" 0 0 "${LIMIT_RAG}"

log_msg "[Stage P2] r2_plus_path_preserve_chain_summary"
run_variant_for_dataset "p2" "r2_plus_path_preserve_chain_summary" "hotpotqa" "hotpot_f2" "${UNIVERSAL_REGIME}" "r2_plus_path_preserve" "inherit" 0 0 "${LIMIT_RAG}"
run_variant_for_dataset "p2" "r2_plus_path_preserve_chain_summary" "2wikimultihopqa" "${WIKI_CORE_PROFILE}" "${UNIVERSAL_REGIME}" "r2_plus_path_preserve" "inherit" 0 0 "${LIMIT_RAG}"

log_msg "[Stage P3] r2_plus_path_preserve_derivation_prompt"
run_variant_for_dataset "p3" "r2_plus_path_preserve_derivation_prompt" "hotpotqa" "hotpot_f2" "${UNIVERSAL_REGIME}" "r2_plus_path_preserve" "inherit" 0 0 "${LIMIT_RAG}"
run_variant_for_dataset "p3" "r2_plus_path_preserve_derivation_prompt" "2wikimultihopqa" "${WIKI_CORE_PROFILE}" "${UNIVERSAL_REGIME}" "r2_plus_path_preserve" "inherit" 0 0 "${LIMIT_RAG}"

log_msg "[Stage P4] r2_plus_path_preserve_chain_summary_plus_derivation"
run_variant_for_dataset "p4" "r2_plus_path_preserve_chain_summary_plus_derivation" "hotpotqa" "hotpot_f2" "${UNIVERSAL_REGIME}" "r2_plus_path_preserve" "inherit" 0 0 "${LIMIT_RAG}"
run_variant_for_dataset "p4" "r2_plus_path_preserve_chain_summary_plus_derivation" "2wikimultihopqa" "${WIKI_CORE_PROFILE}" "${UNIVERSAL_REGIME}" "r2_plus_path_preserve" "inherit" 0 0 "${LIMIT_RAG}"

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

build_round_metrics

log_msg "[DONE] outputs: ${ROUND_ROOT}"
log_msg "[DONE] logs: ${LOG_ROOT}"
log_msg "[DONE] run records: ${RECORD_TSV}"
log_msg "[DONE] preflight hotpot: ${PREFLIGHT_HOTPOT_JSON}"
log_msg "[DONE] preflight wiki: ${PREFLIGHT_WIKI_JSON}"
log_msg "[DONE] universal regime freeze: ${UNIVERSAL_SELECTION_JSON}"
log_msg "[DONE] round metrics: ${ROUND_METRICS_JSON}"
