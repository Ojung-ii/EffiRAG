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
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LIMIT_STAGE_A="${LIMIT_STAGE_A:-1000}"
LIMIT_STAGE_B="${LIMIT_STAGE_B:-1000}"
LIMIT_STAGE_C="${LIMIT_STAGE_C:-1000}"
LIMIT_SANITY="${LIMIT_SANITY:-8}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
RUN_SANITY="${RUN_SANITY:-true}"
EMBEDDING_ENABLED="${EMBEDDING_ENABLED:-true}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
ALLOW_CPU_EMBEDDING="${ALLOW_CPU_EMBEDDING:-false}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
RESUME_ROUND_ROOT="${RESUME_ROUND_ROOT:-}"
STOP_ON_STAGE_A_FAIL="${STOP_ON_STAGE_A_FAIL:-true}"
STOP_ON_STAGE_B_FAIL="${STOP_ON_STAGE_B_FAIL:-true}"
REPRO_F1_TOL="${REPRO_F1_TOL:-0.01}"
REPRO_R5_TOL="${REPRO_R5_TOL:-0.02}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

RAG_BASE_CFG="configs/canonical/rag_entity_first_chunk_grounded.yaml"
RETR_BASE_CFG="configs/canonical/retrieval_entity_first_chunk_grounded.yaml"

# Trusted references (override via env if needed).
HOTPOT_D1_REF_SUMMARY="${HOTPOT_D1_REF_SUMMARY:-outputs/exp_retrieval_gap_recheck/hotpotqa/D1/rag/hotpotqa/20260406_053010_574786/rag_summary.json}"
HOTPOT_F2_REF_SUMMARY="${HOTPOT_F2_REF_SUMMARY:-outputs/exp_retrieval_gap/hotpotqa/F2/rag/hotpotqa/20260405_141352_352678/rag_summary.json}"
WIKI_D1_REF_SUMMARY="${WIKI_D1_REF_SUMMARY:-outputs/exp_retrieval_gap_recheck/2wikimultihopqa/D1/rag/2wikimultihopqa/20260406_054816_597964/rag_summary.json}"
WIKI_E1_REF_SUMMARY="${WIKI_E1_REF_SUMMARY:-outputs/exp_retrieval_gap/2wikimultihopqa/E1/rag/2wikimultihopqa/20260405_144256_613687/rag_summary.json}"
WIKI_W3_REF_SUMMARY="${WIKI_W3_REF_SUMMARY:-}"
D2_HOTPOT_REF_SUMMARY="${D2_HOTPOT_REF_SUMMARY:-outputs/exp_retrieval_gap_recheck/hotpotqa/D2/rag/hotpotqa/20260406_053942_373258/rag_summary.json}"
D2_WIKI_REF_SUMMARY="${D2_WIKI_REF_SUMMARY:-outputs/exp_retrieval_gap/2wikimultihopqa/D2/rag/2wikimultihopqa/20260405_143345_219316/rag_summary.json}"
# External HippoRAG2 references.
HIPPORAG2_HOTPOT_SUMMARY="${HIPPORAG2_HOTPOT_SUMMARY:-/home/ojungii/HippoRAG2/outputs/hotpotqa/hotpotqa_eval_summary_results.json}"
HIPPORAG2_WIKI_SUMMARY="${HIPPORAG2_WIKI_SUMMARY:-/home/ojungii/HippoRAG2/outputs/2wikimultihopqa/2wikimultihopqa_eval_summary_results.json}"

CUDA_AVAILABLE="$(${PYTHON_BIN} - <<'PY'
try:
    import torch
    print("1" if bool(torch.cuda.is_available()) else "0")
except Exception:
    print("0")
PY
)"
if [[ "${EMBEDDING_ENABLED}" == "true" && "${CUDA_AVAILABLE}" != "1" && "${ALLOW_CPU_EMBEDDING}" != "true" ]]; then
  echo "[ERROR] embedding_enabled=true but torch CUDA is unavailable in this environment." >&2
  echo "[ERROR] Current run would build semantic embeddings on CPU and become extremely slow." >&2
  echo "[HINT] Use one of the following:" >&2
  echo "  1) Fix CUDA/PyTorch compatibility, then rerun as-is." >&2
  echo "  2) Run fast fallback: EMBEDDING_ENABLED=false bash scripts/run_regime_matched_repro_and_shared_g1.sh" >&2
  echo "  3) Force CPU embedding (very slow): ALLOW_CPU_EMBEDDING=true bash scripts/run_regime_matched_repro_and_shared_g1.sh" >&2
  exit 2
fi

if [[ -n "${RESUME_ROUND_ROOT}" ]]; then
  ROUND_ROOT="${RESUME_ROUND_ROOT}"
  echo "[INFO] resume mode: ROUND_ROOT=${ROUND_ROOT}"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="outputs/regime_matched_repro_shared_g1/${RUN_STAMP}"
fi

CFG_ROOT="${ROUND_ROOT}/configs"
mkdir -p "${CFG_ROOT}"

RECORD_TSV="${ROUND_ROOT}/run_records.tsv"
if [[ ! -f "${RECORD_TSV}" ]]; then
  printf "stage\tvariant\tdataset\tkind\tsummary_path\n" > "${RECORD_TSV}"
fi

REF_METRICS_JSON="${ROUND_ROOT}/reference_metrics.json"

build_reference_metrics() {
  "${PYTHON_BIN}" - \
    "${REF_METRICS_JSON}" \
    "${HOTPOT_D1_REF_SUMMARY}" \
    "${HOTPOT_F2_REF_SUMMARY}" \
    "${WIKI_D1_REF_SUMMARY}" \
    "${WIKI_E1_REF_SUMMARY}" \
    "${WIKI_W3_REF_SUMMARY}" \
    "${D2_HOTPOT_REF_SUMMARY}" \
    "${D2_WIKI_REF_SUMMARY}" \
    "${HIPPORAG2_HOTPOT_SUMMARY}" \
    "${HIPPORAG2_WIKI_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

(
    out_json,
    hotpot_d1_path,
    hotpot_f2_path,
    wiki_d1_path,
    wiki_e1_path,
    wiki_w3_path,
    d2_hotpot_path,
    d2_wiki_path,
    hippo_hotpot_path,
    hippo_wiki_path,
) = sys.argv[1:11]


def _load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def _metric(summary, *keys):
    for key in keys:
        if key in summary and summary[key] is not None:
            try:
                return float(summary[key])
            except Exception:
                continue
    return 0.0


def _r_at(summary, k):
    direct = _metric(summary, f"supporting_fact_recall_at_{k}", f"Recall@{k}", f"recall_at_{k}", f"R@{k}")
    if direct > 0.0:
        return direct
    recall_at_k = summary.get("supporting_fact_recall_at_k", {}) or {}
    if str(k) in recall_at_k:
        return _metric(recall_at_k, str(k))
    if k in recall_at_k:
        return _metric(recall_at_k, k)
    nested = summary.get("recall_at_k", {}) or {}
    if str(k) in nested:
        return _metric(nested, str(k))
    if k in nested:
        return _metric(nested, k)
    return 0.0


def _read_repro_knobs(summary):
    src = dict(summary.get("retrieval_params", {}) or {})
    out = {}
    keys = [
        "semantic_topn_entity",
        "semantic_topn_chunk",
        "graph_reserve_topn",
        "max_anchors",
        "samples_per_anchor",
        "phase1_run_preshortlist_topm",
        "phase1_full_run_score_topk",
        "final_top_slice_reorder_enabled",
        "final_top_slice_reorder_topk",
        "answer_support_pinning_enabled",
        "answer_support_pinning_min",
    ]
    for k in keys:
        if k in src:
            out[k] = src[k]
    return out


def _clean_text(v):
    if v is None:
        return ""
    return str(v).strip()


def _read_run_profile(summary):
    rp = dict(summary.get("retrieval_params", {}) or {})
    gp = _clean_text(rp.get("global_corpus_path", summary.get("global_corpus_path", "")))
    # keep truly empty path as empty string for regime matching.
    profile = {
        "generator": _clean_text(summary.get("generator", "vllm")) or "vllm",
        "evaluator_mode": _clean_text(summary.get("evaluator_mode", "hipporag2_parity")) or "hipporag2_parity",
        "global_corpus_path": gp,
        "openie_mode": _clean_text(rp.get("openie_mode", "llm")) or "llm",
        "openie_model_name": _clean_text(rp.get("openie_model_name", "")),
        "llm_max_new_tokens": int(float((summary.get("generation_params", {}) or {}).get("max_new_tokens", 64) or 64)),
    }
    return profile


hotpot_d1 = _load_json(hotpot_d1_path)
hotpot_f2 = _load_json(hotpot_f2_path)
wiki_d1 = _load_json(wiki_d1_path)
wiki_e1 = _load_json(wiki_e1_path)

wiki_best_label = "e1"
wiki_best_path = str(Path(wiki_e1_path).resolve())
wiki_best = wiki_e1
if str(wiki_w3_path or "").strip():
    p = Path(wiki_w3_path)
    if p.exists():
        wiki_best_label = "w3"
        wiki_best_path = str(p.resolve())
        wiki_best = _load_json(str(p))

d2_hotpot = _load_json(d2_hotpot_path)
d2_wiki = _load_json(d2_wiki_path)
hippo_hotpot = _load_json(hippo_hotpot_path)
hippo_wiki = _load_json(hippo_wiki_path)

payload = {
    "hotpotqa": {
        "d1": {
            "label": "D1",
            "path": str(Path(hotpot_d1_path).resolve()),
            "f1": _metric(hotpot_d1, "f1", "F1"),
            "em": _metric(hotpot_d1, "em", "EM"),
            "r1": _r_at(hotpot_d1, 1),
            "r5": _r_at(hotpot_d1, 5),
            "r20": _r_at(hotpot_d1, 20),
        },
        "best_non_oracle": {
            "label": "F2",
            "path": str(Path(hotpot_f2_path).resolve()),
            "f1": _metric(hotpot_f2, "f1", "F1"),
            "em": _metric(hotpot_f2, "em", "EM"),
            "r1": _r_at(hotpot_f2, 1),
            "r5": _r_at(hotpot_f2, 5),
            "r20": _r_at(hotpot_f2, 20),
        },
        "d2": {
            "label": "D2",
            "path": str(Path(d2_hotpot_path).resolve()),
            "f1": _metric(d2_hotpot, "f1", "F1"),
            "em": _metric(d2_hotpot, "em", "EM"),
            "r1": _r_at(d2_hotpot, 1),
            "r5": _r_at(d2_hotpot, 5),
            "r20": _r_at(d2_hotpot, 20),
        },
        "hipporag2": {
            "label": "HippoRAG2",
            "path": str(Path(hippo_hotpot_path).resolve()),
            "f1": _metric(hippo_hotpot, "f1", "F1"),
            "em": _metric(hippo_hotpot, "em", "EM"),
            "r1": _r_at(hippo_hotpot, 1),
            "r5": _r_at(hippo_hotpot, 5),
            "r20": _r_at(hippo_hotpot, 20),
        },
    },
    "2wikimultihopqa": {
        "d1": {
            "label": "D1",
            "path": str(Path(wiki_d1_path).resolve()),
            "f1": _metric(wiki_d1, "f1", "F1"),
            "em": _metric(wiki_d1, "em", "EM"),
            "r1": _r_at(wiki_d1, 1),
            "r5": _r_at(wiki_d1, 5),
            "r20": _r_at(wiki_d1, 20),
        },
        "best_non_oracle": {
            "label": wiki_best_label.upper(),
            "path": wiki_best_path,
            "f1": _metric(wiki_best, "f1", "F1"),
            "em": _metric(wiki_best, "em", "EM"),
            "r1": _r_at(wiki_best, 1),
            "r5": _r_at(wiki_best, 5),
            "r20": _r_at(wiki_best, 20),
        },
        "d2": {
            "label": "D2",
            "path": str(Path(d2_wiki_path).resolve()),
            "f1": _metric(d2_wiki, "f1", "F1"),
            "em": _metric(d2_wiki, "em", "EM"),
            "r1": _r_at(d2_wiki, 1),
            "r5": _r_at(d2_wiki, 5),
            "r20": _r_at(d2_wiki, 20),
        },
        "hipporag2": {
            "label": "HippoRAG2",
            "path": str(Path(hippo_wiki_path).resolve()),
            "f1": _metric(hippo_wiki, "f1", "F1"),
            "em": _metric(hippo_wiki, "em", "EM"),
            "r1": _r_at(hippo_wiki, 1),
            "r5": _r_at(hippo_wiki, 5),
            "r20": _r_at(hippo_wiki, 20),
        },
    },
    "wiki_best_label": wiki_best_label,
    "repro_knobs": {
        "hotpot_d1": _read_repro_knobs(hotpot_d1),
        "hotpot_f2": _read_repro_knobs(hotpot_f2),
        "wiki_d1": _read_repro_knobs(wiki_d1),
        "wiki_best": _read_repro_knobs(wiki_best),
    },
    "run_profiles": {
        "hotpot_d1": _read_run_profile(hotpot_d1),
        "hotpot_f2": _read_run_profile(hotpot_f2),
        "wiki_d1": _read_run_profile(wiki_d1),
        "wiki_best": _read_run_profile(wiki_best),
    },
}

for dataset in ("hotpotqa", "2wikimultihopqa"):
    for key in ("d1", "best_non_oracle", "d2", "hipporag2"):
        f1 = float(payload[dataset][key].get("f1", 0.0) or 0.0)
        r5 = float(payload[dataset][key].get("r5", 0.0) or 0.0)
        if f1 <= 0.0 and r5 <= 0.0:
            raise RuntimeError(f"invalid reference metrics for {dataset}/{key}: {payload[dataset][key]}")

out = Path(out_json)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
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
  find "${search_dir}" -type f -name "${summary_name}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | awk '{print $2}' || true
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

query_path = summary_path.with_name(query_name)
if not query_path.exists():
    raise SystemExit(1)

line_count = 0
with query_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            line_count += 1

if expected_limit > 0 and line_count < expected_limit:
    raise SystemExit(1)
if expected_limit <= 0 and line_count <= 0:
    raise SystemExit(1)

print(summary_path)
PY
}

upsert_record() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local kind="$4"
  local summary_path="$5"
  "${PYTHON_BIN}" - "${RECORD_TSV}" "${stage}" "${variant}" "${dataset}" "${kind}" "${summary_path}" <<'PY'
import sys
from pathlib import Path

tsv_path = Path(sys.argv[1])
stage, variant, dataset, kind, summary_path = sys.argv[2:]
header = "stage\tvariant\tdataset\tkind\tsummary_path\n"

records = []
seen = set()
if tsv_path.exists():
    with tsv_path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if first.strip() != header.strip():
            if first.strip():
                parts = first.rstrip("\n").split("\t")
                if len(parts) == 5:
                    key = tuple(parts[:4])
                    records.append(parts)
                    seen.add(key)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            key = tuple(parts[:4])
            if key in seen:
                continue
            records.append(parts)
            seen.add(key)

target = (stage, variant, dataset, kind)
updated = False
for row in records:
    if tuple(row[:4]) == target:
        row[4] = summary_path
        updated = True
        break
if not updated:
    records.append([stage, variant, dataset, kind, summary_path])

tsv_path.parent.mkdir(parents=True, exist_ok=True)
with tsv_path.open("w", encoding="utf-8") as f:
    f.write(header)
    for row in records:
        f.write("\t".join(row) + "\n")
PY
}

assert_cfg_race_safe() {
  local cfg_path="$1"
  "${PYTHON_BIN}" - "${cfg_path}" <<'PY'
import sys
import yaml
from pathlib import Path

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
num_workers = int(cfg.get("num_workers", 1) or 1)
force_rebuild = bool(cfg.get("force_rebuild_graph_index", False))
if num_workers > 1 and force_rebuild:
    raise SystemExit("unsafe cache/meta race config: num_workers>1 with force_rebuild_graph_index=true")
PY
}

create_variant_cfg() {
  local base_cfg="$1"
  local out_cfg="$2"
  local dataset="$3"
  local variant_tag="$4"
  local repro_key="$5"
  local shared_profile="$6"
  local proposal_step="$7"
  local exploration_step="$8"
  local downstream="$9"
  local reorder_topk="${10}"
  local pinning_min="${11}"

  mkdir -p "$(dirname "${out_cfg}")"

  "${PYTHON_BIN}" - \
    "${base_cfg}" \
    "${out_cfg}" \
    "${dataset}" \
    "${variant_tag}" \
    "${repro_key}" \
    "${shared_profile}" \
    "${proposal_step}" \
    "${exploration_step}" \
    "${downstream}" \
    "${reorder_topk}" \
    "${pinning_min}" \
    "${REF_METRICS_JSON}" <<'PY'
import json
import sys
import yaml

(
    base_cfg,
    out_cfg,
    dataset,
    variant_tag,
    repro_key,
    shared_profile,
    proposal_step,
    exploration_step,
    downstream,
    reorder_topk,
    pinning_min,
    ref_json,
) = sys.argv[1:13]

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

refs = json.loads(open(ref_json, "r", encoding="utf-8").read())
knobs = dict((refs.get("repro_knobs", {}) or {}).get(repro_key, {}) or {})

for key in (
    "semantic_topn_entity",
    "semantic_topn_chunk",
    "graph_reserve_topn",
    "max_anchors",
    "samples_per_anchor",
    "phase1_run_preshortlist_topm",
    "phase1_full_run_score_topk",
    "final_top_slice_reorder_enabled",
    "final_top_slice_reorder_topk",
    "answer_support_pinning_enabled",
    "answer_support_pinning_min",
):
    if key in knobs:
        cfg[key] = knobs[key]

cfg["dataset"] = str(dataset)
cfg["canonical_variant_name"] = str(variant_tag)
cfg["shared_budget_profile"] = str(shared_profile)
cfg["shared_budget_proposal_step"] = int(proposal_step)
cfg["shared_budget_exploration_step"] = int(exploration_step)
cfg["oracle_ceiling_f1"] = float(((refs.get(dataset, {}) or {}).get("d2", {}) or {}).get("f1", 0.0) or 0.0)

# Minimal downstream controls only.
downstream_name = str(downstream or "off").strip().lower()
if downstream_name in {"reorder", "reorder_light"}:
    cfg["final_top_slice_reorder_enabled"] = True
    cfg["final_top_slice_reorder_topk"] = max(1, int(reorder_topk))
    cfg["answer_support_pinning_enabled"] = False
elif downstream_name in {"pinning", "pinning_light"}:
    cfg["answer_support_pinning_enabled"] = True
    cfg["answer_support_pinning_min"] = max(1, int(pinning_min))
    cfg["final_top_slice_reorder_enabled"] = False
else:
    cfg["final_top_slice_reorder_enabled"] = False
    cfg["answer_support_pinning_enabled"] = False

with open(out_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
PY

  assert_cfg_race_safe "${out_cfg}"
}

get_run_profile_value() {
  local repro_key="$1"
  local field="$2"
  "${PYTHON_BIN}" - "${REF_METRICS_JSON}" "${repro_key}" "${field}" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
repro_key = sys.argv[2]
field = sys.argv[3]
profile = (payload.get("run_profiles", {}) or {}).get(repro_key, {}) or {}
value = profile.get(field, "")
if value is None:
    value = ""
print(str(value))
PY
}

annotate_summary_with_refs() {
  local summary_path="$1"
  local dataset="$2"
  "${PYTHON_BIN}" - "${summary_path}" "${dataset}" "${REF_METRICS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

from effirag.utils import compute_rag_derived_metrics

summary_path = Path(sys.argv[1])
dataset = sys.argv[2]
ref_json = Path(sys.argv[3])

summary = json.loads(summary_path.read_text(encoding="utf-8"))
refs = json.loads(ref_json.read_text(encoding="utf-8"))

d2_f1 = float(((refs.get(dataset, {}) or {}).get("d2", {}) or {}).get("f1", 0.0) or 0.0)
hippo_f1 = float(((refs.get(dataset, {}) or {}).get("hipporag2", {}) or {}).get("f1", 0.0) or 0.0)
hippo_r5 = float(((refs.get(dataset, {}) or {}).get("hipporag2", {}) or {}).get("r5", 0.0) or 0.0)

summary["oracle_ceiling_f1"] = d2_f1
summary["hipporag2_reference_f1"] = hippo_f1
summary["hipporag2_reference_recall_at_5"] = hippo_r5
summary.update(compute_rag_derived_metrics(summary))

summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

run_retrieval_sanity_variant() {
  local variant="$1"
  local dataset="$2"
  local cfg_path="$3"
  local repro_key="$4"
  local limit="$5"

  local out_dir="${ROUND_ROOT}/sanity_retrieval/${variant}/retrieval"
  mkdir -p "${out_dir}"

  if [[ "${SKIP_COMPLETED}" == "true" ]]; then
    local existing_summary
    existing_summary="$(find_latest_summary "${out_dir}" "${dataset}" "retrieval_summary.json")"
    if [[ -n "${existing_summary}" ]]; then
      if is_complete_summary "${existing_summary}" "${limit}" "retrieval_query_results.jsonl" >/dev/null 2>&1; then
        echo "[SKIP][retrieval-sanity] variant=${variant} dataset=${dataset} (found complete: ${existing_summary})"
        upsert_record "sanity_retrieval" "${variant}" "${dataset}" "retrieval" "${existing_summary}"
        return 0
      fi
      echo "[RESUME][retrieval-sanity] found incomplete artifact, rerun variant=${variant} dataset=${dataset}"
    fi
  fi

  local run_generator run_global_corpus run_openie_mode run_openie_model_name
  run_generator="$(get_run_profile_value "${repro_key}" "generator")"
  run_global_corpus="$(get_run_profile_value "${repro_key}" "global_corpus_path")"
  run_openie_mode="$(get_run_profile_value "${repro_key}" "openie_mode")"
  run_openie_model_name="$(get_run_profile_value "${repro_key}" "openie_model_name")"

  local -a cmd
  cmd=(
    "${PYTHON_BIN}" -m effirag.run_retrieval
    --config "${cfg_path}"
    --dataset "${dataset}"
    --data-path "data/qa/${dataset}.json"
    --graph-cache-dir "${GRAPH_CACHE_DIR}"
    --force-rebuild-graph-index false
    --num-workers 1
    --openie-mode "${run_openie_mode:-llm}"
    --openie-model-name "${run_openie_model_name:-${MODEL_NAME}}"
    --openie-api-base-url "${LLM_BASE_URL}"
    --openie-api-key "${LLM_API_KEY}"
    --embedding-enabled "${EMBEDDING_ENABLED}"
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
    --generator "${run_generator:-vllm}"
    --model-name "${MODEL_NAME}"
    --limit "${limit}"
    --output-dir "${out_dir}"
    --timestamp-output true
  )

  if [[ -n "${run_global_corpus}" ]]; then
    cmd+=(--global-corpus-path "${run_global_corpus}")
  fi

  echo "[RUN][retrieval-sanity] variant=${variant} dataset=${dataset} limit=${limit}"
  "${cmd[@]}"

  local summary_path
  summary_path="$(find_latest_summary "${out_dir}" "${dataset}" "retrieval_summary.json")"
  if [[ -z "${summary_path}" ]]; then
    echo "[ERROR] retrieval_summary.json not found (variant=${variant}, dataset=${dataset})" >&2
    exit 1
  fi

  upsert_record "sanity_retrieval" "${variant}" "${dataset}" "retrieval" "${summary_path}"
}

run_rag_variant() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local cfg_path="$4"
  local limit="$5"
  local repro_key="$6"

  local out_dir="${ROUND_ROOT}/${stage}/${variant}/rag"
  mkdir -p "${out_dir}"

  if [[ "${SKIP_COMPLETED}" == "true" ]]; then
    local existing_summary
    existing_summary="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
    if [[ -n "${existing_summary}" ]]; then
      if is_complete_summary "${existing_summary}" "${limit}" "rag_query_results.jsonl" >/dev/null 2>&1; then
        annotate_summary_with_refs "${existing_summary}" "${dataset}"
        echo "[SKIP][rag] stage=${stage} variant=${variant} dataset=${dataset} (found complete: ${existing_summary})"
        upsert_record "${stage}" "${variant}" "${dataset}" "rag" "${existing_summary}"
        return 0
      fi
      echo "[RESUME][rag] found incomplete artifact, rerun stage=${stage} variant=${variant} dataset=${dataset}"
    fi
  fi

  local run_generator run_evaluator run_global_corpus run_openie_mode run_openie_model_name run_llm_max_new_tokens
  run_generator="$(get_run_profile_value "${repro_key}" "generator")"
  run_evaluator="$(get_run_profile_value "${repro_key}" "evaluator_mode")"
  run_global_corpus="$(get_run_profile_value "${repro_key}" "global_corpus_path")"
  run_openie_mode="$(get_run_profile_value "${repro_key}" "openie_mode")"
  run_openie_model_name="$(get_run_profile_value "${repro_key}" "openie_model_name")"
  run_llm_max_new_tokens="$(get_run_profile_value "${repro_key}" "llm_max_new_tokens")"

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
    --openie-model-name "${run_openie_model_name:-${MODEL_NAME}}"
    --openie-api-base-url "${LLM_BASE_URL}"
    --openie-api-key "${LLM_API_KEY}"
    --embedding-enabled "${EMBEDDING_ENABLED}"
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
    --generator "${run_generator:-vllm}"
    --model-name "${MODEL_NAME}"
    --llm-base-url "${LLM_BASE_URL}"
    --llm-api-key "${LLM_API_KEY}"
    --llm-max-new-tokens "${run_llm_max_new_tokens:-64}"
    --run-qa true
    --retrieval-only false
    --evaluator-mode "${run_evaluator:-hipporag2_parity}"
    --limit "${limit}"
    --output-dir "${out_dir}"
    --timestamp-output true
  )

  if [[ -n "${run_global_corpus}" ]]; then
    cmd+=(--global-corpus-path "${run_global_corpus}")
  fi

  echo "[RUN][rag] stage=${stage} variant=${variant} dataset=${dataset} limit=${limit} profile=${repro_key}"
  "${cmd[@]}"

  local summary_path
  summary_path="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
  if [[ -z "${summary_path}" ]]; then
    echo "[ERROR] rag_summary.json not found (stage=${stage}, variant=${variant}, dataset=${dataset})" >&2
    exit 1
  fi

  annotate_summary_with_refs "${summary_path}" "${dataset}"
  upsert_record "${stage}" "${variant}" "${dataset}" "rag" "${summary_path}"
}

run_variant_for_dataset() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local repro_key="$4"
  local shared_profile="$5"
  local proposal_step="$6"
  local exploration_step="$7"
  local downstream="$8"
  local reorder_topk="$9"
  local pinning_min="${10}"
  local limit="${11}"

  local dataset_cfg_dir="${CFG_ROOT}/${stage}/${variant}/${dataset}"
  local rag_cfg="${dataset_cfg_dir}/rag.yaml"

  create_variant_cfg \
    "${RAG_BASE_CFG}" \
    "${rag_cfg}" \
    "${dataset}" \
    "${variant}" \
    "${repro_key}" \
    "${shared_profile}" \
    "${proposal_step}" \
    "${exploration_step}" \
    "${downstream}" \
    "${reorder_topk}" \
    "${pinning_min}"

  run_rag_variant "${stage}" "${variant}" "${dataset}" "${rag_cfg}" "${limit}" "${repro_key}"
}

run_stage_a_gate() {
  local out_json="${ROUND_ROOT}/stage_a_repro_gate.json"
  local out_md="${ROUND_ROOT}/stage_a_repro_gate.md"
  "${PYTHON_BIN}" - \
    "${RECORD_TSV}" \
    "${REF_METRICS_JSON}" \
    "${REPRO_F1_TOL}" \
    "${REPRO_R5_TOL}" \
    "${WIKI_REPRO_VARIANT}" \
    "${out_json}" \
    "${out_md}" <<'PY'
import json
import sys
from pathlib import Path

record_tsv, ref_json, f1_tol, r5_tol, wiki_repro_variant, out_json, out_md = sys.argv[1:8]
f1_tol = float(f1_tol)
r5_tol = float(r5_tol)

refs = json.loads(Path(ref_json).read_text(encoding="utf-8"))
records = {}
with Path(record_tsv).open("r", encoding="utf-8") as f:
    next(f, None)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 5:
            continue
        stage, variant, dataset, kind, summary_path = parts
        if stage != "stage_a" or kind != "rag":
            continue
        records[(variant, dataset)] = summary_path

spec = [
    ("hotpot_d1_repro", "hotpotqa", "d1"),
    ("hotpot_f2_repro", "hotpotqa", "best_non_oracle"),
    ("wiki_d1_repro", "2wikimultihopqa", "d1"),
    (wiki_repro_variant, "2wikimultihopqa", "best_non_oracle"),
]

rows = []
all_pass = True
for variant, dataset, ref_key in spec:
    summary_path = records.get((variant, dataset), "")
    if not summary_path:
        rows.append({"variant": variant, "dataset": dataset, "reference": ref_key, "status": "missing", "pass": False})
        all_pass = False
        continue

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    current_f1 = float(summary.get("f1", 0.0) or 0.0)
    current_r5 = float(summary.get("supporting_fact_recall_at_5", 0.0) or 0.0)
    if current_r5 <= 0.0:
        at_k = summary.get("supporting_fact_recall_at_k", {}) or {}
        current_r5 = float(at_k.get("5", at_k.get(5, 0.0)) or 0.0)
    if current_r5 <= 0.0:
        nested = summary.get("recall_at_k", {}) or {}
        current_r5 = float(nested.get("5", nested.get(5, 0.0)) or 0.0)

    ref = ((refs.get(dataset, {}) or {}).get(ref_key, {}) or {})
    ref_f1 = float(ref.get("f1", 0.0) or 0.0)
    ref_r5 = float(ref.get("r5", 0.0) or 0.0)
    delta_f1 = current_f1 - ref_f1
    delta_r5 = current_r5 - ref_r5

    passed = (abs(delta_f1) <= f1_tol) and (abs(delta_r5) <= r5_tol)
    all_pass = all_pass and passed

    rows.append(
        {
            "variant": variant,
            "dataset": dataset,
            "reference": ref_key,
            "summary_path": summary_path,
            "f1": current_f1,
            "r5": current_r5,
            "ref_f1": ref_f1,
            "ref_r5": ref_r5,
            "delta_f1": delta_f1,
            "delta_r5": delta_r5,
            "pass": bool(passed),
            "status": "ok" if passed else "drift",
        }
    )

payload = {"f1_tolerance": f1_tol, "r5_tolerance": r5_tol, "all_pass": bool(all_pass), "rows": rows}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

md_lines = [
    "# Stage A Baseline Reproduction Gate",
    "",
    f"- f1_tolerance: {f1_tol:.4f}",
    f"- r5_tolerance: {r5_tol:.4f}",
    f"- all_pass: {all_pass}",
    "",
    "| variant | dataset | ref | F1 | ref_F1 | dF1 | R@5 | ref_R@5 | dR@5 | pass |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    md_lines.append(
        "| {variant} | {dataset} | {reference} | {f1:.4f} | {ref_f1:.4f} | {delta_f1:+.4f} | {r5:.4f} | {ref_r5:.4f} | {delta_r5:+.4f} | {passed} |".format(
            variant=r.get("variant", ""),
            dataset=r.get("dataset", ""),
            reference=r.get("reference", ""),
            f1=float(r.get("f1", 0.0) or 0.0),
            ref_f1=float(r.get("ref_f1", 0.0) or 0.0),
            delta_f1=float(r.get("delta_f1", 0.0) or 0.0),
            r5=float(r.get("r5", 0.0) or 0.0),
            ref_r5=float(r.get("ref_r5", 0.0) or 0.0),
            delta_r5=float(r.get("delta_r5", 0.0) or 0.0),
            passed=str(bool(r.get("pass", False))),
        )
    )
Path(out_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print("PASS" if all_pass else "FAIL")
raise SystemExit(0 if all_pass else 3)
PY
}

run_stage_b_gate() {
  local out_json="${ROUND_ROOT}/stage_b_gate.json"
  local out_md="${ROUND_ROOT}/stage_b_gate.md"
  "${PYTHON_BIN}" - \
    "${RECORD_TSV}" \
    "${REF_METRICS_JSON}" \
    "${WIKI_REPRO_VARIANT}" \
    "${out_json}" \
    "${out_md}" <<'PY'
import json
import sys
from pathlib import Path

record_tsv, ref_json, wiki_repro_variant, out_json, out_md = sys.argv[1:6]
refs = json.loads(Path(ref_json).read_text(encoding="utf-8"))


def _load_metrics(path):
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    r5 = float(s.get("supporting_fact_recall_at_5", 0.0) or 0.0)
    if r5 <= 0.0:
        k = s.get("supporting_fact_recall_at_k", {}) or {}
        r5 = float(k.get("5", k.get(5, 0.0)) or 0.0)
    if r5 <= 0.0:
        k2 = s.get("recall_at_k", {}) or {}
        r5 = float(k2.get("5", k2.get(5, 0.0)) or 0.0)
    f1 = float(s.get("f1", 0.0) or 0.0)
    d2_f1 = float(((refs.get(s.get("dataset", ""), {}) or {}).get("d2", {}) or {}).get("f1", 0.0) or 0.0)
    hippo_f1 = float(((refs.get(s.get("dataset", ""), {}) or {}).get("hipporag2", {}) or {}).get("f1", 0.0) or 0.0)
    hippo_r5 = float(((refs.get(s.get("dataset", ""), {}) or {}).get("hipporag2", {}) or {}).get("r5", 0.0) or 0.0)
    return {
        "f1": f1,
        "r5": r5,
        "oracle_gap": (d2_f1 - f1) if d2_f1 > 0.0 else 0.0,
        "hippo_gap_f1": (hippo_f1 - f1) if hippo_f1 > 0.0 else 0.0,
        "hippo_gap_r5": (hippo_r5 - r5) if hippo_r5 > 0.0 else 0.0,
    }

rows = {}
with Path(record_tsv).open("r", encoding="utf-8") as f:
    next(f, None)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) != 5:
            continue
        stage, variant, dataset, kind, summary_path = p
        if kind != "rag":
            continue
        rows[(stage, variant, dataset)] = summary_path

hotpot_base = _load_metrics(rows[("stage_a", "hotpot_d1_repro", "hotpotqa")])
wiki_base = _load_metrics(rows[("stage_a", "wiki_d1_repro", "2wikimultihopqa")])
hotpot_b = _load_metrics(rows[("stage_b", "hotpot_g1_shared", "hotpotqa")])
wiki_b = _load_metrics(rows[("stage_b", "wiki_g1_shared", "2wikimultihopqa")])

hotpot_pass = (hotpot_b["f1"] >= hotpot_base["f1"]) and (hotpot_b["r5"] >= hotpot_base["r5"])
wiki_pass = (wiki_b["f1"] >= wiki_base["f1"]) and (wiki_b["r5"] >= wiki_base["r5"])

hippo_gap_reduced_any = (
    abs(hotpot_b["hippo_gap_f1"]) < abs(hotpot_base["hippo_gap_f1"]) or
    abs(hotpot_b["hippo_gap_r5"]) < abs(hotpot_base["hippo_gap_r5"]) or
    abs(wiki_b["hippo_gap_f1"]) < abs(wiki_base["hippo_gap_f1"]) or
    abs(wiki_b["hippo_gap_r5"]) < abs(wiki_base["hippo_gap_r5"])
)

oracle_gap_not_worse = (
    hotpot_b["oracle_gap"] <= (hotpot_base["oracle_gap"] + 1e-12) and
    wiki_b["oracle_gap"] <= (wiki_base["oracle_gap"] + 1e-12)
)

all_pass = bool(hotpot_pass and wiki_pass and hippo_gap_reduced_any and oracle_gap_not_worse)

payload = {
    "hotpot": {"baseline": hotpot_base, "stage_b": hotpot_b, "pass": bool(hotpot_pass)},
    "wiki": {"baseline": wiki_base, "stage_b": wiki_b, "pass": bool(wiki_pass)},
    "hippo_gap_reduced_any": bool(hippo_gap_reduced_any),
    "oracle_gap_not_worse": bool(oracle_gap_not_worse),
    "all_pass": bool(all_pass),
}

Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
lines = [
    "# Stage B Gate",
    "",
    f"- hotpot_pass: {hotpot_pass}",
    f"- wiki_pass: {wiki_pass}",
    f"- hippo_gap_reduced_any: {hippo_gap_reduced_any}",
    f"- oracle_gap_not_worse: {oracle_gap_not_worse}",
    f"- all_pass: {all_pass}",
    "",
    "| dataset | baseline_F1 | stageB_F1 | baseline_R@5 | stageB_R@5 | baseline_oracle_gap | stageB_oracle_gap | pass |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    f"| hotpotqa | {hotpot_base['f1']:.4f} | {hotpot_b['f1']:.4f} | {hotpot_base['r5']:.4f} | {hotpot_b['r5']:.4f} | {hotpot_base['oracle_gap']:.4f} | {hotpot_b['oracle_gap']:.4f} | {hotpot_pass} |",
    f"| 2wikimultihopqa | {wiki_base['f1']:.4f} | {wiki_b['f1']:.4f} | {wiki_base['r5']:.4f} | {wiki_b['r5']:.4f} | {wiki_base['oracle_gap']:.4f} | {wiki_b['oracle_gap']:.4f} | {wiki_pass} |",
]
Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

print("PASS" if all_pass else "FAIL")
raise SystemExit(0 if all_pass else 4)
PY
}

build_round_metrics() {
  local out_json="${ROUND_ROOT}/round_metrics.json"
  local out_md="${ROUND_ROOT}/round_metrics.md"
  "${PYTHON_BIN}" - "${RECORD_TSV}" "${REF_METRICS_JSON}" "${out_json}" "${out_md}" <<'PY'
import json
import sys
from pathlib import Path

record_tsv, ref_json, out_json, out_md = sys.argv[1:5]
refs = json.loads(Path(ref_json).read_text(encoding="utf-8"))


def _safe(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def _r_at(summary, k):
    direct = _safe(summary.get(f"supporting_fact_recall_at_{k}", 0.0))
    if direct > 0.0:
        return direct
    bag = summary.get("supporting_fact_recall_at_k", {}) or {}
    if str(k) in bag:
        return _safe(bag[str(k)])
    if k in bag:
        return _safe(bag[k])
    bag2 = summary.get("recall_at_k", {}) or {}
    if str(k) in bag2:
        return _safe(bag2[str(k)])
    if k in bag2:
        return _safe(bag2[k])
    return 0.0


def _row_from_summary(stage, variant, dataset, kind, summary_path, summary):
    sf_recall = _safe(summary.get("supporting_fact_recall", 0.0))
    rendered_sf_recall = _safe(summary.get("rendered_supporting_fact_recall", summary.get("rendered_sf_recall", 0.0)))
    f1 = _safe(summary.get("f1", summary.get("F1", 0.0)))
    em = _safe(summary.get("em", summary.get("EM", 0.0)))
    r1 = _r_at(summary, 1)
    r5 = _r_at(summary, 5)
    r20 = _r_at(summary, 20)

    d1 = ((refs.get(dataset, {}) or {}).get("d1", {}) or {})
    best = ((refs.get(dataset, {}) or {}).get("best_non_oracle", {}) or {})
    d2 = ((refs.get(dataset, {}) or {}).get("d2", {}) or {})
    hippo = ((refs.get(dataset, {}) or {}).get("hipporag2", {}) or {})

    d2_f1 = _safe(d2.get("f1", 0.0))
    hippo_r5 = _safe(hippo.get("r5", 0.0))
    hippo_f1 = _safe(hippo.get("f1", 0.0))

    rendered_retention = _safe(summary.get("rendered_retention", (rendered_sf_recall / sf_recall if sf_recall > 0 else 0.0)))
    answer_conversion = _safe(summary.get("answer_conversion", (f1 / rendered_sf_recall if rendered_sf_recall > 0 else 0.0)))

    return {
        "stage": stage,
        "variant": variant,
        "dataset": dataset,
        "kind": kind,
        "summary_path": summary_path,
        "supporting_fact_recall": sf_recall,
        "Recall@1": r1,
        "Recall@5": r5,
        "Recall@20": r20,
        "rendered_supporting_fact_recall": rendered_sf_recall,
        "EM": em,
        "F1": f1,
        "retrieval_ms": _safe(summary.get("retrieval_latency_ms", summary.get("retrieval_ms", 0.0))),
        "total_ms": _safe(summary.get("total_latency_ms", summary.get("total_ms", 0.0))),
        "rendered_retention": rendered_retention,
        "answer_conversion": answer_conversion,
        "oracle_gap": (d2_f1 - f1) if d2_f1 > 0.0 else 0.0,
        "gap_to_hippo_r5": (hippo_r5 - r5) if hippo_r5 > 0.0 else 0.0,
        "gap_to_hippo_f1": (hippo_f1 - f1) if hippo_f1 > 0.0 else 0.0,
        "delta_R5_vs_D1": r5 - _safe(d1.get("r5", 0.0)),
        "delta_EM_vs_D1": em - _safe(d1.get("em", 0.0)),
        "delta_F1_vs_D1": f1 - _safe(d1.get("f1", 0.0)),
        "delta_R5_vs_best_non_oracle": r5 - _safe(best.get("r5", 0.0)),
        "delta_EM_vs_best_non_oracle": em - _safe(best.get("em", 0.0)),
        "delta_F1_vs_best_non_oracle": f1 - _safe(best.get("f1", 0.0)),
    }

rows = []
with Path(record_tsv).open("r", encoding="utf-8") as f:
    next(f, None)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) != 5:
            continue
        stage, variant, dataset, kind, summary_path = p
        if kind != "rag":
            continue
        try:
            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(_row_from_summary(stage, variant, dataset, kind, summary_path, summary))

for dataset in ("hotpotqa", "2wikimultihopqa"):
    ds = refs.get(dataset, {}) or {}
    for key, variant in (
        ("d1", "D1"),
        ("best_non_oracle", f"best_non_oracle({(ds.get('best_non_oracle', {}) or {}).get('label', '')})"),
        ("d2", "D2_upper_bound"),
        ("hipporag2", "HippoRAG2"),
    ):
        ref = ds.get(key, {}) or {}
        fake_summary = {
            "supporting_fact_recall": 0.0,
            "supporting_fact_recall_at_1": _safe(ref.get("r1", 0.0)),
            "supporting_fact_recall_at_5": _safe(ref.get("r5", 0.0)),
            "supporting_fact_recall_at_20": _safe(ref.get("r20", 0.0)),
            "rendered_supporting_fact_recall": 0.0,
            "em": _safe(ref.get("em", 0.0)),
            "f1": _safe(ref.get("f1", 0.0)),
            "retrieval_latency_ms": 0.0,
            "total_latency_ms": 0.0,
        }
        rows.append(_row_from_summary("reference", variant, dataset, "rag", ref.get("path", ""), fake_summary))

payload = {"wiki_best_label": refs.get("wiki_best_label", "e1"), "records": rows}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# Round Metrics (RAG full only)",
    "",
    "| stage | variant | dataset | R@5 | EM | F1 | oracle_gap | gap_to_hippo_r5 | gap_to_hippo_f1 |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        "| {stage} | {variant} | {dataset} | {r5:.4f} | {em:.4f} | {f1:.4f} | {og:+.4f} | {ghr5:+.4f} | {ghf1:+.4f} |".format(
            stage=r.get("stage", ""),
            variant=r.get("variant", ""),
            dataset=r.get("dataset", ""),
            r5=_safe(r.get("Recall@5", 0.0)),
            em=_safe(r.get("EM", 0.0)),
            f1=_safe(r.get("F1", 0.0)),
            og=_safe(r.get("oracle_gap", 0.0)),
            ghr5=_safe(r.get("gap_to_hippo_r5", 0.0)),
            ghf1=_safe(r.get("gap_to_hippo_f1", 0.0)),
        )
    )
Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

if [[ "${RUN_PREFLIGHT}" == "true" ]]; then
  echo "[CHECK] python -m py_compile effirag/*.py"
  "${PYTHON_BIN}" -m py_compile effirag/*.py

  echo "[CHECK] PYTHONPATH=. python3 -m effirag.config_audit configs/canonical"
  PYTHONPATH=. "${PYTHON_BIN}" -m effirag.config_audit configs/canonical

  echo "[CHECK] PYTHONPATH=. python3 tests/smoke_entity_chunk.py"
  PYTHONPATH=. "${PYTHON_BIN}" tests/smoke_entity_chunk.py
fi

build_reference_metrics
WIKI_BEST_LABEL="$(${PYTHON_BIN} - "${REF_METRICS_JSON}" <<'PY'
import json, sys
print((json.loads(open(sys.argv[1], 'r', encoding='utf-8').read()) or {}).get('wiki_best_label', 'e1'))
PY
)"
WIKI_REPRO_VARIANT="wiki_${WIKI_BEST_LABEL}_repro"

if [[ "${RUN_SANITY}" == "true" ]]; then
  local_hotpot_cfg="${CFG_ROOT}/sanity_retrieval/sanity_hotpot_d1/hotpotqa/retrieval.yaml"
  local_wiki_cfg="${CFG_ROOT}/sanity_retrieval/sanity_wiki_d1/2wikimultihopqa/retrieval.yaml"

  create_variant_cfg "${RETR_BASE_CFG}" "${local_hotpot_cfg}" "hotpotqa" "sanity_hotpot_d1" "hotpot_d1" "off" 0 0 "off" 4 1
  create_variant_cfg "${RETR_BASE_CFG}" "${local_wiki_cfg}" "2wikimultihopqa" "sanity_wiki_d1" "wiki_d1" "off" 0 0 "off" 4 1

  echo "[Sanity Retrieval-only] hotpotqa D1-style (limit=${LIMIT_SANITY})"
  run_retrieval_sanity_variant "sanity_hotpot_d1" "hotpotqa" "${local_hotpot_cfg}" "hotpot_d1" "${LIMIT_SANITY}"

  echo "[Sanity Retrieval-only] 2wikimultihopqa D1-style (limit=${LIMIT_SANITY})"
  run_retrieval_sanity_variant "sanity_wiki_d1" "2wikimultihopqa" "${local_wiki_cfg}" "wiki_d1" "${LIMIT_SANITY}"
fi

echo "[Stage A] Regime-matched baseline reproduction"
run_variant_for_dataset "stage_a" "hotpot_d1_repro" "hotpotqa" "hotpot_d1" "off" 0 0 "off" 4 1 "${LIMIT_STAGE_A}"
run_variant_for_dataset "stage_a" "hotpot_f2_repro" "hotpotqa" "hotpot_f2" "off" 0 0 "pinning_light" 4 1 "${LIMIT_STAGE_A}"
run_variant_for_dataset "stage_a" "wiki_d1_repro" "2wikimultihopqa" "wiki_d1" "off" 0 0 "off" 4 1 "${LIMIT_STAGE_A}"
run_variant_for_dataset "stage_a" "${WIKI_REPRO_VARIANT}" "2wikimultihopqa" "wiki_best" "off" 0 0 "off" 4 1 "${LIMIT_STAGE_A}"

STAGE_A_PASS=false
if run_stage_a_gate; then
  STAGE_A_PASS=true
  echo "[PASS] Stage A reproduction gate"
else
  echo "[FAIL] Stage A reproduction gate"
fi

if [[ "${STAGE_A_PASS}" != "true" ]]; then
  build_round_metrics
  if [[ "${STOP_ON_STAGE_A_FAIL}" == "true" ]]; then
    echo "[STOP] Stage A failed; skipping Stage B/C"
    echo "[DONE] outputs: ${ROUND_ROOT}"
    exit 3
  fi
fi

echo "[Stage B] Shared g1 revalidation"
run_variant_for_dataset "stage_b" "hotpot_g1_shared" "hotpotqa" "hotpot_d1" "proposal_light" 0 0 "off" 4 1 "${LIMIT_STAGE_B}"
run_variant_for_dataset "stage_b" "wiki_g1_shared" "2wikimultihopqa" "wiki_d1" "proposal_light" 0 0 "off" 4 1 "${LIMIT_STAGE_B}"

STAGE_B_PASS=false
if run_stage_b_gate; then
  STAGE_B_PASS=true
  echo "[PASS] Stage B gate"
else
  echo "[FAIL] Stage B gate"
fi

if [[ "${STAGE_B_PASS}" != "true" ]]; then
  build_round_metrics
  if [[ "${STOP_ON_STAGE_B_FAIL}" == "true" ]]; then
    echo "[STOP] Stage B failed; skipping Stage C"
    echo "[DONE] outputs: ${ROUND_ROOT}"
    exit 4
  fi
fi

if [[ "${STAGE_B_PASS}" == "true" ]]; then
  echo "[Stage C] Tiny dataset-specific add-on"
  run_variant_for_dataset "stage_c" "hotpot_next" "hotpotqa" "hotpot_f2" "proposal_light" 0 0 "pinning_light" 4 1 "${LIMIT_STAGE_C}"
  run_variant_for_dataset "stage_c" "wiki_next" "2wikimultihopqa" "wiki_best" "proposal_light" 0 0 "reorder_light" 3 1 "${LIMIT_STAGE_C}"
else
  echo "[SKIP] Stage C skipped because Stage B gate did not pass"
fi

# Cache/meta race guard check across generated configs.
echo "[CHECK] cache/meta race guard across generated configs"
"${PYTHON_BIN}" - "${CFG_ROOT}" <<'PY'
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
unsafe = []
for p in root.rglob("*.yaml"):
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if int(cfg.get("num_workers", 1) or 1) > 1 and bool(cfg.get("force_rebuild_graph_index", False)):
        unsafe.append(str(p))
if unsafe:
    raise SystemExit("unsafe cache/meta race configs:\n" + "\n".join(unsafe))
print("safe")
PY

build_round_metrics

echo "[DONE] outputs: ${ROUND_ROOT}"
echo "[DONE] run records: ${RECORD_TSV}"
echo "[DONE] reference metrics: ${REF_METRICS_JSON}"
echo "[DONE] round metrics: ${ROUND_ROOT}/round_metrics.json"
