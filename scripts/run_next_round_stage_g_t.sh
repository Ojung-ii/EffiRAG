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

LIMIT_STAGE_G="${LIMIT_STAGE_G:-1000}"
LIMIT_STAGE_T="${LIMIT_STAGE_T:-1000}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
EMBEDDING_ENABLED="${EMBEDDING_ENABLED:-true}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
ALLOW_CPU_EMBEDDING="${ALLOW_CPU_EMBEDDING:-false}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
RESUME_ROUND_ROOT="${RESUME_ROUND_ROOT:-}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

CUDA_AVAILABLE="$("${PYTHON_BIN}" - <<'PY'
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
  echo "  2) Run fast fallback: EMBEDDING_ENABLED=false bash scripts/run_next_round_stage_g_t.sh" >&2
  echo "  3) Force CPU embedding (very slow): ALLOW_CPU_EMBEDDING=true bash scripts/run_next_round_stage_g_t.sh" >&2
  exit 2
fi

RAG_BASE_CFG="configs/canonical/rag_entity_first_chunk_grounded.yaml"

if [[ -n "${RESUME_ROUND_ROOT}" ]]; then
  ROUND_ROOT="${RESUME_ROUND_ROOT}"
  echo "[INFO] resume mode: ROUND_ROOT=${ROUND_ROOT}"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  ROUND_ROOT="outputs/next_round_stage_g_t/${RUN_STAMP}"
fi
CFG_ROOT="${ROUND_ROOT}/configs"
mkdir -p "${CFG_ROOT}"

RECORD_TSV="${ROUND_ROOT}/run_records.tsv"
if [[ ! -f "${RECORD_TSV}" ]]; then
  printf "stage\tvariant\tdataset\tkind\tsummary_path\n" > "${RECORD_TSV}"
fi

declare -A D2_CEILING_F1
D2_CEILING_F1[hotpotqa]="0.6374"
D2_CEILING_F1[2wikimultihopqa]="0.5747"

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

is_complete_rag_summary() {
  local summary_path="$1"
  local expected_limit="$2"
  "${PYTHON_BIN}" - "${summary_path}" "${expected_limit}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_limit = int(float(sys.argv[2]))
if not summary_path.exists():
    raise SystemExit(1)

try:
    _ = json.loads(summary_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

query_path = summary_path.with_name("rag_query_results.jsonl")
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
            # Preserve compatibility with existing files missing header.
            if first.strip():
                parts = first.rstrip("\n").split("\t")
                if len(parts) == 5:
                    k = tuple(parts[:4])
                    records.append(parts)
                    seen.add(k)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            key = tuple(parts[:4])
            if key in seen:
                continue
            records.append(parts)
            seen.add(key)

target_key = (stage, variant, dataset, kind)
updated = False
for row in records:
    if tuple(row[:4]) == target_key:
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

create_variant_cfg() {
  local base_cfg="$1"
  local out_cfg="$2"
  local dataset="$3"
  local profile="$4"
  local proposal_step="$5"
  local exploration_step="$6"
  local downstream="$7"
  local oracle_ceiling="$8"
  local variant_tag="$9"

  mkdir -p "$(dirname "${out_cfg}")"

  "${PYTHON_BIN}" - "${base_cfg}" "${out_cfg}" "${dataset}" "${profile}" "${proposal_step}" "${exploration_step}" "${downstream}" "${oracle_ceiling}" "${variant_tag}" <<'PY'
import sys
import yaml

base_cfg, out_cfg, dataset, profile, proposal_step, exploration_step, downstream, oracle_ceiling, variant_tag = sys.argv[1:]

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

cfg["dataset"] = str(dataset)
cfg["canonical_variant_name"] = str(variant_tag)
cfg["shared_budget_profile"] = str(profile)
cfg["shared_budget_proposal_step"] = int(proposal_step)
cfg["shared_budget_exploration_step"] = int(exploration_step)
cfg["oracle_ceiling_f1"] = float(oracle_ceiling)

if downstream == "reorder":
    cfg["final_top_slice_reorder_enabled"] = True
    cfg["final_top_slice_reorder_topk"] = int(cfg.get("final_top_slice_reorder_topk", 4) or 4)
    cfg["answer_support_pinning_enabled"] = False
elif downstream == "pinning":
    cfg["answer_support_pinning_enabled"] = True
    cfg["answer_support_pinning_min"] = max(1, int(cfg.get("answer_support_pinning_min", 1) or 1))
    cfg["final_top_slice_reorder_enabled"] = False
else:
    cfg["final_top_slice_reorder_enabled"] = False
    cfg["answer_support_pinning_enabled"] = False

with open(out_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
PY
}

run_rag_variant() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local cfg_path="$4"
  local limit="$5"

  local out_dir="${ROUND_ROOT}/${stage}/${variant}/rag"
  mkdir -p "${out_dir}"

  if [[ "${SKIP_COMPLETED}" == "true" ]]; then
    local existing_summary
    existing_summary="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
    if [[ -n "${existing_summary}" ]]; then
      if is_complete_rag_summary "${existing_summary}" "${limit}" >/dev/null 2>&1; then
        echo "[SKIP][rag] stage=${stage} variant=${variant} dataset=${dataset} (found complete: ${existing_summary})"
        upsert_record "${stage}" "${variant}" "${dataset}" "rag" "${existing_summary}"
        return 0
      fi
      echo "[RESUME][rag] found incomplete artifact, rerun stage=${stage} variant=${variant} dataset=${dataset}"
    fi
  fi

  echo "[RUN][rag] stage=${stage} variant=${variant} dataset=${dataset} limit=${limit}"
  "${PYTHON_BIN}" -m effirag.run_rag \
    --config "${cfg_path}" \
    --dataset "${dataset}" \
    --data-path "data/qa/${dataset}.json" \
    --global-corpus-path "data/${dataset}_corpus.json" \
    --graph-cache-dir "${GRAPH_CACHE_DIR}" \
    --force-rebuild-graph-index false \
    --num-workers 1 \
    --openie-mode llm \
    --openie-model-name "${MODEL_NAME}" \
    --openie-api-base-url "${LLM_BASE_URL}" \
    --openie-api-key "${LLM_API_KEY}" \
    --embedding-enabled "${EMBEDDING_ENABLED}" \
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
    --generator openai_compat \
    --model-name "${MODEL_NAME}" \
    --llm-base-url "${LLM_BASE_URL}" \
    --llm-api-key "${LLM_API_KEY}" \
    --llm-max-new-tokens 64 \
    --run-qa true \
    --retrieval-only false \
    --evaluator-mode hipporag2_parity \
    --limit "${limit}" \
    --output-dir "${out_dir}" \
    --timestamp-output true

  local summary_path
  summary_path="$(find_latest_summary "${out_dir}" "${dataset}" "rag_summary.json")"
  if [[ -z "${summary_path}" ]]; then
    echo "[ERROR] rag_summary.json not found (stage=${stage}, variant=${variant}, dataset=${dataset})" >&2
    exit 1
  fi
  upsert_record "${stage}" "${variant}" "${dataset}" "rag" "${summary_path}"
}

run_variant_for_dataset() {
  local stage="$1"
  local variant="$2"
  local dataset="$3"
  local profile="$4"
  local proposal_step="$5"
  local exploration_step="$6"
  local downstream="$7"
  local limit="$8"

  local dataset_cfg_dir="${CFG_ROOT}/${stage}/${variant}/${dataset}"
  local rag_cfg="${dataset_cfg_dir}/rag.yaml"
  local oracle_ceiling="${D2_CEILING_F1[${dataset}]}"

  create_variant_cfg "${RAG_BASE_CFG}" "${rag_cfg}" "${dataset}" "${profile}" "${proposal_step}" "${exploration_step}" "${downstream}" "${oracle_ceiling}" "${variant}"

  # RAG includes retrieval, so we run only RAG to avoid duplicated experiments.
  run_rag_variant "${stage}" "${variant}" "${dataset}" "${rag_cfg}" "${limit}"
}

if [[ "${RUN_PREFLIGHT}" == "true" ]]; then
  echo "[CHECK] python -m py_compile effirag/*.py"
  "${PYTHON_BIN}" -m py_compile effirag/*.py

  echo "[CHECK] PYTHONPATH=. python -m effirag.config_audit configs/canonical"
  PYTHONPATH=. "${PYTHON_BIN}" -m effirag.config_audit configs/canonical

  echo "[CHECK] PYTHONPATH=. python tests/smoke_entity_chunk.py"
  PYTHONPATH=. "${PYTHON_BIN}" tests/smoke_entity_chunk.py
fi

DATASETS=(hotpotqa 2wikimultihopqa)

echo "[Stage G] G1 shared proposal expansion"
for dataset in "${DATASETS[@]}"; do
  run_variant_for_dataset "stage_g" "g1_proposal" "${dataset}" "g1_proposal" 0 0 "off" "${LIMIT_STAGE_G}"
done

echo "[Stage G] G2 shared exploration expansion"
for dataset in "${DATASETS[@]}"; do
  run_variant_for_dataset "stage_g" "g2_exploration" "${dataset}" "g2_exploration" 0 0 "off" "${LIMIT_STAGE_G}"
done

G_WINNER_VARIANT="$("${PYTHON_BIN}" - "${RECORD_TSV}" <<'PY'
import json
import sys
from statistics import mean

record_tsv = sys.argv[1]
variants = ["g1_proposal", "g2_exploration"]
values = {v: [] for v in variants}

with open(record_tsv, "r", encoding="utf-8") as f:
    next(f, None)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 5:
            continue
        stage, variant, dataset, kind, summary_path = parts
        if stage != "stage_g" or kind != "rag":
            continue
        if variant not in values:
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as sf:
                summary = json.load(sf)
        except Exception:
            continue
        r5 = float(summary.get("supporting_fact_recall_at_5", 0.0) or 0.0)
        values[variant].append(r5)

means = {k: (mean(v) if v else -1.0) for k, v in values.items()}
winner = max(means.items(), key=lambda kv: (kv[1], kv[0]))[0]
print(winner)
PY
)"

echo "[Stage G] winner profile by mean R@5(from RAG retrieval): ${G_WINNER_VARIANT}"

echo "[Stage G] G3 retrieval winner + minimal downstream(final_top_slice_reorder)"
for dataset in "${DATASETS[@]}"; do
  run_variant_for_dataset "stage_g" "g3_${G_WINNER_VARIANT}_plus_reorder" "${dataset}" "${G_WINNER_VARIANT}" 0 0 "reorder" "${LIMIT_STAGE_G}"
done

echo "[Stage T] Hotpot precision-preserving tuning (exploration + reorder)"
run_variant_for_dataset "stage_t" "t_hotpot_precision" "hotpotqa" "t_hotpot" 0 0 "reorder" "${LIMIT_STAGE_T}"

echo "[Stage T] 2Wiki coverage-first tuning (proposal + pinning)"
run_variant_for_dataset "stage_t" "t_2wiki_coverage" "2wikimultihopqa" "t_2wiki" 0 0 "pinning" "${LIMIT_STAGE_T}"

"${PYTHON_BIN}" - "${RECORD_TSV}" "${ROUND_ROOT}/round_metrics.json" "${G_WINNER_VARIANT}" <<'PY'
import json
import sys
from pathlib import Path

record_tsv = Path(sys.argv[1])
out_json = Path(sys.argv[2])
g_winner = sys.argv[3]

rows = []
with record_tsv.open("r", encoding="utf-8") as f:
    next(f, None)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 5:
            continue
        stage, variant, dataset, kind, summary_path = parts
        summary = {}
        try:
            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        except Exception:
            summary = {}

        row = {
            "stage": stage,
            "variant": variant,
            "dataset": dataset,
            "kind": kind,
            "summary_path": summary_path,
        }
        # Keep common metrics and include all recall-family metrics available in summaries.
        row.update({
            "supporting_fact_recall": float(summary.get("supporting_fact_recall", 0.0) or 0.0),
            "rendered_supporting_fact_recall": float(summary.get("rendered_supporting_fact_recall", 0.0) or 0.0),
            "em": float(summary.get("em", 0.0) or 0.0),
            "f1": float(summary.get("f1", 0.0) or 0.0),
            "retrieval_ms": float(summary.get("retrieval_latency_ms", 0.0) or 0.0),
            "total_ms": float(summary.get("total_latency_ms", 0.0) or 0.0),
            "rendered_retention": float(summary.get("rendered_retention", 0.0) or 0.0),
            "answer_conversion": float(summary.get("answer_conversion", 0.0) or 0.0),
            "oracle_gap": float(summary.get("oracle_gap", 0.0) or 0.0),
        })
        for key, value in summary.items():
            lk = str(key).lower()
            if "recall" not in lk:
                continue
            try:
                row[str(key)] = float(value)
            except Exception:
                continue
        recall_at_k = summary.get("recall_at_k", {})
        if isinstance(recall_at_k, dict):
            for k, value in recall_at_k.items():
                out_key = f"recall_at_{k}"
                try:
                    row[out_key] = float(value)
                except Exception:
                    continue
        rows.append(row)

payload = {
    "stage_g_winner_variant": g_winner,
    "records": rows,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "[DONE] next round outputs: ${ROUND_ROOT}"
echo "[DONE] run records: ${RECORD_TSV}"
echo "[DONE] round metrics: ${ROUND_ROOT}/round_metrics.json"
