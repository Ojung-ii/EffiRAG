#!/usr/bin/env bash
set -euo pipefail

dataset="hotpotqa"
rag_limit=100
retrieval_limit=100
output_root="outputs"
run_hf="false"
run_oracle="false"
hf_model="Qwen/Qwen3.5-2B"
max_context_sentences=10
methods_csv="effirag,naive_graphrag"
run_retrieval="true"
run_rag="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      dataset="$2"
      shift 2
      ;;
    --rag-limit)
      rag_limit="$2"
      shift 2
      ;;
    --retrieval-limit)
      retrieval_limit="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --methods)
      methods_csv="$2"
      shift 2
      ;;
    --run-hf)
      run_hf="true"
      shift
      ;;
    --run-oracle)
      run_oracle="true"
      shift
      ;;
    --hf-model)
      hf_model="$2"
      shift 2
      ;;
    --max-context-sentences)
      max_context_sentences="$2"
      shift 2
      ;;
    --skip-retrieval)
      run_retrieval="false"
      shift
      ;;
    --skip-rag)
      run_rag="false"
      shift
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: scripts/run_toy.sh [options]

Options:
  --dataset <name>                Dataset name (default: hotpotqa)
  --rag-limit <int>               Number of RAG toy samples (default: 3)
  --retrieval-limit <int>         Number of retrieval toy samples (default: 5)
  --output-root <dir>             Root output directory (default: outputs)
  --methods <csv>                 Methods to compare (default: effirag,naive_graphrag)
  --skip-retrieval                Skip retrieval-only comparison
  --skip-rag                      Skip RAG comparison
  --run-hf                        Also run HF generator comparison
  --run-oracle                    Also run oracle generator comparison
  --hf-model <model-name>         HF model name for --run-hf (default: Qwen/Qwen3.5-2B)
  --max-context-sentences <int>   Max context sentences for HF run (default: 10)
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Try: scripts/run_toy.sh --help" >&2
      exit 1
      ;;
  esac
done

IFS=',' read -r -a methods <<< "${methods_csv}"
generators=("heuristic")
if [[ "${run_oracle}" == "true" ]]; then
  generators+=("oracle")
fi
if [[ "${run_hf}" == "true" ]]; then
  generators+=("hf")
fi

run_stamp="$(date -u +%Y%m%d_%H%M%S)"
base_dir="${output_root}/toy_compare_${run_stamp}"

echo "[toy] output root: ${base_dir}"
echo "[toy] methods: ${methods[*]}"
echo "[toy] generators: ${generators[*]}"

if [[ "${run_retrieval}" == "true" ]]; then
  echo "[toy] retrieval compare limit=${retrieval_limit}"
  for method in "${methods[@]}"; do
    out_dir="${base_dir}/retrieval/${method}"
    echo "[toy][retrieval] method=${method}"
    python3 -m effirag.run_retrieval \
      --dataset "${dataset}" \
      --limit "${retrieval_limit}" \
      --method "${method}" \
      --output-dir "${out_dir}"
  done
fi

if [[ "${run_rag}" == "true" ]]; then
  echo "[toy] rag compare limit=${rag_limit}"
  for generator in "${generators[@]}"; do
    for method in "${methods[@]}"; do
      out_dir="${base_dir}/rag/${generator}/${method}"
      echo "[toy][rag] method=${method} generator=${generator}"
      cmd=(
        python3 -m effirag.run_rag
        --dataset "${dataset}"
        --limit "${rag_limit}"
        --method "${method}"
        --generator "${generator}"
        --run-qa true
        --output-dir "${out_dir}"
      )
      if [[ "${generator}" == "hf" ]]; then
        cmd+=(
          --model-name "${hf_model}"
          --max-context-sentences "${max_context_sentences}"
        )
      fi
      "${cmd[@]}"
    done
  done
fi

echo "[toy] done"
summary_logs_dir="${base_dir}/logs"
summary_md="${summary_logs_dir}/toy_summary.md"
summary_json="${summary_logs_dir}/toy_summary.json"
summary_csv="${summary_logs_dir}/toy_summary.csv"
mkdir -p "${summary_logs_dir}"

python3 - <<PY
import csv
import json
from pathlib import Path

base = Path("${base_dir}")
run_retrieval = "${run_retrieval}" == "true"
run_rag = "${run_rag}" == "true"

rows = []

if run_retrieval:
    for fp in sorted((base / "retrieval").glob("*/retrieval_summary.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        rows.append(
            {
                "task": "retrieval",
                "method": data.get("method", fp.parent.name),
                "generator": "-",
                "samples": int(data.get("n_samples", 0)),
                "sf_recall": float(data.get("supporting_fact_recall", 0.0)),
                "sf_precision": float(data.get("supporting_fact_precision", 0.0)),
                "em": "",
                "f1": "",
                "retrieval_ms": float(data.get("retrieval_latency_ms", 0.0)),
                "total_ms": "",
                "summary_path": str(fp),
            }
        )

if run_rag:
    for fp in sorted((base / "rag").glob("*/*/rag_summary.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        rows.append(
            {
                "task": "rag",
                "method": data.get("method", fp.parent.name),
                "generator": data.get("generator", fp.parent.parent.name),
                "samples": int(data.get("n_samples", 0)),
                "sf_recall": float(data.get("supporting_fact_recall", 0.0)),
                "sf_precision": float(data.get("supporting_fact_precision", 0.0)),
                "em": float(data.get("em", 0.0)),
                "f1": float(data.get("f1", 0.0)),
                "retrieval_ms": float(data.get("retrieval_latency_ms", 0.0)),
                "total_ms": float(data.get("total_latency_ms", 0.0)),
                "summary_path": str(fp),
            }
        )

headers = ["task", "method", "generator", "samples", "sf_recall", "sf_precision", "em", "f1", "retrieval_ms", "total_ms"]

def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

md_lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for row in rows:
    md_lines.append("| " + " | ".join(_fmt(row.get(h, "")) for h in headers) + " |")
md_text = "\\n".join(md_lines)
title = "**Total Eval Summary**"

print("[toy][summary-table]")
print(title)
print(md_text)

Path("${summary_md}").write_text(title + "\\n\\n" + md_text + "\\n", encoding="utf-8")
Path("${summary_json}").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")

with Path("${summary_csv}").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers + ["summary_path"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
PY

echo "[toy] saved summary log: ${summary_md}"
echo "[toy] saved summary json: ${summary_json}"
echo "[toy] saved summary csv: ${summary_csv}"
