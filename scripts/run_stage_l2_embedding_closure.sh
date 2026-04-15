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

# Closure tolerance for "effectively zero" criteria.
TOKEN_CLIP_EPS="${TOKEN_CLIP_EPS:-0.0001}"
CHAR_CLIP_EPS="${CHAR_CLIP_EPS:-0.001}"
PROMPT_OVER_EPS="${PROMPT_OVER_EPS:-0.0}"
PROMPT_AUDIT_SAMPLES_PER_DATASET="${PROMPT_AUDIT_SAMPLES_PER_DATASET:-200}"
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

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
ROUND_ROOT="outputs/stage_l2_embedding_closure/${RUN_STAMP}"
mkdir -p "${ROUND_ROOT}"

REF_METRICS_JSON="${ROUND_ROOT}/reference_metrics_stage_l2.json"
LENGTH_AUDIT_JSON="${ROUND_ROOT}/stage_l2_length_audit.json"
LENGTH_AUDIT_MD="${ROUND_ROOT}/stage_l2_length_audit.md"
CLOSURE_JSON="${ROUND_ROOT}/stage_l2_closure_summary.json"
CLOSURE_MD="${ROUND_ROOT}/stage_l2_closure_summary.md"

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

echo "[CHECK] python -m py_compile effirag/*.py"
"${PYTHON_BIN}" -m py_compile effirag/*.py

echo "[CHECK] PYTHONPATH=. python3 -m effirag.config_audit configs/canonical"
PYTHONPATH=. "${PYTHON_BIN}" -m effirag.config_audit configs/canonical

echo "[Stage L2] Build frozen reference bundle (execution embedding limits applied)"
"${PYTHON_BIN}" - \
  "${REF_METRICS_JSON}" \
  "${ROUND_ROOT}" \
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
import shutil
import sys
from pathlib import Path
import yaml

(
    out_json,
    round_root,
    hotpot_d1_path,
    hotpot_f2_path,
    wiki_d1_path,
    wiki_e1_path,
    wiki_w3_path,
    d2_hotpot_path,
    d2_wiki_path,
    hippo_hotpot_path,
    hippo_wiki_path,
) = sys.argv[1:12]

round_root = Path(round_root)
adjusted_dir = round_root / "adjusted_refs"
adjusted_dir.mkdir(parents=True, exist_ok=True)

retr_cfg = yaml.safe_load(Path("configs/canonical/retrieval_entity_first_chunk_grounded.yaml").read_text(encoding="utf-8")) or {}
exec_embedding_max_length = int(retr_cfg.get("embedding_max_length", 320) or 320)
exec_embedding_text_max_chars = int(retr_cfg.get("embedding_text_max_chars", 900) or 900)


def _load(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def _clone_with_exec_embedding_limits(src_path, dst_name):
    src_p = Path(src_path).resolve()
    src = _load(src_p)
    rp = dict(src.get("retrieval_params", {}) or {})
    rp["embedding_max_length"] = int(exec_embedding_max_length)
    rp["embedding_text_max_chars"] = int(exec_embedding_text_max_chars)
    src["retrieval_params"] = rp
    dst = adjusted_dir / dst_name
    dst.write_text(json.dumps(src, ensure_ascii=False, indent=2), encoding="utf-8")
    src_q = src_p.with_name("rag_query_results.jsonl")
    dst_q = dst.with_name("rag_query_results.jsonl")
    if src_q.exists():
        shutil.copy2(str(src_q), str(dst_q))
    return str(dst.resolve())

wiki_best_label = "e1"
wiki_best_path = str(Path(wiki_e1_path).resolve())
if str(wiki_w3_path or "").strip() and Path(wiki_w3_path).exists():
    wiki_best_label = "w3"
    wiki_best_path = str(Path(wiki_w3_path).resolve())

adjusted_hotpot_best = _clone_with_exec_embedding_limits(hotpot_f2_path, "hotpot_f2_adjusted.json")
adjusted_wiki_best = _clone_with_exec_embedding_limits(wiki_best_path, f"wiki_{wiki_best_label}_adjusted.json")

payload = {
    "hotpotqa": {
        "d1": {"label": "D1", "path": str(Path(hotpot_d1_path).resolve())},
        "best_non_oracle": {"label": "F2", "path": adjusted_hotpot_best},
        "d2": {"label": "D2", "path": str(Path(d2_hotpot_path).resolve())},
        "hipporag2": {"label": "HippoRAG2", "path": str(Path(hippo_hotpot_path).resolve())},
    },
    "2wikimultihopqa": {
        "d1": {"label": "D1", "path": str(Path(wiki_d1_path).resolve())},
        "best_non_oracle": {"label": wiki_best_label.upper(), "path": adjusted_wiki_best},
        "d2": {"label": "D2", "path": str(Path(d2_wiki_path).resolve())},
        "hipporag2": {"label": "HippoRAG2", "path": str(Path(hippo_wiki_path).resolve())},
    },
    "wiki_best_label": wiki_best_label,
    "stage_l2_inputs": {
        "reference_hotpot_best_path": str(Path(hotpot_f2_path).resolve()),
        "reference_wiki_best_path": str(Path(wiki_best_path).resolve()),
        "execution_embedding_max_length": int(exec_embedding_max_length),
        "execution_embedding_text_max_chars": int(exec_embedding_text_max_chars),
    },
}
Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(out_json)
PY

resolve_vllm_max_model_len

echo "[Stage L2] Run length audit only"
set +e
"${PYTHON_BIN}" scripts/length_audit_frozen.py \
  --ref-metrics-json "${REF_METRICS_JSON}" \
  --output-json "${LENGTH_AUDIT_JSON}" \
  --output-md "${LENGTH_AUDIT_MD}" \
  --vllm-max-model-len "${VLLM_MAX_MODEL_LEN_RESOLVED}" \
  --vllm-max-model-len-source "${VLLM_MAX_MODEL_LEN_SOURCE}" \
  --prompt-samples-per-dataset "${PROMPT_AUDIT_SAMPLES_PER_DATASET}"
AUDIT_EXIT=$?
set -e

echo "[Stage L2] Summarize closure status"
"${PYTHON_BIN}" - \
  "${LENGTH_AUDIT_JSON}" \
  "${REF_METRICS_JSON}" \
  "${CLOSURE_JSON}" \
  "${CLOSURE_MD}" \
  "${TOKEN_CLIP_EPS}" \
  "${CHAR_CLIP_EPS}" \
  "${PROMPT_OVER_EPS}" <<'PY'
import json
import sys
from pathlib import Path

from effirag.embedding import get_embedding_model_length_info
from effirag.utils import evaluate_clipping_closure

(
    audit_json,
    ref_json,
    out_json,
    out_md,
    token_eps,
    char_eps,
    prompt_eps,
) = sys.argv[1:8]

audit = json.loads(Path(audit_json).read_text(encoding="utf-8"))
refs = json.loads(Path(ref_json).read_text(encoding="utf-8"))
l5 = dict((audit.get("l5_summary", {}) or {}))

exec_cfg = dict((refs.get("stage_l2_inputs", {}) or {}))
embedding_model_name = str((audit.get("l2_embedding_audit", {}) or {}).get("embedding_model_name", "nvidia/NV-Embed-v2"))
model_limits = get_embedding_model_length_info(embedding_model_name)

closure = evaluate_clipping_closure(
    embedding_token_clipped_rate=float(l5.get("embedding_token_clipped_rate", 0.0) or 0.0),
    embedding_char_clipped_rate=float(l5.get("embedding_char_clipped_rate", 0.0) or 0.0),
    prompt_over_limit_rate=float(l5.get("prompt_over_limit_rate", 0.0) or 0.0),
    token_eps=float(token_eps),
    char_eps=float(char_eps),
    prompt_eps=float(prompt_eps),
)

clipped_examples = list((audit.get("l2_embedding_audit", {}) or {}).get("clipped_examples", []) or [])
char_examples = [e for e in clipped_examples if str((e or {}).get("clip_type", "")) == "char"]
token_examples = [e for e in clipped_examples if str((e or {}).get("clip_type", "")) == "token"]
by_dataset = ((audit.get("l2_embedding_audit", {}) or {}).get("by_dataset", {}) or {})
for ds_name, ds_payload in by_dataset.items():
    for kind in ("entity", "chunk"):
        entries = list((((ds_payload or {}).get(kind, {}) or {}).get("examples", []) or []))
        for ex in entries:
            item = dict(ex or {})
            item["dataset"] = ds_name
            if str(item.get("clip_type", "")) == "char":
                char_examples.append(item)
            elif str(item.get("clip_type", "")) == "token":
                token_examples.append(item)

payload = {
    "length_audit_pass_raw": bool(audit.get("length_audit_pass", False)),
    "length_audit_pass_l2": bool(closure.get("pass", False)),
    "closure_status": closure,
    "stage_l2_inputs": exec_cfg,
    "embedding_model_limits": model_limits,
    "key_metrics": {
        "embedding_token_clipped_rate": float(l5.get("embedding_token_clipped_rate", 0.0) or 0.0),
        "embedding_char_clipped_rate": float(l5.get("embedding_char_clipped_rate", 0.0) or 0.0),
        "prompt_over_limit_rate": float(l5.get("prompt_over_limit_rate", 0.0) or 0.0),
        "prompt_token_p95": float(l5.get("prompt_token_p95", 0.0) or 0.0),
        "prompt_token_p99": float(l5.get("prompt_token_p99", 0.0) or 0.0),
        "prompt_token_max": float(l5.get("prompt_token_max", 0.0) or 0.0),
    },
    "clipping_examples": {
        "char_example_count": int(len(char_examples)),
        "token_example_count": int(len(token_examples)),
        "char_examples": char_examples[:20],
        "token_examples": token_examples[:20],
    },
    "recommended_final_settings": {
        "embedding_max_length": int(exec_cfg.get("execution_embedding_max_length", 320) or 320),
        "embedding_text_max_chars": int(exec_cfg.get("execution_embedding_text_max_chars", 900) or 900),
    },
}

Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# Stage L2 Embedding Clipping Closure",
    "",
    f"- length_audit_pass_raw: {payload['length_audit_pass_raw']}",
    f"- length_audit_pass_l2: {payload['length_audit_pass_l2']}",
    f"- embedding_token_clipped_rate: {payload['key_metrics']['embedding_token_clipped_rate']:.6f}",
    f"- embedding_char_clipped_rate: {payload['key_metrics']['embedding_char_clipped_rate']:.6f}",
    f"- prompt_over_limit_rate: {payload['key_metrics']['prompt_over_limit_rate']:.6f}",
    f"- token_eps: {closure['token_eps']}",
    f"- char_eps: {closure['char_eps']}",
    f"- prompt_eps: {closure['prompt_eps']}",
    "",
    "## Recommended Final Settings",
    f"- embedding_max_length: {payload['recommended_final_settings']['embedding_max_length']}",
    f"- embedding_text_max_chars: {payload['recommended_final_settings']['embedding_text_max_chars']}",
    "",
    "## Embedding Model Limits",
    f"- model_name: {model_limits.get('model_name', '')}",
    f"- tokenizer_model_max_length: {model_limits.get('tokenizer_model_max_length', 0)}",
    f"- tokenizer_model_max_length_raw: {model_limits.get('tokenizer_model_max_length_raw', 0)}",
    f"- model_max_seq_length: {model_limits.get('model_max_seq_length', 0)}",
]
Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("PASS" if payload["length_audit_pass_l2"] else "FAIL")
raise SystemExit(0 if payload["length_audit_pass_l2"] else 5)
PY

echo "[DONE] outputs: ${ROUND_ROOT}"
echo "[DONE] stage_l2 audit: ${LENGTH_AUDIT_JSON}"
echo "[DONE] stage_l2 closure: ${CLOSURE_JSON}"

# Do not execute Stage P/Q in this script.
# Exit with Stage L2 closure status.
if [[ -f "${CLOSURE_JSON}" ]]; then
  PASS="$(${PYTHON_BIN} - <<'PY' "${CLOSURE_JSON}"
import json,sys
obj=json.loads(open(sys.argv[1], 'r', encoding='utf-8').read())
print('1' if bool(obj.get('length_audit_pass_l2', False)) else '0')
PY
)"
  if [[ "${PASS}" == "1" ]]; then
    exit 0
  fi
fi

exit 5
