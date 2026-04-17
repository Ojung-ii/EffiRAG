#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
import re
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from effirag.global_index import _collect_semantic_records, _split_passage_chunks, load_corpus_rows
from effirag.registry import get_dataset_loader, register_defaults
from effirag.render import render_context
from effirag.run_rag import _reconstruct_retrieval


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _percentile(values: List[float], q: float) -> float:
    seq = sorted([float(v) for v in values])
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * float(len(seq) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(seq[lo])
    frac = pos - float(lo)
    return float(seq[lo] * (1.0 - frac) + seq[hi] * frac)


def _summary(values: List[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(len(vals)),
        "mean": float(sum(vals) / float(len(vals))),
        "p50": float(_percentile(vals, 0.50)),
        "p95": float(_percentile(vals, 0.95)),
        "p99": float(_percentile(vals, 0.99)),
        "max": float(max(vals)),
    }


def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        return None, f"transformers_unavailable: {exc}"

    # Keep audit deterministic/offline-first: do not retry through network when
    # the environment has no internet access.
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        return tok, "ok"
    except Exception as exc:
        return None, f"tokenizer_load_failed_local_only: {exc}"


def _render_from_params(sample, retrieval, render_params: Dict[str, Any], render_mode: str):
    """
    Call render_context with only supported kwargs for the current codebase.
    This avoids silent zero-sample audits when render signature drifts.
    """
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


def _token_lengths(tokenizer, texts: List[str], add_special_tokens: bool, batch_size: int = 128) -> List[int]:
    out = []
    if tokenizer is None:
        return out
    bs = max(1, int(batch_size))
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        try:
            enc = tokenizer(
                batch,
                add_special_tokens=bool(add_special_tokens),
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            ids = []
            if isinstance(enc, dict):
                ids = enc.get("input_ids", []) or []
            elif hasattr(enc, "get"):
                try:
                    ids = enc.get("input_ids", []) or []
                except Exception:
                    ids = []
            elif hasattr(enc, "input_ids"):
                ids = getattr(enc, "input_ids", []) or []
            out.extend([len(x) for x in ids])
        except Exception:
            # Fallback: rough estimate if tokenizer call fails on a batch.
            out.extend([max(1, int(len(t) / 4)) for t in batch])
    return out


def _find_index_cache_dir(dataset: str, embedding_model_name: str) -> Tuple[str, Dict[str, Any]]:
    root = Path("outputs/index_cache")
    candidates = []
    if not root.exists():
        return "", {}

    prefix = f"{dataset}_corpus_"
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        meta_path = d / "semantic_index_meta.json"
        graph_path = d / "graph.gpickle"
        if not meta_path.exists() or not graph_path.exists():
            continue
        try:
            meta = _read_json(meta_path)
        except Exception:
            continue
        sbc = dict(meta.get("semantic_build_config", {}) or {})
        score = (
            1 if str(meta.get("model_name", "")) == str(embedding_model_name or "") else 0,
            1 if str(sbc.get("graph_mode", "")) == "entity_chunk_graph" else 0,
            1 if str(sbc.get("index_chunk_unit", "")) == "passage" else 0,
            _safe_int(meta.get("entity_count", 0), 0) + _safe_int(meta.get("chunk_count", 0), 0),
            _safe_float(meta_path.stat().st_mtime, 0.0),
        )
        candidates.append((score, str(d), meta))

    if not candidates:
        return "", {}

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, path, meta = candidates[0]
    return path, meta


def _clip_text(text: str, max_chars: int) -> str:
    s = str(text or "").strip()
    if int(max_chars) <= 0:
        return s
    return s[: int(max_chars)]


def _audit_embedding_for_dataset(
    dataset: str,
    index_dir: str,
    embedding_max_length: int,
    embedding_text_max_chars: int,
    tokenizer,
    text_limit: int,
) -> Dict[str, Any]:
    if not index_dir:
        return {
            "dataset": dataset,
            "index_dir": "",
            "error": "index_cache_not_found",
        }

    gpath = Path(index_dir) / "graph.gpickle"
    with gpath.open("rb") as f:
        graph = pickle.load(f)

    entity_records, chunk_records = _collect_semantic_records(graph)
    entity_texts = [str(r.get("text", "") or "") for r in entity_records if str(r.get("text", "") or "").strip()]
    chunk_texts = [str(r.get("text", "") or "") for r in chunk_records if str(r.get("text", "") or "").strip()]

    if int(text_limit) > 0:
        entity_texts = entity_texts[: int(text_limit)]
        chunk_texts = chunk_texts[: int(text_limit)]

    def _audit_texts(texts: List[str], kind: str) -> Dict[str, Any]:
        raw_char_lens = [len(t) for t in texts]
        clipped_texts = [_clip_text(t, embedding_text_max_chars) for t in texts]
        clipped_char_lens = [len(t) for t in clipped_texts]
        char_clipped_idx = [i for i, t in enumerate(texts) if len(t) > int(embedding_text_max_chars)]

        tok_lens = _token_lengths(tokenizer, clipped_texts, add_special_tokens=False, batch_size=128)
        token_clipped_idx = [i for i, n in enumerate(tok_lens) if int(n) > int(embedding_max_length)]

        examples = []
        for i in char_clipped_idx[:10]:
            examples.append(
                {
                    "kind": kind,
                    "clip_type": "char",
                    "raw_chars": int(len(texts[i])),
                    "clipped_chars": int(len(clipped_texts[i])),
                    "tokens_after_char_clip": int(tok_lens[i]) if i < len(tok_lens) else 0,
                    "preview": str(texts[i])[:220],
                }
            )
        for i in token_clipped_idx[:10]:
            examples.append(
                {
                    "kind": kind,
                    "clip_type": "token",
                    "raw_chars": int(len(texts[i])),
                    "clipped_chars": int(len(clipped_texts[i])),
                    "tokens_after_char_clip": int(tok_lens[i]),
                    "preview": str(texts[i])[:220],
                }
            )

        return {
            "count": int(len(texts)),
            "raw_char_summary": _summary(raw_char_lens),
            "clipped_char_summary": _summary(clipped_char_lens),
            "token_summary": _summary(tok_lens),
            "char_clipped_rate": float(len(char_clipped_idx) / len(texts)) if texts else 0.0,
            "token_clipped_rate": float(len(token_clipped_idx) / len(texts)) if texts else 0.0,
            "examples": examples[:20],
        }

    ent = _audit_texts(entity_texts, "entity")
    chk = _audit_texts(chunk_texts, "chunk")

    total_n = int(ent["count"] + chk["count"])
    agg_char_clip = float(
        (ent["char_clipped_rate"] * ent["count"] + chk["char_clipped_rate"] * chk["count"]) / total_n
    ) if total_n > 0 else 0.0
    agg_tok_clip = float(
        (ent["token_clipped_rate"] * ent["count"] + chk["token_clipped_rate"] * chk["count"]) / total_n
    ) if total_n > 0 else 0.0

    return {
        "dataset": dataset,
        "index_dir": index_dir,
        "embedding_max_length": int(embedding_max_length),
        "embedding_text_max_chars": int(embedding_text_max_chars),
        "entity": ent,
        "chunk": chk,
        "aggregate": {
            "count": total_n,
            "char_clipped_rate": agg_char_clip,
            "token_clipped_rate": agg_tok_clip,
        },
    }


def _audit_openie_for_dataset(
    dataset: str,
    corpus_path: str,
    openie_text_max_chars: int,
    chunking_cfg: Dict[str, Any],
    chunk_limit: int,
) -> Dict[str, Any]:
    rows = load_corpus_rows(corpus_path, show_progress=False)
    texts = []
    for row in rows:
        passage = str(row.get("text", "") or "")
        chunks = _split_passage_chunks(
            text=passage,
            strategy=str(chunking_cfg.get("passage_chunking_strategy", "sentence_window") or "sentence_window"),
            size_sentences=_safe_int(chunking_cfg.get("passage_chunk_size_sentences", 3), 3),
            stride_sentences=_safe_int(chunking_cfg.get("passage_chunk_stride_sentences", 2), 2),
            min_sentences=_safe_int(chunking_cfg.get("passage_chunk_min_sentences", 2), 2),
            max_chars=_safe_int(chunking_cfg.get("passage_chunk_max_chars", 900), 900),
        )
        texts.extend(chunks)
        if int(chunk_limit) > 0 and len(texts) >= int(chunk_limit):
            texts = texts[: int(chunk_limit)]
            break

    char_lens = [len(t) for t in texts]
    clipped = [i for i, t in enumerate(texts) if len(t) > int(openie_text_max_chars)]
    examples = [
        {
            "chars": int(len(texts[i])),
            "limit": int(openie_text_max_chars),
            "preview": str(texts[i])[:220],
        }
        for i in clipped[:20]
    ]

    return {
        "dataset": dataset,
        "corpus_path": corpus_path,
        "openie_text_max_chars": int(openie_text_max_chars),
        "count": int(len(texts)),
        "char_summary": _summary(char_lens),
        "char_clipped_rate": float(len(clipped) / len(texts)) if texts else 0.0,
        "examples": examples,
    }


def _build_prompt_text(question: str, context: str) -> str:
    return (
        "You are a QA assistant.\n"
        "Use only the provided context.\n"
        "Return only the final answer span.\n"
        "Do not output reasoning, explanations, or <think> tags.\n"
        "If the question is yes/no, output exactly yes or no.\n"
        f"Question: {question}\n"
        f"Context:\n{context}\n"
        "Final answer:"
    )


def _token_count_from_encoding(enc: Any) -> int:
    ids = []
    if isinstance(enc, dict):
        ids = enc.get("input_ids", []) or []
    elif hasattr(enc, "get"):
        try:
            ids = enc.get("input_ids", []) or []
        except Exception:
            ids = []
    elif hasattr(enc, "input_ids"):
        ids = getattr(enc, "input_ids", []) or []

    # HF tokenizers may return [ids] for single-item batch; flatten once.
    if isinstance(ids, (list, tuple)) and ids:
        first = ids[0]
        if isinstance(first, (list, tuple)):
            return int(len(first))
        return int(len(ids))
    try:
        # tensor-like fallback
        shape = tuple(getattr(ids, "shape", ()))
        if len(shape) == 2:
            return int(shape[1])
        if len(shape) == 1:
            return int(shape[0])
    except Exception:
        pass
    return 0


def _audit_prompt_lengths(
    dataset: str,
    ref_summary_path: str,
    prompt_tokenizer,
    prompt_sample_limit: int,
    max_model_len: int,
    default_max_new_tokens: int,
) -> Dict[str, Any]:
    summary = _read_json(Path(ref_summary_path))
    query_path = Path(ref_summary_path).with_name("rag_query_results.jsonl")
    rows = _read_jsonl(query_path)

    register_defaults()
    loader = get_dataset_loader(dataset)
    samples = loader(split="validation", limit=None, data_path=f"data/qa/{dataset}.json")
    sample_by_qid = {str(s.qid): s for s in samples}

    render_params = dict(summary.get("render_params", {}) or {})
    render_mode = str(summary.get("render_mode_resolved") or summary.get("render_mode_requested") or "flat")
    max_new_tokens = _safe_int((summary.get("generation_params", {}) or {}).get("llm_max_new_tokens", default_max_new_tokens), default_max_new_tokens)

    token_lengths = []
    over_limit = []
    truncated_render = []

    lim = max(1, int(prompt_sample_limit))
    seen = 0
    for row in rows:
        if seen >= lim:
            break
        qid = str(row.get("sample_id", "") or "")
        sample = sample_by_qid.get(qid)
        if sample is None:
            continue
        retrieval_payload = dict(row.get("retrieval", {}) or {})
        if not retrieval_payload:
            continue

        try:
            retrieval = _reconstruct_retrieval(retrieval_payload)
            rendered = _render_from_params(
                sample=sample,
                retrieval=retrieval,
                render_params=render_params,
                render_mode=render_mode,
            )
        except Exception:
            continue

        prompt = _build_prompt_text(sample.question, rendered.text)
        if prompt_tokenizer is not None:
            try:
                toks = prompt_tokenizer(prompt, add_special_tokens=True, truncation=False, return_attention_mask=False)
                plen = _token_count_from_encoding(toks)
                if plen <= 0:
                    plen = max(1, int(len(prompt) / 4))
            except Exception:
                plen = max(1, int(len(prompt) / 4))
        else:
            plen = max(1, int(len(prompt) / 4))

        token_lengths.append(int(plen))
        total = int(plen + max_new_tokens)
        if total > int(max_model_len):
            over_limit.append(
                {
                    "sample_id": qid,
                    "prompt_tokens": int(plen),
                    "max_new_tokens": int(max_new_tokens),
                    "max_model_len": int(max_model_len),
                    "over_by": int(total - int(max_model_len)),
                    "prompt_preview": prompt[:280],
                }
            )
        if bool(rendered.truncated):
            truncated_render.append(
                {
                    "sample_id": qid,
                    "truncated_corridor_count": int(getattr(rendered, "truncated_corridor_count", 0) or 0),
                    "truncated_sentence_count": int(getattr(rendered, "truncated_sentence_count", 0) or 0),
                }
            )
        seen += 1

    return {
        "dataset": dataset,
        "reference_summary_path": ref_summary_path,
        "reference_query_results_path": str(query_path),
        "sample_count": int(seen),
        "max_new_tokens": int(max_new_tokens),
        "max_model_len": int(max_model_len),
        "prompt_token_summary": _summary(token_lengths),
        "prompt_over_limit_rate": float(len(over_limit) / seen) if seen > 0 else 0.0,
        "over_limit_examples": over_limit[:20],
        "render_truncation_rate": float(len(truncated_render) / seen) if seen > 0 else 0.0,
        "render_truncation_examples": truncated_render[:20],
    }


def _extract_line_refs(path: str, patterns: List[str]) -> Dict[str, List[int]]:
    p = Path(path)
    if not p.exists():
        return {}
    lines = p.read_text(encoding="utf-8").splitlines()
    out = {}
    for pat in patterns:
        regex = re.compile(pat)
        hits = []
        for i, line in enumerate(lines, start=1):
            if regex.search(line):
                hits.append(i)
                if len(hits) >= 5:
                    break
        out[pat] = hits
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-metrics-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--vllm-max-model-len", type=int, required=True)
    parser.add_argument("--vllm-max-model-len-source", type=str, default="")
    parser.add_argument("--prompt-samples-per-dataset", type=int, default=200)
    parser.add_argument("--embedding-text-limit", type=int, default=0)
    parser.add_argument("--openie-chunk-limit", type=int, default=0)
    parser.add_argument("--severe-char-clip-threshold", type=float, default=0.01)
    args = parser.parse_args()

    refs = _read_json(Path(args.ref_metrics_json))

    hotpot_d1 = _read_json(Path(refs["hotpotqa"]["d1"]["path"]))
    hotpot_f2 = _read_json(Path(refs["hotpotqa"]["best_non_oracle"]["path"]))
    wiki_d1 = _read_json(Path(refs["2wikimultihopqa"]["d1"]["path"]))
    wiki_best = _read_json(Path(refs["2wikimultihopqa"]["best_non_oracle"]["path"]))

    # Prefer non-oracle frozen references for run-time configs.
    run_ref = {
        "hotpotqa": hotpot_f2,
        "2wikimultihopqa": wiki_best,
    }

    # L1: config/code audit.
    rag_cfg = yaml.safe_load(Path("configs/canonical/rag_entity_first_chunk_grounded.yaml").read_text(encoding="utf-8")) or {}
    retr_cfg = yaml.safe_load(Path("configs/canonical/retrieval_entity_first_chunk_grounded.yaml").read_text(encoding="utf-8")) or {}

    l1 = {
        "reference_values": {
            ds: {
                "embedding_text_max_chars": _safe_int((run_ref[ds].get("retrieval_params", {}) or {}).get("embedding_text_max_chars", 600), 600),
                "embedding_max_length": _safe_int((run_ref[ds].get("retrieval_params", {}) or {}).get("embedding_max_length", 192), 192),
                "openie_text_max_chars": _safe_int((run_ref[ds].get("retrieval_params", {}) or {}).get("openie_text_max_chars", 2200), 2200),
                "openie_max_new_tokens": _safe_int((run_ref[ds].get("retrieval_params", {}) or {}).get("openie_max_new_tokens", 256), 256),
                "max_new_tokens": _safe_int((run_ref[ds].get("generation_params", {}) or {}).get("llm_max_new_tokens", 64), 64),
                "max_context_sentences": _safe_int((run_ref[ds].get("render_params", {}) or {}).get("max_context_sentences", 8), 8),
                "max_corridors_in_context": _safe_int((run_ref[ds].get("render_params", {}) or {}).get("max_corridors_in_context", 3), 3),
            }
            for ds in ("hotpotqa", "2wikimultihopqa")
        },
        "canonical_config_values": {
            "rag": {
                "embedding_text_max_chars": retr_cfg.get("embedding_text_max_chars"),
                "embedding_max_length": retr_cfg.get("embedding_max_length"),
                "openie_text_max_chars": retr_cfg.get("openie_text_max_chars"),
                "openie_max_new_tokens": retr_cfg.get("openie_max_new_tokens"),
                "llm_max_new_tokens": rag_cfg.get("llm_max_new_tokens"),
                "max_context_sentences": rag_cfg.get("max_context_sentences"),
                "max_corridors_in_context": rag_cfg.get("max_corridors_in_context"),
            },
        },
        "code_locations": {
            "effirag/embedding.py": _extract_line_refs(
                "effirag/embedding.py",
                [r"max_length", r"max_chars", r"truncation=True", r"AutoTokenizer"],
            ),
            "effirag/render.py": _extract_line_refs(
                "effirag/render.py",
                [r"truncated_corridor", r"truncated_sentence", r"max_context_sentences", r"max_corridors_in_context"],
            ),
            "effirag/generator.py": _extract_line_refs(
                "effirag/generator.py",
                [r"llm_max_new_tokens", r"Context:", r"Final answer"],
            ),
            "effirag/global_index.py": _extract_line_refs(
                "effirag/global_index.py",
                [r"openie_text_max_chars", r"openie_max_new_tokens", r"_split_passage_chunks", r"embedding_max_length", r"embedding_text_max_chars"],
            ),
        },
        "vllm_max_model_len": {
            "value": int(args.vllm_max_model_len),
            "source": str(args.vllm_max_model_len_source or "unknown"),
        },
    }

    # Tokenizers.
    embedding_model_name = str((run_ref["hotpotqa"].get("retrieval_params", {}) or {}).get("embedding_model_name", "nvidia/NV-Embed-v2"))
    prompt_model_name = str((run_ref["hotpotqa"].get("generation_params", {}) or {}).get("model_name", "Qwen/Qwen2.5-7B-Instruct"))

    emb_tok, emb_tok_status = _load_tokenizer(embedding_model_name)
    prompt_tok, prompt_tok_status = _load_tokenizer(prompt_model_name)

    # L2 embedding audit.
    embedding_by_dataset = {}
    all_entity_token_lens = []
    all_chunk_token_lens = []
    total_count = 0
    total_char_clip = 0.0
    total_tok_clip = 0.0
    all_examples = []

    for ds in ("hotpotqa", "2wikimultihopqa"):
        rp = dict((run_ref[ds].get("retrieval_params", {}) or {}))
        index_dir, index_meta = _find_index_cache_dir(ds, embedding_model_name)
        one = _audit_embedding_for_dataset(
            dataset=ds,
            index_dir=index_dir,
            embedding_max_length=_safe_int(rp.get("embedding_max_length", 192), 192),
            embedding_text_max_chars=_safe_int(rp.get("embedding_text_max_chars", 600), 600),
            tokenizer=emb_tok,
            text_limit=int(args.embedding_text_limit),
        )
        one["index_meta"] = index_meta
        embedding_by_dataset[ds] = one

        if "entity" in one and "chunk" in one:
            ent_tok_p = one["entity"]["token_summary"]
            chk_tok_p = one["chunk"]["token_summary"]
            # store approximate samples for global distribution by repeating p99 values is misleading;
            # instead we aggregate using rates/count and per-dataset summaries for pass criteria.
            total_count += int(one["aggregate"]["count"])
            total_char_clip += float(one["aggregate"]["char_clipped_rate"]) * int(one["aggregate"]["count"])
            total_tok_clip += float(one["aggregate"]["token_clipped_rate"]) * int(one["aggregate"]["count"])
            all_examples.extend((one["entity"].get("examples") or [])[:10])
            all_examples.extend((one["chunk"].get("examples") or [])[:10])

    # For global p95/p99/max we merge per-dataset statistics conservatively via max of dataset p-values.
    entity_p95 = max(
        _safe_float(embedding_by_dataset.get(ds, {}).get("entity", {}).get("token_summary", {}).get("p95", 0.0), 0.0)
        for ds in ("hotpotqa", "2wikimultihopqa")
    )
    entity_p99 = max(
        _safe_float(embedding_by_dataset.get(ds, {}).get("entity", {}).get("token_summary", {}).get("p99", 0.0), 0.0)
        for ds in ("hotpotqa", "2wikimultihopqa")
    )
    entity_max = max(
        _safe_float(embedding_by_dataset.get(ds, {}).get("entity", {}).get("token_summary", {}).get("max", 0.0), 0.0)
        for ds in ("hotpotqa", "2wikimultihopqa")
    )
    chunk_p95 = max(
        _safe_float(embedding_by_dataset.get(ds, {}).get("chunk", {}).get("token_summary", {}).get("p95", 0.0), 0.0)
        for ds in ("hotpotqa", "2wikimultihopqa")
    )
    chunk_p99 = max(
        _safe_float(embedding_by_dataset.get(ds, {}).get("chunk", {}).get("token_summary", {}).get("p99", 0.0), 0.0)
        for ds in ("hotpotqa", "2wikimultihopqa")
    )
    chunk_max = max(
        _safe_float(embedding_by_dataset.get(ds, {}).get("chunk", {}).get("token_summary", {}).get("max", 0.0), 0.0)
        for ds in ("hotpotqa", "2wikimultihopqa")
    )

    embedding_char_clipped_rate = float(total_char_clip / total_count) if total_count > 0 else 0.0
    embedding_token_clipped_rate = float(total_tok_clip / total_count) if total_count > 0 else 0.0

    l2 = {
        "embedding_model_name": embedding_model_name,
        "tokenizer_status": emb_tok_status,
        "by_dataset": embedding_by_dataset,
        "embedding_entity_token_p95": entity_p95,
        "embedding_entity_token_p99": entity_p99,
        "embedding_entity_token_max": entity_max,
        "embedding_chunk_token_p95": chunk_p95,
        "embedding_chunk_token_p99": chunk_p99,
        "embedding_chunk_token_max": chunk_max,
        "embedding_char_clipped_rate": embedding_char_clipped_rate,
        "embedding_token_clipped_rate": embedding_token_clipped_rate,
        "clipped_examples": all_examples[:20],
    }

    # L3 OpenIE audit.
    openie_by_dataset = {}
    total_openie_n = 0
    total_openie_clip = 0.0
    for ds in ("hotpotqa", "2wikimultihopqa"):
        rp = dict((run_ref[ds].get("retrieval_params", {}) or {}))
        chunk_cfg = {
            "passage_chunking_strategy": rp.get("passage_chunking_strategy", "sentence_window"),
            "passage_chunk_size_sentences": rp.get("passage_chunk_size_sentences", 3),
            "passage_chunk_stride_sentences": rp.get("passage_chunk_stride_sentences", 2),
            "passage_chunk_min_sentences": rp.get("passage_chunk_min_sentences", 2),
            "passage_chunk_max_chars": rp.get("passage_chunk_max_chars", 900),
        }
        corpus_path = f"data/{ds}_corpus.json"
        one = _audit_openie_for_dataset(
            dataset=ds,
            corpus_path=corpus_path,
            openie_text_max_chars=_safe_int(rp.get("openie_text_max_chars", 2200), 2200),
            chunking_cfg=chunk_cfg,
            chunk_limit=int(args.openie_chunk_limit),
        )
        openie_by_dataset[ds] = one
        total_openie_n += int(one["count"])
        total_openie_clip += float(one["char_clipped_rate"]) * int(one["count"])

    openie_char_clipped_rate = float(total_openie_clip / total_openie_n) if total_openie_n > 0 else 0.0
    l3 = {
        "by_dataset": openie_by_dataset,
        "openie_char_clipped_rate": openie_char_clipped_rate,
    }

    # L4 Prompt audit using frozen reference query results.
    prompt_by_dataset = {}
    prompt_max_new_tokens = 0
    all_prompt_tokens = []
    over_num = 0
    over_den = 0
    over_examples = []

    for ds in ("hotpotqa", "2wikimultihopqa"):
        ref_path = str(refs[ds]["best_non_oracle"]["path"])
        one = _audit_prompt_lengths(
            dataset=ds,
            ref_summary_path=ref_path,
            prompt_tokenizer=prompt_tok,
            prompt_sample_limit=int(args.prompt_samples_per_dataset),
            max_model_len=int(args.vllm_max_model_len),
            default_max_new_tokens=64,
        )
        prompt_by_dataset[ds] = one
        prompt_max_new_tokens = max(prompt_max_new_tokens, int(one.get("max_new_tokens", 0)))
        s = one.get("prompt_token_summary", {}) or {}
        # conservative global p-values via max across dataset-level p-values
        all_prompt_tokens.append(float(s.get("p95", 0.0) or 0.0))
        all_prompt_tokens.append(float(s.get("p99", 0.0) or 0.0))
        all_prompt_tokens.append(float(s.get("max", 0.0) or 0.0))
        over_num += len(one.get("over_limit_examples", []) or [])
        over_den += int(one.get("sample_count", 0) or 0)
        over_examples.extend((one.get("over_limit_examples") or [])[:10])

    prompt_p95 = max(
        _safe_float(prompt_by_dataset.get(ds, {}).get("prompt_token_summary", {}).get("p95", 0.0), 0.0)
        for ds in ("hotpotqa", "2wikimultihopqa")
    )
    prompt_p99 = max(
        _safe_float(prompt_by_dataset.get(ds, {}).get("prompt_token_summary", {}).get("p99", 0.0), 0.0)
        for ds in ("hotpotqa", "2wikimultihopqa")
    )
    prompt_max = max(
        _safe_float(prompt_by_dataset.get(ds, {}).get("prompt_token_summary", {}).get("max", 0.0), 0.0)
        for ds in ("hotpotqa", "2wikimultihopqa")
    )
    prompt_over_limit_rate = float(over_num / over_den) if over_den > 0 else 0.0

    l4 = {
        "prompt_model_name": prompt_model_name,
        "tokenizer_status": prompt_tok_status,
        "by_dataset": prompt_by_dataset,
        "prompt_token_p95": prompt_p95,
        "prompt_token_p99": prompt_p99,
        "prompt_token_max": prompt_max,
        "prompt_over_limit_rate": prompt_over_limit_rate,
        "max_new_tokens": int(prompt_max_new_tokens),
        "max_model_len": int(args.vllm_max_model_len),
        "over_limit_examples": over_examples[:20],
    }

    # Stage L pass criteria.
    token_clip_ok = float(embedding_token_clipped_rate) <= 1.0e-6
    prompt_over_ok = float(prompt_over_limit_rate) <= 1.0e-12
    prompt_budget_ok = (float(prompt_p99) + float(prompt_max_new_tokens)) <= (0.9 * float(args.vllm_max_model_len))
    severe_char_clip = (float(embedding_char_clipped_rate) > float(args.severe_char_clip_threshold)) or (
        float(openie_char_clipped_rate) > float(args.severe_char_clip_threshold)
    )

    length_audit_pass = bool(token_clip_ok and prompt_over_ok and prompt_budget_ok and (not severe_char_clip))

    recommendations = []
    if not token_clip_ok:
        needed = int(max(entity_max, chunk_max))
        if needed <= 0:
            needed = int(max(entity_p99, chunk_p99))
        recommendations.append(
            {
                "issue": "embedding_token_clipping",
                "recommendation": f"increase embedding_max_length >= {needed}",
            }
        )
    if float(embedding_char_clipped_rate) > 0.0:
        recommendations.append(
            {
                "issue": "embedding_char_clipping",
                "recommendation": "increase embedding_text_max_chars or reduce text construction length (entity context/chunk payload)",
            }
        )
    if float(openie_char_clipped_rate) > 0.0:
        recommendations.append(
            {
                "issue": "openie_char_clipping",
                "recommendation": "increase openie_text_max_chars or reduce upstream passage chunk length",
            }
        )
    if not prompt_over_ok or not prompt_budget_ok:
        recommendations.append(
            {
                "issue": "prompt_length_risk",
                "recommendation": "reduce max_context_sentences or max_total_sentences, and/or lower llm_max_new_tokens or raise vLLM max_model_len",
            }
        )

    payload = {
        "length_audit_pass": bool(length_audit_pass),
        "criteria": {
            "embedding_token_clipped_rate_target": "0 or effectively 0",
            "prompt_over_limit_rate_target": "0",
            "prompt_p99_plus_max_new_tokens_leq": "0.9 * max_model_len",
            "severe_char_clip_threshold": float(args.severe_char_clip_threshold),
        },
        "l1_config_code_audit": l1,
        "l2_embedding_audit": l2,
        "l3_openie_audit": l3,
        "l4_prompt_audit": l4,
        "l5_summary": {
            "embedding_entity_token_p95": entity_p95,
            "embedding_entity_token_p99": entity_p99,
            "embedding_entity_token_max": entity_max,
            "embedding_chunk_token_p95": chunk_p95,
            "embedding_chunk_token_p99": chunk_p99,
            "embedding_chunk_token_max": chunk_max,
            "embedding_char_clipped_rate": embedding_char_clipped_rate,
            "embedding_token_clipped_rate": embedding_token_clipped_rate,
            "openie_char_clipped_rate": openie_char_clipped_rate,
            "prompt_token_p95": prompt_p95,
            "prompt_token_p99": prompt_p99,
            "prompt_token_max": prompt_max,
            "prompt_over_limit_rate": prompt_over_limit_rate,
            "max_model_len": int(args.vllm_max_model_len),
            "max_model_len_source": str(args.vllm_max_model_len_source or "unknown"),
            "max_new_tokens": int(prompt_max_new_tokens),
            "length_audit_pass": bool(length_audit_pass),
            "recommendations": recommendations,
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stage L Length Audit",
        "",
        f"- length_audit_pass: {bool(length_audit_pass)}",
        f"- embedding_token_clipped_rate: {embedding_token_clipped_rate:.6f}",
        f"- embedding_char_clipped_rate: {embedding_char_clipped_rate:.6f}",
        f"- openie_char_clipped_rate: {openie_char_clipped_rate:.6f}",
        f"- prompt_over_limit_rate: {prompt_over_limit_rate:.6f}",
        f"- prompt_p99 + max_new_tokens: {prompt_p99:.1f} + {int(prompt_max_new_tokens)} = {prompt_p99 + float(prompt_max_new_tokens):.1f}",
        f"- 0.9 * max_model_len: {0.9 * float(args.vllm_max_model_len):.1f}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| embedding_entity_token_p95 | {entity_p95:.2f} |",
        f"| embedding_entity_token_p99 | {entity_p99:.2f} |",
        f"| embedding_entity_token_max | {entity_max:.2f} |",
        f"| embedding_chunk_token_p95 | {chunk_p95:.2f} |",
        f"| embedding_chunk_token_p99 | {chunk_p99:.2f} |",
        f"| embedding_chunk_token_max | {chunk_max:.2f} |",
        f"| embedding_char_clipped_rate | {embedding_char_clipped_rate:.6f} |",
        f"| embedding_token_clipped_rate | {embedding_token_clipped_rate:.6f} |",
        f"| openie_char_clipped_rate | {openie_char_clipped_rate:.6f} |",
        f"| prompt_token_p95 | {prompt_p95:.2f} |",
        f"| prompt_token_p99 | {prompt_p99:.2f} |",
        f"| prompt_token_max | {prompt_max:.2f} |",
        f"| prompt_over_limit_rate | {prompt_over_limit_rate:.6f} |",
    ]

    if recommendations:
        lines.append("")
        lines.append("## Recommendations")
        for rec in recommendations:
            lines.append(f"- {rec['issue']}: {rec['recommendation']}")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("PASS" if length_audit_pass else "FAIL")
    return 0 if length_audit_pass else 5


if __name__ == "__main__":
    raise SystemExit(main())
