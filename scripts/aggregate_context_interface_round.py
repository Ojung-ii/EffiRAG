#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.eval_adapters import QASampleIndex, load_qa_index
from effirag.eval_metrics import (  # noqa: E402
    MetricRegistry,
    build_context_interface_abgf_metric_registry,
    metric_definition_notes,
    write_json,
)
from effirag.eval_reports import (  # noqa: E402
    build_flat_markdown_table,
    build_group_table_markdown,
    group_markdown_tables,
    markdown_document_from_group_tables,
    write_markdown,
)
from effirag.utils import safe_div  # noqa: E402

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text(value: Any) -> str:
    return " ".join(_strip(value).lower().split())


def _cmd(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(args), text=True).strip()
    except Exception:
        return ""


def _mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in list(values or [])]
    if not seq:
        return 0.0
    return float(sum(seq) / len(seq))


def _mode_text(values: Sequence[str], default: str = "") -> str:
    freq: Dict[str, int] = {}
    for raw in list(values or []):
        txt = _strip(raw)
        if not txt:
            continue
        freq[txt] = int(freq.get(txt, 0) + 1)
    if not freq:
        return str(default)
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


class TokenCounter:
    def __init__(self, model_name: str) -> None:
        self.model_name = str(model_name or "")
        self.mode = "tokenizer"
        self._cache: Dict[str, int] = {}
        self._tokenizer = None

        if AutoTokenizer is None:
            self.mode = "whitespace_fallback"
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            self._tokenizer = None
            self.mode = "whitespace_fallback"

    def count(self, text: Any) -> int:
        content = str(text or "")
        if not content:
            return 0
        if content in self._cache:
            return int(self._cache[content])

        if self._tokenizer is not None:
            try:
                n = len(self._tokenizer.encode(content, add_special_tokens=False))
                self._cache[content] = int(n)
                return int(n)
            except Exception:
                self._tokenizer = None
                self.mode = "whitespace_fallback"

        n = len([tok for tok in content.split() if tok.strip()])
        self._cache[content] = int(n)
        return int(n)

    @property
    def tokenizer_name(self) -> str:
        if self._tokenizer is None:
            return "whitespace"
        return str(getattr(self._tokenizer, "name_or_path", self.model_name) or self.model_name)


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        rr = csv.DictReader(f, delimiter="\t")
        for row in rr:
            rows.append({str(k): str(v or "") for k, v in dict(row).items()})
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")) or {})


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = str(line or "").strip()
            if not line:
                continue
            out.append(dict(json.loads(line) or {}))
    return out


def _resolve_qa_item(index: QASampleIndex, qid: str, question: str, sample_index: int) -> Optional[Dict[str, Any]]:
    sid = _strip(qid)
    q = _strip(question)
    if sid and sid in index.by_qid:
        return dict(index.by_qid[sid] or {})
    if q:
        rows = list(index.by_question.get(q, []) or [])
        if len(rows) == 1:
            return dict(rows[0] or {})
        if len(rows) > 1 and 0 <= sample_index < len(index.by_index):
            return dict(index.by_index[sample_index] or {})
        if rows:
            return dict(rows[0] or {})
    if 0 <= sample_index < len(index.by_index):
        return dict(index.by_index[sample_index] or {})
    return None


def _lookup_metric(summary: Mapping[str, Any], aliases: Sequence[str], default: float = 0.0) -> float:
    src = dict(summary or {})
    for key in list(aliases or []):
        if key in src:
            return _safe_float(src.get(key), default)
    return float(default)


def _support_match(rendered_context: str, support_text: str) -> bool:
    rendered = _strip(rendered_context)
    support = _strip(support_text)
    if not rendered or not support:
        return False
    if support in rendered:
        return True
    return _normalize_text(support) in _normalize_text(rendered)


def _answer_aliases(qa_item: Mapping[str, Any], row: Mapping[str, Any]) -> List[str]:
    seen = set()
    out: List[str] = []

    def _add(v: Any) -> None:
        vv = _strip(v)
        if not vv:
            return
        low = vv.lower()
        if low in seen:
            return
        seen.add(low)
        out.append(vv)

    _add((qa_item or {}).get("answer"))
    _add((row or {}).get("answer"))
    for v in list((row.get("gold_answers", []) or []) if isinstance(row, Mapping) else []):
        _add(v)

    raw = dict((qa_item or {}).get("raw", {}) or {})
    for key in ("possible_answers", "answers", "answer_aliases", "o_aliases", "aliases"):
        val = raw.get(key)
        if isinstance(val, list):
            for item in val:
                _add(item)
        elif isinstance(val, str):
            _add(val)
    return out


def _answer_surface_stats(rendered_text: str, aliases: Sequence[str]) -> Dict[str, Any]:
    tokens = [tok for tok in _normalize_text(rendered_text).split() if tok]
    if not tokens:
        return {"present": False, "token_count": 0, "position": -1.0}

    matched = set()
    for alias in list(aliases or []):
        alias_tokens = [tok for tok in _normalize_text(alias).split() if tok]
        if not alias_tokens:
            continue
        n = len(alias_tokens)
        if n > len(tokens):
            continue
        for i in range(0, len(tokens) - n + 1):
            if tokens[i : i + n] == alias_tokens:
                for j in range(i, i + n):
                    matched.add(int(j))

    if not matched:
        return {"present": False, "token_count": 0, "position": -1.0}
    first = min(matched)
    return {
        "present": True,
        "token_count": int(len(matched)),
        "position": float(safe_div(float(first), float(max(1, len(tokens))))),
    }


def _failure_bucket(q: Mapping[str, Any]) -> str:
    if _safe_float(q.get("abgf_answer_surface_present_generation_fail", 0.0), 0.0) > 0.5:
        return "abgf_answer_surface_present_generation_fail"
    if _safe_float(q.get("abgf_support_present_generation_fail", 0.0), 0.0) > 0.5:
        return "abgf_support_present_generation_fail"
    if _safe_float(q.get("support_present_answer_surface_absent", 0.0), 0.0) > 0.5:
        return "support_present_answer_surface_absent"
    if _safe_float(q.get("answer_surface_present_but_low_overlap", 0.0), 0.0) > 0.5:
        return "answer_surface_present_but_low_overlap"
    return ""


def _build_per_query_metrics(
    *,
    dataset: str,
    variant: str,
    rows: Sequence[Mapping[str, Any]],
    qa_index: QASampleIndex,
    token_counter: TokenCounter,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(list(rows or [])):
        row = dict(raw or {})
        metrics = dict(row.get("metrics", {}) or {})
        rendered = dict(row.get("rendered", {}) or {})
        rendered_meta = dict(rendered.get("metadata", {}) or {})
        rendering_meta = dict(row.get("rendering", {}) or {})
        gdiag = dict(row.get("generation_diagnostics", {}) or {})

        sample_index = _safe_int(row.get("sample_index", i), i)
        qid = _strip(row.get("sample_id"))
        question = _strip(row.get("question"))
        qa_item = _resolve_qa_item(qa_index, qid=qid, question=question, sample_index=sample_index) or {}

        rendered_context = _strip(rendered.get("text"))
        if not rendered_context:
            rendered_context = "\n".join([_strip(x) for x in list(rendered.get("sentences", []) or []) if _strip(x)])

        rendered_tokens = int(token_counter.count(rendered_context))

        gold_supports = list(qa_item.get("gold_supports", []) or [])
        matched_support_tokens = 0
        support_match_count = 0
        for sup in gold_supports:
            if _support_match(rendered_context, sup):
                support_match_count += 1
                matched_support_tokens += int(token_counter.count(sup))
        matched_support_tokens = min(matched_support_tokens, rendered_tokens)

        aliases = _answer_aliases(qa_item, row)
        surface = _answer_surface_stats(rendered_context, aliases)
        answer_surface_present = bool(surface.get("present", False))
        answer_surface_token_count = int(surface.get("token_count", 0) or 0)
        answer_surface_token_count = min(answer_surface_token_count, rendered_tokens)
        answer_surface_position = _safe_float(surface.get("position", -1.0), -1.0)

        f1 = _safe_float(metrics.get("f1", 0.0), 0.0)
        em = _safe_float(metrics.get("em", 0.0), 0.0)
        qa_executed = _safe_bool(row.get("qa_executed", True), True)
        output_overlap = _safe_float(metrics.get("output_overlap_answer_bearing", 0.0), 0.0)
        matched_gold_total = _safe_float(metrics.get("matched_gold_total", 0.0), 0.0)
        support_present = bool(matched_gold_total > 0.0)
        f1_low = bool(f1 <= 0.01)

        abgf = _safe_float(
            metrics.get("answer_present_but_generation_fail"),
            1.0
            if (_safe_float(metrics.get("answer_bearing_chunk_present", 0.0), 0.0) > 0.5 and qa_executed and f1_low)
            else 0.0,
        )
        abgf_support = _safe_float(
            metrics.get("abgf_support_present_generation_fail"),
            1.0 if (support_present and qa_executed and f1_low) else 0.0,
        )
        abgf_surface = _safe_float(
            metrics.get("abgf_answer_surface_present_generation_fail"),
            1.0 if (answer_surface_present and qa_executed and f1_low) else 0.0,
        )
        support_surface_absent = _safe_float(
            metrics.get("support_present_answer_surface_absent"),
            1.0 if (support_present and (not answer_surface_present)) else 0.0,
        )
        low_overlap = _safe_float(
            metrics.get("answer_surface_present_but_low_overlap"),
            1.0 if (answer_surface_present and output_overlap < 0.35) else 0.0,
        )

        qa_variant = _strip(gdiag.get("qa_utilization_variant"))
        qa_applied = _safe_bool(gdiag.get("qa_utilization_applied", False), False)
        if (not qa_applied) and qa_variant:
            qa_applied = True
        qa_changed = _safe_bool(gdiag.get("qa_utilization_changed", False), False)
        no_rewrite_violation = 1.0 if (qa_applied or qa_changed or bool(qa_variant)) else 0.0

        evidence_order = list(rendered.get("sentence_ids", []) or row.get("rendered_sentence_ids", []) or [])
        corridor_or_path_ids = list(rendered.get("rendered_corridor_ids", []) or row.get("rendered_corridor_ids", []) or [])
        render_variant = _strip(rendered_meta.get("render_variant") or rendering_meta.get("render_variant") or rendered.get("render_mode") or rendering_meta.get("render_mode"))
        if not render_variant:
            render_variant = "corridor_aware_flat"
        prompt_variant = _strip(gdiag.get("prompt_variant") or rendering_meta.get("prompt_variant") or rendered_meta.get("prompt_variant") or "default")
        if not prompt_variant:
            prompt_variant = "default"

        corridor_group_count_q = _safe_float(rendered_meta.get("corridor_group_count"), 0.0)
        if corridor_group_count_q <= 0.0 and corridor_or_path_ids:
            corridor_group_count_q = float(len(corridor_or_path_ids))

        path_block_count_q = _safe_float(
            rendered_meta.get("path_block_count", rendered_meta.get("path_bundle_count", 0.0)),
            0.0,
        )

        role_tag_applied = _safe_bool(rendered_meta.get("role_tagged_compact_applied", False), False)
        if (not role_tag_applied) and _safe_int(rendered_meta.get("role_tag_applied_count", 0), 0) > 0:
            role_tag_applied = True
        role_unknown_rate_q = _safe_float(rendered_meta.get("role_unknown_rate", 0.0), 0.0)
        title_source_info = bool(any("::" in _strip(x) for x in evidence_order))

        out.append(
            {
                "dataset": str(dataset),
                "variant": str(variant),
                "qid": _strip(qa_item.get("qid") or qid or f"{dataset}-{sample_index}"),
                "question": question or _strip(qa_item.get("question")),
                "gold_answer": _strip(qa_item.get("answer") or row.get("answer") or ((row.get("gold_answers", []) or [""])[0])),
                "prediction": _strip(row.get("prediction")),
                "f1": float(f1),
                "em": float(em),
                "rendered_context": rendered_context,
                "rendered_context_tokens": float(rendered_tokens),
                "matched_support_tokens": float(matched_support_tokens),
                "sf_token_density_q": float(safe_div(float(matched_support_tokens), float(max(1, rendered_tokens)))),
                "answer_surface_token_count": float(answer_surface_token_count),
                "answer_token_density_q": float(safe_div(float(answer_surface_token_count), float(max(1, rendered_tokens)))),
                "matched_gold_total": float(matched_gold_total),
                "answer_bearing_chunk_present": float(_safe_float(metrics.get("answer_bearing_chunk_present", 0.0), 0.0)),
                "answer_surface_present": 1.0 if answer_surface_present else 0.0,
                "output_overlap_answer_bearing": float(output_overlap),
                "answer_surface_position": float(answer_surface_position),
                "answer_surface_token_density": float(safe_div(float(answer_surface_token_count), float(max(1, rendered_tokens)))),
                "answer_present_but_generation_fail": float(abgf),
                "abgf_support_present_generation_fail": float(abgf_support),
                "abgf_answer_surface_present_generation_fail": float(abgf_surface),
                "support_present_answer_surface_absent": float(support_surface_absent),
                "answer_surface_present_but_low_overlap": float(low_overlap),
                "qa_utilization_applied": 1.0 if qa_applied else 0.0,
                "qa_utilization_changed": 1.0 if qa_changed else 0.0,
                "no_prediction_rewrite_violation": float(no_rewrite_violation),
                "generation_fallback": 1.0 if _safe_bool(row.get("generation_fallback", False), False) else 0.0,
                "render_variant": str(render_variant),
                "prompt_variant": str(prompt_variant),
                "corridor_group_count_q": float(corridor_group_count_q),
                "path_block_count_q": float(path_block_count_q),
                "role_tag_activation_q": 1.0 if role_tag_applied else 0.0,
                "role_unknown_rate_q": float(max(0.0, min(1.0, role_unknown_rate_q))),
                "title_source_info": 1.0 if title_source_info else 0.0,
                "evidence_order": list(evidence_order),
                "corridor_or_path_ids": list(corridor_or_path_ids),
            }
        )
    return out


def _aggregate_variant(
    *,
    dataset: str,
    variant: str,
    summary: Mapping[str, Any],
    qrows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = [dict(r or {}) for r in list(qrows or [])]
    n = len(rows)
    rendered_sum = float(sum(_safe_float(r.get("rendered_context_tokens", 0.0), 0.0) for r in rows))
    support_tok_sum = float(sum(_safe_float(r.get("matched_support_tokens", 0.0), 0.0) for r in rows))
    answer_tok_sum = float(sum(_safe_float(r.get("answer_surface_token_count", 0.0), 0.0) for r in rows))

    avg_context_tokens = float(_mean([_safe_float(r.get("rendered_context_tokens", 0.0), 0.0) for r in rows]))
    sf_token_density = float(safe_div(support_tok_sum, rendered_sum)) if rendered_sum > 0.0 else 0.0
    answer_token_density = float(safe_div(answer_tok_sum, rendered_sum)) if rendered_sum > 0.0 else 0.0

    f1 = _lookup_metric(summary, ("f1", "F1"), default=0.0)
    em = _lookup_metric(summary, ("em", "EM"), default=0.0)
    avg_ctx_k = float(safe_div(avg_context_tokens, 1000.0))
    f1_per_1k = float(safe_div(f1, avg_ctx_k)) if avg_ctx_k > 0.0 else 0.0

    def _rate(key: str) -> float:
        return float(_mean([_safe_float(r.get(key, 0.0), 0.0) for r in rows]))

    answer_surface_positions = [
        _safe_float(r.get("answer_surface_position", -1.0), -1.0)
        for r in rows
        if _safe_float(r.get("answer_surface_position", -1.0), -1.0) >= 0.0
    ]

    retrieval_ms = _lookup_metric(summary, ("retrieval_ms", "retrieval_latency_ms"), default=0.0)
    generation_ms = _lookup_metric(summary, ("generation_ms", "generation_latency_ms"), default=0.0)
    total_ms = _lookup_metric(summary, ("total_ms", "total_latency_ms"), default=(retrieval_ms + generation_ms))

    out = {
        "dataset": str(dataset),
        "variant": str(variant),
        "query_count": int(n),
        "recall_at_1": _lookup_metric(summary, ("recall_at_1", "supporting_fact_recall_at_1", "R@1"), default=0.0),
        "recall_at_5": _lookup_metric(summary, ("recall_at_5", "supporting_fact_recall_at_5", "R@5"), default=0.0),
        "recall_at_10": _lookup_metric(summary, ("recall_at_10", "supporting_fact_recall_at_10", "R@10"), default=0.0),
        "em": float(em),
        "f1": float(f1),
        "supporting_fact_precision": _lookup_metric(summary, ("supporting_fact_precision", "sf_P"), default=0.0),
        "supporting_fact_recall": _lookup_metric(summary, ("supporting_fact_recall", "sf_R"), default=0.0),
        "supporting_fact_f1": _lookup_metric(summary, ("supporting_fact_f1", "sf_F1"), default=0.0),
        "rendered_sf_recall": _lookup_metric(
            summary,
            ("rendered_supporting_fact_recall", "rendered_sf_recall", "rendered_sf_R"),
            default=0.0,
        ),
        "avg_context_tokens": float(avg_context_tokens),
        "sf_token_density": float(sf_token_density),
        "answer_token_density": float(answer_token_density),
        "F1_per_1k_context_tokens": float(f1_per_1k),
        "answer_bearing_chunk_present": _rate("answer_bearing_chunk_present"),
        "output_overlap_answer_bearing": _rate("output_overlap_answer_bearing"),
        "answer_present_but_generation_fail": _rate("answer_present_but_generation_fail"),
        "abgf_support_present_generation_fail": _rate("abgf_support_present_generation_fail"),
        "abgf_answer_surface_present_generation_fail": _rate("abgf_answer_surface_present_generation_fail"),
        "support_present_answer_surface_absent": _rate("support_present_answer_surface_absent"),
        "answer_surface_present_but_low_overlap": _rate("answer_surface_present_but_low_overlap"),
        "answer_surface_position_avg": float(_mean(answer_surface_positions)),
        "qa_utilization_activation_rate": _rate("qa_utilization_applied"),
        "qa_utilization_changed_rate": _rate("qa_utilization_changed"),
        "no_prediction_rewrite_violation_rate": _rate("no_prediction_rewrite_violation"),
        "render_variant": _mode_text([str(r.get("render_variant", "")) for r in rows], default=""),
        "prompt_variant": _mode_text([str(r.get("prompt_variant", "")) for r in rows], default="default"),
        "corridor_group_count": float(_mean([_safe_float(r.get("corridor_group_count_q", 0.0), 0.0) for r in rows])),
        "path_block_count": float(_mean([_safe_float(r.get("path_block_count_q", 0.0), 0.0) for r in rows])),
        "role_tag_activation_rate": _rate("role_tag_activation_q"),
        "role_unknown_rate": float(_mean([_safe_float(r.get("role_unknown_rate_q", 0.0), 0.0) for r in rows])),
        "retrieval_ms": float(retrieval_ms),
        "generation_ms": float(generation_ms),
        "total_ms": float(total_ms),
        "fallback_rate": _lookup_metric(summary, ("fallback_rate",), default=_rate("generation_fallback")),
    }
    return out


def _delta_rows(rows: Sequence[Mapping[str, Any]], baseline_variant: str) -> List[Dict[str, Any]]:
    items = [dict(x or {}) for x in list(rows or [])]
    idx = {(str(r.get("dataset", "")), str(r.get("variant", ""))): r for r in items}
    out: List[Dict[str, Any]] = []
    for row in items:
        ds = str(row.get("dataset", ""))
        vv = str(row.get("variant", ""))
        base = idx.get((ds, baseline_variant))
        if base is None:
            continue
        avg_base = _safe_float(base.get("avg_context_tokens", 0.0), 0.0)
        avg_now = _safe_float(row.get("avg_context_tokens", 0.0), 0.0)
        pct = float(100.0 * safe_div((avg_now - avg_base), avg_base)) if avg_base > 0.0 else 0.0
        out.append(
            {
                "dataset": ds,
                "variant": vv,
                "baseline_variant": baseline_variant,
                "delta_recall_at_5": _safe_float(row.get("recall_at_5", 0.0), 0.0) - _safe_float(base.get("recall_at_5", 0.0), 0.0),
                "delta_recall_at_10": _safe_float(row.get("recall_at_10", 0.0), 0.0) - _safe_float(base.get("recall_at_10", 0.0), 0.0),
                "delta_em": _safe_float(row.get("em", 0.0), 0.0) - _safe_float(base.get("em", 0.0), 0.0),
                "delta_f1": _safe_float(row.get("f1", 0.0), 0.0) - _safe_float(base.get("f1", 0.0), 0.0),
                "delta_abgf": _safe_float(row.get("answer_present_but_generation_fail", 0.0), 0.0)
                - _safe_float(base.get("answer_present_but_generation_fail", 0.0), 0.0),
                "delta_output_overlap": _safe_float(row.get("output_overlap_answer_bearing", 0.0), 0.0)
                - _safe_float(base.get("output_overlap_answer_bearing", 0.0), 0.0),
                "avg_context_tokens_delta": avg_now - avg_base,
                "avg_context_tokens_increase_pct": pct,
                "sf_token_density_delta": _safe_float(row.get("sf_token_density", 0.0), 0.0)
                - _safe_float(base.get("sf_token_density", 0.0), 0.0),
                "answer_token_density_delta": _safe_float(row.get("answer_token_density", 0.0), 0.0)
                - _safe_float(base.get("answer_token_density", 0.0), 0.0),
                "answer_surface_position_delta": _safe_float(row.get("answer_surface_position_avg", 0.0), 0.0)
                - _safe_float(base.get("answer_surface_position_avg", 0.0), 0.0),
                "generation_ms_delta": _safe_float(row.get("generation_ms", 0.0), 0.0)
                - _safe_float(base.get("generation_ms", 0.0), 0.0),
                "total_ms_delta": _safe_float(row.get("total_ms", 0.0), 0.0) - _safe_float(base.get("total_ms", 0.0), 0.0),
            }
        )
    out.sort(key=lambda x: (x.get("dataset", ""), x.get("variant", "")))
    return out


def _build_interface_audit(per_query_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(r or {}) for r in list(per_query_rows or [])]
    by_ds: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if _strip(row.get("variant")) != "champion_reconfirm":
            continue
        ds = _strip(row.get("dataset"))
        by_ds.setdefault(ds, []).append(row)

    datasets = []
    for ds, ds_rows in sorted(by_ds.items(), key=lambda kv: kv[0]):
        abgf_rows = [r for r in ds_rows if _safe_float(r.get("answer_present_but_generation_fail", 0.0), 0.0) > 0.5]
        ok_rows = [r for r in ds_rows if _safe_float(r.get("answer_present_but_generation_fail", 0.0), 0.0) <= 0.5]

        def _avg(src_rows: Sequence[Mapping[str, Any]], key: str, min_value: Optional[float] = None) -> float:
            vals = [_safe_float(r.get(key, 0.0), 0.0) for r in src_rows]
            if min_value is not None:
                vals = [v for v in vals if v >= float(min_value)]
            return float(_mean(vals))

        dataset_summary = {
            "dataset": ds,
            "query_count": int(len(ds_rows)),
            "abgf_count": int(len(abgf_rows)),
            "non_abgf_count": int(len(ok_rows)),
            "abgf_rate": float(safe_div(len(abgf_rows), max(1, len(ds_rows)))),
            "answer_surface_position_avg_abgf": _avg(abgf_rows, "answer_surface_position", min_value=0.0),
            "answer_surface_position_avg_non_abgf": _avg(ok_rows, "answer_surface_position", min_value=0.0),
            "output_overlap_avg_abgf": _avg(abgf_rows, "output_overlap_answer_bearing"),
            "output_overlap_avg_non_abgf": _avg(ok_rows, "output_overlap_answer_bearing"),
            "avg_context_tokens_abgf": _avg(abgf_rows, "rendered_context_tokens"),
            "avg_context_tokens_non_abgf": _avg(ok_rows, "rendered_context_tokens"),
            "answer_surface_present_rate_abgf": _avg(abgf_rows, "answer_surface_present"),
            "answer_surface_present_rate_non_abgf": _avg(ok_rows, "answer_surface_present"),
            "corridor_or_path_info_rate": _avg(ds_rows, "corridor_or_path_info_present"),
            "title_source_info_rate": _avg(ds_rows, "title_source_info"),
        }
        dataset_summary["answer_surface_position_gap"] = float(
            dataset_summary["answer_surface_position_avg_abgf"] - dataset_summary["answer_surface_position_avg_non_abgf"]
        )
        dataset_summary["output_overlap_gap"] = float(
            dataset_summary["output_overlap_avg_abgf"] - dataset_summary["output_overlap_avg_non_abgf"]
        )
        datasets.append(dataset_summary)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "champion_variant": "champion_reconfirm",
        "datasets": datasets,
    }


def _audit_markdown(audit: Mapping[str, Any]) -> str:
    lines = ["# Context Interface Audit", ""]
    for ds_row in list(audit.get("datasets", []) or []):
        ds = _strip(ds_row.get("dataset"))
        lines.append(f"## {ds}")
        lines.append("")
        lines.append(f"- query_count: {int(_safe_int(ds_row.get('query_count', 0), 0))}")
        lines.append(f"- abgf_rate: {_safe_float(ds_row.get('abgf_rate', 0.0), 0.0):.4f}")
        lines.append(
            f"- answer_surface_position_avg (abgf/non-abgf): "
            f"{_safe_float(ds_row.get('answer_surface_position_avg_abgf', 0.0), 0.0):.4f} / "
            f"{_safe_float(ds_row.get('answer_surface_position_avg_non_abgf', 0.0), 0.0):.4f}"
        )
        lines.append(
            f"- output_overlap_avg (abgf/non-abgf): "
            f"{_safe_float(ds_row.get('output_overlap_avg_abgf', 0.0), 0.0):.4f} / "
            f"{_safe_float(ds_row.get('output_overlap_avg_non_abgf', 0.0), 0.0):.4f}"
        )
        lines.append(
            f"- avg_context_tokens (abgf/non-abgf): "
            f"{_safe_float(ds_row.get('avg_context_tokens_abgf', 0.0), 0.0):.2f} / "
            f"{_safe_float(ds_row.get('avg_context_tokens_non_abgf', 0.0), 0.0):.2f}"
        )
        lines.append(
            f"- answer_surface_present_rate (abgf/non-abgf): "
            f"{_safe_float(ds_row.get('answer_surface_present_rate_abgf', 0.0), 0.0):.4f} / "
            f"{_safe_float(ds_row.get('answer_surface_present_rate_non_abgf', 0.0), 0.0):.4f}"
        )
        lines.append(f"- corridor_or_path_info_rate: {_safe_float(ds_row.get('corridor_or_path_info_rate', 0.0), 0.0):.4f}")
        lines.append(f"- title_source_info_rate: {_safe_float(ds_row.get('title_source_info_rate', 0.0), 0.0):.4f}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--round-root", required=True)
    p.add_argument("--run-records", default="")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--print-tables", default="true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    round_root = Path(args.round_root).resolve()
    run_records_path = Path(args.run_records).resolve() if _strip(args.run_records) else (round_root / "run_records.tsv")

    out_metrics_json = round_root / "context_interface_metrics.json"
    out_metrics_md = round_root / "context_interface_metrics.md"
    out_delta_md = round_root / "context_interface_delta_table.md"
    out_breakdown_md = round_root / "abgf_breakdown_table.md"
    out_compact_md = round_root / "context_compactness_guard_table.md"
    out_render_prompt_md = round_root / "render_prompt_variant_table.md"
    out_audit_md = round_root / "context_interface_audit.md"
    out_audit_json = round_root / "context_interface_audit.json"
    out_examples_jsonl = round_root / "abgf_context_examples.jsonl"
    out_defs_md = round_root / "metric_definition_notes.md"

    token_counter = TokenCounter(args.model_name)
    registry: MetricRegistry = build_context_interface_abgf_metric_registry()

    run_records = _read_tsv(run_records_path)
    active = []
    for rr in run_records:
        if _strip(rr.get("status")) not in {"ok", "skipped"}:
            continue
        if _strip(rr.get("kind")) and _strip(rr.get("kind")) != "rag":
            continue
        summary_path = Path(_strip(rr.get("summary_path")))
        if not summary_path.exists():
            continue
        active.append(dict(rr))

    qa_index_cache: Dict[str, QASampleIndex] = {}
    metrics_rows: List[Dict[str, Any]] = []
    per_query_all: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for rr in active:
        dataset = _strip(rr.get("dataset"))
        variant = _strip(rr.get("variant"))
        qa_path = _strip(rr.get("qa_path"))
        summary_path = Path(_strip(rr.get("summary_path"))).resolve()
        query_path = summary_path.with_name("rag_query_results.jsonl")
        if not query_path.exists():
            failures.append({"dataset": dataset, "variant": variant, "error": f"missing query file: {query_path}"})
            continue
        if qa_path not in qa_index_cache:
            qa_index_cache[qa_path] = load_qa_index(qa_path)
        qa_index = qa_index_cache[qa_path]

        summary = _read_json(summary_path)
        qraw = _read_jsonl(query_path)
        qrows = _build_per_query_metrics(
            dataset=dataset,
            variant=variant,
            rows=qraw,
            qa_index=qa_index,
            token_counter=token_counter,
        )

        for qr in qrows:
            qr["corridor_or_path_info_present"] = 1.0 if bool(qr.get("corridor_or_path_ids", [])) else 0.0

        per_query_all.extend(qrows)
        agg = _aggregate_variant(dataset=dataset, variant=variant, summary=summary, qrows=qrows)
        metrics_rows.append(agg)

    metrics_rows.sort(key=lambda x: (str(x.get("dataset", "")), str(x.get("variant", ""))))

    baseline_index = {
        (str(r.get("dataset", "")), str(r.get("variant", ""))): r
        for r in metrics_rows
    }

    for row in metrics_rows:
        base = baseline_index.get((str(row.get("dataset", "")), "champion_reconfirm"))
        if base is None:
            row["avg_context_tokens_delta"] = 0.0
            row["answer_surface_position_delta"] = 0.0
            continue
        row["avg_context_tokens_delta"] = _safe_float(row.get("avg_context_tokens", 0.0), 0.0) - _safe_float(base.get("avg_context_tokens", 0.0), 0.0)
        row["answer_surface_position_delta"] = _safe_float(row.get("answer_surface_position_avg", 0.0), 0.0) - _safe_float(base.get("answer_surface_position_avg", 0.0), 0.0)

    retrieval_invariance_violations: List[Dict[str, Any]] = []
    for row in metrics_rows:
        ds = str(row.get("dataset", ""))
        vv = str(row.get("variant", ""))
        if vv == "champion_reconfirm":
            continue
        base = baseline_index.get((ds, "champion_reconfirm"))
        if base is None:
            continue
        d5 = abs(_safe_float(row.get("recall_at_5", 0.0), 0.0) - _safe_float(base.get("recall_at_5", 0.0), 0.0))
        d10 = abs(_safe_float(row.get("recall_at_10", 0.0), 0.0) - _safe_float(base.get("recall_at_10", 0.0), 0.0))
        if d5 > 1.0e-9 or d10 > 1.0e-9:
            retrieval_invariance_violations.append(
                {
                    "dataset": ds,
                    "variant": vv,
                    "delta_recall_at_5": float(d5),
                    "delta_recall_at_10": float(d10),
                }
            )

    no_rewrite_violations = [
        {
            "dataset": str(r.get("dataset", "")),
            "variant": str(r.get("variant", "")),
            "qa_utilization_activation_rate": _safe_float(r.get("qa_utilization_activation_rate", 0.0), 0.0),
            "qa_utilization_changed_rate": _safe_float(r.get("qa_utilization_changed_rate", 0.0), 0.0),
            "no_prediction_rewrite_violation_rate": _safe_float(r.get("no_prediction_rewrite_violation_rate", 0.0), 0.0),
        }
        for r in metrics_rows
        if _safe_float(r.get("no_prediction_rewrite_violation_rate", 0.0), 0.0) > 1.0e-12
    ]

    with out_examples_jsonl.open("w", encoding="utf-8") as f:
        for row in per_query_all:
            bucket = _failure_bucket(row)
            abgf_flag = bool(_safe_float(row.get("answer_present_but_generation_fail", 0.0), 0.0) > 0.5)
            if not abgf_flag:
                continue
            item = {
                "dataset": row.get("dataset", ""),
                "qid": row.get("qid", ""),
                "question": row.get("question", ""),
                "gold_answer": row.get("gold_answer", ""),
                "prediction": row.get("prediction", ""),
                "f1": _safe_float(row.get("f1", 0.0), 0.0),
                "abgf_flag": bool(abgf_flag),
                "rendered_context": row.get("rendered_context", ""),
                "evidence_order": list(row.get("evidence_order", []) or []),
                "corridor_or_path_ids_if_available": list(row.get("corridor_or_path_ids", []) or []),
                "title_source_info": bool(_safe_float(row.get("title_source_info", 0.0), 0.0) > 0.5),
                "answer_surface_present": bool(_safe_float(row.get("answer_surface_present", 0.0), 0.0) > 0.5),
                "answer_surface_position": _safe_float(row.get("answer_surface_position", -1.0), -1.0),
                "output_overlap_answer_bearing": _safe_float(row.get("output_overlap_answer_bearing", 0.0), 0.0),
                "failure_note": bucket,
                "variant": row.get("variant", ""),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    delta_rows = _delta_rows(metrics_rows, baseline_variant="champion_reconfirm")

    delta_headers = [
        "dataset",
        "variant",
        "baseline_variant",
        "delta_recall_at_5",
        "delta_recall_at_10",
        "delta_em",
        "delta_f1",
        "delta_abgf",
        "delta_output_overlap",
        "avg_context_tokens_delta",
        "avg_context_tokens_increase_pct",
        "sf_token_density_delta",
        "answer_token_density_delta",
        "answer_surface_position_delta",
        "generation_ms_delta",
        "total_ms_delta",
    ]

    group_tables = group_markdown_tables(records=metrics_rows, registry=registry)
    write_markdown(
        str(out_metrics_md),
        markdown_document_from_group_tables(group_tables, title="Context Interface ABGF Round"),
    )

    metrics_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_root": str(round_root),
        "tokenizer_mode": token_counter.mode,
        "tokenizer_name": token_counter.tokenizer_name,
        "records": metrics_rows,
        "delta_rows": delta_rows,
        "metric_groups": {group: [spec.name for spec in registry.enabled_by_group(group)] for group in registry.groups()},
        "integrity": {
            "retrieval_invariance_violations": retrieval_invariance_violations,
            "no_prediction_rewrite_violations": no_rewrite_violations,
            "failures_count": int(len(failures)),
            "failures": failures,
        },
        "run_context": {
            "git_commit_hash": _cmd(["git", "rev-parse", "HEAD"]),
            "active_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        },
    }
    write_json(str(out_metrics_json), metrics_json)

    breakdown_headers = [
        "dataset",
        "variant",
        "abgf_support_present_generation_fail",
        "abgf_answer_surface_present_generation_fail",
        "support_present_answer_surface_absent",
        "answer_surface_present_but_low_overlap",
    ]
    breakdown_md = build_flat_markdown_table(records=metrics_rows, headers=breakdown_headers, registry=registry)
    write_markdown(str(out_breakdown_md), breakdown_md + "\n")

    compact_md = build_flat_markdown_table(
        records=metrics_rows,
        headers=[
            "dataset",
            "variant",
            "avg_context_tokens",
            "sf_token_density",
            "answer_token_density",
            "F1_per_1k_context_tokens",
            "avg_context_tokens_delta",
        ],
        registry=registry,
    )
    write_markdown(str(out_compact_md), compact_md + "\n")

    render_prompt_md = build_flat_markdown_table(
        records=metrics_rows,
        headers=[
            "dataset",
            "variant",
            "render_variant",
            "prompt_variant",
            "corridor_group_count",
            "path_block_count",
            "role_tag_activation_rate",
            "role_unknown_rate",
        ],
        registry=registry,
    )
    write_markdown(str(out_render_prompt_md), render_prompt_md + "\n")

    delta_md = build_flat_markdown_table(records=delta_rows, headers=delta_headers, registry=registry)
    write_markdown(str(out_delta_md), delta_md + "\n")

    audit = _build_interface_audit(per_query_all)
    write_json(str(out_audit_json), audit)
    write_markdown(str(out_audit_md), _audit_markdown(audit))

    notes = metric_definition_notes(
        registry=registry,
        tokenizer_mode=token_counter.mode,
        tokenizer_name=token_counter.tokenizer_name,
        optional_notes=[
            "Context-to-generation interface round keeps champion retrieval fixed and changes render/prompt only.",
            "ABGF decomposition uses low-F1 threshold <= 0.01.",
            "answer_surface_present uses normalized token-sequence matching against gold answer + aliases.",
            "no prediction rewrite constraint is checked by qa_utilization activation/changed rates.",
            "retrieval invariance check compares R@5 and R@10 against champion_reconfirm.",
        ],
    )
    write_markdown(str(out_defs_md), notes)

    print_tables = _strip(args.print_tables).lower() in {"1", "true", "yes", "y", "on"}
    if print_tables:
        print("## Main QA Table")
        print(
            build_flat_markdown_table(
                records=metrics_rows,
                headers=[
                    "dataset",
                    "variant",
                    "recall_at_5",
                    "em",
                    "f1",
                    "answer_present_but_generation_fail",
                    "output_overlap_answer_bearing",
                    "avg_context_tokens",
                    "total_ms",
                ],
                registry=registry,
            )
        )
        print("")

        print("## ABGF Breakdown Table")
        print(breakdown_md)
        print("")

        print("## Context Compactness Guard Table")
        print(compact_md)
        print("")

        print("## Render/Prompt Variant Table")
        print(render_prompt_md)
        print("")

        print("## Variant Delta Table")
        print(delta_md)
        print("")

    print(str(round_root))


if __name__ == "__main__":
    main()
