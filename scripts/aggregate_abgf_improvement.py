#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.eval_adapters import QASampleIndex, load_qa_index
from effirag.eval_metrics import (
    MetricRegistry,
    build_abgf_improvement_metric_registry,
    metric_definition_notes,
    write_json,
)
from effirag.eval_reports import (
    build_flat_markdown_table,
    build_group_table_markdown,
    group_markdown_tables,
    markdown_document_from_group_tables,
    write_markdown,
)
from effirag.utils import safe_div

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


def _variant_label(raw: str) -> str:
    v = _strip(raw).lower()
    if v in {"answer_normalization_light", "answer_surface_normalization"}:
        return "normalization"
    if v in {"answer_type_aware_extraction"}:
        return "extraction"
    if v in {"answer_verification_light", "evidence_supported_verification"}:
        return "verification"
    return ""


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


def _mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in list(values or [])]
    if not seq:
        return 0.0
    return float(sum(seq) / len(seq))


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

        qa_var = _strip(gdiag.get("qa_utilization_variant"))
        qa_applied = _safe_bool(gdiag.get("qa_utilization_applied", False), False)
        if (not qa_applied) and qa_var:
            qa_applied = True
        qa_changed = _safe_bool(gdiag.get("qa_utilization_changed", False), False)

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
                "qa_utilization_variant": str(qa_var),
                "qa_utilization_family": _variant_label(qa_var),
                "initial_f1": float(_safe_float(gdiag.get("initial_f1", f1), f1)),
                "highlight_applied": 1.0
                if bool(((rendered.get("metadata", {}) or {}).get("answer_cue_highlight_applied", False)))
                else 0.0,
                "generation_fallback": 1.0 if _safe_bool(row.get("generation_fallback", False), False) else 0.0,
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

    def _count_where(pred) -> int:
        return int(sum(1 for r in rows if pred(r)))

    norm_rows = [r for r in rows if str(r.get("qa_utilization_family", "")) == "normalization"]
    ext_rows = [r for r in rows if str(r.get("qa_utilization_family", "")) == "extraction"]
    ver_rows = [r for r in rows if str(r.get("qa_utilization_family", "")) == "verification"]

    def _var_stats(src_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        src = [dict(x or {}) for x in list(src_rows or [])]
        activation = [r for r in src if _safe_float(r.get("qa_utilization_applied", 0.0), 0.0) > 0.5]
        changed = [r for r in activation if _safe_float(r.get("qa_utilization_changed", 0.0), 0.0) > 0.5]
        helped = [
            r
            for r in activation
            if _safe_float(r.get("f1", 0.0), 0.0) > _safe_float(r.get("initial_f1", 0.0), 0.0) + 1.0e-9
        ]
        hurt = [
            r
            for r in activation
            if _safe_float(r.get("f1", 0.0), 0.0) + 1.0e-9 < _safe_float(r.get("initial_f1", 0.0), 0.0)
        ]
        return {
            "activation_rate": float(len(activation) / n) if n > 0 else 0.0,
            "changed_rate": float(len(changed) / n) if n > 0 else 0.0,
            "helped_count": int(len(helped)),
            "hurt_count": int(len(hurt)),
        }

    norm_stats = _var_stats(norm_rows)
    ext_stats = _var_stats(ext_rows)
    ver_stats = _var_stats(ver_rows)

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
        "qa_utilization_activation_rate": _rate("qa_utilization_applied"),
        "qa_utilization_changed_rate": _rate("qa_utilization_changed"),
        "normalization_activation_rate": float(norm_stats["activation_rate"]),
        "normalization_changed_rate": float(norm_stats["changed_rate"]),
        "normalization_helped_count": int(norm_stats["helped_count"]),
        "normalization_hurt_count": int(norm_stats["hurt_count"]),
        "extraction_activation_rate": float(ext_stats["activation_rate"]),
        "extraction_changed_rate": float(ext_stats["changed_rate"]),
        "extraction_helped_count": int(ext_stats["helped_count"]),
        "extraction_hurt_count": int(ext_stats["hurt_count"]),
        "verification_activation_rate": float(ver_stats["activation_rate"]),
        "verification_changed_rate": float(ver_stats["changed_rate"]),
        "verification_helped_count": int(ver_stats["helped_count"]),
        "verification_hurt_count": int(ver_stats["hurt_count"]),
        "highlight_activation_rate": _rate("highlight_applied"),
        "answer_surface_position_avg": float(_mean(answer_surface_positions)),
        "answer_surface_token_density": _rate("answer_surface_token_density"),
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
                "delta_em": _safe_float(row.get("em", 0.0), 0.0) - _safe_float(base.get("em", 0.0), 0.0),
                "delta_f1": _safe_float(row.get("f1", 0.0), 0.0) - _safe_float(base.get("f1", 0.0), 0.0),
                "delta_abgf": _safe_float(row.get("answer_present_but_generation_fail", 0.0), 0.0)
                - _safe_float(base.get("answer_present_but_generation_fail", 0.0), 0.0),
                "delta_output_overlap": _safe_float(row.get("output_overlap_answer_bearing", 0.0), 0.0)
                - _safe_float(base.get("output_overlap_answer_bearing", 0.0), 0.0),
                "avg_context_tokens_delta": avg_now - avg_base,
                "avg_context_tokens_increase_pct": pct,
                "answer_token_density_delta": _safe_float(row.get("answer_token_density", 0.0), 0.0)
                - _safe_float(base.get("answer_token_density", 0.0), 0.0),
                "sf_token_density_delta": _safe_float(row.get("sf_token_density", 0.0), 0.0)
                - _safe_float(base.get("sf_token_density", 0.0), 0.0),
                "generation_ms_delta": _safe_float(row.get("generation_ms", 0.0), 0.0)
                - _safe_float(base.get("generation_ms", 0.0), 0.0),
                "total_ms_delta": _safe_float(row.get("total_ms", 0.0), 0.0) - _safe_float(base.get("total_ms", 0.0), 0.0),
            }
        )
    out.sort(key=lambda x: (x.get("dataset", ""), x.get("variant", "")))
    return out


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

    out_metrics_json = round_root / "abgf_improvement_metrics.json"
    out_metrics_md = round_root / "abgf_improvement_metrics.md"
    out_delta_md = round_root / "abgf_delta_table.md"
    out_breakdown_md = round_root / "abgf_breakdown_table.md"
    out_compact_md = round_root / "context_compactness_guard_table.md"
    out_qa_util_md = round_root / "qa_utilization_activation_table.md"
    out_breakdown_json = round_root / "abgf_breakdown_metrics.json"
    out_breakdown_metrics_md = round_root / "abgf_breakdown_metrics.md"
    out_failure_jsonl = round_root / "abgf_failure_samples.jsonl"
    out_defs_md = round_root / "metric_definition_notes.md"

    token_counter = TokenCounter(args.model_name)
    registry: MetricRegistry = build_abgf_improvement_metric_registry()

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
            failures.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "error": f"missing query file: {query_path}",
                }
            )
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
        per_query_all.extend(qrows)
        agg = _aggregate_variant(dataset=dataset, variant=variant, summary=summary, qrows=qrows)
        metrics_rows.append(agg)

    metrics_rows.sort(key=lambda x: (str(x.get("dataset", "")), str(x.get("variant", ""))))

    # Failure bucket export.
    with out_failure_jsonl.open("w", encoding="utf-8") as f:
        for row in per_query_all:
            bucket = _failure_bucket(row)
            if not bucket:
                continue
            item = {
                "dataset": row.get("dataset", ""),
                "qid": row.get("qid", ""),
                "question": row.get("question", ""),
                "gold_answer": row.get("gold_answer", ""),
                "prediction": row.get("prediction", ""),
                "f1": _safe_float(row.get("f1", 0.0), 0.0),
                "rendered_context": row.get("rendered_context", ""),
                "matched_gold_total": _safe_float(row.get("matched_gold_total", 0.0), 0.0),
                "answer_surface_present": bool(_safe_float(row.get("answer_surface_present", 0.0), 0.0) > 0.5),
                "output_overlap_answer_bearing": _safe_float(row.get("output_overlap_answer_bearing", 0.0), 0.0),
                "failure_bucket": bucket,
                "variant": row.get("variant", ""),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Delta table.
    delta_rows = _delta_rows(metrics_rows, baseline_variant="champion_reconfirm")

    delta_headers = [
        "dataset",
        "variant",
        "baseline_variant",
        "delta_recall_at_5",
        "delta_em",
        "delta_f1",
        "delta_abgf",
        "delta_output_overlap",
        "avg_context_tokens_delta",
        "avg_context_tokens_increase_pct",
        "sf_token_density_delta",
        "answer_token_density_delta",
        "generation_ms_delta",
        "total_ms_delta",
    ]

    # Group tables.
    group_tables = group_markdown_tables(records=metrics_rows, registry=registry)
    write_markdown(str(out_metrics_md), markdown_document_from_group_tables(group_tables, title="ABGF Improvement Round"))

    metrics_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_root": str(round_root),
        "tokenizer_mode": token_counter.mode,
        "tokenizer_name": token_counter.tokenizer_name,
        "records": metrics_rows,
        "delta_rows": delta_rows,
        "metric_groups": {group: [spec.name for spec in registry.enabled_by_group(group)] for group in registry.groups()},
        "failure_sample_path": str(out_failure_jsonl),
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
    write_markdown(str(out_breakdown_metrics_md), breakdown_md + "\n")
    write_json(
        str(out_breakdown_json),
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "round_root": str(round_root),
            "records": [{k: r.get(k) for k in breakdown_headers} for r in metrics_rows],
            "failure_sample_path": str(out_failure_jsonl),
        },
    )

    compact_md = build_group_table_markdown(
        records=metrics_rows,
        registry=registry,
        groups=["Context Compactness Guard Table"],
    )
    write_markdown(str(out_compact_md), compact_md)

    qa_util_md = build_group_table_markdown(
        records=metrics_rows,
        registry=registry,
        groups=["QA Utilization Activation Table"],
    )
    write_markdown(str(out_qa_util_md), qa_util_md)

    delta_md = build_flat_markdown_table(records=delta_rows, headers=delta_headers, registry=registry)
    write_markdown(str(out_delta_md), delta_md + "\n")

    notes = metric_definition_notes(
        registry=registry,
        tokenizer_mode=token_counter.mode,
        tokenizer_name=token_counter.tokenizer_name,
        optional_notes=[
            "ABGF decomposition uses low-F1 threshold <= 0.01.",
            "answer_surface_present uses normalized token-sequence matching against gold answer + available aliases.",
            "answer_surface_present_but_low_overlap uses output_overlap_answer_bearing < 0.35.",
            "Context compactness guard tracks avg_context_tokens and density deltas against champion_reconfirm.",
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
        print(
            build_flat_markdown_table(
                records=metrics_rows,
                headers=[
                    "dataset",
                    "variant",
                    "avg_context_tokens",
                    "sf_token_density",
                    "answer_token_density",
                    "F1_per_1k_context_tokens",
                ],
                registry=registry,
            )
        )
        print("")

        print("## QA Utilization Activation Table")
        print(
            build_flat_markdown_table(
                records=metrics_rows,
                headers=[
                    "dataset",
                    "variant",
                    "qa_utilization_activation_rate",
                    "qa_utilization_changed_rate",
                    "normalization_activation_rate",
                    "extraction_activation_rate",
                    "verification_activation_rate",
                    "highlight_activation_rate",
                ],
                registry=registry,
            )
        )
        print("")

        print("## Variant Delta Table")
        print(delta_md)
        print("")

    print(str(round_root))


if __name__ == "__main__":
    main()
