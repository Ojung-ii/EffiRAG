#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.eval_adapters import QASampleIndex, load_qa_index
from effirag.eval_metrics import (
    MetricRegistry,
    build_prompt_interface_bridge_metric_registry,
    metric_definition_notes,
    write_json,
)
from effirag.eval_reports import (
    build_flat_markdown_table,
    group_markdown_tables,
    markdown_document_from_group_tables,
    write_markdown,
)
from effirag.utils import safe_div

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_REASONING_MARKER_RE = re.compile(r"(^|\n)\s*(thought|reasoning)\s*:", re.IGNORECASE)


_PROMPT_VARIANT_META = {
    "light_separator_render": {
        "added_instruction": "none",
        "one_shot_enabled": 0.0,
        "title_grounded_format_enabled": 0.0,
        "final_answer_only": 1.0,
        "reasoning_forbidden": 1.0,
    },
    "light_separator_bridge_instruction": {
        "added_instruction": "bridge_and_copy_span",
        "one_shot_enabled": 0.0,
        "title_grounded_format_enabled": 0.0,
        "final_answer_only": 1.0,
        "reasoning_forbidden": 1.0,
    },
    "light_separator_copy_span_instruction": {
        "added_instruction": "copy_span_only",
        "one_shot_enabled": 0.0,
        "title_grounded_format_enabled": 0.0,
        "final_answer_only": 1.0,
        "reasoning_forbidden": 1.0,
    },
    "light_separator_title_grounded_format": {
        "added_instruction": "none",
        "one_shot_enabled": 0.0,
        "title_grounded_format_enabled": 1.0,
        "final_answer_only": 1.0,
        "reasoning_forbidden": 1.0,
    },
    "light_separator_final_answer_oneshot": {
        "added_instruction": "final_answer_oneshot",
        "one_shot_enabled": 1.0,
        "title_grounded_format_enabled": 0.0,
        "final_answer_only": 1.0,
        "reasoning_forbidden": 1.0,
    },
}


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


def _percentile(values: Sequence[float], q: float) -> float:
    seq = sorted([_safe_float(v, 0.0) for v in list(values or [])])
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    qq = max(0.0, min(1.0, _safe_float(q, 0.0)))
    pos = qq * float(len(seq) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(seq[lo])
    frac = pos - float(lo)
    return float(seq[lo] * (1.0 - frac) + seq[hi] * frac)


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


def _contains_think_tag(text: str) -> bool:
    low = str(text or "").lower()
    return "<think>" in low or "</think>" in low or bool(_THINK_RE.search(str(text or "")))


def _contains_reasoning_trace(text: str) -> bool:
    raw = str(text or "")
    low = raw.lower()
    if _contains_think_tag(raw):
        return True
    if _REASONING_MARKER_RE.search(raw):
        return True
    if "let's think" in low or "step by step" in low:
        return True
    return False


def _exact_answer_format_violation(prediction: str) -> bool:
    pred = str(prediction or "").strip()
    if not pred:
        return True
    if "\n" in pred:
        return True
    if _contains_think_tag(pred):
        return True
    tok_n = len([tok for tok in pred.split() if tok.strip()])
    if tok_n > 12:
        return True
    low = pred.lower()
    if low.startswith("because ") or low.startswith("the answer is "):
        return True
    return False


def _build_per_query_metrics(
    *,
    dataset: str,
    variant: str,
    rows: Sequence[Mapping[str, Any]],
    qa_index: QASampleIndex,
    token_counter: TokenCounter,
    config_meta: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    prompt_meta = dict(_PROMPT_VARIANT_META.get(str(variant), {}))
    prompt_meta.update({
        "added_instruction": _strip(config_meta.get("added_instruction")) or _strip(prompt_meta.get("added_instruction")),
        "one_shot_enabled": 1.0 if _safe_bool(config_meta.get("one_shot_enabled", prompt_meta.get("one_shot_enabled", False)), False) else 0.0,
        "final_answer_only": 1.0,
        "reasoning_forbidden": 1.0,
    })

    for i, raw in enumerate(list(rows or [])):
        row = dict(raw or {})
        metrics = dict(row.get("metrics", {}) or {})
        rendered = dict(row.get("rendered", {}) or {})
        rendered_meta = dict(rendered.get("metadata", {}) or {})
        rendering_meta = dict(row.get("rendering", {}) or {})
        gdiag = dict(row.get("generation_diagnostics", {}) or {})
        generation = dict(row.get("generation", {}) or {})

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

        f1 = _safe_float(metrics.get("f1", 0.0), 0.0)
        em = _safe_float(metrics.get("em", 0.0), 0.0)
        qa_executed = _safe_bool(row.get("qa_executed", True), True)
        output_overlap = _safe_float(metrics.get("output_overlap_answer_bearing", 0.0), 0.0)
        matched_gold_total = _safe_float(metrics.get("matched_gold_total", 0.0), 0.0)
        support_present = bool(matched_gold_total > 0.0)
        f1_low = bool(f1 <= 0.01)

        abgf = _safe_float(
            metrics.get("answer_present_but_generation_fail"),
            1.0 if (_safe_float(metrics.get("answer_bearing_chunk_present", 0.0), 0.0) > 0.5 and qa_executed and f1_low) else 0.0,
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

        render_variant = _strip(
            rendered_meta.get("render_variant")
            or rendering_meta.get("render_variant")
            or rendered.get("render_mode")
            or rendering_meta.get("render_mode")
            or "corridor_aware_flat"
        )
        prompt_variant = _strip(
            gdiag.get("prompt_variant")
            or (generation.get("metadata", {}) or {}).get("prompt_variant")
            or rendering_meta.get("prompt_variant")
            or rendered_meta.get("prompt_variant")
            or "default"
        )

        retrieval_source = _strip(row.get("retrieval_source"))
        prediction = _strip(row.get("prediction"))
        raw_text = str(generation.get("raw_text", "") or "")
        answer_length_tokens = int(token_counter.count(prediction))

        think_violation = 1.0 if _contains_think_tag(raw_text) else 0.0
        reasoning_violation = 1.0 if _contains_reasoning_trace(raw_text) else 0.0
        exact_violation = 1.0 if _exact_answer_format_violation(prediction) else 0.0
        format_violation = 1.0 if (think_violation > 0.5 or reasoning_violation > 0.5 or exact_violation > 0.5) else 0.0

        title_grounded_enabled = 1.0 if _safe_bool(rendered_meta.get("title_grounded_format_enabled", prompt_meta.get("title_grounded_format_enabled", 0.0)), False) else 0.0

        out.append(
            {
                "dataset": str(dataset),
                "variant": str(variant),
                "qid": _strip(qa_item.get("qid") or qid or f"{dataset}-{sample_index}"),
                "question": question or _strip(qa_item.get("question")),
                "gold_answer": _strip(qa_item.get("answer") or row.get("answer") or ((row.get("gold_answers", []) or [""])[0])),
                "prediction": prediction,
                "raw_text": raw_text,
                "f1": float(f1),
                "em": float(em),
                "rendered_context": rendered_context,
                "rendered_context_tokens": float(rendered_tokens),
                "matched_support_tokens": float(matched_support_tokens),
                "answer_surface_token_count": float(answer_surface_token_count),
                "answer_present_but_generation_fail": float(abgf),
                "abgf_support_present_generation_fail": float(abgf_support),
                "abgf_answer_surface_present_generation_fail": float(abgf_surface),
                "support_present_answer_surface_absent": float(support_surface_absent),
                "answer_surface_present_but_low_overlap": float(low_overlap),
                "output_overlap_answer_bearing": float(output_overlap),
                "answer_bearing_chunk_present": float(_safe_float(metrics.get("answer_bearing_chunk_present", 0.0), 0.0)),
                "qa_utilization_applied": 1.0 if qa_applied else 0.0,
                "qa_utilization_changed": 1.0 if qa_changed else 0.0,
                "no_prediction_rewrite_violation": float(no_rewrite_violation),
                "generation_fallback": 1.0 if _safe_bool(row.get("generation_fallback", False), False) else 0.0,
                "render_variant": str(render_variant),
                "prompt_variant": str(prompt_variant),
                "retrieval_source_precomputed": 1.0 if retrieval_source == "precomputed" else 0.0,
                "retrieval_source_on_the_fly": 1.0 if retrieval_source == "on_the_fly" else 0.0,
                "support_match_count": float(support_match_count),
                "support_total_count": float(len(gold_supports)),
                "reasoning_output_violation": float(reasoning_violation),
                "think_tag_violation": float(think_violation),
                "exact_answer_format_violation": float(exact_violation),
                "format_violation": float(format_violation),
                "answer_length_tokens": float(answer_length_tokens),
                "added_instruction": str(prompt_meta.get("added_instruction", "none") or "none"),
                "one_shot_enabled": float(prompt_meta.get("one_shot_enabled", 0.0)),
                "title_grounded_format_enabled": float(title_grounded_enabled),
                "final_answer_only": float(prompt_meta.get("final_answer_only", 1.0)),
                "reasoning_forbidden": float(prompt_meta.get("reasoning_forbidden", 1.0)),
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

    rendered_tokens = [_safe_float(r.get("rendered_context_tokens", 0.0), 0.0) for r in rows]
    rendered_sum = float(sum(rendered_tokens))
    support_tok_sum = float(sum(_safe_float(r.get("matched_support_tokens", 0.0), 0.0) for r in rows))
    answer_tok_sum = float(sum(_safe_float(r.get("answer_surface_token_count", 0.0), 0.0) for r in rows))

    avg_context_tokens = float(_mean(rendered_tokens))
    median_context_tokens = float(_percentile(rendered_tokens, 0.5))
    p90_context_tokens = float(_percentile(rendered_tokens, 0.9))
    sf_token_density = float(safe_div(support_tok_sum, rendered_sum)) if rendered_sum > 0.0 else 0.0
    answer_token_density = float(safe_div(answer_tok_sum, rendered_sum)) if rendered_sum > 0.0 else 0.0

    f1 = _lookup_metric(summary, ("f1", "F1"), default=0.0)
    em = _lookup_metric(summary, ("em", "EM"), default=0.0)
    avg_ctx_k = float(safe_div(avg_context_tokens, 1000.0))
    f1_per_1k = float(safe_div(f1, avg_ctx_k)) if avg_ctx_k > 0.0 else 0.0

    def _rate(key: str) -> float:
        return float(_mean([_safe_float(r.get(key, 0.0), 0.0) for r in rows]))

    retrieval_ms = _lookup_metric(summary, ("retrieval_ms", "retrieval_latency_ms"), default=0.0)
    generation_ms = _lookup_metric(summary, ("generation_ms", "generation_latency_ms"), default=0.0)
    total_ms = _lookup_metric(summary, ("total_ms", "total_latency_ms"), default=(retrieval_ms + generation_ms))

    out = {
        "dataset": str(dataset),
        "variant": str(variant),
        "query_count": int(len(rows)),
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
        "median_context_tokens": float(median_context_tokens),
        "p90_context_tokens": float(p90_context_tokens),
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
        "retrieval_ms": float(retrieval_ms),
        "generation_ms": float(generation_ms),
        "total_ms": float(total_ms),
        "fallback_rate": _lookup_metric(summary, ("fallback_rate",), default=_rate("generation_fallback")),
        "retrieval_source_precomputed_rate": _rate("retrieval_source_precomputed"),
        "retrieval_source_on_the_fly_rate": _rate("retrieval_source_on_the_fly"),
        "qa_utilization_changed_rate": _rate("qa_utilization_changed"),
        "no_prediction_rewrite_violation_rate": _rate("no_prediction_rewrite_violation"),
        "no_prediction_rewrite_pass": "pass"
        if _rate("qa_utilization_changed") <= 1.0e-12 and _rate("no_prediction_rewrite_violation") <= 1.0e-12
        else "fail",
        "render_variant": _mode_text([str(r.get("render_variant", "")) for r in rows], default=""),
        "prompt_variant": _mode_text([str(r.get("prompt_variant", "")) for r in rows], default="default"),
        "reasoning_output_violation_rate": _rate("reasoning_output_violation"),
        "think_tag_violation_rate": _rate("think_tag_violation"),
        "exact_answer_format_violation_rate": _rate("exact_answer_format_violation"),
        "format_violation_rate": _rate("format_violation"),
        "answer_length_avg": _rate("answer_length_tokens"),
        "added_instruction": _mode_text([str(r.get("added_instruction", "")) for r in rows], default="none"),
        "one_shot_enabled": _rate("one_shot_enabled"),
        "title_grounded_format_enabled": _rate("title_grounded_format_enabled"),
        "final_answer_only": _rate("final_answer_only"),
        "reasoning_forbidden": _rate("reasoning_forbidden"),
    }
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--round-root", required=True)
    p.add_argument("--run-records", default="")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--print-tables", default="true")
    return p


def _read_config_meta(path: str) -> Dict[str, Any]:
    p = Path(str(path or ""))
    if not p.exists():
        return {}
    try:
        return dict(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    except Exception:
        return {}


def main() -> None:
    args = _build_parser().parse_args()
    round_root = Path(args.round_root).resolve()
    run_records_path = Path(args.run_records).resolve() if _strip(args.run_records) else (round_root / "run_records.tsv")

    out_metrics_json = round_root / "prompt_interface_bridge_metrics.json"
    out_metrics_md = round_root / "prompt_interface_bridge_metrics.md"
    out_delta_md = round_root / "prompt_interface_delta_table.md"
    out_retrieval_inv_md = round_root / "retrieval_invariance_report.md"
    out_retrieval_inv_json = round_root / "retrieval_invariance_report.json"
    out_breakdown_md = round_root / "abgf_breakdown_table.md"
    out_context_eff_md = round_root / "context_efficiency_table.md"
    out_prompt_variant_md = round_root / "prompt_variant_table.md"
    out_format_violation_md = round_root / "format_violation_report.md"
    out_defs_md = round_root / "metric_definition_notes.md"

    token_counter = TokenCounter(args.model_name)
    registry: MetricRegistry = build_prompt_interface_bridge_metric_registry()

    run_records = _read_tsv(run_records_path)
    cfg_meta_by_key = {
        (str(r.get("dataset", "")), str(r.get("variant", ""))): _read_config_meta(r.get("config_path", ""))
        for r in run_records
        if _strip(r.get("kind", "rag")) == "rag"
    }
    expected_by_key = {
        (str(r.get("dataset", "")), str(r.get("variant", ""))): {
            "retrieval_fixed_expected": _safe_bool(r.get("retrieval_fixed_expected", False), False),
            "source_available": _safe_bool(r.get("source_available", False), False),
        }
        for r in run_records
        if _strip(r.get("kind", "rag")) == "rag"
    }

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
        cfg_meta = cfg_meta_by_key.get((dataset, variant), {})
        qrows = _build_per_query_metrics(
            dataset=dataset,
            variant=variant,
            rows=qraw,
            qa_index=qa_index,
            token_counter=token_counter,
            config_meta=cfg_meta,
        )

        agg = _aggregate_variant(dataset=dataset, variant=variant, summary=summary, qrows=qrows)
        key = (dataset, variant)
        expected = expected_by_key.get(key, {})
        agg["retrieval_fixed_expected"] = 1.0 if bool(expected.get("retrieval_fixed_expected", False)) else 0.0
        agg["source_available"] = 1.0 if bool(expected.get("source_available", False)) else 0.0
        metrics_rows.append(agg)

    metrics_rows.sort(key=lambda x: (str(x.get("dataset", "")), str(x.get("variant", ""))))
    idx = {(str(r.get("dataset", "")), str(r.get("variant", ""))): r for r in metrics_rows}
    baseline_variant = "light_separator_render"

    for row in metrics_rows:
        ds = str(row.get("dataset", ""))
        base = idx.get((ds, baseline_variant))
        if base is None:
            row["avg_context_tokens_delta"] = 0.0
            row["delta_generation_ms"] = 0.0
            row["delta_em"] = 0.0
            row["delta_f1"] = 0.0
            row["delta_abgf"] = 0.0
            row["delta_output_overlap"] = 0.0
            row["recall_at_1_delta"] = 0.0
            row["recall_at_5_delta"] = 0.0
            row["recall_at_10_delta"] = 0.0
            row["retrieval_invariance_pass"] = "n/a"
            continue

        row["avg_context_tokens_delta"] = float(_safe_float(row.get("avg_context_tokens", 0.0), 0.0) - _safe_float(base.get("avg_context_tokens", 0.0), 0.0))
        row["delta_generation_ms"] = float(_safe_float(row.get("generation_ms", 0.0), 0.0) - _safe_float(base.get("generation_ms", 0.0), 0.0))
        row["delta_em"] = float(_safe_float(row.get("em", 0.0), 0.0) - _safe_float(base.get("em", 0.0), 0.0))
        row["delta_f1"] = float(_safe_float(row.get("f1", 0.0), 0.0) - _safe_float(base.get("f1", 0.0), 0.0))
        row["delta_abgf"] = float(
            _safe_float(row.get("answer_present_but_generation_fail", 0.0), 0.0)
            - _safe_float(base.get("answer_present_but_generation_fail", 0.0), 0.0)
        )
        row["delta_output_overlap"] = float(
            _safe_float(row.get("output_overlap_answer_bearing", 0.0), 0.0)
            - _safe_float(base.get("output_overlap_answer_bearing", 0.0), 0.0)
        )

        d1 = _safe_float(row.get("recall_at_1", 0.0), 0.0) - _safe_float(base.get("recall_at_1", 0.0), 0.0)
        d5 = _safe_float(row.get("recall_at_5", 0.0), 0.0) - _safe_float(base.get("recall_at_5", 0.0), 0.0)
        d10 = _safe_float(row.get("recall_at_10", 0.0), 0.0) - _safe_float(base.get("recall_at_10", 0.0), 0.0)
        row["recall_at_1_delta"] = float(d1)
        row["recall_at_5_delta"] = float(d5)
        row["recall_at_10_delta"] = float(d10)
        row["retrieval_invariance_pass"] = "pass" if (abs(d1) <= 1.0e-9 and abs(d5) <= 1.0e-9 and abs(d10) <= 1.0e-9) else "fail"

    retrieval_invariance_rows = [
        {
            "dataset": str(r.get("dataset", "")),
            "variant": str(r.get("variant", "")),
            "recall_at_1": _safe_float(r.get("recall_at_1", 0.0), 0.0),
            "recall_at_5": _safe_float(r.get("recall_at_5", 0.0), 0.0),
            "recall_at_10": _safe_float(r.get("recall_at_10", 0.0), 0.0),
            "recall_at_1_delta": _safe_float(r.get("recall_at_1_delta", 0.0), 0.0),
            "recall_at_5_delta": _safe_float(r.get("recall_at_5_delta", 0.0), 0.0),
            "recall_at_10_delta": _safe_float(r.get("recall_at_10_delta", 0.0), 0.0),
            "retrieval_invariance_pass": str(r.get("retrieval_invariance_pass", "n/a")),
        }
        for r in metrics_rows
    ]

    retrieval_invariance_violations = [
        r for r in retrieval_invariance_rows if str(r.get("retrieval_invariance_pass", "")).lower() == "fail"
    ]

    retrieval_fixed_violations = []
    for r in metrics_rows:
        if _safe_float(r.get("retrieval_fixed_expected", 0.0), 0.0) < 0.5:
            continue
        if _safe_float(r.get("retrieval_source_precomputed_rate", 0.0), 0.0) < 0.999999:
            retrieval_fixed_violations.append(
                {
                    "dataset": str(r.get("dataset", "")),
                    "variant": str(r.get("variant", "")),
                    "retrieval_source_precomputed_rate": _safe_float(r.get("retrieval_source_precomputed_rate", 0.0), 0.0),
                }
            )

    no_prediction_rewrite_violations = [
        {
            "dataset": str(r.get("dataset", "")),
            "variant": str(r.get("variant", "")),
            "qa_utilization_changed_rate": _safe_float(r.get("qa_utilization_changed_rate", 0.0), 0.0),
            "no_prediction_rewrite_violation_rate": _safe_float(r.get("no_prediction_rewrite_violation_rate", 0.0), 0.0),
        }
        for r in metrics_rows
        if str(r.get("no_prediction_rewrite_pass", "pass")).lower() != "pass"
    ]

    delta_rows = []
    for r in metrics_rows:
        ds = str(r.get("dataset", ""))
        vv = str(r.get("variant", ""))
        base = idx.get((ds, baseline_variant))
        if base is None:
            continue
        delta_rows.append(
            {
                "dataset": ds,
                "variant": vv,
                "delta_em": _safe_float(r.get("delta_em", 0.0), 0.0),
                "delta_f1": _safe_float(r.get("delta_f1", 0.0), 0.0),
                "delta_abgf": _safe_float(r.get("delta_abgf", 0.0), 0.0),
                "delta_output_overlap": _safe_float(r.get("delta_output_overlap", 0.0), 0.0),
                "avg_context_tokens_delta": _safe_float(r.get("avg_context_tokens_delta", 0.0), 0.0),
                "delta_generation_ms": _safe_float(r.get("delta_generation_ms", 0.0), 0.0),
            }
        )
    delta_rows.sort(key=lambda x: (x.get("dataset", ""), x.get("variant", "")))

    # Prompt variant summary table (one row per variant)
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in metrics_rows:
        by_variant[str(r.get("variant", ""))].append(r)

    prompt_variant_rows = []
    for vv, rows in sorted(by_variant.items(), key=lambda kv: kv[0]):
        prompt_variant_rows.append(
            {
                "variant": vv,
                "added_instruction": _mode_text([str(r.get("added_instruction", "")) for r in rows], default="none"),
                "one_shot_enabled": 1.0 if _mean([_safe_float(r.get("one_shot_enabled", 0.0), 0.0) for r in rows]) >= 0.5 else 0.0,
                "title_grounded_format_enabled": 1.0
                if _mean([_safe_float(r.get("title_grounded_format_enabled", 0.0), 0.0) for r in rows]) >= 0.5
                else 0.0,
                "final_answer_only": 1.0 if _mean([_safe_float(r.get("final_answer_only", 1.0), 1.0) for r in rows]) >= 0.5 else 0.0,
                "reasoning_forbidden": 1.0
                if _mean([_safe_float(r.get("reasoning_forbidden", 1.0), 1.0) for r in rows]) >= 0.5
                else 0.0,
            }
        )

    # Decision: focus on bridge instruction as main candidate.
    decision_rows = []
    for ds in sorted(set([str(r.get("dataset", "")) for r in metrics_rows])):
        ds_rows = [r for r in metrics_rows if str(r.get("dataset", "")) == ds]
        if not ds_rows:
            continue
        winner = sorted(ds_rows, key=lambda x: (_safe_float(x.get("f1", 0.0), 0.0), -_safe_float(x.get("answer_present_but_generation_fail", 1.0), 1.0)), reverse=True)[0]
        decision_rows.append(
            {
                "dataset": ds,
                "winner": str(winner.get("variant", "")),
                "winner_f1": _safe_float(winner.get("f1", 0.0), 0.0),
                "winner_abgf": _safe_float(winner.get("answer_present_but_generation_fail", 0.0), 0.0),
            }
        )

    bridge_mainline_decision = "hold"
    target_datasets = ["hotpotqa", "2wikimultihopqa"]
    if all((ds, baseline_variant) in idx and (ds, "light_separator_bridge_instruction") in idx for ds in target_datasets):
        hot_base = idx[("hotpotqa", baseline_variant)]
        hot_v1 = idx[("hotpotqa", "light_separator_bridge_instruction")]
        wk_base = idx[("2wikimultihopqa", baseline_variant)]
        wk_v1 = idx[("2wikimultihopqa", "light_separator_bridge_instruction")]

        ctx_hot_ratio = safe_div(
            _safe_float(hot_v1.get("avg_context_tokens", 0.0), 0.0) - _safe_float(hot_base.get("avg_context_tokens", 0.0), 0.0),
            max(1e-12, _safe_float(hot_base.get("avg_context_tokens", 0.0), 0.0)),
        )
        ctx_wk_ratio = safe_div(
            _safe_float(wk_v1.get("avg_context_tokens", 0.0), 0.0) - _safe_float(wk_base.get("avg_context_tokens", 0.0), 0.0),
            max(1e-12, _safe_float(wk_base.get("avg_context_tokens", 0.0), 0.0)),
        )
        gen_hot_ratio = safe_div(
            _safe_float(hot_v1.get("generation_ms", 0.0), 0.0) - _safe_float(hot_base.get("generation_ms", 0.0), 0.0),
            max(1e-12, _safe_float(hot_base.get("generation_ms", 0.0), 0.0)),
        )
        gen_wk_ratio = safe_div(
            _safe_float(wk_v1.get("generation_ms", 0.0), 0.0) - _safe_float(wk_base.get("generation_ms", 0.0), 0.0),
            max(1e-12, _safe_float(wk_base.get("generation_ms", 0.0), 0.0)),
        )

        cond = [
            _safe_float(hot_v1.get("f1", 0.0), 0.0) >= _safe_float(hot_base.get("f1", 0.0), 0.0),
            _safe_float(wk_v1.get("f1", 0.0), 0.0) > _safe_float(wk_base.get("f1", 0.0), 0.0),
            (_safe_float(hot_v1.get("answer_present_but_generation_fail", 0.0), 0.0) + _safe_float(wk_v1.get("answer_present_but_generation_fail", 0.0), 0.0))
            < (_safe_float(hot_base.get("answer_present_but_generation_fail", 0.0), 0.0) + _safe_float(wk_base.get("answer_present_but_generation_fail", 0.0), 0.0)),
            (_safe_float(hot_v1.get("output_overlap_answer_bearing", 0.0), 0.0) + _safe_float(wk_v1.get("output_overlap_answer_bearing", 0.0), 0.0))
            >= (_safe_float(hot_base.get("output_overlap_answer_bearing", 0.0), 0.0) + _safe_float(wk_base.get("output_overlap_answer_bearing", 0.0), 0.0)),
            ctx_hot_ratio <= 0.05,
            ctx_wk_ratio <= 0.05,
            _safe_float(hot_v1.get("F1_per_1k_context_tokens", 0.0), 0.0) >= _safe_float(hot_base.get("F1_per_1k_context_tokens", 0.0), 0.0),
            _safe_float(wk_v1.get("F1_per_1k_context_tokens", 0.0), 0.0) >= _safe_float(wk_base.get("F1_per_1k_context_tokens", 0.0), 0.0),
            gen_hot_ratio <= 0.10,
            gen_wk_ratio <= 0.10,
            len(retrieval_invariance_violations) == 0,
            len(no_prediction_rewrite_violations) == 0,
            _safe_float(hot_v1.get("reasoning_output_violation_rate", 0.0), 0.0) <= 1.0e-12,
            _safe_float(wk_v1.get("reasoning_output_violation_rate", 0.0), 0.0) <= 1.0e-12,
            _safe_float(hot_v1.get("think_tag_violation_rate", 0.0), 0.0) <= 1.0e-12,
            _safe_float(wk_v1.get("think_tag_violation_rate", 0.0), 0.0) <= 1.0e-12,
        ]
        bridge_mainline_decision = "promote" if all(cond) else "hold"

    variant_decision_rows = [
        {
            "scope": "mainline_candidate",
            "candidate": "light_separator_bridge_instruction",
            "decision": bridge_mainline_decision,
            "note": "primary candidate from prompt-interface bridge resolution round",
        },
        {
            "scope": "stopped_variants",
            "candidate": "",
            "decision": "",
            "note": "",
        },
    ]

    group_tables = group_markdown_tables(records=metrics_rows, registry=registry)
    write_markdown(
        str(out_metrics_md),
        markdown_document_from_group_tables(group_tables, title="Prompt Interface Bridge Resolution Round"),
    )

    metrics_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_root": str(round_root),
        "baseline_variant": baseline_variant,
        "tokenizer_mode": token_counter.mode,
        "tokenizer_name": token_counter.tokenizer_name,
        "records": metrics_rows,
        "delta_rows": delta_rows,
        "prompt_variant_rows": prompt_variant_rows,
        "variant_decision_rows": variant_decision_rows,
        "retrieval_invariance_rows": retrieval_invariance_rows,
        "metric_groups": {group: [spec.name for spec in registry.enabled_by_group(group)] for group in registry.groups()},
        "integrity": {
            "retrieval_invariance_violations": retrieval_invariance_violations,
            "retrieval_fixed_violations": retrieval_fixed_violations,
            "no_prediction_rewrite_violations": no_prediction_rewrite_violations,
            "failures_count": int(len(failures)),
            "failures": failures,
        },
        "run_context": {
            "git_commit_hash": _cmd(["git", "rev-parse", "HEAD"]),
            "active_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        },
    }
    write_json(str(out_metrics_json), metrics_json)

    retrieval_invariance_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_variant": baseline_variant,
        "rows": retrieval_invariance_rows,
        "violations": retrieval_invariance_violations,
        "pass": bool(len(retrieval_invariance_violations) == 0),
    }
    write_json(str(out_retrieval_inv_json), retrieval_invariance_report)

    write_markdown(
        str(out_retrieval_inv_md),
        build_flat_markdown_table(
            records=retrieval_invariance_rows,
            headers=[
                "dataset",
                "variant",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_5_delta",
                "recall_at_10_delta",
                "retrieval_invariance_pass",
            ],
            registry=registry,
        )
        + "\n",
    )

    write_markdown(
        str(out_breakdown_md),
        build_flat_markdown_table(
            records=metrics_rows,
            headers=[
                "dataset",
                "variant",
                "abgf_support_present_generation_fail",
                "abgf_answer_surface_present_generation_fail",
                "support_present_answer_surface_absent",
                "answer_surface_present_but_low_overlap",
            ],
            registry=registry,
        )
        + "\n",
    )

    write_markdown(
        str(out_context_eff_md),
        build_flat_markdown_table(
            records=metrics_rows,
            headers=[
                "dataset",
                "variant",
                "avg_context_tokens",
                "median_context_tokens",
                "p90_context_tokens",
                "answer_token_density",
                "sf_token_density",
                "F1_per_1k_context_tokens",
                "avg_context_tokens_delta",
            ],
            registry=registry,
        )
        + "\n",
    )

    write_markdown(
        str(out_delta_md),
        build_flat_markdown_table(
            records=delta_rows,
            headers=[
                "dataset",
                "variant",
                "delta_em",
                "delta_f1",
                "delta_abgf",
                "delta_output_overlap",
                "avg_context_tokens_delta",
                "delta_generation_ms",
            ],
            registry=registry,
        )
        + "\n",
    )

    write_markdown(
        str(out_prompt_variant_md),
        build_flat_markdown_table(
            records=prompt_variant_rows,
            headers=[
                "variant",
                "added_instruction",
                "one_shot_enabled",
                "title_grounded_format_enabled",
                "final_answer_only",
                "reasoning_forbidden",
            ],
            registry=registry,
        )
        + "\n",
    )

    write_markdown(
        str(out_format_violation_md),
        build_flat_markdown_table(
            records=metrics_rows,
            headers=[
                "dataset",
                "variant",
                "reasoning_output_violation_rate",
                "think_tag_violation_rate",
                "answer_length_avg",
                "exact_answer_format_violation_rate",
                "format_violation_rate",
            ],
            registry=registry,
        )
        + "\n",
    )

    notes = metric_definition_notes(
        registry=registry,
        tokenizer_mode=token_counter.mode,
        tokenizer_name=token_counter.tokenizer_name,
        optional_notes=[
            "Round compares prompt-interface variants on retrieval-fixed evidence sets.",
            "Baseline variant for deltas and invariance is light_separator_render.",
            "Reasoning-output violation checks raw generation text for Thought:/Reasoning: markers and hidden-think tags.",
            "exact_answer_format_violation_rate flags empty/multiline/overlong outputs and explicit reasoning-form answers.",
            "no prediction rewrite pass requires qa_utilization_changed_rate == 0.",
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
                    "em",
                    "f1",
                    "answer_present_but_generation_fail",
                    "output_overlap_answer_bearing",
                    "avg_context_tokens",
                    "generation_ms",
                    "total_ms",
                ],
                registry=registry,
            )
        )
        print("")

        print("## ABGF Breakdown Table")
        print(
            build_flat_markdown_table(
                records=metrics_rows,
                headers=[
                    "dataset",
                    "variant",
                    "abgf_support_present_generation_fail",
                    "abgf_answer_surface_present_generation_fail",
                    "support_present_answer_surface_absent",
                    "answer_surface_present_but_low_overlap",
                ],
                registry=registry,
            )
        )
        print("")

        print("## Context Efficiency Table")
        print(
            build_flat_markdown_table(
                records=metrics_rows,
                headers=[
                    "dataset",
                    "variant",
                    "avg_context_tokens",
                    "median_context_tokens",
                    "p90_context_tokens",
                    "answer_token_density",
                    "sf_token_density",
                    "F1_per_1k_context_tokens",
                    "avg_context_tokens_delta",
                ],
                registry=registry,
            )
        )
        print("")

        print("## Prompt Variant Table")
        print(
            build_flat_markdown_table(
                records=prompt_variant_rows,
                headers=[
                    "variant",
                    "added_instruction",
                    "one_shot_enabled",
                    "title_grounded_format_enabled",
                    "final_answer_only",
                    "reasoning_forbidden",
                ],
                registry=registry,
            )
        )
        print("")

        print("## Retrieval Invariance Table")
        print(
            build_flat_markdown_table(
                records=retrieval_invariance_rows,
                headers=[
                    "dataset",
                    "variant",
                    "recall_at_1",
                    "recall_at_5",
                    "recall_at_10",
                    "recall_at_5_delta",
                    "recall_at_10_delta",
                    "retrieval_invariance_pass",
                ],
                registry=registry,
            )
        )
        print("")

        print("## Format Violation Table")
        print(
            build_flat_markdown_table(
                records=metrics_rows,
                headers=[
                    "dataset",
                    "variant",
                    "reasoning_output_violation_rate",
                    "think_tag_violation_rate",
                    "answer_length_avg",
                    "exact_answer_format_violation_rate",
                    "format_violation_rate",
                ],
                registry=registry,
            )
        )
        print("")

        print("## Variant Decision Table")
        print(
            build_flat_markdown_table(
                records=variant_decision_rows,
                headers=["scope", "candidate", "decision", "note"],
                registry=registry,
            )
        )
        print("")

    print(str(round_root))


if __name__ == "__main__":
    main()
