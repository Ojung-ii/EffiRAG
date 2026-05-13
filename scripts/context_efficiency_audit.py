#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from effirag.eval_adapters import (
    adapt_effirag_from_summary,
    adapt_hipporag2_from_outputs,
    discover_effirag_summary_path,
    load_qa_index,
)
from effirag.eval_metrics import (
    MetricRegistry,
    aggregate_context_efficiency_metrics,
    build_context_efficiency_metric_registry,
    metric_definition_notes,
    validate_context_efficiency_metrics,
    write_json,
    write_tsv,
)
from effirag.eval_reports import (
    build_flat_markdown_table,
    build_group_table_markdown,
    group_markdown_tables,
    markdown_document_from_group_tables,
    write_markdown,
)
from effirag.qa_metrics import exact_match_score, token_f1_score
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


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _parse_csv(raw: str) -> List[str]:
    out = []
    for token in str(raw or "").split(","):
        tok = token.strip()
        if tok:
            out.append(tok)
    return out


def _normalize_for_match(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


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
        try:
            return str(getattr(self._tokenizer, "name_or_path", self.model_name) or self.model_name)
        except Exception:
            return str(self.model_name)


def _support_match(
    rendered_context: str,
    support_text: str,
    *,
    fuzzy_threshold: float = 0.0,
) -> bool:
    rendered = _strip(rendered_context)
    support = _strip(support_text)
    if not rendered or not support:
        return False

    if support in rendered:
        return True

    rendered_norm = _normalize_for_match(rendered)
    support_norm = _normalize_for_match(support)
    if support_norm and support_norm in rendered_norm:
        return True

    if fuzzy_threshold > 0.0 and support_norm and rendered_norm:
        ratio = SequenceMatcher(a=support_norm, b=rendered_norm).ratio()
        return bool(ratio >= float(fuzzy_threshold))
    return False


def _answer_bearing_match(text: str, gold_answer: str) -> bool:
    t = _normalize_for_match(text)
    a = _normalize_for_match(gold_answer)
    if (not t) or (not a):
        return False
    return bool(a in t)


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        txt = _strip(value)
        return [txt] if txt else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            txt = _strip(item)
            if txt:
                out.append(txt)
        return out
    return []


def compute_query_context_metrics(
    record: Mapping[str, Any],
    token_counter: TokenCounter,
    *,
    fuzzy_support_threshold: float = 0.0,
) -> Dict[str, Any]:
    row = dict(record or {})
    rendered_context = _strip(row.get("rendered_context"))
    rendered_units = _as_text_list(row.get("rendered_items"))
    if (not rendered_units) and rendered_context:
        rendered_units = [rendered_context]
    raw_candidates = _as_text_list(row.get("raw_candidate_context"))

    rendered_context_tokens = int(token_counter.count(rendered_context))
    raw_candidate_tokens: Optional[int]
    if raw_candidates:
        raw_candidate_tokens = int(sum(token_counter.count(x) for x in raw_candidates))
    else:
        raw_candidate_tokens = None

    gold_supports = _as_text_list(row.get("gold_supports"))
    support_match_count = 0
    matched_support_tokens = 0
    for support_text in gold_supports:
        matched = _support_match(
            rendered_context,
            support_text,
            fuzzy_threshold=fuzzy_support_threshold,
        )
        if matched:
            support_match_count += 1
            matched_support_tokens += int(token_counter.count(support_text))
    matched_support_tokens = min(int(matched_support_tokens), int(rendered_context_tokens))

    gold_answer = _strip(row.get("gold_answer"))
    answer_bearing_hits = 0
    answer_bearing_tokens = 0
    for item in rendered_units:
        if _answer_bearing_match(item, gold_answer):
            answer_bearing_hits += 1
            answer_bearing_tokens += int(token_counter.count(item))
    answer_bearing_tokens = min(int(answer_bearing_tokens), int(rendered_context_tokens))

    support_density = float(safe_div(float(matched_support_tokens), float(rendered_context_tokens)))
    answer_density = float(safe_div(float(answer_bearing_tokens), float(rendered_context_tokens)))

    compression_ratio: Optional[float] = None
    context_reduction_rate: Optional[float] = None
    if raw_candidate_tokens is not None and raw_candidate_tokens > 0:
        compression_ratio = float(safe_div(float(rendered_context_tokens), float(raw_candidate_tokens)))
        context_reduction_rate = float(1.0 - compression_ratio)

    pred = _strip(row.get("prediction"))
    query_em = exact_match_score(pred, gold_answer) if gold_answer else 0.0
    query_f1 = token_f1_score(pred, gold_answer) if gold_answer else 0.0

    timing = dict(row.get("timing", {}) or {})
    retrieval_ms = _safe_float(timing.get("retrieval_ms", 0.0), 0.0)
    generation_ms = _safe_float(timing.get("generation_ms", 0.0), 0.0)
    total_ms = _safe_float(timing.get("total_ms", 0.0), 0.0)
    if total_ms <= 0.0 and (retrieval_ms > 0.0 or generation_ms > 0.0):
        total_ms = float(retrieval_ms + generation_ms)

    return {
        "dataset": _strip(row.get("dataset")),
        "variant": _strip(row.get("variant")),
        "qid": _strip(row.get("qid")),
        "rendered_context_tokens": float(rendered_context_tokens),
        "raw_candidate_tokens": float(raw_candidate_tokens) if raw_candidate_tokens is not None else None,
        "compression_ratio": compression_ratio,
        "context_reduction_rate": context_reduction_rate,
        "support_total_count": float(len(gold_supports)),
        "support_match_count": float(support_match_count),
        "support_hits": float(support_match_count),
        "matched_support_tokens": float(matched_support_tokens),
        "supporting_fact_token_density": float(support_density),
        "non_support_token_ratio": float(max(0.0, 1.0 - support_density)),
        "answer_bearing_hits": float(answer_bearing_hits),
        "answer_bearing_tokens": float(answer_bearing_tokens),
        "answer_bearing_token_density": float(answer_density),
        "non_answer_bearing_token_ratio": float(max(0.0, 1.0 - answer_density)),
        "answer_bearing_match": 1.0 if answer_bearing_hits > 0 else 0.0,
        "missing_rendered_context": bool(rendered_context_tokens <= 0),
        "query_em": float(query_em),
        "query_f1": float(query_f1),
        "retrieval_ms": float(retrieval_ms),
        "generation_ms": float(generation_ms),
        "total_ms": float(total_ms),
    }


def _expected_count(total_dataset_count: int, n_samples: int) -> int:
    if n_samples > 0:
        return min(int(total_dataset_count), int(n_samples))
    return int(total_dataset_count)


def _build_delta_rows(
    rows: Sequence[Mapping[str, Any]],
    baseline_variant: str,
    metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    all_rows = [dict(r or {}) for r in list(rows or [])]
    idx = {(str(r.get("dataset", "")), str(r.get("variant", ""))): r for r in all_rows}
    out: List[Dict[str, Any]] = []
    for row in all_rows:
        ds = str(row.get("dataset", ""))
        variant = str(row.get("variant", ""))
        base = idx.get((ds, baseline_variant))
        if base is None:
            continue
        delta = {
            "dataset": ds,
            "variant": variant,
            "baseline_variant": baseline_variant,
        }
        for name in list(metric_names or []):
            lhs = _safe_float(row.get(name, 0.0), 0.0)
            rhs = _safe_float(base.get(name, 0.0), 0.0)
            delta[f"delta_{name}"] = float(lhs - rhs)
        out.append(delta)
    out.sort(key=lambda x: (x.get("dataset", ""), x.get("variant", "")))
    return out


def _write_run_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = [dict(r or {}) for r in list(records or [])]
    if not rows:
        path.write_text(
            "dataset\tvariant\tsource\tstatus\tn_queries\tsummary_path\tquery_path\terror_message\n",
            encoding="utf-8",
        )
        return
    headers = [
        "dataset",
        "variant",
        "source",
        "status",
        "n_queries",
        "summary_path",
        "query_path",
        "error_message",
    ]
    write_tsv(str(path), headers, rows)


def _load_run_payload(
    *,
    dataset: str,
    variant: str,
    qa_path: str,
    effirag_output_root: str,
    hippo_output_root: str,
    n_samples: int,
) -> Dict[str, Any]:
    qa_index = load_qa_index(qa_path)
    vv = variant.strip().lower()
    if vv in {"hipporag2", "hipporag", "hipporag2_tangent", "hipporag__tangent__tempna"}:
        return adapt_hipporag2_from_outputs(
            hippo_output_root=hippo_output_root,
            dataset=dataset,
            variant=variant,
            qa_index=qa_index,
            n_samples=n_samples,
        )

    summary_path, meta = discover_effirag_summary_path(
        effirag_output_root=effirag_output_root,
        dataset=dataset,
        variant=variant,
    )
    if summary_path is None:
        raise FileNotFoundError(
            f"EffiRAG summary not found for dataset={dataset}, variant={variant}. warnings={meta.get('warnings', [])}"
        )
    return adapt_effirag_from_summary(
        summary_path=str(summary_path),
        dataset=dataset,
        variant=variant,
        qa_index=qa_index,
        n_samples=n_samples,
    )


def _validate_em_f1_consistency(
    *,
    aggregated: Mapping[str, Any],
    per_query: Sequence[Mapping[str, Any]],
    tolerance: float = 0.03,
) -> List[str]:
    seq = list(per_query or [])
    if not seq:
        return []
    em_ref = _safe_float(aggregated.get("em", 0.0), 0.0)
    f1_ref = _safe_float(aggregated.get("f1", 0.0), 0.0)
    em_re = sum(_safe_float(r.get("query_em", 0.0), 0.0) for r in seq) / float(len(seq))
    f1_re = sum(_safe_float(r.get("query_f1", 0.0), 0.0) for r in seq) / float(len(seq))

    issues: List[str] = []
    if abs(float(em_ref) - float(em_re)) > float(tolerance):
        issues.append(
            f"EM mismatch vs recomputed: summary={em_ref:.4f}, recomputed={em_re:.4f}, tol={tolerance:.4f}"
        )
    if abs(float(f1_ref) - float(f1_re)) > float(tolerance):
        issues.append(
            f"F1 mismatch vs recomputed: summary={f1_ref:.4f}, recomputed={f1_re:.4f}, tol={tolerance:.4f}"
        )
    return issues


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    parser.add_argument("--variants", default="champion_config,hipporag2,baseline_mid_reconfirm,ref_reconfirm")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--qa-root", default="data/qa")
    parser.add_argument("--corpus-root", default="")
    parser.add_argument("--effirag-output-root", required=True)
    parser.add_argument("--hippo-output-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--graph-cache-dir", default="")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--print-tables", default="true")
    parser.add_argument("--fuzzy-support-threshold", type=float, default=0.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    datasets = _parse_csv(args.datasets)
    variants = _parse_csv(args.variants)
    if not datasets:
        raise SystemExit("empty --datasets")
    if not variants:
        raise SystemExit("empty --variants")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    round_root = Path(args.output_root).resolve() / run_stamp
    round_root.mkdir(parents=True, exist_ok=True)

    out_metrics_json = round_root / "context_efficiency_metrics.json"
    out_metrics_md = round_root / "context_efficiency_metrics.md"
    out_delta_md = round_root / "context_efficiency_delta_table.md"
    out_token_md = round_root / "context_efficiency_token_table.md"
    out_density_md = round_root / "context_efficiency_density_table.md"
    out_qa_md = round_root / "context_efficiency_qa_table.md"
    out_diag_json = round_root / "context_efficiency_diagnostics.json"
    out_defs_md = round_root / "metric_definition_notes.md"
    out_run_records = round_root / "run_records.tsv"
    out_failures = round_root / "failures.json"
    out_env = round_root / "environment_snapshot.json"

    token_counter = TokenCounter(args.model_name)
    registry: MetricRegistry = build_context_efficiency_metric_registry()

    env_snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "git_commit_hash": _cmd(["git", "rev-parse", "HEAD"]),
        "active_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "cwd": str(Path.cwd().resolve()),
        "datasets": datasets,
        "variants": variants,
        "n_samples": int(args.n_samples),
        "qa_root": str(Path(args.qa_root).resolve()),
        "corpus_root": str(args.corpus_root),
        "effirag_output_root": str(Path(args.effirag_output_root).resolve()),
        "hippo_output_root": str(Path(args.hippo_output_root).resolve()),
        "graph_cache_dir": str(args.graph_cache_dir),
        "llm_base_url": str(args.llm_base_url),
        "llm_api_key": "EMPTY" if str(args.llm_api_key) == "EMPTY" else "***",
        "model_name": str(args.model_name),
        "tokenizer_mode": str(token_counter.mode),
        "tokenizer_name": str(token_counter.tokenizer_name),
    }
    write_json(str(out_env), env_snapshot)

    rows: List[Dict[str, Any]] = []
    diagnostics_runs: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    qa_size_by_dataset: Dict[str, int] = {}
    for ds in datasets:
        qa_path = Path(args.qa_root).resolve() / f"{ds}.json"
        if qa_path.exists():
            qa_size_by_dataset[ds] = len(load_qa_index(str(qa_path)).by_index)
        else:
            qa_size_by_dataset[ds] = 0

    for dataset in datasets:
        qa_path = Path(args.qa_root).resolve() / f"{dataset}.json"
        if not qa_path.exists():
            err = f"missing QA file: {qa_path}"
            for variant in variants:
                rec = {
                    "dataset": dataset,
                    "variant": variant,
                    "source": "unknown",
                    "status": "failed",
                    "n_queries": "0",
                    "summary_path": "",
                    "query_path": "",
                    "error_message": err,
                }
                run_records.append(rec)
                failures.append(rec)
            continue

        for variant in variants:
            source = "hipporag2" if variant.strip().lower().startswith("hipporag") else "effirag"
            try:
                payload = _load_run_payload(
                    dataset=dataset,
                    variant=variant,
                    qa_path=str(qa_path),
                    effirag_output_root=args.effirag_output_root,
                    hippo_output_root=args.hippo_output_root,
                    n_samples=int(args.n_samples),
                )
                records = list(payload.get("records", []) or [])
                per_query = [
                    compute_query_context_metrics(
                        rec,
                        token_counter,
                        fuzzy_support_threshold=float(args.fuzzy_support_threshold or 0.0),
                    )
                    for rec in records
                ]
                agg = aggregate_context_efficiency_metrics(
                    per_query,
                    base_metrics=dict(payload.get("summary_metrics", {}) or {}),
                )
                agg["dataset"] = str(dataset)
                agg["variant"] = str(variant)

                expected_count = _expected_count(qa_size_by_dataset.get(dataset, 0), int(args.n_samples))
                integrity_issues = validate_context_efficiency_metrics(agg)
                if expected_count > 0 and int(len(records)) != int(expected_count):
                    integrity_issues.append(
                        f"query count mismatch: expected={expected_count}, actual={len(records)}"
                    )
                summary_n_samples = _safe_int(payload.get("summary_n_samples", 0), 0)
                if summary_n_samples <= 0 or int(summary_n_samples) == int(len(records)):
                    integrity_issues.extend(
                        _validate_em_f1_consistency(
                            aggregated=agg,
                            per_query=per_query,
                            tolerance=0.03,
                        )
                    )

                rows.append(agg)
                diag_row = {
                    "dataset": dataset,
                    "variant": variant,
                    "source": source,
                    "n_queries": int(len(records)),
                    "summary_n_samples": _safe_int(payload.get("summary_n_samples", 0), 0),
                    "summary_path": str(payload.get("summary_path", "")),
                    "query_path": str(payload.get("query_path", "")),
                    "warnings": list(payload.get("warnings", []) or []),
                    "integrity_issues": integrity_issues,
                    "missing_rendered_context_rate": _safe_float(agg.get("missing_rendered_context_rate", 0.0), 0.0),
                }
                diagnostics_runs.append(diag_row)

                run_records.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "source": source,
                        "status": "ok",
                        "n_queries": str(len(records)),
                        "summary_path": str(payload.get("summary_path", "")),
                        "query_path": str(payload.get("query_path", "")),
                        "error_message": "",
                    }
                )
            except Exception as exc:
                rec = {
                    "dataset": dataset,
                    "variant": variant,
                    "source": source,
                    "status": "failed",
                    "n_queries": "0",
                    "summary_path": "",
                    "query_path": "",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
                run_records.append(rec)
                failures.append(rec)
                diagnostics_runs.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "source": source,
                        "n_queries": 0,
                        "summary_path": "",
                        "query_path": "",
                        "warnings": [],
                        "integrity_issues": [rec["error_message"]],
                        "missing_rendered_context_rate": 1.0,
                    }
                )

    rows.sort(key=lambda x: (str(x.get("dataset", "")), str(x.get("variant", ""))))
    _write_run_records(out_run_records, run_records)
    write_json(str(out_failures), {"count": len(failures), "items": failures})

    group_tables = group_markdown_tables(records=rows, registry=registry)
    full_md = markdown_document_from_group_tables(group_tables, title="Context Efficiency Audit")
    write_markdown(str(out_metrics_md), full_md)

    metrics_bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_root": str(round_root),
        "tokenizer_mode": token_counter.mode,
        "tokenizer_name": token_counter.tokenizer_name,
        "records": rows,
        "metric_groups": {group: [spec.name for spec in registry.enabled_by_group(group)] for group in registry.groups()},
        "failures_count": len(failures),
    }
    write_json(str(out_metrics_json), metrics_bundle)

    delta_metric_names = [
        name
        for name in [
            "recall_at_5",
            "supporting_fact_f1",
            "avg_context_tokens",
            "supporting_fact_token_density",
            "answer_bearing_token_density",
            "EM_per_1k_context_tokens",
            "F1_per_1k_context_tokens",
            "em",
            "f1",
            "retrieval_ms",
            "total_ms",
        ]
        if registry.get(name) is not None
    ]
    delta_rows = _build_delta_rows(rows, baseline_variant="baseline_mid_reconfirm", metric_names=delta_metric_names)
    delta_headers = ["dataset", "variant", "baseline_variant"] + [f"delta_{name}" for name in delta_metric_names]
    delta_md = build_flat_markdown_table(records=delta_rows, headers=delta_headers, registry=registry)
    write_markdown(str(out_delta_md), delta_md + "\n")

    token_groups = ["Context Efficiency Metrics", "Token-normalized Utility Metrics"]
    token_md = build_group_table_markdown(records=rows, registry=registry, groups=token_groups)
    write_markdown(str(out_token_md), token_md)

    density_metric_names = [
        name
        for name in registry.enabled_names()
        if (
            ("density" in name)
            or ("non_support_token_ratio" == name)
            or ("non_answer_bearing_token_ratio" == name)
            or ("support_text_match_rate" == name)
            or ("answer_bearing_match_rate" == name)
        )
    ]
    density_headers = ["dataset", "variant"] + density_metric_names
    density_md = build_flat_markdown_table(records=rows, headers=density_headers, registry=registry)
    write_markdown(str(out_density_md), density_md + "\n")

    qa_metric_names = [
        name
        for name in [
            "supporting_fact_precision",
            "supporting_fact_recall",
            "supporting_fact_f1",
            "context_precision",
            "em",
            "f1",
            "retrieval_ms",
            "generation_ms",
            "total_ms",
        ]
        if registry.get(name) is not None
    ]
    qa_headers = ["dataset", "variant"] + qa_metric_names
    qa_md = build_flat_markdown_table(records=rows, headers=qa_headers, registry=registry)
    write_markdown(str(out_qa_md), qa_md + "\n")

    metric_notes_extra: List[str] = []
    if token_counter.mode == "whitespace_fallback":
        metric_notes_extra.append(
            "Tokenizer loading failed; token counting used whitespace fallback (tokenizer_mode=whitespace_fallback)."
        )
    if float(args.fuzzy_support_threshold or 0.0) > 0.0:
        metric_notes_extra.append(
            f"Conservative fuzzy support matching enabled with threshold={float(args.fuzzy_support_threshold):.3f}."
        )
    else:
        metric_notes_extra.append("Support matching used exact substring + normalized whitespace substring only.")

    defs_md = metric_definition_notes(
        registry=registry,
        tokenizer_mode=token_counter.mode,
        tokenizer_name=token_counter.tokenizer_name,
        optional_notes=metric_notes_extra,
    )
    write_markdown(str(out_defs_md), defs_md)

    markdown_test_ok = bool(_strip(full_md))
    json_schema_test_ok = bool(isinstance(metrics_bundle.get("records"), list))
    metric_sanity_pass = True
    metric_sanity_issues: List[str] = []
    for row in rows:
        issues = validate_context_efficiency_metrics(row)
        if issues:
            metric_sanity_pass = False
            metric_sanity_issues.append(
                f"{row.get('dataset', '')}/{row.get('variant', '')}: " + "; ".join(issues)
            )

    diagnostics_bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_root": str(round_root),
        "tokenizer_mode": token_counter.mode,
        "tokenizer_name": token_counter.tokenizer_name,
        "runs": diagnostics_runs,
        "checks": {
            "context_efficiency_metric_unit_sanity_check": {
                "pass": bool(metric_sanity_pass),
                "issues": metric_sanity_issues,
            },
            "markdown_output_generation_test": {
                "pass": bool(markdown_test_ok),
                "details": "context_efficiency_metrics.md non-empty",
            },
            "json_export_schema_test": {
                "pass": bool(json_schema_test_ok),
                "details": "context_efficiency_metrics.json has list-valued records",
            },
        },
        "failures_count": len(failures),
    }
    write_json(str(out_diag_json), diagnostics_bundle)

    print_tables = str(args.print_tables).strip().lower() in {"1", "true", "yes", "y", "on"}
    if print_tables:
        for group in registry.groups():
            print(f"## {group}")
            print(group_tables.get(group, ""))
            print("")

    print(str(round_root))


if __name__ == "__main__":
    main()
