#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.canonical_scoring import CANONICAL_PRUNED_WEIGHT_FIELDS
from effirag.pipeline_trace import trace_active_pipeline
from effirag.types import RenderedContext, Sample
from effirag.generator import _build_qa_prompt


def _parse_csv(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _find_latest_summary(round_root: Path, stage: str, profile: str, dataset: str) -> Optional[Path]:
    base = round_root / "runs" / stage / profile / "rag" / dataset
    if not base.exists():
        return None
    cands = sorted(base.glob("*/rag_summary.json"))
    if not cands:
        return None
    return cands[-1]


def _metric(summary: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in summary and summary.get(key) is not None:
            try:
                return float(summary.get(key))
            except Exception:
                continue
    return None


def _as_rendered_context(obj: Mapping[str, Any], sample_id: str) -> RenderedContext:
    md = dict((obj or {}).get("metadata", {}) or {})
    return RenderedContext(
        sample_id=str(sample_id),
        method="effirag",
        text=str((obj or {}).get("text", "") or ""),
        sentences=[str(x) for x in ((obj or {}).get("sentences", []) or [])],
        sentence_ids=[str(x) for x in ((obj or {}).get("sentence_ids", []) or [])],
        truncated=bool((obj or {}).get("truncated", False)),
        render_mode=str((obj or {}).get("render_mode", "flat") or "flat"),
        rendered_corridor_ids=[str(x) for x in ((obj or {}).get("rendered_corridor_ids", []) or [])],
        truncated_corridor_count=int((obj or {}).get("truncated_corridor_count", 0) or 0),
        truncated_sentence_count=int((obj or {}).get("truncated_sentence_count", 0) or 0),
        retrieval_selected_sentence_ids=[str(x) for x in ((obj or {}).get("retrieval_selected_sentence_ids", []) or [])],
        metadata=md,
    )


def _sample_for_prompt(sample_id: str, question: str, answer: str) -> Sample:
    return Sample(
        qid=str(sample_id),
        question=str(question or ""),
        answer=str(answer or ""),
        contexts=[],
        supporting_facts=[],
        metadata={},
    )


def _hashes_from_query_jsonl(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "n_rows": 0,
            "context_hash": "",
            "prompt_hash": "",
            "selected_sentence_ids_hash": "",
            "rendered_sentence_ids_hash": "",
            "sample_ids_hash": "",
        }

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(dict(json.loads(raw)))
            except Exception:
                continue

    rows.sort(key=lambda r: str(r.get("sample_id", r.get("sample_index", ""))))

    context_parts: List[str] = []
    prompt_parts: List[str] = []
    selected_sid_parts: List[str] = []
    rendered_sid_parts: List[str] = []
    sample_ids: List[str] = []

    for row in rows:
        sid = str(row.get("sample_id", row.get("sample_index", "")))
        sample_ids.append(sid)

        rendered_obj = dict(row.get("rendered", {}) or {})
        rendered = _as_rendered_context(rendered_obj, sample_id=sid)

        context_parts.append(sid + "\t" + str(rendered.text))
        rendered_sid_parts.append(sid + "\t" + "|".join([str(x) for x in (row.get("rendered_sentence_ids", []) or [])]))
        selected_sid_parts.append(
            sid + "\t" + "|".join([str(x) for x in (row.get("retrieval_selected_sentence_ids", []) or [])])
        )

        sample = _sample_for_prompt(
            sample_id=sid,
            question=str(row.get("question", "") or ""),
            answer=str(row.get("answer", "") or ""),
        )
        prompt_text = _build_qa_prompt(sample=sample, rendered=rendered, cfg=None)
        prompt_parts.append(sid + "\t" + prompt_text)

    return {
        "exists": True,
        "n_rows": int(len(rows)),
        "context_hash": _sha256_text("\n".join(context_parts)),
        "prompt_hash": _sha256_text("\n".join(prompt_parts)),
        "selected_sentence_ids_hash": _sha256_text("\n".join(selected_sid_parts)),
        "rendered_sentence_ids_hash": _sha256_text("\n".join(rendered_sid_parts)),
        "sample_ids_hash": _sha256_text("\n".join(sample_ids)),
    }


def _load_cfg_for_trace(round_root: Path, stage: str, profile: str, dataset: str) -> Optional[Dict[str, Any]]:
    path = round_root / "configs" / stage / profile / dataset / "rag.yaml"
    if not path.exists():
        return None
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _mk_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(str(h) for h in headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _delta(after: Optional[float], before: Optional[float]) -> Optional[float]:
    if after is None or before is None:
        return None
    return float(after - before)


def _format_delta(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def _all_zero_deltas(metric_delta: Mapping[str, Optional[float]], keys: Iterable[str], atol: float = 1e-12) -> bool:
    for key in keys:
        vv = metric_delta.get(key)
        if vv is None:
            return False
        if abs(float(vv)) > float(atol):
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Compare canonical_copy_span before/after phase-3.5 pruning.")
    p.add_argument("--before-round", default="outputs/canonical_pipeline_simplification_phase3/20260508_193149")
    p.add_argument("--after-round", required=True)
    p.add_argument("--out-root", default="outputs/canonical_mainline_pruning_phase3_5")
    p.add_argument("--timestamp", default="")
    p.add_argument("--stage", default="stage_b_full")
    p.add_argument("--profile", default="canonical_copy_span")
    p.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    args = p.parse_args()

    before_round = (REPO_ROOT / str(args.before_round)).resolve() if not str(args.before_round).startswith("/") else Path(args.before_round).resolve()
    after_round = (REPO_ROOT / str(args.after_round)).resolve() if not str(args.after_round).startswith("/") else Path(args.after_round).resolve()
    stamp = str(args.timestamp or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (REPO_ROOT / str(args.out_root)).resolve() if not str(args.out_root).startswith("/") else Path(args.out_root).resolve()
    round_root = out_root / stamp
    report_root = round_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)

    datasets = _parse_csv(args.datasets)
    profile = str(args.profile)
    stage = str(args.stage)

    post_rows: List[Dict[str, Any]] = []
    hash_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        before_summary_path = _find_latest_summary(before_round, stage, profile, dataset)
        after_summary_path = _find_latest_summary(after_round, stage, profile, dataset)
        if before_summary_path is None or after_summary_path is None:
            post_rows.append(
                {
                    "dataset": dataset,
                    "status": "missing_summary",
                    "before_summary_path": str(before_summary_path or ""),
                    "after_summary_path": str(after_summary_path or ""),
                }
            )
            continue

        before_summary = _read_json(before_summary_path)
        after_summary = _read_json(after_summary_path)

        before_query = before_summary_path.with_name("rag_query_results.jsonl")
        after_query = after_summary_path.with_name("rag_query_results.jsonl")
        before_hash = _hashes_from_query_jsonl(before_query)
        after_hash = _hashes_from_query_jsonl(after_query)

        metric_before = {
            "Recall@1": _metric(before_summary, "recall_at_1", "R@1"),
            "Recall@5": _metric(before_summary, "recall_at_5", "R@5"),
            "Recall@10": _metric(before_summary, "recall_at_10", "R@10"),
            "EM": _metric(before_summary, "em"),
            "F1": _metric(before_summary, "f1"),
            "avg_context_tokens": _metric(before_summary, "avg_context_tokens", "prompt_tokens_avg"),
            "retrieval_ms": _metric(before_summary, "retrieval_ms", "retrieval_latency_ms"),
            "equivalent_evidence_coverage": _metric(before_summary, "equivalent_evidence_coverage"),
        }
        metric_after = {
            "Recall@1": _metric(after_summary, "recall_at_1", "R@1"),
            "Recall@5": _metric(after_summary, "recall_at_5", "R@5"),
            "Recall@10": _metric(after_summary, "recall_at_10", "R@10"),
            "EM": _metric(after_summary, "em"),
            "F1": _metric(after_summary, "f1"),
            "avg_context_tokens": _metric(after_summary, "avg_context_tokens", "prompt_tokens_avg"),
            "retrieval_ms": _metric(after_summary, "retrieval_ms", "retrieval_latency_ms"),
            "equivalent_evidence_coverage": _metric(after_summary, "equivalent_evidence_coverage"),
        }
        metric_delta = {k: _delta(metric_after.get(k), metric_before.get(k)) for k in metric_before.keys()}

        hash_equal = {
            "sample_ids_hash_equal": before_hash.get("sample_ids_hash") == after_hash.get("sample_ids_hash"),
            "context_hash_equal": before_hash.get("context_hash") == after_hash.get("context_hash"),
            "prompt_hash_equal": before_hash.get("prompt_hash") == after_hash.get("prompt_hash"),
            "selected_sentence_ids_hash_equal": before_hash.get("selected_sentence_ids_hash")
            == after_hash.get("selected_sentence_ids_hash"),
            "rendered_sentence_ids_hash_equal": before_hash.get("rendered_sentence_ids_hash")
            == after_hash.get("rendered_sentence_ids_hash"),
        }

        core_delta_zero = _all_zero_deltas(
            metric_delta,
            [
                "Recall@1",
                "Recall@5",
                "Recall@10",
                "EM",
                "F1",
                "avg_context_tokens",
                "equivalent_evidence_coverage",
            ],
            atol=1e-12,
        )
        decision = "accept" if (core_delta_zero and all(bool(v) for v in hash_equal.values())) else "investigate"

        post_rows.append(
            {
                "dataset": dataset,
                "status": "ok",
                "before_summary_path": str(before_summary_path),
                "after_summary_path": str(after_summary_path),
                "before_query_path": str(before_query),
                "after_query_path": str(after_query),
                "metric_before": metric_before,
                "metric_after": metric_after,
                "metric_delta": metric_delta,
                "decision": decision,
            }
        )
        hash_rows.append(
            {
                "dataset": dataset,
                "before_hash": before_hash,
                "after_hash": after_hash,
                "hash_equal": hash_equal,
            }
        )

        cfg = _load_cfg_for_trace(after_round, stage, profile, dataset)
        if cfg is not None:
            trace_rows.append(
                {
                    "dataset": dataset,
                    "trace": trace_active_pipeline(cfg, dataset=dataset, profile=profile),
                }
            )

    post_payload = {
        "before_round": str(before_round),
        "after_round": str(after_round),
        "stage": stage,
        "profile": profile,
        "datasets": datasets,
        "rows": post_rows,
    }
    hash_payload = {
        "before_round": str(before_round),
        "after_round": str(after_round),
        "stage": stage,
        "profile": profile,
        "datasets": datasets,
        "rows": hash_rows,
    }
    trace_payload = {
        "after_round": str(after_round),
        "stage": stage,
        "profile": profile,
        "datasets": datasets,
        "rows": trace_rows,
    }
    pruned_terms_payload = {
        "canonical_pruned_weight_terms": list(CANONICAL_PRUNED_WEIGHT_FIELDS),
        "phase": "phase3_5_canonical_mainline_pruning",
    }

    _write_json(report_root / "post_pruning_parity.json", post_payload)
    _write_json(report_root / "hash_parity.json", hash_payload)
    _write_json(report_root / "active_pipeline_trace_after_pruning.json", trace_payload)
    _write_json(report_root / "pruned_terms.json", pruned_terms_payload)

    table_rows: List[List[Any]] = []
    for row in post_rows:
        if row.get("status") != "ok":
            table_rows.append([row.get("dataset"), "NA", "NA", "NA", "NA", "NA", "NA", "NA", "missing_summary"])
            continue
        ds = str(row.get("dataset"))
        delta = dict(row.get("metric_delta", {}) or {})
        hh = next((x for x in hash_rows if str(x.get("dataset")) == ds), {})
        he = dict((hh or {}).get("hash_equal", {}) or {})
        table_rows.append(
            [
                ds,
                _format_delta(delta.get("Recall@5")),
                _format_delta(delta.get("EM")),
                _format_delta(delta.get("F1")),
                _format_delta(delta.get("avg_context_tokens")),
                _format_delta(delta.get("retrieval_ms")),
                str(bool(he.get("context_hash_equal", False))),
                str(bool(he.get("prompt_hash_equal", False))),
                str(row.get("decision")),
            ]
        )

    summary_lines: List[str] = []
    summary_lines.append("# Canonical Mainline Pruning (Phase-3.5)")
    summary_lines.append("")
    summary_lines.append(f"- before_round: `{before_round}`")
    summary_lines.append(f"- after_round: `{after_round}`")
    summary_lines.append(f"- stage: `{stage}`")
    summary_lines.append(f"- profile: `{profile}`")
    summary_lines.append("")
    summary_lines.append(
        _mk_table(
            ["dataset", "dR@5", "dEM", "dF1", "dTokens", "dRetrievalMs", "context_hash", "prompt_hash", "decision"],
            table_rows,
        )
    )
    summary_lines.append("")
    summary_lines.append("## Artifacts")
    summary_lines.append(f"- reports: `{report_root}`")
    summary_lines.append(f"- post_pruning_parity: `{report_root / 'post_pruning_parity.json'}`")
    summary_lines.append(f"- hash_parity: `{report_root / 'hash_parity.json'}`")
    summary_lines.append(f"- active_pipeline_trace_after_pruning: `{report_root / 'active_pipeline_trace_after_pruning.json'}`")
    summary_lines.append(f"- pruned_terms: `{report_root / 'pruned_terms.json'}`")

    (round_root / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(str(round_root))


if __name__ == "__main__":
    main()
