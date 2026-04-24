from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


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


def _strip_text(value: Any) -> str:
    return str(value or "").strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lim = int(limit or 0)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if lim > 0 and len(rows) >= lim:
                break
    return rows


def _context_by_title(sample: Mapping[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    raw = sample.get("context", [])
    if isinstance(raw, dict):
        titles = list(raw.get("title", []) or [])
        sentences = list(raw.get("sentences", []) or [])
        for title, sents in zip(titles, sentences):
            t = _strip_text(title)
            if not t:
                continue
            if isinstance(sents, list):
                out[t] = [_strip_text(x) for x in sents if _strip_text(x)]
            elif isinstance(sents, str):
                val = _strip_text(sents)
                out[t] = [val] if val else []
        return out

    if not isinstance(raw, list):
        return out

    for item in raw:
        if isinstance(item, list) and len(item) >= 2:
            title = _strip_text(item[0])
            if not title:
                continue
            sents = item[1]
            if isinstance(sents, list):
                out[title] = [_strip_text(x) for x in sents if _strip_text(x)]
            else:
                text = _strip_text(sents)
                out[title] = [text] if text else []
        elif isinstance(item, dict):
            title = _strip_text(item.get("title") or item.get("document_title") or item.get("name"))
            if not title:
                continue
            sents_val = item.get("sentences")
            if isinstance(sents_val, list):
                out[title] = [_strip_text(x) for x in sents_val if _strip_text(x)]
            elif isinstance(sents_val, str):
                text = _strip_text(sents_val)
                out[title] = [text] if text else []
    return out


def _supporting_facts(sample: Mapping[str, Any]) -> List[Tuple[str, int]]:
    raw = sample.get("supporting_facts", [])
    facts: List[Tuple[str, int]] = []
    if isinstance(raw, dict):
        titles = list(raw.get("title", []) or [])
        sent_ids = list(raw.get("sent_id", raw.get("sent_ids", [])) or [])
        for title, sent_idx in zip(titles, sent_ids):
            t = _strip_text(title)
            if not t:
                continue
            facts.append((t, _safe_int(sent_idx, 0)))
        return facts

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, list) and len(item) >= 2:
                title = _strip_text(item[0])
                if title:
                    facts.append((title, _safe_int(item[1], 0)))
            elif isinstance(item, dict):
                title = _strip_text(item.get("title") or item.get("doc_title") or item.get("document_title"))
                if title:
                    facts.append((title, _safe_int(item.get("sent_id", item.get("sent_idx", 0)), 0)))
    return facts


def _support_texts(sample: Mapping[str, Any]) -> List[str]:
    by_title = _context_by_title(sample)
    out: List[str] = []
    seen = set()
    for title, sent_idx in _supporting_facts(sample):
        sents = list(by_title.get(title, []) or [])
        if sent_idx < 0 or sent_idx >= len(sents):
            continue
        text = _strip_text(sents[sent_idx])
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


@dataclass
class QASampleIndex:
    by_index: List[Dict[str, Any]]
    by_qid: Dict[str, Dict[str, Any]]
    by_question: Dict[str, List[Dict[str, Any]]]


def load_qa_index(qa_path: str) -> QASampleIndex:
    path = Path(qa_path).resolve()
    payload = _load_json(path)
    rows: List[Mapping[str, Any]]
    if isinstance(payload, dict):
        rows = list(payload.get("data", payload.get("examples", [])) or [])
    else:
        rows = list(payload or [])

    by_index: List[Dict[str, Any]] = []
    by_qid: Dict[str, Dict[str, Any]] = {}
    by_question: Dict[str, List[Dict[str, Any]]] = {}

    for idx, raw in enumerate(rows):
        sample = dict(raw or {})
        qid = _strip_text(sample.get("id") or sample.get("_id") or sample.get("qid") or f"idx-{idx}")
        question = _strip_text(sample.get("question"))
        answer = _strip_text(sample.get("answer"))
        support_texts = _support_texts(sample)
        item = {
            "qid": qid,
            "question": question,
            "answer": answer,
            "gold_supports": support_texts,
            "raw": sample,
        }
        by_index.append(item)
        if qid and qid not in by_qid:
            by_qid[qid] = item
        if question:
            by_question.setdefault(question, []).append(item)
    return QASampleIndex(by_index=by_index, by_qid=by_qid, by_question=by_question)


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({str(k): str(v) for k, v in dict(row).items()})
    return rows


def discover_effirag_round_root(effirag_output_root: str) -> Optional[Path]:
    root = Path(effirag_output_root).resolve()
    if (root / "run_records.tsv").exists():
        return root
    if not root.exists():
        return None
    candidates = sorted([p for p in root.iterdir() if p.is_dir() and (p / "run_records.tsv").exists()])
    if not candidates:
        return None
    return candidates[-1]


def _pick_run_record(
    run_records: Sequence[Mapping[str, str]],
    dataset: str,
    variant: str,
) -> Optional[Dict[str, str]]:
    hits = []
    for row in list(run_records or []):
        if str(row.get("dataset", "")) != str(dataset):
            continue
        if str(row.get("variant", "")) != str(variant):
            continue
        if str(row.get("status", "")) not in {"ok", "skipped"}:
            continue
        summary_path = _strip_text(row.get("summary_path", ""))
        if not summary_path:
            continue
        if not Path(summary_path).exists():
            continue
        hits.append(dict(row))
    if not hits:
        return None
    hits.sort(key=lambda x: str(x.get("end_ts", x.get("start_ts", ""))))
    return hits[-1]


def discover_effirag_summary_path(
    effirag_output_root: str,
    dataset: str,
    variant: str,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    warnings: List[str] = []
    round_root = discover_effirag_round_root(effirag_output_root)
    if round_root is None:
        warnings.append(f"run_records.tsv not found under {effirag_output_root}")
        return None, {"round_root": "", "run_row": {}, "warnings": warnings}

    run_records_path = round_root / "run_records.tsv"
    rows = _read_tsv(run_records_path)
    hit = _pick_run_record(rows, dataset=dataset, variant=variant)
    if hit is None:
        warnings.append(f"missing run record for dataset={dataset}, variant={variant}")
        return None, {"round_root": str(round_root), "run_row": {}, "warnings": warnings}

    summary_path = Path(_strip_text(hit.get("summary_path", ""))).resolve()
    if not summary_path.exists():
        warnings.append(f"summary_path does not exist: {summary_path}")
        return None, {"round_root": str(round_root), "run_row": hit, "warnings": warnings}
    return summary_path, {"round_root": str(round_root), "run_row": hit, "warnings": warnings}


def _resolve_qa_item(index: QASampleIndex, qid: str, question: str, sample_index: int) -> Optional[Dict[str, Any]]:
    q = _strip_text(question)
    sid = _strip_text(qid)
    if sid and sid in index.by_qid:
        return index.by_qid[sid]
    if q:
        candidates = list(index.by_question.get(q, []) or [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1 and 0 <= sample_index < len(index.by_index):
            return index.by_index[sample_index]
        if candidates:
            return candidates[0]
    if 0 <= sample_index < len(index.by_index):
        return index.by_index[sample_index]
    return None


def _summary_to_base_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    s = dict(summary or {})
    out: Dict[str, float] = {
        "recall_at_1": _safe_float(s.get("recall_at_1", s.get("R@1", s.get("Recall@1", 0.0))), 0.0),
        "recall_at_5": _safe_float(s.get("recall_at_5", s.get("R@5", s.get("Recall@5", 0.0))), 0.0),
        "recall_at_10": _safe_float(s.get("recall_at_10", s.get("R@10", s.get("Recall@10", 0.0))), 0.0),
        "recall_at_20": _safe_float(s.get("recall_at_20", s.get("R@20", s.get("Recall@20", 0.0))), 0.0),
        "hit_at_5": _safe_float(s.get("hit_at_5", s.get("Hit@5", 0.0)), 0.0),
        "hit_at_10": _safe_float(s.get("hit_at_10", s.get("Hit@10", 0.0)), 0.0),
        "mrr_at_10": _safe_float(s.get("mrr_at_10", s.get("MRR@10", 0.0)), 0.0),
        "ndcg_at_10": _safe_float(s.get("ndcg_at_10", s.get("nDCG@10", 0.0)), 0.0),
        "context_precision": _safe_float(s.get("context_precision", s.get("ContextPrecision", 0.0)), 0.0),
        "supporting_fact_precision": _safe_float(
            s.get("supporting_fact_precision", s.get("sf_P", s.get("sf_precision", 0.0))),
            0.0,
        ),
        "supporting_fact_recall": _safe_float(
            s.get("supporting_fact_recall", s.get("sf_R", s.get("sf_recall", 0.0))),
            0.0,
        ),
        "supporting_fact_f1": _safe_float(
            s.get("supporting_fact_f1", s.get("sf_F1", s.get("sf_f1", 0.0))),
            0.0,
        ),
        "rendered_supporting_fact_precision": _safe_float(
            s.get("rendered_supporting_fact_precision", s.get("rendered_sf_P", 0.0)),
            0.0,
        ),
        "rendered_supporting_fact_recall": _safe_float(
            s.get("rendered_supporting_fact_recall", s.get("rendered_sf_R", 0.0)),
            0.0,
        ),
        "rendered_supporting_fact_f1": _safe_float(
            s.get("rendered_supporting_fact_f1", s.get("rendered_sf_F1", 0.0)),
            0.0,
        ),
        "em": _safe_float(s.get("em", s.get("EM", 0.0)), 0.0),
        "f1": _safe_float(s.get("f1", s.get("F1", 0.0)), 0.0),
        "retrieval_ms": _safe_float(s.get("retrieval_ms", s.get("retrieval_latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(s.get("generation_ms", s.get("generation_latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(s.get("total_ms", s.get("total_latency_ms", 0.0)), 0.0),
    }
    return out


def adapt_effirag_from_summary(
    *,
    summary_path: str,
    dataset: str,
    variant: str,
    qa_index: QASampleIndex,
    n_samples: int = 0,
) -> Dict[str, Any]:
    summary_fp = Path(summary_path).resolve()
    summary = dict(_load_json(summary_fp))
    query_path = summary_fp.with_name("rag_query_results.jsonl")
    if not query_path.exists():
        raise FileNotFoundError(f"missing query results: {query_path}")

    rows = _load_jsonl(query_path, limit=int(n_samples))

    adapted: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for i, row in enumerate(rows):
        sample_index = _safe_int(row.get("sample_index", i), i)
        sample_id = _strip_text(row.get("sample_id"))
        question = _strip_text(row.get("question"))
        qa_item = _resolve_qa_item(qa_index, qid=sample_id, question=question, sample_index=sample_index)
        if qa_item is None:
            warnings.append(f"qa lookup miss (dataset={dataset}, variant={variant}, idx={sample_index}, qid={sample_id})")

        gold_answer = _strip_text(
            (qa_item or {}).get("answer")
            or row.get("answer")
            or ((row.get("gold_answers", []) or [""])[0])
        )
        gold_supports = list((qa_item or {}).get("gold_supports", []) or [])

        retrieval = dict(row.get("retrieval", {}) or {})
        rendered = dict(row.get("rendered", {}) or {})
        efficiency = dict(row.get("efficiency", {}) or {})
        retrieved_items = [str(x) for x in list(retrieval.get("selected_sentences", []) or []) if _strip_text(x)]
        rendered_sentences = [str(x) for x in list(rendered.get("sentences", []) or []) if _strip_text(x)]
        rendered_context = _strip_text(rendered.get("text")) or "\n".join(rendered_sentences)

        adapted.append(
            {
                "dataset": str(dataset),
                "variant": str(variant),
                "qid": _strip_text((qa_item or {}).get("qid") or sample_id or f"{dataset}-{sample_index}"),
                "question": question or _strip_text((qa_item or {}).get("question")),
                "gold_answer": gold_answer,
                "prediction": _strip_text(row.get("prediction")),
                "gold_supports": gold_supports,
                "retrieved_items": retrieved_items,
                "rendered_context": rendered_context,
                "rendered_items": rendered_sentences,
                "raw_candidate_context": retrieved_items if retrieved_items else None,
                "timing": {
                    "retrieval_ms": _safe_float(efficiency.get("retrieval_latency_ms", 0.0), 0.0),
                    "generation_ms": _safe_float(
                        efficiency.get("generation_ms", efficiency.get("generation_latency_ms", 0.0)),
                        0.0,
                    ),
                    "total_ms": _safe_float(efficiency.get("total_latency_ms", 0.0), 0.0),
                },
                "query_metrics": dict(row.get("metrics", {}) or {}),
                "source": "effirag",
                "source_sample_index": int(sample_index),
                "source_sample_id": sample_id,
            }
        )

    return {
        "dataset": str(dataset),
        "variant": str(variant),
        "records": adapted,
        "summary_metrics": _summary_to_base_metrics(summary),
        "summary_n_samples": _safe_int(summary.get("n_samples", 0), 0),
        "summary_path": str(summary_fp),
        "query_path": str(query_path),
        "warnings": warnings,
    }


def _hipporag_variant_to_run_name(variant: str) -> str:
    v = _strip_text(variant).lower()
    if v in {"hipporag2", "hipporag", "hipporag2_tangent", "hipporag__tangent__tempna"}:
        return "hipporag__tangent__tempNA"
    return _strip_text(variant) or "hipporag__tangent__tempNA"


def _find_hipporag_query_solution_path(dataset_dir: Path, run_name: str, dataset: str) -> Optional[Path]:
    run_dir = dataset_dir / run_name
    preferred = run_dir / f"{dataset}__{run_name}__query_solutions.json"
    if preferred.exists():
        return preferred
    candidates = sorted(run_dir.glob("*__query_solutions.json"))
    if candidates:
        return candidates[-1]
    fallback = dataset_dir / f"{dataset}_query_solutions.json"
    if fallback.exists() and fallback.stat().st_size > 0:
        return fallback
    return None


def _find_hipporag_all_metadata_path(run_dir: Path, dataset: str, run_name: str) -> Optional[Path]:
    preferred = run_dir / f"{dataset}__{run_name}__all_metadata.json"
    if preferred.exists():
        return preferred
    candidates = sorted(run_dir.glob("*__all_metadata.json"))
    if candidates:
        return candidates[-1]
    return None


def _find_hipporag_metric_eval_summary(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("metric_eval_*/metric_round_metrics.json"))
    if candidates:
        return candidates[-1]
    return None


def adapt_hipporag2_from_outputs(
    *,
    hippo_output_root: str,
    dataset: str,
    variant: str,
    qa_index: QASampleIndex,
    n_samples: int = 0,
) -> Dict[str, Any]:
    dataset_dir = Path(hippo_output_root).resolve() / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"missing HippoRAG2 dataset dir: {dataset_dir}")

    run_name = _hipporag_variant_to_run_name(variant)
    run_dir = dataset_dir / run_name
    query_solution_path = _find_hipporag_query_solution_path(dataset_dir, run_name=run_name, dataset=dataset)
    if query_solution_path is None:
        raise FileNotFoundError(f"query solutions not found under {dataset_dir}")
    if not run_dir.exists():
        run_dir = query_solution_path.parent

    query_solutions = list(_load_json(query_solution_path) or [])
    if int(n_samples) > 0:
        query_solutions = query_solutions[: int(n_samples)]
    n_rows = len(query_solutions)

    warnings: List[str] = []
    metric_eval_summary_path = _find_hipporag_metric_eval_summary(run_dir)
    dataset_eval_summary_path = dataset_dir / f"{dataset}_eval_summary_results.json"
    base_metrics: Dict[str, float]
    summary_n_samples = 0
    if metric_eval_summary_path is not None and metric_eval_summary_path.exists():
        summary_payload = dict(_load_json(metric_eval_summary_path) or {})
        base_metrics = _summary_to_base_metrics(summary_payload)
        summary_n_samples = _safe_int(summary_payload.get("n_samples", 0), 0)
    elif dataset_eval_summary_path.exists():
        summary_payload = dict(_load_json(dataset_eval_summary_path) or {})
        base_metrics = _summary_to_base_metrics(summary_payload)
        summary_n_samples = int(n_rows)
        if n_rows > 0:
            total_retrieval_sec = _safe_float(summary_payload.get("total_retrieval_time", 0.0), 0.0)
            base_metrics["retrieval_ms"] = float((total_retrieval_sec * 1000.0) / float(n_rows))
            base_metrics["generation_ms"] = 0.0
            base_metrics["total_ms"] = base_metrics["retrieval_ms"]
        warnings.append("generation_ms unavailable in HippoRAG2 summary; fallback to retrieval-only timing")
    else:
        summary_payload = {}
        base_metrics = _summary_to_base_metrics(summary_payload)
        warnings.append("HippoRAG2 summary not found; base metrics defaulted to 0")

    all_metadata_path = _find_hipporag_all_metadata_path(run_dir, dataset=dataset, run_name=run_name)
    all_metadata = []
    if all_metadata_path is not None and all_metadata_path.exists():
        all_metadata = list(_load_json(all_metadata_path) or [])

    adapted: List[Dict[str, Any]] = []
    for i, row in enumerate(query_solutions):
        question = _strip_text(row.get("question"))
        qa_item = _resolve_qa_item(qa_index, qid="", question=question, sample_index=i)
        if qa_item is None:
            warnings.append(f"qa lookup miss (dataset={dataset}, variant={variant}, idx={i})")

        docs = [str(x) for x in list(row.get("docs", []) or []) if _strip_text(x)]
        rendered_context = "\n\n".join(docs)
        prediction = _strip_text(row.get("answer"))
        gold_answer = _strip_text((qa_item or {}).get("answer"))
        if not gold_answer:
            gold_answers = list(row.get("gold_answers", []) or [])
            gold_answer = _strip_text(gold_answers[0] if gold_answers else "")

        query_meta = {}
        if i < len(all_metadata) and isinstance(all_metadata[i], dict):
            query_meta = dict(all_metadata[i])

        adapted.append(
            {
                "dataset": str(dataset),
                "variant": str(variant),
                "qid": _strip_text((qa_item or {}).get("qid") or f"{dataset}-{i}"),
                "question": question or _strip_text((qa_item or {}).get("question")),
                "gold_answer": gold_answer,
                "prediction": prediction,
                "gold_supports": list((qa_item or {}).get("gold_supports", []) or []),
                "retrieved_items": docs,
                "rendered_context": rendered_context,
                "rendered_items": docs,
                "raw_candidate_context": docs if docs else None,
                "timing": {
                    "retrieval_ms": _safe_float(base_metrics.get("retrieval_ms", 0.0), 0.0),
                    "generation_ms": _safe_float(base_metrics.get("generation_ms", 0.0), 0.0),
                    "total_ms": _safe_float(base_metrics.get("total_ms", 0.0), 0.0),
                },
                "query_metrics": {},
                "source": "hipporag2",
                "source_sample_index": int(i),
                "source_sample_id": "",
                "query_metadata": query_meta,
            }
        )

    return {
        "dataset": str(dataset),
        "variant": str(variant),
        "records": adapted,
        "summary_metrics": base_metrics,
        "summary_n_samples": int(summary_n_samples if summary_n_samples > 0 else n_rows),
        "summary_path": str(metric_eval_summary_path or dataset_eval_summary_path),
        "query_path": str(query_solution_path),
        "warnings": warnings,
    }
