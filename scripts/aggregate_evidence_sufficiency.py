#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique, load_popqa
from effirag.metrics import supporting_fact_match_details
from effirag.utils import content_tokens, markdown_table


WORD_RE = re.compile(r"[A-Za-z0-9]+")


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


def _safe_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return bool(default)
    text = str(v).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _norm_text(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())


def _tokenize(s: Any) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(str(s or ""))]


def _token_set(s: Any) -> set:
    return set(content_tokens(str(s or "")))


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0.0:
        return 0.0
    return float(num / den)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _pairwise_jaccard(sentences: List[str]) -> float:
    if len(sentences) <= 1:
        return 0.0
    sets = [_token_set(s) for s in sentences]
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a and not b:
                continue
            union = a.union(b)
            if not union:
                continue
            scores.append(float(len(a.intersection(b)) / len(union)))
    return _mean(scores)


def _parse_title(unit_id: str) -> str:
    sid = str(unit_id or "").strip()
    if sid.startswith("chunk::"):
        payload = sid[len("chunk::") :]
        if "::" in payload:
            return str(payload.rsplit("::", 1)[0])
    if "::" in sid:
        return str(sid.rsplit("::", 1)[0])
    return sid


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({str(k): str(v or "") for k, v in row.items()})
    return rows


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = str(line or "").strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _is_answer_bearing(text: str, gold_answers: List[str]) -> bool:
    norm_text = _norm_text(text)
    if not norm_text:
        return False
    text_tok = _token_set(norm_text)
    for ans in gold_answers:
        norm_ans = _norm_text(ans)
        if not norm_ans:
            continue
        if len(norm_ans) >= 3 and norm_ans in norm_text:
            return True
        ans_tok = _token_set(norm_ans)
        if not ans_tok:
            continue
        inter = text_tok.intersection(ans_tok)
        if len(inter) >= max(1, int(math.ceil(0.70 * len(ans_tok)))):
            return True
    return False


def _overlap_score(prediction: str, evidence_texts: List[str]) -> float:
    pred_tok = _token_set(prediction)
    if not pred_tok:
        return 0.0
    joined = " ".join([str(x or "") for x in evidence_texts]).strip()
    ev_tok = _token_set(joined)
    if not ev_tok:
        return 0.0
    return float(_safe_ratio(len(pred_tok.intersection(ev_tok)), len(pred_tok)))


def _support_sentence_tokens(sample) -> List[Tuple[str, set]]:
    lookup = {}
    for doc in list(getattr(sample, "contexts", []) or []):
        title = str(getattr(doc, "title", "") or "")
        sents = list(getattr(doc, "sentences", []) or [])
        for idx, sent in enumerate(sents):
            lookup[(title, int(idx))] = _token_set(sent)

    out = []
    for title, idx in list(getattr(sample, "supporting_facts", []) or []):
        t = str(title or "")
        ii = _safe_int(idx, 0)
        toks = set(lookup.get((t, ii), set()))
        if toks:
            out.append((t, toks))
    return out


def _equivalent_unit(text: str, title: str, gold_titles: set, gold_support_tokens: List[Tuple[str, set]]) -> bool:
    if title and title in gold_titles:
        return True
    txt = _token_set(text)
    if not txt:
        return False
    best = 0.0
    for _gtitle, gtoks in gold_support_tokens:
        if not gtoks:
            continue
        inter = txt.intersection(gtoks)
        if len(inter) < 3:
            continue
        score = float(_safe_ratio(len(inter), len(gtoks)))
        if score > best:
            best = score
    return bool(best >= 0.50)


def _extract_main_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    r5 = _safe_float(
        summary.get(
            "supporting_fact_recall_at_5",
            ((summary.get("supporting_fact_recall_at_k", {}) or {}).get("5", 0.0)),
        ),
        0.0,
    )
    if r5 <= 0.0:
        r5 = _safe_float((summary.get("recall_at_k", {}) or {}).get("5", 0.0), 0.0)
    if r5 <= 0.0:
        r5 = _safe_float(summary.get("Recall@5", 0.0), 0.0)

    fallback_rate = _safe_float(
        summary.get(
            "stagewise_fallback_rate",
            ((summary.get("stagewise_loss_funnel", {}) or {}).get("fallback_rate", 0.0)),
        ),
        0.0,
    )

    return {
        "R@5": float(r5),
        "EM": _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0),
        "F1": _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0),
        "retrieval_ms": _safe_float(summary.get("retrieval_latency_ms", summary.get("retrieval_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_latency_ms", summary.get("generation_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_latency_ms", summary.get("total_ms", 0.0)), 0.0),
        "fallback_rate": float(fallback_rate),
    }


def _recompute_path_metrics(
    raw_strict: float,
    raw_ab_hit: float,
    path_complete_rate: float,
    bridge_to_answer_path_hit: float,
    bridge_purity: float,
    run_bridge_coverage: float,
    rendered_support_recall: float,
    strict_support_recall: float,
    answer_bearing_chunk_present: bool,
    equivalent_evidence_hit: bool,
    title_hit_ratio: float,
) -> Tuple[float, float, Dict[str, float]]:
    strict = float(max(0.0, raw_strict))
    ab_hit = float(max(0.0, raw_ab_hit))

    if strict <= 0.0:
        bridge_term = max(bridge_to_answer_path_hit, bridge_purity, run_bridge_coverage)
        answer_term = max(rendered_support_recall, strict_support_recall, 1.0 if answer_bearing_chunk_present else 0.0)
        strict = float(
            max(
                0.0,
                min(
                    1.0,
                    min(
                        max(0.0, path_complete_rate),
                        max(0.0, bridge_term),
                        max(0.0, answer_term),
                        max(0.0, title_hit_ratio),
                    ),
                ),
            )
        )

    if ab_hit <= 0.0:
        ab_signal = 1.0 if answer_bearing_chunk_present else (1.0 if equivalent_evidence_hit else 0.0)
        ab_hit = float(max(0.0, min(1.0, strict * ab_signal)))

    return strict, ab_hit, {
        "raw_strict_path_complete": float(raw_strict),
        "raw_answer_bearing_path_hit": float(raw_ab_hit),
        "repaired_strict_path_complete": float(strict),
        "repaired_answer_bearing_path_hit": float(ab_hit),
    }


def _dataset_family(dataset: str) -> str:
    d = str(dataset or "").strip().lower()
    if d.startswith("hotpot"):
        return "answerability-friendly multi-hop"
    if d.startswith("2wiki"):
        return "conversion-sensitive multi-hop"
    if d.startswith("musique"):
        return "harder compositional multi-hop"
    if d.startswith("popqa"):
        return "retrieval-light / prior-heavy open-domain"
    return "unknown"


def _family_bottleneck(row: Dict[str, Any]) -> str:
    strict_r = _safe_float(row.get("strict_support_recall", 0.0), 0.0)
    sufficient = _safe_float(row.get("minimal_support_subset_coverage", 0.0), 0.0)
    eq_cov = _safe_float(row.get("equivalent_evidence_coverage", 0.0), 0.0)
    fail = _safe_float(row.get("answer_present_but_generation_fail_rate", 0.0), 0.0)
    answer_present = _safe_float(row.get("answer_bearing_chunk_present_rate", 0.0), 0.0)
    if answer_present >= 0.60 and fail >= 0.35:
        return "rendered->generation conversion"
    if strict_r < 0.20 and sufficient < 0.30:
        return "retrieval evidence sufficiency"
    if eq_cov - strict_r >= 0.15:
        return "strict metric under-captures equivalent evidence"
    return "mixed"


def _family_budget_fit(row: Dict[str, Any]) -> str:
    strict_r = _safe_float(row.get("strict_support_recall", 0.0), 0.0)
    sufficient = _safe_float(row.get("minimal_support_subset_coverage", 0.0), 0.0)
    fail = _safe_float(row.get("answer_present_but_generation_fail_rate", 0.0), 0.0)
    if sufficient >= strict_r and fail >= 0.30:
        return "bounded budget likely adequate for retrieval; conversion is limiting"
    if sufficient < 0.25:
        return "bounded budget may be tight for this family"
    return "bounded budget appears acceptable"


def _load_samples_for_dataset(dataset: str, qa_path: str):
    d = str(dataset or "").strip().lower()
    loader = None
    split = "validation"
    if d.startswith("hotpot"):
        loader = load_hotpotqa
        split = "validation"
    elif d.startswith("2wiki") or d.startswith("twowiki"):
        loader = load_2wikimultihopqa
        split = "validation"
    elif d.startswith("musique"):
        loader = load_musique
        split = "validation"
    elif d.startswith("popqa"):
        loader = load_popqa
        split = "test"
    else:
        raise ValueError(f"unsupported_dataset:{dataset}")
    samples = loader(split=split, limit=None, data_path=qa_path)
    return {str(s.qid): s for s in list(samples or [])}


def _query_diag_row(
    dataset: str,
    variant: str,
    row: Dict[str, Any],
    sample,
) -> Dict[str, Any]:
    metrics = dict((row.get("metrics", {}) or {}))
    retrieval = dict((row.get("retrieval", {}) or {}))
    diagnostics = dict((retrieval.get("diagnostics", {}) or {}))
    rendered = dict((row.get("rendered", {}) or {}))
    rendered_meta = dict((rendered.get("metadata", {}) or {}))

    rendered_sentence_ids = [str(x or "") for x in list((row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or []))]
    rendered_sentences = [str(x or "") for x in list(((rendered.get("sentences", []) or [])))]

    prediction = str(row.get("prediction", "") or "")
    answer = str(row.get("answer", "") or "")
    gold_answers = [str(x or "").strip() for x in list((row.get("gold_answers", []) or [])) if str(x or "").strip()]
    if not gold_answers and answer:
        gold_answers = [answer]

    strict_support_recall = _safe_float(metrics.get("supporting_fact_recall", 0.0), 0.0)
    rendered_support_recall = _safe_float(metrics.get("rendered_supporting_fact_recall", 0.0), 0.0)
    em = _safe_float(metrics.get("em", metrics.get("EM", 0.0)), 0.0)
    f1 = _safe_float(metrics.get("f1", metrics.get("F1", 0.0)), 0.0)
    answer_correct = 1 if (em >= 0.5 or f1 >= 0.5) else 0

    graph_mode = str(diagnostics.get("graph_mode", "current_entity_graph") or "current_entity_graph")
    sf_match = supporting_fact_match_details(
        sample=sample,
        unit_ids=rendered_sentence_ids,
        unit_texts=rendered_sentences,
        graph_mode=graph_mode,
    )

    gold_total = int(sf_match.get("gold_total", 0) or 0)
    matched_total = int(sf_match.get("matched_gold_total", 0) or 0)
    strict_support_recall_recomputed = _safe_ratio(float(matched_total), float(max(1, gold_total)))

    gold_titles = {str(t) for t, _ in list(getattr(sample, "supporting_facts", []) or []) if str(t or "")}
    rendered_titles = {_parse_title(sid) for sid in rendered_sentence_ids if str(sid or "")}
    matched_gold_titles = {str(x).split("::", 1)[0] for x in list(sf_match.get("matched_gold_sentence_ids", []) or [])}
    title_hit_count = int(len(gold_titles.intersection(rendered_titles)))
    unique_gold_title_count = int(len(gold_titles))
    required_titles = int(1 if unique_gold_title_count <= 1 else 2)
    title_hit_ratio = _safe_ratio(float(title_hit_count), float(required_titles))

    answer_bearing_indices = []
    answer_bearing_texts = []
    equivalent_flags = []

    support_token_map = _support_sentence_tokens(sample)
    for idx, (sid, sent) in enumerate(zip(rendered_sentence_ids, rendered_sentences), start=1):
        sent = str(sent or "")
        title = _parse_title(sid)
        is_ans = _is_answer_bearing(sent, gold_answers)
        if is_ans:
            answer_bearing_indices.append(int(idx))
            answer_bearing_texts.append(sent)
        equivalent_flags.append(bool(_equivalent_unit(sent, title, gold_titles, support_token_map)))

    answer_bearing_chunk_present = bool(len(answer_bearing_indices) > 0)
    num_answer_bearing_chunks = int(len(answer_bearing_indices))
    first_answer_chunk_rank = int(answer_bearing_indices[0]) if answer_bearing_indices else 0

    meta_bundle_count = _safe_int(rendered_meta.get("raw_focus_answer_bearing_bundle_count", 0), 0)
    first_bundle_rank_meta = _safe_int(rendered_meta.get("raw_focus_first_answer_bearing_bundle_rank", 0), 0)

    if meta_bundle_count > 0:
        num_answer_bearing_bundles = int(meta_bundle_count)
    else:
        num_answer_bearing_bundles = int(1 if answer_bearing_chunk_present else 0)

    answer_bearing_bundle_present = bool(num_answer_bearing_bundles > 0)
    first_answer_bundle_rank = int(first_bundle_rank_meta if first_bundle_rank_meta > 0 else first_answer_chunk_rank)

    rendered_text = str((rendered.get("text", "") or ""))
    if answer_bearing_chunk_present and rendered_text:
        first_text = str(answer_bearing_texts[0] or "")
        pos = rendered_text.find(first_text)
        if pos < 0:
            pos = 0
        end_pos = min(len(rendered_text), pos + len(first_text))
        answer_bearing_token_start_ratio = float(_safe_ratio(float(pos), float(max(1, len(rendered_text)))))
        answer_bearing_token_end_ratio = float(_safe_ratio(float(end_pos), float(max(1, len(rendered_text)))))
    else:
        answer_bearing_token_start_ratio = 0.0
        answer_bearing_token_end_ratio = 0.0

    equivalent_unit_count = int(sum(1 for x in equivalent_flags if bool(x)))
    equivalent_evidence_hit = bool(equivalent_unit_count > 0)

    minimal_support_subset_hit = bool(
        answer_bearing_chunk_present
        and (
            title_hit_count >= required_titles
            or (
                equivalent_evidence_hit
                and strict_support_recall >= 0.20
                and title_hit_count >= 1
            )
        )
    )

    path_complete_rate = _safe_float(diagnostics.get("path_complete_rate", 0.0), 0.0)
    bridge_to_answer_path_hit = _safe_float(diagnostics.get("bridge_to_answer_path_hit", 0.0), 0.0)
    bridge_purity = _safe_float(diagnostics.get("bridge_purity", 0.0), 0.0)
    run_bridge_coverage = _safe_float(diagnostics.get("run_bridge_coverage", 0.0), 0.0)
    raw_strict_path_complete = _safe_float(diagnostics.get("strict_path_complete_rate", 0.0), 0.0)
    raw_answer_bearing_path_hit = _safe_float(diagnostics.get("answer_bearing_path_hit", 0.0), 0.0)

    strict_path_complete, answer_bearing_path_hit, repaired_meta = _recompute_path_metrics(
        raw_strict=raw_strict_path_complete,
        raw_ab_hit=raw_answer_bearing_path_hit,
        path_complete_rate=path_complete_rate,
        bridge_to_answer_path_hit=bridge_to_answer_path_hit,
        bridge_purity=bridge_purity,
        run_bridge_coverage=run_bridge_coverage,
        rendered_support_recall=rendered_support_recall,
        strict_support_recall=strict_support_recall,
        answer_bearing_chunk_present=answer_bearing_chunk_present,
        equivalent_evidence_hit=equivalent_evidence_hit,
        title_hit_ratio=title_hit_ratio,
    )

    overlap_score = _overlap_score(prediction=prediction, evidence_texts=answer_bearing_texts)
    output_overlap_answer_bearing = 1 if overlap_score >= 0.30 else 0

    if len(answer_bearing_indices) >= 2:
        evidence_dispersion_score = float(_safe_ratio(float(max(answer_bearing_indices) - min(answer_bearing_indices)), float(max(1, len(rendered_sentence_ids) - 1))))
    else:
        evidence_dispersion_score = 0.0
    evidence_duplication_score = float(_pairwise_jaccard(answer_bearing_texts))

    gdiag = dict((row.get("generation_diagnostics", {}) or {}))
    evidence_truncation_flag = 1 if (
        _safe_bool(rendered.get("truncated", False), False)
        or _safe_int(gdiag.get("truncated_sentence_count", 0), 0) > 0
        or _safe_int(gdiag.get("truncated_corridor_count", 0), 0) > 0
    ) else 0

    fallback = 1 if _safe_bool(row.get("generation_fallback", False), False) else 0
    answer_present_but_generation_fail = 1 if (answer_bearing_chunk_present and answer_correct <= 0 and fallback <= 0) else 0

    return {
        "query_id": str(row.get("sample_id", "") or ""),
        "dataset": str(dataset),
        "variant": str(variant),
        "answer_correct": int(answer_correct),
        "fallback": int(fallback),
        "strict_support_recall": float(strict_support_recall),
        "strict_support_recall_recomputed": float(strict_support_recall_recomputed),
        "answer_bearing_chunk_present": int(1 if answer_bearing_chunk_present else 0),
        "answer_bearing_bundle_present": int(1 if answer_bearing_bundle_present else 0),
        "answer_bearing_path_hit": float(answer_bearing_path_hit),
        "strict_path_complete": float(strict_path_complete),
        "equivalent_evidence_hit": int(1 if equivalent_evidence_hit else 0),
        "minimal_support_subset_hit": int(1 if minimal_support_subset_hit else 0),
        "answer_present_but_generation_fail": int(answer_present_but_generation_fail),
        "first_answer_chunk_rank": int(first_answer_chunk_rank),
        "first_answer_bundle_rank": int(first_answer_bundle_rank),
        "output_overlap_answer_bearing": int(output_overlap_answer_bearing),
        "output_overlap_answer_bearing_score": float(overlap_score),
        "num_gold_support_facts": int(gold_total),
        "num_matched_gold_support_facts": int(matched_total),
        "num_equivalent_evidence_units": int(equivalent_unit_count),
        "evidence_dispersion_score": float(evidence_dispersion_score),
        "evidence_duplication_score": float(evidence_duplication_score),
        "evidence_truncation_flag": int(evidence_truncation_flag),
        "answer_bearing_token_start_ratio": float(answer_bearing_token_start_ratio),
        "answer_bearing_token_end_ratio": float(answer_bearing_token_end_ratio),
        "num_answer_bearing_chunks": int(num_answer_bearing_chunks),
        "num_answer_bearing_bundles": int(num_answer_bearing_bundles),
        "raw_strict_path_complete": float(repaired_meta["raw_strict_path_complete"]),
        "raw_answer_bearing_path_hit": float(repaired_meta["raw_answer_bearing_path_hit"]),
        "repaired_strict_path_complete": float(repaired_meta["repaired_strict_path_complete"]),
        "repaired_answer_bearing_path_hit": float(repaired_meta["repaired_answer_bearing_path_hit"]),
        "path_complete_rate_raw": float(path_complete_rate),
        "bridge_to_answer_path_hit_raw": float(bridge_to_answer_path_hit),
        "bridge_purity_raw": float(bridge_purity),
        "run_bridge_coverage_raw": float(run_bridge_coverage),
        "title_hit_count": int(title_hit_count),
        "required_title_hits": int(required_titles),
        "title_hit_ratio": float(title_hit_ratio),
    }


def _agg_subset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "strict_support_recall": 0.0,
            "answer_bearing_chunk_present_rate": 0.0,
            "answer_bearing_bundle_present_rate": 0.0,
            "answer_bearing_path_hit_rate": 0.0,
            "strict_path_complete_rate": 0.0,
            "equivalent_evidence_coverage": 0.0,
            "minimal_support_subset_coverage": 0.0,
            "answer_present_but_generation_fail_rate": 0.0,
            "output_overlap_answer_bearing_rate": 0.0,
            "avg_first_answer_chunk_rank": 0.0,
            "avg_first_answer_bundle_rank": 0.0,
            "avg_evidence_dispersion_score": 0.0,
            "avg_evidence_duplication_score": 0.0,
            "avg_answer_bearing_token_start_ratio": 0.0,
            "avg_answer_bearing_token_end_ratio": 0.0,
            "avg_num_equivalent_evidence_units": 0.0,
            "strict_recall_low_but_correct_ratio": 0.0,
            "answer_bearing_present_generation_fail_ratio": 0.0,
            "answer_correct_rate": 0.0,
            "fallback_rate": 0.0,
        }

    n = len(rows)
    first_chunk = [float(r["first_answer_chunk_rank"]) for r in rows if _safe_int(r["first_answer_chunk_rank"], 0) > 0]
    first_bundle = [float(r["first_answer_bundle_rank"]) for r in rows if _safe_int(r["first_answer_bundle_rank"], 0) > 0]

    strict_low_correct = [r for r in rows if _safe_float(r["strict_support_recall"], 0.0) < 0.20 and _safe_int(r["answer_correct"], 0) > 0]
    correct_rows = [r for r in rows if _safe_int(r["answer_correct"], 0) > 0]
    ab_present_rows = [r for r in rows if _safe_int(r["answer_bearing_chunk_present"], 0) > 0]
    ab_present_gen_fail = [r for r in ab_present_rows if _safe_int(r["answer_present_but_generation_fail"], 0) > 0]

    return {
        "count": int(n),
        "strict_support_recall": _mean([_safe_float(r["strict_support_recall"], 0.0) for r in rows]),
        "answer_bearing_chunk_present_rate": _mean([_safe_int(r["answer_bearing_chunk_present"], 0) for r in rows]),
        "answer_bearing_bundle_present_rate": _mean([_safe_int(r["answer_bearing_bundle_present"], 0) for r in rows]),
        "answer_bearing_path_hit_rate": _mean([_safe_float(r["answer_bearing_path_hit"], 0.0) for r in rows]),
        "strict_path_complete_rate": _mean([_safe_float(r["strict_path_complete"], 0.0) for r in rows]),
        "equivalent_evidence_coverage": _mean([_safe_int(r["equivalent_evidence_hit"], 0) for r in rows]),
        "minimal_support_subset_coverage": _mean([_safe_int(r["minimal_support_subset_hit"], 0) for r in rows]),
        "answer_present_but_generation_fail_rate": _mean([_safe_int(r["answer_present_but_generation_fail"], 0) for r in rows]),
        "output_overlap_answer_bearing_rate": _mean([_safe_int(r["output_overlap_answer_bearing"], 0) for r in rows]),
        "avg_first_answer_chunk_rank": _mean(first_chunk),
        "avg_first_answer_bundle_rank": _mean(first_bundle),
        "avg_evidence_dispersion_score": _mean([_safe_float(r["evidence_dispersion_score"], 0.0) for r in rows]),
        "avg_evidence_duplication_score": _mean([_safe_float(r["evidence_duplication_score"], 0.0) for r in rows]),
        "avg_answer_bearing_token_start_ratio": _mean([_safe_float(r["answer_bearing_token_start_ratio"], 0.0) for r in rows]),
        "avg_answer_bearing_token_end_ratio": _mean([_safe_float(r["answer_bearing_token_end_ratio"], 0.0) for r in rows]),
        "avg_num_equivalent_evidence_units": _mean([_safe_float(r["num_equivalent_evidence_units"], 0.0) for r in rows]),
        "strict_recall_low_but_correct_ratio": _safe_ratio(float(len(strict_low_correct)), float(max(1, len(correct_rows)))),
        "answer_bearing_present_generation_fail_ratio": _safe_ratio(float(len(ab_present_gen_fail)), float(max(1, len(ab_present_rows)))),
        "answer_correct_rate": _mean([_safe_int(r["answer_correct"], 0) for r in rows]),
        "fallback_rate": _mean([_safe_int(r["fallback"], 0) for r in rows]),
    }


def _fmt(v: Any, nd: int = 4) -> str:
    return f"{_safe_float(v, 0.0):.{nd}f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate evidence sufficiency diagnostics over multi-dataset runs.")
    p.add_argument("--round-root", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--run-records", default=None)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    p.add_argument("--metric-integrity-report", default=None)
    p.add_argument("--query-diag-jsonl", default=None)
    p.add_argument("--query-diag-csv", default=None)
    p.add_argument("--base-variants", default="baseline_mid_reconfirm,ref_reconfirm")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    round_root = Path(args.round_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else (round_root / "run_manifest.json")
    run_records_path = Path(args.run_records).resolve() if args.run_records else (round_root / "run_records.tsv")
    output_json = Path(args.output_json).resolve() if args.output_json else (round_root / "round_metrics.json")
    output_md = Path(args.output_md).resolve() if args.output_md else (round_root / "round_metrics.md")
    integrity_json = Path(args.metric_integrity_report).resolve() if args.metric_integrity_report else (round_root / "metric_integrity_report.json")
    query_jsonl = Path(args.query_diag_jsonl).resolve() if args.query_diag_jsonl else (round_root / "evidence_sufficiency_query_diagnostics_all.jsonl")
    query_csv = Path(args.query_diag_csv).resolve() if args.query_diag_csv else (round_root / "evidence_sufficiency_query_diagnostics_all.csv")

    base_variants = [x.strip() for x in str(args.base_variants or "").split(",") if x.strip()]

    manifest = _read_json(manifest_path)
    run_records = _read_tsv(run_records_path)

    # Load dataset sample lookup from manifest-resolved QA paths.
    dataset_paths = dict((manifest.get("dataset_paths", {}) or {}))
    sample_lookup_by_dataset: Dict[str, Dict[str, Any]] = {}
    load_errors: Dict[str, str] = {}
    for dataset, pinfo in dataset_paths.items():
        qa_path = str((pinfo or {}).get("qa_path", "") or "").strip()
        if not qa_path:
            load_errors[str(dataset)] = "missing_qa_path"
            continue
        try:
            sample_lookup_by_dataset[str(dataset)] = _load_samples_for_dataset(str(dataset), qa_path)
        except Exception as exc:
            sample_lookup_by_dataset[str(dataset)] = {}
            load_errors[str(dataset)] = str(exc)

    # Build run list from actual records (includes optional s4 runs).
    active_runs: List[Dict[str, Any]] = []
    for row in run_records:
        if str(row.get("kind", "")) != "rag":
            continue
        status = str(row.get("status", "") or "")
        if status not in {"ok", "skipped"}:
            continue
        sp = str(row.get("summary_path", "") or "").strip()
        if not sp:
            continue
        summary_path = Path(sp)
        if not summary_path.exists():
            continue
        run = {
            "stage": str(row.get("stage", "") or ""),
            "variant": str(row.get("variant", "") or ""),
            "dataset": str(row.get("dataset", "") or ""),
            "profile": str(row.get("profile", "") or ""),
            "summary_path": str(summary_path.resolve()),
            "log_path": str(row.get("log_path", "") or ""),
            "status": status,
        }
        active_runs.append(run)

    query_rows: List[Dict[str, Any]] = []
    performance_rows: List[Dict[str, Any]] = []
    integrity_rows: List[Dict[str, Any]] = []

    for run in active_runs:
        dataset = str(run["dataset"])
        variant = str(run["variant"])
        summary_path = Path(run["summary_path"])
        summary = _read_json(summary_path)
        main_metrics = _extract_main_metrics(summary)

        performance_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "stage": str(run.get("stage", "")),
                "profile": str(run.get("profile", "")),
                **main_metrics,
            }
        )

        qpath = summary_path.with_name("rag_query_results.jsonl")
        qrows = _read_jsonl(qpath)

        dataset_samples = sample_lookup_by_dataset.get(dataset, {})
        if not dataset_samples:
            continue

        raw_strict_vals = []
        raw_ab_vals = []
        repaired_strict_vals = []
        repaired_ab_vals = []
        overlap_vals = []

        for row in qrows:
            qid = str(row.get("sample_id", "") or "")
            sample = dataset_samples.get(qid)
            if sample is None:
                # tolerant fallback: skip unknown sample ids
                continue
            qdiag = _query_diag_row(dataset=dataset, variant=variant, row=row, sample=sample)
            query_rows.append(qdiag)
            raw_strict_vals.append(_safe_float(qdiag.get("raw_strict_path_complete", 0.0), 0.0))
            raw_ab_vals.append(_safe_float(qdiag.get("raw_answer_bearing_path_hit", 0.0), 0.0))
            repaired_strict_vals.append(_safe_float(qdiag.get("strict_path_complete", 0.0), 0.0))
            repaired_ab_vals.append(_safe_float(qdiag.get("answer_bearing_path_hit", 0.0), 0.0))
            overlap_vals.append(_safe_float(qdiag.get("output_overlap_answer_bearing", 0.0), 0.0))

        raw_strict_mean = _mean(raw_strict_vals)
        raw_ab_mean = _mean(raw_ab_vals)
        repaired_strict_mean = _mean(repaired_strict_vals)
        repaired_ab_mean = _mean(repaired_ab_vals)
        overlap_mean = _mean(overlap_vals)

        issue_flags = []
        if raw_strict_mean <= 1.0e-9 and repaired_strict_mean >= 0.02:
            issue_flags.append("strict_path_complete_join_or_formula_mismatch")
        if raw_ab_mean <= 1.0e-9 and repaired_ab_mean >= 0.02:
            issue_flags.append("answer_bearing_path_hit_join_or_formula_mismatch")
        if overlap_mean <= 1.0e-9 and _safe_float(main_metrics.get("F1", 0.0), 0.0) > 0.05:
            issue_flags.append("output_overlap_answer_bearing_missing_path")

        integrity_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "stage": str(run.get("stage", "")),
                "query_count": int(len(repaired_strict_vals)),
                "raw_strict_path_complete_mean": float(raw_strict_mean),
                "raw_answer_bearing_path_hit_mean": float(raw_ab_mean),
                "repaired_strict_path_complete_mean": float(repaired_strict_mean),
                "repaired_answer_bearing_path_hit_mean": float(repaired_ab_mean),
                "output_overlap_answer_bearing_rate": float(overlap_mean),
                "issues": list(issue_flags),
            }
        )

    integrity_issue_count = int(sum(len(list(r.get("issues", []) or [])) for r in integrity_rows))
    integrity_status = "ok"
    if integrity_issue_count > 0:
        integrity_status = "partial"
    if not query_rows:
        integrity_status = "broken"

    integrity_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_root": str(round_root),
        "integrity_status": str(integrity_status),
        "dataset_load_errors": dict(load_errors),
        "checks": integrity_rows,
        "notes": [
            "strict_path_complete and answer_bearing_path_hit are repaired conservatively when raw diagnostics collapse to zero.",
            "repaired metrics are diagnostic-only and do not modify retrieval core behavior.",
        ],
    }
    integrity_json.write_text(json.dumps(integrity_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write query-level diagnostics.
    query_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with query_jsonl.open("w", encoding="utf-8") as f:
        for row in query_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    query_fields = [
        "query_id",
        "dataset",
        "variant",
        "answer_correct",
        "fallback",
        "strict_support_recall",
        "strict_support_recall_recomputed",
        "answer_bearing_chunk_present",
        "answer_bearing_bundle_present",
        "answer_bearing_path_hit",
        "strict_path_complete",
        "equivalent_evidence_hit",
        "minimal_support_subset_hit",
        "answer_present_but_generation_fail",
        "first_answer_chunk_rank",
        "first_answer_bundle_rank",
        "output_overlap_answer_bearing",
        "output_overlap_answer_bearing_score",
        "num_gold_support_facts",
        "num_matched_gold_support_facts",
        "num_equivalent_evidence_units",
        "evidence_dispersion_score",
        "evidence_duplication_score",
        "evidence_truncation_flag",
        "answer_bearing_token_start_ratio",
        "answer_bearing_token_end_ratio",
        "num_answer_bearing_chunks",
        "num_answer_bearing_bundles",
        "raw_strict_path_complete",
        "raw_answer_bearing_path_hit",
        "repaired_strict_path_complete",
        "repaired_answer_bearing_path_hit",
        "path_complete_rate_raw",
        "bridge_to_answer_path_hit_raw",
        "bridge_purity_raw",
        "run_bridge_coverage_raw",
        "title_hit_count",
        "required_title_hits",
        "title_hit_ratio",
    ]
    with query_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=query_fields)
        w.writeheader()
        for row in query_rows:
            w.writerow({k: row.get(k, "") for k in query_fields})

    # Aggregate by dataset+variant.
    group = defaultdict(list)
    for row in query_rows:
        key = (str(row.get("dataset", "")), str(row.get("variant", "")))
        group[key].append(row)

    evidence_table: List[Dict[str, Any]] = []
    gap_table: List[Dict[str, Any]] = []
    decomposition_table: List[Dict[str, Any]] = []

    for (dataset, variant), rows in sorted(group.items(), key=lambda x: (x[0][0], x[0][1])):
        agg = _agg_subset(rows)
        evidence_row = {
            "dataset": dataset,
            "variant": variant,
            **agg,
        }
        evidence_table.append(evidence_row)

        gap_table.append(
            {
                "dataset": dataset,
                "variant": variant,
                "strict_recall_minus_sufficient_gap": float(agg["strict_support_recall"] - agg["minimal_support_subset_coverage"]),
                "strict_recall_minus_equivalent_gap": float(agg["strict_support_recall"] - agg["equivalent_evidence_coverage"]),
                "sufficient_recall_minus_answer_correct_gap": float(agg["minimal_support_subset_coverage"] - agg["answer_correct_rate"]),
            }
        )

        correct_rows = [r for r in rows if _safe_int(r.get("answer_correct", 0), 0) > 0]
        incorrect_rows = [r for r in rows if _safe_int(r.get("answer_correct", 0), 0) <= 0]
        fail_rows = [r for r in rows if _safe_int(r.get("answer_present_but_generation_fail", 0), 0) > 0]
        non_fail_rows = [r for r in rows if _safe_int(r.get("answer_present_but_generation_fail", 0), 0) <= 0]

        decomposition_table.extend(
            [
                {
                    "dataset": dataset,
                    "variant": variant,
                    "subset": "correct",
                    **_agg_subset(correct_rows),
                },
                {
                    "dataset": dataset,
                    "variant": variant,
                    "subset": "incorrect",
                    **_agg_subset(incorrect_rows),
                },
                {
                    "dataset": dataset,
                    "variant": variant,
                    "subset": "generation_fail",
                    **_agg_subset(fail_rows),
                },
                {
                    "dataset": dataset,
                    "variant": variant,
                    "subset": "non_fail",
                    **_agg_subset(non_fail_rows),
                },
            ]
        )

    # Join main performance with evidence table.
    perf_idx = {(r["dataset"], r["variant"]): r for r in performance_rows}

    # Efficiency deltas vs baseline per dataset.
    eff_table = []
    for dataset in sorted({r["dataset"] for r in performance_rows}):
        base = perf_idx.get((dataset, "baseline_mid_reconfirm"))
        for variant in sorted({r["variant"] for r in performance_rows if r["dataset"] == dataset}):
            row = perf_idx[(dataset, variant)]
            if base is None:
                delta = {k: 0.0 for k in ["dR5", "dEM", "dF1", "dRetr", "dGen", "dTotal"]}
            else:
                delta = {
                    "dR5": float(_safe_float(row["R@5"], 0.0) - _safe_float(base["R@5"], 0.0)),
                    "dEM": float(_safe_float(row["EM"], 0.0) - _safe_float(base["EM"], 0.0)),
                    "dF1": float(_safe_float(row["F1"], 0.0) - _safe_float(base["F1"], 0.0)),
                    "dRetr": float(_safe_float(row["retrieval_ms"], 0.0) - _safe_float(base["retrieval_ms"], 0.0)),
                    "dGen": float(_safe_float(row["generation_ms"], 0.0) - _safe_float(base["generation_ms"], 0.0)),
                    "dTotal": float(_safe_float(row["total_ms"], 0.0) - _safe_float(base["total_ms"], 0.0)),
                }
            f1_per_100 = float(_safe_ratio(_safe_float(row["F1"], 0.0) * 100.0, max(1.0e-8, _safe_float(row["total_ms"], 0.0))))
            eff_table.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    **delta,
                    "F1_per_100ms": float(f1_per_100),
                }
            )

    # Family summary uses baseline/ref pair when available.
    family_summary = []
    ev_idx = {(r["dataset"], r["variant"]): r for r in evidence_table}
    datasets_all = sorted({r["dataset"] for r in evidence_table})
    for dataset in datasets_all:
        ref = ev_idx.get((dataset, "ref_reconfirm")) or ev_idx.get((dataset, "baseline_mid_reconfirm"))
        if not ref:
            continue
        family_summary.append(
            {
                "dataset": dataset,
                "family": _dataset_family(dataset),
                "strict_recall_level": float(_safe_float(ref.get("strict_support_recall", 0.0), 0.0)),
                "sufficient_recall_level": float(_safe_float(ref.get("minimal_support_subset_coverage", 0.0), 0.0)),
                "equivalent_evidence_coverage": float(_safe_float(ref.get("equivalent_evidence_coverage", 0.0), 0.0)),
                "generation_fail_level": float(_safe_float(ref.get("answer_present_but_generation_fail_rate", 0.0), 0.0)),
                "estimated_bottleneck": _family_bottleneck(ref),
                "bounded_budget_fit": _family_budget_fit(ref),
            }
        )

    # Stage4 recommendation condition.
    wiki_ref = ev_idx.get(("2wikimultihopqa", "ref_reconfirm"), {})
    stage4_condition_met = bool(
        integrity_status in {"ok", "partial"}
        and _safe_float(wiki_ref.get("answer_bearing_chunk_present_rate", 0.0), 0.0) >= 0.70
        and _safe_float(wiki_ref.get("answer_present_but_generation_fail_rate", 0.0), 0.0) >= 0.35
    )

    # Optional stage4 rows (if already appended in run_records).
    stage4_rows = [
        r
        for r in performance_rows
        if str(r.get("variant", "")).startswith("wiki_scaffold_ab_") or str(r.get("variant", "")) == "wiki_ref_reconfirm"
    ]

    # Reference integrity summary.
    reference_integrity = {
        "datasets": sorted({r.get("dataset", "") for r in performance_rows if r.get("variant") in base_variants}),
        "variants": list(base_variants),
        "completed_pairs": int(sum(1 for r in performance_rows if r.get("variant") in base_variants)),
        "fallback_rate_max": float(max([_safe_float(r.get("fallback_rate", 0.0), 0.0) for r in performance_rows] or [0.0])),
        "all_ok": bool(len(performance_rows) >= 1),
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_root": str(round_root),
        "methodology_note": "Evidence sufficiency proof round: retrieval core fixed, focus on strict-vs-sufficient evidence diagnostics.",
        "reference_integrity": reference_integrity,
        "metric_integrity_report_path": str(integrity_json),
        "metric_integrity_status": str(integrity_status),
        "main_performance": performance_rows,
        "evidence_sufficiency_table": evidence_table,
        "strict_vs_sufficient_gap_table": gap_table,
        "correct_incorrect_decomposition": decomposition_table,
        "efficiency_table": eff_table,
        "family_summary": family_summary,
        "optional_stage4_results": stage4_rows,
        "stage4_condition_met": bool(stage4_condition_met),
        "query_diagnostics_jsonl": str(query_jsonl),
        "query_diagnostics_csv": str(query_csv),
    }

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report.
    lines = []
    lines.append("# Evidence Sufficiency Proof Round")
    lines.append("")
    lines.append("- This round is for diagnosis/proof, not retrieval-core improvement.")
    lines.append("- Retrieval core and universal regime are fixed; strict-vs-sufficient evidence is audited across four datasets.")
    lines.append("")

    lines.append("## 1. Reference Integrity")
    lines.append("")
    lines.append(f"- integrity_status: {integrity_status}")
    lines.append(f"- completed_pairs: {reference_integrity['completed_pairs']}")
    lines.append(f"- fallback_rate_max: {_fmt(reference_integrity['fallback_rate_max'], 4)}")
    lines.append(f"- metric_integrity_report: {integrity_json}")
    lines.append("")

    lines.append("## 2. Metric Integrity Report")
    lines.append("")
    integ_headers = [
        "dataset",
        "variant",
        "raw_strict_path_complete_mean",
        "repaired_strict_path_complete_mean",
        "raw_answer_bearing_path_hit_mean",
        "repaired_answer_bearing_path_hit_mean",
        "output_overlap_answer_bearing_rate",
        "issues",
    ]
    integ_rows = []
    for r in integrity_rows:
        integ_rows.append(
            [
                r.get("dataset", ""),
                r.get("variant", ""),
                _fmt(r.get("raw_strict_path_complete_mean", 0.0), 4),
                _fmt(r.get("repaired_strict_path_complete_mean", 0.0), 4),
                _fmt(r.get("raw_answer_bearing_path_hit_mean", 0.0), 4),
                _fmt(r.get("repaired_answer_bearing_path_hit_mean", 0.0), 4),
                _fmt(r.get("output_overlap_answer_bearing_rate", 0.0), 4),
                ",".join(list(r.get("issues", []) or [])),
            ]
        )
    lines.append(markdown_table(integ_headers, integ_rows))
    lines.append("")

    lines.append("## 3. Evidence Sufficiency Table")
    lines.append("")
    ev_headers = [
        "dataset",
        "variant",
        "strict_support_recall",
        "answer_bearing_chunk_present_rate",
        "answer_bearing_bundle_present_rate",
        "answer_bearing_path_hit_rate",
        "equivalent_evidence_coverage",
        "minimal_support_subset_coverage",
        "answer_present_but_generation_fail_rate",
        "output_overlap_answer_bearing_rate",
        "avg_first_answer_chunk_rank",
        "avg_first_answer_bundle_rank",
        "avg_evidence_dispersion_score",
        "avg_evidence_duplication_score",
    ]
    ev_rows = []
    for r in evidence_table:
        ev_rows.append(
            [
                r.get("dataset", ""),
                r.get("variant", ""),
                _fmt(r.get("strict_support_recall", 0.0), 4),
                _fmt(r.get("answer_bearing_chunk_present_rate", 0.0), 4),
                _fmt(r.get("answer_bearing_bundle_present_rate", 0.0), 4),
                _fmt(r.get("answer_bearing_path_hit_rate", 0.0), 4),
                _fmt(r.get("equivalent_evidence_coverage", 0.0), 4),
                _fmt(r.get("minimal_support_subset_coverage", 0.0), 4),
                _fmt(r.get("answer_present_but_generation_fail_rate", 0.0), 4),
                _fmt(r.get("output_overlap_answer_bearing_rate", 0.0), 4),
                _fmt(r.get("avg_first_answer_chunk_rank", 0.0), 3),
                _fmt(r.get("avg_first_answer_bundle_rank", 0.0), 3),
                _fmt(r.get("avg_evidence_dispersion_score", 0.0), 4),
                _fmt(r.get("avg_evidence_duplication_score", 0.0), 4),
            ]
        )
    lines.append(markdown_table(ev_headers, ev_rows))
    lines.append("")

    lines.append("## 4. Strict vs Sufficient Gap Table")
    lines.append("")
    gap_headers = [
        "dataset",
        "variant",
        "strict_recall_minus_sufficient_gap",
        "strict_recall_minus_equivalent_gap",
        "sufficient_recall_minus_answer_correct_gap",
    ]
    gap_rows = []
    for r in gap_table:
        gap_rows.append(
            [
                r.get("dataset", ""),
                r.get("variant", ""),
                _fmt(r.get("strict_recall_minus_sufficient_gap", 0.0), 4),
                _fmt(r.get("strict_recall_minus_equivalent_gap", 0.0), 4),
                _fmt(r.get("sufficient_recall_minus_answer_correct_gap", 0.0), 4),
            ]
        )
    lines.append(markdown_table(gap_headers, gap_rows))
    lines.append("")

    lines.append("## 5. Correct/Incorrect Decomposition")
    lines.append("")
    dec_headers = [
        "dataset",
        "variant",
        "subset",
        "count",
        "strict_support_recall",
        "minimal_support_subset_coverage",
        "answer_present_but_generation_fail_rate",
        "output_overlap_answer_bearing_rate",
        "strict_recall_low_but_correct_ratio",
        "answer_bearing_present_generation_fail_ratio",
    ]
    dec_rows = []
    for r in decomposition_table:
        dec_rows.append(
            [
                r.get("dataset", ""),
                r.get("variant", ""),
                r.get("subset", ""),
                str(_safe_int(r.get("count", 0), 0)),
                _fmt(r.get("strict_support_recall", 0.0), 4),
                _fmt(r.get("minimal_support_subset_coverage", 0.0), 4),
                _fmt(r.get("answer_present_but_generation_fail_rate", 0.0), 4),
                _fmt(r.get("output_overlap_answer_bearing_rate", 0.0), 4),
                _fmt(r.get("strict_recall_low_but_correct_ratio", 0.0), 4),
                _fmt(r.get("answer_bearing_present_generation_fail_ratio", 0.0), 4),
            ]
        )
    lines.append(markdown_table(dec_headers, dec_rows))
    lines.append("")

    lines.append("## 6. Four-dataset Family Summary")
    lines.append("")
    fam_headers = [
        "dataset",
        "family",
        "strict_recall_level",
        "sufficient_recall_level",
        "equivalent_evidence_coverage",
        "generation_fail_level",
        "estimated_bottleneck",
        "bounded_budget_fit",
    ]
    fam_rows = []
    for r in family_summary:
        fam_rows.append(
            [
                r.get("dataset", ""),
                r.get("family", ""),
                _fmt(r.get("strict_recall_level", 0.0), 4),
                _fmt(r.get("sufficient_recall_level", 0.0), 4),
                _fmt(r.get("equivalent_evidence_coverage", 0.0), 4),
                _fmt(r.get("generation_fail_level", 0.0), 4),
                r.get("estimated_bottleneck", ""),
                r.get("bounded_budget_fit", ""),
            ]
        )
    lines.append(markdown_table(fam_headers, fam_rows))
    lines.append("")

    lines.append("## 7. Optional 2Wiki Scaffold A/B")
    lines.append("")
    lines.append(f"- stage4_condition_met: {str(stage4_condition_met).lower()}")
    if stage4_rows:
        s4_headers = ["dataset", "variant", "R@5", "EM", "F1", "retrieval_ms", "generation_ms", "total_ms", "fallback_rate"]
        s4_rows = []
        for r in stage4_rows:
            s4_rows.append(
                [
                    r.get("dataset", ""),
                    r.get("variant", ""),
                    _fmt(r.get("R@5", 0.0), 4),
                    _fmt(r.get("EM", 0.0), 4),
                    _fmt(r.get("F1", 0.0), 4),
                    _fmt(r.get("retrieval_ms", 0.0), 2),
                    _fmt(r.get("generation_ms", 0.0), 2),
                    _fmt(r.get("total_ms", 0.0), 2),
                    _fmt(r.get("fallback_rate", 0.0), 4),
                ]
            )
        lines.append(markdown_table(s4_headers, s4_rows))
    else:
        lines.append("- no stage4 runs in this round")
    lines.append("")

    lines.append("## 8. Interpretation")
    lines.append("")
    lines.append("- strict recall vs sufficient recall should be interpreted with conservative equivalent/minimal heuristics.")
    lines.append("- low strict recall does not automatically imply missing answer-bearing evidence.")
    lines.append("- when answer-bearing presence is high but generation-fail remains high, conversion is the primary bottleneck.")
    lines.append("")

    lines.append("## 9. Definitions (Conservative)")
    lines.append("")
    lines.append("- strict_support_recall: exact gold supporting-fact matching from evaluator.")
    lines.append("- equivalent_evidence_hit: title-aligned or high-overlap support-like rendered unit (conservative lexical heuristic).")
    lines.append("- minimal_support_subset_hit: answer-bearing evidence plus at least minimal title-support coverage (1 title for single-title, otherwise 2 titles).")
    lines.append("- strict_path_complete / answer_bearing_path_hit: repaired conservatively when raw diagnostics collapse to zero.")

    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "round_metrics_json": str(output_json),
                "round_metrics_md": str(output_md),
                "metric_integrity_report": str(integrity_json),
                "query_diagnostics_jsonl": str(query_jsonl),
                "query_diagnostics_csv": str(query_csv),
                "integrity_status": integrity_status,
                "stage4_condition_met": bool(stage4_condition_met),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
