from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .evidence_flow_audit import stage_metrics
from .utils import content_tokens, markdown_table, mean_or_zero, safe_div


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_BOTTLENECK_RULES: Dict[str, float] = {
    "candidate_recall_gap": 0.15,
    "candidate_overlap_absent": 0.05,
    "selection_overlap_drop": 0.25,
    "selection_recall_drop": 0.15,
    "selection_candidate_min": 0.50,
    "unit_prompt_tokens_high": 800.0,
    "unit_avg_tokens_per_item_high": 85.0,
    "unit_chunk_like_rate_high": 0.35,
    "density_recall_min": 0.45,
    "density_precision_max": 0.35,
    "density_redundancy_high": 0.45,
    "density_f1_per_1k_prompt_low": 0.30,
    "generation_recall_min": 0.60,
    "generation_precision_min": 0.35,
    "generation_f1_low": 0.20,
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


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = _safe_text(line)
            if not line:
                continue
            try:
                rows.append(dict(json.loads(line)))
            except Exception:
                continue
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in list(rows or []):
            f.write(json.dumps(dict(row or {}), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames or []))
        writer.writeheader()
        for row in list(rows or []):
            writer.writerow(dict(row or {}))


def load_json_or_jsonl(path: Path) -> Any:
    if not path.exists():
        return {}
    text = _safe_text(path.read_text(encoding="utf-8"))
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        rows = read_jsonl(path)
        if rows:
            return rows
        return {}


def parse_key_value_items(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in list(items or []):
        token = _safe_text(item)
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = _safe_text(key)
        value = _safe_text(value)
        if not key or not value:
            continue
        out[key] = Path(value).resolve()
    return out


def rough_token_count(text: str) -> int:
    return int(len(TOKEN_RE.findall(str(text or ""))))


def token_count_for_texts(texts: Iterable[str]) -> int:
    total = 0
    for text in list(texts or []):
        total += rough_token_count(str(text or ""))
    return int(total)


def parse_sentence_id_title(sentence_id: str) -> str:
    sid = _safe_text(sentence_id)
    if not sid:
        return ""
    parts = sid.split("::")
    if len(parts) >= 2:
        return _safe_text(parts[1])
    return sid


def normalized_text(text: str) -> str:
    return " ".join(_safe_text(text).lower().split())


def stage_from_row(row: Mapping[str, Any], stage: str) -> Tuple[List[str], List[str], str]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    rendered = dict((row.get("rendered", {}) or {}))
    diagnostics = dict((retrieval.get("diagnostics", {}) or {}))
    graph_mode = _safe_text(diagnostics.get("graph_mode", "current_entity_graph")) or "current_entity_graph"

    if stage == "candidate":
        ids = list(retrieval.get("candidate_sentence_ids", []) or [])
        texts = list(retrieval.get("candidate_sentences", []) or [])
        if not ids:
            ids = list(diagnostics.get("candidate_text_unit_ids", []) or [])
            texts = list(diagnostics.get("candidate_texts", []) or [])
        return [str(x) for x in ids], [str(x) for x in texts], graph_mode
    if stage == "selected":
        ids = list(retrieval.get("selected_sentence_ids", []) or [])
        texts = list(retrieval.get("selected_sentences", []) or [])
        if not ids:
            ids = list(diagnostics.get("selected_text_unit_ids", []) or [])
            texts = list(diagnostics.get("selected_texts", []) or [])
        return [str(x) for x in ids], [str(x) for x in texts], graph_mode
    if stage == "rendered":
        ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
        texts = list(rendered.get("sentences", []) or [])
        return [str(x) for x in ids], [str(x) for x in texts], graph_mode
    raise ValueError(f"Unsupported stage: {stage}")


def candidate_provenance_available(row: Mapping[str, Any]) -> bool:
    retrieval = dict((row.get("retrieval", {}) or {}))
    diagnostics = dict((retrieval.get("diagnostics", {}) or {}))
    direct_ids = list(retrieval.get("candidate_sentence_ids", []) or [])
    diag_ids = list(diagnostics.get("candidate_text_unit_ids", []) or [])
    if direct_ids or diag_ids:
        return True
    union_count = _safe_int(diagnostics.get("num_candidates_union", 0), 0)
    return bool(union_count <= 0)


def token_decomposition_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    rendered = dict((row.get("rendered", {}) or {}))
    generation_diag = dict((row.get("generation_diagnostics", {}) or {}))
    retrieval = dict((row.get("retrieval", {}) or {}))
    diagnostics = dict((retrieval.get("diagnostics", {}) or {}))
    metadata = dict((rendered.get("metadata", {}) or {}))

    prompt_tokens = _safe_int(
        generation_diag.get(
            "prompt_tokens",
            row.get("prompt_tokens", 0),
        ),
        0,
    )
    sentences = [str(x or "") for x in list(rendered.get("sentences", []) or [])]
    sentence_ids = [str(x or "") for x in list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])]
    titles = [parse_sentence_id_title(sid) for sid in sentence_ids if parse_sentence_id_title(sid)]

    evidence_tokens = _safe_int(metadata.get("evidence_text_tokens", token_count_for_texts(sentences)), 0)
    metadata_tokens = _safe_int(metadata.get("metadata_tokens", token_count_for_texts(titles)), 0)
    separator_line_count = _safe_int(
        metadata.get("separator_tokens", metadata.get("separator_line_count", diagnostics.get("separator_line_count", 0))),
        0,
    )
    separator_tokens = max(separator_line_count, max(0, len(sentences) - 1))
    instruction_tokens = _safe_int(
        generation_diag.get(
            "instruction_tokens",
            row.get("instruction_tokens", 0),
        ),
        0,
    )

    sum_tokens = int(evidence_tokens + metadata_tokens + separator_tokens + instruction_tokens)
    gap = int(prompt_tokens - sum_tokens)
    gap_abs = abs(gap)
    gap_ratio = float(safe_div(float(gap_abs), float(max(1, prompt_tokens))))
    warning = bool(prompt_tokens > 0 and (gap_abs > 64 or gap_ratio > 0.15))

    return {
        "prompt_tokens": int(prompt_tokens),
        "evidence_tokens": int(evidence_tokens),
        "metadata_tokens": int(metadata_tokens),
        "instruction_tokens": int(instruction_tokens),
        "separator_tokens": int(separator_tokens),
        "decomposition_sum_tokens": int(sum_tokens),
        "decomposition_gap_tokens": int(gap),
        "decomposition_gap_ratio": float(gap_ratio),
        "decomposition_warning": bool(warning),
    }


def _item_is_chunk_like(text: str, unit_id: str = "") -> bool:
    sid = _safe_text(unit_id).lower()
    tok = rough_token_count(text)
    sentence_boundary_count = len(re.findall(r"[.!?]", str(text or "")))
    if sid.startswith("chunk::"):
        return True
    if tok >= 60:
        return True
    if sentence_boundary_count >= 2:
        return True
    return False


def unit_contract_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    rendered = dict((row.get("rendered", {}) or {}))
    metadata = dict((rendered.get("metadata", {}) or {}))
    sentences = [str(x or "") for x in list(rendered.get("sentences", []) or [])]
    sentence_ids = [str(x or "") for x in list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])]
    token_counts = [rough_token_count(text) for text in sentences]
    chunk_like_flags = [
        _item_is_chunk_like(sentences[idx] if idx < len(sentences) else "", sentence_ids[idx] if idx < len(sentence_ids) else "")
        for idx in range(max(len(sentences), len(sentence_ids)))
    ]
    item_count = max(len(sentences), len(sentence_ids))
    chunk_like_count = int(sum(1 for x in chunk_like_flags if x))
    chunk_like_rate = float(
        _safe_float(
            metadata.get("chunk_like_rate", safe_div(float(chunk_like_count), float(max(1, item_count)))),
            safe_div(float(chunk_like_count), float(max(1, item_count))),
        )
    )
    avg_tokens = float(mean_or_zero(token_counts))
    if "avg_item_tokens" in metadata:
        avg_tokens = float(_safe_float(metadata.get("avg_item_tokens", avg_tokens), avg_tokens))
    max_tokens = int(max(token_counts) if token_counts else 0)
    if "max_item_tokens" in metadata:
        max_tokens = int(_safe_int(metadata.get("max_item_tokens", max_tokens), max_tokens))
    sentence_level = bool(chunk_like_rate < 0.5)
    if "sentence_level_rate" in metadata:
        sentence_level = bool(_safe_float(metadata.get("sentence_level_rate", 0.0), 0.0) >= 0.5)
    rendered_unit_type = _safe_text(metadata.get("selected_unit_type", ""))
    if not rendered_unit_type:
        rendered_unit_type = "chunk" if chunk_like_rate >= 0.5 else "sentence"
    if rendered_unit_type not in {"chunk", "sentence"}:
        rendered_unit_type = "chunk" if chunk_like_rate >= 0.5 else "sentence"

    return {
        "rendered_unit_type": rendered_unit_type,
        "is_sentence_level": bool(sentence_level),
        "is_chunk_like": bool(chunk_like_rate >= 0.5),
        "rendered_sentence_count": int(item_count),
        "chunk_like_rate": float(chunk_like_rate),
        "avg_tokens_per_rendered_item": float(avg_tokens),
        "avg_tokens_per_item": float(avg_tokens),
        "max_item_tokens": int(max_tokens),
    }


def density_metrics_from_row(
    row: Mapping[str, Any],
    *,
    prompt_tokens: int,
    rendered_sf_r: float,
    rendered_sf_p: float,
    rendered_sf_f1: float,
) -> Dict[str, Any]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    diagnostics = dict((retrieval.get("diagnostics", {}) or {}))
    rendered = dict((row.get("rendered", {}) or {}))
    sentences = [str(x or "") for x in list(rendered.get("sentences", []) or [])]
    sentence_ids = [str(x or "") for x in list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])]
    titles = [parse_sentence_id_title(sid) for sid in sentence_ids if parse_sentence_id_title(sid)]

    redundancy_rate = diagnostics.get("redundancy_rate", None)
    if redundancy_rate is None:
        normalized = [normalized_text(text) for text in sentences if normalized_text(text)]
        unique_ratio = safe_div(float(len(set(normalized))), float(max(1, len(normalized))))
        redundancy_rate = 1.0 - unique_ratio

    source_repetition_rate = 0.0
    if titles:
        source_repetition_rate = 1.0 - safe_div(float(len(set(titles))), float(len(titles)))

    token_list: List[str] = []
    for text in sentences:
        token_list.extend([str(tok) for tok in content_tokens(text)])
    token_counter = Counter(token_list)
    repeated = sum(max(0, count - 1) for count in token_counter.values())
    entity_repetition_rate = float(safe_div(float(repeated), float(max(1, len(token_list)))))

    coverage_gain_per_token = float(safe_div(float(rendered_sf_r), float(max(1, prompt_tokens))))
    sf_f1_per_1k_prompt = float(safe_div(float(rendered_sf_f1) * 1000.0, float(max(1, prompt_tokens))))

    return {
        "redundancy_rate": float(_safe_float(redundancy_rate, 0.0)),
        "source_repetition_rate": float(source_repetition_rate),
        "entity_repetition_rate": float(entity_repetition_rate),
        "coverage_gain_per_token": float(coverage_gain_per_token),
        "sf_F1_per_1k_prompt": float(sf_f1_per_1k_prompt),
        "rendered_sf_P": float(rendered_sf_p),
        "rendered_sf_R": float(rendered_sf_r),
        "rendered_sf_F1": float(rendered_sf_f1),
    }


def overlap_ratio(reference_ids: Iterable[str], probe_ids: Iterable[str]) -> float:
    ref = {str(x) for x in list(reference_ids or []) if str(x)}
    if not ref:
        return 0.0
    probe = {str(x) for x in list(probe_ids or []) if str(x)}
    return float(safe_div(float(len(ref.intersection(probe))), float(len(ref))))


def dominant_lost_stage_from_overlap(
    candidate_overlap: float,
    selected_overlap: float,
    rendered_overlap: float,
    *,
    eps: float = 1.0e-12,
) -> str:
    cand = _safe_float(candidate_overlap, 0.0)
    sel = _safe_float(selected_overlap, 0.0)
    ren = _safe_float(rendered_overlap, 0.0)
    if cand <= eps:
        return "candidate"
    if sel + eps < cand:
        return "selection"
    if ren + eps < sel:
        return "rendered"
    return "not_lost"


def legacy_overlap_for_query(
    legacy_rendered_ids: Iterable[str],
    candidate_ids: Iterable[str],
    selected_ids: Iterable[str],
    rendered_ids: Iterable[str],
) -> Dict[str, Any]:
    legacy_ids = [str(x) for x in list(legacy_rendered_ids or []) if str(x)]
    cand_overlap = overlap_ratio(legacy_ids, candidate_ids)
    sel_overlap = overlap_ratio(legacy_ids, selected_ids)
    ren_overlap = overlap_ratio(legacy_ids, rendered_ids)
    return {
        "legacy_candidate_overlap": float(cand_overlap),
        "legacy_selected_overlap": float(sel_overlap),
        "legacy_rendered_overlap": float(ren_overlap),
        "dominant_lost_stage": dominant_lost_stage_from_overlap(cand_overlap, sel_overlap, ren_overlap),
    }


def _score_trace_to_items(score_trace: Any) -> List[Tuple[str, Dict[str, Any]]]:
    if isinstance(score_trace, dict):
        out: List[Tuple[str, Dict[str, Any]]] = []
        for sid, payload in score_trace.items():
            out.append((str(sid or ""), dict(payload or {}) if isinstance(payload, Mapping) else {}))
        return out
    if isinstance(score_trace, list):
        out = []
        for item in score_trace:
            if not isinstance(item, Mapping):
                continue
            sid = _safe_text(item.get("sentence_id", item.get("unit_id", item.get("id", ""))))
            payload = dict(item)
            out.append((sid, payload))
        return out
    return []


def extract_utility_component_rows(
    *,
    dataset: str,
    profile: str,
    qid: str,
    row: Mapping[str, Any],
    selected_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    diagnostics = dict((retrieval.get("diagnostics", {}) or {}))
    gl_diag = dict((diagnostics.get("gl_rcedr_diag", {}) or {}))
    score_trace = gl_diag.get("score_trace", {})
    trace_items = _score_trace_to_items(score_trace)
    if not trace_items:
        return []

    selected_rank_map: Dict[str, int] = {str(sid): idx + 1 for idx, sid in enumerate(list(selected_ids or []))}
    core_ids = {str(x) for x in list(gl_diag.get("core_sentence_ids", []) or [])}
    out: List[Dict[str, Any]] = []
    for sid, payload in trace_items:
        evidence_gain = _safe_float(payload.get("evidence_gain", payload.get("G", 0.0)), 0.0)
        bridge_gain = _safe_float(payload.get("bridge_gain", payload.get("B", 0.0)), 0.0)
        redundancy = _safe_float(payload.get("redundancy", payload.get("R", 0.0)), 0.0)
        token_cost = _safe_float(payload.get("token_cost", payload.get("cost_C", payload.get("C", 0.0))), 0.0)
        score = _safe_float(payload.get("score", payload.get("final_utility", 0.0)), 0.0)
        rank = int(selected_rank_map.get(str(sid), 0))
        status = "selected" if rank > 0 else "rejected"
        out.append(
            {
                "dataset": dataset,
                "profile": profile,
                "qid": qid,
                "sentence_id": str(sid),
                "evidence_gain_G": float(evidence_gain),
                "bridge_gain_B": float(bridge_gain),
                "redundancy_R": float(redundancy),
                "cost_C": float(token_cost),
                "final_utility": float(score),
                "selected_rank": int(rank),
                "selected_or_rejected": status,
                "is_core_preserve": bool(str(sid) in core_ids),
            }
        )
    return out


def classify_bottleneck_taxonomy(
    row: Mapping[str, Any],
    *,
    rules: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    cfg = dict(DEFAULT_BOTTLENECK_RULES)
    if rules:
        for k, v in dict(rules).items():
            cfg[str(k)] = _safe_float(v, cfg.get(str(k), 0.0))

    legacy_rendered_sf_r = _safe_float(row.get("legacy_rendered_sf_R", 0.0))
    candidate_sf_r = _safe_float(row.get("candidate_sf_R", 0.0))
    selected_sf_r = _safe_float(row.get("selected_sf_R", 0.0))
    rendered_sf_r = _safe_float(row.get("rendered_sf_R", 0.0))
    rendered_sf_p = _safe_float(row.get("rendered_sf_P", 0.0))
    candidate_overlap_raw = row.get("legacy_candidate_overlap", row.get("candidate_overlap_with_legacy", None))
    selected_overlap_raw = row.get("legacy_selected_overlap", row.get("selected_overlap_with_legacy", None))
    candidate_overlap_known = candidate_overlap_raw is not None
    selected_overlap_known = selected_overlap_raw is not None
    candidate_overlap = _safe_float(candidate_overlap_raw, 0.0)
    selected_overlap = _safe_float(selected_overlap_raw, 0.0)
    rendered_overlap = _safe_float(row.get("legacy_rendered_overlap", row.get("rendered_overlap_with_legacy", 0.0)))
    avg_tokens_per_item = _safe_float(row.get("avg_tokens_per_item", row.get("avg_tokens_per_rendered_item", 0.0)))
    prompt_tokens = _safe_float(row.get("prompt_tokens", 0.0))
    chunk_like_rate = _safe_float(row.get("chunk_like_rate", 0.0))
    redundancy_rate = _safe_float(row.get("redundancy_rate", 0.0))
    sf_f1_per_1k_prompt = _safe_float(row.get("sf_F1_per_1k_prompt", 0.0))
    qa_f1 = _safe_float(row.get("QA_F1", row.get("qa_f1", 0.0)))

    candidate_bottleneck = False
    if legacy_rendered_sf_r > 0.0 and candidate_sf_r + cfg["candidate_recall_gap"] < legacy_rendered_sf_r:
        candidate_bottleneck = True
    if candidate_overlap_known and candidate_overlap <= cfg["candidate_overlap_absent"] and legacy_rendered_sf_r > 0.0:
        candidate_bottleneck = True

    selection_bottleneck = False
    if (
        candidate_overlap_known
        and selected_overlap_known
        and candidate_overlap >= 0.50
        and selected_overlap + cfg["selection_overlap_drop"] < candidate_overlap
    ):
        selection_bottleneck = True
    if candidate_sf_r >= cfg["selection_candidate_min"] and selected_sf_r + cfg["selection_recall_drop"] < candidate_sf_r:
        selection_bottleneck = True

    unit_bottleneck = False
    if prompt_tokens >= cfg["unit_prompt_tokens_high"]:
        unit_bottleneck = True
    if avg_tokens_per_item >= cfg["unit_avg_tokens_per_item_high"]:
        unit_bottleneck = True
    if chunk_like_rate >= cfg["unit_chunk_like_rate_high"]:
        unit_bottleneck = True

    density_bottleneck = False
    if rendered_sf_r >= cfg["density_recall_min"] and rendered_sf_p <= cfg["density_precision_max"]:
        density_bottleneck = True
    if redundancy_rate >= cfg["density_redundancy_high"]:
        density_bottleneck = True
    if sf_f1_per_1k_prompt <= cfg["density_f1_per_1k_prompt_low"] and rendered_sf_r >= cfg["density_recall_min"]:
        density_bottleneck = True

    generation_bottleneck = False
    if (
        rendered_sf_r >= cfg["generation_recall_min"]
        and rendered_sf_p >= cfg["generation_precision_min"]
        and qa_f1 <= cfg["generation_f1_low"]
    ):
        generation_bottleneck = True

    dominant = "none"
    if candidate_bottleneck:
        dominant = "candidate"
    elif selection_bottleneck:
        dominant = "selection"
    elif unit_bottleneck and density_bottleneck:
        dominant = "unit_or_density"
    elif unit_bottleneck:
        dominant = "unit_contract"
    elif density_bottleneck:
        dominant = "density"
    elif generation_bottleneck:
        dominant = "generation"

    return {
        "candidate_bottleneck": bool(candidate_bottleneck),
        "selection_bottleneck": bool(selection_bottleneck),
        "unit_bottleneck": bool(unit_bottleneck),
        "density_bottleneck": bool(density_bottleneck),
        "generation_bottleneck": bool(generation_bottleneck),
        "dominant_bottleneck": dominant,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], metric_keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in list(metric_keys or []):
        out[key] = float(mean_or_zero([_safe_float(row.get(key, 0.0), 0.0) for row in list(rows or [])]))
    return out


def aggregate_query_rows_by_profile_dataset(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        dataset = _safe_text(row.get("dataset", ""))
        profile = _safe_text(row.get("profile", ""))
        buckets[(dataset, profile)].append(dict(row))

    metric_keys = [
        "candidate_sf_R",
        "selected_sf_R",
        "rendered_sf_R",
        "rendered_sf_P",
        "rendered_sf_F1",
        "candidate_sf_P",
        "selected_sf_P",
        "prompt_tokens",
        "rendered_tokens",
        "rendered_sf_F1_per_1k_tokens",
        "legacy_candidate_overlap",
        "legacy_selected_overlap",
        "legacy_rendered_overlap",
        "redundancy_rate",
        "source_repetition_rate",
        "entity_repetition_rate",
        "sf_F1_per_1k_prompt",
        "chunk_like_rate",
        "avg_tokens_per_item",
    ]
    out: List[Dict[str, Any]] = []
    for (dataset, profile), group in sorted(buckets.items()):
        agg = _aggregate(group, metric_keys)
        qa_rows = [item for item in group if bool(item.get("qa_metric_available", False))]
        qa_em = float(mean_or_zero([_safe_float(item.get("QA_EM", 0.0), 0.0) for item in qa_rows]))
        qa_f1 = float(mean_or_zero([_safe_float(item.get("QA_F1", 0.0), 0.0) for item in qa_rows]))
        dominant_counter = Counter([_safe_text(item.get("dominant_bottleneck", "")) for item in group])
        lost_stage_counter = Counter([_safe_text(item.get("dominant_lost_stage", "")) for item in group])
        out.append(
            {
                "dataset": dataset,
                "profile": profile,
                "num_queries": int(len(group)),
                "qa_num_queries": int(len(qa_rows)),
                "QA_EM": float(qa_em),
                "QA_F1": float(qa_f1),
                **agg,
                "candidate_contains_legacy_evidence_rate": float(
                    mean_or_zero([1.0 if bool(item.get("candidate_contains_legacy_evidence", False)) else 0.0 for item in group])
                ),
                "selector_drop_rate": float(
                    mean_or_zero([_safe_float(item.get("selector_drop_rate", 0.0), 0.0) for item in group])
                ),
                "dominant_bottleneck": str(dominant_counter.most_common(1)[0][0]) if dominant_counter else "none",
                "dominant_lost_stage": str(lost_stage_counter.most_common(1)[0][0]) if lost_stage_counter else "unknown",
            }
        )
    return out


def summarize_token_decomposition(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        buckets[(_safe_text(row.get("dataset", "")), _safe_text(row.get("profile", "")))].append(dict(row))
    out: List[Dict[str, Any]] = []
    for (dataset, profile), group in sorted(buckets.items()):
        out.append(
            {
                "dataset": dataset,
                "profile": profile,
                "prompt_tokens": float(mean_or_zero([_safe_float(r.get("prompt_tokens", 0.0), 0.0) for r in group])),
                "evidence_tokens": float(mean_or_zero([_safe_float(r.get("evidence_tokens", 0.0), 0.0) for r in group])),
                "metadata_tokens": float(mean_or_zero([_safe_float(r.get("metadata_tokens", 0.0), 0.0) for r in group])),
                "instruction_tokens": float(
                    mean_or_zero([_safe_float(r.get("instruction_tokens", 0.0), 0.0) for r in group])
                ),
                "separator_tokens": float(mean_or_zero([_safe_float(r.get("separator_tokens", 0.0), 0.0) for r in group])),
                "avg_tokens_per_item": float(mean_or_zero([_safe_float(r.get("avg_tokens_per_item", 0.0), 0.0) for r in group])),
                "decomposition_warning_rate": float(
                    mean_or_zero([1.0 if bool(r.get("decomposition_warning", False)) else 0.0 for r in group])
                ),
            }
        )
    return out


def summarize_unit_contract(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        buckets[(_safe_text(row.get("dataset", "")), _safe_text(row.get("profile", "")))].append(dict(row))
    out: List[Dict[str, Any]] = []
    for (dataset, profile), group in sorted(buckets.items()):
        max_item_tokens = 0
        for item in group:
            max_item_tokens = max(max_item_tokens, _safe_int(item.get("max_item_tokens", 0), 0))
        out.append(
            {
                "dataset": dataset,
                "profile": profile,
                "sentence_level_rate": float(
                    mean_or_zero([1.0 if bool(item.get("is_sentence_level", False)) else 0.0 for item in group])
                ),
                "chunk_like_rate": float(mean_or_zero([_safe_float(item.get("chunk_like_rate", 0.0), 0.0) for item in group])),
                "avg_item_tokens": float(mean_or_zero([_safe_float(item.get("avg_tokens_per_item", 0.0), 0.0) for item in group])),
                "max_item_tokens": int(max_item_tokens),
            }
        )
    return out


def summarize_utility_component_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        buckets[(_safe_text(row.get("dataset", "")), _safe_text(row.get("profile", "")))].append(dict(row))
    out: List[Dict[str, Any]] = []
    for (dataset, profile), group in sorted(buckets.items()):
        selected = [r for r in group if _safe_text(r.get("selected_or_rejected", "")) == "selected"]
        rejected = [r for r in group if _safe_text(r.get("selected_or_rejected", "")) != "selected"]
        out.append(
            {
                "dataset": dataset,
                "profile": profile,
                "num_scored_units": int(len(group)),
                "avg_G_selected": float(mean_or_zero([_safe_float(r.get("evidence_gain_G", 0.0), 0.0) for r in selected])),
                "avg_B_selected": float(mean_or_zero([_safe_float(r.get("bridge_gain_B", 0.0), 0.0) for r in selected])),
                "avg_R_selected": float(mean_or_zero([_safe_float(r.get("redundancy_R", 0.0), 0.0) for r in selected])),
                "avg_C_selected": float(mean_or_zero([_safe_float(r.get("cost_C", 0.0), 0.0) for r in selected])),
                "avg_G_rejected": float(mean_or_zero([_safe_float(r.get("evidence_gain_G", 0.0), 0.0) for r in rejected])),
                "avg_C_rejected": float(mean_or_zero([_safe_float(r.get("cost_C", 0.0), 0.0) for r in rejected])),
            }
        )
    return out


def summarize_bottleneck_taxonomy(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        buckets[(_safe_text(row.get("dataset", "")), _safe_text(row.get("profile", "")))].append(dict(row))
    out: List[Dict[str, Any]] = []
    for (dataset, profile), group in sorted(buckets.items()):
        counter = Counter([_safe_text(row.get("dominant_bottleneck", "none")) for row in group])
        out.append(
            {
                "dataset": dataset,
                "profile": profile,
                "candidate_bottleneck": float(
                    mean_or_zero([1.0 if bool(r.get("candidate_bottleneck", False)) else 0.0 for r in group])
                ),
                "selection_bottleneck": float(
                    mean_or_zero([1.0 if bool(r.get("selection_bottleneck", False)) else 0.0 for r in group])
                ),
                "unit_bottleneck": float(mean_or_zero([1.0 if bool(r.get("unit_bottleneck", False)) else 0.0 for r in group])),
                "density_bottleneck": float(
                    mean_or_zero([1.0 if bool(r.get("density_bottleneck", False)) else 0.0 for r in group])
                ),
                "generation_bottleneck": float(
                    mean_or_zero([1.0 if bool(r.get("generation_bottleneck", False)) else 0.0 for r in group])
                ),
                "dominant_bottleneck": str(counter.most_common(1)[0][0]) if counter else "none",
            }
        )
    return out


def champion_summary_rows(
    *,
    aggregate_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    profiles: Sequence[str],
) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in list(aggregate_rows or []):
        by_key[(_safe_text(row.get("dataset", "")), _safe_text(row.get("profile", "")))] = dict(row)

    def pick_max(dataset: str, metric: str) -> Tuple[str, float]:
        best_profile = ""
        best_value = float("-inf")
        for profile in list(profiles or []):
            row = by_key.get((dataset, profile))
            if not row:
                continue
            value = _safe_float(row.get(metric, 0.0), 0.0)
            if value > best_value:
                best_profile = profile
                best_value = value
        if best_value == float("-inf"):
            best_value = 0.0
        return best_profile, float(best_value)

    out: List[Dict[str, Any]] = []
    for dataset in list(datasets or []):
        for role, profile, metric in [
            ("legacy teacher", "legacy_sota", "rendered_sf_R"),
            ("clean baseline", "unified_large", "rendered_sf_R"),
            ("candidate recall champion", "unified_candidate_recall_boost_v1", "candidate_sf_R"),
        ]:
            row = by_key.get((dataset, profile), {})
            out.append(
                {
                    "role": role,
                    "dataset": dataset,
                    "profile": profile,
                    "key metric": metric,
                    "value": float(_safe_float(row.get(metric, 0.0), 0.0)),
                }
            )
        eff_profile, eff_value = pick_max(dataset, "rendered_sf_F1_per_1k_tokens")
        out.append(
            {
                "role": "retrieval efficiency champion",
                "dataset": dataset,
                "profile": eff_profile,
                "key metric": "rendered_sf_F1_per_1k_tokens",
                "value": float(eff_value),
            }
        )
        recall_profile, recall_value = pick_max(dataset, "rendered_sf_R")
        out.append(
            {
                "role": "rendered recall champion",
                "dataset": dataset,
                "profile": recall_profile,
                "key metric": "rendered_sf_R",
                "value": float(recall_value),
            }
        )
        qa_profile, qa_value = pick_max(dataset, "QA_F1")
        out.append(
            {
                "role": "QA champion",
                "dataset": dataset,
                "profile": qa_profile,
                "key metric": "QA_F1",
                "value": float(qa_value),
            }
        )
    return out


def load_optional_qa_index(
    path: Path,
    dataset: str,
    default_profile: str = "",
) -> Tuple[Dict[Tuple[str, str, str], Dict[str, float]], Dict[Tuple[str, str], List[Dict[str, float]]]]:
    payload = load_json_or_jsonl(path)
    if not payload:
        return {}, {}

    out_profile: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    out_dataset: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)

    def maybe_add(item: Mapping[str, Any], fallback_dataset: str) -> None:
        qid = _safe_text(
            item.get(
                "qid",
                item.get(
                    "sample_id",
                    item.get("id", ""),
                ),
            )
        )
        if not qid:
            return
        ds = _safe_text(item.get("dataset", fallback_dataset)) or fallback_dataset
        profile = _safe_text(item.get("profile", item.get("variant", default_profile)))
        metrics = item.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        em = _safe_float(
            item.get(
                "em",
                item.get(
                    "EM",
                    item.get(
                        "qa_em",
                        item.get(
                            "QA_EM",
                            metrics.get("em", metrics.get("EM", metrics.get("qa_em", metrics.get("QA_EM", 0.0)))),
                        ),
                    ),
                ),
            ),
            0.0,
        )
        f1 = _safe_float(
            item.get(
                "f1",
                item.get(
                    "F1",
                    item.get(
                        "qa_f1",
                        item.get(
                            "QA_F1",
                            metrics.get("f1", metrics.get("F1", metrics.get("qa_f1", metrics.get("QA_F1", 0.0)))),
                        ),
                    ),
                ),
            ),
            0.0,
        )
        generation_diag = item.get("generation_diagnostics", {})
        if not isinstance(generation_diag, Mapping):
            generation_diag = {}
        prompt_tokens = _safe_int(
            item.get("prompt_tokens", generation_diag.get("prompt_tokens", item.get("prompt_token_count", 0))),
            0,
        )
        completion_tokens = _safe_int(
            item.get("completion_tokens", generation_diag.get("completion_tokens", item.get("completion_token_count", 0))),
            0,
        )
        record = {
            "QA_EM": float(em),
            "QA_F1": float(f1),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "qa_profile": str(profile),
            "qa_metric_available": True,
        }
        if profile:
            out_profile[(ds, profile, qid)] = record
        out_dataset[(ds, qid)].append(record)

    def walk(obj: Any, fallback_dataset: str) -> None:
        if isinstance(obj, list):
            for item in obj:
                walk(item, fallback_dataset)
            return
        if isinstance(obj, dict):
            maybe_add(obj, fallback_dataset)
            for key in ["rows", "items", "records", "results", "data", "query_level", "samples", "predictions"]:
                if key in obj:
                    walk(obj.get(key), fallback_dataset)
            # Also handle qid->payload maps.
            looks_like_map = bool(obj) and all(isinstance(v, Mapping) for v in obj.values())
            if looks_like_map:
                for k, v in obj.items():
                    if not isinstance(v, Mapping):
                        continue
                    inner = dict(v)
                    inner.setdefault("qid", str(k))
                    walk(inner, fallback_dataset)

    walk(payload, dataset)
    return out_profile, out_dataset


def merge_optional_qa_metrics(
    *,
    dataset: str,
    profile: str,
    qid: str,
    base_row: Mapping[str, Any],
    qa_profile_index: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    qa_dataset_index: Mapping[Tuple[str, str], List[Mapping[str, Any]]],
) -> Dict[str, Any]:
    merged = dict(base_row or {})
    key_profile = (dataset, profile, qid)
    if key_profile in qa_profile_index:
        matched = dict(qa_profile_index[key_profile] or {})
        merged.update(matched)
        merged["qa_metric_available"] = bool(matched.get("qa_metric_available", True))
        return merged
    dataset_records = list(qa_dataset_index.get((dataset, qid), []) or [])
    if not dataset_records:
        return merged
    exact_profile = [
        dict(record or {})
        for record in dataset_records
        if _safe_text((record or {}).get("qa_profile", "")) == profile
    ]
    if len(exact_profile) == 1:
        matched = exact_profile[0]
        merged.update(matched)
        merged["qa_metric_available"] = bool(matched.get("qa_metric_available", True))
        return merged
    compatible = [
        dict(record or {})
        for record in dataset_records
        if _safe_text((record or {}).get("qa_profile", "")) in {"", profile}
    ]
    if len(compatible) == 1:
        matched = compatible[0]
        merged.update(matched)
        merged["qa_metric_available"] = bool(matched.get("qa_metric_available", True))
    return merged


def build_phase6j_markdown(
    *,
    datasets: Sequence[str],
    profiles: Sequence[str],
    roots: Mapping[str, Path],
    champion_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    overlap_rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Mapping[str, Any]],
    unit_rows: Sequence[Mapping[str, Any]],
    utility_rows: Sequence[Mapping[str, Any]],
    taxonomy_rows: Sequence[Mapping[str, Any]],
    phase_label: str = "Phase-6J",
    audit_title: str = "Champion-based Bottleneck Audit",
    purpose_text: str = "Reposition bottlenecks against legacy and profile champions before any method patching.",
) -> str:
    agg_by_key = {
        (_safe_text(r.get("dataset", "")), _safe_text(r.get("profile", ""))): dict(r)
        for r in list(aggregate_rows or [])
    }
    overlap_by_key = {
        (_safe_text(r.get("dataset", "")), _safe_text(r.get("profile", ""))): dict(r)
        for r in list(overlap_rows or [])
    }
    token_by_key = {
        (_safe_text(r.get("dataset", "")), _safe_text(r.get("profile", ""))): dict(r)
        for r in list(token_rows or [])
    }
    unit_by_key = {
        (_safe_text(r.get("dataset", "")), _safe_text(r.get("profile", ""))): dict(r)
        for r in list(unit_rows or [])
    }
    utility_by_key = {
        (_safe_text(r.get("dataset", "")), _safe_text(r.get("profile", ""))): dict(r)
        for r in list(utility_rows or [])
    }
    taxonomy_by_key = {
        (_safe_text(r.get("dataset", "")), _safe_text(r.get("profile", ""))): dict(r)
        for r in list(taxonomy_rows or [])
    }

    lines: List[str] = []
    lines.append(f"# {phase_label} {audit_title}")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(str(purpose_text or "").strip() or "TBD")
    lines.append("")
    lines.append("## 2. Compared Profiles")
    lines.append("")
    for profile in list(profiles or []):
        lines.append(f"- {profile}: {str(roots.get(profile, Path('')))}")
    lines.append("")
    lines.append("## 3. Champion Summary")
    lines.append("")
    lines.append("| role | dataset | profile | key metric | value |")
    lines.append("|---|---|---|---|---:|")
    for row in list(champion_rows or []):
        lines.append(
            f"| {_safe_text(row.get('role', ''))} | {_safe_text(row.get('dataset', ''))} | {_safe_text(row.get('profile', ''))} | "
            f"{_safe_text(row.get('key metric', ''))} | {float(_safe_float(row.get('value', 0.0), 0.0)):.4f} |"
        )
    lines.append("")
    lines.append("## 4. Evidence Flow")
    lines.append("")
    lines.append(
        "| dataset | profile | candidate_sf_R | selected_sf_R | rendered_sf_R | rendered_sf_P | "
        "rendered_tokens | prompt_tokens | rendered_sf_F1_per_1k_tokens |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in list(datasets or []):
        for profile in list(profiles or []):
            row = agg_by_key.get((dataset, profile))
            if not row:
                continue
            lines.append(
                f"| {dataset} | {profile} | {float(_safe_float(row.get('candidate_sf_R', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('selected_sf_R', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('rendered_sf_R', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('rendered_sf_P', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('rendered_tokens', 0.0), 0.0)):.1f} | "
                f"{float(_safe_float(row.get('prompt_tokens', 0.0), 0.0)):.1f} | "
                f"{float(_safe_float(row.get('rendered_sf_F1_per_1k_tokens', 0.0), 0.0)):.4f} |"
            )
    lines.append("")
    lines.append("## 5. Legacy Overlap")
    lines.append("")
    lines.append("| dataset | profile | candidate_overlap | selected_overlap | rendered_overlap | dominant_lost_stage |")
    lines.append("|---|---|---:|---:|---:|---|")
    for dataset in list(datasets or []):
        for profile in list(profiles or []):
            if profile == "legacy_sota":
                continue
            row = overlap_by_key.get((dataset, profile))
            if not row:
                continue
            lines.append(
                f"| {dataset} | {profile} | {float(_safe_float(row.get('legacy_candidate_overlap', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('legacy_selected_overlap', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('legacy_rendered_overlap', 0.0), 0.0)):.4f} | "
                f"{_safe_text(row.get('dominant_lost_stage', 'unknown'))} |"
            )
    lines.append("")
    lines.append("## 6. Token Decomposition")
    lines.append("")
    lines.append(
        "| dataset | profile | prompt_tokens | evidence_tokens | metadata_tokens | instruction_tokens | separator_tokens | avg_tokens_per_item |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for dataset in list(datasets or []):
        for profile in list(profiles or []):
            row = token_by_key.get((dataset, profile))
            if not row:
                continue
            lines.append(
                f"| {dataset} | {profile} | {float(_safe_float(row.get('prompt_tokens', 0.0), 0.0)):.1f} | "
                f"{float(_safe_float(row.get('evidence_tokens', 0.0), 0.0)):.1f} | "
                f"{float(_safe_float(row.get('metadata_tokens', 0.0), 0.0)):.1f} | "
                f"{float(_safe_float(row.get('instruction_tokens', 0.0), 0.0)):.1f} | "
                f"{float(_safe_float(row.get('separator_tokens', 0.0), 0.0)):.1f} | "
                f"{float(_safe_float(row.get('avg_tokens_per_item', 0.0), 0.0)):.1f} |"
            )
    lines.append("")
    lines.append("## 7. Unit Contract")
    lines.append("")
    lines.append("| dataset | profile | sentence_level_rate | chunk_like_rate | avg_item_tokens | max_item_tokens |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for dataset in list(datasets or []):
        for profile in list(profiles or []):
            row = unit_by_key.get((dataset, profile))
            if not row:
                continue
            lines.append(
                f"| {dataset} | {profile} | {float(_safe_float(row.get('sentence_level_rate', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('chunk_like_rate', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('avg_item_tokens', 0.0), 0.0)):.2f} | "
                f"{int(_safe_int(row.get('max_item_tokens', 0), 0))} |"
            )
    lines.append("")
    lines.append("## 8. Redundancy and Density")
    lines.append("")
    lines.append(
        "| dataset | profile | redundancy_rate | source_repetition_rate | entity_repetition_rate | rendered_sf_P | sf_F1_per_1k_prompt |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for dataset in list(datasets or []):
        for profile in list(profiles or []):
            row = agg_by_key.get((dataset, profile))
            if not row:
                continue
            lines.append(
                f"| {dataset} | {profile} | {float(_safe_float(row.get('redundancy_rate', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('source_repetition_rate', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('entity_repetition_rate', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('rendered_sf_P', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('sf_F1_per_1k_prompt', 0.0), 0.0)):.4f} |"
            )
    lines.append("")
    lines.append("## 9. Utility Component Audit")
    lines.append("")
    lines.append(
        "| dataset | profile | avg_G_selected | avg_B_selected | avg_R_selected | avg_C_selected | avg_G_rejected | avg_C_rejected |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for dataset in list(datasets or []):
        for profile in list(profiles or []):
            row = utility_by_key.get((dataset, profile))
            if not row:
                continue
            lines.append(
                f"| {dataset} | {profile} | {float(_safe_float(row.get('avg_G_selected', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('avg_B_selected', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('avg_R_selected', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('avg_C_selected', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('avg_G_rejected', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('avg_C_rejected', 0.0), 0.0)):.4f} |"
            )
    lines.append("")
    lines.append("## 10. Bottleneck Taxonomy")
    lines.append("")
    lines.append(
        "| dataset | profile | candidate_bottleneck | selection_bottleneck | unit_bottleneck | density_bottleneck | generation_bottleneck | dominant_bottleneck |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for dataset in list(datasets or []):
        for profile in list(profiles or []):
            row = taxonomy_by_key.get((dataset, profile))
            if not row:
                continue
            lines.append(
                f"| {dataset} | {profile} | {float(_safe_float(row.get('candidate_bottleneck', 0.0), 0.0)):.3f} | "
                f"{float(_safe_float(row.get('selection_bottleneck', 0.0), 0.0)):.3f} | "
                f"{float(_safe_float(row.get('unit_bottleneck', 0.0), 0.0)):.3f} | "
                f"{float(_safe_float(row.get('density_bottleneck', 0.0), 0.0)):.3f} | "
                f"{float(_safe_float(row.get('generation_bottleneck', 0.0), 0.0)):.3f} | "
                f"{_safe_text(row.get('dominant_bottleneck', 'none'))} |"
            )
    lines.append("")
    lines.append("## 11. QA Smoke")
    lines.append("")
    lines.append("| dataset | profile | qa_num_queries | QA_EM | QA_F1 |")
    lines.append("|---|---|---:|---:|---:|")
    for dataset in list(datasets or []):
        for profile in list(profiles or []):
            row = agg_by_key.get((dataset, profile))
            if not row:
                continue
            lines.append(
                f"| {dataset} | {profile} | {int(_safe_int(row.get('qa_num_queries', 0), 0))} | "
                f"{float(_safe_float(row.get('QA_EM', 0.0), 0.0)):.4f} | "
                f"{float(_safe_float(row.get('QA_F1', 0.0), 0.0)):.4f} |"
            )
    lines.append("")
    lines.append("## 12. Decision")
    lines.append("")
    lines.append("- Fine-grained improvement or large redesign?")
    lines.append("- Which bottleneck should be targeted next?")
    lines.append("")
    return "\n".join(lines)
