import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "who",
    "what",
    "when",
    "where",
    "which",
    "how",
}


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


def content_tokens(text):
    return [t for t in tokenize(text) if len(t) > 2 and t not in STOPWORDS]


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def is_effectively_zero(value: float, eps: float) -> bool:
    try:
        v = abs(float(value))
        e = abs(float(eps))
    except Exception:
        return False
    return bool(v <= e)


def evaluate_clipping_closure(
    *,
    embedding_token_clipped_rate: float,
    embedding_char_clipped_rate: float,
    prompt_over_limit_rate: float,
    token_eps: float = 1.0e-4,
    char_eps: float = 1.0e-3,
    prompt_eps: float = 1.0e-12,
) -> Dict[str, Any]:
    token_rate = float(embedding_token_clipped_rate or 0.0)
    char_rate = float(embedding_char_clipped_rate or 0.0)
    prompt_rate = float(prompt_over_limit_rate or 0.0)
    token_ok = is_effectively_zero(token_rate, token_eps)
    char_ok = is_effectively_zero(char_rate, char_eps)
    prompt_ok = is_effectively_zero(prompt_rate, prompt_eps)
    return {
        "token_rate": token_rate,
        "char_rate": char_rate,
        "prompt_over_limit_rate": prompt_rate,
        "token_eps": float(token_eps),
        "char_eps": float(char_eps),
        "prompt_eps": float(prompt_eps),
        "token_ok": bool(token_ok),
        "char_ok": bool(char_ok),
        "prompt_ok": bool(prompt_ok),
        "pass": bool(token_ok and char_ok and prompt_ok),
    }


def compute_rag_derived_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    summary = dict(summary or {})
    sf_recall = float(summary.get("supporting_fact_recall", 0.0) or 0.0)
    recall_at_k = summary.get("supporting_fact_recall_at_k", {}) or {}
    recall_at_5 = float(
        summary.get(
            "supporting_fact_recall_at_5",
            recall_at_k.get("5", recall_at_k.get(5, summary.get("recall_at_5", 0.0))),
        )
        or 0.0
    )
    rendered_sf_recall = float(
        summary.get(
            "rendered_supporting_fact_recall",
            summary.get("rendered_sf_recall", 0.0),
        )
        or 0.0
    )
    f1 = float(summary.get("f1", summary.get("F1", 0.0)) or 0.0)
    oracle_ceiling_f1 = float(
        summary.get(
            "oracle_ceiling_f1",
            summary.get(
                "oracle_reference_f1",
                summary.get("d2_f1", 0.0),
            ),
        )
        or 0.0
    )
    if oracle_ceiling_f1 <= 0.0 and str(summary.get("oracle_mode", "") or "").strip().lower() == "oracle_upper_bound":
        oracle_ceiling_f1 = float(f1)
    hipporag2_reference_f1 = float(
        summary.get(
            "hipporag2_reference_f1",
            summary.get("hippo_reference_f1", 0.0),
        )
        or 0.0
    )
    hipporag2_reference_recall_at_5 = float(
        summary.get(
            "hipporag2_reference_recall_at_5",
            summary.get("hippo_reference_recall_at_5", 0.0),
        )
        or 0.0
    )
    retrieval_ms = float(
        summary.get(
            "retrieval_latency_ms",
            summary.get("retrieval_ms", 0.0),
        )
        or 0.0
    )
    total_ms = float(
        summary.get(
            "total_latency_ms",
            summary.get("total_ms", 0.0),
        )
        or 0.0
    )
    f1_per_100ms = float(safe_div(f1 * 100.0, total_ms)) if total_ms > 0.0 else 0.0
    r5_per_100ms = float(safe_div(recall_at_5 * 100.0, retrieval_ms)) if retrieval_ms > 0.0 else 0.0
    oracle_gap = float(oracle_ceiling_f1 - f1) if oracle_ceiling_f1 > 0.0 else 0.0
    oracle_gap_per_ms = float(safe_div(oracle_gap, total_ms)) if total_ms > 0.0 else 0.0
    return {
        "rendered_retention": float(safe_div(rendered_sf_recall, sf_recall)),
        "answer_conversion": float(safe_div(f1, rendered_sf_recall)),
        "oracle_gap": float(oracle_gap),
        "retrieval_ms": float(retrieval_ms),
        "total_ms": float(total_ms),
        "f1_per_100ms": float(f1_per_100ms),
        "r5_per_100ms": float(r5_per_100ms),
        "oracle_gap_per_ms": float(oracle_gap_per_ms),
        "gap_to_hippo_f1": (
            float(hipporag2_reference_f1 - f1) if hipporag2_reference_f1 > 0.0 else 0.0
        ),
        "gap_to_hippo_r5": (
            float(hipporag2_reference_recall_at_5 - recall_at_5)
            if hipporag2_reference_recall_at_5 > 0.0
            else 0.0
        ),
    }


STAGEWISE_LOSS_KEYS = (
    "anchor_hit_rate",
    "proposal_hit_rate",
    "phase1_hit_rate",
    "final_chunk_hit_rate",
    "rendered_retention",
    # Connector-aware retrieval diagnostics (PAMAE-style rounds)
    "seed_anchor_recall",
    "seed_bridge_recall",
    "seed_answer_recall",
    "seed_diversity",
    "seed_redundancy",
    "run_bridge_coverage",
    "corridor_role_coverage",
    "answer_preserve_activation_rate",
    "preserved_answer_usefulness",
    "path_complete_rate",
    "bridge_answer_pair_retention",
    "incomplete_path_rate",
    "chain_compactness",
    "path_preserve_activation_rate",
    "anchor_bridge_balance",
    "bridge_purity",
    "answer_side_density",
    "support_set_compactness",
    "bridge_noise_ratio",
    "useful_bridge_rate",
    "bridge_to_answer_path_hit",
    "conversion_after_bridge",
    "conversion_after_preserve",
    "conversion_after_path_preserve",
    "hotpot_overpreserve_rate",
    "answer_conversion",
    "em_conversion",
    "f1_per_r5",
    "error_bucket_bridge_missing",
    "error_bucket_answer_side_missing",
    "error_bucket_bridge_present_noisy",
    "error_bucket_answer_present_generation_fail",
    "redundancy_rate",
    "connector_quality",
)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_jsonl_rows(path):
    rows = []
    fp = Path(path)
    if not fp.exists():
        return rows
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = str(line or "").strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _extract_stagewise_from_row(row):
    retrieval = ((row or {}).get("retrieval", {}) or {})
    diagnostics = (retrieval.get("diagnostics", {}) or {})
    funnel = diagnostics.get("stagewise_loss_funnel", {}) or {}
    if not isinstance(funnel, dict):
        funnel = {}

    metrics = {}
    for key in STAGEWISE_LOSS_KEYS:
        if key in funnel:
            metrics[key] = _safe_float(funnel.get(key, 0.0), 0.0)

    if "anchor_hit_rate" not in metrics and "anchor_hit_rate" in diagnostics:
        metrics["anchor_hit_rate"] = _safe_float(diagnostics.get("anchor_hit_rate", 0.0), 0.0)
    if "proposal_hit_rate" not in metrics and "proposal_hit_rate" in diagnostics:
        metrics["proposal_hit_rate"] = _safe_float(diagnostics.get("proposal_hit_rate", 0.0), 0.0)
    if "phase1_hit_rate" not in metrics and "phase1_hit_rate" in diagnostics:
        metrics["phase1_hit_rate"] = _safe_float(diagnostics.get("phase1_hit_rate", 0.0), 0.0)
    if "final_chunk_hit_rate" not in metrics and "final_chunk_hit_rate" in diagnostics:
        metrics["final_chunk_hit_rate"] = _safe_float(diagnostics.get("final_chunk_hit_rate", 0.0), 0.0)
    for key in (
        "seed_anchor_recall",
        "seed_bridge_recall",
        "seed_answer_recall",
        "seed_diversity",
        "seed_redundancy",
        "run_bridge_coverage",
        "corridor_role_coverage",
        "answer_preserve_activation_rate",
        "preserved_answer_usefulness",
        "path_complete_rate",
        "bridge_answer_pair_retention",
        "incomplete_path_rate",
        "chain_compactness",
        "path_preserve_activation_rate",
        "anchor_bridge_balance",
        "bridge_purity",
        "answer_side_density",
        "support_set_compactness",
        "bridge_noise_ratio",
        "useful_bridge_rate",
        "bridge_to_answer_path_hit",
        "conversion_after_preserve",
        "conversion_after_path_preserve",
        "hotpot_overpreserve_rate",
        "redundancy_rate",
        "connector_quality",
    ):
        if key not in metrics and key in diagnostics:
            metrics[key] = _safe_float(diagnostics.get(key, 0.0), 0.0)

    m = ((row or {}).get("metrics", {}) or {})
    sf_recall = _safe_float(m.get("supporting_fact_recall", 0.0), 0.0)
    rendered_sf_recall = _safe_float(m.get("rendered_supporting_fact_recall", 0.0), 0.0)
    f1 = _safe_float(m.get("f1", m.get("F1", 0.0)), 0.0)
    em = _safe_float(m.get("em", m.get("EM", 0.0)), 0.0)
    supporting_fact_precision = _safe_float(m.get("supporting_fact_precision", 0.0), 0.0)
    rendered_supporting_fact_precision = _safe_float(m.get("rendered_supporting_fact_precision", 0.0), 0.0)
    recall_at_5 = _safe_float(m.get("supporting_fact_recall_at_5", 0.0), 0.0)
    if recall_at_5 <= 0.0:
        at_k = m.get("supporting_fact_recall_at_k", {}) or {}
        if isinstance(at_k, dict):
            recall_at_5 = _safe_float(at_k.get("5", at_k.get(5, 0.0)), 0.0)

    if "rendered_retention" not in metrics and sf_recall > 0.0:
        metrics["rendered_retention"] = float(safe_div(rendered_sf_recall, sf_recall))

    metrics["answer_conversion"] = float(safe_div(f1, rendered_sf_recall))
    metrics["em_conversion"] = float(safe_div(em, rendered_sf_recall))
    metrics["f1_per_r5"] = float(safe_div(f1, recall_at_5))

    run_bridge_coverage = _safe_float(diagnostics.get("run_bridge_coverage", 0.0), 0.0)
    seed_bridge_recall = _safe_float(diagnostics.get("seed_bridge_recall", 0.0), 0.0)
    seed_answer_recall = _safe_float(diagnostics.get("seed_answer_recall", 0.0), 0.0)
    corridor_role_coverage = _safe_float(diagnostics.get("corridor_role_coverage", 0.0), 0.0)
    answer_preserve_activation_rate = _safe_float(
        diagnostics.get("answer_preserve_activation_rate", metrics.get("answer_preserve_activation_rate", 0.0)),
        0.0,
    )
    preserved_answer_usefulness = _safe_float(
        diagnostics.get("preserved_answer_usefulness", metrics.get("preserved_answer_usefulness", 0.0)),
        0.0,
    )
    path_complete_rate = _safe_float(
        diagnostics.get("path_complete_rate", metrics.get("path_complete_rate", 0.0)),
        0.0,
    )
    bridge_answer_pair_retention = _safe_float(
        diagnostics.get("bridge_answer_pair_retention", metrics.get("bridge_answer_pair_retention", 0.0)),
        0.0,
    )
    incomplete_path_rate = _safe_float(
        diagnostics.get("incomplete_path_rate", metrics.get("incomplete_path_rate", 0.0)),
        0.0,
    )
    chain_compactness = _safe_float(
        diagnostics.get("chain_compactness", metrics.get("chain_compactness", 0.0)),
        0.0,
    )
    path_preserve_activation_rate = _safe_float(
        diagnostics.get("path_preserve_activation_rate", metrics.get("path_preserve_activation_rate", 0.0)),
        0.0,
    )
    anchor_bridge_balance = _safe_float(
        diagnostics.get("anchor_bridge_balance", metrics.get("anchor_bridge_balance", 0.0)),
        0.0,
    )
    bridge_purity = _safe_float(diagnostics.get("bridge_purity", metrics.get("bridge_purity", 0.0)), 0.0)
    answer_side_density = _safe_float(
        diagnostics.get("answer_side_density", metrics.get("answer_side_density", 0.0)),
        0.0,
    )
    support_set_compactness = _safe_float(
        diagnostics.get("support_set_compactness", metrics.get("support_set_compactness", 0.0)),
        0.0,
    )
    bridge_noise_ratio = _safe_float(
        diagnostics.get("bridge_noise_ratio", metrics.get("bridge_noise_ratio", 1.0 - bridge_purity)),
        0.0,
    )
    hotpot_overpreserve_rate = _safe_float(
        diagnostics.get("hotpot_overpreserve_rate", metrics.get("hotpot_overpreserve_rate", 0.0)),
        0.0,
    )
    redundancy_rate = _safe_float(diagnostics.get("redundancy_rate", 0.0), 0.0)
    bridge_noise_ratio = max(0.0, min(1.0, bridge_noise_ratio))

    bridge_signal = max(run_bridge_coverage, seed_bridge_recall)
    answer_signal = max(rendered_sf_recall, seed_answer_recall)
    bridge_present = bool(bridge_signal >= 0.35)
    answer_side_present = bool(max(answer_signal, answer_side_density) >= 0.25)
    qa_executed = bool((row or {}).get("qa_executed", True))
    generation_fail = bool(qa_executed and f1 <= 0.01)
    bridge_noisy = bool(
        bridge_present
        and answer_side_present
        and (
            bridge_noise_ratio >= 0.55
            or (
                max(supporting_fact_precision, rendered_supporting_fact_precision) < 0.25
                and (redundancy_rate >= 0.60 or f1 < 0.25)
            )
        )
    )

    metrics["bridge_purity"] = max(0.0, min(1.0, bridge_purity))
    metrics["answer_side_density"] = max(0.0, min(1.0, answer_side_density))
    metrics["support_set_compactness"] = max(0.0, min(1.0, support_set_compactness))
    metrics["bridge_noise_ratio"] = float(bridge_noise_ratio)
    metrics["answer_preserve_activation_rate"] = max(0.0, min(1.0, answer_preserve_activation_rate))
    metrics["preserved_answer_usefulness"] = max(0.0, min(1.0, preserved_answer_usefulness))
    metrics["path_complete_rate"] = max(0.0, min(1.0, path_complete_rate))
    metrics["bridge_answer_pair_retention"] = max(0.0, min(1.0, bridge_answer_pair_retention))
    metrics["incomplete_path_rate"] = max(0.0, min(1.0, incomplete_path_rate))
    metrics["chain_compactness"] = max(0.0, min(1.0, chain_compactness))
    metrics["path_preserve_activation_rate"] = max(0.0, min(1.0, path_preserve_activation_rate))
    metrics["anchor_bridge_balance"] = max(0.0, min(1.0, anchor_bridge_balance))
    metrics["hotpot_overpreserve_rate"] = max(0.0, min(1.0, hotpot_overpreserve_rate))
    metrics["conversion_after_bridge"] = (
        1.0 if (bridge_present and answer_side_present and (f1 > 0.0 or em > 0.0)) else 0.0
    )
    metrics["conversion_after_preserve"] = (
        1.0
        if (
            answer_preserve_activation_rate >= 0.5
            and bridge_present
            and answer_side_present
            and (f1 > 0.0 or em > 0.0)
        )
        else 0.0
    )
    metrics["conversion_after_path_preserve"] = (
        1.0
        if (
            path_preserve_activation_rate >= 0.25
            and bridge_present
            and answer_side_present
            and path_complete_rate >= 0.45
            and (f1 > 0.0 or em > 0.0)
        )
        else 0.0
    )

    metrics.setdefault(
        "useful_bridge_rate",
        1.0 if (bridge_present and answer_side_present and bridge_purity >= 0.35 and (f1 > 0.0 or em > 0.0)) else 0.0,
    )
    metrics.setdefault(
        "bridge_to_answer_path_hit",
        1.0
        if (bridge_present and answer_side_present and corridor_role_coverage >= 0.45 and bridge_purity >= 0.35)
        else 0.0,
    )

    metrics["error_bucket_bridge_missing"] = 0.0
    metrics["error_bucket_answer_side_missing"] = 0.0
    metrics["error_bucket_bridge_present_noisy"] = 0.0
    metrics["error_bucket_answer_present_generation_fail"] = 0.0
    if not bridge_present:
        metrics["error_bucket_bridge_missing"] = 1.0
    elif not answer_side_present:
        metrics["error_bucket_answer_side_missing"] = 1.0
    elif generation_fail:
        metrics["error_bucket_answer_present_generation_fail"] = 1.0
    elif bridge_noisy:
        metrics["error_bucket_bridge_present_noisy"] = 1.0
    return metrics


def _compute_stagewise_summary(rows):
    seq = list(rows or [])
    if not seq:
        return {}

    values = {key: [] for key in STAGEWISE_LOSS_KEYS}
    rows_with = 0
    for row in seq:
        item = _extract_stagewise_from_row(row)
        if not item:
            continue
        rows_with += 1
        for key in STAGEWISE_LOSS_KEYS:
            if key in item:
                values[key].append(_safe_float(item[key], 0.0))

    if rows_with <= 0:
        return {}

    summary = {
        "query_count": int(len(seq)),
        "query_with_stagewise": int(rows_with),
    }
    for key in STAGEWISE_LOSS_KEYS:
        vals = list(values.get(key, []) or [])
        if not vals:
            continue
        summary[key] = float(mean_or_zero(vals))
        summary[f"{key}_count"] = int(len(vals))
    return summary


def _attach_stagewise_summary(path, payload):
    out = dict(payload or {})
    p = Path(path)
    name = str(p.name or "")
    if name not in {"rag_summary.json", "retrieval_summary.json"}:
        return out

    query_name = "rag_query_results.jsonl" if name == "rag_summary.json" else "retrieval_query_results.jsonl"
    rows = _read_jsonl_rows(p.with_name(query_name))
    stagewise_summary = _compute_stagewise_summary(rows)
    if stagewise_summary:
        out["stagewise_loss_funnel"] = stagewise_summary
        for key in STAGEWISE_LOSS_KEYS:
            if key in stagewise_summary:
                out[f"stagewise_{key}"] = float(stagewise_summary[key])
    return out


def mean_or_zero(values):
    if not values:
        return 0.0
    return float(mean(values))


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    ensure_parent(path)
    payload_to_write = dict(payload or {})
    payload_to_write = _attach_stagewise_summary(path, payload_to_write)
    if Path(path).name == "rag_summary.json":
        payload_to_write.update(compute_rag_derived_metrics(payload_to_write))
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload_to_write, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    ensure_parent(path)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def timestamp_for_filename():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def timestamp_iso_utc():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def markdown_table(headers, rows):
    safe_headers = [str(h) for h in headers]
    lines = [
        "| " + " | ".join(safe_headers) + " |",
        "| " + " | ".join(["---"] * len(safe_headers)) + " |",
    ]
    for row in rows:
        safe_row = [str(cell) for cell in row]
        lines.append("| " + " | ".join(safe_row) + " |")
    return "\n".join(lines)


def infer_config_filename_hints(path):
    name = Path(path).name.lower()
    hints = {}
    if "entity_chunk_graph" in name:
        hints["graph_mode"] = "entity_chunk_graph"
    elif any(tok in name for tok in ["fusion", "semantic_bridge", "baseline_aggressive", "top1corr"]):
        hints["graph_mode"] = "current_entity_graph"

    if "mixed_diffusion" in name:
        hints["chunk_node_enabled_in_diffusion"] = True
    elif "entity_seed_chunk_lift" in name:
        hints["chunk_node_enabled_in_diffusion"] = False

    return hints


def validate_config_against_filename(path, payload):
    payload = dict(payload or {})
    hints = infer_config_filename_hints(path)
    warnings = []
    for key, expected in hints.items():
        if key not in payload:
            continue
        actual = payload.get(key)
        if actual != expected:
            warnings.append({
                "field": str(key),
                "expected_from_filename": expected,
                "actual_in_yaml": actual,
            })

    graph_mode = str(payload.get("graph_mode", "current_entity_graph") or "current_entity_graph")
    if graph_mode == "entity_chunk_graph":
        if int(payload.get("samples_per_anchor", 1) or 1) <= 1:
            warnings.append({
                "field": "samples_per_anchor",
                "expected_from_strategy": ">=2 for stable run selection",
                "actual_in_yaml": payload.get("samples_per_anchor", 1),
            })
        if int(payload.get("phase1_run_preshortlist_topm", 1) or 1) <= 1:
            warnings.append({
                "field": "phase1_run_preshortlist_topm",
                "expected_from_strategy": ">=2 for stable run selection",
                "actual_in_yaml": payload.get("phase1_run_preshortlist_topm", 1),
            })
        if int(payload.get("phase1_full_run_score_topk", 1) or 1) <= 1:
            warnings.append({
                "field": "phase1_full_run_score_topk",
                "expected_from_strategy": ">=2 for stable run selection",
                "actual_in_yaml": payload.get("phase1_full_run_score_topk", 1),
            })
    return warnings


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def load_yaml(path):
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in yaml file: {path}")
    warnings = validate_config_against_filename(path, data)
    data["_config_file"] = str(path)
    data["_config_audit"] = {
        "warning_count": int(len(warnings)),
        "warnings": warnings,
    }
    return data
