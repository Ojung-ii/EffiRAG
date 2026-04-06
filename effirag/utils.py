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


def compute_rag_derived_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    summary = dict(summary or {})
    sf_recall = float(summary.get("supporting_fact_recall", 0.0) or 0.0)
    rendered_sf_recall = float(
        summary.get(
            "rendered_supporting_fact_recall",
            summary.get("rendered_sf_recall", 0.0),
        )
        or 0.0
    )
    f1 = float(summary.get("f1", summary.get("F1", 0.0)) or 0.0)
    return {
        "rendered_retention": float(safe_div(rendered_sf_recall, sf_recall)),
        "answer_conversion": float(safe_div(f1, rendered_sf_recall)),
    }


STAGEWISE_LOSS_KEYS = (
    "anchor_hit_rate",
    "proposal_hit_rate",
    "phase1_hit_rate",
    "final_chunk_hit_rate",
    "rendered_retention",
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

    if "rendered_retention" not in metrics:
        m = ((row or {}).get("metrics", {}) or {})
        sf_recall = _safe_float(m.get("supporting_fact_recall", 0.0), 0.0)
        rendered_sf_recall = _safe_float(m.get("rendered_supporting_fact_recall", 0.0), 0.0)
        if sf_recall > 0.0:
            metrics["rendered_retention"] = float(safe_div(rendered_sf_recall, sf_recall))
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
