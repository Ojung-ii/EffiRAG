#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(dict(json.loads(raw)))
            except Exception:
                continue
    return rows


def _sample_id(row: dict, idx: int) -> str:
    for key in ("sample_id", "qid", "id", "query_id"):
        if row.get(key) is not None and str(row.get(key)).strip():
            return str(row.get(key))
    return f"row_{idx}"


def _as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _first_nonnull(*values):
    for v in values:
        if v is not None:
            return v
    return None


def _extract_selected_ids(row: dict):
    retrieval = dict(row.get("retrieval", {}) or {})
    diagnostics = dict(retrieval.get("diagnostics", {}) or {})
    rendered = dict(row.get("rendered", {}) or {})
    candidates = [
        row.get("selected_text_unit_ids"),
        row.get("retrieval_selected_sentence_ids"),
        row.get("rendered_sentence_ids"),
        rendered.get("sentence_ids"),
        diagnostics.get("selected_text_unit_ids"),
        retrieval.get("selected_sentence_ids"),
        row.get("selected_chunks"),
        row.get("retrieved_texts"),
        row.get("selected_texts"),
    ]
    for c in candidates:
        if isinstance(c, list) and c:
            return [str(x) for x in c]
    return []


def _extract_seed_ids(row: dict):
    retrieval = dict(row.get("retrieval", {}) or {})
    seeds = retrieval.get("seeds")
    if isinstance(seeds, list):
        return [str(x) for x in seeds]
    diag = dict(retrieval.get("diagnostics", {}) or {})
    alt = diag.get("seed_nodes")
    if isinstance(alt, list):
        return [str(x) for x in alt]
    return []


def _extract_pair_ids(row: dict):
    retrieval = dict(row.get("retrieval", {}) or {})
    diag = dict(retrieval.get("diagnostics", {}) or {})
    pairs = diag.get("pair_shortlist")
    out = []
    if isinstance(pairs, list):
        for p in pairs:
            if isinstance(p, dict):
                a = str(p.get("anchor", ""))
                z = str(p.get("seed", ""))
                if a or z:
                    out.append(f"{a}::{z}")
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append(f"{p[0]}::{p[1]}")
    return out


def _extract_em_f1(row: dict):
    metrics = dict(row.get("metrics", {}) or {})
    em = _first_nonnull(
        _as_float(metrics.get("em")),
        _as_float(row.get("em")),
    )
    f1 = _first_nonnull(
        _as_float(metrics.get("f1")),
        _as_float(row.get("f1")),
    )
    return em, f1


def _extract_context_tokens(row: dict):
    gen = dict(row.get("generation_diagnostics", {}) or {})
    eff = dict(row.get("efficiency", {}) or {})
    return _first_nonnull(
        _as_float(row.get("context_tokens")),
        _as_float(gen.get("prompt_tokens")),
        _as_float(eff.get("prompt_tokens")),
    )


def _extract_retrieval_ms(row: dict):
    eff = dict(row.get("efficiency", {}) or {})
    return _first_nonnull(
        _as_float(row.get("retrieval_ms")),
        _as_float(eff.get("retrieval_latency_ms")),
        _as_float(eff.get("retrieval_ms")),
    )


def _extract_dataset(row: dict):
    for key in ("dataset", "dataset_name", "source_dataset"):
        v = row.get(key)
        if v:
            return str(v)
    retrieval = dict(row.get("retrieval", {}) or {})
    diag = dict(retrieval.get("diagnostics", {}) or {})
    v = diag.get("dataset")
    if v:
        return str(v)
    return "unknown"


def _mean_or_none(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(mean(vals))


def _safe_div(a, b):
    if b == 0:
        return None
    return float(a) / float(b)


def _compute_overlap_stats(base_rows: dict, var_rows: dict):
    base_ids = set(base_rows.keys())
    var_ids = set(var_rows.keys())
    matched_ids = sorted(base_ids.intersection(var_ids))
    missing_in_variant = sorted(base_ids.difference(var_ids))
    extra_in_variant = sorted(var_ids.difference(base_ids))

    top1_matches = 0
    top1_compared = 0
    jaccards = []
    overlap_vs_baseline = []
    seed_jaccards = []
    pair_jaccards = []
    em_deltas = []
    f1_deltas = []
    token_deltas = []
    retrieval_ms_deltas = []

    by_dataset = {}

    for sid in matched_ids:
        b = base_rows[sid]
        v = var_rows[sid]
        dataset = _extract_dataset(b)
        by_dataset.setdefault(
            dataset,
            {
                "matched": 0,
                "top1_match_count": 0,
                "top1_compared": 0,
                "jaccard": [],
                "em_delta": [],
                "f1_delta": [],
            },
        )
        by_dataset[dataset]["matched"] += 1

        b_sel = _extract_selected_ids(b)
        v_sel = _extract_selected_ids(v)
        b_set = set(b_sel)
        v_set = set(v_sel)

        if b_sel and v_sel:
            top1_compared += 1
            by_dataset[dataset]["top1_compared"] += 1
            if str(b_sel[0]) == str(v_sel[0]):
                top1_matches += 1
                by_dataset[dataset]["top1_match_count"] += 1

        union = b_set.union(v_set)
        inter = b_set.intersection(v_set)
        if union:
            jac = _safe_div(len(inter), len(union))
            jaccards.append(jac)
            by_dataset[dataset]["jaccard"].append(jac)
        if b_set:
            overlap_vs_baseline.append(_safe_div(len(inter), len(b_set)))

        b_seed = set(_extract_seed_ids(b))
        v_seed = set(_extract_seed_ids(v))
        seed_union = b_seed.union(v_seed)
        if seed_union:
            seed_jaccards.append(_safe_div(len(b_seed.intersection(v_seed)), len(seed_union)))

        b_pair = set(_extract_pair_ids(b))
        v_pair = set(_extract_pair_ids(v))
        pair_union = b_pair.union(v_pair)
        if pair_union:
            pair_jaccards.append(_safe_div(len(b_pair.intersection(v_pair)), len(pair_union)))

        b_em, b_f1 = _extract_em_f1(b)
        v_em, v_f1 = _extract_em_f1(v)
        if b_em is not None and v_em is not None:
            d = float(v_em - b_em)
            em_deltas.append(d)
            by_dataset[dataset]["em_delta"].append(d)
        if b_f1 is not None and v_f1 is not None:
            d = float(v_f1 - b_f1)
            f1_deltas.append(d)
            by_dataset[dataset]["f1_delta"].append(d)

        b_tok = _extract_context_tokens(b)
        v_tok = _extract_context_tokens(v)
        if b_tok is not None and v_tok is not None:
            token_deltas.append(float(v_tok - b_tok))

        b_ms = _extract_retrieval_ms(b)
        v_ms = _extract_retrieval_ms(v)
        if b_ms is not None and v_ms is not None:
            retrieval_ms_deltas.append(float(v_ms - b_ms))

    per_dataset = {}
    for ds, payload in by_dataset.items():
        per_dataset[ds] = {
            "matched_samples": int(payload["matched"]),
            "top1_match_rate": _safe_div(payload["top1_match_count"], payload["top1_compared"]),
            "selected_text_jaccard_mean": _mean_or_none(payload["jaccard"]),
            "em_delta_mean": _mean_or_none(payload["em_delta"]),
            "f1_delta_mean": _mean_or_none(payload["f1_delta"]),
        }

    return {
        "sample_count_baseline": int(len(base_ids)),
        "sample_count_variant": int(len(var_ids)),
        "matched_sample_count": int(len(matched_ids)),
        "missing_sample_ids_in_variant": missing_in_variant,
        "extra_sample_ids_in_variant": extra_in_variant,
        "selected_text_top1_match_rate": _safe_div(top1_matches, top1_compared),
        "selected_text_topk_jaccard_mean": _mean_or_none(jaccards),
        "selected_text_overlap_vs_baseline_mean": _mean_or_none(overlap_vs_baseline),
        "seed_overlap_jaccard_mean": _mean_or_none(seed_jaccards),
        "shortlisted_pair_overlap_jaccard_mean": _mean_or_none(pair_jaccards),
        "em_delta_mean": _mean_or_none(em_deltas),
        "f1_delta_mean": _mean_or_none(f1_deltas),
        "context_token_delta_mean": _mean_or_none(token_deltas),
        "retrieval_latency_ms_delta_mean": _mean_or_none(retrieval_ms_deltas),
        "per_dataset_summary": per_dataset,
    }


def _to_markdown(payload: dict, baseline: str, variant: str):
    lines = []
    lines.append("# Phase5 Score Ablation Compare")
    lines.append("")
    lines.append(f"- baseline: `{baseline}`")
    lines.append(f"- variant: `{variant}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    rows = [
        ("sample_count_baseline", payload.get("sample_count_baseline")),
        ("sample_count_variant", payload.get("sample_count_variant")),
        ("matched_sample_count", payload.get("matched_sample_count")),
        ("selected_text_top1_match_rate", payload.get("selected_text_top1_match_rate")),
        ("selected_text_topk_jaccard_mean", payload.get("selected_text_topk_jaccard_mean")),
        ("selected_text_overlap_vs_baseline_mean", payload.get("selected_text_overlap_vs_baseline_mean")),
        ("seed_overlap_jaccard_mean", payload.get("seed_overlap_jaccard_mean")),
        ("shortlisted_pair_overlap_jaccard_mean", payload.get("shortlisted_pair_overlap_jaccard_mean")),
        ("em_delta_mean", payload.get("em_delta_mean")),
        ("f1_delta_mean", payload.get("f1_delta_mean")),
        ("context_token_delta_mean", payload.get("context_token_delta_mean")),
        ("retrieval_latency_ms_delta_mean", payload.get("retrieval_latency_ms_delta_mean")),
    ]
    for k, v in rows:
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.6f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Per Dataset")
    lines.append("")
    lines.append("| dataset | matched | top1_match_rate | jaccard_mean | dEM | dF1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    per_dataset = dict(payload.get("per_dataset_summary", {}) or {})
    for ds in sorted(per_dataset.keys()):
        row = dict(per_dataset.get(ds, {}) or {})
        lines.append(
            "| {ds} | {matched} | {t1} | {jac} | {dem} | {df1} |".format(
                ds=ds,
                matched=row.get("matched_samples", 0),
                t1="NA" if row.get("top1_match_rate") is None else f"{float(row.get('top1_match_rate')):.6f}",
                jac="NA" if row.get("selected_text_jaccard_mean") is None else f"{float(row.get('selected_text_jaccard_mean')):.6f}",
                dem="NA" if row.get("em_delta_mean") is None else f"{float(row.get('em_delta_mean')):.6f}",
                df1="NA" if row.get("f1_delta_mean") is None else f"{float(row.get('f1_delta_mean')):.6f}",
            )
        )
    lines.append("")
    missing = list(payload.get("missing_sample_ids_in_variant", []) or [])
    extra = list(payload.get("extra_sample_ids_in_variant", []) or [])
    lines.append(f"- missing_sample_ids_in_variant: {len(missing)}")
    lines.append(f"- extra_sample_ids_in_variant: {len(extra)}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs ablation JSONL for Phase5 score diagnostics.")
    parser.add_argument("--baseline", required=True, help="Path to baseline rag_query_results.jsonl")
    parser.add_argument("--variant", required=True, help="Path to variant rag_query_results.jsonl")
    parser.add_argument("--output", required=True, help="Path to output JSON report")
    parser.add_argument("--markdown-output", default="", help="Optional Markdown output path")
    args = parser.parse_args()

    baseline_path = Path(str(args.baseline)).expanduser().resolve()
    variant_path = Path(str(args.variant)).expanduser().resolve()
    output_path = Path(str(args.output)).expanduser().resolve()
    markdown_path = Path(str(args.markdown_output)).expanduser().resolve() if str(args.markdown_output).strip() else None

    baseline_rows_raw = _read_jsonl(baseline_path)
    variant_rows_raw = _read_jsonl(variant_path)
    baseline_rows = {_sample_id(r, i): r for i, r in enumerate(baseline_rows_raw)}
    variant_rows = {_sample_id(r, i): r for i, r in enumerate(variant_rows_raw)}

    result = {
        "baseline": str(baseline_path),
        "variant": str(variant_path),
        "stats": _compute_overlap_stats(baseline_rows, variant_rows),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            _to_markdown(result["stats"], baseline=str(baseline_path), variant=str(variant_path)),
            encoding="utf-8",
        )

    print(str(output_path))


if __name__ == "__main__":
    main()
