#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _pick_best(rows, baseline_variant: str):
    non_base = [r for r in rows if str(r.get("variant", "")) != baseline_variant]
    if not non_base:
        return None
    guarded = [r for r in non_base if float(r.get("delta_recall_at_20", 0.0)) >= -0.01]
    pool = guarded if guarded else non_base
    return max(
        pool,
        key=lambda r: (
            float(r.get("delta_f1", -1e9)),
            float(r.get("delta_em", -1e9)),
            float(r.get("delta_rendered_recall", -1e9)),
            -float(r.get("delta_total_ms", 0.0)),
        ),
    )


def _metric(d: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in d:
            x = _to_float(d.get(k))
            if x is not None:
                return x
    return None


def _dataset_rows(dataset: str, summary: Dict[str, Any], hippo_ref: Dict[str, Any]):
    base_var = str(summary.get("baseline_variant", "baseline"))
    rows = list(summary.get("rows", []) or [])
    baseline = next((r for r in rows if str(r.get("variant", "")) == base_var), None)
    best = _pick_best(rows, base_var)

    hip = dict((hippo_ref or {}).get(dataset, {}) or {})
    hip_row = {
        "dataset": dataset,
        "method": "HippoRAG2",
        "variant": "hipporag2_reference",
        "Recall@1": _metric(hip, "Recall@1", "recall_at_1", "R@1"),
        "Recall@5": _metric(hip, "Recall@5", "recall_at_5", "R@5"),
        "Recall@20": _metric(hip, "Recall@20", "recall_at_20", "R@20"),
        "rendered_recall": _metric(hip, "rendered_recall", "Rendered", "rendered_supporting_fact_recall"),
        "EM": _metric(hip, "EM", "em"),
        "F1": _metric(hip, "F1", "f1"),
        "retrieval_ms": _metric(hip, "retrieval_ms", "retrieval_latency_ms"),
        "total_ms": _metric(hip, "total_ms", "total_latency_ms"),
    }

    def from_eff(name: str, tag: str, r: Optional[Dict[str, Any]]):
        if not r:
            return {
                "dataset": dataset,
                "method": name,
                "variant": tag,
                "Recall@1": None,
                "Recall@5": None,
                "Recall@20": None,
                "rendered_recall": None,
                "EM": None,
                "F1": None,
                "retrieval_ms": None,
                "total_ms": None,
            }
        return {
            "dataset": dataset,
            "method": name,
            "variant": str(r.get("variant", tag)),
            "Recall@1": _to_float(r.get("recall_at_1")),
            "Recall@5": _to_float(r.get("recall_at_5")),
            "Recall@20": _to_float(r.get("recall_at_20")),
            "rendered_recall": _to_float(r.get("rendered_recall")),
            "EM": _to_float(r.get("em")),
            "F1": _to_float(r.get("f1")),
            "retrieval_ms": _to_float(r.get("retrieval_ms")),
            "total_ms": _to_float(r.get("total_ms")),
        }

    eff_base = from_eff("EffiRAG(frozen_baseline)", base_var, baseline)
    eff_best = from_eff("EffiRAG(best_fusion)", "best_fusion", best)

    def add_delta(row: Dict[str, Any], ref: Dict[str, Any]):
        row = dict(row)
        for k, dk in [("Recall@20", "dR20_vs_HippoRAG2"), ("EM", "dEM_vs_HippoRAG2"), ("F1", "dF1_vs_HippoRAG2")]:
            rv = row.get(k)
            hv = ref.get(k)
            row[dk] = (rv - hv) if (rv is not None and hv is not None) else None
        for k, dk in [("retrieval_ms", "dRetrievalMs_vs_HippoRAG2"), ("total_ms", "dTotalMs_vs_HippoRAG2")]:
            rv = row.get(k)
            hv = ref.get(k)
            row[dk] = (rv - hv) if (rv is not None and hv is not None) else None
        return row

    out = [hip_row, add_delta(eff_base, hip_row), add_delta(eff_best, hip_row)]
    return out, base_var, (best or {}).get("variant", "")


def _fmt(v, d=4):
    if v is None:
        return "N/A"
    return f"{v:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HippoRAG2 gap tracking table for fusion experiments")
    ap.add_argument("--hotpot-summary", required=True)
    ap.add_argument("--wiki-summary", required=True)
    ap.add_argument("--hipporag-ref", default="")
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--git-branch", default="")
    ap.add_argument("--git-commit", default="")
    ap.add_argument("--git-tag", default="")
    args = ap.parse_args()

    hot = _load_json(args.hotpot_summary)
    wiki = _load_json(args.wiki_summary)

    hippo_ref = {}
    if args.hipporag_ref and Path(args.hipporag_ref).exists():
        hippo_ref = _load_json(args.hipporag_ref)

    hot_rows, hot_base, hot_best = _dataset_rows("hotpotqa", hot, hippo_ref)
    wiki_rows, wiki_base, wiki_best = _dataset_rows("2wikimultihopqa", wiki, hippo_ref)

    payload = {
        "experiment_family": "next_method_design",
        "question_being_answered": "How much does fusion reduce the HippoRAG2 gap?",
        "baseline_reference": "frozen baseline",
        "frozen_config_reference": "configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml",
        "dataset_scope": ["hotpotqa", "2wikimultihopqa"],
        "changed_components": ["retrieval_delivery_fusion", "chunk_grounded_delivery"],
        "git_branch": args.git_branch,
        "git_commit": args.git_commit,
        "git_tag": args.git_tag,
        "hipporag_ref_path": str(args.hipporag_ref or ""),
        "rows": hot_rows + wiki_rows,
        "selected_candidates": {
            "hotpotqa": {"baseline": hot_base, "best_fusion": hot_best},
            "2wikimultihopqa": {"baseline": wiki_base, "best_fusion": wiki_best},
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cols = [
        "dataset",
        "method",
        "variant",
        "Recall@1",
        "Recall@5",
        "Recall@20",
        "rendered_recall",
        "EM",
        "F1",
        "retrieval_ms",
        "total_ms",
        "dR20_vs_HippoRAG2",
        "dEM_vs_HippoRAG2",
        "dF1_vs_HippoRAG2",
        "dRetrievalMs_vs_HippoRAG2",
        "dTotalMs_vs_HippoRAG2",
    ]

    lines = [
        "# HippoRAG2 Gap Tracking",
        "",
        "- experiment_family: `next_method_design`",
        "- baseline_reference: `frozen baseline`",
        f"- git_branch: {args.git_branch}",
        f"- git_commit: {args.git_commit}",
        f"- git_tag: {args.git_tag}",
        f"- hipporag_ref_path: {args.hipporag_ref or 'N/A'}",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]

    for r in payload["rows"]:
        row = [
            str(r.get("dataset", "")),
            str(r.get("method", "")),
            str(r.get("variant", "")),
            _fmt(r.get("Recall@1")),
            _fmt(r.get("Recall@5")),
            _fmt(r.get("Recall@20")),
            _fmt(r.get("rendered_recall")),
            _fmt(r.get("EM")),
            _fmt(r.get("F1")),
            _fmt(r.get("retrieval_ms"), d=2),
            _fmt(r.get("total_ms"), d=2),
            _fmt(r.get("dR20_vs_HippoRAG2")),
            _fmt(r.get("dEM_vs_HippoRAG2")),
            _fmt(r.get("dF1_vs_HippoRAG2")),
            _fmt(r.get("dRetrievalMs_vs_HippoRAG2"), d=2),
            _fmt(r.get("dTotalMs_vs_HippoRAG2"), d=2),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Questions",
            "1. fusion 후에도 가장 큰 gap이 Recall@20인지, F1인지를 위 표의 dR20/dF1로 확인.",
            "2. dR20 변화가 작고 dF1이 개선되면 delivery-side 개선 효과로 해석.",
            "3. dR20와 dF1 모두 큰 음수면 retrieval 개선과 delivery 개선 결합이 추가 필요.",
            "",
        ]
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
