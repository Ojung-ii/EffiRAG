#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _select_best(rows, baseline_variant: str, max_recall_drop: float = 0.01):
    non_base = [r for r in rows if str(r.get("variant", "")) != baseline_variant]
    guarded = [r for r in non_base if float(r.get("delta_recall_at_20", 0.0)) >= -abs(max_recall_drop)]
    pool = guarded if guarded else non_base
    if not pool:
        return None
    return max(
        pool,
        key=lambda r: (
            float(r.get("delta_f1", -1e9)),
            float(r.get("delta_em", -1e9)),
            float(r.get("delta_rendered_recall", -1e9)),
            -float(r.get("delta_total_ms", 0.0)),
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize retrieval+delivery fusion final interpretation")
    ap.add_argument("--hotpot-summary", required=True)
    ap.add_argument("--wiki-summary", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--git-branch", default="")
    ap.add_argument("--git-commit", default="")
    ap.add_argument("--git-tag", default="")
    args = ap.parse_args()

    hotpot = _load(args.hotpot_summary)
    wiki = _load(args.wiki_summary)

    h_base = str(hotpot.get("baseline_variant", "hotpot_baseline"))
    w_base = str(wiki.get("baseline_variant", "2wiki_baseline"))
    h_rows = list(hotpot.get("rows", []) or [])
    w_rows = list(wiki.get("rows", []) or [])

    h_best = _select_best(h_rows, h_base, max_recall_drop=0.01)
    w_best = _select_best(w_rows, w_base, max_recall_drop=0.01)

    result = {
        "experiment_family": "next_method_design",
        "question_being_answered": "Can retrieval+chunk-grounded delivery fusion reduce remaining quality gaps without changing frozen core?",
        "baseline_reference": "frozen baseline (speed_default=aggressive, quality_variant=top1corr_t1)",
        "frozen_config_reference": "configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml",
        "dataset_scope": ["hotpotqa", "2wikimultihopqa"],
        "changed_components": ["retrieval_delivery_fusion", "semantic_union", "chunk_grounded_delivery"],
        "git_branch": args.git_branch,
        "git_commit": args.git_commit,
        "git_tag": args.git_tag,
        "hotpot": {
            "baseline_variant": h_base,
            "best_variant": (h_best or {}).get("variant", ""),
            "best_row": h_best,
            "summary_path": str(Path(args.hotpot_summary).resolve()),
        },
        "wiki": {
            "baseline_variant": w_base,
            "best_variant": (w_best or {}).get("variant", ""),
            "best_row": w_best,
            "summary_path": str(Path(args.wiki_summary).resolve()),
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Retrieval + Delivery Fusion Final Interpretation",
        "",
        "- experiment_family: `next_method_design`",
        "- baseline_reference: `frozen baseline (speed_default=aggressive, quality_variant=top1corr_t1)`",
        f"- git_branch: {args.git_branch}",
        f"- git_commit: {args.git_commit}",
        f"- git_tag: {args.git_tag}",
        "",
        "## HotpotQA",
    ]
    if h_best:
        lines.append(
            f"- best fusion candidate: `{h_best.get('variant', '')}` "
            f"(ΔR20={float(h_best.get('delta_recall_at_20', 0.0)):+.4f}, "
            f"Δrendered={float(h_best.get('delta_rendered_recall', 0.0)):+.4f}, "
            f"ΔEM={float(h_best.get('delta_em', 0.0)):+.4f}, "
            f"ΔF1={float(h_best.get('delta_f1', 0.0)):+.4f}, "
            f"Δtotal_ms={float(h_best.get('delta_total_ms', 0.0)):+.2f})"
        )
    else:
        lines.append("- no eligible non-baseline candidate found.")

    lines.extend(["", "## 2Wiki"])
    if w_best:
        lines.append(
            f"- best fusion candidate: `{w_best.get('variant', '')}` "
            f"(ΔR20={float(w_best.get('delta_recall_at_20', 0.0)):+.4f}, "
            f"Δrendered={float(w_best.get('delta_rendered_recall', 0.0)):+.4f}, "
            f"ΔEM={float(w_best.get('delta_em', 0.0)):+.4f}, "
            f"ΔF1={float(w_best.get('delta_f1', 0.0)):+.4f}, "
            f"Δtotal_ms={float(w_best.get('delta_total_ms', 0.0)):+.2f})"
        )
    else:
        lines.append("- no eligible non-baseline candidate found.")

    lines.extend(
        [
            "",
            "## Decision",
            "- Hotpot: prioritize `semantic_union + chunk_grounded_bridge` only if it outperforms both standalone retrieval and standalone delivery variants.",
            "- 2Wiki: prioritize shallow package/backfill variants that improve EM/F1 with bounded latency penalty.",
            "- Keep full chunk indexing as follow-up track unless fusion gains stall.",
            "",
        ]
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
