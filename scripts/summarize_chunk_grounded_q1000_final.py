#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_rows(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data.get("rows", []) or [])
    return rows


def _baseline(rows: List[dict], name: str = "baseline") -> Optional[dict]:
    for r in rows:
        if str(r.get("variant", "")) == name:
            return r
    return None


def _find_variant(rows: List[dict], name: str) -> Optional[dict]:
    for r in rows:
        if str(r.get("variant", "")) == name:
            return r
    return None


def _fmt(v, nd=4):
    return f"{float(v):.{nd}f}"


def _delta_row(row: Optional[dict], base: Optional[dict]) -> Optional[dict]:
    if row is None or base is None:
        return None
    out = dict(row)
    for key in ["recall_at_20", "rendered_recall", "em", "f1", "retrieval_ms", "total_ms"]:
        out[f"delta_{key}"] = float(row.get(key, 0.0)) - float(base.get(key, 0.0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final interpretation for chunk-grounded q1000 revalidation.")
    parser.add_argument("--hotpot-summary", type=str, required=True)
    parser.add_argument("--wiki-summary", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--git-branch", type=str, default="")
    parser.add_argument("--git-commit", type=str, default="")
    parser.add_argument("--git-tag", type=str, default="")
    args = parser.parse_args()

    hotpot_rows = _load_rows(Path(args.hotpot_summary))
    wiki_rows = _load_rows(Path(args.wiki_summary))

    hp_base = _baseline(hotpot_rows)
    wk_base = _baseline(wiki_rows)

    hp_bridge = _delta_row(_find_variant(hotpot_rows, "corridor_lift_grounded_bridge"), hp_base)
    hp_grounded = _delta_row(_find_variant(hotpot_rows, "corridor_lift_grounded"), hp_base)
    hp_pkg = _delta_row(_find_variant(hotpot_rows, "package_score_grounded"), hp_base)

    wk_pkg = _delta_row(_find_variant(wiki_rows, "package_score_basic"), wk_base)
    wk_backfill = _delta_row(_find_variant(wiki_rows, "sentence_backfill_window_wide"), wk_base)
    wk_corr = _delta_row(_find_variant(wiki_rows, "corridor_lift_grounded"), wk_base)

    # Select candidates with simple policy aligned to request.
    hotpot_candidates = [x for x in [hp_bridge, hp_grounded, hp_pkg] if x is not None]
    hotpot_best = None
    if hotpot_candidates:
        hotpot_best = max(hotpot_candidates, key=lambda r: (float(r.get("delta_f1", -1e9)), float(r.get("delta_em", -1e9))))

    wiki_candidates = [x for x in [wk_pkg, wk_backfill, wk_corr] if x is not None]
    wiki_best = None
    if wiki_candidates:
        wiki_best = max(
            wiki_candidates,
            key=lambda r: (
                float(r.get("delta_f1", -1e9)) + 0.5 * float(r.get("delta_em", -1e9)),
                -abs(float(r.get("delta_recall_at_20", 0.0))),
                -max(0.0, float(r.get("delta_total_ms", 0.0))),
            ),
        )

    result = {
        "experiment_family": "next_method_design",
        "question_being_answered": "q1000 revalidation of chunk-grounded corridor candidates",
        "baseline_reference": "cg_baseline",
        "frozen_config_reference": "configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml",
        "dataset_scope": ["hotpotqa", "2wikimultihopqa"],
        "changed_components": ["chunk_grounded_context_q1000", "evidence_packaging", "render_delivery"],
        "git_branch": args.git_branch,
        "git_commit": args.git_commit,
        "git_tag": args.git_tag,
        "hotpot": {
            "baseline": hp_base,
            "corridor_lift_grounded_bridge": hp_bridge,
            "corridor_lift_grounded": hp_grounded,
            "package_score_grounded_optional": hp_pkg,
            "best_candidate": hotpot_best,
        },
        "wiki": {
            "baseline": wk_base,
            "package_score_basic": wk_pkg,
            "sentence_backfill_window_wide": wk_backfill,
            "corridor_lift_grounded_optional": wk_corr,
            "best_candidate": wiki_best,
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Chunk-Grounded Context q1000 Final Interpretation",
        "",
        "- experiment_family: `next_method_design`",
        "- baseline_reference: `cg_baseline`",
        "- frozen_config_reference: `configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml`",
        f"- git_branch: {args.git_branch}",
        f"- git_commit: {args.git_commit}",
        f"- git_tag: {args.git_tag}",
        "",
        "## HotpotQA",
    ]

    if hp_bridge is not None:
        lines.append(
            "- `corridor_lift_grounded_bridge`: "
            f"ΔR20={_fmt(hp_bridge.get('delta_recall_at_20', 0.0))}, "
            f"Δrendered={_fmt(hp_bridge.get('delta_rendered_recall', 0.0))}, "
            f"ΔEM={_fmt(hp_bridge.get('delta_em', 0.0))}, "
            f"ΔF1={_fmt(hp_bridge.get('delta_f1', 0.0))}, "
            f"Δtotal_ms={_fmt(hp_bridge.get('delta_total_ms', 0.0), 2)}"
        )
    if hp_grounded is not None:
        lines.append(
            "- `corridor_lift_grounded`: "
            f"ΔR20={_fmt(hp_grounded.get('delta_recall_at_20', 0.0))}, "
            f"Δrendered={_fmt(hp_grounded.get('delta_rendered_recall', 0.0))}, "
            f"ΔEM={_fmt(hp_grounded.get('delta_em', 0.0))}, "
            f"ΔF1={_fmt(hp_grounded.get('delta_f1', 0.0))}, "
            f"Δtotal_ms={_fmt(hp_grounded.get('delta_total_ms', 0.0), 2)}"
        )
    if hotpot_best is not None:
        lines.append(f"- Hotpot quality candidate: `{hotpot_best.get('variant', '')}`")

    lines += ["", "## 2Wiki"]
    if wk_pkg is not None:
        lines.append(
            "- `package_score_basic`: "
            f"ΔR20={_fmt(wk_pkg.get('delta_recall_at_20', 0.0))}, "
            f"Δrendered={_fmt(wk_pkg.get('delta_rendered_recall', 0.0))}, "
            f"ΔEM={_fmt(wk_pkg.get('delta_em', 0.0))}, "
            f"ΔF1={_fmt(wk_pkg.get('delta_f1', 0.0))}, "
            f"Δtotal_ms={_fmt(wk_pkg.get('delta_total_ms', 0.0), 2)}"
        )
    if wk_backfill is not None:
        lines.append(
            "- `sentence_backfill_window_wide`: "
            f"ΔR20={_fmt(wk_backfill.get('delta_recall_at_20', 0.0))}, "
            f"Δrendered={_fmt(wk_backfill.get('delta_rendered_recall', 0.0))}, "
            f"ΔEM={_fmt(wk_backfill.get('delta_em', 0.0))}, "
            f"ΔF1={_fmt(wk_backfill.get('delta_f1', 0.0))}, "
            f"Δtotal_ms={_fmt(wk_backfill.get('delta_total_ms', 0.0), 2)}"
        )
    if wiki_best is not None:
        lines.append(f"- 2Wiki quality candidate: `{wiki_best.get('variant', '')}`")

    lines += [
        "",
        "## Final Answers",
        "1. Hotpot에서 strong chunk-grounded corridor가 q1000에서도 유지되는가?",
        "- Above deltas from q1000 revalidation determine this; select `corridor_lift_grounded_bridge` if EM/F1 remain positive with stable Recall@20.",
        "2. 2Wiki에서는 strong grounding보다 shallow package/backfill이 더 적합한가?",
        "- Compare `package_score_basic` and `sentence_backfill_window_wide` against optional strong corridor variant.",
        "3. retrieval core를 유지한 채 delivery만 바꿔 F1 gap을 줄일 수 있는가?",
        "- If EM/F1 improve while Recall@20 is flat, gain is delivery-side packaging gain.",
        "4. full chunk indexing은 지금 당장 필요한가?",
        "- If q1000 gains are consistent on Hotpot/2Wiki, keep full chunk indexing as follow-up track; otherwise escalate to separate chunk-primary branch.",
        "",
    ]

    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
