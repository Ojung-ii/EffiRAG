#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


CATEGORIES = [
    "UQ_ENTITY_MISS",
    "SEED_SELECTION_MISS",
    "REFINEMENT_DRIFT",
    "SEED_TO_EVIDENCE_MISS",
    "CANDIDATE_CHAIN_INFEASIBLE",
    "FINAL_SELECTION_FAILED",
    "SELECTED_NOT_SUFFICIENT",
    "QA_PROMPT_FAILED",
    "OTHER",
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _run_dirs(root: Path) -> List[Path]:
    return sorted(p.parent for p in root.glob("*/*/*/rag_summary.json"))


def _parts(root: Path, run_dir: Path) -> Dict[str, str]:
    parts = list(run_dir.relative_to(root).parts)
    return {
        "profile": parts[0] if len(parts) > 0 else "",
        "variant": parts[1] if len(parts) > 1 else "",
        "dataset": parts[2] if len(parts) > 2 else "",
    }


def _classify(q: Dict[str, Any]) -> str:
    if _safe_float(q.get("entity_universe_size", 0.0), 0.0) <= 0:
        return "UQ_ENTITY_MISS"
    if bool(q.get("phase8_pamae_enabled", False)) and _safe_float(q.get("best_seed_score", 0.0), 0.0) <= 0.0:
        return "SEED_SELECTION_MISS"
    if bool(q.get("phase8_pamae_enabled", False)) and _safe_float(q.get("num_changed_seeds", 0.0), 0.0) > 0.0 and _safe_float(q.get("candidate_gold_recall", 0.0), 0.0) <= 0.0:
        return "REFINEMENT_DRIFT"
    if bool(q.get("phase8_pamae_enabled", False)) and _safe_float(q.get("seed_evidence_gold_hit_rate_eval_only", 0.0), 0.0) <= 0.0 and _safe_float(q.get("candidate_gold_recall", 0.0), 0.0) <= 0.0:
        return "SEED_TO_EVIDENCE_MISS"
    if _safe_float(q.get("chain_unit_oracle_feasible", 0.0), 0.0) <= 0.0:
        return "CANDIDATE_CHAIN_INFEASIBLE"
    if _safe_float(q.get("candidate_gold_recall", 0.0), 0.0) > _safe_float(q.get("sf_recall", 0.0), 0.0) + 0.2:
        return "FINAL_SELECTION_FAILED"
    if _safe_float(q.get("sf_recall", 0.0), 0.0) > _safe_float(q.get("f1", 0.0), 0.0) + 0.2:
        return "QA_PROMPT_FAILED"
    if _safe_float(q.get("sf_recall", 0.0), 0.0) < 0.5:
        return "SELECTED_NOT_SUFFICIENT"
    return "OTHER"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase8 PAMAE-inspired entity seeding failures.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    counts = defaultdict(Counter)
    examples = defaultdict(list)
    for run_dir in _run_dirs(root):
        parts = _parts(root, run_dir)
        run_key = f"{parts['dataset']}/{parts['profile']}/{parts['variant']}"
        for q in _read_jsonl(run_dir / "phase8_query_trace.jsonl"):
            cat = _classify(q)
            row = {**parts, "query_id": str(q.get("query_id", "")), "category": cat, "question": str(q.get("question", ""))}
            rows.append(row)
            counts[run_key][cat] += 1
            if len(examples[(run_key, cat)]) < 3:
                examples[(run_key, cat)].append(row)

    (root / "phase8_failure_attribution.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )

    lines = ["# Phase8 PAMAE Failure Analysis", ""]
    for run_key in sorted(counts):
        total = sum(counts[run_key].values())
        lines.append(f"## {run_key}")
        lines.append("")
        lines.append("| category | count | rate |")
        lines.append("| --- | ---: | ---: |")
        for cat in CATEGORIES:
            cnt = int(counts[run_key].get(cat, 0))
            rate = float(cnt / total) if total else 0.0
            lines.append(f"| {cat} | {cnt} | {rate:.4f} |")
        lines.append("")
        for cat in CATEGORIES:
            exs = examples.get((run_key, cat), [])
            if not exs:
                continue
            lines.append(f"Examples for {cat}:")
            for ex in exs:
                lines.append(f"- {ex['query_id']}: {ex['question'][:160]}")
            lines.append("")

    out_md = root / "phase8_pamae_failure_analysis.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
