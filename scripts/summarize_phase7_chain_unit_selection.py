#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


VARIANTS = [
    "baseline_bq",
    "source_balanced_128",
    "source_balanced_160",
    "chain_unit_selection",
]
DATASETS = ["hotpotqa", "2wikimultihopqa"]
PROFILES = ["legacy_512_10", "balanced_384_8"]


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
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


def _mean(values: Iterable[Any]) -> float:
    seq = [_safe_float(v, 0.0) for v in list(values or [])]
    return float(sum(seq) / float(len(seq))) if seq else 0.0


def _bool_mean(values: Iterable[Any]) -> float:
    return _mean(1.0 if bool(v) else 0.0 for v in list(values or []))


def _fmt4(value: Any) -> str:
    return f"{_safe_float(value):.4f}"


def _fmt2(value: Any) -> str:
    return f"{_safe_float(value):.2f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def _summary_value(summary: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in summary:
            return _safe_float(summary.get(key), 0.0)
    return 0.0


def _oracle_row(run_dir: Path) -> Dict[str, Any]:
    payload = _read_json(run_dir / "oracle_context_results.json", {})
    rows = list(payload.get("summary_rows", []) or [])
    if not rows:
        return {}
    for mode in ("current_phase7", "phase7_short", "lightrag_short"):
        for row in rows:
            if str(row.get("prompt_mode", "") or "") == mode:
                return dict(row)
    return dict(rows[0])


def _diag(row: Dict[str, Any], key: str) -> Dict[str, Any]:
    return dict(row.get(key, {}) or {})


def _row_for(root: Path, profile: str, dataset: str, variant: str) -> Dict[str, Any]:
    run_dir = root / profile / variant / dataset
    summary = dict(_read_json(run_dir / "rag_summary.json", {}) or {})
    traces = _read_jsonl(run_dir / "phase7_query_trace.jsonl")
    oracle = _oracle_row(run_dir)

    candidate_recalls = [_safe_float(t.get("candidate_sf_recall_eval_only", 0.0), 0.0) for t in traces]
    selected_recalls = [_safe_float(t.get("selected_sf_recall_eval_only", 0.0), 0.0) for t in traces]
    rendered_recalls = [_safe_float(t.get("rendered_sf_recall_eval_only", 0.0), 0.0) for t in traces]
    feasibility = [_diag(t, "candidate_chain_feasibility") for t in traces]
    unit_diag = [_diag(t, "chain_unit_diagnostics") for t in traces]
    unit_gold = [_diag(t, "chain_unit_gold_eval_only") for t in traces]
    stage_rows = [_diag(t, "stage_ms") for t in traces]

    row = {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "run_dir": str(run_dir),
        "num_queries": int(len(traces)),
        "EM": _summary_value(summary, "EM", "em"),
        "F1": _summary_value(summary, "F1", "f1"),
        "SF_R": _summary_value(summary, "sf_recall", "SF_R", "supporting_fact_recall"),
        "SF_P": _summary_value(summary, "sf_precision", "SF_P", "supporting_fact_precision"),
        "SF_F1": _summary_value(summary, "sf_f1", "SF_F1", "supporting_fact_f1"),
        "avg_context_tokens": _summary_value(summary, "avg_context_tokens", "context_tokens_avg", "prompt_tokens_avg"),
        "avg_selected_atoms": _mean(t.get("num_selected_atoms", 0.0) for t in traces),
        "retrieval_ms": _summary_value(summary, "retrieval_ms", "retrieval_latency_ms"),
        "generation_ms": _summary_value(summary, "generation_ms"),
        "total_ms": _summary_value(summary, "total_ms", "total_latency_ms"),
        "candidate_gold_partial": _bool_mean(f.get("candidate_gold_partial", False) for f in feasibility),
        "candidate_gold_full": _bool_mean(f.get("candidate_gold_full", False) for f in feasibility),
        "candidate_gold_recall": _mean(f.get("candidate_gold_recall", 0.0) for f in feasibility),
        "candidate_graph_gold_connected": _bool_mean(f.get("candidate_graph_gold_connected", False) for f in feasibility),
        "minimal_gold_chain_atoms": _mean(f.get("minimal_gold_chain_atoms", 0.0) for f in feasibility),
        "minimal_gold_chain_tokens": _mean(f.get("minimal_gold_chain_tokens", 0.0) for f in feasibility),
        "budget_feasible_gold_chain": _bool_mean(f.get("budget_feasible_gold_chain", False) for f in feasibility),
        "chain_unit_oracle_feasible": _bool_mean(f.get("chain_unit_oracle_feasible", False) for f in feasibility),
        "carrier_feasible_but_atom_failed": _bool_mean(f.get("carrier_feasible_but_atom_failed", False) for f in feasibility),
        "selected_partial_gold_hit": _mean(1.0 if r > 0.0 else 0.0 for r in selected_recalls),
        "selected_full_gold_coverage": _mean(1.0 if r >= 0.999999 else 0.0 for r in selected_recalls),
        "selected_gold_recall": _mean(selected_recalls),
        "rendered_full_gold_coverage": _mean(1.0 if r >= 0.999999 else 0.0 for r in rendered_recalls),
        "selected_context_F1": _safe_float(oracle.get("selected_context_F1", 0.0), 0.0),
        "gold_support_context_F1": _safe_float(oracle.get("gold_support_F1", 0.0), 0.0),
        "selected_plus_gold_F1": _safe_float(oracle.get("selected_plus_gold_F1", 0.0), 0.0),
        "num_single_units": _mean(d.get("num_single_units", 0.0) for d in unit_diag),
        "num_pair_units": _mean(d.get("num_pair_units", 0.0) for d in unit_diag),
        "num_explicit_transition_units": _mean(d.get("num_explicit_transition_units", 0.0) for d in unit_diag),
        "num_same_title_units": _mean(d.get("num_same_title_units", 0.0) for d in unit_diag),
        "num_same_carrier_units": _mean(d.get("num_same_carrier_units", 0.0) for d in unit_diag),
        "num_selected_units": _mean(d.get("num_selected_units", 0.0) for d in unit_diag),
        "selected_pair_unit_rate": _mean(t.get("selected_pair_unit_rate", 0.0) for t in traces),
        "selected_explicit_transition_unit_rate": _mean(t.get("selected_explicit_transition_unit_rate", 0.0) for t in traces),
        "selected_same_title_unit_rate": _mean(t.get("selected_same_title_unit_rate", 0.0) for t in traces),
        "selected_same_carrier_unit_rate": _mean(t.get("selected_same_carrier_unit_rate", 0.0) for t in traces),
        "gold_unit_partial": _bool_mean(d.get("gold_unit_partial", False) for d in unit_gold),
        "gold_unit_full": _bool_mean(d.get("gold_unit_full", False) for d in unit_gold),
        "gold_units_selected": _mean(d.get("gold_units_selected", 0.0) for d in unit_gold),
        "source_balanced_union_ms": _mean(s.get("source_balanced_union", 0.0) for s in stage_rows),
        "chain_unit_build_ms": _mean(s.get("chain_unit_build", 0.0) for s in stage_rows),
        "chain_unit_selection_ms": _mean(s.get("chain_unit_selection", 0.0) for s in stage_rows),
        "local_graph_ms": _mean(s.get("local_graph_build", 0.0) for s in stage_rows),
        "corridor_ms": _mean(s.get("corridor_extraction", 0.0) for s in stage_rows),
    }
    row["phase1_partial_gold_hit"] = _mean(1.0 if r > 0.0 else 0.0 for r in candidate_recalls)
    row["phase1_full_gold_coverage"] = _mean(1.0 if r >= 0.999999 else 0.0 for r in candidate_recalls)
    row["phase1_gold_recall"] = _mean(candidate_recalls)
    row["oracle_gap"] = float(row["gold_support_context_F1"] - row["selected_context_F1"])
    return row


def _aggregate(rows: List[Dict[str, Any]], variant: str) -> Dict[str, float]:
    subset = [r for r in rows if r["variant"] == variant and int(r.get("num_queries", 0)) > 0]
    keys = [
        "F1",
        "SF_P",
        "retrieval_ms",
        "candidate_gold_full",
        "candidate_gold_recall",
        "budget_feasible_gold_chain",
        "chain_unit_oracle_feasible",
        "selected_full_gold_coverage",
        "selected_gold_recall",
        "selected_context_F1",
        "oracle_gap",
        "chain_unit_build_ms",
        "chain_unit_selection_ms",
    ]
    return {key: _mean(r.get(key, 0.0) for r in subset) for key in keys}


def _report(rows: List[Dict[str, Any]]) -> str:
    sb128 = _aggregate(rows, "source_balanced_128")
    sb160 = _aggregate(rows, "source_balanced_160")
    chain = _aggregate(rows, "chain_unit_selection")

    feasible_avg = chain["chain_unit_oracle_feasible"]
    sb160_feasible_delta = sb160["chain_unit_oracle_feasible"] - sb128["chain_unit_oracle_feasible"]
    selected_full_delta = chain["selected_full_gold_coverage"] - sb128["selected_full_gold_coverage"]
    selected_context_delta = chain["selected_context_F1"] - sb128["selected_context_F1"]
    gap_delta = chain["oracle_gap"] - sb128["oracle_gap"]
    f1_delta = chain["F1"] - sb128["F1"]
    sfp_delta = chain["SF_P"] - sb128["SF_P"]
    retrieval_delta_pct = (
        (chain["retrieval_ms"] - sb128["retrieval_ms"]) / sb128["retrieval_ms"]
        if sb128["retrieval_ms"] > 0.0
        else 0.0
    )
    chain_latency = chain["chain_unit_build_ms"] + chain["chain_unit_selection_ms"]

    if feasible_avg < 0.20:
        proposal_issue = "candidate pool/unit graph oracle feasibility is low; proposal/index coverage remains the bottleneck."
    elif chain["selected_full_gold_coverage"] < feasible_avg * 0.5:
        proposal_issue = "oracle-feasible chains exist more often than selected full chains; the selection objective is still weak."
    else:
        proposal_issue = "chain-unit selection is preserving a meaningful share of feasible chains."

    should_mainline = (
        (selected_full_delta > 0.0 or selected_context_delta > 0.0 or gap_delta < 0.0 or f1_delta > 0.0)
        and sfp_delta >= -0.05
        and retrieval_delta_pct <= 0.30
    )

    return "\n".join(
        [
            "# Phase7 Chain-Unit Selection Report",
            "",
            f"1. Is the current candidate pool oracle-feasible? {'yes' if feasible_avg >= 0.20 else 'no'} (chain_unit_oracle_feasible={feasible_avg:.4f})",
            f"2. Does source_balanced_160 improve feasibility over 128? {'yes' if sb160_feasible_delta > 1.0e-9 else 'no'} ({sb160_feasible_delta:+.4f})",
            f"3. Does chain_unit_selection improve selected full-chain coverage? {'yes' if selected_full_delta > 1.0e-9 else 'no'} ({selected_full_delta:+.4f} vs source_balanced_128)",
            f"4. Does chain_unit_selection improve selected_context_F1? {'yes' if selected_context_delta > 1.0e-9 else 'no'} ({selected_context_delta:+.4f})",
            f"5. Does chain_unit_selection reduce oracle_gap? {'yes' if gap_delta < -1.0e-9 else 'no'} ({gap_delta:+.4f})",
            f"6. Does final F1 improve? {'yes' if f1_delta > 1.0e-9 else 'no'} ({f1_delta:+.4f})",
            f"7. Does SF-P collapse? {'yes' if sfp_delta < -0.05 else 'no'} ({sfp_delta:+.4f})",
            f"8. Does chain-unit selection add acceptable latency? {'yes' if retrieval_delta_pct <= 0.30 else 'no'} ({retrieval_delta_pct:+.2%}, chain_unit_ms={chain_latency:.2f})",
            f"9. If chain_unit_oracle_feasible is low, what proposal/index issue remains? {proposal_issue}",
            f"10. If feasible is high but selected_full is low, what selection issue remains? {proposal_issue}",
            f"11. Should chain-unit selection become the next Phase7 mainline? {'yes' if should_mainline else 'no'}",
            "",
            "Aggregate comparison is averaged over available dataset/profile cells. Missing runs are ignored in aggregate deltas.",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/phase7_evidence_flow/chain_unit_selection_experiment")
    args = ap.parse_args()
    root = Path(args.root)
    rows = [_row_for(root, profile, dataset, variant) for profile in PROFILES for dataset in DATASETS for variant in VARIANTS]

    md = [
        "# Phase7 Chain-Unit Selection Summary",
        "",
        "## Main Table",
        _table(
            ["dataset", "profile", "variant", "EM", "F1", "SF-R", "SF-P", "SF-F1", "avg_tokens", "avg_atoms", "retrieval_ms", "total_ms"],
            [
                [
                    r["dataset"],
                    r["profile"],
                    r["variant"],
                    _fmt4(r["EM"]),
                    _fmt4(r["F1"]),
                    _fmt4(r["SF_R"]),
                    _fmt4(r["SF_P"]),
                    _fmt4(r["SF_F1"]),
                    _fmt2(r["avg_context_tokens"]),
                    _fmt2(r["avg_selected_atoms"]),
                    _fmt2(r["retrieval_ms"]),
                    _fmt2(r["total_ms"]),
                ]
                for r in rows
            ],
        ),
        "",
        "## Feasibility Table",
        _table(
            ["dataset", "profile", "variant", "cand_partial", "cand_full", "cand_recall", "graph_connected", "min_atoms", "min_tokens", "budget_feasible", "unit_oracle", "carrier_gap"],
            [
                [
                    r["dataset"],
                    r["profile"],
                    r["variant"],
                    _fmt4(r["candidate_gold_partial"]),
                    _fmt4(r["candidate_gold_full"]),
                    _fmt4(r["candidate_gold_recall"]),
                    _fmt4(r["candidate_graph_gold_connected"]),
                    _fmt2(r["minimal_gold_chain_atoms"]),
                    _fmt2(r["minimal_gold_chain_tokens"]),
                    _fmt4(r["budget_feasible_gold_chain"]),
                    _fmt4(r["chain_unit_oracle_feasible"]),
                    _fmt4(r["carrier_feasible_but_atom_failed"]),
                ]
                for r in rows
            ],
        ),
        "",
        "## Selected Table",
        _table(
            ["dataset", "profile", "variant", "selected_full", "selected_recall", "rendered_full", "selected_F1", "gold_oracle_F1", "selected_plus_gold_F1", "oracle_gap"],
            [
                [
                    r["dataset"],
                    r["profile"],
                    r["variant"],
                    _fmt4(r["selected_full_gold_coverage"]),
                    _fmt4(r["selected_gold_recall"]),
                    _fmt4(r["rendered_full_gold_coverage"]),
                    _fmt4(r["selected_context_F1"]),
                    _fmt4(r["gold_support_context_F1"]),
                    _fmt4(r["selected_plus_gold_F1"]),
                    _fmt4(r["oracle_gap"]),
                ]
                for r in rows
            ],
        ),
        "",
        "## Chain-Unit Table",
        _table(
            ["dataset", "profile", "variant", "single_units", "pair_units", "explicit_units", "same_title", "same_carrier", "selected_units", "pair_sel", "explicit_sel", "gold_unit_partial", "gold_unit_full", "gold_units_selected", "chain_unit_ms"],
            [
                [
                    r["dataset"],
                    r["profile"],
                    r["variant"],
                    _fmt2(r["num_single_units"]),
                    _fmt2(r["num_pair_units"]),
                    _fmt2(r["num_explicit_transition_units"]),
                    _fmt2(r["num_same_title_units"]),
                    _fmt2(r["num_same_carrier_units"]),
                    _fmt2(r["num_selected_units"]),
                    _fmt4(r["selected_pair_unit_rate"]),
                    _fmt4(r["selected_explicit_transition_unit_rate"]),
                    _fmt4(r["gold_unit_partial"]),
                    _fmt4(r["gold_unit_full"]),
                    _fmt2(r["gold_units_selected"]),
                    _fmt2(r["chain_unit_build_ms"] + r["chain_unit_selection_ms"]),
                ]
                for r in rows
            ],
        ),
        "",
        "## Timing Table",
        _table(
            ["dataset", "profile", "variant", "source_union_ms", "chain_build_ms", "chain_select_ms", "local_graph_ms", "corridor_ms", "retrieval_ms"],
            [
                [
                    r["dataset"],
                    r["profile"],
                    r["variant"],
                    _fmt2(r["source_balanced_union_ms"]),
                    _fmt2(r["chain_unit_build_ms"]),
                    _fmt2(r["chain_unit_selection_ms"]),
                    _fmt2(r["local_graph_ms"]),
                    _fmt2(r["corridor_ms"]),
                    _fmt2(r["retrieval_ms"]),
                ]
                for r in rows
            ],
        ),
        "",
    ]

    root.mkdir(parents=True, exist_ok=True)
    summary_md = root / "chain_unit_selection_summary.md"
    summary_json = root / "chain_unit_selection_summary.json"
    report_md = root / "phase7_chain_unit_selection_report.md"
    summary_md.write_text("\n".join(md), encoding="utf-8")
    summary_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_report(rows), encoding="utf-8")
    print(summary_md)
    print(summary_json)
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
