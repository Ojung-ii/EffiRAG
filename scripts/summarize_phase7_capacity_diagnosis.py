#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


def _safe_int(v: Any, d: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(d)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = str(line or "").strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def _sf_f1(r: float, p: float) -> float:
    rr = _safe_float(r, 0.0)
    pp = _safe_float(p, 0.0)
    if (rr + pp) <= 0.0:
        return 0.0
    return float(2.0 * rr * pp / (rr + pp))


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "null"
    return f"{_safe_float(v, 0.0):.{nd}f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    h = "| " + " | ".join(headers) + " |"
    s = "| " + " | ".join(["---"] * len(headers)) + " |"
    b = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([h, s] + b)


def _profile_caps(profile: str) -> Dict[str, int]:
    if profile == "balanced_384_8":
        return {"tokens": 384, "atoms": 8}
    if profile == "legacy_512_10":
        return {"tokens": 512, "atoms": 10}
    if profile == "capacity_512_16":
        return {"tokens": 512, "atoms": 16}
    if profile == "capacity_768_16":
        return {"tokens": 768, "atoms": 16}
    return {"tokens": 0, "atoms": 0}


def _oracle_row(run_dir: Path, dataset: str) -> Dict[str, Any]:
    payload = _read_json(run_dir / "oracle_context_results.json")
    rows = list(payload.get("summary_rows", []) or [])
    for row in rows:
        if str(row.get("dataset", "")) == str(dataset) and str(row.get("prompt_mode", "")) == "current_phase7":
            return dict(row)
    return {}


def _coverage_rates(oracle_gap_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not oracle_gap_rows:
        return {
            "phase1_full_gold_coverage": 0.0,
            "selected_full_gold_coverage": 0.0,
            "rendered_full_gold_coverage": 0.0,
        }
    p1 = 0
    sel = 0
    ren = 0
    n = 0
    for row in oracle_gap_rows:
        gold = dict(row.get("gold_survival_eval_only", {}) or {})
        gtot = _safe_int(gold.get("gold_support_count", 0), 0)
        if gtot <= 0:
            continue
        n += 1
        if _safe_int(gold.get("gold_in_phase1", 0), 0) >= gtot:
            p1 += 1
        if _safe_int(gold.get("gold_selected", 0), 0) >= gtot:
            sel += 1
        if _safe_int(gold.get("gold_rendered", 0), 0) >= gtot:
            ren += 1
    if n <= 0:
        return {
            "phase1_full_gold_coverage": 0.0,
            "selected_full_gold_coverage": 0.0,
            "rendered_full_gold_coverage": 0.0,
        }
    return {
        "phase1_full_gold_coverage": float(p1 / float(n)),
        "selected_full_gold_coverage": float(sel / float(n)),
        "rendered_full_gold_coverage": float(ren / float(n)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Phase7 capacity diagnosis outputs.")
    ap.add_argument("--root", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Missing root: {root}")

    rows: List[Dict[str, Any]] = []
    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        dataset = dataset_dir.name
        for profile_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            profile = profile_dir.name
            rag_summary = _read_json(profile_dir / "rag_summary.json")
            if not rag_summary:
                continue
            query_rows = _read_jsonl(profile_dir / "phase7_query_trace.jsonl")
            oracle_gap_rows = _read_jsonl(profile_dir / "phase7_oracle_gap_trace.jsonl")
            oracle_row = _oracle_row(profile_dir, dataset=dataset)
            caps = _profile_caps(profile)
            cov = _coverage_rates(oracle_gap_rows)

            token_cap = int(caps.get("tokens", 0))
            atom_cap = int(caps.get("atoms", 0))
            atom_exhausted = 0
            token_exhausted = 0
            bts_count = 0
            avg_sel_atoms: List[float] = []
            avg_sel_tokens: List[float] = []
            avg_rem_tokens: List[float] = []
            avg_rem_atoms: List[float] = []
            for q in query_rows:
                sel_atoms = _safe_int(q.get("num_selected_atoms", 0), 0)
                sel_tokens = _safe_int(q.get("selected_token_count", q.get("rendered_context_tokens", 0)), 0)
                bf = _safe_int(q.get("budget_filtered_candidates_eval_only", 0), 0)
                if atom_cap > 0 and sel_atoms >= atom_cap:
                    atom_exhausted += 1
                if bf > 0:
                    token_exhausted += 1
                if str(q.get("empty_candidate_reason", "")) == "ALL_CANDIDATES_FILTERED_BY_TOKEN_BUDGET" or bf > 0:
                    bts_count += 1
                avg_sel_atoms.append(float(sel_atoms))
                avg_sel_tokens.append(float(sel_tokens))
                if token_cap > 0:
                    avg_rem_tokens.append(float(max(0, token_cap - sel_tokens)))
                if atom_cap > 0:
                    avg_rem_atoms.append(float(max(0, atom_cap - sel_atoms)))

            n = max(1, len(query_rows))
            sf_r = _safe_float(rag_summary.get("supporting_fact_recall", 0.0), 0.0)
            sf_p = _safe_float(rag_summary.get("supporting_fact_precision", 0.0), 0.0)
            row = {
                "dataset": dataset,
                "profile": profile,
                "n": len(query_rows),
                "EM": _safe_float(rag_summary.get("em", 0.0), 0.0),
                "F1": _safe_float(rag_summary.get("f1", 0.0), 0.0),
                "SF_R": sf_r,
                "SF_P": sf_p,
                "SF_F1": _sf_f1(sf_r, sf_p),
                "avg_context_tokens": _safe_float(rag_summary.get("prompt_tokens_avg", 0.0), 0.0),
                "F1_per_1k": (
                    float(_safe_float(rag_summary.get("f1", 0.0), 0.0) / max(1.0, _safe_float(rag_summary.get("prompt_tokens_avg", 0.0), 0.0)) * 1000.0)
                    if _safe_float(rag_summary.get("prompt_tokens_avg", 0.0), 0.0) > 0
                    else 0.0
                ),
                "retrieval_ms": _safe_float(rag_summary.get("retrieval_latency_ms", 0.0), 0.0),
                "generation_ms": _safe_float(rag_summary.get("total_latency_ms", 0.0), 0.0)
                - _safe_float(rag_summary.get("retrieval_latency_ms", 0.0), 0.0),
                "total_ms": _safe_float(rag_summary.get("total_latency_ms", 0.0), 0.0),
                "phase1_full_gold_coverage": cov["phase1_full_gold_coverage"],
                "selected_full_gold_coverage": cov["selected_full_gold_coverage"],
                "rendered_full_gold_coverage": cov["rendered_full_gold_coverage"],
                "selected_context_F1": (
                    _safe_float(oracle_row.get("selected_context_F1"), 0.0) if oracle_row else None
                ),
                "gold_support_context_F1": (
                    _safe_float(oracle_row.get("gold_support_F1"), 0.0) if oracle_row else None
                ),
                "selected_plus_gold_F1": (
                    _safe_float(oracle_row.get("selected_plus_gold_F1"), 0.0) if oracle_row else None
                ),
                "oracle_gap": (
                    _safe_float(oracle_row.get("gold_support_F1"), 0.0)
                    - _safe_float(oracle_row.get("selected_context_F1"), 0.0)
                    if oracle_row
                    else None
                ),
                "budget_too_small_count": int(bts_count),
                "atom_cap_exhausted_rate": float(atom_exhausted / float(n)),
                "token_cap_exhausted_rate": float(token_exhausted / float(n)),
                "avg_selected_atoms": _mean(avg_sel_atoms),
                "avg_selected_tokens": _mean(avg_sel_tokens),
                "avg_remaining_token_budget": _mean(avg_rem_tokens),
                "avg_remaining_atom_budget": _mean(avg_rem_atoms),
                "run_dir": str(profile_dir.resolve()),
            }
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No capacity diagnosis runs found under: {root}")

    main_rows: List[List[str]] = []
    oracle_rows: List[List[str]] = []
    cap_rows: List[List[str]] = []
    for r in sorted(rows, key=lambda x: (x["dataset"], x["profile"])):
        main_rows.append(
            [
                str(r["dataset"]),
                str(r["profile"]),
                _fmt(r["EM"], 4),
                _fmt(r["F1"], 4),
                _fmt(r["SF_R"], 4),
                _fmt(r["SF_P"], 4),
                _fmt(r["avg_context_tokens"], 2),
                _fmt(r["avg_selected_atoms"], 2),
                _fmt(r["retrieval_ms"], 2),
                _fmt(r["total_ms"], 2),
            ]
        )
        oracle_rows.append(
            [
                str(r["dataset"]),
                str(r["profile"]),
                _fmt(r["selected_context_F1"], 4) if r["selected_context_F1"] is not None else "null",
                _fmt(r["gold_support_context_F1"], 4) if r["gold_support_context_F1"] is not None else "null",
                _fmt(r["selected_plus_gold_F1"], 4) if r["selected_plus_gold_F1"] is not None else "null",
                _fmt(r["oracle_gap"], 4) if r["oracle_gap"] is not None else "null",
            ]
        )
        cap_rows.append(
            [
                str(r["dataset"]),
                str(r["profile"]),
                _fmt(r["atom_cap_exhausted_rate"], 4),
                _fmt(r["token_cap_exhausted_rate"], 4),
                _fmt(r["avg_remaining_token_budget"], 2),
                str(int(r["budget_too_small_count"])),
            ]
        )

    md: List[str] = []
    md.append("# Phase7 Capacity Diagnosis Report")
    md.append("")
    md.append("## Main Table")
    md.append(
        _table(
            ["dataset", "profile", "EM", "F1", "SF-R", "SF-P", "avg_tokens", "avg_atoms", "retrieval_ms", "total_ms"],
            main_rows,
        )
    )
    md.append("")
    md.append("## Oracle Gap Table")
    md.append(
        _table(
            ["dataset", "profile", "selected_F1", "gold_oracle_F1", "selected_plus_gold_F1", "oracle_gap"],
            oracle_rows,
        )
    )
    md.append("")
    md.append("## Capacity Table")
    md.append(
        _table(
            ["dataset", "profile", "atom_cap_exhausted", "token_cap_exhausted", "avg_remaining_tokens", "budget_too_small_count"],
            cap_rows,
        )
    )
    md.append("")
    md.append("## Interpretation")
    md.append("- `atom_cap_exhausted`가 높고 `token_cap_exhausted`가 낮으면 atom cap 제약이 주요 원인입니다.")
    md.append("- `token_cap_exhausted`가 높고 `avg_remaining_tokens`가 낮으면 token cap 제약이 주요 원인입니다.")
    md.append("- capacity 확장 시 `oracle_gap`이 줄고 `SF-P`가 급락하지 않으면 확장 프로파일이 유망합니다.")

    atom_exhaust_mean = _mean([_safe_float(r["atom_cap_exhausted_rate"], 0.0) for r in rows])
    token_exhaust_mean = _mean([_safe_float(r["token_cap_exhausted_rate"], 0.0) for r in rows])
    by_profile: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_profile.setdefault(str(r["profile"]), []).append(r)
    profile_scores: Dict[str, float] = {}
    for p, items in by_profile.items():
        mean_f1 = _mean([_safe_float(x.get("F1", 0.0), 0.0) for x in items])
        mean_gap = _mean([_safe_float(x.get("oracle_gap", 0.0), 0.0) for x in items if x.get("oracle_gap") is not None])
        mean_sf_p = _mean([_safe_float(x.get("SF_P", 0.0), 0.0) for x in items])
        # lower gap is better -> subtract
        profile_scores[p] = float(mean_f1 + (0.2 * mean_sf_p) - (0.5 * mean_gap))
    recommended_profile = sorted(profile_scores.items(), key=lambda x: x[1], reverse=True)[0][0] if profile_scores else "unknown"

    md.append("")
    md.append("## Explicit Answers")
    md.append(f"- Is BUDGET_TOO_SMALL mainly token-cap limited? {'yes' if token_exhaust_mean > atom_exhaust_mean else 'partially/no'}")
    md.append(f"- Is BUDGET_TOO_SMALL mainly atom-cap limited? {'yes' if atom_exhaust_mean >= token_exhaust_mean else 'partially/no'}")
    md.append("- Does increasing capacity reduce oracle gap? check `Oracle gap table` profile deltas.")
    md.append("- Does increasing capacity increase noise and reduce precision? check `SF-P` in main table.")
    md.append(f"- Recommended next main profile: `{recommended_profile}` (heuristic aggregate over F1/SF-P/oracle-gap).")

    report_path = root / "phase7_capacity_diagnosis_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    json_path = root / "phase7_capacity_diagnosis_summary.json"
    json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
