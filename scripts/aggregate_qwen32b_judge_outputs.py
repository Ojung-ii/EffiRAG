#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = _strip(line)
            if not text:
                continue
            rows.append(dict(json.loads(text) or {}))
    return rows


def _mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _group_metrics(
    rows: Sequence[Mapping[str, Any]],
    parse_fail_ids: Set[str],
    retry_ids: Set[str],
) -> Dict[str, Any]:
    n = len(rows)
    if n <= 0:
        return {
            "n": 0,
            "avg_score": 0.0,
            "llm_sufficient_rate": 0.0,
            "llm_minimal_sufficient_rate": 0.0,
            "answer_surface_present_rate": 0.0,
            "equivalent_evidence_rate": 0.0,
            "avg_minimal_evidence_count": 0.0,
            "judge_error_rate": 0.0,
            "parse_failure_rate": 0.0,
            "retry_rate": 0.0,
        }

    scores = [_safe_float(r.get("score"), 0.0) for r in rows]
    sufficient = [1.0 if _safe_float(r.get("score"), -1) >= 2 else 0.0 for r in rows]
    minimal = [1.0 if _safe_float(r.get("score"), -1) == 3 else 0.0 for r in rows]
    answer_surface = [1.0 if bool(r.get("answer_surface_present")) else 0.0 for r in rows]
    equivalent = [1.0 if bool(r.get("equivalent_evidence")) else 0.0 for r in rows]
    error_flags = [1.0 if _strip(r.get("error")) else 0.0 for r in rows]

    sufficient_rows = [r for r in rows if _safe_float(r.get("score"), -1) >= 2]
    minimal_counts = [
        float(len(list(r.get("minimal_evidence_ids", []) or [])))
        for r in sufficient_rows
        if isinstance(r.get("minimal_evidence_ids"), list)
    ]

    ids = {_strip(r.get("judge_id")) for r in rows if _strip(r.get("judge_id"))}
    parse_rate = float(len(ids.intersection(parse_fail_ids)) / len(ids)) if ids else 0.0
    retry_rate = float(len(ids.intersection(retry_ids)) / len(ids)) if ids else 0.0

    return {
        "n": int(n),
        "avg_score": _mean(scores),
        "llm_sufficient_rate": _mean(sufficient),
        "llm_minimal_sufficient_rate": _mean(minimal),
        "answer_surface_present_rate": _mean(answer_surface),
        "equivalent_evidence_rate": _mean(equivalent),
        "avg_minimal_evidence_count": _mean(minimal_counts),
        "judge_error_rate": _mean(error_flags),
        "parse_failure_rate": parse_rate,
        "retry_rate": retry_rate,
    }


def _render_table(rows: Sequence[Mapping[str, Any]], group_cols: Sequence[str]) -> str:
    header_cols = list(group_cols) + [
        "n",
        "avg_score",
        "llm_sufficient_rate",
        "llm_minimal_sufficient_rate",
        "answer_surface_present_rate",
        "equivalent_evidence_rate",
        "avg_minimal_evidence_count",
        "judge_error_rate",
        "parse_failure_rate",
        "retry_rate",
    ]
    lines: List[str] = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(["---"] * len(header_cols)) + " |",
    ]
    for r in rows:
        vals: List[str] = []
        for c in group_cols:
            vals.append(str(r.get(c, "")))
        vals.extend(
            [
                str(int(r.get("n", 0))),
                f"{_safe_float(r.get('avg_score')):.4f}",
                f"{_safe_float(r.get('llm_sufficient_rate')):.4f}",
                f"{_safe_float(r.get('llm_minimal_sufficient_rate')):.4f}",
                f"{_safe_float(r.get('answer_surface_present_rate')):.4f}",
                f"{_safe_float(r.get('equivalent_evidence_rate')):.4f}",
                f"{_safe_float(r.get('avg_minimal_evidence_count')):.4f}",
                f"{_safe_float(r.get('judge_error_rate')):.4f}",
                f"{_safe_float(r.get('parse_failure_rate')):.4f}",
                f"{_safe_float(r.get('retry_rate')):.4f}",
            ]
        )
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Qwen32B judged outputs into blind/unblinded summaries.")
    ap.add_argument("--round-dir", required=True)
    args = ap.parse_args()

    round_dir = Path(args.round_dir).resolve()
    manifest_path = round_dir / "manifest.json"
    judge_outputs_dir = round_dir / "judge_outputs"
    summary_dir = round_dir / "summary"
    logs_dir = round_dir / "logs"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = dict(_read_json(manifest_path) or {})

    judged_rows: List[Dict[str, Any]] = []
    for fp in sorted(judge_outputs_dir.glob("*_judged.jsonl")):
        judged_rows.extend(_read_jsonl(fp))

    parse_fail_ids = {
        _strip(r.get("judge_id"))
        for r in _read_jsonl(logs_dir / "parse_failures.jsonl")
        if _strip(r.get("judge_id"))
    }
    retry_ids = {
        _strip(r.get("judge_id"))
        for r in _read_jsonl(logs_dir / "retry_records.jsonl")
        if _strip(r.get("judge_id"))
    }

    # Blind summary
    blind_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in judged_rows:
        ds = _strip(row.get("dataset"))
        sid = _strip(row.get("system_id"))
        blind_groups[(ds, sid)].append(row)

    blind_rows: List[Dict[str, Any]] = []
    for (ds, sid), rows in sorted(blind_groups.items()):
        met = _group_metrics(rows, parse_fail_ids=parse_fail_ids, retry_ids=retry_ids)
        out = {"dataset": ds, "system_id": sid}
        out.update(met)
        blind_rows.append(out)

    blind_payload = {
        "generated_at_utc": _utc_now_iso(),
        "round_dir": str(round_dir),
        "group_by": ["dataset", "system_id"],
        "rows": blind_rows,
    }
    _write_json(summary_dir / "llm_judge_summary_blind.json", blind_payload)
    (summary_dir / "llm_judge_summary_blind.md").write_text(
        "# LLM Judge Summary (Blind)\n\n" + _render_table(blind_rows, ["dataset", "system_id"]),
        encoding="utf-8",
    )

    # Unblinded summary
    mapping_root = Path(_strip(manifest.get("mapping_root"))).resolve()
    mapping_path = mapping_root / "record_mapping.jsonl"
    mapping_rows = _read_jsonl(mapping_path) if mapping_path.exists() else []
    map_by_id = {_strip(r.get("judge_id")): dict(r) for r in mapping_rows}
    missing_mapping = 0

    enriched: List[Dict[str, Any]] = []
    for row in judged_rows:
        jid = _strip(row.get("judge_id"))
        mp = map_by_id.get(jid)
        if mp is None:
            missing_mapping += 1
            continue
        rr = dict(row)
        rr["true_system"] = _strip(mp.get("true_system"))
        rr["true_variant"] = _strip(mp.get("true_variant")) or _strip(mp.get("source_system_name"))
        rr["dataset"] = _strip(rr.get("dataset")) or _strip(mp.get("dataset"))
        enriched.append(rr)

    unblind_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        ds = _strip(row.get("dataset"))
        ts = _strip(row.get("true_system"))
        tv = _strip(row.get("true_variant"))
        unblind_groups[(ds, ts, tv)].append(row)

    unblind_rows: List[Dict[str, Any]] = []
    for (ds, ts, tv), rows in sorted(unblind_groups.items()):
        met = _group_metrics(rows, parse_fail_ids=parse_fail_ids, retry_ids=retry_ids)
        out = {"dataset": ds, "true_system": ts, "true_variant": tv}
        out.update(met)
        unblind_rows.append(out)

    unblind_payload = {
        "generated_at_utc": _utc_now_iso(),
        "round_dir": str(round_dir),
        "group_by": ["dataset", "true_system", "true_variant"],
        "missing_mapping_count": int(missing_mapping),
        "rows": unblind_rows,
    }
    _write_json(summary_dir / "llm_judge_summary_unblinded.json", unblind_payload)
    (summary_dir / "llm_judge_summary_unblinded.md").write_text(
        "# LLM Judge Summary (Unblinded)\n\n"
        + _render_table(unblind_rows, ["dataset", "true_system", "true_variant"]),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "blind_rows": len(blind_rows),
                "unblinded_rows": len(unblind_rows),
                "missing_mapping_count": int(missing_mapping),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
