#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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
            t = _strip(line)
            if not t:
                continue
            rows.append(dict(json.loads(t) or {}))
    return rows


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        rr = csv.DictReader(f, delimiter="\t")
        for row in rr:
            rows.append({str(k): str(v or "") for k, v in dict(row).items()})
    return rows


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _render_md(report: Mapping[str, Any]) -> str:
    checks = list(report.get("checks", []) or [])
    warnings = list(report.get("warnings", []) or [])
    errors = list(report.get("errors", []) or [])
    counts = dict(report.get("counts", {}) or {})

    lines: List[str] = [
        "# Qwen32B Judge Output Validation Report",
        "",
        f"- generated_at_utc: {report.get('generated_at_utc')}",
        f"- round_dir: {report.get('round_dir')}",
        f"- pass: {report.get('pass')}",
        "",
        "## Counts",
        "```json",
        json.dumps(counts, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Checks",
        "| name | pass | details |",
        "| --- | --- | --- |",
    ]
    for c in checks:
        lines.append(f"| {c.get('name')} | {c.get('pass')} | {c.get('details')} |")
    lines.extend(["", "## Warnings"])
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- none")
    lines.extend(["", "## Errors"])
    if errors:
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _add_check(name: str, cond: bool, details: str, checks: List[Dict[str, Any]], errors: List[str]) -> None:
    checks.append({"name": name, "pass": bool(cond), "details": details})
    if not cond:
        errors.append(f"{name}: {details}")


def _expected_from_input(input_root: Path) -> Tuple[Dict[str, Set[int]], Dict[str, str], Dict[str, Dict[str, str]]]:
    evidence_ids: Dict[str, Set[int]] = {}
    chunk_of: Dict[str, str] = {}
    meta_of: Dict[str, Dict[str, str]] = {}
    for fp in sorted(input_root.glob("*.jsonl")):
        if fp.name.endswith("_judged.jsonl"):
            continue
        for row in _read_jsonl(fp):
            jid = _strip(row.get("judge_id"))
            if not jid:
                continue
            ev_ids = {
                int(x.get("id"))
                for x in list(row.get("evidence", []) or [])
                if str(x.get("id", "")).isdigit()
            }
            evidence_ids[jid] = ev_ids
            chunk_of[jid] = fp.name
            meta_of[jid] = {
                "dataset": _strip(row.get("dataset")),
                "qid": _strip(row.get("qid")),
                "system_id": _strip(row.get("system_id")),
            }
    return evidence_ids, chunk_of, meta_of


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Qwen32B judged outputs.")
    ap.add_argument("--round-dir", required=True)
    args = ap.parse_args()

    round_dir = Path(args.round_dir).resolve()
    manifest_path = round_dir / "manifest.json"
    run_records_path = round_dir / "run_records.tsv"
    expected_ids_path = round_dir / "expected_judge_ids.jsonl"
    judge_outputs_dir = round_dir / "judge_outputs"
    logs_dir = round_dir / "logs"
    validation_dir = round_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = dict(_read_json(manifest_path) or {})
    input_root = Path(_strip(manifest.get("input_root"))).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input_root from manifest not found: {input_root}")

    evidence_ids, chunk_of, meta_of = _expected_from_input(input_root)
    if expected_ids_path.exists():
        rows = _read_jsonl(expected_ids_path)
        expected_ids = {_strip(r.get("judge_id")) for r in rows if _strip(r.get("judge_id"))}
    else:
        expected_ids = set(evidence_ids.keys())

    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []
    errors: List[str] = []

    output_records: List[Dict[str, Any]] = []
    output_by_id: Dict[str, Dict[str, Any]] = {}
    duplicate_ids = 0
    parse_errors = 0
    for fp in sorted(judge_outputs_dir.glob("*_judged.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                t = _strip(line)
                if not t:
                    continue
                try:
                    row = dict(json.loads(t) or {})
                except Exception:
                    parse_errors += 1
                    continue
                output_records.append(row)
                jid = _strip(row.get("judge_id"))
                if jid in output_by_id:
                    duplicate_ids += 1
                output_by_id[jid] = row

    _add_check("output_json_parse", parse_errors == 0, f"parse_errors={parse_errors}", checks, errors)
    _add_check("output_judge_id_unique", duplicate_ids == 0, f"duplicate_ids={duplicate_ids}", checks, errors)

    expected_count = len(expected_ids)
    output_count = sum(1 for jid in output_by_id.keys() if jid in expected_ids)
    _add_check(
        "input_output_count_match",
        expected_count == output_count,
        f"expected={expected_count}, output={output_count}",
        checks,
        errors,
    )

    missing_ids = sorted(expected_ids - set(output_by_id.keys()))
    _add_check(
        "all_input_judge_ids_in_output",
        len(missing_ids) == 0,
        f"missing={len(missing_ids)}",
        checks,
        errors,
    )

    score_invalid = 0
    sufficient_mismatch = 0
    minimal_subset_violation = 0
    warn_sufficient_empty_minimal = 0
    warn_insufficient_with_minimal = 0
    judge_error_count = 0

    for jid in sorted(expected_ids):
        row = output_by_id.get(jid)
        if row is None:
            continue
        score = row.get("score")
        if not isinstance(score, int):
            try:
                score = int(score)
            except Exception:
                score = None
        if score not in {0, 1, 2, 3}:
            score_invalid += 1
            continue

        sufficient = row.get("sufficient")
        if not _is_bool(sufficient) or bool(sufficient) != bool(score >= 2):
            sufficient_mismatch += 1

        mids = row.get("minimal_evidence_ids", [])
        if mids is None:
            mids = []
        if not isinstance(mids, list):
            minimal_subset_violation += 1
            continue
        valid = evidence_ids.get(jid, set())
        for m in mids:
            try:
                mm = int(m)
            except Exception:
                minimal_subset_violation += 1
                continue
            if mm not in valid:
                minimal_subset_violation += 1

        if score >= 2 and len(mids) == 0:
            warn_sufficient_empty_minimal += 1
        if score < 2 and len(mids) > 0:
            warn_insufficient_with_minimal += 1

        if _strip(row.get("error")):
            judge_error_count += 1

    _add_check("score_enum_0_1_2_3", score_invalid == 0, f"invalid={score_invalid}", checks, errors)
    _add_check(
        "sufficient_equals_score_ge_2",
        sufficient_mismatch == 0,
        f"mismatch={sufficient_mismatch}",
        checks,
        errors,
    )
    _add_check(
        "minimal_evidence_ids_subset_of_input_evidence_ids",
        minimal_subset_violation == 0,
        f"violations={minimal_subset_violation}",
        checks,
        errors,
    )

    if warn_sufficient_empty_minimal > 0:
        warnings.append(
            f"score>=2 but minimal_evidence_ids empty: {warn_sufficient_empty_minimal}"
        )
    if warn_insufficient_with_minimal > 0:
        warnings.append(
            f"score<2 but minimal_evidence_ids non-empty: {warn_insufficient_with_minimal}"
        )

    # Parse failure / retry rates.
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
    judge_error_rate = float(judge_error_count / expected_count) if expected_count > 0 else 0.0
    parse_failure_rate = float(len(parse_fail_ids) / expected_count) if expected_count > 0 else 0.0
    retry_rate = float(len(retry_ids) / expected_count) if expected_count > 0 else 0.0

    # Chunk-level count checks.
    run_rows = _read_tsv(run_records_path)
    chunk_mismatch = 0
    for rr in run_rows:
        status = _strip(rr.get("status"))
        if status not in {"ok", "skipped_existing"}:
            continue
        chunk_file = _strip(rr.get("chunk_file"))
        out_file = _strip(rr.get("output_file"))
        expected_n = _safe_int(rr.get("expected_output_count"), 0)
        if not out_file:
            out_file = Path(chunk_file).stem + "_judged.jsonl"
        actual_n = len(_read_jsonl(judge_outputs_dir / out_file))
        if actual_n != expected_n:
            chunk_mismatch += 1
            warnings.append(
                f"chunk count mismatch: {chunk_file} expected={expected_n} actual={actual_n}"
            )
    _add_check(
        "chunk_input_output_count_match",
        chunk_mismatch == 0,
        f"chunk_mismatch={chunk_mismatch}",
        checks,
        errors,
    )

    report = {
        "generated_at_utc": _utc_now_iso(),
        "round_dir": str(round_dir),
        "pass": len(errors) == 0,
        "counts": {
            "expected_records": expected_count,
            "output_records": output_count,
            "missing_records": len(missing_ids),
            "judge_error_count": judge_error_count,
            "judge_error_rate": judge_error_rate,
            "parse_failure_record_count": len(parse_fail_ids),
            "parse_failure_rate": parse_failure_rate,
            "retry_record_count": len(retry_ids),
            "retry_rate": retry_rate,
        },
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "missing_judge_ids": missing_ids[:200],
    }

    _write_json(validation_dir / "judge_output_validation_report.json", report)
    (validation_dir / "judge_output_validation_report.md").write_text(_render_md(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "pass": report["pass"],
                "errors": len(errors),
                "warnings": len(warnings),
                "parse_failure_rate": parse_failure_rate,
                "retry_rate": retry_rate,
            },
            ensure_ascii=False,
        )
    )
    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
