#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip(value: Any) -> str:
    return str(value or "").strip()


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
            line = _strip(line)
            if not line:
                continue
            rows.append(dict(json.loads(line) or {}))
    return rows


def _contains_hidden_system_leak(record: Mapping[str, Any], banned_tokens: Sequence[str]) -> bool:
    blob = json.dumps(dict(record or {}), ensure_ascii=False).lower()
    for tok in banned_tokens:
        if tok and tok.lower() in blob:
            return True
    return False


def _is_blind_label(system_id: str) -> bool:
    return bool(re.match(r"^[A-Z]{1,2}$", _strip(system_id)))


def _check_evidence_ids(evidence: Sequence[Mapping[str, Any]]) -> bool:
    ids = [int(item.get("id", -1)) for item in list(evidence or [])]
    if not ids:
        return False
    return ids == list(range(1, len(ids) + 1))


def _render_markdown(report: Mapping[str, Any]) -> str:
    checks = list(report.get("checks", []) or [])
    warnings = list(report.get("warnings", []) or [])
    errors = list(report.get("errors", []) or [])
    counts = dict(report.get("counts", {}) or {})

    lines: List[str] = [
        "# Input Validation Report",
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
    for chk in checks:
        lines.append(f"| {chk.get('name')} | {chk.get('pass')} | {chk.get('details', '')} |")

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


def _check(name: str, cond: bool, details: str, checks: List[Dict[str, Any]], errors: List[str]) -> None:
    checks.append({"name": name, "pass": bool(cond), "details": details})
    if not cond:
        errors.append(f"{name}: {details}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate generated LLM judge input files.")
    ap.add_argument("--round-dir", required=True, help="outputs/llm_judge_input_round/<timestamp> directory")
    args = ap.parse_args()

    round_dir = Path(args.round_dir).resolve()
    manifest_path = round_dir / "manifest.json"
    judge_inputs_dir = round_dir / "judge_inputs"
    judge_chunks_dir = round_dir / "judge_chunks"
    hidden_mapping_dir = round_dir / "hidden_mapping"
    validation_dir = round_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []
    errors: List[str] = []

    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = dict(_read_json(manifest_path) or {})
    systems = [str(x) for x in list(manifest.get("systems", []) or [])]
    chunk_size = int(manifest.get("chunk_size", 50) or 50)
    datasets = [str(x) for x in list(manifest.get("datasets", []) or [])]

    all_records_path = judge_inputs_dir / "all_records.jsonl"
    mapping_path = hidden_mapping_dir / "record_mapping.jsonl"
    all_records = _read_jsonl(all_records_path)
    mapping_rows = _read_jsonl(mapping_path)

    _check(
        "jsonl_parse_all_records",
        len(all_records) > 0,
        f"records={len(all_records)} from {all_records_path}",
        checks,
        errors,
    )
    _check(
        "jsonl_parse_mapping",
        len(mapping_rows) > 0,
        f"mapping_rows={len(mapping_rows)} from {mapping_path}",
        checks,
        errors,
    )

    judge_ids = [_strip(r.get("judge_id")) for r in all_records]
    unique_ids = set(judge_ids)
    _check(
        "judge_id_unique",
        len(judge_ids) == len(unique_ids),
        f"total={len(judge_ids)} unique={len(unique_ids)}",
        checks,
        errors,
    )

    empty_question = sum(1 for r in all_records if not _strip(r.get("question")))
    empty_gold = sum(1 for r in all_records if not _strip(r.get("gold_answer")))
    empty_evidence = sum(1 for r in all_records if not list(r.get("evidence", []) or []))
    bad_evidence_id = sum(
        1
        for r in all_records
        if not _check_evidence_ids(list(r.get("evidence", []) or []))
    )
    bad_system_id = sum(1 for r in all_records if not _is_blind_label(_strip(r.get("system_id"))))

    _check("question_non_empty", empty_question == 0, f"empty={empty_question}", checks, errors)
    _check("gold_answer_non_empty", empty_gold == 0, f"empty={empty_gold}", checks, errors)
    _check("evidence_non_empty", empty_evidence == 0, f"empty={empty_evidence}", checks, errors)
    _check("evidence_ids_contiguous", bad_evidence_id == 0, f"bad={bad_evidence_id}", checks, errors)
    _check("system_id_blind_label", bad_system_id == 0, f"bad={bad_system_id}", checks, errors)

    banned_tokens = ["effirag", "hipporag2", "hipporag", "light_separator", "copy_span"]
    banned_tokens.extend([s.lower() for s in systems])
    leak_count = sum(1 for r in all_records if _contains_hidden_system_leak(r, banned_tokens))
    _check(
        "no_system_name_leak_in_judge_input",
        leak_count == 0,
        f"leak_count={leak_count}",
        checks,
        errors,
    )

    # Chunk checks
    chunk_files = sorted(judge_chunks_dir.glob("*.jsonl"))
    chunk_total = 0
    oversize_chunks = 0
    for fp in chunk_files:
        rows = _read_jsonl(fp)
        chunk_total += len(rows)
        if len(rows) > chunk_size:
            oversize_chunks += 1
    _check(
        "chunk_files_exist",
        len(chunk_files) > 0,
        f"chunk_files={len(chunk_files)}",
        checks,
        errors,
    )
    _check(
        "chunk_size_respected",
        oversize_chunks == 0,
        f"oversize_chunks={oversize_chunks}, chunk_size={chunk_size}",
        checks,
        errors,
    )

    mapping_ids = [_strip(r.get("judge_id")) for r in mapping_rows]
    _check(
        "mapping_judge_id_match",
        set(mapping_ids) == unique_ids,
        f"mapping={len(set(mapping_ids))}, input={len(unique_ids)}",
        checks,
        errors,
    )

    # Dataset/system balance by true system names from mapping.
    by_dataset_system: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in mapping_rows:
        ds = _strip(row.get("dataset"))
        sys_name = _strip(row.get("source_system_name") or row.get("true_variant") or row.get("true_system"))
        by_dataset_system[ds][sys_name] += 1
    balance_ok = True
    balance_details: List[str] = []
    for ds, cnt_map in by_dataset_system.items():
        vals = sorted(cnt_map.values())
        if not vals:
            continue
        if vals[0] != vals[-1]:
            balance_ok = False
        balance_details.append(f"{ds}:{dict(cnt_map)}")
    _check(
        "dataset_system_count_balance",
        balance_ok,
        "; ".join(balance_details),
        checks,
        errors,
    )

    # Missing and duplicate qid checks.
    missing_qid = sum(1 for r in all_records if not _strip(r.get("qid")))
    if missing_qid > 0:
        warnings.append(f"missing qid count: {missing_qid}")

    dup_qid_issues = 0
    per_ds_variant_seen: Dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in mapping_rows:
        ds = _strip(row.get("dataset"))
        tv = _strip(row.get("true_variant") or row.get("source_system_name"))
        qid = _strip(row.get("qid"))
        key = (ds, tv)
        if qid in per_ds_variant_seen[key]:
            dup_qid_issues += 1
        else:
            per_ds_variant_seen[key].add(qid)
    if dup_qid_issues > 0:
        warnings.append(f"duplicate qid within dataset+variant count: {dup_qid_issues}")

    long_evidence_count = 0
    for rec in all_records:
        evidence = list(rec.get("evidence", []) or [])
        for item in evidence:
            if len(_strip(item.get("text"))) > 1800:
                long_evidence_count += 1
    if long_evidence_count > 0:
        warnings.append(f"very long evidence text count (>1800 chars): {long_evidence_count}")

    # Per-dataset file presence checks
    per_dataset_files_ok = True
    details = []
    for ds in datasets:
        fp = judge_inputs_dir / f"{ds}_records.jsonl"
        ok = fp.exists()
        per_dataset_files_ok = per_dataset_files_ok and ok
        details.append(f"{ds}:{'ok' if ok else 'missing'}")
    _check("dataset_record_files_exist", per_dataset_files_ok, ", ".join(details), checks, errors)

    report = {
        "generated_at_utc": _utc_now_iso(),
        "round_dir": str(round_dir),
        "pass": len(errors) == 0,
        "counts": {
            "all_records": len(all_records),
            "mapping_rows": len(mapping_rows),
            "unique_judge_ids": len(unique_ids),
            "chunk_files": len(chunk_files),
            "chunk_total_records": chunk_total,
            "datasets": datasets,
        },
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }
    _write_json(validation_dir / "input_validation_report.json", report)
    (validation_dir / "input_validation_report.md").write_text(_render_markdown(report), encoding="utf-8")

    print(json.dumps({"pass": report["pass"], "errors": len(errors), "warnings": len(warnings)}, ensure_ascii=False))
    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
