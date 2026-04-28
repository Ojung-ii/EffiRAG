#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate judged output JSONL files.")
    ap.add_argument("--round-dir", required=True)
    ap.add_argument("--judge-output-dir", required=True)
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    round_dir = Path(args.round_dir).resolve()
    judge_out_dir = Path(args.judge_output_dir).resolve()
    mapping_path = round_dir / "hidden_mapping" / "record_mapping.jsonl"
    schema_path = round_dir / "aggregation" / "expected_judge_output_schema.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"missing mapping file: {mapping_path}")

    mapping_rows = _read_jsonl(mapping_path)
    expected_ids = {str(r.get("judge_id")) for r in mapping_rows}
    seen_ids = set()
    errors: List[str] = []
    warnings: List[str] = []
    judged_count = 0

    if schema_path.exists():
        schema = dict(_read_json(schema_path) or {})
        required = list(schema.get("required", []) or [])
    else:
        required = [
            "judge_id",
            "sufficient",
            "score",
            "minimal_evidence_ids",
            "answer_surface_present",
            "equivalent_evidence",
            "insufficient_reason",
            "error",
        ]

    for fp in sorted(judge_out_dir.glob("*.jsonl")):
        rows = _read_jsonl(fp)
        for i, row in enumerate(rows):
            judged_count += 1
            judge_id = _strip(row.get("judge_id"))
            if not judge_id:
                errors.append(f"{fp.name}:{i}: missing judge_id")
                continue
            if judge_id in seen_ids:
                errors.append(f"{fp.name}:{i}: duplicate judge_id={judge_id}")
            seen_ids.add(judge_id)
            if judge_id not in expected_ids:
                warnings.append(f"{fp.name}:{i}: unknown judge_id={judge_id}")
            for key in required:
                if key not in row:
                    errors.append(f"{fp.name}:{i}: missing required field `{key}`")

            score = row.get("score")
            if _strip(score) != "":
                try:
                    s = float(score)
                    if s < 0 or s > 4:
                        errors.append(f"{fp.name}:{i}: score out of range [0,4] -> {score}")
                except Exception:
                    errors.append(f"{fp.name}:{i}: invalid score -> {score}")

            mids = row.get("minimal_evidence_ids")
            if mids is not None and not isinstance(mids, list):
                errors.append(f"{fp.name}:{i}: minimal_evidence_ids must be list or null")

    missing = sorted(expected_ids - seen_ids)
    if missing:
        warnings.append(f"missing judged ids count={len(missing)}")

    report = {
        "generated_at_utc": _utc_now_iso(),
        "round_dir": str(round_dir),
        "judge_output_dir": str(judge_out_dir),
        "judged_count": int(judged_count),
        "expected_count": int(len(expected_ids)),
        "missing_count": int(len(missing)),
        "errors": errors,
        "warnings": warnings,
        "pass": len(errors) == 0,
    }
    out_json = Path(args.output_json).resolve() if _strip(args.output_json) else (round_dir / "aggregation" / "judge_output_validation_report.json")
    _write_json(out_json, report)
    print(json.dumps({"pass": report["pass"], "errors": len(errors), "warnings": len(warnings)}, ensure_ascii=False))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
