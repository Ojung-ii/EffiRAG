#!/usr/bin/env python3
"""Validate Phase-6S artifact consistency.

Strict mode fails when any artifact consistency rule is violated.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _read_json_with_error(path: Path) -> Tuple[Dict[str, Any], str]:
    if not path.exists():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        return {}, f"parse_error:{type(exc).__name__}"
    if not isinstance(payload, Mapping):
        return {}, "not_mapping"
    return dict(payload), ""


def _infer_profile_from_manifest(manifest: Mapping[str, Any], record: Mapping[str, Any], run_name: str) -> str:
    rec_profile = _safe_text(record.get("profile"))
    if rec_profile:
        return rec_profile
    man_profile = _safe_text(manifest.get("profile"))
    if man_profile:
        return man_profile
    mode = _safe_text(manifest.get("mode"))
    profile_name = _safe_text(manifest.get("profile_name"))
    if mode == "locked_precomputed" or "copy_span_instruction_4ds" in profile_name:
        return "legacy_sota"
    return run_name


def load_scheduled_runs(out_root: Path) -> Dict[str, Any]:
    qa_root = out_root / "qa_runs"
    root_manifest_path = out_root / "phase6s_run_manifest.json"
    root_manifest_exists = root_manifest_path.exists()
    schedule_source = "root_manifest"
    schedule_warnings: List[str] = []
    manifest_errors: List[str] = []
    runs: List[Dict[str, Any]] = []

    if root_manifest_exists:
        payload, err = _read_json_with_error(root_manifest_path)
        if err:
            manifest_errors.append(f"root_manifest_{err}")
        else:
            raw_runs = payload.get("runs", [])
            if not isinstance(raw_runs, list):
                manifest_errors.append("root_manifest_runs_not_list")
            else:
                for row in raw_runs:
                    if not isinstance(row, Mapping):
                        manifest_errors.append("root_manifest_run_row_not_mapping")
                        continue
                    run_name = _safe_text(row.get("run_name"))
                    dataset = _safe_text(row.get("dataset"))
                    profile = _safe_text(row.get("profile"))
                    if not run_name:
                        manifest_errors.append("root_manifest_run_name_missing")
                        continue
                    runs.append(
                        {
                            "run_name": run_name,
                            "dataset": dataset,
                            "profile": profile,
                            "gpu": _safe_text(row.get("gpu")),
                            "process_group": _safe_text(row.get("process_group")),
                            "role": _safe_text(row.get("role")),
                            "kind": _safe_text(row.get("kind")),
                            "scheduled": bool(row.get("scheduled", True)),
                        }
                    )
    else:
        schedule_source = "fallback_manifest_scan"
        schedule_warnings.append("WARNING: root manifest missing; reconstructed from per-run manifests.")
        for run_manifest_path in sorted(qa_root.glob("*/run_manifest.json")):
            run_name = run_manifest_path.parent.name
            manifest, err = _read_json_with_error(run_manifest_path)
            if err:
                manifest_errors.append(f"fallback_manifest_parse_error:{run_name}:{err}")
                continue
            records = manifest.get("records", [])
            if not isinstance(records, list) or not records:
                runs.append(
                    {
                        "run_name": run_name,
                        "dataset": "",
                        "profile": _safe_text(manifest.get("profile")),
                        "gpu": "",
                        "process_group": "",
                        "role": "unknown",
                        "kind": "fallback",
                        "scheduled": True,
                    }
                )
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                dataset = _safe_text(record.get("dataset"))
                profile = _infer_profile_from_manifest(manifest, record, run_name)
                runs.append(
                    {
                        "run_name": run_name,
                        "dataset": dataset,
                        "profile": profile,
                        "gpu": "",
                        "process_group": "",
                        "role": "unknown",
                        "kind": "fallback",
                        "scheduled": True,
                    }
                )

    # duplicate run_name detection
    seen: Dict[str, int] = {}
    for row in runs:
        name = _safe_text(row.get("run_name"))
        if not name:
            continue
        seen[name] = int(seen.get(name, 0) + 1)
    duplicate_run_names = sorted([name for name, cnt in seen.items() if cnt > 1])

    return {
        "runs": runs,
        "schedule_source": schedule_source,
        "schedule_warnings": schedule_warnings,
        "manifest_errors": manifest_errors,
        "root_manifest_exists": root_manifest_exists,
        "root_manifest_path": str(root_manifest_path),
        "duplicate_run_names": duplicate_run_names,
    }


@dataclass
class AttemptEntry:
    path: str
    has_summary: bool
    has_query: bool
    summary_path: str
    query_path: str
    summary_parse_error: str



def scan_run_attempts(out_root: Path) -> Dict[str, Dict[str, Any]]:
    qa_root = out_root / "qa_runs"
    run_attempts: Dict[str, Dict[str, Any]] = {}
    if not qa_root.exists():
        return run_attempts

    for run_dir in sorted([p for p in qa_root.iterdir() if p.is_dir()]):
        run_name = run_dir.name
        attempt_map: Dict[Path, Dict[str, Any]] = {}

        for summary_path in run_dir.rglob("rag_summary.json"):
            attempt_dir = summary_path.parent
            row = attempt_map.setdefault(
                attempt_dir,
                {
                    "has_summary": False,
                    "has_query": False,
                    "summary_path": "",
                    "query_path": "",
                    "summary_parse_error": "",
                },
            )
            row["has_summary"] = True
            row["summary_path"] = str(summary_path)
            _, err = _read_json_with_error(summary_path)
            if err:
                row["summary_parse_error"] = err

        for query_path in run_dir.rglob("rag_query_results.jsonl"):
            attempt_dir = query_path.parent
            row = attempt_map.setdefault(
                attempt_dir,
                {
                    "has_summary": False,
                    "has_query": False,
                    "summary_path": "",
                    "query_path": "",
                    "summary_parse_error": "",
                },
            )
            row["has_query"] = True
            row["query_path"] = str(query_path)

        entries: List[AttemptEntry] = []
        for attempt_dir, row in sorted(attempt_map.items(), key=lambda kv: str(kv[0])):
            entries.append(
                AttemptEntry(
                    path=str(attempt_dir),
                    has_summary=bool(row.get("has_summary", False)),
                    has_query=bool(row.get("has_query", False)),
                    summary_path=_safe_text(row.get("summary_path")),
                    query_path=_safe_text(row.get("query_path")),
                    summary_parse_error=_safe_text(row.get("summary_parse_error")),
                )
            )

        complete_entries = [e for e in entries if e.has_summary and e.has_query]
        incomplete_entries = [e for e in entries if (e.has_summary or e.has_query) and not (e.has_summary and e.has_query)]
        parse_error_entries = [e for e in entries if e.summary_parse_error]

        latest_complete: AttemptEntry | None = None
        if complete_entries:
            latest_complete = sorted(complete_entries, key=lambda e: e.path)[-1]

        run_attempts[run_name] = {
            "run_name": run_name,
            "attempt_count": len(entries),
            "complete_attempt_count": len(complete_entries),
            "incomplete_attempt_count": len(incomplete_entries),
            "attempt_entries": [e.__dict__ for e in entries],
            "complete_attempt_paths": [e.path for e in complete_entries],
            "incomplete_attempt_paths": [e.path for e in incomplete_entries],
            "parse_error_entries": [
                {"attempt_path": e.path, "summary_path": e.summary_path, "error": e.summary_parse_error}
                for e in parse_error_entries
            ],
            "latest_complete_path": latest_complete.path if latest_complete else "",
            "latest_complete_summary_path": latest_complete.summary_path if latest_complete else "",
            "latest_complete_query_path": latest_complete.query_path if latest_complete else "",
            "latest_complete_summary_parse_error": latest_complete.summary_parse_error if latest_complete else "",
        }

    return run_attempts


def collect_phase6s_artifacts(out_root: Path) -> Dict[str, Any]:
    schedule_info = load_scheduled_runs(out_root)
    scheduled_runs = list(schedule_info.get("runs", []) or [])
    run_attempts = scan_run_attempts(out_root)

    scheduled_names = [_safe_text(r.get("run_name")) for r in scheduled_runs if _safe_text(r.get("run_name"))]
    scheduled_name_set = set(scheduled_names)
    attempt_name_set = set(run_attempts.keys())

    statuses: List[Dict[str, Any]] = []
    completed_runs: List[Dict[str, Any]] = []
    artifact_warnings: List[Dict[str, Any]] = []

    for run in scheduled_runs:
        run_name = _safe_text(run.get("run_name"))
        dataset = _safe_text(run.get("dataset"))
        profile = _safe_text(run.get("profile"))
        info = run_attempts.get(run_name)
        if info is None:
            statuses.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "status": "scheduled_but_missing",
                    "reason": "no_attempt_directory",
                }
            )
            continue

        attempt_count = int(info.get("attempt_count", 0))
        complete_count = int(info.get("complete_attempt_count", 0))
        incomplete_count = int(info.get("incomplete_attempt_count", 0))
        latest_summary_path = Path(_safe_text(info.get("latest_complete_summary_path")))
        latest_query_path = Path(_safe_text(info.get("latest_complete_query_path")))

        if attempt_count > 1:
            artifact_warnings.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "status": "multi_attempt_run_names",
                    "attempt_count": attempt_count,
                    "complete_attempt_count": complete_count,
                    "incomplete_attempt_count": incomplete_count,
                    "path": "; ".join(info.get("complete_attempt_paths", []) or []),
                }
            )
        if complete_count > 1:
            artifact_warnings.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "status": "multi_complete_attempt",
                    "attempt_count": attempt_count,
                    "complete_attempt_count": complete_count,
                    "incomplete_attempt_count": incomplete_count,
                    "path": "; ".join(info.get("complete_attempt_paths", []) or []),
                }
            )
        if incomplete_count > 0:
            artifact_warnings.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "status": "incomplete_attempt",
                    "attempt_count": attempt_count,
                    "complete_attempt_count": complete_count,
                    "incomplete_attempt_count": incomplete_count,
                    "path": "; ".join(info.get("incomplete_attempt_paths", []) or []),
                }
            )

        if complete_count >= 1:
            if not latest_summary_path.exists() or not latest_query_path.exists():
                statuses.append(
                    {
                        "run_name": run_name,
                        "dataset": dataset,
                        "profile": profile,
                        "status": "scheduled_but_incomplete",
                        "reason": "latest_complete_paths_missing",
                    }
                )
                continue
            summary_payload, parse_err = _read_json_with_error(latest_summary_path)
            if parse_err:
                statuses.append(
                    {
                        "run_name": run_name,
                        "dataset": dataset,
                        "profile": profile,
                        "status": "parse_error",
                        "reason": f"latest_complete_summary_{parse_err}",
                        "path": str(latest_summary_path),
                    }
                )
                artifact_warnings.append(
                    {
                        "run_name": run_name,
                        "dataset": dataset,
                        "profile": profile,
                        "status": "parse_error",
                        "attempt_count": attempt_count,
                        "complete_attempt_count": complete_count,
                        "incomplete_attempt_count": incomplete_count,
                        "path": str(latest_summary_path),
                    }
                )
                continue

            statuses.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "status": "completed",
                    "reason": "latest_complete_attempt",
                }
            )
            completed_runs.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "summary_path": str(latest_summary_path),
                    "query_path": str(latest_query_path),
                    "summary": summary_payload,
                    "attempt_count": attempt_count,
                    "complete_attempt_count": complete_count,
                    "incomplete_attempt_count": incomplete_count,
                    "latest_complete_path": _safe_text(info.get("latest_complete_path")),
                }
            )
        else:
            status = "scheduled_but_incomplete" if attempt_count > 0 else "scheduled_but_missing"
            reason = "attempts_exist_without_complete_pair" if attempt_count > 0 else "no_attempt_directory"
            statuses.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "status": status,
                    "reason": reason,
                }
            )

    extra_names = sorted(list(attempt_name_set - scheduled_name_set))
    for run_name in extra_names:
        info = run_attempts.get(run_name, {})
        statuses.append(
            {
                "run_name": run_name,
                "dataset": "",
                "profile": "",
                "status": "extra_not_in_manifest",
                "reason": "run_directory_exists_but_not_scheduled",
            }
        )
        artifact_warnings.append(
            {
                "run_name": run_name,
                "dataset": "",
                "profile": "",
                "status": "extra_not_in_manifest",
                "attempt_count": int(info.get("attempt_count", 0)),
                "complete_attempt_count": int(info.get("complete_attempt_count", 0)),
                "incomplete_attempt_count": int(info.get("incomplete_attempt_count", 0)),
                "path": "; ".join(info.get("complete_attempt_paths", []) or info.get("incomplete_attempt_paths", [])),
            }
        )

    parse_error_count = 0
    incomplete_attempt_total = 0
    multi_attempt_run_names = 0
    for run_name, info in run_attempts.items():
        incomplete_attempt_total += int(info.get("incomplete_attempt_count", 0))
        if int(info.get("attempt_count", 0)) > 1:
            multi_attempt_run_names += 1
        parse_error_count += len(info.get("parse_error_entries", []) or [])

    completed_run_names = sorted([_safe_text(r.get("run_name")) for r in completed_runs if _safe_text(r.get("run_name"))])

    strict_errors: List[str] = []
    if not bool(schedule_info.get("root_manifest_exists", False)):
        strict_errors.append("root_manifest_missing")
    for err in list(schedule_info.get("manifest_errors", []) or []):
        strict_errors.append(f"manifest_error:{err}")
    for dup in list(schedule_info.get("duplicate_run_names", []) or []):
        strict_errors.append(f"duplicate_run_name:{dup}")

    status_by_run = {(_safe_text(r.get("run_name"))): _safe_text(r.get("status")) for r in statuses}
    for run_name in sorted(scheduled_name_set):
        st = status_by_run.get(run_name, "scheduled_but_missing")
        if st != "completed":
            strict_errors.append(f"scheduled_not_completed:{run_name}:{st}")

    for run_name in extra_names:
        strict_errors.append(f"extra_not_in_manifest:{run_name}")
    if multi_attempt_run_names > 0:
        strict_errors.append(f"multi_attempt_run_names:{multi_attempt_run_names}")
    if incomplete_attempt_total > 0:
        strict_errors.append(f"incomplete_attempts:{incomplete_attempt_total}")
    if parse_error_count > 0:
        strict_errors.append(f"parse_errors:{parse_error_count}")

    counts = {
        "scheduled_runs": len(scheduled_names),
        "run_names_with_attempts": len(attempt_name_set),
        "completed_run_names": len(set(completed_run_names)),
        "incomplete_attempts": int(incomplete_attempt_total),
        "extra_not_in_manifest": len(extra_names),
        "multi_attempt_run_names": int(multi_attempt_run_names),
        "parse_errors": int(parse_error_count),
    }

    return {
        "out_root": str(out_root),
        "schedule_source": _safe_text(schedule_info.get("schedule_source")),
        "schedule_warnings": list(schedule_info.get("schedule_warnings", []) or []),
        "manifest_errors": list(schedule_info.get("manifest_errors", []) or []),
        "root_manifest_exists": bool(schedule_info.get("root_manifest_exists", False)),
        "root_manifest_path": _safe_text(schedule_info.get("root_manifest_path")),
        "duplicate_run_names": list(schedule_info.get("duplicate_run_names", []) or []),
        "scheduled_runs": scheduled_runs,
        "run_attempts": run_attempts,
        "statuses": statuses,
        "completed_runs": completed_runs,
        "artifact_warnings": artifact_warnings,
        "counts": counts,
        "strict_errors": strict_errors,
    }


def _print_report(report: Mapping[str, Any]) -> None:
    counts = dict(report.get("counts", {}) or {})
    print("[Phase-6S artifact validation]")
    print(f"out_root={_safe_text(report.get('out_root'))}")
    print(f"schedule_source={_safe_text(report.get('schedule_source'))}")
    print(f"root_manifest_exists={bool(report.get('root_manifest_exists', False))}")
    print(f"scheduled_runs={counts.get('scheduled_runs', 0)}")
    print(f"run_names_with_attempts={counts.get('run_names_with_attempts', 0)}")
    print(f"completed_run_names={counts.get('completed_run_names', 0)}")
    print(f"incomplete_attempts={counts.get('incomplete_attempts', 0)}")
    print(f"extra_not_in_manifest={counts.get('extra_not_in_manifest', 0)}")
    print(f"multi_attempt_run_names={counts.get('multi_attempt_run_names', 0)}")
    print(f"parse_errors={counts.get('parse_errors', 0)}")

    schedule_warnings = list(report.get("schedule_warnings", []) or [])
    if schedule_warnings:
        print("schedule_warnings:")
        for warning in schedule_warnings:
            print(f"- {warning}")

    manifest_errors = list(report.get("manifest_errors", []) or [])
    if manifest_errors:
        print("manifest_errors:")
        for err in manifest_errors:
            print(f"- {err}")

    strict_errors = list(report.get("strict_errors", []) or [])
    if strict_errors:
        print("strict_errors:")
        for err in strict_errors:
            print(f"- {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase-6S artifacts")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", default="", help="Optional path to dump validation JSON")
    args = parser.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve()
    report = collect_phase6s_artifacts(out_root)

    if _safe_text(args.json):
        out_json = Path(_safe_text(args.json)).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _print_report(report)

    if args.strict and list(report.get("strict_errors", []) or []):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
