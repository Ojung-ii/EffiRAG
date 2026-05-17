from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _scripts() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "scripts" / "summarize_phase6s_unified_acr_rcedr_results.py",
        root / "scripts" / "validate_phase6s_artifacts.py",
    )


def _create_complete_attempt(
    *,
    out_root: Path,
    run_name: str,
    dataset: str,
    profile: str,
    ts: str,
    f1: float = 0.4,
) -> Path:
    attempt_dir = out_root / "qa_runs" / run_name / dataset / dataset / ts
    attempt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = attempt_dir / "rag_summary.json"
    _write_json(
        summary_path,
        {
            "dataset": dataset,
            "qa_executed_samples": 5,
            "em": 0.2,
            "f1": f1,
            "prompt_tokens_avg": 300.0,
            "completion_tokens_avg": 20.0,
            "retrieval_ms": 1200.0,
            "generation_ms": 40.0,
            "total_ms": 1300.0,
        },
    )
    (attempt_dir / "rag_query_results.jsonl").write_text(
        json.dumps(
            {
                "retrieval": {
                    "latency_ms": 1200.0,
                    "diagnostics": {
                        "latency_breakdown_ms": {
                            "sentence_rerank_ms": 11.0,
                            "top1_correction_ms": 0.0,
                        },
                        "unified_acr_rcedr_diag": {"selection_ms": 22.0},
                        "gl_rcedr_diag": {"selection_ms": 0.0},
                        "answerability_selection_diag": {"selection_ms": 0.0},
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        out_root / "qa_runs" / run_name / "run_manifest.json",
        {
            "profile": profile,
            "records": [
                {
                    "dataset": dataset,
                    "profile": profile,
                    "summary_path": str(summary_path),
                }
            ],
        },
    )
    return summary_path


def _create_incomplete_attempt_query_only(*, out_root: Path, run_name: str, dataset: str, ts: str) -> None:
    attempt_dir = out_root / "qa_runs" / run_name / dataset / dataset / ts
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "rag_query_results.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(
        out_root / "qa_runs" / run_name / "run_manifest.json",
        {
            "records": [
                {
                    "dataset": dataset,
                    "profile": "unified_acr_rcedr_v12",
                    "summary_path": "",
                }
            ]
        },
    )


def _write_root_manifest(out_root: Path, runs: list[dict]) -> None:
    _write_json(
        out_root / "phase6s_run_manifest.json",
        {
            "phase": "phase6s_unified_acr_rcedr",
            "sample_size": 5,
            "out_root": str(out_root),
            "created_at": "2026-05-17T00:00:00Z",
            "runs": runs,
        },
    )


def _extract_between(md: str, start: str, end: str) -> str:
    return md.split(start, 1)[1].split(end, 1)[0]


def test_phase6s_root_manifest_and_completed_outputs_are_consistent(tmp_path: Path) -> None:
    summarize, validate = _scripts()
    out_root = tmp_path / "phase6s_ok"

    runs = [
        {"run_name": "run_hotpot", "dataset": "hotpotqa", "profile": "unified_large", "scheduled": True},
        {
            "run_name": "run_2wiki",
            "dataset": "2wikimultihopqa",
            "profile": "unified_acr_rcedr_v12",
            "scheduled": True,
        },
    ]
    _write_root_manifest(out_root, runs)
    _create_complete_attempt(
        out_root=out_root,
        run_name="run_hotpot",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_100000_000001",
    )
    _create_complete_attempt(
        out_root=out_root,
        run_name="run_2wiki",
        dataset="2wikimultihopqa",
        profile="unified_acr_rcedr_v12",
        ts="20260517_100000_000002",
    )

    strict = subprocess.run(
        [sys.executable, str(validate), "--out-root", str(out_root), "--strict"],
        text=True,
        capture_output=True,
    )
    assert strict.returncode == 0, strict.stdout + "\n" + strict.stderr

    subprocess.run([sys.executable, str(summarize), "--out-root", str(out_root)], check=True, text=True)
    md = (out_root / "PHASE6S_UNIFIED_ACR_RCEDR_SUMMARY.md").read_text(encoding="utf-8")
    assert "| scheduled_runs | 2 |" in md
    assert "| completed_run_names | 2 |" in md
    assert "No artifact warnings." in md


def test_phase6s_incomplete_and_extra_are_reported_and_strict_fails(tmp_path: Path) -> None:
    summarize, validate = _scripts()
    out_root = tmp_path / "phase6s_bad"

    runs = [
        {
            "run_name": "run_hotpot",
            "dataset": "hotpotqa",
            "profile": "unified_acr_rcedr_v12",
            "scheduled": True,
        }
    ]
    _write_root_manifest(out_root, runs)
    _create_incomplete_attempt_query_only(
        out_root=out_root,
        run_name="run_hotpot",
        dataset="hotpotqa",
        ts="20260517_100000_000001",
    )
    _create_complete_attempt(
        out_root=out_root,
        run_name="extra_run",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_100000_000002",
    )

    non_strict = subprocess.run(
        [sys.executable, str(validate), "--out-root", str(out_root)],
        text=True,
        capture_output=True,
    )
    assert non_strict.returncode == 0
    assert "incomplete_attempts=1" in non_strict.stdout
    assert "extra_not_in_manifest=1" in non_strict.stdout

    strict = subprocess.run(
        [sys.executable, str(validate), "--out-root", str(out_root), "--strict"],
        text=True,
        capture_output=True,
    )
    assert strict.returncode != 0

    subprocess.run([sys.executable, str(summarize), "--out-root", str(out_root)], check=True, text=True)
    md = (out_root / "PHASE6S_UNIFIED_ACR_RCEDR_SUMMARY.md").read_text(encoding="utf-8")
    assert "scheduled_but_incomplete" in md
    assert "extra_not_in_manifest" in md


def test_phase6s_duplicate_run_name_in_manifest_fails_strict(tmp_path: Path) -> None:
    _, validate = _scripts()
    out_root = tmp_path / "phase6s_dup"

    runs = [
        {"run_name": "dup", "dataset": "hotpotqa", "profile": "unified_large", "scheduled": True},
        {
            "run_name": "dup",
            "dataset": "2wikimultihopqa",
            "profile": "unified_acr_rcedr_v12",
            "scheduled": True,
        },
    ]
    _write_root_manifest(out_root, runs)
    _create_complete_attempt(
        out_root=out_root,
        run_name="dup",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_100000_000001",
    )

    strict = subprocess.run(
        [sys.executable, str(validate), "--out-root", str(out_root), "--strict"],
        text=True,
        capture_output=True,
    )
    assert strict.returncode != 0
    assert "duplicate_run_name:dup" in strict.stdout


def test_phase6s_multi_attempt_warns_and_strict_fails(tmp_path: Path) -> None:
    summarize, validate = _scripts()
    out_root = tmp_path / "phase6s_multi"

    runs = [
        {
            "run_name": "run_hotpot",
            "dataset": "hotpotqa",
            "profile": "unified_acr_rcedr_v12",
            "scheduled": True,
        }
    ]
    _write_root_manifest(out_root, runs)
    _create_complete_attempt(
        out_root=out_root,
        run_name="run_hotpot",
        dataset="hotpotqa",
        profile="unified_acr_rcedr_v12",
        ts="20260517_100000_000001",
    )
    _create_complete_attempt(
        out_root=out_root,
        run_name="run_hotpot",
        dataset="hotpotqa",
        profile="unified_acr_rcedr_v12",
        ts="20260517_100000_000002",
    )

    strict = subprocess.run(
        [sys.executable, str(validate), "--out-root", str(out_root), "--strict"],
        text=True,
        capture_output=True,
    )
    assert strict.returncode != 0
    assert "multi_attempt_run_names:1" in strict.stdout

    subprocess.run([sys.executable, str(summarize), "--out-root", str(out_root)], check=True, text=True)
    md = (out_root / "PHASE6S_UNIFIED_ACR_RCEDR_SUMMARY.md").read_text(encoding="utf-8")
    assert "multi_attempt_run_names" in md
    assert "multi_complete_attempt" in md


def test_phase6s_main_table_has_only_completed_rows_and_no_synthetic_zero_rows(tmp_path: Path) -> None:
    summarize, _ = _scripts()
    out_root = tmp_path / "phase6s_main"

    runs = [
        {"run_name": "run_hotpot", "dataset": "hotpotqa", "profile": "unified_large", "scheduled": True}
    ]
    _write_root_manifest(out_root, runs)
    _create_complete_attempt(
        out_root=out_root,
        run_name="run_hotpot",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_100000_000001",
    )

    subprocess.run([sys.executable, str(summarize), "--out-root", str(out_root)], check=True, text=True)
    md = (out_root / "PHASE6S_UNIFIED_ACR_RCEDR_SUMMARY.md").read_text(encoding="utf-8")

    sec1 = _extract_between(md, "## 1. Executed QA Results", "## 2. Main HotpotQA/2Wiki Comparison")
    assert "| hotpotqa | unified_large |" in sec1
    assert "run_failure_or_missing" not in sec1
    assert " 0.0000 |" not in sec1

    sec5 = _extract_between(md, "## 5. Completeness Check", "## 6. Artifact Consistency")
    assert "not_scheduled" in sec5
