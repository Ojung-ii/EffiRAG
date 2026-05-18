from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scripts() -> tuple[Path, Path]:
    root = _root()
    return (
        root / "scripts" / "validate_phase6s_artifacts.py",
        root / "scripts" / "summarize_phase6s_unified_acr_rcedr_results.py",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _create_attempt(
    *,
    out_root: Path,
    run_name: str,
    dataset: str,
    profile: str,
    ts: str,
    prompt_tokens: float = 300.0,
    f1: float = 0.4,
    with_summary: bool = True,
    with_query: bool = True,
) -> None:
    attempt_dir = out_root / "qa_runs" / run_name / dataset / dataset / ts
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if with_summary:
        _write_json(
            attempt_dir / "rag_summary.json",
            {
                "dataset": dataset,
                "qa_executed_samples": 5,
                "em": 0.2,
                "f1": f1,
                "prompt_tokens_avg": prompt_tokens,
                "completion_tokens_avg": 10.0,
                "retrieval_ms": 1234.0,
                "generation_ms": 45.0,
                "total_ms": 1300.0,
            },
        )
    if with_query:
        (attempt_dir / "rag_query_results.jsonl").write_text(
            json.dumps(
                {
                    "retrieval": {
                        "latency_ms": 1200.0,
                        "diagnostics": {
                            "latency_breakdown_ms": {
                                "sentence_rerank_ms": 10.0,
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
                    "summary_path": str(attempt_dir / "rag_summary.json"),
                }
            ],
        },
    )


def _section(md: str, start: str, end: str) -> str:
    return md.split(start, 1)[1].split(end, 1)[0]


def test_strict_clean_passes_for_fresh_complete_root(tmp_path: Path) -> None:
    validate, _ = _scripts()
    out_root = tmp_path / "phase6s_clean"
    _write_root_manifest(
        out_root,
        [{"run_name": "run_hotpot", "dataset": "hotpotqa", "profile": "unified_large", "scheduled": True}],
    )
    _create_attempt(
        out_root=out_root,
        run_name="run_hotpot",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_000001_000001",
    )

    p = subprocess.run(
        [sys.executable, str(validate), "--out-root", str(out_root), "--strict-clean"],
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0, p.stdout + "\n" + p.stderr


def test_strict_clean_fails_with_duplicate_manifest_run_name(tmp_path: Path) -> None:
    validate, _ = _scripts()
    out_root = tmp_path / "phase6s_dup"
    _write_root_manifest(
        out_root,
        [
            {"run_name": "dup", "dataset": "hotpotqa", "profile": "unified_large", "scheduled": True},
            {"run_name": "dup", "dataset": "2wikimultihopqa", "profile": "unified_large", "scheduled": True},
        ],
    )
    _create_attempt(
        out_root=out_root,
        run_name="dup",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_000001_000001",
    )

    p = subprocess.run(
        [sys.executable, str(validate), "--out-root", str(out_root), "--strict-clean"],
        text=True,
        capture_output=True,
    )
    assert p.returncode != 0
    assert "duplicate_run_name:dup" in p.stdout


def test_allow_stale_warns_but_does_not_fail(tmp_path: Path) -> None:
    validate, _ = _scripts()
    out_root = tmp_path / "phase6s_stale"
    _write_root_manifest(
        out_root,
        [{"run_name": "run_hotpot", "dataset": "hotpotqa", "profile": "unified_large", "scheduled": True}],
    )
    _create_attempt(
        out_root=out_root,
        run_name="run_hotpot",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_000001_000001",
    )
    _create_attempt(
        out_root=out_root,
        run_name="run_hotpot",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_000001_000002",
    )
    _create_attempt(
        out_root=out_root,
        run_name="extra_run",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_000001_000003",
    )
    _create_attempt(
        out_root=out_root,
        run_name="broken_run",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_000001_000004",
        with_summary=False,
        with_query=True,
    )

    p = subprocess.run(
        [sys.executable, str(validate), "--out-root", str(out_root), "--allow-stale"],
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0
    assert "multi_attempt_run_names=1" in p.stdout
    assert "extra_not_in_manifest=2" in p.stdout
    assert "incomplete_attempts=1" in p.stdout

    p_strict = subprocess.run(
        [sys.executable, str(validate), "--out-root", str(out_root), "--strict-clean"],
        text=True,
        capture_output=True,
    )
    assert p_strict.returncode != 0


def test_summary_moves_empty_context_to_result_warnings(tmp_path: Path) -> None:
    validate, summarize = _scripts()
    out_root = tmp_path / "phase6s_summary"
    _write_root_manifest(
        out_root,
        [{"run_name": "run_hotpot", "dataset": "hotpotqa", "profile": "unified_large", "scheduled": True}],
    )
    _create_attempt(
        out_root=out_root,
        run_name="run_hotpot",
        dataset="hotpotqa",
        profile="unified_large",
        ts="20260517_000001_000001",
        prompt_tokens=0.0,
        f1=0.0,
    )

    p = subprocess.run(
        [sys.executable, str(validate), "--out-root", str(out_root), "--strict-clean"],
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0

    subprocess.run([sys.executable, str(summarize), "--out-root", str(out_root)], check=True, text=True)
    md = (out_root / "PHASE6S_UNIFIED_ACR_RCEDR_SUMMARY.md").read_text(encoding="utf-8")

    sec1 = _section(md, "## 1. Executed QA Results", "## 2. Main HotpotQA/2Wiki Comparison")
    assert "| hotpotqa | unified_large |" in sec1
    assert "run_failure_or_missing" not in sec1

    sec7 = _section(md, "## 7. Artifact Warnings", "## 8. Result Warnings")
    assert "empty_context" not in sec7

    sec8 = _section(md, "## 8. Result Warnings", "## 9. Compared Profiles")
    assert "empty_context" in sec8
