from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_phase6s_summary_separates_completed_and_completeness_statuses(tmp_path: Path):
    out_root = tmp_path / "phase6s_smoke_like"
    qa_root = out_root / "qa_runs"
    qa_root.mkdir(parents=True, exist_ok=True)

    # Root schedule: one completed, one scheduled missing, one parse error.
    root_manifest = {
        "phase": "phase6s_unified_acr_rcedr",
        "sample_size": 5,
        "out_root": str(out_root),
        "created_at": "2026-05-17T00:00:00Z",
        "runs": [
            {
                "run_name": "ok_hotpot",
                "dataset": "hotpotqa",
                "profile": "unified_acr_rcedr_v12",
                "gpu": "0",
                "process_group": "A",
                "role": "main",
                "scheduled": True,
            },
            {
                "run_name": "missing_hotpot",
                "dataset": "hotpotqa",
                "profile": "unified_acr_rcedr_v12_beam3",
                "gpu": "1",
                "process_group": "D",
                "role": "ablation",
                "scheduled": True,
            },
            {
                "run_name": "bad_2wiki",
                "dataset": "2wikimultihopqa",
                "profile": "unified_acr_rcedr_v12_no_bridge",
                "gpu": "1",
                "process_group": "D",
                "role": "ablation",
                "scheduled": True,
            },
        ],
    }
    _write_json(out_root / "phase6s_run_manifest.json", root_manifest)

    # Completed run
    ok_summary = qa_root / "ok_hotpot" / "hotpotqa" / "hotpotqa" / "ts" / "rag_summary.json"
    _write_json(
        ok_summary,
        {
            "dataset": "hotpotqa",
            "qa_executed_samples": 5,
            "em": 0.2,
            "f1": 0.4,
            "prompt_tokens_avg": 300.0,
            "completion_tokens_avg": 15.0,
            "retrieval_ms": 1234.0,
            "generation_ms": 45.0,
            "total_ms": 1300.0,
        },
    )
    _write_json(
        qa_root / "ok_hotpot" / "run_manifest.json",
        {
            "profile": "unified_acr_rcedr_v12",
            "records": [
                {
                    "dataset": "hotpotqa",
                    "profile": "unified_acr_rcedr_v12",
                    "summary_path": str(ok_summary),
                }
            ],
        },
    )
    (ok_summary.parent / "rag_query_results.jsonl").write_text(
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

    # Parse error run: summary file exists but invalid JSON.
    bad_summary = qa_root / "bad_2wiki" / "2wikimultihopqa" / "2wikimultihopqa" / "ts" / "rag_summary.json"
    bad_summary.parent.mkdir(parents=True, exist_ok=True)
    bad_summary.write_text("{not-json", encoding="utf-8")
    _write_json(
        qa_root / "bad_2wiki" / "run_manifest.json",
        {
            "profile": "unified_acr_rcedr_v12_no_bridge",
            "records": [
                {
                    "dataset": "2wikimultihopqa",
                    "profile": "unified_acr_rcedr_v12_no_bridge",
                    "summary_path": str(bad_summary),
                }
            ],
        },
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_phase6s_unified_acr_rcedr_results.py"
    subprocess.run(
        [sys.executable, str(script), "--out-root", str(out_root)],
        check=True,
        text=True,
    )

    md_path = out_root / "PHASE6S_UNIFIED_ACR_RCEDR_SUMMARY.md"
    md = md_path.read_text(encoding="utf-8")

    # Executed table should contain only completed runs and never synthetic failure labels.
    sec1 = md.split("## 1. Executed QA Results", 1)[1].split("## 2. Main HotpotQA/2Wiki Comparison", 1)[0]
    assert "| hotpotqa | unified_acr_rcedr_v12 |" in sec1
    assert "run_failure_or_missing" not in sec1
    assert "missing_hotpot" not in sec1

    # Completeness section should carry non-scheduled / missing / parse statuses.
    sec5 = md.split("## 5. Completeness Check", 1)[1].split("## 6. Artifact Consistency", 1)[0]
    assert "scheduled_but_missing" in sec5 or "scheduled_but_incomplete" in sec5
    assert "not_scheduled" in sec5
