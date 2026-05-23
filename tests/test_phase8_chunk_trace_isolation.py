import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_phase8_chunk_summarizer_latest_run_dedupes_query_ids(tmp_path):
    root = tmp_path / "phase8_chunk_medoid"
    run_dir = root / "hotpotqa" / "legacy_512_10" / "chunk_pamae_k5" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "rag_summary.json").write_text(json.dumps({"num_queries": 1}), encoding="utf-8")
    base = {
        "run_id": "run_001",
        "timestamp": "20260101T000000",
        "dataset": "hotpotqa",
        "profile": "legacy_512_10",
        "variant": "chunk_pamae_k5",
        "query_id": "q1",
    }
    _write_jsonl(
        run_dir / "rag_query_results.jsonl",
        [
            {
                **base,
                "sample_id": "q1",
                "run_timestamp": "20260101T000000",
                "metrics": {"supporting_fact_recall": 0.0, "supporting_fact_precision": 0.0, "supporting_fact_f1": 0.0, "f1": 0.0},
                "efficiency": {"retrieval_latency_ms": 10.0, "generation_ms": 0.0, "total_latency_ms": 10.0},
                "latency_breakdown_ms": {"retrieval_total_ms": 10.0, "phase8_chunk_medoid_proposal_ms": 3.0},
            }
        ],
    )
    duplicate_trace = [{**base, "chunk_universe_size": 10}, {**base, "chunk_universe_size": 20}]
    for name in [
        "phase7_query_trace.jsonl",
        "phase7_diagnostics.jsonl",
        "phase8_chunk_query_trace.jsonl",
        "phase8_chunk_stage_timing.jsonl",
        "phase8_chunk_seed_trace.jsonl",
        "phase8_chunk_evidence_trace.jsonl",
    ]:
        _write_jsonl(run_dir / name, duplicate_trace)

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_phase8_chunk_medoid.py",
            "--root",
            str(root),
            "--latest-only",
            "--dedupe-query-id",
            "--expected-limit",
            "1",
        ],
        check=True,
    )
    payload = json.loads((root / "phase8_chunk_probe_summary.json").read_text(encoding="utf-8"))
    assert payload["runs"][0]["num_queries"] == 1
    assert payload["runs"][0]["trace_warning_count"] > 0
    report = (root / "phase8_chunk_trace_repair_report.md").read_text(encoding="utf-8")
    assert "duplicate_rows" in report


def test_phase8_chunk_failure_analysis_excludes_rejected_variants(tmp_path):
    root = tmp_path / "phase8_chunk_medoid"
    for variant in ["chunk_pamae_k5", "chunk_bridge_refine_k5"]:
        run_dir = root / "hotpotqa" / "legacy_512_10" / variant / "run_001"
        run_dir.mkdir(parents=True)
        (run_dir / "rag_summary.json").write_text(json.dumps({"num_queries": 1}), encoding="utf-8")
        row = {
            "run_id": "run_001",
            "timestamp": "20260101T000000",
            "dataset": "hotpotqa",
            "profile": "legacy_512_10",
            "variant": variant,
            "query_id": "q1",
            "sample_id": "q1",
            "metrics": {"supporting_fact_recall": 0.0, "supporting_fact_precision": 0.0, "supporting_fact_f1": 0.0, "f1": 0.0},
            "efficiency": {"retrieval_latency_ms": 10.0, "generation_ms": 0.0, "total_latency_ms": 10.0},
        }
        _write_jsonl(run_dir / "rag_query_results.jsonl", [row])
        _write_jsonl(run_dir / "phase8_chunk_query_trace.jsonl", [{**row, "chunk_universe_size": 1}])

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_phase8_chunk_medoid_failures.py",
            "--root",
            str(root),
            "--latest-only",
            "--dedupe-query-id",
        ],
        check=True,
    )
    payload = json.loads((root / "phase8_chunk_probe_failure_analysis.json").read_text(encoding="utf-8"))
    assert {row["variant"] for row in payload["summary"]} == {"chunk_pamae_k5"}
