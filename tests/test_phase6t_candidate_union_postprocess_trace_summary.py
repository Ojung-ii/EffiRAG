import json
import subprocess
import sys
from pathlib import Path


def _write_run(root: Path, run_name: str, f1: float, retrieval_ms: float) -> None:
    run_dir = root / "qa_runs" / run_name / "hotpotqa" / "hotpotqa" / "20260518_000000_000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rag_summary.json").write_text(
        json.dumps(
            {
                "f1": f1,
                "em": 0.1,
                "retrieval_ms": retrieval_ms,
                "total_ms": retrieval_ms + 1000.0,
                "prompt_tokens_avg": 300.0,
                "supporting_fact_recall": 0.5,
                "supporting_fact_precision": 0.4,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "rag_query_results.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "q1",
                "latency_breakdown_ms": {"retrieval_ms": retrieval_ms, "proposal_total_ms": 1000.0},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_phase6t_candidate_union_postprocess_trace_summary(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    trace_log = logs / "candidate_union_postprocess_trace.log"
    trace_log.write_text(
        "\n".join(
            [
                '[CAND_UNION_POST_TRACE] qid=q1 trace_id=q1 mode=diag_on step=candidate_union_outer_start t=1.000000 dt_prev_ms=0.000 dt_total_ms=0.000 extra={}',
                '[CAND_UNION_POST_TRACE] qid=q1 trace_id=q1 mode=diag_on step=candidate_union_inner_start t=1.100000 dt_prev_ms=100.000 dt_total_ms=100.000 extra={}',
                '[CAND_UNION_POST_TRACE] qid=q1 trace_id=q1 mode=diag_on step=candidate_union_inner_done t=2.100000 dt_prev_ms=1000.000 dt_total_ms=1100.000 extra={"candidate_union_total_ms":1000.0}',
                '[CAND_UNION_POST_TRACE] qid=q1 trace_id=q1 mode=diag_on step=candidate_result_receive_done t=3.100000 dt_prev_ms=1000.000 dt_total_ms=2100.000 extra={}',
                '[CAND_UNION_POST_TRACE] qid=q1 trace_id=q1 mode=diag_on step=candidate_union_outer_done t=4.100000 dt_prev_ms=1000.000 dt_total_ms=3100.000 extra={"candidate_union_outer_total_ms":3100.0}',
                '[CAND_UNION_POST_TRACE] qid=q2 trace_id=q2 mode=diag_off step=candidate_union_outer_start t=1.000000 dt_prev_ms=0.000 dt_total_ms=0.000 extra={}',
                '[CAND_UNION_POST_TRACE] qid=q2 trace_id=q2 mode=diag_off step=candidate_union_inner_start t=1.050000 dt_prev_ms=50.000 dt_total_ms=50.000 extra={}',
                '[CAND_UNION_POST_TRACE] qid=q2 trace_id=q2 mode=diag_off step=candidate_union_inner_done t=1.550000 dt_prev_ms=500.000 dt_total_ms=550.000 extra={"candidate_union_total_ms":500.0}',
                '[CAND_UNION_POST_TRACE] qid=q2 trace_id=q2 mode=diag_off step=candidate_result_receive_done t=2.050000 dt_prev_ms=500.000 dt_total_ms=1050.000 extra={}',
                '[CAND_UNION_POST_TRACE] qid=q2 trace_id=q2 mode=diag_off step=candidate_union_outer_done t=2.550000 dt_prev_ms=500.000 dt_total_ms=1550.000 extra={"candidate_union_outer_total_ms":1550.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "phase": "phase6t_candidate_union_postprocess_trace",
        "sample_size": 5,
        "out_root": str(out_root),
        "runs": [
            {
                "run_name": "posttrace_diag_on_hotpotqa",
                "dataset": "hotpotqa",
                "profile": "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank",
                "scheduled": True,
            },
            {
                "run_name": "posttrace_diag_off_hotpotqa",
                "dataset": "hotpotqa",
                "profile": "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank",
                "scheduled": True,
            },
        ],
    }
    (out_root / "phase6s_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_run(out_root, "posttrace_diag_on_hotpotqa", f1=0.45, retrieval_ms=12000.0)
    _write_run(out_root, "posttrace_diag_off_hotpotqa", f1=0.44, retrieval_ms=10000.0)

    script = Path("scripts/summarize_phase6t_candidate_union_postprocess_trace.py").resolve()
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--trace-log",
            str(trace_log),
            "--out-root",
            str(out_root),
        ],
        check=True,
    )

    out_json = out_root / "candidate_union_postprocess_trace_summary.json"
    out_md = out_root / "PHASE6T_CANDIDATE_UNION_POSTPROCESS_TRACE_SUMMARY.md"

    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "mode_trace" in payload
    assert "diag_on" in payload["mode_trace"]
    assert "diag_off" in payload["mode_trace"]

    md = out_md.read_text(encoding="utf-8")
    assert "Diagnostics On/Off Comparison" in md
    assert "inner_done_to_outer_done_ms" in md
