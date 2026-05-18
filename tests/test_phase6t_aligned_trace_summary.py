import json
import subprocess
import sys
from pathlib import Path


def test_aligned_trace_summary_parser(tmp_path: Path):
    out_root = tmp_path / "out"
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    trace = logs / "proposal_aligned_trace.log"
    trace.write_text(
        "\n".join(
            [
                "[PROP_ALIGNED_TRACE] qid=q1 trace_id=q1 scope=outer step=proposal_start t=1.000000 dt_prev_ms=0.000 dt_total_ms=0.000 extra={}",
                "[PROP_ALIGNED_TRACE] qid=q1 trace_id=q1 scope=outer step=candidate_union_outer_start t=2.000000 dt_prev_ms=1000.000 dt_total_ms=1000.000 extra={}",
                "[PROP_ALIGNED_TRACE] qid=q1 trace_id=q1 scope=candidate_union step=candidate_union_inner_start t=2.100000 dt_prev_ms=100.000 dt_total_ms=1100.000 extra={}",
                "[PROP_ALIGNED_TRACE] qid=q1 trace_id=q1 scope=candidate_union step=candidate_union_inner_done t=3.100000 dt_prev_ms=1000.000 dt_total_ms=2100.000 extra={\"candidate_union_total_ms\": 1000.0}",
                "[PROP_ALIGNED_TRACE] qid=q1 trace_id=q1 scope=outer step=candidate_union_outer_done t=4.100000 dt_prev_ms=1000.000 dt_total_ms=3100.000 extra={}",
                "[PROP_ALIGNED_TRACE] qid=q1 trace_id=q1 scope=outer step=phase1_run_generation_start t=4.600000 dt_prev_ms=500.000 dt_total_ms=3600.000 extra={}",
                "[PROP_ALIGNED_TRACE] qid=q1 trace_id=q1 scope=outer step=proposal_end t=5.000000 dt_prev_ms=400.000 dt_total_ms=4000.000 extra={\"proposal_total_ms\": 4000.0}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    script = Path("scripts/summarize_phase6t_aligned_trace.py").resolve()
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--trace-log",
            str(trace),
            "--out-root",
            str(out_root),
        ],
        check=True,
    )

    md = out_root / "PHASE6T_ALIGNED_PROPOSAL_TRACE_SUMMARY.md"
    js = out_root / "proposal_aligned_trace_summary.json"
    assert md.exists()
    assert js.exists()

    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["query_count"] == 1
    row = payload["per_query"][0]
    assert row["qid"] == "q1"
    assert abs(float(row["outer_candidate_union_ms"]) - 2100.0) < 1e-6
    assert abs(float(row["inner_candidate_union_ms"]) - 1000.0) < 1e-6
    assert abs(float(row["outer_start_to_inner_start_ms"]) - 100.0) < 1e-6
    assert abs(float(row["inner_done_to_outer_done_ms"]) - 1000.0) < 1e-6

    content = md.read_text(encoding="utf-8")
    assert "Candidate Union Boundary Table" in content
    assert "candidate_union_inner_done" in content
