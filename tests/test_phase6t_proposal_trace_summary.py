import json
import subprocess
import sys
from pathlib import Path


def test_proposal_trace_summary_parser(tmp_path: Path):
    out_root = tmp_path / "out"
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    trace = logs / "proposal_trace.log"
    trace.write_text(
        "\n".join(
            [
                '[PROP_TRACE] qid=q1 step=proposal_start dt_prev_ms=0.000 dt_total_ms=0.000 extra={}',
                '[PROP_TRACE] qid=q1 step=query_setup_done dt_prev_ms=10.000 dt_total_ms=10.000 extra={"num_query_entities": 3}',
                '[PROP_TRACE] qid=q1 step=candidate_union_done dt_prev_ms=100.000 dt_total_ms=110.000 extra={"after_dedup": 25}',
                '[PROP_TRACE] qid=q1 step=phase1_run_scoring_done dt_prev_ms=900.000 dt_total_ms=1010.000 extra={"num_run_candidates": 3}',
                '[PROP_TRACE] qid=q1 step=proposal_end dt_prev_ms=20.000 dt_total_ms=1030.000 extra={"proposal_total_ms": 1000.0, "render_candidate_count": 8}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    script = Path("scripts/summarize_phase6t_proposal_trace.py").resolve()
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

    md = out_root / "PHASE6T_PROPOSAL_TRACE_SUMMARY.md"
    js = out_root / "proposal_trace_summary.json"
    assert md.exists()
    assert js.exists()

    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["query_count"] == 1
    assert payload["interval_count"] >= 1
    assert payload["per_query"][0]["qid"] == "q1"
    assert abs(payload["per_query"][0]["proposal_total_ms"] - 1000.0) < 1e-6

    content = md.read_text(encoding="utf-8")
    assert "Interval Statistics" in content
    assert "phase1_run_scoring_done" in content
