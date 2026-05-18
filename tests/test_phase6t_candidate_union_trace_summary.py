import json
import subprocess
import sys
from pathlib import Path


def test_candidate_union_trace_summary_parser(tmp_path: Path):
    out_root = tmp_path / "out"
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    trace = logs / "candidate_union_trace.log"
    trace.write_text(
        "\n".join(
            [
                '[CAND_UNION_TRACE] qid=q1 step=candidate_union_start dt_prev_ms=0.000 dt_total_ms=0.000 extra={}',
                '[CAND_UNION_TRACE] qid=q1 step=semantic_entity_candidates_done dt_prev_ms=100.000 dt_total_ms=100.000 extra={"num_semantic_entities": 50}',
                '[CAND_UNION_TRACE] qid=q1 step=candidate_merge_done dt_prev_ms=400.000 dt_total_ms=500.000 extra={"num_after_merge": 120}',
                '[CAND_UNION_TRACE] qid=q1 step=candidate_union_done dt_prev_ms=500.000 dt_total_ms=1000.000 extra={"candidate_union_total_ms": 1000.0, "after_topk": 80}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    script = Path("scripts/summarize_phase6t_candidate_union_trace.py").resolve()
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

    md = out_root / "PHASE6T_CANDIDATE_UNION_TRACE_SUMMARY.md"
    js = out_root / "candidate_union_trace_summary.json"
    assert md.exists()
    assert js.exists()

    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["query_count"] == 1
    assert payload["interval_count"] >= 1
    assert payload["per_query"][0]["qid"] == "q1"
    assert abs(payload["per_query"][0]["candidate_union_total_ms"] - 1000.0) < 1e-6

    content = md.read_text(encoding="utf-8")
    assert "Interval Statistics" in content
    assert "candidate_union_done" in content
