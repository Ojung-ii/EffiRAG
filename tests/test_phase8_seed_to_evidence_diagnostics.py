from pathlib import Path

from effirag.phase8_diagnostics import evidence_proposal, load_runs


def test_phase8_candidate_count_from_evidence_trace():
    query = {
        "evidence": {
            "evidence_proposal": {
                "num_seed_atoms": 3,
                "num_seed_carriers": 2,
                "num_final_evidence_candidates": 5,
            }
        }
    }
    proposal = evidence_proposal(query)
    assert proposal["num_final_evidence_candidates"] == 5


def test_missing_trace_reported_as_missing_not_zero(tmp_path: Path):
    run_dir = tmp_path / "legacy_512_10" / "pamae_seed_k5_refine" / "hotpotqa"
    run_dir.mkdir(parents=True)
    (run_dir / "rag_summary.json").write_text('{"num_queries": 100}', encoding="utf-8")
    runs = load_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["status"] == "MISSING"
    assert runs[0]["actual_num_queries"] == 0
    assert runs[0]["files"]["phase8_query_trace.jsonl"] is False
