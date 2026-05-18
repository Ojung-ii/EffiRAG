import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mk_run(out_root: Path, run_name: str, profile: str, retrieval_ms: float):
    run_dir = out_root / "qa_runs" / run_name / "hotpotqa" / "hotpotqa" / "20260518_000000_000001"
    summary = {
        "dataset": "hotpotqa",
        "em": 0.30,
        "f1": 0.50,
        "prompt_tokens_avg": 500.0,
        "retrieval_ms": retrieval_ms,
        "total_ms": retrieval_ms + 200.0,
        "supporting_fact_recall": 0.5,
        "supporting_fact_precision": 0.2,
        "bridge_noise_ratio": 0.9,
        "answer_bearing_chunk_present": 0.8,
        "answer_present_but_generation_fail": 0.1,
    }
    _write_json(run_dir / "rag_summary.json", summary)

    qrows = [
        {
            "sample_id": f"{run_name}-q0",
            "latency_breakdown_ms": {
                "retrieval_ms": retrieval_ms,
                "total_ms": retrieval_ms + 200.0,
                "proposal_total_ms": retrieval_ms * 0.9,
                "proposal_time_ms": retrieval_ms * 0.9,
                "semantic_entity_scan_ms": 200.0,
                "semantic_chunk_scan_ms": 100.0,
                "proposal_union_ms": 300.0,
                "proposal_subgraph_build_ms": 500.0,
                "unattributed_proposal_ms": 50.0,
            },
            "retrieval": {
                "diagnostics": {
                    "num_anchor_candidates": 4,
                    "num_union_candidates": 100,
                    "num_ppr_calls": 3,
                    "num_pair_candidates": 6,
                    "num_corridor_candidates": 4,
                    "objective_eval_calls": 80,
                }
            },
        },
        {
            "sample_id": f"{run_name}-q1",
            "latency_breakdown_ms": {
                "retrieval_ms": retrieval_ms + 10.0,
                "total_ms": retrieval_ms + 210.0,
                "proposal_total_ms": (retrieval_ms + 10.0) * 0.9,
                "proposal_time_ms": (retrieval_ms + 10.0) * 0.9,
                "semantic_entity_scan_ms": 210.0,
                "semantic_chunk_scan_ms": 110.0,
                "proposal_union_ms": 320.0,
                "proposal_subgraph_build_ms": 510.0,
                "unattributed_proposal_ms": 55.0,
            },
            "retrieval": {
                "diagnostics": {
                    "num_anchor_candidates": 5,
                    "num_union_candidates": 110,
                    "num_ppr_calls": 3,
                    "num_pair_candidates": 7,
                    "num_corridor_candidates": 5,
                    "objective_eval_calls": 82,
                }
            },
        },
    ]
    _write_jsonl(run_dir / "rag_query_results.jsonl", qrows)


def test_phase6t_proposal_bottleneck_summary_outputs(tmp_path: Path):
    out_root = tmp_path / "out"
    manifest = {
        "phase": "phase6t_proposal_bottleneck_profile",
        "runs": [
            {
                "run_name": "fast_hotpotqa",
                "dataset": "hotpotqa",
                "profile": "unified_acr_rcedr_v12_sota_contract_fast",
                "scheduled": True,
            },
            {
                "run_name": "fast_no_sentence_rerank_hotpotqa",
                "dataset": "hotpotqa",
                "profile": "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank",
                "scheduled": True,
            },
        ],
    }
    _write_json(out_root / "phase6s_run_manifest.json", manifest)
    _write_json(out_root / "phase6t_run_manifest.json", manifest)

    _mk_run(out_root, "fast_hotpotqa", "unified_acr_rcedr_v12_sota_contract_fast", 20000.0)
    _mk_run(
        out_root,
        "fast_no_sentence_rerank_hotpotqa",
        "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank",
        19800.0,
    )

    script = Path("scripts/summarize_phase6t_proposal_bottleneck_profile.py").resolve()
    cmd = [
        sys.executable,
        str(script),
        "--out-root",
        str(out_root),
        "--dataset",
        "hotpotqa",
    ]
    subprocess.run(cmd, check=True)

    md_path = out_root / "PHASE6T_PROPOSAL_BOTTLENECK_PROFILE.md"
    profile_json_path = out_root / "retrieval_profile_summary.json"
    assert md_path.exists()
    assert profile_json_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    assert "## 3. Proposal Breakdown" in md_text
    assert "## 8. Bottleneck Ranking" in md_text
    assert "run_failure_or_missing" not in md_text

    run1_profile = out_root / "qa_runs" / "fast_hotpotqa" / "retrieval_profile_summary.json"
    run1_per_query = out_root / "qa_runs" / "fast_hotpotqa" / "retrieval_profile_per_query.jsonl"
    assert run1_profile.exists()
    assert run1_per_query.exists()

    payload = json.loads(profile_json_path.read_text(encoding="utf-8"))
    assert "overall_stage_ranking" in payload
    assert len(payload.get("rows", [])) == 2
