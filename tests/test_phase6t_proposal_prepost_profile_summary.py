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


def _mk_run(out_root: Path, run_name: str, profile: str):
    run_dir = out_root / "qa_runs" / run_name / "hotpotqa" / "hotpotqa" / "20260518_000000_000001"
    summary = {
        "dataset": "hotpotqa",
        "em": 0.31,
        "f1": 0.49,
        "prompt_tokens_avg": 520.0,
        "retrieval_ms": 1100.0,
        "total_ms": 1300.0,
        "supporting_fact_recall": 0.41,
        "supporting_fact_precision": 0.22,
    }
    _write_json(run_dir / "rag_summary.json", summary)

    rows = [
        {
            "sample_id": "q0",
            "latency_breakdown_ms": {
                "retrieval_ms": 1100.0,
                "proposal_total_ms": 900.0,
                "proposal_pre_union_ms": 250.0,
                "proposal_union_total_ms": 300.0,
                "proposal_post_union_ms": 320.0,
                "proposal_known_total_ms": 870.0,
                "unattributed_proposal_ms": 30.0,
                "proposal_unattributed_share": 30.0 / 900.0,
                "query_preprocess_ms": 20.0,
                "anchor_candidate_lookup_ms": 25.0,
                "phase1_run_generation_ms": 45.0,
                "phase1_run_scoring_ms": 90.0,
                "corridor_feature_extraction_ms": 35.0,
                "post_union_misc_ms": 12.0,
            },
            "retrieval": {
                "diagnostics": {
                    "num_query_entities": 6,
                    "num_anchor_candidates": 12,
                    "semantic_score_map_size": 40,
                    "num_run_candidates": 3,
                    "num_shortlisted_runs": 2,
                    "num_corridor_candidates": 8,
                }
            },
        }
    ]
    _write_jsonl(run_dir / "rag_query_results.jsonl", rows)


def test_proposal_prepost_summary_generation(tmp_path: Path):
    out_root = tmp_path / "out"
    profile = "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank"
    manifest = {
        "phase": "phase6t_proposal_prepost_profile",
        "runs": [
            {
                "run_name": "fast_no_sentence_rerank_hotpotqa",
                "dataset": "hotpotqa",
                "profile": profile,
                "scheduled": True,
            }
        ],
    }
    _write_json(out_root / "phase6s_run_manifest.json", manifest)
    _write_json(out_root / "phase6t_run_manifest.json", manifest)
    _mk_run(out_root, "fast_no_sentence_rerank_hotpotqa", profile)

    script = Path("scripts/summarize_phase6t_proposal_prepost_profile.py").resolve()
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-root",
            str(out_root),
            "--dataset",
            "hotpotqa",
            "--profile",
            profile,
        ],
        check=True,
    )

    summary_json = out_root / "proposal_pre_post_profile_summary.json"
    assert summary_json.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(payload.get("rows", [])) == 1

    run_profile = payload["run_profiles"][0]
    stage = run_profile["stage_metrics"]
    assert abs(stage["proposal_pre_union_ms"]["avg"] - 250.0) < 1e-6
    assert abs(stage["proposal_union_total_ms"]["avg"] - 300.0) < 1e-6
    assert abs(stage["proposal_post_union_ms"]["avg"] - 320.0) < 1e-6

    per_query_path = out_root / "qa_runs" / "fast_no_sentence_rerank_hotpotqa" / "proposal_pre_post_profile_per_query.jsonl"
    assert per_query_path.exists()
    row = json.loads(per_query_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["proposal_total_ms"] == 900.0
    assert row["proposal_known_total_ms"] == 870.0
    assert row["proposal_unattributed_ms"] == 30.0

    md = (out_root / "PHASE6T_PROPOSAL_PRE_POST_PROFILE.md").read_text(encoding="utf-8")
    assert "run_failure_or_missing" not in md
    assert "Proposal Pre/Union/Post Breakdown" in md


def test_proposal_prepost_overlap_note(tmp_path: Path):
    out_root = tmp_path / "out_overlap"
    profile = "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank"
    manifest = {
        "phase": "phase6t_proposal_prepost_profile",
        "runs": [
            {
                "run_name": "fast_no_sentence_rerank_hotpotqa",
                "dataset": "hotpotqa",
                "profile": profile,
                "scheduled": True,
            }
        ],
    }
    _write_json(out_root / "phase6s_run_manifest.json", manifest)
    _write_json(out_root / "phase6t_run_manifest.json", manifest)

    run_dir = out_root / "qa_runs" / "fast_no_sentence_rerank_hotpotqa" / "hotpotqa" / "hotpotqa" / "20260518_000000_000001"
    _write_json(
        run_dir / "rag_summary.json",
        {"dataset": "hotpotqa", "em": 0.0, "f1": 0.0, "prompt_tokens_avg": 500.0, "retrieval_ms": 1000.0, "total_ms": 1200.0},
    )
    _write_jsonl(
        run_dir / "rag_query_results.jsonl",
        [
            {
                "sample_id": "q0",
                "latency_breakdown_ms": {
                    "proposal_total_ms": 100.0,
                    "proposal_pre_union_ms": 80.0,
                    "proposal_union_total_ms": 30.0,
                    "proposal_post_union_ms": 20.0,
                },
                "retrieval": {"diagnostics": {"proposal_timer_overlap_note": "nested_or_overlapping_timers_detected"}},
            }
        ],
    )

    script = Path("scripts/summarize_phase6t_proposal_prepost_profile.py").resolve()
    subprocess.run([sys.executable, str(script), "--out-root", str(out_root)], check=True)

    payload = json.loads((out_root / "proposal_pre_post_profile_summary.json").read_text(encoding="utf-8"))
    rp = payload["run_profiles"][0]
    assert rp["overlap_detected"] is True
    md = (out_root / "PHASE6T_PROPOSAL_PRE_POST_PROFILE.md").read_text(encoding="utf-8")
    assert "nested_or_overlapping_timers_detected" in md
