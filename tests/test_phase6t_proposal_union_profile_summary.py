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
        "em": 0.30,
        "f1": 0.50,
        "prompt_tokens_avg": 500.0,
        "retrieval_ms": 1000.0,
        "total_ms": 1200.0,
        "supporting_fact_recall": 0.4,
        "supporting_fact_precision": 0.2,
    }
    _write_json(run_dir / "rag_summary.json", summary)

    rows = [
        {
            "sample_id": "q0",
            "latency_breakdown_ms": {
                "retrieval_ms": 1000.0,
                "total_ms": 1200.0,
                "proposal_total_ms": 900.0,
                "proposal_time_ms": 900.0,
                "proposal_union_total_ms": 400.0,
                "proposal_union_ms": 400.0,
                "semantic_candidate_union_ms": 400.0,
                "raw_semantic_entity_fetch_ms": 120.0,
                "raw_semantic_chunk_fetch_ms": 80.0,
                "graph_reserve_fetch_ms": 60.0,
                "entity_to_chunk_expand_ms": 40.0,
                "candidate_materialization_ms": 50.0,
                "score_merge_ms": 70.0,
                "dedup_ms": 30.0,
                "sort_topk_ms": 20.0,
            },
            "retrieval": {
                "diagnostics": {
                    "num_raw_semantic_entities": 50,
                    "num_raw_semantic_chunks": 30,
                    "num_chunk_candidates_after_dedup": 20,
                    "num_candidate_objects_built": 20,
                }
            },
        }
    ]
    _write_jsonl(run_dir / "rag_query_results.jsonl", rows)


def test_proposal_union_summary_alias_dedup(tmp_path: Path):
    out_root = tmp_path / "out"
    profile = "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank"
    manifest = {
        "phase": "phase6t_proposal_union_profile",
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

    script = Path("scripts/summarize_phase6t_proposal_union_profile.py").resolve()
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

    summary_json = out_root / "proposal_union_profile_summary.json"
    assert summary_json.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(payload.get("rows", [])) == 1

    run_profile = payload["run_profiles"][0]
    stage = run_profile["stage_metrics"]
    assert abs(stage["proposal_union_total_ms"]["avg"] - 400.0) < 1e-6

    per_query_path = out_root / "qa_runs" / "fast_no_sentence_rerank_hotpotqa" / "proposal_union_profile_per_query.jsonl"
    assert per_query_path.exists()
    line = per_query_path.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert row["proposal_union_total_ms"] == 400.0
    # avoid synthetic failure rows
    md = (out_root / "PHASE6T_PROPOSAL_UNION_PROFILE.md").read_text(encoding="utf-8")
    assert "run_failure_or_missing" not in md


def test_proposal_union_optimization_summary(tmp_path: Path):
    baseline = tmp_path / "baseline"
    optimized = tmp_path / "optimized"

    base_payload = {
        "rows": [{"EM": 0.3, "F1": 0.5, "prompt_tokens_avg": 500.0, "supporting_fact_recall": 0.4, "supporting_fact_precision": 0.2}],
        "run_profiles": [
            {
                "stage_metrics": {
                    "retrieval_ms": {"avg": 1000.0},
                    "proposal_total_ms": {"avg": 900.0},
                    "proposal_union_total_ms": {"avg": 400.0},
                    "candidate_materialization_ms": {"avg": 200.0},
                    "unattributed_proposal_ms": {"avg": 100.0},
                }
            }
        ],
    }
    opt_payload = {
        "rows": [{"EM": 0.3, "F1": 0.5, "prompt_tokens_avg": 500.0, "supporting_fact_recall": 0.4, "supporting_fact_precision": 0.2}],
        "run_profiles": [
            {
                "stage_metrics": {
                    "retrieval_ms": {"avg": 800.0},
                    "proposal_total_ms": {"avg": 700.0},
                    "proposal_union_total_ms": {"avg": 250.0},
                    "candidate_materialization_ms": {"avg": 80.0},
                    "unattributed_proposal_ms": {"avg": 60.0},
                }
            }
        ],
    }
    _write_json(baseline / "proposal_union_profile_summary.json", base_payload)
    _write_json(optimized / "proposal_union_profile_summary.json", opt_payload)

    script = Path("scripts/summarize_phase6t_proposal_union_optimization.py").resolve()
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline-out-root",
            str(baseline),
            "--optimized-out-root",
            str(optimized),
        ],
        check=True,
    )

    assert (optimized / "PHASE6T_PROPOSAL_UNION_OPTIMIZATION_SUMMARY.md").exists()
    assert (optimized / "proposal_union_optimization_summary.json").exists()
