from __future__ import annotations

import json
from pathlib import Path
import sys

from effirag.config import RetrievalConfig
from effirag.retrieval import _select_phase1_diversity_reserve


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import summarize_phase6x_early_collapse_recovery as phase6x_summary  # noqa: E402
from validate_phase6s_artifacts import collect_phase6s_artifacts  # noqa: E402


def _run(run_id, seeds, hybrid=0.5, semantic=0.5, graph=0.5, bridge=0.5, grounding=0.0):
    return {
        "run_id": int(run_id),
        "seeds": set(seeds),
        "hybrid_run_score": float(hybrid),
        "run_score_components": {
            "semantic_coverage": float(semantic),
            "structural_connectivity": float(graph),
            "bridge_path_completeness": float(bridge),
            "entity_chunk_grounding_score": float(grounding),
            "cheap_pre_components": {
                "max_semantic_seed": float(semantic),
                "graph_seed_mass": float(graph),
                "bridge_proxy": float(bridge),
            },
        },
    }


def _row(sample_id: str, prediction: str, reserve_diag: dict | None = None):
    return {
        "sample_id": sample_id,
        "question": "Q",
        "prediction": prediction,
        "metrics": {
            "em": 1.0 if prediction == "A" else 0.0,
            "f1": 1.0 if prediction == "A" else 0.0,
            "supporting_fact_recall": 0.5,
            "supporting_fact_precision": 0.2,
        },
        "efficiency": {
            "retrieval_latency_ms": 100.0,
            "generation_latency_ms": 20.0,
            "total_latency_ms": 120.0,
        },
        "retrieval": {
            "selected_sentence_ids": ["chunk::Title::0"],
            "diagnostics": dict(reserve_diag or {}),
        },
        "rendered_sentence_ids": ["chunk::Title::0"],
    }


def test_baseline_unchanged_when_reserve_disabled():
    cfg = RetrievalConfig()
    cfg.phase1_diversity_reserve_enabled = False
    ranked = [_run(1, {"a", "b"}), _run(2, {"c", "d"})]
    reserve, diag = _select_phase1_diversity_reserve(ranked, [ranked[0]], cfg)
    assert reserve is None
    assert diag["enabled"] is False
    assert diag["added"] is False


def test_reserve_count_is_at_most_one():
    cfg = RetrievalConfig()
    cfg.phase1_diversity_reserve_enabled = True
    cfg.phase1_diversity_reserve_count = 1
    ranked = [_run(1, {"a", "b"}, 0.9), _run(2, {"a", "b"}, 0.8), _run(3, {"c", "d"}, 0.7)]
    reserve, diag = _select_phase1_diversity_reserve(ranked, [ranked[0]], cfg)
    assert reserve is not None
    assert int(diag["added_run_id"]) == 3
    assert int(diag["rank_after_reserve"]) == 2


def test_reserve_selection_does_not_depend_on_gold_fields():
    cfg = RetrievalConfig()
    cfg.phase1_diversity_reserve_enabled = True
    ranked = [_run(1, {"a", "b"}, 0.9), {**_run(2, {"c", "d"}, 0.8), "gold_answer": "A", "gold_support": True}]
    reserve, diag = _select_phase1_diversity_reserve(ranked, [ranked[0]], cfg)
    assert reserve is not None
    assert int(diag["added_run_id"]) == 2


def test_reserve_not_forced_into_final_selection_by_default():
    baseline_rows = [_row("s1", "A")]
    reserve_diag = {
        "phase1_reserve_added": True,
        "phase1_reserve_seed_or_run_id": 3,
        "phase1_reserve_source_seed_id": "seed-x",
        "phase1_reserve_reason": "nonredundant_best",
        "phase1_reserve_score": 0.7,
        "phase1_reserve_rank_before_reserve": 2,
        "phase1_reserve_rank_after_reserve": 3,
        "phase1_reserve_in_phase1_shortlist": True,
        "phase1_reserve_reaches_phase2": True,
        "phase1_reserve_has_sentence_candidates": True,
        "phase1_reserve_enters_ABR_candidate_pool": True,
        "phase1_reserve_ABR_score": 1.2,
        "phase1_reserve_ABR_rank": 4,
        "phase1_reserve_selected_by_ABR": True,
        "phase1_reserve_rendered": True,
        "phase1_reserve_unit_ids": ["chunk::Title::0"],
    }
    variant_rows = [_row("s1", "A", reserve_diag=reserve_diag)]
    _, reserve_rows = phase6x_summary._build_paired_rows("hotpotqa", baseline_rows, variant_rows)
    assert len(reserve_rows) == 1
    rr = reserve_rows[0]
    assert rr["reserve_forced_selected"] is False
    assert rr["reserve_forced_rendered"] is False
    assert rr["selected_because_of_forced_pin"] is False


def test_manifest_contains_exactly_four_scheduled_runs(tmp_path: Path):
    out_root = tmp_path / "phase6x"
    manifest = {
        "runs": [
            {"run_name": "r1", "dataset": "hotpotqa", "profile": "p1", "scheduled": True},
            {"run_name": "r2", "dataset": "hotpotqa", "profile": "p2", "scheduled": True},
            {"run_name": "r3", "dataset": "2wikimultihopqa", "profile": "p1", "scheduled": True},
            {"run_name": "r4", "dataset": "2wikimultihopqa", "profile": "p2", "scheduled": True},
        ]
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "phase6s_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = collect_phase6s_artifacts(out_root)
    assert int(report["counts"]["scheduled_runs"]) == 4


def test_paired_sample_id_mismatch_zero_when_identical():
    rows_a = [{"sample_id": "s1"}, {"sample_id": "s2"}]
    rows_b = [{"sample_id": "s1"}, {"sample_id": "s2"}]
    assert phase6x_summary._sample_id_mismatch_count(rows_a, rows_b) == 0


def test_paired_sample_id_mismatch_detected():
    rows_a = [{"sample_id": "s1"}, {"sample_id": "s2"}]
    rows_b = [{"sample_id": "s1"}, {"sample_id": "s3"}]
    assert phase6x_summary._sample_id_mismatch_count(rows_a, rows_b) == 2


def test_validation_catches_missing_run_outputs(tmp_path: Path):
    out_root = tmp_path / "phase6x_missing"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"runs": [{"run_name": "r1", "dataset": "hotpotqa", "profile": "p1", "scheduled": True}]}
    (out_root / "phase6s_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = collect_phase6s_artifacts(out_root)
    assert int(report["counts"]["scheduled_runs"]) == 1
    assert "scheduled_not_completed:r1:scheduled_but_missing" in list(report.get("strict_clean_errors", []) or [])
