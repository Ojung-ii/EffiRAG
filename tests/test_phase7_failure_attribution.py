from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_phase7_query_intent_failures.py"
    spec = importlib.util.spec_from_file_location("phase7_failure_analyzer", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_failure_attribution_final_selection_failed():
    mod = _load_module()
    diag = {
        "gold_supporting_facts": [{"title": "Doc A", "sent_idx": 0, "unit_id": "Doc A::0"}],
        "phase1_seed_ids": ["Doc A::0", "Doc B::1"],
        "phase2_candidate_ids": ["Doc A::0", "Doc B::1"],
        "selected_evidence_ids": ["Doc B::1"],
        "rendered_evidence_ids": ["Doc B::1"],
        "selected_to_rendered_match": True,
        "query_f1": 0.0,
    }
    qtrace = {
        "candidate_source_contribution_eval_only": {
            "gold_found_by_semantic": 1,
            "gold_found_by_entity_title": 1,
            "gold_found_by_relation_cue": 0,
            "gold_found_by_answer_type": 0,
            "gold_found_by_graph_flow": 1,
        },
        "num_selected_atoms": 8,
    }
    oracle = {
        "selected_context_f1": 0.0,
        "gold_support_context_f1": 1.0,
        "phase1_oracle_context_f1": 1.0,
        "selected_plus_gold_context_f1": 1.0,
    }
    result = mod._classify(
        dataset="hotpotqa",
        profile="balanced_384_8",
        variant="intent_p1",
        qid="q1",
        diag_row=diag,
        qtrace_row=qtrace,
        oracle_row=oracle,
        rag_row={},
    )
    assert result["primary_stage"] == "FINAL_SELECTION_FAILED"


def test_failure_attribution_phase1_not_found_semantic_anchor_failed():
    mod = _load_module()
    diag = {
        "gold_supporting_facts": [{"title": "Doc A", "sent_idx": 0, "unit_id": "Doc A::0"}],
        "phase1_seed_ids": ["Doc B::1"],
        "phase2_candidate_ids": ["Doc B::1"],
        "selected_evidence_ids": ["Doc B::1"],
        "rendered_evidence_ids": ["Doc B::1"],
        "selected_to_rendered_match": True,
        "query_f1": 0.0,
    }
    qtrace = {
        "candidate_source_contribution_eval_only": {
            "gold_found_by_semantic": 0,
            "gold_found_by_entity_title": 0,
            "gold_found_by_relation_cue": 0,
            "gold_found_by_answer_type": 0,
            "gold_found_by_graph_flow": 0,
        },
        "num_selected_atoms": 8,
    }
    oracle = {
        "selected_context_f1": 0.0,
        "gold_support_context_f1": 1.0,
        "phase1_oracle_context_f1": 0.0,
        "selected_plus_gold_context_f1": 0.0,
    }
    result = mod._classify(
        dataset="2wikimultihopqa",
        profile="balanced_384_8",
        variant="baseline_bq",
        qid="q2",
        diag_row=diag,
        qtrace_row=qtrace,
        oracle_row=oracle,
        rag_row={},
    )
    assert result["primary_stage"] == "PHASE1_NOT_FOUND"
    assert result["secondary_stage"] == "SEMANTIC_ANCHOR_FAILED"

