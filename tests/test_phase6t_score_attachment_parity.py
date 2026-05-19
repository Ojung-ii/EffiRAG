from __future__ import annotations

from scripts.compare_phase6t_candidate_score_parity import (
    _candidate_output_impact,
    _classify_prediction_change,
    _classify_reason,
)


def _row(*, candidate_id: str = "e::southern", bridge=None, bridge_missing=True, source="none"):
    return {
        "candidate_id": candidate_id,
        "final_score": 0.5,
        "semantic_score": 0.5,
        "graph_score": 0.5,
        "bridge_score": bridge,
        "corridor_score": None,
        "redundancy_score": None,
        "answerability_score": None,
        "missing_flags": {
            "semantic_missing": False,
            "graph_missing": False,
            "bridge_missing": bool(bridge_missing),
            "corridor_missing": True,
            "redundancy_missing": True,
        },
        "score_source": {
            "semantic": "semantic_scores",
            "graph": "proposal_scores",
            "bridge": source,
            "corridor": "none",
            "redundancy": "none",
        },
        "selected": False,
        "rendered": False,
    }


def test_bridge_no_score_default_expected_alignment():
    baseline = _row(bridge=None, bridge_missing=True, source="none")
    optimized = _row(bridge=None, bridge_missing=True, source="none")
    reason = _classify_reason(baseline, optimized)
    assert reason != "missing_score_default_diff"


def test_source_priority_consistency_when_component_sources_match():
    baseline = _row(bridge=0.0, bridge_missing=False, source="path_scores")
    optimized = _row(bridge=0.0, bridge_missing=False, source="path_scores")
    reason = _classify_reason(baseline, optimized)
    # parity should not be reported as missing-default/source-priority mismatch
    assert reason != "missing_score_default_diff"


def test_tail_candidate_output_impact_classification():
    row = {"selected": False, "rendered": False}
    impact = _candidate_output_impact(
        candidate_key="e::tail_only",
        topk_keys=["e::top1", "e::top2"],
        row=row,
    )
    assert impact == 0


def test_prediction_change_generation_nondeterminism_classification():
    cls = _classify_prediction_change(
        prediction_changed=True,
        b_selected=["s1", "s2"],
        o_selected=["s1", "s2"],
        b_rendered=["s1", "s2"],
        o_rendered=["s1", "s2"],
        b_context_hash="ctx",
        o_context_hash="ctx",
        b_prompt_hash="prompt",
        o_prompt_hash="prompt",
    )
    assert cls == "generation_nondeterminism_candidate"


def test_prediction_change_retrieval_drift_classification():
    cls = _classify_prediction_change(
        prediction_changed=True,
        b_selected=["s1", "s2"],
        o_selected=["s1", "s3"],
        b_rendered=["s1", "s2"],
        o_rendered=["s1", "s3"],
        b_context_hash="ctxA",
        o_context_hash="ctxB",
        b_prompt_hash="promptA",
        o_prompt_hash="promptB",
    )
    assert cls == "retrieval_drift"
