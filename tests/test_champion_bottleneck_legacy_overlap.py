from __future__ import annotations

from effirag.champion_bottleneck_audit import legacy_overlap_for_query


def test_legacy_overlap_candidate_miss():
    out = legacy_overlap_for_query(
        legacy_rendered_ids=["A::1", "B::2"],
        candidate_ids=["C::3"],
        selected_ids=["C::3"],
        rendered_ids=["C::3"],
    )
    assert out["legacy_candidate_overlap"] == 0.0
    assert out["dominant_lost_stage"] == "candidate"


def test_legacy_overlap_selection_drop():
    out = legacy_overlap_for_query(
        legacy_rendered_ids=["A::1", "B::2"],
        candidate_ids=["A::1", "B::2", "C::3"],
        selected_ids=["A::1"],
        rendered_ids=["A::1"],
    )
    assert out["legacy_candidate_overlap"] > out["legacy_selected_overlap"]
    assert out["dominant_lost_stage"] == "selection"


def test_legacy_overlap_rendered_drop_and_not_lost():
    rendered_drop = legacy_overlap_for_query(
        legacy_rendered_ids=["A::1", "B::2"],
        candidate_ids=["A::1", "B::2"],
        selected_ids=["A::1", "B::2"],
        rendered_ids=["A::1"],
    )
    assert rendered_drop["dominant_lost_stage"] == "rendered"

    not_lost = legacy_overlap_for_query(
        legacy_rendered_ids=["A::1", "B::2"],
        candidate_ids=["A::1", "B::2"],
        selected_ids=["A::1", "B::2"],
        rendered_ids=["A::1", "B::2"],
    )
    assert not_lost["dominant_lost_stage"] == "not_lost"

