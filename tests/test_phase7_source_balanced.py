from effirag.phase7_source_balanced import source_balanced_union


def _row(source_id, score, tags=()):
    return {
        "node": f"s::{source_id}",
        "source_id": source_id,
        "proposal_score": score,
        "flow_score": score / 10.0,
        "semantic_score": score / 20.0,
        "source_tags": set(tags),
    }


def test_source_balanced_union_respects_quotas_and_deduplicates():
    shared = _row("A::0", 10.0, {"semantic"})
    rows = {
        "semantic": [shared, _row("B::0", 9.0, {"semantic"})],
        "entity_title": [_row("A::0", 8.0, {"entity_title"}), _row("C::0", 7.0, {"entity_title"})],
        "graph_flow": [_row("D::0", 6.0, {"graph_flow"})],
        "anchor_neighborhood": [_row("E::0", 5.0, {"anchor_neighborhood"})],
    }
    selected, diag = source_balanced_union(
        ranked_candidates=rows,
        quotas={
            "semantic": 1,
            "entity_title": 1,
            "graph_flow": 1,
            "anchor_neighborhood": 1,
        },
        top_m=4,
        fill_remaining=True,
    )

    ids = [row["source_id"] for row in selected]
    assert ids == ["A::0", "C::0", "D::0", "E::0"]
    assert len(ids) == len(set(ids))
    assert set(selected[0]["source_tags"]) == {"semantic", "entity_title"}
    assert diag["selected_by_counts"]["semantic"] == 1
    assert diag["selected_by_counts"]["entity_title"] == 1
    assert diag["quota_fill_rates"]["graph_flow"] == 1.0


def test_source_balanced_union_backfills_by_existing_score():
    rows = {
        "semantic": [_row("A::0", 1.0, {"semantic"})],
        "entity_title": [],
        "graph_flow": [_row("B::0", 10.0, {"graph_flow"}), _row("C::0", 9.0, {"graph_flow"})],
        "anchor_neighborhood": [],
    }
    selected, diag = source_balanced_union(
        ranked_candidates=rows,
        quotas={
            "semantic": 1,
            "entity_title": 1,
            "graph_flow": 0,
            "anchor_neighborhood": 0,
        },
        top_m=3,
        fill_remaining=True,
    )

    assert [row["source_id"] for row in selected] == ["A::0", "B::0", "C::0"]
    assert diag["fill_remaining_count"] == 2

