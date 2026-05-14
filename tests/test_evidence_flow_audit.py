from effirag.evidence_flow_audit import (
    classify_failure_type,
    infer_lost_stage,
    ordered_unique_ids_texts,
    stage_metrics,
)
from effirag.types import ContextDocument, Sample


def _sample():
    return Sample(
        qid="q1",
        question="Who wrote the book and where was the author born?",
        answer="X",
        contexts=[
            ContextDocument(
                title="DocA",
                sentences=[
                    "Alice wrote the book.",
                    "She was born in Seoul.",
                ],
            ),
            ContextDocument(
                title="DocB",
                sentences=[
                    "Bob edited the manuscript.",
                ],
            ),
        ],
        supporting_facts=[("DocA", 0), ("DocA", 1)],
    )


def test_ordered_unique_ids_texts_dedup():
    ids, texts = ordered_unique_ids_texts(
        ["DocA::0", "DocA::0", "DocA::1", ""],
        ["a", "dup-a", "b", "c"],
    )
    assert ids == ["DocA::0", "DocA::1"]
    assert texts == ["a", "b"]


def test_stage_metrics_sentence_mode():
    sample = _sample()
    out = stage_metrics(
        sample=sample,
        unit_ids=["DocA::0", "DocB::0"],
        unit_texts=["Alice wrote the book.", "Bob edited the manuscript."],
        graph_mode="current_entity_graph",
    )
    assert out["gold_support_count"] == 2
    assert out["matched_gold_count"] == 1
    assert abs(float(out["sf_P"]) - 0.5) < 1e-9
    assert abs(float(out["sf_R"]) - 0.5) < 1e-9
    assert abs(float(out["sf_F1"]) - 0.5) < 1e-9
    assert int(out["tokens"]) > 0


def test_failure_priority():
    retrieval_miss = classify_failure_type(
        candidate_stage={"id_mapping_available": True, "matched_gold_count": 0, "sf_R": 0.0, "sf_F1_per_1k_tokens": 0.0},
        selected_stage={"sf_R": 0.0},
        rendered_stage={"sf_R": 0.0, "sf_F1_per_1k_tokens": 0.0},
    )
    assert retrieval_miss == "retrieval_miss"

    selector_drop = classify_failure_type(
        candidate_stage={"id_mapping_available": True, "matched_gold_count": 1, "sf_R": 1.0, "sf_F1_per_1k_tokens": 0.4},
        selected_stage={"sf_R": 0.5},
        rendered_stage={"sf_R": 0.5, "sf_F1_per_1k_tokens": 0.4},
    )
    assert selector_drop == "selector_drop"

    render_drop = classify_failure_type(
        candidate_stage={"id_mapping_available": True, "matched_gold_count": 1, "sf_R": 1.0, "sf_F1_per_1k_tokens": 0.4},
        selected_stage={"sf_R": 1.0},
        rendered_stage={"sf_R": 0.5, "sf_F1_per_1k_tokens": 0.4},
    )
    assert render_drop == "render_drop"

    low_density = classify_failure_type(
        candidate_stage={"id_mapping_available": True, "matched_gold_count": 1, "sf_R": 1.0, "sf_F1_per_1k_tokens": 0.4},
        selected_stage={"sf_R": 1.0},
        rendered_stage={"sf_R": 1.0, "sf_F1_per_1k_tokens": 0.2},
    )
    assert low_density == "low_density"

    ready = classify_failure_type(
        candidate_stage={"id_mapping_available": True, "matched_gold_count": 1, "sf_R": 1.0, "sf_F1_per_1k_tokens": 0.4},
        selected_stage={"sf_R": 1.0},
        rendered_stage={"sf_R": 1.0, "sf_F1_per_1k_tokens": 0.4},
    )
    assert ready == "generation_ready"


def test_infer_lost_stage():
    assert infer_lost_stage(0.8, 0.6, 0.6, 0.6) == "candidate"
    assert infer_lost_stage(0.8, 0.8, 0.5, 0.5) == "selected"
    assert infer_lost_stage(0.8, 0.8, 0.8, 0.5) == "rendered"
    assert infer_lost_stage(0.8, 0.8, 0.8, 0.8) == "not_lost"
