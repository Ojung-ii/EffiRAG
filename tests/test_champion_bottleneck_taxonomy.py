from __future__ import annotations

from effirag.champion_bottleneck_audit import classify_bottleneck_taxonomy


def test_candidate_bottleneck_rule():
    row = {
        "legacy_rendered_sf_R": 0.9,
        "candidate_sf_R": 0.4,
        "legacy_candidate_overlap": 0.0,
    }
    out = classify_bottleneck_taxonomy(row)
    assert out["candidate_bottleneck"] is True
    assert out["dominant_bottleneck"] == "candidate"


def test_selection_bottleneck_rule():
    row = {
        "legacy_rendered_sf_R": 0.8,
        "candidate_sf_R": 0.8,
        "selected_sf_R": 0.5,
        "legacy_candidate_overlap": 0.9,
        "legacy_selected_overlap": 0.3,
    }
    out = classify_bottleneck_taxonomy(row)
    assert out["selection_bottleneck"] is True
    assert out["dominant_bottleneck"] == "selection"


def test_unit_contract_bottleneck_rule():
    row = {
        "legacy_rendered_sf_R": 0.2,
        "candidate_sf_R": 0.2,
        "selected_sf_R": 0.2,
        "rendered_sf_R": 0.2,
        "rendered_sf_P": 0.2,
        "prompt_tokens": 1200,
        "avg_tokens_per_item": 95.0,
        "chunk_like_rate": 0.6,
    }
    out = classify_bottleneck_taxonomy(row)
    assert out["unit_bottleneck"] is True
    assert out["dominant_bottleneck"] == "unit_contract"


def test_density_bottleneck_rule():
    row = {
        "legacy_rendered_sf_R": 0.5,
        "candidate_sf_R": 0.5,
        "selected_sf_R": 0.5,
        "rendered_sf_R": 0.7,
        "rendered_sf_P": 0.2,
        "redundancy_rate": 0.7,
        "sf_F1_per_1k_prompt": 0.1,
        "prompt_tokens": 200,
        "avg_tokens_per_item": 20.0,
        "chunk_like_rate": 0.0,
    }
    out = classify_bottleneck_taxonomy(row)
    assert out["density_bottleneck"] is True
    assert out["dominant_bottleneck"] == "density"


def test_generation_bottleneck_rule():
    row = {
        "legacy_rendered_sf_R": 0.2,
        "candidate_sf_R": 0.7,
        "selected_sf_R": 0.7,
        "rendered_sf_R": 0.8,
        "rendered_sf_P": 0.5,
        "QA_F1": 0.0,
        "prompt_tokens": 200,
        "avg_tokens_per_item": 20.0,
        "chunk_like_rate": 0.0,
        "redundancy_rate": 0.1,
        "sf_F1_per_1k_prompt": 0.7,
    }
    out = classify_bottleneck_taxonomy(row)
    assert out["generation_bottleneck"] is True
    assert out["dominant_bottleneck"] == "generation"

