from __future__ import annotations

from effirag.gl_rcedr import resolve_dynamic_weights


def _base():
    return {
        "recall": 0.34,
        "bridge_path": 0.22,
        "diversity": 0.16,
        "stability": 0.12,
        "redundancy": 0.16,
        "cost": 0.12,
    }


def test_dynamic_control_boosts_bridge_weight_when_dispersion_is_high():
    weights = resolve_dynamic_weights(
        base_weights=_base(),
        graph_dispersion=0.80,
        coverage_saturation=0.60,
        redundancy_ratio=0.20,
        dynamic_control_enabled=True,
    )
    assert weights["bridge_path"] > _base()["bridge_path"]


def test_dynamic_control_boosts_recall_when_coverage_is_low():
    weights = resolve_dynamic_weights(
        base_weights=_base(),
        graph_dispersion=0.30,
        coverage_saturation=0.20,
        redundancy_ratio=0.20,
        dynamic_control_enabled=True,
    )
    assert weights["recall"] > _base()["recall"]


def test_dynamic_control_boosts_redundancy_penalty_when_overlap_is_high():
    weights = resolve_dynamic_weights(
        base_weights=_base(),
        graph_dispersion=0.30,
        coverage_saturation=0.70,
        redundancy_ratio=0.60,
        dynamic_control_enabled=True,
    )
    assert weights["redundancy"] > _base()["redundancy"]


def test_no_dynamic_control_keeps_base_weights():
    base = _base()
    weights = resolve_dynamic_weights(
        base_weights=base,
        graph_dispersion=0.80,
        coverage_saturation=0.20,
        redundancy_ratio=0.60,
        dynamic_control_enabled=False,
    )
    assert weights == base
