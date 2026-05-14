from __future__ import annotations

from effirag.gl_rcedr import SeedSetCandidate, score_seed_set, select_best_seed_set


def _seed(
    sid: str,
    *,
    recall: float,
    bridge: float,
    diversity: float,
    stability: float,
    redundancy: float,
    cost: float,
) -> SeedSetCandidate:
    return SeedSetCandidate(
        seed_set_id=sid,
        sentence_ids=(f"{sid}_1", f"{sid}_2"),
        recall_proxy=recall,
        bridge_path_proxy=bridge,
        diversity_proxy=diversity,
        stability_proxy=stability,
        redundancy_penalty=redundancy,
        compactness_penalty=cost,
    )


def test_seed_set_scoring_prefers_high_recall_bridge_and_stability():
    weights = {
        "recall": 0.34,
        "bridge_path": 0.22,
        "diversity": 0.16,
        "stability": 0.12,
        "redundancy": 0.16,
        "cost": 0.12,
    }
    strong = _seed("strong", recall=0.80, bridge=0.72, diversity=0.61, stability=0.75, redundancy=0.12, cost=0.24)
    weak = _seed("weak", recall=0.62, bridge=0.31, diversity=0.45, stability=0.40, redundancy=0.21, cost=0.25)
    assert score_seed_set(strong, weights=weights, use_stability=True, use_bridge_path=True) > score_seed_set(
        weak, weights=weights, use_stability=True, use_bridge_path=True
    )


def test_seed_set_scoring_penalizes_redundancy_and_cost():
    weights = {
        "recall": 0.34,
        "bridge_path": 0.22,
        "diversity": 0.16,
        "stability": 0.12,
        "redundancy": 0.20,
        "cost": 0.16,
    }
    dense = _seed("dense", recall=0.74, bridge=0.52, diversity=0.50, stability=0.62, redundancy=0.18, cost=0.20)
    bloated = _seed("bloated", recall=0.75, bridge=0.53, diversity=0.50, stability=0.62, redundancy=0.56, cost=0.66)
    assert score_seed_set(dense, weights=weights, use_stability=True, use_bridge_path=True) > score_seed_set(
        bloated, weights=weights, use_stability=True, use_bridge_path=True
    )


def test_select_best_seed_set_respects_bridge_path_ablation():
    weights = {
        "recall": 0.34,
        "bridge_path": 0.22,
        "diversity": 0.16,
        "stability": 0.12,
        "redundancy": 0.16,
        "cost": 0.12,
    }
    bridge_heavy = _seed(
        "bridge_heavy",
        recall=0.64,
        bridge=0.88,
        diversity=0.54,
        stability=0.56,
        redundancy=0.15,
        cost=0.22,
    )
    recall_heavy = _seed(
        "recall_heavy",
        recall=0.78,
        bridge=0.30,
        diversity=0.56,
        stability=0.56,
        redundancy=0.15,
        cost=0.22,
    )
    best_on, _ = select_best_seed_set(
        [bridge_heavy, recall_heavy],
        weights=weights,
        use_stability=True,
        use_bridge_path=True,
    )
    best_off, _ = select_best_seed_set(
        [bridge_heavy, recall_heavy],
        weights=weights,
        use_stability=True,
        use_bridge_path=False,
    )
    assert best_on is not None and best_on.seed_set_id == "bridge_heavy"
    assert best_off is not None and best_off.seed_set_id == "recall_heavy"
