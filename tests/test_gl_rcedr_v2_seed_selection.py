from __future__ import annotations

from effirag.gl_rcedr import SeedSetCandidate, select_best_seed_set_v2
from effirag.marginal_evidence_utility import MarginalUtilityWeights


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
        sentence_ids=(f"{sid}_a", f"{sid}_b"),
        recall_proxy=recall,
        bridge_path_proxy=bridge,
        diversity_proxy=diversity,
        stability_proxy=stability,
        redundancy_penalty=redundancy,
        compactness_penalty=cost,
    )


def test_v2_seed_selection_prefers_high_gain_low_redundancy():
    weights = MarginalUtilityWeights(evidence_gain=0.52, bridge_gain=0.28, redundancy=0.22, token_cost=0.14)
    stable_cover = _seed("stable_cover", recall=0.82, bridge=0.60, diversity=0.64, stability=0.74, redundancy=0.10, cost=0.20)
    noisy_redundant = _seed("noisy_redundant", recall=0.80, bridge=0.62, diversity=0.40, stability=0.30, redundancy=0.56, cost=0.21)
    best, _scores = select_best_seed_set_v2(
        [stable_cover, noisy_redundant],
        weights=weights,
        bridge_lambda=1.0,
        seed_stability_weight=0.15,
        seed_diversity_weight=0.15,
        use_stability=True,
        use_bridge_path=True,
        redundancy_enabled=True,
        cost_enabled=True,
        density_first=False,
    )
    assert best is not None
    assert best.seed_set_id == "stable_cover"


def test_v2_seed_selection_responds_to_bridge_lambda():
    weights = MarginalUtilityWeights(evidence_gain=0.52, bridge_gain=0.28, redundancy=0.22, token_cost=0.14)
    bridge_heavy = _seed("bridge_heavy", recall=0.68, bridge=0.90, diversity=0.50, stability=0.45, redundancy=0.14, cost=0.22)
    recall_heavy = _seed("recall_heavy", recall=0.83, bridge=0.20, diversity=0.54, stability=0.45, redundancy=0.14, cost=0.22)
    best_low, scores_low = select_best_seed_set_v2(
        [bridge_heavy, recall_heavy],
        weights=weights,
        bridge_lambda=0.85,
        seed_stability_weight=0.15,
        seed_diversity_weight=0.15,
        use_stability=True,
        use_bridge_path=True,
        redundancy_enabled=True,
        cost_enabled=True,
        density_first=False,
    )
    best_high, scores_high = select_best_seed_set_v2(
        [bridge_heavy, recall_heavy],
        weights=weights,
        bridge_lambda=1.35,
        seed_stability_weight=0.15,
        seed_diversity_weight=0.15,
        use_stability=True,
        use_bridge_path=True,
        redundancy_enabled=True,
        cost_enabled=True,
        density_first=False,
    )
    assert best_low is not None and best_high is not None
    gap_low = float(scores_low["bridge_heavy"]) - float(scores_low["recall_heavy"])
    gap_high = float(scores_high["bridge_heavy"]) - float(scores_high["recall_heavy"])
    assert gap_high > gap_low
