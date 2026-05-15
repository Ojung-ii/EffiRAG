from __future__ import annotations

from effirag.marginal_evidence_utility import (
    MarginalUtilityWeights,
    marginal_evidence_utility,
    resolve_adaptive_bridge_lambda,
)


def test_evidence_gain_increases_utility():
    w = MarginalUtilityWeights(evidence_gain=0.6, bridge_gain=0.2, redundancy=0.2, token_cost=0.1)
    low = marginal_evidence_utility(
        evidence_gain=0.30,
        bridge_gain=0.20,
        redundancy=0.10,
        token_cost=0.10,
        weights=w,
        bridge_lambda=1.0,
    )
    high = marginal_evidence_utility(
        evidence_gain=0.80,
        bridge_gain=0.20,
        redundancy=0.10,
        token_cost=0.10,
        weights=w,
        bridge_lambda=1.0,
    )
    assert high > low


def test_bridge_gain_respects_adaptive_lambda():
    w = MarginalUtilityWeights(evidence_gain=0.4, bridge_gain=0.4, redundancy=0.1, token_cost=0.1)
    lambda_low = resolve_adaptive_bridge_lambda(graph_dispersion=0.1, adaptive_enabled=True)
    lambda_high = resolve_adaptive_bridge_lambda(graph_dispersion=0.9, adaptive_enabled=True)
    score_low = marginal_evidence_utility(
        evidence_gain=0.4,
        bridge_gain=0.8,
        redundancy=0.1,
        token_cost=0.1,
        weights=w,
        bridge_lambda=lambda_low,
    )
    score_high = marginal_evidence_utility(
        evidence_gain=0.4,
        bridge_gain=0.8,
        redundancy=0.1,
        token_cost=0.1,
        weights=w,
        bridge_lambda=lambda_high,
    )
    assert lambda_high >= lambda_low
    assert score_high >= score_low


def test_redundancy_penalty_lowers_utility():
    w = MarginalUtilityWeights(evidence_gain=0.5, bridge_gain=0.2, redundancy=0.5, token_cost=0.1)
    low_overlap = marginal_evidence_utility(
        evidence_gain=0.6,
        bridge_gain=0.3,
        redundancy=0.1,
        token_cost=0.1,
        weights=w,
        bridge_lambda=1.0,
    )
    high_overlap = marginal_evidence_utility(
        evidence_gain=0.6,
        bridge_gain=0.3,
        redundancy=0.9,
        token_cost=0.1,
        weights=w,
        bridge_lambda=1.0,
    )
    assert low_overlap > high_overlap


def test_token_cost_is_soft_not_hard_core_drop():
    w = MarginalUtilityWeights(evidence_gain=0.7, bridge_gain=0.2, redundancy=0.1, token_cost=0.2)
    high_gain_high_cost = marginal_evidence_utility(
        evidence_gain=0.9,
        bridge_gain=0.4,
        redundancy=0.1,
        token_cost=0.9,
        weights=w,
        bridge_lambda=1.0,
    )
    low_gain_low_cost = marginal_evidence_utility(
        evidence_gain=0.4,
        bridge_gain=0.2,
        redundancy=0.1,
        token_cost=0.1,
        weights=w,
        bridge_lambda=1.0,
    )
    assert high_gain_high_cost > low_gain_low_cost

