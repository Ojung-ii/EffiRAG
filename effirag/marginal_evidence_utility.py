from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(value))))


@dataclass(frozen=True)
class MarginalUtilityWeights:
    evidence_gain: float = 0.52
    bridge_gain: float = 0.28
    redundancy: float = 0.22
    token_cost: float = 0.14


def resolve_adaptive_bridge_lambda(
    *,
    graph_dispersion: float,
    base: float = 1.0,
    alpha: float = 0.30,
    minimum: float = 0.85,
    maximum: float = 1.35,
    adaptive_enabled: bool = True,
) -> float:
    if not adaptive_enabled:
        return clamp(float(base), float(minimum), float(maximum))
    raw = float(base) * float(1.0 + float(alpha) * max(0.0, float(graph_dispersion)))
    return clamp(raw, float(minimum), float(maximum))


def marginal_evidence_utility(
    *,
    evidence_gain: float,
    bridge_gain: float,
    redundancy: float,
    token_cost: float,
    weights: MarginalUtilityWeights,
    bridge_lambda: float = 1.0,
    redundancy_enabled: bool = True,
    cost_enabled: bool = True,
    density_first: bool = False,
) -> float:
    e = float(max(0.0, min(1.0, float(evidence_gain))))
    b = float(max(0.0, min(1.0, float(bridge_gain))))
    r = float(max(0.0, min(1.0, float(redundancy))))
    c = float(max(0.0, min(1.0, float(token_cost))))

    if density_first:
        # Failure-hypothesis ablation: over-prioritize compactness.
        c = clamp(c + 0.20, 0.0, 1.0)
        e = clamp(e * 0.85, 0.0, 1.0)

    score = float(weights.evidence_gain) * e + float(bridge_lambda) * float(weights.bridge_gain) * b
    if redundancy_enabled:
        score -= float(weights.redundancy) * r
    if cost_enabled:
        score -= float(weights.token_cost) * c
    return float(score)

