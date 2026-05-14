from __future__ import annotations

from dataclasses import dataclass

from effirag.evidence_density_rerank import rerank_evidence_density


@dataclass
class DummyCfg:
    evidence_density_rerank_enabled: bool = True
    bridge_path_utility_enabled: bool = True
    coverage_gain_enabled: bool = True
    redundancy_penalty_enabled: bool = True
    token_cost_penalty_enabled: bool = True
    density_budget_awareness_enabled: bool = True
    density_semantic_weight: float = 0.42
    density_bridge_path_weight: float = 0.24
    density_coverage_weight: float = 0.24
    density_redundancy_weight: float = 0.18
    density_token_cost_weight: float = 0.22
    max_selected_candidates: int = 24
    max_rendered_candidates: int = 24
    max_rendered_tokens: int = 340
    min_render_topn: int = 1
    dataset: str = "hotpotqa"


def _feat(
    base: float,
    *,
    rank: int = 10,
    corridor_score: float = 0.0,
    main: bool = False,
    support: bool = False,
    connector: bool = False,
    corridors: list[str] | None = None,
):
    return {
        "base_retrieval_score": float(base),
        "best_corridor_rank": int(rank),
        "best_corridor_score": float(corridor_score),
        "is_main_candidate": bool(main),
        "is_support_candidate": bool(support),
        "is_connector_adjacent": bool(connector),
        "corridor_ids": list(corridors or []),
    }


def _run(question, ids, texts, feats, cfg=None):
    return rerank_evidence_density(
        question_text=question,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=feats,
        cfg=cfg or DummyCfg(),
    )


def test_rerank_is_deterministic():
    cfg = DummyCfg()
    ids = ["s1", "s2", "s3"]
    texts = [
        "alpha bridge clue",
        "beta support clue",
        "gamma extra clue",
    ]
    feats = {
        "s1": _feat(0.91, main=True, rank=1, corridor_score=0.9, corridors=["c1"]),
        "s2": _feat(0.89, support=True, rank=2, corridor_score=0.8, corridors=["c2"]),
        "s3": _feat(0.80, rank=3, corridor_score=0.7, corridors=["c3"]),
    }

    a = _run("alpha beta bridge", ids, texts, feats, cfg=cfg)
    b = _run("alpha beta bridge", ids, texts, feats, cfg=cfg)
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert a[2]["score_trace"] == b[2]["score_trace"]


def test_coverage_gain_promotes_new_query_coverage():
    cfg = DummyCfg(
        density_semantic_weight=0.25,
        density_bridge_path_weight=0.0,
        density_coverage_weight=1.0,
        density_redundancy_weight=0.0,
        density_token_cost_weight=0.0,
    )
    ids = ["s1", "s2", "s3"]
    texts = [
        "alpha base fact",
        "alpha duplicate fact",
        "beta novel fact",
    ]
    feats = {
        "s1": _feat(1.00, rank=1, corridors=["c1"]),
        "s2": _feat(0.85, rank=2, corridors=["c1"]),
        "s3": _feat(0.85, rank=2, corridors=["c2"]),
    }

    selected_ids, _, _ = _run("alpha beta", ids, texts, feats, cfg=cfg)
    assert selected_ids[0] == "s1"
    assert "s3" in selected_ids[:2]


def test_redundancy_penalty_avoids_duplicate_candidate():
    cfg = DummyCfg(
        density_semantic_weight=0.10,
        density_bridge_path_weight=0.0,
        density_coverage_weight=0.20,
        density_redundancy_weight=1.20,
        density_token_cost_weight=0.0,
        max_selected_candidates=2,
        max_rendered_candidates=2,
    )
    ids = ["s1", "s2", "s3"]
    texts = [
        "alpha duplicate evidence segment",
        "alpha duplicate evidence segment",
        "beta distinct support clue",
    ]
    feats = {
        "s1": _feat(0.95, rank=1, corridors=["c1"]),
        "s2": _feat(0.94, rank=2, corridors=["c1"]),
        "s3": _feat(0.80, rank=3, corridors=["c2"]),
    }

    selected_ids, _, _ = _run("alpha beta", ids, texts, feats, cfg=cfg)
    assert "s1" in selected_ids
    assert "s2" not in selected_ids
    assert "s3" in selected_ids


def test_token_cost_penalty_prefers_compact_candidate():
    cfg = DummyCfg(
        density_semantic_weight=1.0,
        density_bridge_path_weight=0.0,
        density_coverage_weight=0.0,
        density_redundancy_weight=0.0,
        density_token_cost_weight=1.2,
        max_selected_candidates=1,
        max_rendered_candidates=1,
    )
    ids = ["long", "short"]
    texts = [
        " ".join([f"longtok{i}" for i in range(60)]),
        "short compact clue tokens",
    ]
    feats = {
        "long": _feat(1.0, rank=1),
        "short": _feat(1.0, rank=1),
    }

    selected_ids, _, _ = _run("compact clue", ids, texts, feats, cfg=cfg)
    assert selected_ids == ["short"]


def test_bridge_path_utility_can_preserve_connector_candidate():
    cfg = DummyCfg(
        density_semantic_weight=0.10,
        density_bridge_path_weight=1.0,
        density_coverage_weight=0.0,
        density_redundancy_weight=0.0,
        density_token_cost_weight=0.0,
        max_selected_candidates=1,
        max_rendered_candidates=1,
    )
    ids = ["plain", "bridge"]
    texts = [
        "plain support sentence",
        "connector bridge sentence",
    ]
    feats = {
        "plain": _feat(0.95, rank=1, corridor_score=0.1, main=True, support=False, connector=False),
        "bridge": _feat(0.80, rank=1, corridor_score=0.9, main=True, support=True, connector=True),
    }

    selected_ids, _, diag = _run("connector support", ids, texts, feats, cfg=cfg)
    assert selected_ids == ["bridge"]
    assert diag["bridge_path_utility_enabled"] is True


def test_selection_respects_candidate_and_token_budgets():
    cfg = DummyCfg(
        max_selected_candidates=2,
        max_rendered_candidates=2,
        max_rendered_tokens=8,
        min_render_topn=1,
        density_semantic_weight=0.6,
        density_bridge_path_weight=0.0,
        density_coverage_weight=0.4,
        density_redundancy_weight=0.0,
        density_token_cost_weight=0.2,
    )
    ids = ["s1", "s2", "s3"]
    texts = [
        "alpha one two",  # 3
        "beta one two three",  # 4
        "gamma one two three four five",  # 6
    ]
    feats = {
        "s1": _feat(0.95, rank=1, corridors=["c1"]),
        "s2": _feat(0.90, rank=2, corridors=["c2"]),
        "s3": _feat(0.89, rank=3, corridors=["c3"]),
    }

    selected_ids, _, diag = _run("alpha beta gamma", ids, texts, feats, cfg=cfg)
    assert len(selected_ids) <= 2
    assert int(diag["tokens_after"]) <= 8
    assert int(diag["selected_count_after"]) <= 2


def test_rerank_is_dataset_independent():
    ids = ["s1", "s2", "s3"]
    texts = [
        "alpha bridge clue",
        "beta support clue",
        "gamma extra clue",
    ]
    feats = {
        "s1": _feat(0.91, main=True, rank=1, corridor_score=0.9, corridors=["c1"]),
        "s2": _feat(0.89, support=True, rank=2, corridor_score=0.8, corridors=["c2"]),
        "s3": _feat(0.80, rank=3, corridor_score=0.7, corridors=["c3"]),
    }
    hotpot_cfg = DummyCfg(dataset="hotpotqa")
    wiki_cfg = DummyCfg(dataset="2wikimultihopqa")

    hotpot = _run("alpha beta bridge", ids, texts, feats, cfg=hotpot_cfg)
    wiki = _run("alpha beta bridge", ids, texts, feats, cfg=wiki_cfg)
    assert hotpot[0] == wiki[0]
    assert hotpot[2]["score_trace"] == wiki[2]["score_trace"]
