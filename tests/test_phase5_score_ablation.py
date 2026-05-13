from types import SimpleNamespace

import networkx as nx

from effirag.config import RetrievalConfig
from effirag.retrieval import (
    _apply_final_text_rerank,
    _phase2_pair_shortlist,
    _weighted_sum_active_components,
    run_effirag,
)
from effirag.run_rag import _build_parser
from effirag.types import ContextDocument, Sample


def _pair_cfg(**overrides):
    base = dict(
        pair_shortlist_topb=5,
        tau=4,
        ablation_no_pair_semantic=False,
        ablation_no_pair_bridge=False,
        score_component_trace_enabled=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _pair_fixture():
    g = nx.Graph()
    g.add_edge("a", "s")
    g.add_edge("b", "s")
    runs = [
        {
            "run_id": 1,
            "seeds": ["s"],
            "seed_score_map": {"s": 0.8},
            "anchor_scores": {"a": {"s": 0.6}, "b": {"s": 0.5}},
        }
    ]
    anchors = ["a", "b"]
    query_sim_map = {"s": 0.0}
    support_sim_map = {"s": 0.0}
    return runs, anchors, g, query_sim_map, support_sim_map


def test_phase5a_config_defaults_are_disabled():
    cfg = RetrievalConfig()
    assert cfg.ablation_no_pair_semantic is False
    assert cfg.ablation_no_pair_bridge is False
    assert cfg.ablation_no_final_text_rerank is False
    assert cfg.score_component_trace_enabled is False
    assert cfg.score_component_trace_topn == 20


def test_phase5a_cli_parser_accepts_flags():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--ablation-no-pair-semantic",
            "true",
            "--ablation-no-pair-bridge",
            "false",
            "--ablation-no-final-text-rerank",
            "true",
            "--score-component-trace-enabled",
            "true",
            "--score-component-trace-topn",
            "7",
        ]
    )
    assert args.ablation_no_pair_semantic == "true"
    assert args.ablation_no_pair_bridge == "false"
    assert args.ablation_no_final_text_rerank == "true"
    assert args.score_component_trace_enabled == "true"
    assert args.score_component_trace_topn == 7


def test_weighted_sum_active_components_no_disabled_matches_direct_sum():
    components = {
        "anchor_align": 0.6,
        "seed_strength": 0.8,
        "semantic_rel": 0.5,
        "dist_score": 0.8,
        "bridge_potential": 1.0,
    }
    weights = {
        "anchor_align": 0.30,
        "seed_strength": 0.25,
        "semantic_rel": 0.20,
        "dist_score": 0.15,
        "bridge_potential": 0.10,
    }
    score, effective = _weighted_sum_active_components(components, weights, disabled_keys=set())
    direct = sum(weights[k] * components[k] for k in weights.keys())
    assert abs(score - direct) < 1e-12
    assert set(effective.keys()) == set(weights.keys())


def test_weighted_sum_active_components_renormalizes_when_disabled():
    components = {
        "anchor_align": 0.6,
        "seed_strength": 0.8,
        "semantic_rel": 0.5,
        "dist_score": 0.8,
        "bridge_potential": 1.0,
    }
    weights = {
        "anchor_align": 0.30,
        "seed_strength": 0.25,
        "semantic_rel": 0.20,
        "dist_score": 0.15,
        "bridge_potential": 0.10,
    }
    score, effective = _weighted_sum_active_components(components, weights, disabled_keys={"semantic_rel"})
    assert "semantic_rel" not in effective
    expected = (
        0.375 * components["anchor_align"]
        + 0.3125 * components["seed_strength"]
        + 0.1875 * components["dist_score"]
        + 0.125 * components["bridge_potential"]
    )
    assert abs(score - expected) < 1e-12


def test_weighted_sum_active_components_all_disabled_raises():
    components = {"x": 1.0}
    weights = {"x": 1.0}
    try:
        _weighted_sum_active_components(components, weights, disabled_keys={"x"})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "disabled" in str(e).lower()


def test_pair_scoring_default_formula_path_exact():
    runs, anchors, g, qmap, smap = _pair_fixture()
    cfg = _pair_cfg()
    out = _phase2_pair_shortlist(runs, anchors, g, qmap, smap, cfg)
    item = next(x for x in out if x["anchor"] == "a" and x["seed"] == "s")
    expected = 0.30 * 0.6 + 0.25 * 0.8 + 0.20 * 0.5 + 0.15 * 0.8 + 0.10 * 1.0
    assert abs(float(item["pair_proxy_score"]) - expected) < 1e-12


def test_pair_scoring_ablation_no_pair_semantic():
    runs, anchors, g, qmap, smap = _pair_fixture()
    cfg = _pair_cfg(ablation_no_pair_semantic=True)
    out = _phase2_pair_shortlist(runs, anchors, g, qmap, smap, cfg)
    item = next(x for x in out if x["anchor"] == "a" and x["seed"] == "s")
    expected = 0.375 * 0.6 + 0.3125 * 0.8 + 0.1875 * 0.8 + 0.125 * 1.0
    assert abs(float(item["pair_proxy_score"]) - expected) < 1e-12


def test_pair_scoring_ablation_no_pair_bridge():
    runs, anchors, g, qmap, smap = _pair_fixture()
    cfg = _pair_cfg(ablation_no_pair_bridge=True)
    out = _phase2_pair_shortlist(runs, anchors, g, qmap, smap, cfg)
    item = next(x for x in out if x["anchor"] == "a" and x["seed"] == "s")
    expected = (1.0 / 3.0) * 0.6 + (5.0 / 18.0) * 0.8 + (2.0 / 9.0) * 0.5 + (1.0 / 6.0) * 0.8
    assert abs(float(item["pair_proxy_score"]) - expected) < 1e-12


def test_pair_scoring_ablation_no_pair_semantic_and_bridge():
    runs, anchors, g, qmap, smap = _pair_fixture()
    cfg = _pair_cfg(ablation_no_pair_semantic=True, ablation_no_pair_bridge=True)
    out = _phase2_pair_shortlist(runs, anchors, g, qmap, smap, cfg)
    item = next(x for x in out if x["anchor"] == "a" and x["seed"] == "s")
    expected = (3.0 / 7.0) * 0.6 + (5.0 / 14.0) * 0.8 + (3.0 / 14.0) * 0.8
    assert abs(float(item["pair_proxy_score"]) - expected) < 1e-12


def test_pair_score_trace_does_not_change_pair_ranking():
    runs, anchors, g, qmap, smap = _pair_fixture()
    cfg_plain = _pair_cfg(score_component_trace_enabled=False)
    cfg_trace = _pair_cfg(score_component_trace_enabled=True)
    out_plain = _phase2_pair_shortlist(runs, anchors, g, qmap, smap, cfg_plain)
    out_trace = _phase2_pair_shortlist(runs, anchors, g, qmap, smap, cfg_trace)
    assert [(x["anchor"], x["seed"]) for x in out_plain] == [(x["anchor"], x["seed"]) for x in out_trace]
    assert abs(float(out_plain[0]["pair_proxy_score"]) - float(out_trace[0]["pair_proxy_score"])) < 1e-12
    assert "pair_score_components" in out_trace[0]


def test_apply_final_text_rerank_default_path_calls_rerank(monkeypatch):
    called = {"n": 0}

    def _mock_rerank(**kwargs):
        called["n"] += 1
        return {
            "ranked_sentence_ids": list(kwargs["sentence_ids"]),
            "ranked_sentence_texts": list(kwargs["sentence_texts"]),
            "applied": True,
            "error": "",
            "rerank_topn": 2,
            "similarity_by_sentence_id": {},
            "fused_score_by_sentence_id": {},
        }

    monkeypatch.setattr("effirag.retrieval.rerank_sentences_by_embedding", _mock_rerank)
    cfg = SimpleNamespace(
        sentence_rerank_enabled=True,
        embedding_enabled=True,
        ablation_no_final_text_rerank=False,
        embedding_model_name="dummy",
        embedding_weight=0.35,
        embedding_rerank_topn=80,
        embedding_batch_size=16,
        embedding_max_length=128,
        embedding_text_max_chars=512,
    )
    _apply_final_text_rerank(
        question="q",
        selected_sentence_ids=["s1", "s2"],
        selected_sentences=["t1", "t2"],
        selected_sentence_score_map={"s1": 0.9, "s2": 0.8},
        cfg=cfg,
    )
    assert called["n"] == 1


def test_apply_final_text_rerank_ablation_skips_rerank(monkeypatch):
    called = {"n": 0}

    def _mock_rerank(**kwargs):
        called["n"] += 1
        return {}

    monkeypatch.setattr("effirag.retrieval.rerank_sentences_by_embedding", _mock_rerank)
    cfg = SimpleNamespace(
        sentence_rerank_enabled=True,
        embedding_enabled=True,
        ablation_no_final_text_rerank=True,
        embedding_model_name="dummy",
        embedding_weight=0.35,
        embedding_rerank_topn=80,
        embedding_batch_size=16,
        embedding_max_length=128,
        embedding_text_max_chars=512,
    )
    _, _, diag, calls, _ = _apply_final_text_rerank(
        question="q",
        selected_sentence_ids=["s1", "s2"],
        selected_sentences=["t1", "t2"],
        selected_sentence_score_map={"s1": 0.9, "s2": 0.8},
        cfg=cfg,
    )
    assert called["n"] == 0
    assert calls == 0
    assert diag.get("error") == "ablation_no_final_text_rerank"


def test_score_trace_enabled_adds_diag_without_selection_change():
    sample = Sample(
        qid="phase5a_t1",
        question="What city is the capital of France?",
        answer="Paris",
        contexts=[
            ContextDocument(
                title="Paris",
                sentences=[
                    "Paris is the capital of France.",
                    "Paris is known as the City of Light.",
                ],
            ),
            ContextDocument(title="France", sentences=["France is in Europe."]),
        ],
        supporting_facts=[("Paris", 0)],
    )
    cfg_plain = RetrievalConfig(
        embedding_enabled=False,
        sentence_rerank_enabled=False,
        score_component_trace_enabled=False,
        samples_per_anchor=2,
        max_anchors=2,
        seed_k=2,
        candidate_top_t=8,
    )
    cfg_trace = RetrievalConfig(
        embedding_enabled=False,
        sentence_rerank_enabled=False,
        score_component_trace_enabled=True,
        score_component_trace_topn=5,
        samples_per_anchor=2,
        max_anchors=2,
        seed_k=2,
        candidate_top_t=8,
    )
    r_plain = run_effirag(sample, cfg_plain)
    r_trace = run_effirag(sample, cfg_trace)
    assert list(r_plain.selected_sentence_ids) == list(r_trace.selected_sentence_ids)
    plain_corridors = [str((c or {}).get("corridor_id", "")) for c in (r_plain.corridors or [])]
    trace_corridors = [str((c or {}).get("corridor_id", "")) for c in (r_trace.corridors or [])]
    assert plain_corridors == trace_corridors
    assert "score_component_trace" not in (r_plain.diagnostics or {})
    assert bool((r_trace.diagnostics or {}).get("score_component_trace", {}).get("enabled", False)) is True
