import inspect

import effirag.sentence_contract_render as sentence_contract_render


def test_adaptive_span_relevance_beats_bridge_only_under_weak_prior():
    score_rel, feats_rel = sentence_contract_render._adaptive_support_span_score(
        span_text="OpenAI studies alignment",
        query_tokens={"openai", "alignment"},
        anchor_tokens={"openai"},
        bridge_tokens={"connectoralpha"},
        use_query_signal=True,
        use_anchor_signal=True,
        use_bridge_signal=True,
        bridge_mode="weak",
        soft_length_penalty_enabled=True,
        rank_index=0,
        total_items=3,
    )
    score_bridge_only, feats_bridge_only = sentence_contract_render._adaptive_support_span_score(
        span_text="ConnectorAlpha",
        query_tokens={"openai", "alignment"},
        anchor_tokens={"openai"},
        bridge_tokens={"connectoralpha"},
        use_query_signal=True,
        use_anchor_signal=True,
        use_bridge_signal=True,
        bridge_mode="weak",
        soft_length_penalty_enabled=True,
        rank_index=0,
        total_items=3,
    )

    assert feats_rel["query_hit"] > 0.0
    assert feats_bridge_only["bridge_hit"] > 0.0
    assert score_rel > score_bridge_only


def test_adaptive_span_soft_length_penalty_penalizes_long_span():
    short_score, short_feats = sentence_contract_render._adaptive_support_span_score(
        span_text="alignment proof",
        query_tokens={"alignment"},
        anchor_tokens=set(),
        bridge_tokens=set(),
        use_query_signal=True,
        use_anchor_signal=True,
        use_bridge_signal=False,
        bridge_mode="conditional",
        soft_length_penalty_enabled=True,
        rank_index=0,
        total_items=2,
    )
    long_score, long_feats = sentence_contract_render._adaptive_support_span_score(
        span_text="alignment " + ("token " * 40),
        query_tokens={"alignment"},
        anchor_tokens=set(),
        bridge_tokens=set(),
        use_query_signal=True,
        use_anchor_signal=True,
        use_bridge_signal=False,
        bridge_mode="conditional",
        soft_length_penalty_enabled=True,
        rank_index=0,
        total_items=2,
    )
    assert long_feats["length_penalty"] > short_feats["length_penalty"]
    assert long_score < short_score


def test_adaptive_span_no_length_penalty_ablation_disables_penalty():
    _score, feats = sentence_contract_render._adaptive_support_span_score(
        span_text="alignment " + ("token " * 50),
        query_tokens={"alignment"},
        anchor_tokens=set(),
        bridge_tokens=set(),
        use_query_signal=True,
        use_anchor_signal=True,
        use_bridge_signal=False,
        bridge_mode="conditional",
        soft_length_penalty_enabled=False,
        rank_index=0,
        total_items=2,
    )
    assert feats["length_penalty"] == 0.0


def test_adaptive_span_scoring_does_not_use_dataset_answer_or_gold_support():
    score_src = inspect.getsource(sentence_contract_render._adaptive_support_span_score).lower()
    pick_src = inspect.getsource(sentence_contract_render._pick_adaptive_support_span).lower()
    combined = score_src + "\n" + pick_src

    banned_terms = [
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
        "popqa",
        "supporting_facts",
        "gold_support",
        "answer",
    ]
    for term in banned_terms:
        assert term not in combined
