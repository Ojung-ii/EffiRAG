from types import SimpleNamespace

from effirag.answerability_selection import apply_answerability_constrained_selection


def _cfg(**overrides):
    base = {
        "answerability_selection_enabled": True,
        "answerability_selection_mode": "greedy",
        "answerability_selection_beam_size": 3,
        "answerability_selection_max_atoms": 4,
        "answerability_selection_max_tokens": 18,
        "answerability_selection_hard_token_budget_enabled": True,
        "answerability_selection_use_answerability": True,
        "answerability_selection_use_structure": True,
        "answerability_selection_use_noise_penalty": True,
        "answerability_selection_use_cost_penalty": True,
        "answerability_selection_ordering_enabled": True,
        "answerability_selection_atom_span_max_sentences": 2,
        "answerability_selection_structure_weight": 0.65,
        "answerability_selection_noise_weight": 0.55,
        "answerability_selection_cost_weight": 0.35,
        "answerability_selection_length_penalty_weight": 0.04,
        "answerability_selection_log_diagnostics": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_greedy_selection_prefers_high_marginal_gain_and_respects_budget():
    ids = ["DocA::0", "DocB::1", "DocA::2", "DocC::0"]
    texts = [
        "OpenAI founded the lab in 2015.",
        "ConnectorAlpha links OpenAI to lab.",
        "OpenAI founded the lab in 2015.",  # redundant with DocA::0
        "Generic background sentence with little relevance.",
    ]
    feature_table = {
        "DocA::0": {"corridor_ids": ["c1"], "is_main_candidate": True, "is_support_candidate": False, "is_connector_adjacent": False, "locality_score": 1.0},
        "DocB::1": {"corridor_ids": ["c1"], "is_main_candidate": False, "is_support_candidate": True, "is_connector_adjacent": True, "locality_score": 0.8},
        "DocA::2": {"corridor_ids": ["c1"], "is_main_candidate": True, "is_support_candidate": False, "is_connector_adjacent": False, "locality_score": 0.6},
        "DocC::0": {"corridor_ids": ["c3"], "is_main_candidate": False, "is_support_candidate": False, "is_connector_adjacent": False, "locality_score": 0.2},
    }

    selected_ids, selected_texts, diag = apply_answerability_constrained_selection(
        question_text="Who founded the lab connected through ConnectorAlpha?",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=feature_table,
        cfg=_cfg(),
    )

    assert diag["applied"] is True
    assert "DocA::0" in selected_ids
    assert "DocB::1" in selected_ids
    assert "DocA::2" not in selected_ids  # redundancy penalty should suppress this duplicate
    assert int(diag["selected_tokens"]) <= 18
    assert len(selected_ids) == len(selected_texts)


def test_high_answerability_signal_is_not_removed_by_cost_only():
    ids = ["DocA::0", "DocX::0"]
    texts = [
        "OpenAI founded the lab in 2015 with explicit answer-bearing details.",
        "background filler filler filler filler filler filler",
    ]
    feature_table = {
        "DocA::0": {"corridor_ids": ["c1"], "is_main_candidate": True, "is_support_candidate": False, "is_connector_adjacent": False, "locality_score": 1.0},
        "DocX::0": {"corridor_ids": ["c2"], "is_main_candidate": False, "is_support_candidate": False, "is_connector_adjacent": False, "locality_score": 0.3},
    }
    selected_ids, _, _ = apply_answerability_constrained_selection(
        question_text="Who founded the lab?",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=feature_table,
        cfg=_cfg(answerability_selection_max_tokens=10),
    )
    assert "DocA::0" in selected_ids
