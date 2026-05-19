from types import SimpleNamespace

from effirag.unified_acr_rcedr_selector import apply_unified_acr_rcedr_selection


def _base_cfg(**overrides):
    cfg = {
        "unified_acr_rcedr_enabled": True,
        "unified_acr_rcedr_mode": "greedy",
        "unified_acr_rcedr_beam_size": 3,
        "unified_acr_rcedr_max_atoms": 2,
        "unified_acr_rcedr_max_tokens": 120,
        "unified_acr_rcedr_hard_token_budget_enabled": True,
        "unified_acr_rcedr_use_answerability_gain": True,
        "unified_acr_rcedr_use_bridge_gain": True,
        "unified_acr_rcedr_use_redundancy_penalty": True,
        "unified_acr_rcedr_use_cost_penalty": False,
        "unified_acr_rcedr_lambda_bridge": 0.28,
        "unified_acr_rcedr_mu_redundancy": 0.22,
        "unified_acr_rcedr_answerability_weight": 1.0,
        "unified_acr_rcedr_atom_span_max_sentences": 2,
        "unified_acr_rcedr_length_penalty_weight": 0.04,
        "unified_acr_rcedr_chain_aware_enabled": False,
        "unified_acr_rcedr_role_aware_redundancy_enabled": False,
        "unified_acr_rcedr_chain_gain_weight": 0.15,
        "unified_acr_rcedr_role_redundancy_relax": 0.5,
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _sample_inputs():
    sentence_ids = ["A::0", "B::0", "C::0"]
    sentence_texts = [
        "Alpha City is in Country X.",
        "Country X borders Beta Region and links trade routes.",
        "Unrelated trivia about sports finals.",
    ]
    table = {
        "A::0": {
            "is_main_candidate": True,
            "is_connector_adjacent": False,
            "is_support_candidate": True,
            "bridge_gain": 0.2,
            "corridor_ids": ["c1"],
            "locality_score": 0.4,
        },
        "B::0": {
            "is_main_candidate": False,
            "is_connector_adjacent": True,
            "is_support_candidate": True,
            "bridge_gain": 0.9,
            "corridor_ids": ["c1"],
            "locality_score": 0.8,
        },
        "C::0": {
            "is_main_candidate": False,
            "is_connector_adjacent": False,
            "is_support_candidate": False,
            "bridge_gain": 0.0,
            "corridor_ids": ["c2"],
            "locality_score": 0.1,
        },
    }
    question = "Which region borders the country where Alpha City is located?"
    return question, sentence_ids, sentence_texts, table


def test_chain_aware_diag_fields_present_and_disabled_by_default():
    q, ids, texts, table = _sample_inputs()
    out_ids, out_texts, diag = apply_unified_acr_rcedr_selection(
        question_text=q,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_base_cfg(),
    )

    assert len(out_ids) >= 1
    assert len(out_texts) == len(out_ids)
    assert diag.get("applied") is True
    assert diag.get("chain_aware_enabled") is False
    assert diag.get("role_aware_redundancy_enabled") is False
    assert "chain_gain_total_avg" in diag
    assert "role_aware_redundancy_applied_avg" in diag


def test_chain_aware_can_be_enabled_without_schema_break():
    q, ids, texts, table = _sample_inputs()
    out_ids, out_texts, diag = apply_unified_acr_rcedr_selection(
        question_text=q,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_base_cfg(
            unified_acr_rcedr_chain_aware_enabled=True,
            unified_acr_rcedr_role_aware_redundancy_enabled=True,
            unified_acr_rcedr_chain_gain_weight=0.15,
            unified_acr_rcedr_role_redundancy_relax=0.5,
        ),
    )

    assert len(out_ids) >= 1
    assert len(out_texts) == len(out_ids)
    assert diag.get("applied") is True
    assert diag.get("chain_aware_enabled") is True
    assert diag.get("role_aware_redundancy_enabled") is True
    assert diag.get("chain_gain_weight") == 0.15
    assert diag.get("role_redundancy_relax") == 0.5
    assert isinstance(diag.get("chain_gain_total_avg"), float)
    assert isinstance(diag.get("role_aware_redundancy_applied_avg"), float)
