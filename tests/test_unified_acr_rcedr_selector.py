from __future__ import annotations

import inspect
from types import SimpleNamespace

import effirag.unified_acr_rcedr_selector as unified_selector


def _cfg(**overrides):
    base = {
        "unified_acr_rcedr_enabled": True,
        "unified_acr_rcedr_mode": "greedy",
        "unified_acr_rcedr_beam_size": 3,
        "unified_acr_rcedr_max_atoms": 4,
        "unified_acr_rcedr_max_tokens": 14,
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
        "unified_acr_rcedr_log_diagnostics": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_unified_selector_respects_hard_budget_with_first_atom_fallback():
    ids = ["DocA::0", "DocB::0"]
    texts = [
        "OpenAI founded the lab in 2015 and established the first research roadmap with many details.",
        "Background sentence unrelated to the founding event.",
    ]
    feature_table = {
        "DocA::0": {"corridor_ids": ["c1"], "is_main_candidate": True, "locality_score": 0.9},
        "DocB::0": {"corridor_ids": ["c2"], "is_support_candidate": False, "locality_score": 0.1},
    }
    selected_ids, selected_texts, diag = unified_selector.apply_unified_acr_rcedr_selection(
        question_text="Who founded the lab in 2015?",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=feature_table,
        cfg=_cfg(unified_acr_rcedr_max_tokens=6),
    )

    assert diag["applied"] is True
    assert len(selected_ids) == len(selected_texts) >= 1
    assert selected_ids[0] == "DocA::0"
    assert int(diag["selected_tokens"]) >= 6
    assert diag["hard_token_budget_enabled"] is True


def test_bridge_gain_toggle_changes_second_atom_selection():
    ids = ["DocA::0", "DocB::0"]
    texts = [
        "The lab was founded by Ada.",
        "ConnectorAlpha bridge sentence with structural context.",
    ]
    feature_table = {
        "DocA::0": {"corridor_ids": ["c1"], "is_main_candidate": True, "locality_score": 0.9},
        "DocB::0": {
            "corridor_ids": ["c1"],
            "is_connector_adjacent": True,
            "bridge_gain": 1.0,
            "locality_score": 0.7,
        },
    }

    with_bridge_ids, _, with_bridge_diag = unified_selector.apply_unified_acr_rcedr_selection(
        question_text="Who founded the lab?",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=feature_table,
        cfg=_cfg(unified_acr_rcedr_use_bridge_gain=True),
    )
    no_bridge_ids, _, no_bridge_diag = unified_selector.apply_unified_acr_rcedr_selection(
        question_text="Who founded the lab?",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=feature_table,
        cfg=_cfg(unified_acr_rcedr_use_bridge_gain=False),
    )

    assert with_bridge_diag["use_bridge_gain"] is True
    assert no_bridge_diag["use_bridge_gain"] is False
    assert len(with_bridge_ids) >= len(no_bridge_ids)
    assert "DocA::0" in with_bridge_ids
    assert "DocA::0" in no_bridge_ids


def test_unified_selector_code_does_not_use_dataset_name():
    src = inspect.getsource(unified_selector.apply_unified_acr_rcedr_selection).lower()
    for banned in ("hotpotqa", "2wikimultihopqa", "musique", "popqa"):
        assert banned not in src
