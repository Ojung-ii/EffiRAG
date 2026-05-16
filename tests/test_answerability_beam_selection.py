from types import SimpleNamespace

from effirag.answerability_selection import apply_answerability_constrained_selection


def _cfg(mode: str):
    return SimpleNamespace(
        answerability_selection_enabled=True,
        answerability_selection_mode=mode,
        answerability_selection_beam_size=3,
        answerability_selection_max_atoms=4,
        answerability_selection_max_tokens=28,
        answerability_selection_hard_token_budget_enabled=True,
        answerability_selection_use_answerability=True,
        answerability_selection_use_structure=True,
        answerability_selection_use_noise_penalty=True,
        answerability_selection_use_cost_penalty=True,
        answerability_selection_ordering_enabled=True,
        answerability_selection_atom_span_max_sentences=2,
        answerability_selection_structure_weight=0.65,
        answerability_selection_noise_weight=0.55,
        answerability_selection_cost_weight=0.35,
        answerability_selection_length_penalty_weight=0.04,
        answerability_selection_log_diagnostics=True,
    )


def test_beam3_finds_score_no_worse_than_greedy_and_is_deterministic():
    ids = ["DocA::0", "DocB::0", "DocC::0", "DocD::0"]
    texts = [
        "OpenAI founded the lab in 2015.",
        "ConnectorAlpha links OpenAI and the target lab.",
        "The target lab is in California.",
        "Background note with weak overlap.",
    ]
    feature_table = {
        "DocA::0": {"corridor_ids": ["c1"], "is_main_candidate": True, "is_support_candidate": False, "is_connector_adjacent": False, "locality_score": 1.0},
        "DocB::0": {"corridor_ids": ["c1"], "is_main_candidate": False, "is_support_candidate": True, "is_connector_adjacent": True, "locality_score": 0.9},
        "DocC::0": {"corridor_ids": ["c1"], "is_main_candidate": False, "is_support_candidate": True, "is_connector_adjacent": False, "locality_score": 0.7},
        "DocD::0": {"corridor_ids": ["c4"], "is_main_candidate": False, "is_support_candidate": False, "is_connector_adjacent": False, "locality_score": 0.2},
    }

    greedy_ids, _, greedy_diag = apply_answerability_constrained_selection(
        question_text="Who founded the target lab and what is the linked path?",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=feature_table,
        cfg=_cfg("greedy"),
    )
    beam_ids_a, _, beam_diag_a = apply_answerability_constrained_selection(
        question_text="Who founded the target lab and what is the linked path?",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=feature_table,
        cfg=_cfg("beam3"),
    )
    beam_ids_b, _, beam_diag_b = apply_answerability_constrained_selection(
        question_text="Who founded the target lab and what is the linked path?",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=feature_table,
        cfg=_cfg("beam3"),
    )

    assert greedy_diag["applied"] is True
    assert beam_diag_a["applied"] is True
    assert float(beam_diag_a["final_score"]) >= float(greedy_diag["final_score"]) - 1e-9
    assert beam_ids_a == beam_ids_b
    assert float(beam_diag_a["final_score"]) == float(beam_diag_b["final_score"])
