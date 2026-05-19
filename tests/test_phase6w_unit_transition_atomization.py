from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import inspect
import sys

from effirag.render import render_context
from effirag.types import ContextDocument, RetrievalResult, Sample
from effirag.unified_acr_rcedr_selector import (
    _atomize_selected_evidence,
    _normalize_space,
    apply_unified_acr_rcedr_selection,
)
from effirag.utils import content_tokens


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import summarize_phase6w_unit_transition_atomization as phase6w_sum  # noqa: E402


def _cfg(**overrides):
    base = {
        "unified_acr_rcedr_enabled": True,
        "unified_acr_rcedr_mode": "greedy",
        "unified_acr_rcedr_beam_size": 3,
        "unified_acr_rcedr_max_atoms": 3,
        "unified_acr_rcedr_max_tokens": 220,
        "unified_acr_rcedr_hard_token_budget_enabled": True,
        "unified_acr_rcedr_use_answerability_gain": True,
        "unified_acr_rcedr_use_bridge_gain": True,
        "unified_acr_rcedr_use_redundancy_penalty": True,
        "unified_acr_rcedr_use_cost_penalty": False,
        "unified_acr_rcedr_lambda_bridge": 0.28,
        "unified_acr_rcedr_mu_redundancy": 0.22,
        "unified_acr_rcedr_answerability_weight": 1.0,
        "unified_acr_rcedr_atomization_enabled": True,
        "unified_acr_rcedr_atom_span_max_sentences": 2,
        "unified_acr_rcedr_length_penalty_weight": 0.04,
        "unified_acr_rcedr_chain_aware_enabled": False,
        "unified_acr_rcedr_role_aware_redundancy_enabled": False,
        "unified_acr_rcedr_role_balanced_enabled": False,
        "unified_acr_rcedr_redundancy_recalibrated_enabled": False,
        "unified_acr_rcedr_semantic_sufficiency_enabled": False,
        "unified_acr_rcedr_semantic_answerability_weight": 0.20,
        "unified_acr_rcedr_semantic_use_as_prior_only": True,
        "unified_acr_rcedr_chain_gain_weight": 0.15,
        "unified_acr_rcedr_role_balance_weight": 0.08,
        "unified_acr_rcedr_role_balance_max_gain_per_step": 0.08,
        "unified_acr_rcedr_role_balance_max_token_jaccard": 0.45,
        "unified_acr_rcedr_role_balance_missing_only": True,
        "unified_acr_rcedr_role_redundancy_relax": 0.75,
        "unified_acr_rcedr_role_redundancy_max_overlap": 0.45,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _sample_inputs():
    question = "Which region borders the country where Alpha city is located?"
    sentence_ids = ["A::0", "B::0", "C::0"]
    sentence_texts = [
        "Alpha city is in country X. The city is in Europe.",
        "Country X borders Beta region. Beta region is mountainous.",
        "Unrelated sports trivia sentence.",
    ]
    table = {
        "A::0": {"is_main_candidate": True, "corridor_ids": ["c1"], "locality_score": 0.7},
        "B::0": {
            "is_support_candidate": True,
            "is_connector_adjacent": True,
            "bridge_gain": 0.9,
            "corridor_ids": ["c1"],
            "locality_score": 0.8,
        },
        "C::0": {"corridor_ids": ["c2"], "locality_score": 0.1},
    }
    return question, sentence_ids, sentence_texts, table


def _atomize_text(text: str, *, atomization_enabled: bool, span_cap: int) -> str:
    q_tokens = {str(tok) for tok in content_tokens("Which region borders country X")}
    q_entities = {"alpha", "beta"}
    rel_tokens = {"borders", "country", "region"}
    bridge_tokens = {"country", "region"}
    atoms = _atomize_selected_evidence(
        sentence_ids=["X::0"],
        sentence_texts=[text],
        question_tokens=q_tokens,
        question_entities=q_entities,
        relation_tokens=rel_tokens,
        bridge_tokens=bridge_tokens,
        expected_answer_type="generic",
        sentence_feature_table={"X::0": {"corridor_ids": ["c1"], "locality_score": 0.5}},
        atomization_enabled=atomization_enabled,
        max_span_sentences=span_cap,
        length_penalty_weight=0.04,
        answerability_weight=1.0,
        lambda_bridge=0.28,
        use_answerability_gain=True,
        use_bridge_gain=True,
    )
    return atoms[0].text


def _sample_and_retrieval():
    docs = [
        ContextDocument("Doc A", ["Alpha city is in country X.", "Country X borders Beta region."]),
        ContextDocument("Doc B", ["Bridge clue sentence.", "Answer-bearing clue sentence."]),
        ContextDocument("Doc C", ["Distractor sentence."]),
    ]
    sample = Sample(
        qid="q_phase6w_render_package",
        question="Which region borders the country where Alpha city is located?",
        answer="Beta region",
        contexts=docs,
    )
    retrieval = RetrievalResult(
        sample_id="q_phase6w_render_package",
        method="effirag",
        anchors=["Alpha city", "country X"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["Doc A::0", "Doc A::1", "Doc B::0", "Doc C::0"],
        selected_sentences=[
            "Alpha city is in country X.",
            "Country X borders Beta region.",
            "Bridge clue sentence.",
            "Distractor sentence.",
        ],
        corridors=[
            {
                "corridor_id": "c1",
                "corridor_score": 1.0,
                "main_path_sentence_ids": ["Doc A::0", "Doc A::1"],
                "support_sentence_ids": ["Doc B::0"],
                "connector_adjacent_sentence_ids": ["Doc B::0"],
                "sentence_score_map": {
                    "Doc A::0": 0.9,
                    "Doc A::1": 0.85,
                    "Doc B::0": 0.7,
                    "Doc C::0": 0.1,
                },
            }
        ],
        diagnostics={},
    )
    return sample, retrieval


def test_baseline_unchanged_with_default_atomization_flag():
    q, ids, texts, table = _sample_inputs()
    out1_ids, out1_texts, _ = apply_unified_acr_rcedr_selection(
        question_text=q,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_cfg(),
    )
    out2_ids, out2_texts, _ = apply_unified_acr_rcedr_selection(
        question_text=q,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_cfg(unified_acr_rcedr_atomization_enabled=True),
    )
    assert out1_ids == out2_ids
    assert out1_texts == out2_texts


def test_no_atomization_disables_span_splitting():
    text = "Alpha city is in country X. Country X borders Beta region."
    atomized = _atomize_text(text, atomization_enabled=True, span_cap=1)
    non_atomized = _atomize_text(text, atomization_enabled=False, span_cap=1)
    assert non_atomized == _normalize_space(text)
    assert atomized != ""
    assert non_atomized != ""
    assert atomized != non_atomized


def test_atom_span3_caps_atom_sentence_window():
    text = "S1. S2. S3. S4."
    span3 = _atomize_text(text, atomization_enabled=True, span_cap=3)
    assert len([x for x in span3.split(".") if x.strip()]) <= 3


def test_render_package_keeps_selected_core_and_respects_budget():
    sample, retrieval = _sample_and_retrieval()
    common = dict(
        max_context_sentences=8,
        render_mode="corridor_aware_flat",
        order_strategy="score",
        selector_aware_render_enabled=True,
        render_selected_only=True,
        render_include_source_titles="minimal",
        render_include_metadata="minimal",
        render_enforce_actual_prompt_budget=True,
        max_prompt_tokens=128,
    )
    off = render_context(
        sample,
        retrieval,
        legacy_contract_top_slice_reorder_enabled=False,
        legacy_contract_minimal_package_enabled=False,
        **common,
    )
    on = render_context(
        sample,
        retrieval,
        legacy_contract_top_slice_reorder_enabled=False,
        legacy_contract_minimal_package_enabled=True,
        legacy_contract_max_extra_sentences_per_selected=1,
        legacy_contract_max_total_extra_sentences=2,
        legacy_contract_preserve_token_budget=True,
        **common,
    )
    assert set(off.sentence_ids).issubset(set(on.sentence_ids))
    assert len(on.sentence_ids) <= len(off.sentence_ids) + 2


def test_render_package_emits_skip_reasons_when_metadata_missing():
    sample = Sample(
        qid="q_skip",
        question="What is the evidence?",
        answer="evidence",
        contexts=[ContextDocument("DocZ", ["Compact chunk evidence.", "Neighbor chunk evidence."])],
    )
    retrieval = RetrievalResult(
        sample_id="q_skip",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["chunk::DocZ::0"],
        selected_sentences=["Compact chunk evidence."],
        candidate_sentence_ids=["chunk::DocZ::0", "chunk::DocZ::1"],
        candidate_sentences=["Compact chunk evidence.", "Neighbor chunk evidence."],
        corridors=[],
        diagnostics={},
    )
    rendered = render_context(
        sample,
        retrieval,
        max_context_sentences=4,
        render_mode="corridor_aware_flat",
        selector_aware_render_enabled=True,
        render_selected_only=True,
        legacy_contract_top_slice_reorder_enabled=False,
        legacy_contract_minimal_package_enabled=True,
        legacy_contract_max_extra_sentences_per_selected=1,
        legacy_contract_max_total_extra_sentences=2,
        legacy_contract_preserve_token_budget=True,
    )
    meta = dict(rendered.metadata or {})
    assert isinstance(meta.get("minimal_package_skip_reason_counts", {}), dict)
    assert isinstance(meta.get("minimal_package_dominant_skip_reason", ""), str)


def test_query_type_classifier_is_offline_only_helper():
    assert phase6w_sum._infer_query_type("Who was born first, A or B?") in {"comparison", "date_or_number", "person_location_org"}
    assert phase6w_sum._infer_query_type("Is Alpha city in country X?") == "yes_no"
    src = inspect.getsource(phase6w_sum._infer_query_type)
    assert "dataset" not in src.lower()


def test_runner_enforces_paired_query_id_check():
    runner = (ROOT / "scripts" / "run_phase6w_unit_transition_atomization_4proc.sh").read_text(encoding="utf-8")
    assert "check_sample_ids_match_baseline" in runner
    assert "wait_for_sample_ids" in runner
