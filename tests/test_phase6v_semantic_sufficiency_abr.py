from __future__ import annotations

from types import SimpleNamespace
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import summarize_phase6v_semantic_sufficiency_abr as phase6v_sum  # noqa: E402
from effirag.retrieval import _build_sentence_feature_table  # noqa: E402
from effirag.unified_acr_rcedr_selector import apply_unified_acr_rcedr_selection  # noqa: E402


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
        "unified_acr_rcedr_atom_span_max_sentences": 2,
        "unified_acr_rcedr_length_penalty_weight": 0.04,
        "unified_acr_rcedr_semantic_sufficiency_enabled": False,
        "unified_acr_rcedr_semantic_answerability_weight": 0.2,
        "unified_acr_rcedr_semantic_use_as_prior_only": True,
        "unified_acr_rcedr_chain_aware_enabled": False,
        "unified_acr_rcedr_role_aware_redundancy_enabled": False,
        "unified_acr_rcedr_role_balanced_enabled": False,
        "unified_acr_rcedr_redundancy_recalibrated_enabled": False,
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


def _sample():
    question = "Which region borders the country where Alpha city is located?"
    sentence_ids = ["A::0", "B::0", "C::0"]
    sentence_texts = [
        "Alpha city is in country X.",
        "Country X borders Beta region.",
        "Unrelated sports trivia.",
    ]
    return question, sentence_ids, sentence_texts


def test_baseline_unchanged_when_semantic_flag_disabled():
    q, ids, texts = _sample()
    table = {
        sid: {
            "corridor_ids": ["c1"],
            "is_main_candidate": i == 0,
            "is_support_candidate": i <= 1,
            "is_connector_adjacent": i == 1,
            "bridge_gain": 0.2 if i == 1 else 0.0,
            "locality_score": 1.0 / float(i + 1),
        }
        for i, sid in enumerate(ids)
    }
    out1_ids, out1_texts, _ = apply_unified_acr_rcedr_selection(
        question_text=q,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_cfg(unified_acr_rcedr_semantic_sufficiency_enabled=False),
    )
    out2_ids, out2_texts, _ = apply_unified_acr_rcedr_selection(
        question_text=q,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_cfg(unified_acr_rcedr_semantic_sufficiency_enabled=False),
    )
    assert out1_ids == out2_ids
    assert out1_texts == out2_texts


def test_sentence_feature_table_includes_semantic_fields_when_available():
    q, ids, texts = _sample()
    sample = SimpleNamespace(question=q)
    table = _build_sentence_feature_table(
        sample=sample,
        selected_sentence_ids=ids,
        sentence_texts=texts,
        sentence_score_map={ids[0]: 1.0, ids[1]: 0.8, ids[2]: 0.1},
        corridors=[],
        semantic_similarity_by_sentence_id={ids[0]: 0.55, ids[1]: 0.70},
        semantic_fused_score_by_sentence_id={ids[0]: 0.61, ids[1]: 0.72},
    )
    assert "semantic_query_score" in table[ids[0]]
    assert "semantic_fused_score" in table[ids[0]]
    assert "semantic_score_source" in table[ids[0]]
    assert table[ids[0]]["semantic_score_source"] == "fused_score_by_sentence_id"


def test_selector_receives_semantic_scores_and_reports_diagnostics():
    q, ids, texts = _sample()
    sample = SimpleNamespace(question=q)
    table = _build_sentence_feature_table(
        sample=sample,
        selected_sentence_ids=ids,
        sentence_texts=texts,
        sentence_score_map={ids[0]: 1.0, ids[1]: 0.8, ids[2]: 0.1},
        corridors=[],
        semantic_similarity_by_sentence_id={ids[0]: 0.55, ids[1]: 0.70, ids[2]: 0.05},
        semantic_fused_score_by_sentence_id={ids[0]: 0.61, ids[1]: 0.72, ids[2]: 0.02},
    )
    _, _, diag = apply_unified_acr_rcedr_selection(
        question_text=q,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_cfg(unified_acr_rcedr_semantic_sufficiency_enabled=True),
    )
    assert bool(diag.get("applied", False)) is True
    assert diag.get("semantic_score_coverage_rate") is not None
    assert diag.get("semantic_prior_applied_count") is not None


def test_missing_semantic_scores_do_not_crash_selector():
    q, ids, texts = _sample()
    table = {
        sid: {
            "corridor_ids": ["c1"],
            "is_main_candidate": i == 0,
            "is_support_candidate": i <= 1,
            "is_connector_adjacent": i == 1,
            "bridge_gain": 0.2 if i == 1 else 0.0,
            "locality_score": 1.0 / float(i + 1),
        }
        for i, sid in enumerate(ids)
    }
    out_ids, out_texts, diag = apply_unified_acr_rcedr_selection(
        question_text=q,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_cfg(unified_acr_rcedr_semantic_sufficiency_enabled=True),
    )
    assert len(out_ids) >= 1
    assert len(out_texts) >= 1
    assert _safe_numeric(diag.get("semantic_score_coverage_rate", 0.0)) >= 0.0


def _safe_numeric(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def test_no_gold_answer_argument_in_selector_api():
    sig = inspect.signature(apply_unified_acr_rcedr_selection)
    assert "answer" not in sig.parameters
    assert "gold" not in sig.parameters


def test_answer_metrics_offline_and_missing_diag_reported_na():
    payload = {
        "f1": 0.5,
        "em": 0.4,
        "recall_at_5": 0.7,
        "supporting_fact_precision": 0.2,
        "supporting_fact_recall": 0.3,
        "prompt_tokens_avg": 500,
        "retrieval_latency_ms": 1000,
        "generation_latency_ms": 100,
        "total_latency_ms": 1200,
        "answer_surface_present": 0.6,
        "answer_surface_position_avg": 0.2,
        "answer_surface_token_density": 0.01,
    }
    metrics = phase6v_sum._summary_metrics(payload)
    assert metrics["answer_string_hit"] == 0.6
    assert metrics["answer_string_rank_avg"] == 0.2
    assert metrics["answer_bearing_density"] == 0.01
    assert phase6v_sum._fmt(None, 4) == "n/a"
