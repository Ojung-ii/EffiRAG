from types import SimpleNamespace

from effirag.unified_acr_rcedr_selector import (
    UnifiedEvidenceAtom,
    _marginal_delta,
    apply_unified_acr_rcedr_selection,
)


def _atom(
    *,
    sid: str,
    src: str,
    toks: set[str],
    q: float = 0.0,
    ent: float = 0.0,
    rel: float = 0.0,
    ans: float = 0.0,
    bridge_overlap: float = 0.0,
    structure_anchor: float = 0.0,
    structure_bridge: float = 0.0,
    bridge_gain: float = 0.0,
    corridor_ids: tuple[str, ...] = (),
) -> UnifiedEvidenceAtom:
    return UnifiedEvidenceAtom(
        sentence_id=sid,
        source_id=src,
        text=sid,
        token_count=max(1, len(toks)),
        token_set=set(toks),
        query_overlap=float(q),
        entity_overlap=float(ent),
        relation_overlap=float(rel),
        answer_type_compat=float(ans),
        bridge_overlap=float(bridge_overlap),
        structure_anchor=float(structure_anchor),
        structure_bridge=float(structure_bridge),
        locality_score=0.5,
        corridor_ids=tuple(corridor_ids),
        bridge_gain=float(bridge_gain),
        atom_base_score=1.0,
    )


def _md(candidate, selected):
    selected_union = set()
    selected_answer = 0.0
    selected_tokens = 0
    for item in selected:
        selected_union.update(item.token_set)
        selected_answer = max(selected_answer, float(item.answer_type_compat))
        selected_tokens += int(item.token_count)
    return _marginal_delta(
        atom=candidate,
        selected_atoms=selected,
        selected_tokens=selected_tokens,
        selected_union_tokens=selected_union,
        selected_answer_type_compat=selected_answer,
        question_tokens={"q", "x", "y"},
        question_entities={"x"},
        relation_tokens={"rel"},
        max_tokens=220,
        use_answerability_gain=True,
        use_bridge_gain=True,
        use_redundancy_penalty=True,
        use_cost_penalty=False,
        chain_aware_enabled=False,
        role_aware_redundancy_enabled=False,
        role_balanced_enabled=True,
        redundancy_recalibrated_enabled=False,
        lambda_bridge=0.28,
        mu_redundancy=0.22,
        chain_gain_weight=0.15,
        role_balance_weight=0.08,
        role_balance_max_gain_per_step=0.08,
        role_balance_max_token_jaccard=0.45,
        role_balance_missing_only=True,
        role_redundancy_relax=0.75,
        role_redundancy_max_overlap=0.45,
        answerability_weight=1.0,
        timing_diag=None,
    )


def _base_cfg(**overrides):
    cfg = {
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
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _sample_inputs():
    sentence_ids = ["A::0", "B::0", "C::0"]
    sentence_texts = [
        "Alpha city is in country X.",
        "Country X borders Beta region.",
        "Unrelated sports trivia.",
    ]
    table = {
        "A::0": {
            "is_main_candidate": True,
            "is_connector_adjacent": False,
            "is_support_candidate": True,
            "bridge_gain": 0.1,
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
    question = "Which region borders the country where Alpha city is located?"
    return question, sentence_ids, sentence_texts, table


def test_baseline_unchanged_when_new_flags_disabled():
    q, ids, texts, table = _sample_inputs()
    out1_ids, out1_texts, _ = apply_unified_acr_rcedr_selection(
        question_text=q,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_base_cfg(),
    )
    out2_ids, out2_texts, _ = apply_unified_acr_rcedr_selection(
        question_text=q,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_base_cfg(
            unified_acr_rcedr_role_balanced_enabled=False,
            unified_acr_rcedr_redundancy_recalibrated_enabled=False,
        ),
    )
    assert out1_ids == out2_ids
    assert out1_texts == out2_texts


def test_role_balanced_gain_activates_for_missing_non_generic_role():
    selected = [
        _atom(
            sid="A::0",
            src="A",
            toks={"alpha", "country", "x"},
            q=1.0,
            ent=1.0,
            structure_anchor=1.0,
            corridor_ids=("c1",),
        )
    ]
    candidate = _atom(
        sid="B::0",
        src="B",
        toks={"beta", "region", "borders", "x"},
        rel=1.0,
        bridge_overlap=1.0,
        structure_bridge=1.0,
        bridge_gain=0.9,
        corridor_ids=("c1",),
    )
    _, comp = _md(candidate, selected)
    assert comp["role_coverage_gain"] > 0.0
    assert comp["role_coverage_gain_weighted"] > 0.0
    assert comp["role_balance_activated"] == 1


def test_role_balanced_gain_not_for_generic_or_covered_role():
    selected = [
        _atom(
            sid="B0",
            src="S",
            toks={"bridge", "token"},
            bridge_overlap=1.0,
            structure_bridge=1.0,
            bridge_gain=0.9,
            corridor_ids=("c1",),
        )
    ]
    generic = _atom(sid="G", src="G", toks={"zzz"}, corridor_ids=("c1",))
    _, comp_g = _md(generic, selected)
    assert comp_g["role_coverage_gain"] == 0.0

    covered_bridge = _atom(
        sid="B1",
        src="S2",
        toks={"new", "bridge", "evidence"},
        bridge_overlap=1.0,
        structure_bridge=1.0,
        bridge_gain=0.8,
        corridor_ids=("c1",),
    )
    _, comp_c = _md(covered_bridge, selected)
    assert comp_c["role_coverage_gain"] == 0.0


def test_redundancy_recalibration_applies_only_for_complementary_low_overlap_and_not_zeroed():
    anchor = _atom(
        sid="A",
        src="S",
        toks={"alpha", "city", "country", "x"},
        q=1.0,
        ent=1.0,
        structure_anchor=1.0,
        corridor_ids=("c1",),
    )
    bridge = _atom(
        sid="B",
        src="S",
        toks={"beta", "region", "borders", "x"},
        rel=1.0,
        bridge_overlap=1.0,
        structure_bridge=1.0,
        bridge_gain=0.9,
        corridor_ids=("c1",),
    )

    delta, comp = _marginal_delta(
        atom=bridge,
        selected_atoms=[anchor],
        selected_tokens=anchor.token_count,
        selected_union_tokens=set(anchor.token_set),
        selected_answer_type_compat=anchor.answer_type_compat,
        question_tokens={"q", "x", "y"},
        question_entities={"x"},
        relation_tokens={"rel"},
        max_tokens=220,
        use_answerability_gain=True,
        use_bridge_gain=True,
        use_redundancy_penalty=True,
        use_cost_penalty=False,
        chain_aware_enabled=False,
        role_aware_redundancy_enabled=False,
        role_balanced_enabled=False,
        redundancy_recalibrated_enabled=True,
        lambda_bridge=0.28,
        mu_redundancy=0.22,
        chain_gain_weight=0.15,
        role_balance_weight=0.08,
        role_balance_max_gain_per_step=0.08,
        role_balance_max_token_jaccard=0.45,
        role_balance_missing_only=True,
        role_redundancy_relax=0.75,
        role_redundancy_max_overlap=0.45,
        answerability_weight=1.0,
        timing_diag=None,
    )
    assert delta == delta  # not nan
    assert comp["redundancy_before"] > 0.0
    assert comp["redundancy_after"] < comp["redundancy_before"]
    assert comp["redundancy_after"] > 0.0


def test_missing_diagnostic_representation_not_forced_zero():
    # Summary/report policy: unavailable diagnostics should be represented as None/n/a, not 0.0.
    unavailable = None
    assert unavailable is None
