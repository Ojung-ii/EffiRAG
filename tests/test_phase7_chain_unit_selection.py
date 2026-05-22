from effirag.phase7_config import Phase7Config
from effirag.phase7_evidence_flow import (
    Phase7EvidenceAtom,
    _build_chain_units,
    _chain_feasibility_diagnostics,
    _unit_delta,
)


def _atom(source_id, title, text, entities=(), carrier=""):
    return Phase7EvidenceAtom(
        atom_id=f"s::{source_id}",
        source_id=source_id,
        title=title,
        text=text,
        token_count=max(1, len(text.split())),
        semantic_score=0.5,
        flow_score=0.5,
        anchor_distance=1.0,
        anchor_reachability=0.5,
        bridge_entity_coverage=0.0,
        relation_term_coverage=0.0,
        source_key=title,
        normalized_token_set=set(text.lower().split()),
        entity_set=set(entities),
        carrier_id=carrier,
        answerability_base=0.5,
        bridge_base=0.2,
    )


def test_chain_unit_builder_uses_strict_pair_types_only():
    a = _atom("A::0", "Alpha", "Alpha mentions Beta", entities={"beta"}, carrier="c1")
    b = _atom("B::0", "Beta", "Beta answer sentence", entities=set(), carrier="c2")
    c = _atom("C::0", "Gamma", "Gamma shares common entity", entities={"shared"}, carrier="c3")
    d = _atom("D::0", "Delta", "Delta also shares common entity", entities={"shared"}, carrier="c4")
    p7 = Phase7Config(chain_unit_enabled=True, chain_unit_max_pair_units=10)

    units, diag = _build_chain_units([a, b, c, d], p7)
    pair_ids = {u.unit_id: u.unit_type for u in units if len(u.atoms) == 2}

    assert pair_ids["A::0||B::0"] == "explicit_transition_pair"
    assert "C::0||D::0" not in pair_ids
    assert diag["num_single_units"] == 4
    assert diag["num_explicit_transition_units"] == 1
    assert diag["unit_preview"][0]["unit_type"] == "explicit_transition_pair"


def test_chain_feasibility_requires_connected_gold_chain():
    a = _atom("A::0", "Alpha", "Alpha mentions Beta", entities={"beta"}, carrier="c1")
    b = _atom("B::0", "Beta", "Beta answer sentence", carrier="c2")
    p7 = Phase7Config(chain_unit_enabled=True, chain_unit_max_pair_units=10)
    units, _diag = _build_chain_units([a, b], p7)

    feasibility, gold_eval = _chain_feasibility_diagnostics(
        atoms=[a, b],
        units=units,
        gold_unit_ids={"A::0", "B::0"},
        selected_ids={"A::0"},
        max_selected_atoms=10,
        max_context_tokens=128,
    )

    assert feasibility["candidate_gold_full"] is True
    assert feasibility["candidate_graph_gold_connected"] is True
    assert feasibility["budget_feasible_gold_chain"] is True
    assert feasibility["chain_unit_oracle_feasible"] is True
    assert gold_eval["gold_unit_full"] is True


def test_chain_unit_delta_rewards_pair_unit_over_single_connection():
    selected = [_atom("A::0", "Alpha", "Alpha mentions Beta", entities={"beta"}, carrier="c1")]
    cand = _atom("B::0", "Beta", "Beta answer sentence", carrier="c2")
    p7 = Phase7Config(chain_unit_enabled=True)
    units, _diag = _build_chain_units(selected + [cand], p7)
    pair = next(u for u in units if u.unit_id == "A::0||B::0")

    terms = _unit_delta(
        unit=pair,
        selected=selected,
        selected_query_tokens=set(),
        selected_bridge_entities=set(),
        selected_anchor_coverage=set(),
        selected_corridor_paths=set(),
        selected_sources=set(),
        max_selected_atoms=10,
        max_context_tokens=128,
        selected_token_count=sum(a.token_count for a in selected),
        p7=p7,
    )

    assert terms["unit_type"] == "explicit_transition_pair"
    assert terms["C_unit"] >= 1.0
    assert terms["final_gain"] > terms["A_unit"]
