import inspect

import effirag.answerability_selection as acr


def _atom(
    *,
    sid: str,
    source: str,
    text: str,
    token_set: set[str],
    token_count: int,
    query_overlap: float,
    entity_overlap: float,
    relation_overlap: float,
    answer_type_compat: float,
    bridge_overlap: float,
    structure_anchor: float,
    structure_bridge: float,
    structure_prior: float,
    corridor_ids: tuple[str, ...],
    atom_score: float,
) -> acr.EvidenceAtom:
    return acr.EvidenceAtom(
        sentence_id=sid,
        source_id=source,
        text=text,
        token_count=token_count,
        token_set=set(token_set),
        query_overlap=query_overlap,
        entity_overlap=entity_overlap,
        relation_overlap=relation_overlap,
        answer_type_compat=answer_type_compat,
        bridge_overlap=bridge_overlap,
        structure_anchor=structure_anchor,
        structure_bridge=structure_bridge,
        structure_prior=structure_prior,
        corridor_ids=tuple(corridor_ids),
        atom_score=atom_score,
    )


def test_answerability_structure_noise_cost_components_behave_as_expected():
    atoms = [
        _atom(
            sid="DocA::0",
            source="DocA",
            text="OpenAI founded the lab in 2015.",
            token_set={"openai", "founded", "lab", "2015"},
            token_count=8,
            query_overlap=2.0,
            entity_overlap=1.0,
            relation_overlap=1.0,
            answer_type_compat=1.0,
            bridge_overlap=0.0,
            structure_anchor=1.0,
            structure_bridge=0.0,
            structure_prior=0.8,
            corridor_ids=("c1",),
            atom_score=4.0,
        ),
        _atom(
            sid="DocB::1",
            source="DocB",
            text="ConnectorAlpha links OpenAI and the target lab.",
            token_set={"connectoralpha", "openai", "target", "lab"},
            token_count=9,
            query_overlap=1.0,
            entity_overlap=1.0,
            relation_overlap=1.0,
            answer_type_compat=0.5,
            bridge_overlap=1.0,
            structure_anchor=0.0,
            structure_bridge=1.0,
            structure_prior=0.8,
            corridor_ids=("c1",),
            atom_score=3.5,
        ),
        _atom(
            sid="DocA::2",
            source="DocA",
            text="OpenAI founded the lab in 2015 with extra repeated words.",
            token_set={"openai", "founded", "lab", "2015", "extra", "repeated"},
            token_count=20,
            query_overlap=2.0,
            entity_overlap=1.0,
            relation_overlap=1.0,
            answer_type_compat=1.0,
            bridge_overlap=0.0,
            structure_anchor=1.0,
            structure_bridge=0.0,
            structure_prior=0.7,
            corridor_ids=("c1",),
            atom_score=3.2,
        ),
    ]
    question_tokens = {"who", "founded", "openai", "lab"}
    question_entities = {"openai"}
    relation_tokens = {"who", "founded"}

    compact = acr._score_selected_set(
        atoms=atoms,
        selected_indices=[0, 1],
        question_tokens=question_tokens,
        question_entities=question_entities,
        relation_tokens=relation_tokens,
        max_tokens=60,
        max_atoms=8,
        use_answerability=True,
        use_structure=True,
        use_noise_penalty=True,
        use_cost_penalty=True,
        structure_weight=0.65,
        noise_weight=0.55,
        cost_weight=0.35,
    )
    redundant = acr._score_selected_set(
        atoms=atoms,
        selected_indices=[0, 2],
        question_tokens=question_tokens,
        question_entities=question_entities,
        relation_tokens=relation_tokens,
        max_tokens=60,
        max_atoms=8,
        use_answerability=True,
        use_structure=True,
        use_noise_penalty=True,
        use_cost_penalty=True,
        structure_weight=0.65,
        noise_weight=0.55,
        cost_weight=0.35,
    )
    long_cost = acr._score_selected_set(
        atoms=atoms,
        selected_indices=[0, 1, 2],
        question_tokens=question_tokens,
        question_entities=question_entities,
        relation_tokens=relation_tokens,
        max_tokens=40,
        max_atoms=8,
        use_answerability=True,
        use_structure=True,
        use_noise_penalty=True,
        use_cost_penalty=True,
        structure_weight=0.65,
        noise_weight=0.55,
        cost_weight=0.35,
    )

    assert compact["A"] >= redundant["A"]
    assert compact["S"] >= 0.0
    assert redundant["N"] > compact["N"]
    assert long_cost["C"] > compact["C"]


def test_answerability_component_code_does_not_use_dataset_or_gold_support():
    score_src = inspect.getsource(acr._score_selected_set).lower()
    apply_src = inspect.getsource(acr.apply_answerability_constrained_selection).lower()
    merged = score_src + "\n" + apply_src
    for banned in [
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
        "popqa",
        "supporting_facts",
        "gold_support",
    ]:
        assert banned not in merged
    assert "answer_support_pinning" not in merged
