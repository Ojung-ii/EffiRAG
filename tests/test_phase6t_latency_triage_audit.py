from __future__ import annotations

from pathlib import Path

import yaml

from scripts.audit_phase6t_latency_triage import run_audit


def _write_cfg(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_cfg() -> dict:
    return {
        "unified_acr_rcedr_enabled": True,
        "gl_rcedr_enabled": False,
        "answerability_selection_enabled": False,
        "unified_acr_rcedr_hard_token_budget_enabled": True,
        "unified_acr_rcedr_use_answerability_gain": True,
        "unified_acr_rcedr_use_bridge_gain": True,
        "unified_acr_rcedr_use_redundancy_penalty": True,
        "unified_acr_rcedr_use_cost_penalty": False,
        "answer_support_pinning_enabled": False,
        "corridor_answer_preserve_guarded_hotpot_enabled": False,
        "final_top_slice_reorder_enabled": False,
        "prompt_variant": "light_separator_copy_span_instruction",
        "order_strategy": "score+light_separator_render",
        "render_mode": "corridor_aware_flat",
        "delivery_mode": "sentence_compressed",
        "max_context_sentences": 8,
        "max_total_sentences": 8,
        "max_sentences": 8,
        "target_prompt_tokens": 600,
        "max_prompt_tokens": 700,
        "top_corridors": 3,
        "max_corridors_in_context": 3,
        "bridge_candidate_induction_enabled": False,
    }


def _write_profile(config_root: Path, profile: str, extra: dict | None = None) -> None:
    for dataset in ["hotpotqa", "2wikimultihopqa"]:
        cfg = _base_cfg()
        if extra:
            cfg.update(extra)
        _write_cfg(config_root / profile / f"{dataset}.yaml", cfg)


def test_phase6t_latency_triage_audit_passes(tmp_path: Path) -> None:
    config_root = tmp_path / "cfgs"
    _write_profile(config_root, "unified_acr_rcedr_v12_sota_contract_unified")
    _write_profile(config_root, "unified_acr_rcedr_v12_sota_contract_rerank20", {"embedding_rerank_topn": 20})
    _write_profile(
        config_root,
        "unified_acr_rcedr_v12_sota_contract_search3x3",
        {"max_anchors": 3, "samples_per_anchor": 3},
    )
    _write_profile(
        config_root,
        "unified_acr_rcedr_v12_sota_contract_fast",
        {
            "embedding_rerank_topn": 20,
            "max_anchors": 3,
            "samples_per_anchor": 3,
            "ppr_mc_walks": 256,
            "corridor_top_bc": 10,
            "ppr_subgraph_max_nodes": 15000,
        },
    )

    report = run_audit(
        profiles=[
            "unified_acr_rcedr_v12_sota_contract_unified",
            "unified_acr_rcedr_v12_sota_contract_rerank20",
            "unified_acr_rcedr_v12_sota_contract_search3x3",
            "unified_acr_rcedr_v12_sota_contract_fast",
        ],
        datasets=["hotpotqa", "2wikimultihopqa"],
        config_root=config_root,
    )
    assert report["status"] == "pass"
    assert report["errors"] == []


def test_phase6t_latency_triage_audit_fails_on_dataset_heuristic(tmp_path: Path) -> None:
    config_root = tmp_path / "cfgs"
    _write_profile(config_root, "unified_acr_rcedr_v12_sota_contract_unified")
    _write_profile(config_root, "unified_acr_rcedr_v12_sota_contract_rerank20", {"embedding_rerank_topn": 20})
    _write_profile(
        config_root,
        "unified_acr_rcedr_v12_sota_contract_search3x3",
        {"max_anchors": 3, "samples_per_anchor": 3},
    )
    _write_profile(
        config_root,
        "unified_acr_rcedr_v12_sota_contract_fast",
        {
            "embedding_rerank_topn": 20,
            "max_anchors": 3,
            "samples_per_anchor": 3,
            "ppr_mc_walks": 256,
            "corridor_top_bc": 10,
            "ppr_subgraph_max_nodes": 15000,
        },
    )
    bad_cfg = _base_cfg()
    bad_cfg["corridor_answer_preserve_guarded_hotpot_enabled"] = True
    _write_cfg(config_root / "unified_acr_rcedr_v12_sota_contract_unified" / "hotpotqa.yaml", bad_cfg)

    report = run_audit(
        profiles=["unified_acr_rcedr_v12_sota_contract_unified"],
        datasets=["hotpotqa", "2wikimultihopqa"],
        config_root=config_root,
    )
    assert report["status"] == "fail"
    assert any("corridor_answer_preserve_guarded_hotpot_enabled=true" in e for e in report["errors"])


def test_phase6t_latency_triage_audit_supports_fast_isolation_profiles(tmp_path: Path) -> None:
    config_root = tmp_path / "cfgs"
    _write_profile(
        config_root,
        "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank",
        {
            "embedding_rerank_topn": 0,
            "sentence_rerank_enabled": False,
            "max_anchors": 3,
            "samples_per_anchor": 3,
            "ppr_mc_walks": 256,
            "corridor_top_bc": 10,
            "ppr_subgraph_max_nodes": 15000,
        },
    )
    _write_profile(
        config_root,
        "unified_acr_rcedr_v12_sota_contract_fast_rerank5",
        {
            "embedding_rerank_topn": 5,
            "max_anchors": 3,
            "samples_per_anchor": 3,
            "ppr_mc_walks": 256,
            "corridor_top_bc": 10,
            "ppr_subgraph_max_nodes": 15000,
        },
    )
    _write_profile(
        config_root,
        "unified_acr_rcedr_v12_sota_contract_fast_proposal_ultralight",
        {
            "embedding_rerank_topn": 20,
            "max_anchors": 2,
            "samples_per_anchor": 2,
            "ppr_mc_walks": 128,
            "corridor_top_bc": 6,
            "ppr_subgraph_max_nodes": 8000,
        },
    )

    report = run_audit(
        profiles=[
            "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank",
            "unified_acr_rcedr_v12_sota_contract_fast_rerank5",
            "unified_acr_rcedr_v12_sota_contract_fast_proposal_ultralight",
        ],
        datasets=["hotpotqa", "2wikimultihopqa"],
        config_root=config_root,
    )
    assert report["status"] == "pass"
    assert report["errors"] == []
