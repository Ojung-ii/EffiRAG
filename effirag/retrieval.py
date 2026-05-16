import multiprocessing as mp
import os
import random
import time
import hashlib
import weakref
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import networkx as nx
import numpy as np

from .anchors import select_lexical_anchors
from .canonical_objective import (
    apply_canonical_copy_span_objective,
    canonical_effective_reference_mode,
    is_canonical_copy_span_mode,
)
from .canonical_scoring import (
    canonical_run_score,
    canonical_run_surrogate_loss,
    canonical_run_weight_bundle,
    canonical_seed_score_map,
)
from .candidate_recall_boost import apply_candidate_recall_boost
from .config import RetrievalConfig
from .embedding import cosine_similarity, encode_texts, rerank_sentences_by_embedding, topk_cosine_similarity
from .evidence_density_rerank import rerank_evidence_density
from .answerability_selection import apply_answerability_constrained_selection
from .gl_rcedr import apply_gl_rcedr
from .global_index import load_or_build_global_index, load_semantic_index
from .graph import build_document_entity_graph
from .legacy_objectives import (
    apply_legacy_objective_flag_overrides,
    is_legacy_objective_mode,
    normalize_objective_mode,
)
from .metrics import supporting_fact_match_details
from .registry import register_method
from .types import AnchorResult, RetrievalResult
from .utils import content_tokens

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

_GLOBAL_INDEX_MEMO = {}
_GLOBAL_SEMANTIC_MEMO = {}
_PPR_GRAPH_STRUCT_MEMO = weakref.WeakKeyDictionary()
_PPR_GRAPH_STRUCT_MEMO_FALLBACK = {}
_PPR_PARALLEL_STRUCT = None
_QUERY_EMBED_MEMO = {}


def _safe_int_cast(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_shared_budget_profile_defaults(profile):
    name = str(profile or "off").strip().lower()
    aliases = {
        "baseline": "off",
        "d1": "off",
        "none": "off",
        "g1": "g1_proposal",
        "proposal_light": "g1_proposal",
        "shared_proposal": "g1_proposal",
        "shared_g1": "g1_proposal",
        "stage_c": "g1_proposal",
        "stage_c1": "g1_proposal",
        "stage_c2": "g1_proposal",
        "g2": "g2_exploration",
        "g3": "g_shared",
        "stage_g1": "g1_proposal",
        "stage_g2": "g2_exploration",
        "stage_g3": "g_shared",
        "stage_g_shared": "g_shared",
        "stage_t_hotpot": "t_hotpot",
        "stage_t_2wiki": "t_2wiki",
        "t_hotpotqa": "t_hotpot",
        "t_2wikimultihopqa": "t_2wiki",
    }
    name = aliases.get(name, name)
    defaults = {
        "off": (0, 0),
        "g1_proposal": (1, 0),
        "g2_exploration": (0, 1),
        "g_shared": (1, 1),
        "t_hotpot": (0, 2),
        "t_2wiki": (2, 0),
    }
    if name not in defaults:
        name = "off"
    proposal_step, exploration_step = defaults[name]
    return name, int(proposal_step), int(exploration_step)


def _apply_shared_budget_profile_once(cfg):
    already = bool(getattr(cfg, "_shared_budget_profile_applied", False))
    if already:
        return dict(getattr(cfg, "_shared_budget_profile_diag", {}) or {})

    requested = str(getattr(cfg, "shared_budget_profile", "off") or "off").strip().lower()
    resolved, default_proposal_step, default_exploration_step = _resolve_shared_budget_profile_defaults(requested)
    proposal_step = max(0, _safe_int_cast(getattr(cfg, "shared_budget_proposal_step", 0), 0))
    exploration_step = max(0, _safe_int_cast(getattr(cfg, "shared_budget_exploration_step", 0), 0))
    if proposal_step <= 0 and exploration_step <= 0:
        proposal_step = int(default_proposal_step)
        exploration_step = int(default_exploration_step)

    updates = {}

    def _bump(field_name, delta, min_value=1):
        if int(delta) <= 0:
            return
        before = _safe_int_cast(getattr(cfg, field_name, min_value), min_value)
        after = max(int(min_value), int(before + int(delta)))
        if after == before:
            return
        setattr(cfg, field_name, int(after))
        updates[field_name] = {
            "before": int(before),
            "after": int(after),
            "delta": int(after - before),
        }

    _bump("semantic_topn_entity", 2 * int(proposal_step), min_value=1)
    _bump("semantic_topn_chunk", 1 * int(proposal_step), min_value=1)
    _bump("graph_reserve_topn", 1 * int(proposal_step), min_value=0)

    _bump("max_anchors", 1 * int(exploration_step), min_value=1)
    _bump("samples_per_anchor", 1 * int(exploration_step), min_value=1)
    _bump("phase1_run_preshortlist_topm", 1 * int(exploration_step), min_value=1)
    _bump("phase1_full_run_score_topk", 1 * int(exploration_step), min_value=1)

    shortlist = max(1, _safe_int_cast(getattr(cfg, "phase1_run_shortlist_topk", 1), 1))
    preshort = max(1, _safe_int_cast(getattr(cfg, "phase1_run_preshortlist_topm", shortlist), shortlist))
    full_topk_before = _safe_int_cast(getattr(cfg, "phase1_full_run_score_topk", preshort), preshort)
    full_topk = max(shortlist, preshort, full_topk_before)
    if full_topk != full_topk_before:
        setattr(cfg, "phase1_full_run_score_topk", int(full_topk))
        updates["phase1_full_run_score_topk"] = {
            "before": int(full_topk_before),
            "after": int(full_topk),
            "delta": int(full_topk - full_topk_before),
        }

    force_rebuild_auto_disabled = False
    workers = max(1, _safe_int_cast(getattr(cfg, "num_workers", 1), 1))
    force_rebuild = bool(getattr(cfg, "force_rebuild_graph_index", False))
    if workers > 1 and force_rebuild:
        setattr(cfg, "force_rebuild_graph_index", False)
        force_rebuild_auto_disabled = True
        updates["force_rebuild_graph_index"] = {
            "before": True,
            "after": False,
            "reason": "num_workers_cache_race_guard",
        }

    diag = {
        "requested_profile": str(requested),
        "resolved_profile": str(resolved),
        "proposal_step": int(proposal_step),
        "exploration_step": int(exploration_step),
        "applied": bool(proposal_step > 0 or exploration_step > 0),
        "updates": dict(updates),
        "force_rebuild_graph_index_auto_disabled": bool(force_rebuild_auto_disabled),
        "num_workers": int(workers),
    }
    setattr(cfg, "_shared_budget_profile_applied", True)
    setattr(cfg, "_shared_budget_profile_diag", dict(diag))
    return dict(diag)


def _resolve_retrieval_objective_mode(cfg):
    mode = str(getattr(cfg, "retrieval_objective_mode", "baseline") or "baseline").strip().lower()
    return normalize_objective_mode(mode)


def _resolve_retrieval_objective_flags(cfg):
    mode = _resolve_retrieval_objective_mode(cfg)
    if is_canonical_copy_span_mode(mode):
        apply_canonical_copy_span_objective(cfg, dataset=getattr(cfg, "dataset", None), mode=mode)

    hybrid = bool(getattr(cfg, "hybrid_anchor_recall_enabled", False))
    bridge_induction = bool(getattr(cfg, "bridge_candidate_induction_enabled", False))
    role_chunk = bool(getattr(cfg, "role_aware_chunk_scoring_enabled", False))
    coverage = bool(getattr(cfg, "coverage_selection_enabled", False))
    corridor_compact = bool(getattr(cfg, "corridor_compact_shaping_enabled", False))
    corridor_answer_preserve = bool(getattr(cfg, "corridor_answer_preserve_enabled", False))
    corridor_bridge_purity = bool(getattr(cfg, "corridor_bridge_purity_shaping_enabled", False))
    corridor_path_preserve = bool(getattr(cfg, "corridor_path_preserve_enabled", False))
    corridor_path_preserve_compact = bool(getattr(cfg, "corridor_path_preserve_compact_enabled", False))
    corridor_path_preserve_guarded = bool(getattr(cfg, "corridor_path_preserve_guarded_enabled", False))
    corridor_path_preserve_compact_lite = bool(getattr(cfg, "corridor_path_preserve_compact_lite_enabled", False))
    corridor_answer_preserve_guarded_hotpot = bool(
        getattr(cfg, "corridor_answer_preserve_guarded_hotpot_enabled", False)
    )
    corridor_answer_preserve_confidence_gated = bool(
        getattr(cfg, "corridor_answer_preserve_confidence_gated_enabled", False)
    )

    if mode == "hybrid_anchor_recall":
        hybrid = True
    elif mode == "bridge_candidate_induction":
        bridge_induction = True
    elif mode == "role_aware_chunk_scoring":
        role_chunk = True
    elif mode == "coverage_selection":
        coverage = True
    elif mode == "bridge_coverage_full":
        hybrid = True
        bridge_induction = True
        role_chunk = True
        coverage = True
    elif is_legacy_objective_mode(mode):
        legacy_flags = apply_legacy_objective_flag_overrides(
            mode,
            {
                "mode": mode,
                "hybrid_anchor_recall": hybrid,
                "bridge_candidate_induction": bridge_induction,
                "role_aware_chunk_scoring": role_chunk,
                "coverage_selection": coverage,
                "corridor_compact_shaping": corridor_compact,
                "corridor_answer_preserve_shaping": corridor_answer_preserve,
                "corridor_bridge_purity_shaping": corridor_bridge_purity,
                "corridor_path_preserve_shaping": corridor_path_preserve,
                "corridor_path_preserve_compact_shaping": corridor_path_preserve_compact,
                "corridor_path_preserve_guarded_shaping": corridor_path_preserve_guarded,
                "corridor_path_preserve_compact_lite_shaping": corridor_path_preserve_compact_lite,
                "corridor_answer_preserve_guarded_hotpot": corridor_answer_preserve_guarded_hotpot,
                "corridor_answer_preserve_confidence_gated": corridor_answer_preserve_confidence_gated,
            },
        )
        hybrid = bool(legacy_flags.get("hybrid_anchor_recall", hybrid))
        bridge_induction = bool(legacy_flags.get("bridge_candidate_induction", bridge_induction))
        role_chunk = bool(legacy_flags.get("role_aware_chunk_scoring", role_chunk))
        coverage = bool(legacy_flags.get("coverage_selection", coverage))
        corridor_compact = bool(legacy_flags.get("corridor_compact_shaping", corridor_compact))
        corridor_answer_preserve = bool(legacy_flags.get("corridor_answer_preserve_shaping", corridor_answer_preserve))
        corridor_bridge_purity = bool(legacy_flags.get("corridor_bridge_purity_shaping", corridor_bridge_purity))
        corridor_path_preserve = bool(legacy_flags.get("corridor_path_preserve_shaping", corridor_path_preserve))
        corridor_path_preserve_compact = bool(
            legacy_flags.get("corridor_path_preserve_compact_shaping", corridor_path_preserve_compact)
        )
        corridor_path_preserve_guarded = bool(
            legacy_flags.get("corridor_path_preserve_guarded_shaping", corridor_path_preserve_guarded)
        )
        corridor_path_preserve_compact_lite = bool(
            legacy_flags.get("corridor_path_preserve_compact_lite_shaping", corridor_path_preserve_compact_lite)
        )
        corridor_answer_preserve_guarded_hotpot = bool(
            legacy_flags.get("corridor_answer_preserve_guarded_hotpot", corridor_answer_preserve_guarded_hotpot)
        )
        corridor_answer_preserve_confidence_gated = bool(
            legacy_flags.get("corridor_answer_preserve_confidence_gated", corridor_answer_preserve_confidence_gated)
        )

    return {
        "mode": str(mode),
        "hybrid_anchor_recall": bool(hybrid),
        "bridge_candidate_induction": bool(bridge_induction),
        "role_aware_chunk_scoring": bool(role_chunk),
        "coverage_selection": bool(coverage),
        "corridor_compact_shaping": bool(corridor_compact),
        "corridor_answer_preserve_shaping": bool(corridor_answer_preserve),
        "corridor_bridge_purity_shaping": bool(corridor_bridge_purity),
        "corridor_path_preserve_shaping": bool(corridor_path_preserve),
        "corridor_path_preserve_compact_shaping": bool(corridor_path_preserve_compact),
        "corridor_path_preserve_guarded_shaping": bool(corridor_path_preserve_guarded),
        "corridor_path_preserve_compact_lite_shaping": bool(corridor_path_preserve_compact_lite),
        "corridor_answer_preserve_guarded_hotpot": bool(corridor_answer_preserve_guarded_hotpot),
        "corridor_answer_preserve_confidence_gated": bool(corridor_answer_preserve_confidence_gated),
    }


def _set_cfg_attr_if_changed(cfg, field_name, value, updates):
    before = getattr(cfg, field_name, None)
    try:
        same = float(before) == float(value)
    except Exception:
        same = before == value
    if same:
        return
    setattr(cfg, field_name, value)
    updates[field_name] = {
        "before": before,
        "after": value,
    }


def _apply_connector_objective_profile(cfg, objective_flags):
    flags = dict(objective_flags or {})
    requested_mode = normalize_objective_mode(str(flags.get("mode", "baseline") or "baseline").strip().lower())
    flags["mode"] = str(requested_mode)
    mode = str(requested_mode)
    canonical_diag = {}
    canonical_mode = is_canonical_copy_span_mode(requested_mode)
    if canonical_mode:
        canonical_diag = apply_canonical_copy_span_objective(
            cfg,
            dataset=getattr(cfg, "dataset", None),
            mode=requested_mode,
        )
        mode = canonical_effective_reference_mode(getattr(cfg, "dataset", None), mode=requested_mode)

    updates = {}
    profile_applied = "none"
    flags["corridor_answer_preserve_guarded_hotpot"] = False
    flags["corridor_answer_preserve_confidence_gated"] = False
    flags["corridor_path_preserve_shaping"] = False
    flags["corridor_path_preserve_compact_shaping"] = False
    flags["corridor_path_preserve_guarded_shaping"] = False
    flags["corridor_path_preserve_compact_lite_shaping"] = False

    def _apply_run_bridge_aware():
        _set_cfg_attr_if_changed(cfg, "run_score_semantic_weight", 0.22, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_anchor_weight", 0.18, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_structure_weight", 0.22, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_bridge_weight", 0.22, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_redundancy_weight", 0.08, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_pair_coverage_weight", 0.04, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_bridge_completeness_weight", 0.04, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_entity_chunk_grounding_weight", 0.0, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_anchor_dispersion_penalty", 0.0, updates)
        _set_cfg_attr_if_changed(cfg, "seed_objective_bridge_weight", 0.28, updates)
        _set_cfg_attr_if_changed(cfg, "seed_objective_anchor_coverage_weight", 0.12, updates)

    def _apply_run_role_balanced():
        _set_cfg_attr_if_changed(cfg, "run_score_semantic_weight", 0.20, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_anchor_weight", 0.20, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_structure_weight", 0.20, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_bridge_weight", 0.16, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_redundancy_weight", 0.08, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_pair_coverage_weight", 0.08, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_bridge_completeness_weight", 0.05, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_entity_chunk_grounding_weight", 0.03, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_anchor_dispersion_penalty", 0.08, updates)
        _set_cfg_attr_if_changed(cfg, "seed_objective_bridge_weight", 0.24, updates)
        _set_cfg_attr_if_changed(cfg, "seed_objective_anchor_coverage_weight", 0.20, updates)

    def _apply_run_bridge_lite():
        _set_cfg_attr_if_changed(cfg, "run_score_semantic_weight", 0.24, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_anchor_weight", 0.20, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_structure_weight", 0.24, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_bridge_weight", 0.18, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_redundancy_weight", 0.10, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_pair_coverage_weight", 0.02, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_bridge_completeness_weight", 0.02, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_entity_chunk_grounding_weight", 0.0, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_anchor_dispersion_penalty", 0.0, updates)
        _set_cfg_attr_if_changed(cfg, "seed_objective_bridge_weight", 0.24, updates)
        _set_cfg_attr_if_changed(cfg, "seed_objective_anchor_coverage_weight", 0.14, updates)

    def _apply_run_connector_core():
        _set_cfg_attr_if_changed(cfg, "run_score_semantic_weight", 0.22, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_anchor_weight", 0.18, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_structure_weight", 0.22, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_bridge_weight", 0.24, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_redundancy_weight", 0.08, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_pair_coverage_weight", 0.03, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_bridge_completeness_weight", 0.03, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_entity_chunk_grounding_weight", 0.0, updates)
        _set_cfg_attr_if_changed(cfg, "run_score_anchor_dispersion_penalty", 0.0, updates)
        _set_cfg_attr_if_changed(cfg, "seed_objective_bridge_weight", 0.28, updates)
        _set_cfg_attr_if_changed(cfg, "seed_objective_anchor_coverage_weight", 0.16, updates)

    def _configure_corridor_shaping(
        compact_enabled,
        answer_preserve_enabled,
        bridge_purity_enabled,
        guarded_hotpot_enabled=False,
        confidence_gated_enabled=False,
        path_preserve_enabled=False,
        path_preserve_compact_enabled=False,
        path_preserve_guarded_enabled=False,
        path_preserve_compact_lite_enabled=False,
    ):
        compact_flag = bool(compact_enabled)
        answer_flag = bool(answer_preserve_enabled)
        purity_flag = bool(bridge_purity_enabled)
        guarded_hotpot_flag = bool(guarded_hotpot_enabled)
        confidence_gated_flag = bool(confidence_gated_enabled)
        path_preserve_flag = bool(path_preserve_enabled)
        path_preserve_compact_flag = bool(path_preserve_compact_enabled)
        path_preserve_guarded_flag = bool(path_preserve_guarded_enabled)
        path_preserve_compact_lite_flag = bool(path_preserve_compact_lite_enabled)
        _set_cfg_attr_if_changed(cfg, "corridor_compact_shaping_enabled", compact_flag, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_enabled", answer_flag, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_bridge_purity_shaping_enabled", purity_flag, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_enabled", path_preserve_flag, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_compact_enabled", path_preserve_compact_flag, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_guarded_enabled", path_preserve_guarded_flag, updates)
        _set_cfg_attr_if_changed(
            cfg, "corridor_path_preserve_compact_lite_enabled", path_preserve_compact_lite_flag, updates
        )
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_guarded_hotpot_enabled", guarded_hotpot_flag, updates)
        _set_cfg_attr_if_changed(
            cfg, "corridor_answer_preserve_confidence_gated_enabled", confidence_gated_flag, updates
        )
        _set_cfg_attr_if_changed(cfg, "corridor_compact_weight", (0.18 if compact_flag else 0.0), updates)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_weight", (0.22 if answer_flag else 0.0), updates)
        _set_cfg_attr_if_changed(cfg, "corridor_bridge_purity_weight", (0.22 if purity_flag else 0.0), updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_weight", (0.26 if path_preserve_flag else 0.0), updates)
        _set_cfg_attr_if_changed(
            cfg,
            "corridor_path_preserve_incomplete_penalty_weight",
            (0.14 if path_preserve_flag else 0.0),
            updates,
        )
        _set_cfg_attr_if_changed(
            cfg, "corridor_path_preserve_compact_weight", (0.14 if path_preserve_compact_flag else 0.0), updates
        )
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_anchor_threshold", 0.35, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_bridge_threshold", 0.35, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_answer_threshold", 0.55, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_pair_threshold", 0.45, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_guard_scale", 0.70, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_guard_min_complete_score", 0.70, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_guard_min_pair_retention", 0.55, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_guard_min_answer_density", 0.62, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_compact_lite_scale", 0.65, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_compact_lite_overflow_coeff", 0.015, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_compact_lite_overflow_cap", 0.08, updates)
        # Guard only used when answer-preserve shaping is on.
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_min_density", 0.30, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_light_boost_scale", 0.65, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_bridge_consistency_threshold", 0.55, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_confidence_threshold", 0.62, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_hotpot_anchor_bridge_sufficient", 0.70, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_hotpot_answer_density_cap", 0.72, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_hotpot_quota_cap", 0.40, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_hotpot_relax_factor", 0.35, updates)
        _set_cfg_attr_if_changed(cfg, "hotpot_overpreserve_answer_share_threshold", 0.10, updates)

    _configure_corridor_shaping(False, False, False)

    if mode == "run_objective_bridge_aware":
        profile_applied = "run_bridge_aware"
        _apply_run_bridge_aware()
        flags["bridge_candidate_induction"] = False
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
    elif mode == "run_objective_role_balanced":
        profile_applied = "run_role_balanced"
        _apply_run_role_balanced()
        flags["bridge_candidate_induction"] = False
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
    elif mode == "corridor_role_constrained":
        profile_applied = "corridor_role_constrained"
        flags["coverage_selection"] = True
        flags["bridge_candidate_induction"] = False
        _set_cfg_attr_if_changed(cfg, "coverage_selection_role_weight", 0.55, updates)
        _set_cfg_attr_if_changed(cfg, "coverage_selection_redundancy_weight", 0.22, updates)
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
    elif mode == "bridge_aware_run_plus_role_constrained_corridor":
        profile_applied = "bridge_run_plus_corridor"
        _apply_run_bridge_aware()
        flags["coverage_selection"] = True
        flags["bridge_candidate_induction"] = False
        _set_cfg_attr_if_changed(cfg, "coverage_selection_role_weight", 0.55, updates)
        _set_cfg_attr_if_changed(cfg, "coverage_selection_redundancy_weight", 0.22, updates)
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
    elif mode == "seed_quality_analysis":
        profile_applied = "seed_quality_analysis"
        flags["bridge_candidate_induction"] = False
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
    elif mode in {"r2_bridge_only", "r2_connector_core"}:
        profile_applied = "r2_bridge_only_lite"
        _apply_run_bridge_lite()
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
        flags["corridor_path_preserve_shaping"] = False
        flags["corridor_path_preserve_compact_shaping"] = False
    elif mode == "r2_plus_r3_anchor_light":
        profile_applied = "r2_plus_r3_anchor_light"
        _apply_run_bridge_lite()
        _set_cfg_attr_if_changed(cfg, "role_chunk_weight_anchor_match", 0.40, updates)
        _set_cfg_attr_if_changed(cfg, "role_chunk_weight_bridge_support", 0.30, updates)
        _set_cfg_attr_if_changed(cfg, "role_chunk_weight_answer_likelihood", 0.18, updates)
        _set_cfg_attr_if_changed(cfg, "role_chunk_weight_path_consistency", 0.12, updates)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = True
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
    elif mode == "r2_plus_r3_answer_light":
        profile_applied = "r2_plus_r3_answer_light"
        _apply_run_bridge_lite()
        _set_cfg_attr_if_changed(cfg, "role_chunk_weight_anchor_match", 0.24, updates)
        _set_cfg_attr_if_changed(cfg, "role_chunk_weight_bridge_support", 0.28, updates)
        _set_cfg_attr_if_changed(cfg, "role_chunk_weight_answer_likelihood", 0.34, updates)
        _set_cfg_attr_if_changed(cfg, "role_chunk_weight_path_consistency", 0.14, updates)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = True
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
        flags["corridor_path_preserve_shaping"] = False
        flags["corridor_path_preserve_compact_shaping"] = False
    elif mode == "r2_plus_path_preserve":
        profile_applied = "r2_plus_path_preserve"
        _apply_run_bridge_lite()
        _configure_corridor_shaping(
            False,
            False,
            False,
            guarded_hotpot_enabled=False,
            confidence_gated_enabled=False,
            path_preserve_enabled=True,
            path_preserve_compact_enabled=False,
            path_preserve_guarded_enabled=False,
            path_preserve_compact_lite_enabled=False,
        )
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
        flags["corridor_path_preserve_shaping"] = True
        flags["corridor_path_preserve_compact_shaping"] = False
        flags["corridor_path_preserve_guarded_shaping"] = False
        flags["corridor_path_preserve_compact_lite_shaping"] = False
    elif mode == "r2_plus_path_preserve_compact":
        profile_applied = "r2_plus_path_preserve_compact"
        _apply_run_bridge_lite()
        _configure_corridor_shaping(
            True,
            False,
            False,
            guarded_hotpot_enabled=False,
            confidence_gated_enabled=False,
            path_preserve_enabled=True,
            path_preserve_compact_enabled=True,
            path_preserve_guarded_enabled=False,
            path_preserve_compact_lite_enabled=False,
        )
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = True
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
        flags["corridor_path_preserve_shaping"] = True
        flags["corridor_path_preserve_compact_shaping"] = True
        flags["corridor_path_preserve_guarded_shaping"] = False
        flags["corridor_path_preserve_compact_lite_shaping"] = False
    elif mode == "r2_plus_path_preserve_guarded":
        profile_applied = "r2_plus_path_preserve_guarded"
        _apply_run_bridge_lite()
        _configure_corridor_shaping(
            False,
            False,
            False,
            guarded_hotpot_enabled=False,
            confidence_gated_enabled=False,
            path_preserve_enabled=True,
            path_preserve_compact_enabled=False,
            path_preserve_guarded_enabled=True,
            path_preserve_compact_lite_enabled=False,
        )
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_weight", 0.20, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_incomplete_penalty_weight", 0.10, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_guard_scale", 0.60, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_guard_min_complete_score", 0.74, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_guard_min_pair_retention", 0.60, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_guard_min_answer_density", 0.66, updates)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
        flags["corridor_path_preserve_shaping"] = True
        flags["corridor_path_preserve_compact_shaping"] = False
        flags["corridor_path_preserve_guarded_shaping"] = True
        flags["corridor_path_preserve_compact_lite_shaping"] = False
    elif mode == "r2_plus_path_preserve_compact_lite":
        profile_applied = "r2_plus_path_preserve_compact_lite"
        _apply_run_bridge_lite()
        _configure_corridor_shaping(
            True,
            False,
            False,
            guarded_hotpot_enabled=False,
            confidence_gated_enabled=False,
            path_preserve_enabled=True,
            path_preserve_compact_enabled=True,
            path_preserve_guarded_enabled=False,
            path_preserve_compact_lite_enabled=True,
        )
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_weight", 0.23, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_compact_weight", 0.08, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_compact_lite_scale", 0.55, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_compact_lite_overflow_coeff", 0.012, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_path_preserve_compact_lite_overflow_cap", 0.06, updates)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = True
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
        flags["corridor_path_preserve_shaping"] = True
        flags["corridor_path_preserve_compact_shaping"] = True
        flags["corridor_path_preserve_guarded_shaping"] = False
        flags["corridor_path_preserve_compact_lite_shaping"] = True
    elif mode == "seed_run_connector_core":
        profile_applied = "seed_run_connector_core"
        _apply_run_connector_core()
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
    elif mode == "seed_run_connector_core_corridor_compact":
        profile_applied = "seed_run_connector_core_corridor_compact"
        _apply_run_connector_core()
        _configure_corridor_shaping(True, False, False)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = True
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = False
    elif mode == "seed_run_connector_core_corridor_answer_preserve":
        profile_applied = "seed_run_connector_core_corridor_answer_preserve"
        _apply_run_connector_core()
        _configure_corridor_shaping(False, True, False)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = True
        flags["corridor_bridge_purity_shaping"] = False
    elif mode == "seed_run_connector_core_corridor_bridge_purity":
        profile_applied = "seed_run_connector_core_corridor_bridge_purity"
        _apply_run_connector_core()
        _configure_corridor_shaping(False, False, True)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = False
        flags["corridor_bridge_purity_shaping"] = True
    elif mode == "seed_run_connector_core_corridor_compact_answer_preserve":
        profile_applied = "seed_run_connector_core_corridor_compact_answer_preserve"
        _apply_run_connector_core()
        _configure_corridor_shaping(True, True, False)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = True
        flags["corridor_answer_preserve_shaping"] = True
        flags["corridor_bridge_purity_shaping"] = False
    elif mode == "p3_answer_preserve_base":
        profile_applied = "p3_answer_preserve_base"
        _apply_run_connector_core()
        _configure_corridor_shaping(False, True, False)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = True
        flags["corridor_bridge_purity_shaping"] = False
    elif mode == "p3_answer_preserve_guarded_hotpot":
        profile_applied = "p3_answer_preserve_guarded_hotpot"
        _apply_run_connector_core()
        _configure_corridor_shaping(False, True, False, guarded_hotpot_enabled=True, confidence_gated_enabled=False)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = True
        flags["corridor_bridge_purity_shaping"] = False
        flags["corridor_answer_preserve_guarded_hotpot"] = True
    elif mode == "p3_answer_preserve_confidence_gated":
        profile_applied = "p3_answer_preserve_confidence_gated"
        _apply_run_connector_core()
        _configure_corridor_shaping(False, True, False, guarded_hotpot_enabled=False, confidence_gated_enabled=True)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_weight", 0.18, updates)
        _set_cfg_attr_if_changed(cfg, "corridor_answer_preserve_light_boost_scale", 0.55, updates)
        flags["bridge_candidate_induction"] = True
        flags["role_aware_chunk_scoring"] = False
        flags["coverage_selection"] = False
        flags["corridor_compact_shaping"] = False
        flags["corridor_answer_preserve_shaping"] = True
        flags["corridor_bridge_purity_shaping"] = False
        flags["corridor_answer_preserve_confidence_gated"] = True

    if canonical_mode:
        canonical_diag = apply_canonical_copy_span_objective(
            cfg,
            dataset=getattr(cfg, "dataset", None),
            mode=requested_mode,
        )
        flags["bridge_candidate_induction"] = bool(getattr(cfg, "bridge_candidate_induction_enabled", False))
        flags["role_aware_chunk_scoring"] = bool(getattr(cfg, "role_aware_chunk_scoring_enabled", False))
        flags["coverage_selection"] = bool(getattr(cfg, "coverage_selection_enabled", False))
        flags["corridor_compact_shaping"] = bool(getattr(cfg, "corridor_compact_shaping_enabled", False))
        flags["corridor_answer_preserve_shaping"] = bool(getattr(cfg, "corridor_answer_preserve_enabled", False))
        flags["corridor_bridge_purity_shaping"] = bool(getattr(cfg, "corridor_bridge_purity_shaping_enabled", False))
        flags["corridor_path_preserve_shaping"] = bool(getattr(cfg, "corridor_path_preserve_enabled", False))
        flags["corridor_path_preserve_compact_shaping"] = bool(
            getattr(cfg, "corridor_path_preserve_compact_enabled", False)
        )
        flags["corridor_path_preserve_guarded_shaping"] = bool(
            getattr(cfg, "corridor_path_preserve_guarded_enabled", False)
        )
        flags["corridor_path_preserve_compact_lite_shaping"] = bool(
            getattr(cfg, "corridor_path_preserve_compact_lite_enabled", False)
        )
        flags["corridor_answer_preserve_guarded_hotpot"] = bool(
            getattr(cfg, "corridor_answer_preserve_guarded_hotpot_enabled", False)
        )
        flags["corridor_answer_preserve_confidence_gated"] = bool(
            getattr(cfg, "corridor_answer_preserve_confidence_gated_enabled", False)
        )
        profile_applied = f"canonical_copy_span::{mode}"

    diag = {
        "mode": requested_mode,
        "effective_mode": mode,
        "profile_applied": profile_applied,
        "flags_after": dict(flags),
        "weight_updates": dict(updates),
        "canonical": dict(canonical_diag) if canonical_mode else {},
    }
    return flags, diag


def _get_global_graph(cfg):
    corpus_path = str(getattr(cfg, "global_corpus_path", "") or "").strip()
    prebuilt_igraph_path = str(getattr(cfg, "prebuilt_igraph_path", "") or "").strip()
    if not corpus_path and not prebuilt_igraph_path:
        return None, {}

    cache_dir = str(getattr(cfg, "graph_cache_dir", "") or "outputs/index_cache").strip()
    force_rebuild = bool(getattr(cfg, "force_rebuild_graph_index", False))
    prebuilt_igraph_format = str(getattr(cfg, "prebuilt_igraph_format", "hipporag_pickle") or "hipporag_pickle").strip()
    prebuilt_entity_token_limit = int(getattr(cfg, "prebuilt_entity_token_limit", 6))
    graph_mode = str(getattr(cfg, "graph_mode", "current_entity_graph") or "current_entity_graph").strip().lower()
    if graph_mode not in {"current_entity_graph", "entity_chunk_graph"}:
        graph_mode = "current_entity_graph"
    index_chunk_unit = str(getattr(cfg, "index_chunk_unit", "auto") or "auto").strip().lower()
    if index_chunk_unit in {"", "auto"}:
        index_chunk_unit = "passage" if graph_mode == "entity_chunk_graph" else "sentence"
    elif index_chunk_unit in {"passage", "chunk", "document", "doc"}:
        index_chunk_unit = "passage"
    else:
        index_chunk_unit = "sentence"
    openie_mode = str(getattr(cfg, "openie_mode", "llm") or "llm").strip().lower()
    openie_model_name = str(getattr(cfg, "openie_model_name", "") or "").strip()
    openie_text_max_chars = int(getattr(cfg, "openie_text_max_chars", 2200))
    openie_max_new_tokens = int(getattr(cfg, "openie_max_new_tokens", 256))
    openie_local_files_only = bool(getattr(cfg, "openie_local_files_only", True))
    openie_retry_attempts = int(getattr(cfg, "openie_retry_attempts", 3))
    openie_retry_backoff_sec = float(getattr(cfg, "openie_retry_backoff_sec", 0.2))
    openie_error_sample_limit = int(getattr(cfg, "openie_error_sample_limit", 20))
    openie_api_base_url = str(getattr(cfg, "openie_api_base_url", "") or "").strip()
    openie_api_key = str(getattr(cfg, "openie_api_key", "") or "").strip()
    openie_api_timeout_sec = float(getattr(cfg, "openie_api_timeout_sec", 120.0))
    openie_parallel_workers = int(getattr(cfg, "openie_parallel_workers", 4))
    openie_log_every = int(getattr(cfg, "openie_log_every", 200))
    embedding_enabled = bool(getattr(cfg, "embedding_enabled", False))
    embedding_model_name = str(getattr(cfg, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2")
    embedding_batch_size = int(getattr(cfg, "embedding_batch_size", 16))
    embedding_max_length = int(getattr(cfg, "embedding_max_length", 192))
    embedding_text_max_chars = int(getattr(cfg, "embedding_text_max_chars", 600))
    memo_key = (
        str(Path(corpus_path).resolve()) if corpus_path else "",
        str(Path(prebuilt_igraph_path).resolve()) if prebuilt_igraph_path else "",
        prebuilt_igraph_format,
        prebuilt_entity_token_limit,
        graph_mode,
        index_chunk_unit,
        embedding_enabled,
        embedding_model_name,
        embedding_batch_size,
        embedding_max_length,
        embedding_text_max_chars,
        str(Path(cache_dir).resolve()),
        openie_mode,
        openie_model_name,
        openie_text_max_chars,
        openie_max_new_tokens,
        openie_local_files_only,
        openie_retry_attempts,
        openie_retry_backoff_sec,
        openie_error_sample_limit,
        openie_api_base_url,
        openie_api_key,
        openie_api_timeout_sec,
        openie_parallel_workers,
        openie_log_every,
    )

    if (not force_rebuild) and memo_key in _GLOBAL_INDEX_MEMO:
        graph, meta = _GLOBAL_INDEX_MEMO[memo_key]
        meta = dict(meta)
        meta["memory_cache_hit"] = True
        return graph, meta

    graph, meta = load_or_build_global_index(
        corpus_path=corpus_path,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        prebuilt_igraph_path=prebuilt_igraph_path,
        prebuilt_igraph_format=prebuilt_igraph_format,
        prebuilt_entity_token_limit=prebuilt_entity_token_limit,
        embedding_enabled=embedding_enabled,
        embedding_model_name=embedding_model_name,
        embedding_batch_size=embedding_batch_size,
        embedding_max_length=embedding_max_length,
        embedding_text_max_chars=embedding_text_max_chars,
        graph_mode=graph_mode,
        index_chunk_unit=index_chunk_unit,
        openie_mode=openie_mode,
        openie_model_name=openie_model_name,
        openie_text_max_chars=openie_text_max_chars,
        openie_max_new_tokens=openie_max_new_tokens,
        openie_local_files_only=openie_local_files_only,
        openie_retry_attempts=openie_retry_attempts,
        openie_retry_backoff_sec=openie_retry_backoff_sec,
        openie_error_sample_limit=openie_error_sample_limit,
        openie_api_base_url=openie_api_base_url,
        openie_api_key=openie_api_key,
        openie_api_timeout_sec=openie_api_timeout_sec,
        openie_parallel_workers=openie_parallel_workers,
        openie_log_every=openie_log_every,
    )
    _GLOBAL_INDEX_MEMO[memo_key] = (graph, dict(meta))
    meta = dict(meta)
    meta["memory_cache_hit"] = False
    return graph, meta


def _get_graph_struct(g):
    try:
        cached = _PPR_GRAPH_STRUCT_MEMO.get(g)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    # Fallback path for environments where graph objects cannot be weak-keyed.
    fallback_key = id(g)
    fallback_cached = _PPR_GRAPH_STRUCT_MEMO_FALLBACK.get(fallback_key)
    if fallback_cached is not None:
        ref, struct = fallback_cached
        if ref() is g:
            return struct
        _PPR_GRAPH_STRUCT_MEMO_FALLBACK.pop(fallback_key, None)

    nodes = list(g.nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    neighbors = [[] for _ in range(len(nodes))]
    degrees = [0 for _ in range(len(nodes))]
    for idx, node in enumerate(nodes):
        nbrs = [node_to_idx[nbr] for nbr in g.neighbors(node) if nbr in node_to_idx]
        neighbors[idx] = nbrs
        degrees[idx] = len(nbrs)
    dangling = [idx for idx, deg in enumerate(degrees) if deg == 0]

    struct = {
        "nodes": nodes,
        "node_to_idx": node_to_idx,
        "neighbors": neighbors,
        "degrees": degrees,
        "dangling": dangling,
        "n": len(nodes),
    }
    try:
        _PPR_GRAPH_STRUCT_MEMO[g] = struct
    except TypeError:
        _PPR_GRAPH_STRUCT_MEMO_FALLBACK[fallback_key] = (weakref.ref(g), struct)
    return struct


def _resolve_ppr_engine(cfg, graph_scope, graph):
    raw = str(getattr(cfg, "ppr_engine", "auto") or "auto").strip().lower()
    if raw in {"power", "mc"}:
        return raw

    # auto: keep small local graphs in exact mode, switch large global/prebuilt graphs to stochastic MC.
    if graph_scope in {"global_corpus", "prebuilt_igraph"} and graph.number_of_nodes() >= 50000:
        return "mc"
    return "power"


def _build_anchor_subgraph(g, anchors, hops, max_nodes):
    if int(hops) <= 0 or int(max_nodes) <= 0 or not anchors:
        return g

    visited = set()
    q = deque()
    for anchor in anchors:
        if anchor in g and anchor not in visited:
            visited.add(anchor)
            q.append((anchor, 0))

    while q and len(visited) < int(max_nodes):
        node, depth = q.popleft()
        if depth >= int(hops):
            continue
        for nbr in g.neighbors(node):
            if nbr in visited:
                continue
            visited.add(nbr)
            if len(visited) >= int(max_nodes):
                break
            q.append((nbr, depth + 1))

    if not visited or len(visited) >= g.number_of_nodes():
        return g
    return g.subgraph(list(visited)).copy()


def _power_ppr_idx(struct, source_idx, alpha, max_iter, tol):
    n = int(struct["n"])
    if n <= 0:
        return {}

    teleport = float(alpha)
    damping = 1.0 - float(alpha)
    neighbors = struct["neighbors"]
    degrees = struct["degrees"]
    dangling = struct["dangling"]

    ranks = [0.0 for _ in range(n)]
    ranks[source_idx] = 1.0

    for _ in range(max(1, int(max_iter))):
        new_ranks = [0.0 for _ in range(n)]
        new_ranks[source_idx] = teleport

        if dangling:
            dangling_mass = damping * sum(ranks[idx] for idx in dangling)
            if dangling_mass > 0.0:
                new_ranks[source_idx] += dangling_mass

        for node_idx, prev_score in enumerate(ranks):
            deg = degrees[node_idx]
            if deg <= 0 or prev_score <= 0.0:
                continue
            share = damping * prev_score / float(deg)
            for nbr_idx in neighbors[node_idx]:
                new_ranks[nbr_idx] += share

        err = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
        ranks = new_ranks
        if err < n * float(tol):
            break

    norm = sum(ranks)
    if norm <= 0.0:
        return {source_idx: 1.0}
    return {idx: (score / norm) for idx, score in enumerate(ranks) if score > 0.0}


def _mc_ppr_idx(struct, source_idx, alpha, num_walks, max_steps, seed, min_score):
    n = int(struct["n"])
    if n <= 0:
        return {}

    neighbors = struct["neighbors"]
    rng = random.Random(int(seed))
    walk_count = max(1, int(num_walks))
    step_cap = max(1, int(max_steps))
    teleport = float(alpha)

    visits = {}
    total = 0
    for _ in range(walk_count):
        cur = source_idx
        visits[cur] = visits.get(cur, 0) + 1
        total += 1
        for _ in range(step_cap):
            if rng.random() < teleport:
                cur = source_idx
            else:
                nbrs = neighbors[cur]
                if not nbrs:
                    cur = source_idx
                else:
                    cur = nbrs[rng.randrange(len(nbrs))]
            visits[cur] = visits.get(cur, 0) + 1
            total += 1

    if total <= 0:
        return {source_idx: 1.0}
    cutoff = max(0.0, float(min_score))
    return {idx: (cnt / float(total)) for idx, cnt in visits.items() if (cnt / float(total)) >= cutoff}


def _mc_task(payload):
    if _PPR_PARALLEL_STRUCT is None:
        raise RuntimeError("MC parallel worker has no graph structure.")
    source_idx, alpha, num_walks, max_steps, seed, min_score = payload
    return _mc_ppr_idx(
        struct=_PPR_PARALLEL_STRUCT,
        source_idx=int(source_idx),
        alpha=float(alpha),
        num_walks=int(num_walks),
        max_steps=int(max_steps),
        seed=int(seed),
        min_score=float(min_score),
    )


def _personalized_pagerank(g, source, alpha, max_iter=100, tol=1.0e-6, min_score=0.0):
    if source not in g:
        return {}
    struct = _get_graph_struct(g)
    source_idx = struct["node_to_idx"].get(source)
    if source_idx is None:
        return {}
    idx_scores = _power_ppr_idx(struct, source_idx=source_idx, alpha=alpha, max_iter=max_iter, tol=tol)
    nodes = struct["nodes"]
    cutoff = max(0.0, float(min_score))
    return {nodes[idx]: score for idx, score in idx_scores.items() if score >= cutoff}


def _compute_ppr_batch(g, sources, cfg, engine, run_seed, show_progress, desc):
    struct = _get_graph_struct(g)
    nodes = struct["nodes"]
    min_score = float(getattr(cfg, "ppr_min_score", 0.0))
    alpha = float(getattr(cfg, "ppr_alpha", 0.15))

    if not sources:
        return {}

    if engine == "mc":
        tasks = []
        seed_base = int(getattr(cfg, "random_seed", 42)) * 1000003 + int(run_seed) * 10007
        for s_idx, source in enumerate(sources):
            source_idx = struct["node_to_idx"].get(source)
            if source_idx is None:
                tasks.append(None)
                continue
            tasks.append(
                (
                    int(source_idx),
                    alpha,
                    int(getattr(cfg, "ppr_mc_walks", 512)),
                    int(getattr(cfg, "ppr_mc_max_steps", 24)),
                    seed_base + s_idx * 7919,
                    min_score,
                )
            )

        use_parallel = False
        ppr_workers = max(1, int(getattr(cfg, "ppr_parallel_workers", 1)))
        # Avoid nested process pools when dataset-level workers are already used.
        if int(getattr(cfg, "num_workers", 1)) <= 1 and ppr_workers > 1:
            try:
                use_parallel = mp.get_start_method(allow_none=True) == "fork"
            except Exception:
                use_parallel = False

        outputs = []
        if use_parallel:
            global _PPR_PARALLEL_STRUCT
            _PPR_PARALLEL_STRUCT = struct
            valid_tasks = [task for task in tasks if task is not None]
            with ProcessPoolExecutor(max_workers=ppr_workers) as ex:
                valid_outputs = list(
                    tqdm(
                        ex.map(_mc_task, valid_tasks),
                        total=len(valid_tasks),
                        desc=desc,
                        leave=False,
                        disable=not show_progress,
                    )
                )
            _PPR_PARALLEL_STRUCT = None
            it = iter(valid_outputs)
            for task in tasks:
                outputs.append({} if task is None else next(it))
        else:
            for task in tqdm(
                tasks,
                total=len(tasks),
                desc=desc,
                leave=False,
                disable=not show_progress,
            ):
                if task is None:
                    outputs.append({})
                else:
                    outputs.append(_mc_task(task) if _PPR_PARALLEL_STRUCT is not None else _mc_ppr_idx(struct, *task))

        mapped = {}
        for source, idx_scores in zip(sources, outputs):
            mapped[source] = {nodes[idx]: score for idx, score in idx_scores.items()}
        return mapped

    mapped = {}
    for source in tqdm(
        sources,
        total=len(sources),
        desc=desc,
        leave=False,
        disable=not show_progress,
    ):
        if source not in struct["node_to_idx"]:
            mapped[source] = {}
            continue
        idx_scores = _power_ppr_idx(
            struct=struct,
            source_idx=int(struct["node_to_idx"][source]),
            alpha=alpha,
            max_iter=int(getattr(cfg, "ppr_power_max_iter", 100)),
            tol=float(getattr(cfg, "ppr_power_tol", 1.0e-6)),
        )
        mapped[source] = {nodes[idx]: score for idx, score in idx_scores.items() if score >= min_score}
    return mapped


def _topk_nodes(scores, k):
    if k <= 0:
        return []
    return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def _stochastic_perturb_graph(g, rng, drop_prob):
    if float(drop_prob) <= 0.0:
        return g
    h = g.copy()
    if h.number_of_edges() == 0:
        return h

    for u, v in list(h.edges()):
        if rng.random() < drop_prob:
            h.remove_edge(u, v)

    if h.number_of_edges() == 0:
        return g.copy()
    return h


def _get_distance_map(g, source, cutoff, distance_map_cache=None):
    if source not in g:
        return {}
    cap = max(1, int(cutoff))
    if isinstance(distance_map_cache, dict):
        key = (source, cap)
        cached = distance_map_cache.get(key)
        if isinstance(cached, dict):
            return cached
    try:
        dmap = nx.single_source_shortest_path_length(g, source, cutoff=cap)
    except Exception:
        dmap = {source: 0}
    if isinstance(distance_map_cache, dict):
        distance_map_cache[(source, cap)] = dmap
    return dmap


def _min_set_distance(g, node, seeds, tau, distance_map_cache=None):
    cap = max(1, int(tau))
    if not seeds or node not in g:
        return cap

    best = cap
    for z in seeds:
        if z not in g:
            continue
        if z == node:
            return 0
        dmap = _get_distance_map(g, z, cap, distance_map_cache=distance_map_cache)
        dist = int(dmap.get(node, cap))
        if dist < best:
            best = dist
            if best <= 0:
                break
    return min(best, cap)


def _greedy_seed_set(g, candidates, weights, seed_k, tau, distance_map_cache=None):
    # Same greedy objective as before, but precompute candidate-to-candidate
    # bounded distances once so we avoid repeated BFS calls in the inner loop.
    candidate_list = [node for node in _ordered_unique(candidates) if node in g]
    if not candidate_list:
        return set()

    k = max(1, int(seed_k))
    cap = max(1, int(tau))
    n = len(candidate_list)

    w = np.asarray([float(weights.get(node, 0.0)) for node in candidate_list], dtype=np.float32)
    if w.size <= 0:
        w = np.ones((n,), dtype=np.float32)

    dist_mat = np.full((n, n), float(cap), dtype=np.float32)
    node_to_idx = {node: idx for idx, node in enumerate(candidate_list)}
    for seed_node, j in node_to_idx.items():
        dmap = _get_distance_map(g, seed_node, cap, distance_map_cache=distance_map_cache)
        dist_mat[j, j] = 0.0
        for node, i in node_to_idx.items():
            if node == seed_node:
                continue
            dist_mat[i, j] = float(min(cap, int(dmap.get(node, cap))))

    selected_idx = []
    remaining_idx = set(range(n))
    current_best = np.full((n,), float(cap), dtype=np.float32)

    while len(selected_idx) < k and remaining_idx:
        best_idx = None
        best_obj = float("inf")
        for idx in remaining_idx:
            trial_best = np.minimum(current_best, dist_mat[:, idx])
            obj = float(np.dot(w, trial_best))
            if obj < best_obj:
                best_obj = obj
                best_idx = idx
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)
        current_best = np.minimum(current_best, dist_mat[:, best_idx])

    return {candidate_list[i] for i in selected_idx}


def _pair_support(anchor_scores, seed_scores, node):
    return sqrt(anchor_scores.get(node, 0.0) * seed_scores.get(node, 0.0))


def _ordered_unique(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _normalize_map(scores):
    if not scores:
        return {}
    vals = [float(v) for v in scores.values()]
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        if hi > 0.0:
            return {k: 1.0 for k in scores}
        return {k: 0.0 for k in scores}
    span = hi - lo
    return {k: (float(v) - lo) / span for k, v in scores.items()}


def _entity_surface_tokens_from_node(node_id):
    raw = str(node_id or "")
    if raw.startswith("e::"):
        raw = raw[3:]
    raw = raw.replace("_", " ")
    return [str(t).strip().lower() for t in content_tokens(raw) if str(t).strip()]


def _augment_anchors_with_semantic(anchors, semantic_entity_scores, sample, g, cfg):
    base_anchors = [str(a) for a in list(anchors or []) if str(a) in g]
    max_anchors = max(1, int(getattr(cfg, "max_anchors", 5)))
    add_topn = max(0, int(getattr(cfg, "hybrid_anchor_semantic_topn", 2)))
    if add_topn <= 0 or not semantic_entity_scores:
        return base_anchors[:max_anchors], {
            "applied": False,
            "reason": "disabled_or_no_semantic_entities",
            "added": [],
        }

    q_tokens = set(content_tokens(str(getattr(sample, "question", "") or "")))
    alias_boost = float(getattr(cfg, "hybrid_anchor_alias_boost", 0.10))
    ranked = []
    for node, raw_score in (semantic_entity_scores or {}).items():
        node_id = str(node)
        if node_id not in g:
            continue
        if str((g.nodes[node_id] or {}).get("node_type", "")).strip().lower() != "entity":
            continue
        sem = 0.5 * (float(raw_score) + 1.0)
        alias_tokens = set(_entity_surface_tokens_from_node(node_id))
        overlap = 1.0 if (q_tokens and alias_tokens.intersection(q_tokens)) else 0.0
        support_freq = 0
        for nbr in g.neighbors(node_id):
            if _is_chunk_like_node(g, nbr):
                support_freq += 1
        rarity = 1.0 / (1.0 + float(support_freq))
        score = 0.70 * float(sem) + float(alias_boost) * float(overlap) + 0.20 * float(rarity)
        ranked.append((node_id, float(score)))
    ranked.sort(key=lambda x: x[1], reverse=True)

    added = []
    merged = list(base_anchors)
    for node, _ in ranked[: max(add_topn * 6, 24)]:
        if node in merged:
            continue
        merged.append(node)
        added.append(node)
        if len(added) >= add_topn:
            break

    if len(merged) > max_anchors:
        keep_priority = set(base_anchors[:max_anchors])
        merged_ranked = []
        for node in merged:
            pr = 1.0 if node in keep_priority else 0.0
            sem = float(semantic_entity_scores.get(node, 0.0))
            merged_ranked.append((node, pr, sem))
        merged_ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        merged = [n for n, _, _ in merged_ranked[:max_anchors]]
        added = [n for n in added if n in set(merged)]

    return merged, {
        "applied": bool(len(added) > 0),
        "max_anchors": int(max_anchors),
        "semantic_add_topn": int(add_topn),
        "added": [str(a) for a in added],
        "final_anchors": [str(a) for a in merged],
    }


def _induce_bridge_candidates(g, anchors, semantic_scores, cfg):
    a_list = [str(a) for a in list(anchors or []) if str(a) in g]
    if len(a_list) < 2:
        return set(), {}, {
            "applied": False,
            "reason": "not_enough_anchors",
            "candidate_count": 0,
            "pair_count": 0,
        }

    tau = max(2, int(getattr(cfg, "bridge_candidate_max_hops", getattr(cfg, "tau", 4))))
    per_pair_topn = max(1, int(getattr(cfg, "bridge_candidate_per_pair_topn", 8)))
    global_topn = max(per_pair_topn, int(getattr(cfg, "bridge_candidate_global_topn", 48)))
    w_sem = max(0.0, float(getattr(cfg, "bridge_candidate_semantic_weight", 0.45)))
    w_path = max(0.0, float(getattr(cfg, "bridge_candidate_path_weight", 0.35)))
    w_deg = max(0.0, float(getattr(cfg, "bridge_candidate_degree_weight", 0.20)))
    w_sum = w_sem + w_path + w_deg
    if w_sum <= 0.0:
        w_sem, w_path, w_deg, w_sum = 0.45, 0.35, 0.20, 1.0
    w_sem /= w_sum
    w_path /= w_sum
    w_deg /= w_sum

    dmaps = {}
    for anchor in a_list:
        dmaps[anchor] = _get_distance_map(g, anchor, tau, distance_map_cache={})

    scores = {}
    pair_count = 0
    for i in range(len(a_list)):
        for j in range(i + 1, len(a_list)):
            pair_count += 1
            a = a_list[i]
            b = a_list[j]
            da = dmaps.get(a, {})
            db = dmaps.get(b, {})
            local = []
            shared_nodes = set(da.keys()).intersection(set(db.keys()))
            for node in shared_nodes:
                if node in {a, b}:
                    continue
                if node not in g:
                    continue
                ntype = str((g.nodes[node] or {}).get("node_type", "")).strip().lower()
                if (ntype != "entity") and (not _is_chunk_like_node(g, node)):
                    continue
                d1 = int(da.get(node, tau + 1))
                d2 = int(db.get(node, tau + 1))
                if d1 > tau or d2 > tau:
                    continue
                path_consistency = 1.0 - (float(d1 + d2) / float(max(1, 2 * tau)))
                sem = 0.5 * (float(semantic_scores.get(node, 0.0)) + 1.0)
                deg = min(1.0, float(g.degree(node)) / 20.0)
                score = w_sem * float(sem) + w_path * float(path_consistency) + w_deg * float(deg)
                local.append((str(node), float(score)))
            local.sort(key=lambda x: x[1], reverse=True)
            for node, score in local[:per_pair_topn]:
                scores[node] = max(float(scores.get(node, -1.0e9)), float(score))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:global_topn]
    out_nodes = {str(node) for node, _ in ranked if str(node) in g}
    out_scores = {str(node): float(score) for node, score in ranked if str(node) in g}
    return out_nodes, out_scores, {
        "applied": bool(len(out_nodes) > 0),
        "candidate_count": int(len(out_nodes)),
        "pair_count": int(pair_count),
        "tau": int(tau),
        "per_pair_topn": int(per_pair_topn),
        "global_topn": int(global_topn),
    }


def _node_semantic_text(g, node):
    if node not in g:
        return ""
    data = g.nodes[node]
    node_type = str(data.get("node_type", "") or "")
    if node_type == "sentence" or _is_chunk_like_node(g, node):
        title = str(data.get("title", "") or "").strip()
        text = str(data.get("text", data.get("content", "")) or "").strip()
        if title:
            return f"Title: {title}\nPassage: {text}".strip()
        return text
    if node_type == "entity":
        token = str(data.get("token", "") or "").strip()
        content = str(data.get("content", "") or "").strip()
        label = token or content or str(node).split("::", 1)[-1]
        support = []
        for nbr in g.neighbors(node):
            nbr_data = g.nodes[nbr]
            if _is_chunk_like_node(g, nbr):
                txt = str(nbr_data.get("text", "") or "").strip()
                if txt:
                    support.append(txt)
            if len(support) >= 2:
                break
        parts = [f"Entity: {label}"]
        if support:
            parts.append("Context: " + " ".join(support))
        return "\n".join(parts).strip()
    if node_type == "relation":
        subj = str(data.get("subject", "") or "").strip()
        pred = str(data.get("predicate", "") or "").strip()
        obj = str(data.get("object", "") or "").strip()
        return " ".join([x for x in [subj, pred, obj] if x]).strip()
    return str(node)


def _semantic_state_cache_key(global_index_meta):
    semantic_meta = (global_index_meta or {}).get("semantic_index", {}) or {}
    return (
        str(semantic_meta.get("index_dir", "")),
        str(semantic_meta.get("entity_embeddings_path", "")),
        str(semantic_meta.get("chunk_embeddings_path", "")),
        str(semantic_meta.get("entity_ids_path", "")),
        str(semantic_meta.get("chunk_ids_path", "")),
        str(semantic_meta.get("model_name", "")),
    )


def _get_global_semantic_state(global_index_meta):
    semantic_meta = (global_index_meta or {}).get("semantic_index", {}) or {}
    if not bool(semantic_meta.get("enabled", False)):
        return None
    key = _semantic_state_cache_key(global_index_meta)
    if key in _GLOBAL_SEMANTIC_MEMO:
        return _GLOBAL_SEMANTIC_MEMO[key]
    try:
        state = load_semantic_index(semantic_meta, mmap_mode="r")
    except Exception:
        state = None
    if state is not None:
        _GLOBAL_SEMANTIC_MEMO[key] = state
    return state


def _normalize_query_text(question, max_chars):
    text = " ".join(str(question or "").strip().split())
    if int(max_chars) > 0 and len(text) > int(max_chars):
        text = text[: int(max_chars)]
    return text


def _query_embedding_cache_path(cfg, model_name, max_length, max_chars, text):
    cache_dir = str(getattr(cfg, "query_embedding_cache_dir", "outputs/query_embedding_cache") or "outputs/query_embedding_cache").strip()
    payload = "|".join(
        [
            str(model_name),
            str(int(max_length)),
            str(int(max_chars)),
            str(text),
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.npy"


def _atomic_save_npy(path, array):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}.npy"
    try:
        np.save(tmp, np.asarray(array, dtype=np.float32))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def _query_embedding(question, cfg):
    model_name = str(getattr(cfg, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2")
    max_length = int(getattr(cfg, "query_embedding_max_length", 96))
    max_chars = int(getattr(cfg, "query_embedding_text_max_chars", 384))
    use_cache = bool(getattr(cfg, "query_embedding_cache_enabled", True))
    query_text = _normalize_query_text(question, max_chars=max_chars)
    if not query_text:
        query_text = str(question or "")

    memo_key = (
        str(model_name),
        int(max_length),
        int(max_chars),
        str(query_text),
    )
    if use_cache and memo_key in _QUERY_EMBED_MEMO:
        cached = _QUERY_EMBED_MEMO[memo_key]
        return np.asarray(cached, dtype=np.float32), "", False, True

    cache_path = _query_embedding_cache_path(
        cfg=cfg,
        model_name=model_name,
        max_length=max_length,
        max_chars=max_chars,
        text=query_text,
    )
    if use_cache and cache_path.exists():
        try:
            arr = np.asarray(np.load(cache_path), dtype=np.float32)
            if arr.size > 0:
                norm = float(np.linalg.norm(arr))
                if norm > 0.0:
                    vec = arr / norm
                    _QUERY_EMBED_MEMO[memo_key] = vec
                    return vec, "", False, True
        except Exception:
            pass

    out = encode_texts(
        texts=[query_text],
        model_name=model_name,
        batch_size=1,
        max_length=max_length,
        max_chars=max_chars,
        instruction="Given a question, retrieve relevant phrases that are mentioned in this question.",
    )
    if not out.get("ok", False):
        return None, str(out.get("error", "query_embedding_failed")), True, False
    vectors = out.get("vectors", []) or []
    if not vectors:
        return None, "query_embedding_empty", True, False
    qvec = np.asarray(vectors[0], dtype=np.float32)
    norm = float(np.linalg.norm(qvec))
    if norm <= 0.0:
        return None, "query_embedding_zero_norm", True, False
    qvec = qvec / norm
    if use_cache:
        _QUERY_EMBED_MEMO[memo_key] = qvec
        try:
            _atomic_save_npy(cache_path, qvec.astype(np.float32, copy=False))
        except Exception:
            pass
    return qvec, "", True, False


def _semantic_top_candidates(query_vec, semantic_state, cfg):
    if semantic_state is None or query_vec is None:
        return [], {}
    topn = max(0, int(getattr(cfg, "semantic_topn", 50)))
    if topn <= 0:
        return [], {}
    scan_bs = max(128, int(getattr(cfg, "semantic_scan_batch_size", 8192)))
    scores = {}
    ent_idx, ent_scores = topk_cosine_similarity(
        query_vector=query_vec,
        matrix=semantic_state.get("entity_embeddings"),
        topn=topn,
        scan_batch_size=scan_bs,
    )
    for idx, score in zip(ent_idx.tolist(), ent_scores.tolist()):
        if idx < 0 or idx >= len(semantic_state.get("entity_ids", [])):
            continue
        node = str(semantic_state["entity_ids"][idx])
        scores[node] = max(float(score), scores.get(node, -1.0))

    chunk_idx, chunk_scores = topk_cosine_similarity(
        query_vector=query_vec,
        matrix=semantic_state.get("chunk_embeddings"),
        topn=topn,
        scan_batch_size=scan_bs,
    )
    for idx, score in zip(chunk_idx.tolist(), chunk_scores.tolist()):
        if idx < 0 or idx >= len(semantic_state.get("chunk_ids", [])):
            continue
        node = str(semantic_state["chunk_ids"][idx])
        scores[node] = max(float(score), scores.get(node, -1.0))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    return [node for node, _ in ranked], {node: float(score) for node, score in ranked}


def _normalize_lexical_key(text):
    raw = str(text or "")
    if not raw:
        return ""
    norm = "".join(ch.lower() if ch.isalnum() else " " for ch in raw)
    return " ".join(norm.split())


def _entity_shortlist_tier1(query_text, anchors, g, semantic_state, cfg):
    if semantic_state is None or g is None:
        return [], {"tier1_candidate_count": 0, "tier1_shortlist_count": 0}
    tier1_cap = max(32, int(getattr(cfg, "entity_lookup_tier1_topk", 256)))
    alias_token_limit = max(4, int(getattr(cfg, "entity_lookup_alias_token_limit", 12)))
    ent2chk = semantic_state.get("entity_topk_chunks_cache", {}) if isinstance(semantic_state, dict) else {}
    chk2ent = semantic_state.get("chunk_topk_entities_cache", {}) if isinstance(semantic_state, dict) else {}
    ent2chk = ent2chk if isinstance(ent2chk, dict) else {}
    chk2ent = chk2ent if isinstance(chk2ent, dict) else {}
    ent_idx = semantic_state.get("entity_id_to_idx", {}) if isinstance(semantic_state, dict) else {}
    ent_idx = ent_idx if isinstance(ent_idx, dict) else {}

    scores = {}

    def add_entity(node, score):
        node_id = str(node)
        if node_id not in ent_idx:
            return
        if node_id not in g:
            return
        if g.nodes[node_id].get("node_type") != "entity":
            return
        scores[node_id] = max(float(scores.get(node_id, -1.0e9)), float(score))

    for arank, anchor in enumerate(anchors):
        if anchor not in g:
            continue
        a_type = g.nodes[anchor].get("node_type")
        if a_type == "entity":
            add_entity(anchor, 1.00 - 0.05 * float(arank))
        elif _is_chunk_like_node(g, anchor):
            for erank, ent in enumerate(list(chk2ent.get(str(anchor), []) or [])[:8]):
                add_entity(ent, 0.90 - 0.03 * float(erank) - 0.02 * float(arank))

    lexical_keys = set()
    q_norm = _normalize_lexical_key(query_text)
    if q_norm:
        lexical_keys.add(q_norm)
    for anchor in anchors:
        if anchor not in g:
            continue
        a_data = g.nodes[anchor]
        label = str(a_data.get("token", "") or a_data.get("content", "") or anchor).strip()
        a_norm = _normalize_lexical_key(label)
        if a_norm:
            lexical_keys.add(a_norm)

    token_pool = []
    if q_norm:
        token_pool.extend(q_norm.split()[:alias_token_limit])
    for key in list(lexical_keys):
        token_pool.extend(key.split()[: max(1, alias_token_limit // 2)])
    for tok in token_pool:
        if len(tok) >= 3:
            lexical_keys.add(tok)

    for key in list(lexical_keys):
        if not key:
            continue
        alias_node = f"e::{key}"
        if alias_node in g and g.nodes[alias_node].get("node_type") == "entity":
            src_type = str(g.nodes[alias_node].get("source_node_type", "") or "")
            if src_type == "alias":
                linked = [
                    str(n)
                    for n in g.neighbors(alias_node)
                    if g.nodes[n].get("node_type") == "entity"
                    and str(g.nodes[n].get("source_node_type", "") or "") != "alias"
                ]
                if linked:
                    for lrank, node in enumerate(linked[:8]):
                        add_entity(node, 0.95 - 0.03 * float(lrank))
                else:
                    add_entity(alias_node, 0.90)
            else:
                add_entity(alias_node, 0.90)

        if key in g and g.nodes[key].get("node_type") == "entity":
            add_entity(key, 0.88)

    seed_entities = [node for node, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:32]]
    for srank, ent in enumerate(seed_entities):
        linked_chunks = list(ent2chk.get(str(ent), []) or [])
        for crank, chunk in enumerate(linked_chunks[:8]):
            linked_entities = list(chk2ent.get(str(chunk), []) or [])
            for erank, linked_ent in enumerate(linked_entities[:6]):
                add_entity(
                    linked_ent,
                    0.80 - 0.02 * float(srank) - 0.03 * float(crank) - 0.02 * float(erank),
                )

    for arank, anchor in enumerate(anchors):
        if anchor not in g:
            continue
        direct_neighbors = list(g.neighbors(anchor))
        for nbr in direct_neighbors[:64]:
            ntype = g.nodes[nbr].get("node_type")
            if ntype == "entity":
                add_entity(nbr, 0.72 - 0.02 * float(arank))
            elif ntype == "relation":
                for nbr2 in g.neighbors(nbr):
                    if g.nodes[nbr2].get("node_type") == "entity":
                        add_entity(nbr2, 0.68 - 0.02 * float(arank))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    shortlist = [node for node, _ in ranked[:tier1_cap]]
    return shortlist, {
        "tier1_candidate_count": int(len(scores)),
        "tier1_shortlist_count": int(len(shortlist)),
    }


def _entity_lookup_two_tier(query_vec, semantic_state, g, anchors, query_text, cfg):
    topn_entity = max(
        0,
        int(
            getattr(
                cfg,
                "semantic_topn_entity",
                getattr(cfg, "semantic_topn", 50),
            )
        ),
    )
    if topn_entity <= 0 or semantic_state is None:
        return {}, {"entity_ms": 0.0, "entity_mode": "disabled", "tier1_candidate_count": 0, "tier1_shortlist_count": 0}

    start = time.perf_counter()
    shortlist, shortlist_diag = _entity_shortlist_tier1(
        query_text=query_text,
        anchors=anchors,
        g=g,
        semantic_state=semantic_state,
        cfg=cfg,
    )

    entity_id_to_idx = semantic_state.get("entity_id_to_idx", {}) if isinstance(semantic_state, dict) else {}
    entity_ids = [node for node in shortlist if node in entity_id_to_idx]
    entity_scores = {}
    mode = "two_tier_shortlist"

    if entity_ids:
        idx = [int(entity_id_to_idx[node]) for node in entity_ids]
        vec_mat = np.asarray(semantic_state.get("entity_embeddings")[idx], dtype=np.float32)
        qvec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        sims = np.matmul(vec_mat, qvec)
        for node, sim in zip(entity_ids, sims.tolist()):
            entity_scores[node] = float(sim)

    fallback_topn = max(0, int(getattr(cfg, "entity_lookup_global_fallback_topn", 8)))
    min_needed = max(1, int(min(topn_entity, fallback_topn))) if fallback_topn > 0 else topn_entity
    need_fallback = int(len(entity_scores)) < int(min_needed)
    scan_bs = max(128, int(getattr(cfg, "semantic_scan_batch_size", 8192)))
    if need_fallback and fallback_topn > 0:
        idx, sims = topk_cosine_similarity(
            query_vector=query_vec,
            matrix=semantic_state.get("entity_embeddings"),
            topn=max(topn_entity, fallback_topn),
            scan_batch_size=scan_bs,
        )
        for i, score in zip(idx.tolist(), sims.tolist()):
            if i < 0 or i >= len(semantic_state.get("entity_ids", [])):
                continue
            node = str(semantic_state["entity_ids"][i])
            if node in g and g.nodes[node].get("node_type") == "entity":
                entity_scores[node] = max(float(entity_scores.get(node, -1.0e9)), float(score))
        mode = "two_tier+global_fallback"

    ranked = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:topn_entity]
    elapsed_ms = float((time.perf_counter() - start) * 1000.0)
    return {node: float(score) for node, score in ranked}, {
        "entity_ms": elapsed_ms,
        "entity_mode": mode,
        "tier1_candidate_count": int(shortlist_diag.get("tier1_candidate_count", 0)),
        "tier1_shortlist_count": int(shortlist_diag.get("tier1_shortlist_count", 0)),
        "tier1_dense_count": int(len(entity_ids)),
    }


def _dense_rerank_from_shortlist(query_vec, matrix, candidate_indices, topn):
    if query_vec is None or matrix is None:
        return [], []
    if not candidate_indices:
        return [], []
    k = max(0, int(topn))
    if k <= 0:
        return [], []
    uniq = sorted({int(i) for i in candidate_indices if int(i) >= 0})
    if not uniq:
        return [], []

    idx_arr = np.asarray(uniq, dtype=np.int64)
    vec_mat = np.asarray(matrix[idx_arr], dtype=np.float32)
    if vec_mat.size <= 0:
        return [], []
    qvec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    sims = np.matmul(vec_mat, qvec)
    if sims.size <= 0:
        return [], []

    topk = int(min(len(uniq), k))
    if topk <= 0:
        return [], []
    if topk >= len(uniq):
        top_local = np.arange(len(uniq), dtype=np.int64)
    else:
        top_local = np.argpartition(sims, -topk)[-topk:]
    top_local = top_local[np.argsort(sims[top_local])[::-1]]
    out_idx = idx_arr[top_local]
    out_scores = sims[top_local]
    return out_idx.tolist(), out_scores.tolist()


def _chunk_shortlist_tier1(query_text, anchors, g, semantic_state, entity_scores, cfg, prior_scores=None):
    if semantic_state is None or g is None:
        return [], {"tier1_candidate_count": 0, "tier1_shortlist_count": 0}

    topn_chunk = max(
        0,
        int(
            getattr(
                cfg,
                "semantic_topn_chunk",
                max(1, int(getattr(cfg, "semantic_topn", 50)) // 2),
            )
        ),
    )
    if topn_chunk <= 0:
        return [], {"tier1_candidate_count": 0, "tier1_shortlist_count": 0}
    tier1_cap = max(topn_chunk * 4, int(getattr(cfg, "chunk_lookup_tier1_topk", 256)))
    neighbor_cap = max(8, int(getattr(cfg, "chunk_lookup_anchor_cache_topn", 8)))

    ent2chk = semantic_state.get("entity_topk_chunks_cache", {}) if isinstance(semantic_state, dict) else {}
    chk2ent = semantic_state.get("chunk_topk_entities_cache", {}) if isinstance(semantic_state, dict) else {}
    support_map = semantic_state.get("entity_to_chunks", {}) if isinstance(semantic_state, dict) else {}
    chunk_idx = semantic_state.get("chunk_id_to_idx", {}) if isinstance(semantic_state, dict) else {}
    ent2chk = ent2chk if isinstance(ent2chk, dict) else {}
    chk2ent = chk2ent if isinstance(chk2ent, dict) else {}
    support_map = support_map if isinstance(support_map, dict) else {}
    chunk_idx = chunk_idx if isinstance(chunk_idx, dict) else {}

    scores = {}

    def add_chunk(node, score):
        node_id = str(node)
        if node_id not in chunk_idx:
            return
        if node_id not in g:
            return
        if not _is_chunk_like_node(g, node_id):
            return
        scores[node_id] = max(float(scores.get(node_id, -1.0e9)), float(score))

    for node, score in (prior_scores or {}).items():
        add_chunk(node, float(score))

    seed_entities = [str(n) for n, _ in sorted((entity_scores or {}).items(), key=lambda x: x[1], reverse=True)[:24]]
    for erank, ent in enumerate(seed_entities):
        linked = []
        linked.extend(list(ent2chk.get(ent, []) or []))
        linked.extend(list(support_map.get(ent, []) or []))
        seen = set()
        compact = []
        for chunk in linked:
            chunk_id = str(chunk)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            compact.append(chunk_id)
        for crank, chunk in enumerate(compact[:12]):
            base = 0.92 - 0.03 * float(crank) - 0.02 * float(erank)
            add_chunk(chunk, base)
            linked_entities = list(chk2ent.get(str(chunk), []) or [])
            for irank, ent2 in enumerate(linked_entities[:3]):
                for jrank, chunk2 in enumerate(list(ent2chk.get(str(ent2), []) or [])[:2]):
                    add_chunk(chunk2, 0.78 - 0.03 * float(irank) - 0.02 * float(jrank))

    q_tokens = set(content_tokens(query_text))
    for arank, anchor in enumerate(anchors):
        if anchor not in g:
            continue
        a_type = g.nodes[anchor].get("node_type")
        if _is_chunk_like_node(g, anchor):
            add_chunk(anchor, 0.90 - 0.03 * float(arank))
        elif a_type == "entity":
            linked = []
            linked.extend(list(ent2chk.get(str(anchor), []) or []))
            linked.extend(list(support_map.get(str(anchor), []) or []))
            seen = set()
            compact = []
            for chunk in linked:
                chunk_id = str(chunk)
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                compact.append(chunk_id)
            for crank, chunk in enumerate(compact[:neighbor_cap]):
                add_chunk(chunk, 0.88 - 0.03 * float(crank) - 0.02 * float(arank))

        direct = list(g.neighbors(anchor))
        added = 0
        for nbr in direct:
            ntype = g.nodes[nbr].get("node_type")
            if _is_chunk_like_node(g, nbr):
                label = str(g.nodes[nbr].get("text", g.nodes[nbr].get("content", "")) or "")
                overlap = float(len(q_tokens.intersection(set(content_tokens(label))))) if q_tokens else 0.0
                add_chunk(nbr, 0.72 + min(0.18, 0.04 * overlap))
                added += 1
            elif ntype == "relation":
                for nbr2 in g.neighbors(nbr):
                    if _is_chunk_like_node(g, nbr2):
                        add_chunk(nbr2, 0.68)
                        added += 1
            if added >= neighbor_cap:
                break

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    shortlist = [node for node, _ in ranked[:tier1_cap]]
    return shortlist, {
        "tier1_candidate_count": int(len(scores)),
        "tier1_shortlist_count": int(len(shortlist)),
    }


def _chunk_lookup_two_tier(query_vec, semantic_state, g, anchors, query_text, entity_scores, cfg, prior_scores=None, allow_global_fallback=True):
    topn_chunk = max(
        0,
        int(
            getattr(
                cfg,
                "semantic_topn_chunk",
                max(1, int(getattr(cfg, "semantic_topn", 50)) // 2),
            )
        ),
    )
    if topn_chunk <= 0 or semantic_state is None:
        return {}, {
            "chunk_ms": 0.0,
            "chunk_mode": "disabled",
            "chunk_tier1_candidate_count": 0,
            "chunk_tier1_shortlist_count": 0,
            "chunk_tier1_dense_count": 0,
        }

    start = time.perf_counter()
    shortlist, shortlist_diag = _chunk_shortlist_tier1(
        query_text=query_text,
        anchors=anchors,
        g=g,
        semantic_state=semantic_state,
        entity_scores=entity_scores,
        cfg=cfg,
        prior_scores=prior_scores,
    )
    chunk_idx = semantic_state.get("chunk_id_to_idx", {}) if isinstance(semantic_state, dict) else {}
    shortlist_idx = [int(chunk_idx[node]) for node in shortlist if node in chunk_idx]
    chunk_scores = {}

    dense_cap = max(topn_chunk, int(getattr(cfg, "chunk_lookup_dense_topk", max(topn_chunk * 4, 64))))
    local_idx, local_scores = _dense_rerank_from_shortlist(
        query_vec=query_vec,
        matrix=semantic_state.get("chunk_embeddings"),
        candidate_indices=shortlist_idx,
        topn=dense_cap,
    )
    for idx, score in zip(local_idx, local_scores):
        if idx < 0 or idx >= len(semantic_state.get("chunk_ids", [])):
            continue
        node = str(semantic_state["chunk_ids"][idx])
        chunk_scores[node] = max(float(chunk_scores.get(node, -1.0e9)), float(score))

    for node, score in (prior_scores or {}).items():
        chunk_scores[str(node)] = max(float(chunk_scores.get(str(node), -1.0e9)), float(score))

    mode = "two_tier_shortlist"
    fallback_topn = max(0, int(getattr(cfg, "chunk_lookup_global_fallback_topn", 6)))
    min_needed = max(1, int(min(topn_chunk, fallback_topn))) if fallback_topn > 0 else topn_chunk
    need_fallback = int(len(chunk_scores)) < int(min_needed)
    scan_bs = max(128, int(getattr(cfg, "semantic_scan_batch_size", 8192)))
    if allow_global_fallback and need_fallback and fallback_topn > 0:
        idx, sims = topk_cosine_similarity(
            query_vector=query_vec,
            matrix=semantic_state.get("chunk_embeddings"),
            topn=max(topn_chunk, fallback_topn),
            scan_batch_size=scan_bs,
        )
        for i, score in zip(idx.tolist(), sims.tolist()):
            if i < 0 or i >= len(semantic_state.get("chunk_ids", [])):
                continue
            node = str(semantic_state["chunk_ids"][i])
            if node in g and _is_chunk_like_node(g, node):
                chunk_scores[node] = max(float(chunk_scores.get(node, -1.0e9)), float(score))
        mode = "two_tier+global_fallback"

    ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:topn_chunk]
    elapsed_ms = float((time.perf_counter() - start) * 1000.0)
    return {node: float(score) for node, score in ranked}, {
        "chunk_ms": elapsed_ms,
        "chunk_mode": mode,
        "chunk_tier1_candidate_count": int(shortlist_diag.get("tier1_candidate_count", 0)),
        "chunk_tier1_shortlist_count": int(shortlist_diag.get("tier1_shortlist_count", 0)),
        "chunk_tier1_dense_count": int(len(shortlist_idx)),
    }


def _semantic_topk_candidates_by_type(query_vec, semantic_state, cfg, g=None, anchors=None, query_text=""):
    if semantic_state is None or query_vec is None:
        return {}, {}, {"entity_ms": 0.0, "chunk_ms": 0.0, "entity_mode": "disabled", "chunk_mode": "disabled"}

    topn_entity = max(
        0,
        int(
            getattr(
                cfg,
                "semantic_topn_entity",
                getattr(cfg, "semantic_topn", 50),
            )
        ),
    )
    topn_chunk = max(
        0,
        int(
            getattr(
                cfg,
                "semantic_topn_chunk",
                max(1, int(getattr(cfg, "semantic_topn", 50)) // 2),
            )
        ),
    )
    scan_bs = max(128, int(getattr(cfg, "semantic_scan_batch_size", 8192)))
    entity_two_tier = bool(getattr(cfg, "entity_lookup_use_two_tier", True))
    anchor_list = [str(a) for a in list(anchors or [])]
    qtext = str(query_text or "")
    chunk_lookup_strategy = str(getattr(cfg, "semantic_chunk_lookup_strategy", "adaptive") or "adaptive").strip().lower()
    if chunk_lookup_strategy not in {"adaptive", "full", "cache_only", "two_tier"}:
        chunk_lookup_strategy = "adaptive"
    cache_min_ratio = float(getattr(cfg, "semantic_chunk_cache_min_ratio", 0.8))
    cache_min_ratio = max(0.0, min(1.0, cache_min_ratio))

    diag = {"entity_ms": 0.0, "chunk_ms": 0.0, "entity_mode": "disabled", "chunk_mode": "disabled"}
    entity_scores = {}
    if topn_entity > 0:
        if entity_two_tier and g is not None:
            entity_scores, ent_diag = _entity_lookup_two_tier(
                query_vec=query_vec,
                semantic_state=semantic_state,
                g=g,
                anchors=anchor_list,
                query_text=qtext,
                cfg=cfg,
            )
            diag["entity_ms"] = float(ent_diag.get("entity_ms", 0.0))
            diag["entity_mode"] = str(ent_diag.get("entity_mode", "two_tier"))
            diag["entity_tier1_candidate_count"] = int(ent_diag.get("tier1_candidate_count", 0))
            diag["entity_tier1_shortlist_count"] = int(ent_diag.get("tier1_shortlist_count", 0))
            diag["entity_tier1_dense_count"] = int(ent_diag.get("tier1_dense_count", 0))
        else:
            ent_start = time.perf_counter()
            ent_idx, ent_scores = topk_cosine_similarity(
                query_vector=query_vec,
                matrix=semantic_state.get("entity_embeddings"),
                topn=topn_entity,
                scan_batch_size=scan_bs,
            )
            for idx, score in zip(ent_idx.tolist(), ent_scores.tolist()):
                if idx < 0 or idx >= len(semantic_state.get("entity_ids", [])):
                    continue
                node = str(semantic_state["entity_ids"][idx])
                entity_scores[node] = max(float(score), float(entity_scores.get(node, -1.0)))
            diag["entity_ms"] = float((time.perf_counter() - ent_start) * 1000.0)
            diag["entity_mode"] = "cache_full_scan"

    entity_scores = dict(sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:topn_entity])

    # Cache-first chunk shortcut from offline entity->chunk support top-k mapping.
    cache_chunk_scores = {}
    support_cache = semantic_state.get("entity_topk_chunks_cache", {}) if isinstance(semantic_state, dict) else {}
    if topn_chunk > 0 and isinstance(support_cache, dict) and entity_scores:
        for ent_rank, (ent_node, ent_score) in enumerate(entity_scores.items()):
            linked_chunks = list(support_cache.get(str(ent_node), []) or [])
            # A small expansion per entity is enough for proposal gating.
            per_ent_limit = max(2, min(topn_chunk, 8))
            for rank, chunk_node in enumerate(linked_chunks[:per_ent_limit]):
                node_id = str(chunk_node)
                if g is not None and node_id not in g:
                    continue
                bonus = float(ent_score) * (0.92 - 0.03 * float(rank)) * (0.96 - 0.02 * float(ent_rank))
                cache_chunk_scores[node_id] = max(float(cache_chunk_scores.get(node_id, -1.0)), float(bonus))
        cache_chunk_scores = dict(sorted(cache_chunk_scores.items(), key=lambda x: x[1], reverse=True)[:topn_chunk])

    # Anchor-aware chunk Tier1 shortcut from cache/support mapping before dense retrieval.
    anchor_cache_topn = max(2, int(getattr(cfg, "chunk_lookup_anchor_cache_topn", 8)))
    if topn_chunk > 0 and g is not None and isinstance(semantic_state, dict) and anchor_list and chunk_lookup_strategy != "full":
        for arank, anchor in enumerate(anchor_list):
            if anchor not in g:
                continue
            anchor_type = g.nodes[anchor].get("node_type")
            if _is_chunk_like_node(g, anchor):
                cache_chunk_scores[str(anchor)] = max(
                    float(cache_chunk_scores.get(str(anchor), -1.0)),
                    0.88 - 0.03 * float(arank),
                )
            if anchor_type == "entity" and isinstance(support_cache, dict):
                for crank, chunk_id in enumerate(list(support_cache.get(str(anchor), []) or [])[:anchor_cache_topn]):
                    node_id = str(chunk_id)
                    if node_id not in g:
                        continue
                    cache_chunk_scores[node_id] = max(
                        float(cache_chunk_scores.get(node_id, -1.0)),
                        0.86 - 0.03 * float(crank) - 0.02 * float(arank),
                    )
            cache_nodes, cache_scores = _anchor_cache_reserve_candidates(
                g=g,
                anchor=anchor,
                topn=anchor_cache_topn,
                semantic_state=semantic_state,
            )
            for node_id in cache_nodes:
                if node_id not in g or (not _is_chunk_like_node(g, node_id)):
                    continue
                prior = float(cache_scores.get(node_id, 0.0))
                boosted = 0.82 * prior + 0.04 * (1.0 / float(arank + 1))
                cache_chunk_scores[node_id] = max(float(cache_chunk_scores.get(node_id, -1.0)), boosted)
        cache_chunk_scores = dict(sorted(cache_chunk_scores.items(), key=lambda x: x[1], reverse=True)[:topn_chunk])

    chunk_scores = {}
    if topn_chunk > 0:
        if chunk_lookup_strategy == "full":
            chunk_start = time.perf_counter()
            chunk_idx, chunk_scores_arr = topk_cosine_similarity(
                query_vector=query_vec,
                matrix=semantic_state.get("chunk_embeddings"),
                topn=topn_chunk,
                scan_batch_size=scan_bs,
            )
            for idx, score in zip(chunk_idx.tolist(), chunk_scores_arr.tolist()):
                if idx < 0 or idx >= len(semantic_state.get("chunk_ids", [])):
                    continue
                node = str(semantic_state["chunk_ids"][idx])
                chunk_scores[node] = max(float(score), float(chunk_scores.get(node, -1.0)))
            for node, score in cache_chunk_scores.items():
                chunk_scores[node] = max(float(chunk_scores.get(node, -1.0e9)), float(score))
            chunk_scores = dict(sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:topn_chunk])
            diag["chunk_ms"] = float((time.perf_counter() - chunk_start) * 1000.0)
            diag["chunk_mode"] = "cache+dense_full"
        elif chunk_lookup_strategy == "cache_only":
            chunk_scores = dict(sorted(cache_chunk_scores.items(), key=lambda x: x[1], reverse=True)[:topn_chunk])
            diag["chunk_ms"] = 0.0
            diag["chunk_mode"] = "cache_shortcut"
        elif chunk_lookup_strategy == "two_tier":
            chunk_scores, chunk_diag = _chunk_lookup_two_tier(
                query_vec=query_vec,
                semantic_state=semantic_state,
                g=g,
                anchors=anchor_list,
                query_text=qtext,
                entity_scores=entity_scores,
                cfg=cfg,
                prior_scores=cache_chunk_scores,
                allow_global_fallback=True,
            )
            diag["chunk_ms"] = float(chunk_diag.get("chunk_ms", 0.0))
            diag["chunk_mode"] = str(chunk_diag.get("chunk_mode", "two_tier"))
            diag["chunk_tier1_candidate_count"] = int(chunk_diag.get("chunk_tier1_candidate_count", 0))
            diag["chunk_tier1_shortlist_count"] = int(chunk_diag.get("chunk_tier1_shortlist_count", 0))
            diag["chunk_tier1_dense_count"] = int(chunk_diag.get("chunk_tier1_dense_count", 0))
        else:
            min_cache_hits = int(max(1, round(float(topn_chunk) * cache_min_ratio)))
            if int(len(cache_chunk_scores)) >= min_cache_hits:
                chunk_scores = dict(sorted(cache_chunk_scores.items(), key=lambda x: x[1], reverse=True)[:topn_chunk])
                diag["chunk_ms"] = 0.0
                diag["chunk_mode"] = "cache_shortcut"
            else:
                chunk_scores, chunk_diag = _chunk_lookup_two_tier(
                    query_vec=query_vec,
                    semantic_state=semantic_state,
                    g=g,
                    anchors=anchor_list,
                    query_text=qtext,
                    entity_scores=entity_scores,
                    cfg=cfg,
                    prior_scores=cache_chunk_scores,
                    allow_global_fallback=True,
                )
                diag["chunk_ms"] = float(chunk_diag.get("chunk_ms", 0.0))
                diag["chunk_mode"] = str(chunk_diag.get("chunk_mode", "two_tier"))
                diag["chunk_tier1_candidate_count"] = int(chunk_diag.get("chunk_tier1_candidate_count", 0))
                diag["chunk_tier1_shortlist_count"] = int(chunk_diag.get("chunk_tier1_shortlist_count", 0))
                diag["chunk_tier1_dense_count"] = int(chunk_diag.get("chunk_tier1_dense_count", 0))

    # Use chunk->entity support cache to recover entity diversity without full-entity dense scan.
    chunk_to_entity_cache = (
        semantic_state.get("chunk_topk_entities_cache", {}) if isinstance(semantic_state, dict) else {}
    )
    entity_boost_count = 0
    if topn_entity > 0 and isinstance(chunk_to_entity_cache, dict) and chunk_scores:
        for chunk_rank, (chunk_node, chunk_score) in enumerate(chunk_scores.items()):
            linked_entities = list(chunk_to_entity_cache.get(str(chunk_node), []) or [])
            for ent_rank, entity_node in enumerate(linked_entities[:8]):
                ent_id = str(entity_node)
                if g is not None and ent_id not in g:
                    continue
                boost = float(chunk_score) * (0.90 - 0.03 * float(chunk_rank)) * (0.94 - 0.02 * float(ent_rank))
                prev = float(entity_scores.get(ent_id, -1.0))
                if boost > prev:
                    entity_scores[ent_id] = float(boost)
                    entity_boost_count += 1
    entity_scores = dict(sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:topn_entity])
    diag["entity_chunk_cache_boost_count"] = int(entity_boost_count)
    return entity_scores, chunk_scores, diag


def _load_candidate_vectors(nodes, g, cfg, semantic_state):
    vectors = {}
    fallback_enabled = bool(getattr(cfg, "candidate_embedding_fallback_enabled", False))
    pending_nodes = []
    pending_texts = []
    for node in nodes:
        vec = None
        if semantic_state is not None and node in g:
            node_type = g.nodes[node].get("node_type")
            if node_type == "entity":
                idx = semantic_state.get("entity_id_to_idx", {}).get(str(node))
                if idx is not None:
                    vec = np.asarray(semantic_state["entity_embeddings"][idx], dtype=np.float32)
            elif _is_chunk_like_node(g, node):
                idx = semantic_state.get("chunk_id_to_idx", {}).get(str(node))
                if idx is not None:
                    vec = np.asarray(semantic_state["chunk_embeddings"][idx], dtype=np.float32)
        if vec is not None and vec.size > 0:
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vectors[node] = vec / norm
                continue

        if fallback_enabled:
            text = _node_semantic_text(g, node)
            if text:
                pending_nodes.append(node)
                pending_texts.append(text)

    if pending_nodes:
        out = encode_texts(
            texts=pending_texts,
            model_name=str(getattr(cfg, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2"),
            batch_size=int(getattr(cfg, "embedding_batch_size", 16)),
            max_length=int(getattr(cfg, "embedding_max_length", 192)),
            max_chars=int(getattr(cfg, "embedding_text_max_chars", 600)),
            instruction="",
        )
        if out.get("ok", False):
            for node, vec in zip(pending_nodes, out.get("vectors", []) or []):
                arr = np.asarray(vec, dtype=np.float32)
                if arr.size <= 0:
                    continue
                norm = float(np.linalg.norm(arr))
                if norm <= 0.0:
                    continue
                vectors[node] = arr / norm
            return vectors, ""
        return vectors, str(out.get("error", "candidate_embedding_failed"))
    return vectors, ""


def _entity_support_similarity(g, node, query_sim_map):
    if node not in g:
        return 0.0
    if g.nodes[node].get("node_type") != "entity":
        return float(query_sim_map.get(node, 0.0))

    best = 0.0
    for nbr in g.neighbors(node):
        nbr_type = g.nodes[nbr].get("node_type")
        if _is_chunk_like_node(g, nbr):
            best = max(best, float(query_sim_map.get(nbr, 0.0)))
        elif nbr_type == "relation":
            for nbr2 in g.neighbors(nbr):
                if _is_chunk_like_node(g, nbr2):
                    best = max(best, float(query_sim_map.get(nbr2, 0.0)))
    return best


def _weighted_mean(score_map, keys, weight_map=None):
    if not keys:
        return 0.0
    if not weight_map:
        vals = [float(score_map.get(k, 0.0)) for k in keys]
        return float(sum(vals) / max(len(vals), 1))

    total_w = 0.0
    total_v = 0.0
    for key in keys:
        w = max(0.0, float(weight_map.get(key, 0.0)))
        v = float(score_map.get(key, 0.0))
        total_w += w
        total_v += w * v
    if total_w <= 0.0:
        vals = [float(score_map.get(k, 0.0)) for k in keys]
        return float(sum(vals) / max(len(vals), 1))
    return float(total_v / total_w)


def _seed_bridge_bonus_map(candidates, anchors, anchor_distance_maps, tau):
    out = {}
    cands = list(candidates or [])
    if not cands:
        return out
    a_list = list(anchors or [])
    cap = max(1, int(tau))
    n_anchor = max(1, len(a_list))
    for node in cands:
        if not a_list:
            out[node] = 0.0
            continue
        connected = 0
        for anchor in a_list:
            dmap = anchor_distance_maps.get(anchor, {})
            if int(dmap.get(node, cap + 1)) <= cap:
                connected += 1
        base = float(connected) / float(n_anchor)
        if len(a_list) >= 2 and connected >= 2:
            base = min(1.0, base + 0.2)
        out[node] = max(0.0, min(1.0, float(base)))
    return out


def _seed_chunk_grounding_bonus_map(candidates, g, semantic_state, query_sim_map, support_sim_map):
    out = {}
    cands = list(candidates or [])
    if not cands or g is None or semantic_state is None:
        return out

    ent2chk = semantic_state.get("entity_topk_chunks_cache", {}) if isinstance(semantic_state, dict) else {}
    chk2ent = semantic_state.get("chunk_topk_entities_cache", {}) if isinstance(semantic_state, dict) else {}
    support_map = semantic_state.get("entity_to_chunks", {}) if isinstance(semantic_state, dict) else {}
    ent2chk = ent2chk if isinstance(ent2chk, dict) else {}
    chk2ent = chk2ent if isinstance(chk2ent, dict) else {}
    support_map = support_map if isinstance(support_map, dict) else {}

    for node in cands:
        if node not in g:
            continue
        node_type = str(g.nodes[node].get("node_type", "") or "")
        vals = []
        if node_type == "entity":
            linked = []
            linked.extend(list(ent2chk.get(str(node), []) or []))
            linked.extend(list(support_map.get(str(node), []) or []))
            seen = set()
            compact = []
            for chunk in linked:
                chunk_id = str(chunk)
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                compact.append(chunk_id)
            for chunk in compact[:12]:
                if chunk not in g:
                    continue
                qv = float(query_sim_map.get(chunk, 0.0))
                sv = float(support_sim_map.get(chunk, 0.0))
                vals.append(0.5 * (max(qv, sv) + 1.0))
        elif _is_chunk_like_node(g, node):
            linked_entities = list(chk2ent.get(str(node), []) or [])
            for ent in linked_entities[:12]:
                ent_id = str(ent)
                if ent_id not in g:
                    continue
                qv = float(query_sim_map.get(ent_id, 0.0))
                sv = float(support_sim_map.get(ent_id, 0.0))
                vals.append(0.5 * (max(qv, sv) + 1.0))

        if vals:
            vals.sort(reverse=True)
            top = vals[:3]
            out[node] = float(sum(top) / float(max(len(top), 1)))
        else:
            out[node] = 0.0
    return out


def _seed_hybrid_scores(
    candidates,
    agg_scores,
    query_sim,
    anchor_sim,
    support_sim,
    cfg,
    bridge_bonus_map=None,
    chunk_grounding_bonus_map=None,
):
    # Candidate-level seed scoring.
    # This stage ranks individual seed candidates (node-wise utility).
    # In canonical_copy_span, optional candidate-level bridge/grounding boosts
    # are pruned and excluded from the score blend.
    graph_raw = {node: float(agg_scores.get(node, 0.0)) for node in candidates}
    graph_norm = _normalize_map(graph_raw)

    semantic_raw = {}
    for node in candidates:
        sem = max(float(query_sim.get(node, 0.0)), float(support_sim.get(node, 0.0)))
        semantic_raw[node] = 0.5 * (sem + 1.0)
    semantic_norm = _normalize_map(semantic_raw)

    anchor_raw = {node: 0.5 * (float(anchor_sim.get(node, 0.0)) + 1.0) for node in candidates}
    anchor_norm = _normalize_map(anchor_raw)

    bridge_raw = {node: max(0.0, min(1.0, float((bridge_bonus_map or {}).get(node, 0.0)))) for node in candidates}
    bridge_norm = _normalize_map(bridge_raw)

    grounding_raw = {
        node: max(0.0, min(1.0, float((chunk_grounding_bonus_map or {}).get(node, 0.0)))) for node in candidates
    }
    grounding_norm = _normalize_map(grounding_raw)

    if is_canonical_copy_span_mode(str(getattr(cfg, "retrieval_objective_mode", "") or "").strip().lower()):
        score = canonical_seed_score_map(
            candidates=candidates,
            graph_norm=graph_norm,
            semantic_norm=semantic_norm,
            anchor_norm=anchor_norm,
            cfg=cfg,
        )
        return score, graph_norm, semantic_norm, anchor_norm, bridge_norm, grounding_norm

    w_sem = max(0.0, float(getattr(cfg, "seed_score_semantic_weight", 0.30)))
    w_graph = max(0.0, float(getattr(cfg, "seed_score_graph_weight", 0.50)))
    w_anchor = max(0.0, float(getattr(cfg, "seed_score_anchor_weight", 0.20)))
    w_bridge = max(0.0, float(getattr(cfg, "seed_score_bridge_weight", 0.0)))
    w_ground = max(0.0, float(getattr(cfg, "seed_score_chunk_grounding_weight", 0.0)))
    norm = w_sem + w_graph + w_anchor + w_bridge + w_ground
    if norm <= 0.0:
        w_graph = 1.0
        w_sem = 0.0
        w_anchor = 0.0
        w_bridge = 0.0
        w_ground = 0.0
        norm = 1.0
    w_sem /= norm
    w_graph /= norm
    w_anchor /= norm
    w_bridge /= norm
    w_ground /= norm

    score = {}
    for node in candidates:
        score[node] = (
            w_graph * float(graph_norm.get(node, 0.0))
            + w_sem * float(semantic_norm.get(node, 0.0))
            + w_anchor * float(anchor_norm.get(node, 0.0))
            + w_bridge * float(bridge_norm.get(node, 0.0))
            + w_ground * float(grounding_norm.get(node, 0.0))
        )
    return score, graph_norm, semantic_norm, anchor_norm, bridge_norm, grounding_norm


def _seed_selection_objective_weights(
    candidates,
    seed_score_map,
    bridge_bonus_map,
    chunk_grounding_bonus_map,
    anchors,
    anchor_distance_maps,
    cfg,
    tau,
):
    # Seed-set objective scoring.
    # This stage scores a candidate *set* for compact multi-hop coverage.
    # Unlike optional candidate-level boosts in _seed_hybrid_scores, these
    # objective terms (bridge/grounding/anchor-coverage) are retained as part
    # of seed-set selection behavior in canonical retrieval.
    base = {node: float(seed_score_map.get(node, 0.0)) for node in candidates}
    bridge_w = max(0.0, float(getattr(cfg, "seed_objective_bridge_weight", 0.20)))
    ground_w = max(0.0, float(getattr(cfg, "seed_objective_grounding_weight", 0.15)))
    anchor_cov_w = max(0.0, float(getattr(cfg, "seed_objective_anchor_coverage_weight", 0.10)))
    cap = max(1, int(tau))
    out = {}
    for node in candidates:
        anchor_hits = 0
        for anchor in anchors:
            dmap = anchor_distance_maps.get(anchor, {}) if isinstance(anchor_distance_maps, dict) else {}
            if int(dmap.get(node, cap + 1)) <= cap:
                anchor_hits += 1
        anchor_cov = float(anchor_hits) / float(max(len(anchors), 1))
        out[node] = (
            float(base.get(node, 0.0))
            + bridge_w * float((bridge_bonus_map or {}).get(node, 0.0))
            + ground_w * float((chunk_grounding_bonus_map or {}).get(node, 0.0))
            + anchor_cov_w * float(anchor_cov)
        )
    return out


def _bridge_utility(g, anchors, nodes, tau, distance_map_cache=None):
    if not anchors or len(anchors) < 2 or not nodes:
        return 0.0
    max_hops = max(1, int(tau))
    anchor_maps = {
        anchor: _get_distance_map(g, anchor, max_hops, distance_map_cache=distance_map_cache)
        for anchor in anchors
        if anchor in g
    }
    useful = 0
    for node in nodes:
        connected = 0
        for anchor in anchors:
            if node == anchor:
                connected += 1
                continue
            if node not in g or anchor not in anchor_maps:
                continue
            d = int(anchor_maps[anchor].get(node, max_hops + 1))
            if d <= max_hops:
                connected += 1
        if connected >= 2:
            useful += 1
    return float(useful / max(len(nodes), 1))


def _dispersion_penalty(g, seeds, tau, distance_map_cache=None):
    seeds = list(seeds or [])
    if len(seeds) <= 1:
        return 0.0
    cap = max(1, int(tau))
    pairs = 0
    total = 0.0
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a = seeds[i]
            b = seeds[j]
            if a not in g or b not in g:
                d = cap
            else:
                dmap = _get_distance_map(g, a, cap, distance_map_cache=distance_map_cache)
                d = int(dmap.get(b, cap))
            total += float(min(cap, d))
            pairs += 1
    if pairs <= 0:
        return 0.0
    return float((total / float(pairs)) / float(cap))


def _semantic_redundancy_penalty(seeds, node_vectors):
    items = [node for node in list(seeds or []) if node in node_vectors]
    if len(items) <= 1:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sim = cosine_similarity(node_vectors.get(items[i]), node_vectors.get(items[j]))
            total += max(0.0, float(sim))
            pairs += 1
    if pairs <= 0:
        return 0.0
    return float(total / float(pairs))


def _run_structural_connectivity(g, surrogate_universe, surrogate_weights, seeds, tau, distance_map_cache=None):
    if not surrogate_universe:
        return 0.0, 0.0, 0.0
    cap = max(1, int(tau))
    total_w = 0.0
    total_keep = 0.0
    loss = 0.0
    for node in surrogate_universe:
        w = max(0.0, float(surrogate_weights.get(node, 0.0)))
        if w <= 0.0:
            continue
        d = _min_set_distance(g, node, seeds, cap, distance_map_cache=distance_map_cache)
        keep = 1.0 - (float(d) / float(cap))
        total_w += w
        total_keep += w * keep
        loss += w * d
    if total_w <= 0.0:
        return 0.0, 0.0, loss
    base = float(total_keep / total_w)
    dispersion = _dispersion_penalty(g, seeds, cap, distance_map_cache=distance_map_cache)
    adjusted = max(0.0, base - 0.5 * dispersion)
    return adjusted, dispersion, loss


def _run_pair_coverage_score(anchors, seeds, anchor_distance_maps, tau):
    a_list = list(anchors or [])
    s_list = list(seeds or [])
    if not a_list or not s_list:
        return 0.0
    cap = max(1, int(tau))
    total = 0
    covered = 0
    for anchor in a_list:
        dmap = anchor_distance_maps.get(anchor, {}) if isinstance(anchor_distance_maps, dict) else {}
        for seed in s_list:
            total += 1
            if int(dmap.get(seed, cap + 1)) <= cap:
                covered += 1
    if total <= 0:
        return 0.0
    return float(covered) / float(total)


def _run_bridge_path_completeness(anchors, run_nodes, anchor_distance_maps, tau):
    a_list = list(anchors or [])
    r_nodes = list(run_nodes or [])
    if len(a_list) < 2 or not r_nodes:
        return 0.0
    cap = max(1, int(tau))
    total_pairs = 0
    complete_pairs = 0
    for i in range(len(a_list)):
        for j in range(i + 1, len(a_list)):
            total_pairs += 1
            a = a_list[i]
            b = a_list[j]
            da = anchor_distance_maps.get(a, {}) if isinstance(anchor_distance_maps, dict) else {}
            db = anchor_distance_maps.get(b, {}) if isinstance(anchor_distance_maps, dict) else {}
            ok = False
            for node in r_nodes:
                if int(da.get(node, cap + 1)) <= cap and int(db.get(node, cap + 1)) <= cap:
                    ok = True
                    break
            if ok:
                complete_pairs += 1
    if total_pairs <= 0:
        return 0.0
    return float(complete_pairs) / float(total_pairs)


def _run_entity_chunk_grounding_score(seeds, run_nodes, semantic_state):
    s_list = [str(x) for x in list(seeds or [])]
    r_set = {str(x) for x in list(run_nodes or [])}
    if not s_list or not r_set or semantic_state is None:
        return 0.0
    ent2chk = semantic_state.get("entity_topk_chunks_cache", {}) if isinstance(semantic_state, dict) else {}
    chk2ent = semantic_state.get("chunk_topk_entities_cache", {}) if isinstance(semantic_state, dict) else {}
    support_map = semantic_state.get("entity_to_chunks", {}) if isinstance(semantic_state, dict) else {}
    ent2chk = ent2chk if isinstance(ent2chk, dict) else {}
    chk2ent = chk2ent if isinstance(chk2ent, dict) else {}
    support_map = support_map if isinstance(support_map, dict) else {}

    considered = 0
    grounded = 0
    for seed in s_list:
        # entity-like seed
        linked_chunks = []
        linked_chunks.extend(list(ent2chk.get(seed, []) or []))
        linked_chunks.extend(list(support_map.get(seed, []) or []))
        if linked_chunks:
            considered += 1
            if any(str(c) in r_set for c in linked_chunks):
                grounded += 1
            continue
        # chunk-like seed
        linked_entities = list(chk2ent.get(seed, []) or [])
        if linked_entities:
            considered += 1
            if any(str(e) in r_set for e in linked_entities):
                grounded += 1
    if considered <= 0:
        return 0.0
    return float(grounded) / float(considered)


def _run_anchor_dispersion_penalty(anchors, seeds, anchor_distance_maps, tau):
    a_list = list(anchors or [])
    s_list = list(seeds or [])
    if not a_list or len(a_list) <= 1 or not s_list:
        return 0.0
    cap = max(1, int(tau))
    rates = []
    for anchor in a_list:
        dmap = anchor_distance_maps.get(anchor, {}) if isinstance(anchor_distance_maps, dict) else {}
        hit = 0
        for seed in s_list:
            if int(dmap.get(seed, cap + 1)) <= cap:
                hit += 1
        rates.append(float(hit) / float(max(len(s_list), 1)))
    arr = np.asarray(rates, dtype=np.float32)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr))


def _text_unit_type_for_node(g, node):
    node_type = str(g.nodes[node].get("node_type", "") or "").strip().lower()
    if node_type == "sentence":
        return "sentence"
    if node_type in {"chunk", "passage", "document"}:
        return "chunk"
    return "non_text"


def _text_unit_id(g, node):
    unit_type = _text_unit_type_for_node(g, node)
    if unit_type == "sentence":
        return str(g.nodes[node].get("sentence_id", node))
    if unit_type == "chunk":
        return str(g.nodes[node].get("chunk_id", node))
    return None


def _sentence_node_to_id(g, node):
    return _text_unit_id(g, node)


def _safe_ratio(numer, denom):
    try:
        n = float(numer)
        d = float(denom)
    except Exception:
        return 0.0
    if d <= 0.0:
        return 0.0
    return float(n / d)


def _corridor_sentence_sets(corridors):
    out = []
    for corridor in list(corridors or []):
        main_ids = [str(x) for x in list(corridor.get("main_path_sentence_ids", []) or []) if str(x)]
        support_ids = [str(x) for x in list(corridor.get("support_sentence_ids", []) or []) if str(x)]
        sent_set = set(main_ids + support_ids)
        if sent_set:
            out.append(sent_set)
    return out


def _mean_pairwise_jaccard(sets_list):
    sets = list(sets_list or [])
    if len(sets) <= 1:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a = sets[i]
            b = sets[j]
            union = len(a.union(b))
            if union <= 0:
                continue
            inter = len(a.intersection(b))
            total += float(inter) / float(union)
            count += 1
    if count <= 0:
        return 0.0
    return float(total / float(count))


def _compute_connector_retrieval_metrics(
    g,
    anchors,
    chosen_run,
    filtered_corridors,
    anchor_distance_maps,
    query_sim_map,
    support_sim_map,
    node_vectors,
    tau,
    cfg=None,
):
    a_list = [str(a) for a in list(anchors or []) if str(a)]
    seeds = [str(s) for s in list((chosen_run or {}).get("seeds", []) or []) if str(s)]
    cap = max(1, int(tau))
    answer_threshold = max(0.50, float(getattr(cfg, "seed_answer_threshold", 0.0) or 0.0))
    if answer_threshold <= 0.0:
        answer_threshold = 0.65

    covered_anchors = 0
    for anchor in a_list:
        dmap = (anchor_distance_maps or {}).get(anchor, {}) if isinstance(anchor_distance_maps, dict) else {}
        if any(int(dmap.get(seed, cap + 1)) <= cap for seed in seeds):
            covered_anchors += 1
    seed_anchor_recall = float(_safe_ratio(covered_anchors, len(a_list)))
    seed_bridge_recall = float(
        _run_bridge_path_completeness(
            anchors=a_list,
            run_nodes=seeds,
            anchor_distance_maps=anchor_distance_maps,
            tau=cap,
        )
    )

    candidate_nodes = _ordered_unique(list((chosen_run or {}).get("candidates", []) or []) + list(seeds))
    answer_like = {}
    for node in candidate_nodes:
        qv = float((query_sim_map or {}).get(node, 0.0))
        sv = float((support_sim_map or {}).get(node, 0.0))
        answer_like[node] = max(0.0, min(1.0, 0.5 * (max(qv, sv) + 1.0)))

    positive_nodes = [node for node, score in answer_like.items() if float(score) >= float(answer_threshold)]
    if not positive_nodes and answer_like:
        ranked_pos = sorted(answer_like.items(), key=lambda x: x[1], reverse=True)
        floor_topk = max(1, min(4, len(ranked_pos)))
        positive_nodes = [node for node, _ in ranked_pos[:floor_topk]]
    seed_answer_hits = [seed for seed in seeds if seed in set(positive_nodes)]
    if positive_nodes:
        seed_answer_recall = float(_safe_ratio(len(seed_answer_hits), len(positive_nodes)))
    else:
        seed_answer_recall = float(
            np.mean([float(answer_like.get(seed, 0.0)) for seed in seeds]) if seeds else 0.0
        )

    seed_diversity = float(_dispersion_penalty(g, seeds, cap, distance_map_cache={}) if seeds else 0.0)
    seed_redundancy = float(_semantic_redundancy_penalty(seeds, node_vectors or {}) if seeds else 0.0)

    run_nodes = list((chosen_run or {}).get("score_nodes", []) or [])
    if not run_nodes:
        run_nodes = list((chosen_run or {}).get("candidates", []) or [])
    run_components = dict((chosen_run or {}).get("run_score_components", {}) or {})
    run_bridge_utility = float(run_components.get("bridge_utility", 0.0) or 0.0)
    run_bridge_complete = float(run_components.get("bridge_path_completeness", 0.0) or 0.0)
    run_bridge_proxy = float(
        _run_bridge_path_completeness(
            anchors=a_list,
            run_nodes=run_nodes,
            anchor_distance_maps=anchor_distance_maps,
            tau=cap,
        )
    )
    run_bridge_coverage = float(max(run_bridge_utility, run_bridge_complete, run_bridge_proxy))

    def _weighted_avg(items, default=0.0):
        total_w = 0.0
        total_v = 0.0
        for value, weight in list(items or []):
            w = max(0.0, float(weight))
            v = float(value)
            total_w += w
            total_v += w * v
        if total_w <= 0.0:
            return float(default)
        return float(total_v / total_w)

    max_anchor = 0.0
    max_bridge = 0.0
    max_answer = 0.0
    bridge_purity_items = []
    answer_side_density_items = []
    compactness_items = []
    noise_ratio_items = []
    preserve_activation_items = []
    preserve_usefulness_items = []
    path_complete_items = []
    bridge_answer_pair_items = []
    incomplete_path_items = []
    chain_compactness_items = []
    path_preserve_activation_items = []
    strict_path_complete_items = []
    answer_bearing_path_items = []
    equivalent_evidence_coverage_items = []
    role_share_items = []
    legacy_path_complete_weight = 0.0
    false_path_weight = 0.0
    ranked_corridors = sorted(
        list(filtered_corridors or []),
        key=lambda c: float(c.get("corridor_score", 0.0)),
        reverse=True,
    )
    for ridx, corridor in enumerate(ranked_corridors):
        comp = dict(corridor.get("final_score_components", {}) or {})
        anchor_score = float(comp.get("anchor_alignment", corridor.get("anchor_alignment", 0.0)) or 0.0)
        bridge_score = float(comp.get("bridge_utility", corridor.get("bridge_potential", 0.0)) or 0.0)
        answer_score = float(
            comp.get(
                "answer_alignment",
                comp.get("answer_support", corridor.get("semantic_relevance", 0.0)),
            )
            or 0.0
        )
        max_anchor = max(float(max_anchor), max(0.0, min(1.0, float(anchor_score))))
        max_bridge = max(float(max_bridge), max(0.0, min(1.0, float(bridge_score))))
        max_answer = max(float(max_answer), max(0.0, min(1.0, float(answer_score))))
        support_ids = list(corridor.get("support_unit_ids", corridor.get("support_sentence_ids", [])) or [])
        connector_ids = list(
            corridor.get("connector_adjacent_unit_ids", corridor.get("connector_adjacent_sentence_ids", [])) or []
        )
        support_span = len(_ordered_unique([str(x) for x in (support_ids + connector_ids) if str(x)]))
        bridge_purity_val = float(comp.get("bridge_purity", bridge_score) or 0.0)
        answer_side_density_val = float(comp.get("answer_side_density", answer_score) or 0.0)
        compactness_val = float(comp.get("support_set_compactness", (1.0 / float(1.0 + max(0, support_span - 1)))) or 0.0)
        noise_ratio_val = float(comp.get("bridge_noise_ratio", (1.0 - bridge_purity_val)) or 0.0)
        preserve_activation_val = 1.0 if bool(comp.get("answer_preserve_applied", False)) else 0.0
        preserve_usefulness_val = float(comp.get("answer_preserve_usefulness", 0.0) or 0.0)
        path_complete_val = 1.0 if bool(comp.get("path_complete", False)) else float(comp.get("path_complete_score", 0.0) or 0.0)
        bridge_answer_pair_val = float(comp.get("bridge_answer_pair_retention", 0.0) or 0.0)
        incomplete_path_val = float(comp.get("incomplete_path_indicator", 0.0) or 0.0)
        chain_compactness_val = float(comp.get("chain_compactness", compactness_val) or compactness_val)
        path_preserve_activation_val = 1.0 if bool(comp.get("path_preserve_applied", False)) else 0.0
        strict_path_complete_val = 0.0
        if float(path_complete_val) > 0.0:
            strict_path_complete_val = float(
                max(
                    0.0,
                    min(
                        float(path_complete_val),
                        float(bridge_answer_pair_val),
                        float(bridge_purity_val),
                        float(answer_side_density_val),
                        float(chain_compactness_val),
                    ),
                )
            )
        answer_bearing_path_val = (
            1.0
            if (
                float(strict_path_complete_val) >= 0.45
                and float(answer_score) >= 0.40
                and float(answer_side_density_val) >= 0.40
            )
            else 0.0
        )
        false_path_val = 1.0 if (float(path_complete_val) >= 0.50 and float(answer_bearing_path_val) <= 0.0) else 0.0
        equivalent_evidence_coverage_val = float(
            max(
                0.0,
                min(
                    1.0,
                    max(float(answer_score), float(answer_side_density_val))
                    * (0.70 + 0.30 * max(0.0, min(1.0, float(bridge_purity_val)))),
                ),
            )
        )
        if float(path_complete_val) < 0.20:
            equivalent_evidence_coverage_val = float(max(0.0, min(1.0, 0.85 * equivalent_evidence_coverage_val)))
        role_total = max(1.0e-8, float(anchor_score) + float(bridge_score) + float(answer_score))
        role_share_items.append(
            (
                float(anchor_score) / float(role_total),
                float(bridge_score) / float(role_total),
                float(answer_score) / float(role_total),
                1.0 / float(ridx + 1),
            )
        )
        weight = 1.0 / float(ridx + 1)
        bridge_purity_items.append((max(0.0, min(1.0, bridge_purity_val)), weight))
        answer_side_density_items.append((max(0.0, min(1.0, answer_side_density_val)), weight))
        compactness_items.append((max(0.0, min(1.0, compactness_val)), weight))
        noise_ratio_items.append((max(0.0, min(1.0, noise_ratio_val)), weight))
        preserve_activation_items.append((float(preserve_activation_val), weight))
        preserve_usefulness_items.append((max(0.0, min(1.0, preserve_usefulness_val)), weight))
        path_complete_items.append((max(0.0, min(1.0, path_complete_val)), weight))
        bridge_answer_pair_items.append((max(0.0, min(1.0, bridge_answer_pair_val)), weight))
        incomplete_path_items.append((max(0.0, min(1.0, incomplete_path_val)), weight))
        chain_compactness_items.append((max(0.0, min(1.0, chain_compactness_val)), weight))
        path_preserve_activation_items.append((float(path_preserve_activation_val), weight))
        strict_path_complete_items.append((max(0.0, min(1.0, strict_path_complete_val)), weight))
        answer_bearing_path_items.append((max(0.0, min(1.0, answer_bearing_path_val)), weight))
        equivalent_evidence_coverage_items.append((max(0.0, min(1.0, equivalent_evidence_coverage_val)), weight))
        if float(path_complete_val) >= 0.50:
            legacy_path_complete_weight += float(max(0.0, weight))
            false_path_weight += float(max(0.0, weight) * max(0.0, min(1.0, false_path_val)))

    corridor_role_coverage = float((max_anchor + max_bridge + max_answer) / 3.0)
    bridge_purity = max(0.0, min(1.0, _weighted_avg(bridge_purity_items, max_bridge)))
    answer_side_density = max(0.0, min(1.0, _weighted_avg(answer_side_density_items, max_answer)))
    support_set_compactness = max(0.0, min(1.0, _weighted_avg(compactness_items, 1.0)))
    bridge_noise_ratio = max(0.0, min(1.0, _weighted_avg(noise_ratio_items, 1.0 - bridge_purity)))
    answer_preserve_activation_rate = max(0.0, min(1.0, _weighted_avg(preserve_activation_items, 0.0)))
    preserved_answer_usefulness = max(0.0, min(1.0, _weighted_avg(preserve_usefulness_items, 0.0)))
    path_complete_rate = max(0.0, min(1.0, _weighted_avg(path_complete_items, 0.0)))
    bridge_answer_pair_retention = max(0.0, min(1.0, _weighted_avg(bridge_answer_pair_items, 0.0)))
    incomplete_path_rate = max(0.0, min(1.0, _weighted_avg(incomplete_path_items, 0.0)))
    chain_compactness = max(0.0, min(1.0, _weighted_avg(chain_compactness_items, support_set_compactness)))
    path_preserve_activation_rate = max(0.0, min(1.0, _weighted_avg(path_preserve_activation_items, 0.0)))
    strict_path_complete_rate = max(0.0, min(1.0, _weighted_avg(strict_path_complete_items, 0.0)))
    answer_bearing_path_hit = max(0.0, min(1.0, _weighted_avg(answer_bearing_path_items, 0.0)))
    equivalent_evidence_coverage = max(
        0.0,
        min(1.0, _weighted_avg(equivalent_evidence_coverage_items, max(answer_side_density, path_complete_rate))),
    )
    false_path_rate = float(_safe_ratio(false_path_weight, legacy_path_complete_weight))
    false_path_rate = max(0.0, min(1.0, false_path_rate))
    role_weight_sum = float(sum(max(0.0, w) for _, _, _, w in role_share_items))
    if role_weight_sum <= 0.0:
        anchor_share = bridge_share = answer_share = 1.0 / 3.0
    else:
        anchor_share = float(sum(max(0.0, w) * a for a, _, _, w in role_share_items) / role_weight_sum)
        bridge_share = float(sum(max(0.0, w) * b for _, b, _, w in role_share_items) / role_weight_sum)
        answer_share = float(sum(max(0.0, w) * c for _, _, c, w in role_share_items) / role_weight_sum)
    anchor_bridge_balance = max(0.0, min(1.0, 1.0 - (max(anchor_share, bridge_share, answer_share) - min(anchor_share, bridge_share, answer_share))))
    bridge_present = bool(run_bridge_coverage >= 0.35)
    answer_side_present = bool(max_answer >= 0.35 or answer_side_density >= 0.35)
    useful_bridge_rate = 1.0 if (bridge_present and answer_side_present and bridge_purity >= 0.40) else 0.0
    bridge_to_answer_path_hit = (
        1.0
        if (bridge_present and answer_side_present and corridor_role_coverage >= 0.45 and bridge_purity >= 0.35)
        else 0.0
    )
    conversion_after_bridge_proxy = (
        1.0 if (bridge_present and answer_side_present and answer_side_density >= 0.35 and bridge_purity >= 0.35) else 0.0
    )
    hotpot_overpreserve_threshold = max(
        0.0, min(1.0, float(getattr(cfg, "hotpot_overpreserve_answer_share_threshold", 0.10)))
    )
    hotpot_overpreserve_rate = 0.0
    dataset_name = str(getattr(cfg, "dataset", "") or "").strip().lower()
    if dataset_name.startswith("hotpot"):
        hotpot_overpreserve_rate = (
            1.0
            if (
                answer_preserve_activation_rate >= 0.5
                and answer_share > (max(anchor_share, bridge_share) + float(hotpot_overpreserve_threshold))
            )
            else 0.0
        )
    conversion_after_preserve_proxy = (
        1.0
        if (
            answer_preserve_activation_rate >= 0.5
            and bridge_present
            and answer_side_present
            and preserved_answer_usefulness >= 0.35
        )
        else 0.0
    )
    conversion_after_path_preserve_proxy = (
        1.0
        if (
            path_preserve_activation_rate >= 0.25
            and bridge_present
            and answer_side_present
            and path_complete_rate >= 0.45
            and incomplete_path_rate <= 0.55
        )
        else 0.0
    )

    redundancy_rate = float(_mean_pairwise_jaccard(_corridor_sentence_sets(filtered_corridors)))

    return {
        "seed_anchor_recall": float(seed_anchor_recall),
        "seed_bridge_recall": float(seed_bridge_recall),
        "seed_answer_recall": float(seed_answer_recall),
        "seed_diversity": float(seed_diversity),
        "seed_redundancy": float(seed_redundancy),
        "run_bridge_coverage": float(run_bridge_coverage),
        "corridor_role_coverage": float(corridor_role_coverage),
        "answer_preserve_activation_rate": float(answer_preserve_activation_rate),
        "preserved_answer_usefulness": float(preserved_answer_usefulness),
        "path_complete_rate": float(path_complete_rate),
        "strict_path_complete_rate": float(strict_path_complete_rate),
        "answer_bearing_path_hit": float(answer_bearing_path_hit),
        "false_path_rate": float(false_path_rate),
        "equivalent_evidence_coverage": float(equivalent_evidence_coverage),
        "bridge_answer_pair_retention": float(bridge_answer_pair_retention),
        "incomplete_path_rate": float(incomplete_path_rate),
        "chain_compactness": float(chain_compactness),
        "path_preserve_activation_rate": float(path_preserve_activation_rate),
        "anchor_bridge_balance": float(anchor_bridge_balance),
        "anchor_role_share": float(anchor_share),
        "bridge_role_share": float(bridge_share),
        "answer_role_share": float(answer_share),
        "bridge_purity": float(bridge_purity),
        "answer_side_density": float(answer_side_density),
        "support_set_compactness": float(support_set_compactness),
        "bridge_noise_ratio": float(bridge_noise_ratio),
        "useful_bridge_rate": float(useful_bridge_rate),
        "bridge_to_answer_path_hit": float(bridge_to_answer_path_hit),
        "conversion_after_bridge_proxy": float(conversion_after_bridge_proxy),
        "conversion_after_preserve_proxy": float(conversion_after_preserve_proxy),
        "conversion_after_path_preserve_proxy": float(conversion_after_path_preserve_proxy),
        "hotpot_overpreserve_rate": float(hotpot_overpreserve_rate),
        "redundancy_rate": float(redundancy_rate),
        "connector_quality": float(
            (
                seed_bridge_recall
                + run_bridge_coverage
                + corridor_role_coverage
                + path_complete_rate
                + bridge_purity
                + chain_compactness
            )
            / 6.0
        ),
    }


def _gold_text_unit_ids_from_sample(sample, graph_mode):
    _ = graph_mode
    ids = set()
    for title, sent_idx in list(getattr(sample, "supporting_facts", []) or []):
        t = str(title or "").strip()
        if not t:
            continue
        try:
            idx = int(sent_idx)
        except Exception:
            idx = 0
        ids.add(f"{t}::{idx}")
    return ids


def _gold_entity_tokens_from_sample(sample):
    doc_map = {}
    for doc in list(getattr(sample, "contexts", []) or []):
        title = str(getattr(doc, "title", "") or "").strip()
        if not title:
            continue
        if title not in doc_map:
            doc_map[title] = list(getattr(doc, "sentences", []) or [])

    tokens = set()
    for title, sent_idx in list(getattr(sample, "supporting_facts", []) or []):
        t = str(title or "").strip()
        if not t:
            continue
        tokens.update(content_tokens(t))
        sentences = list(doc_map.get(t, []) or [])
        try:
            idx = int(sent_idx)
        except Exception:
            idx = -1
        if 0 <= idx < len(sentences):
            tokens.update(content_tokens(str(sentences[idx] or "")))
    return {str(tok).strip().lower() for tok in tokens if str(tok).strip()}


def _candidate_text_unit_ids(g, nodes):
    out = set()
    text_map = {}
    if g is None:
        return out, text_map
    for node in list(nodes or []):
        if node not in g:
            continue
        unit_id = _text_unit_id(g, node)
        if unit_id:
            sid = str(unit_id)
            out.add(sid)
            txt = str((g.nodes[node] or {}).get("text", "") or "").strip()
            if txt and sid not in text_map:
                text_map[sid] = txt
    return out, text_map


def _phase1_candidate_nodes(shortlisted_runs):
    nodes = set()
    for run in list(shortlisted_runs or []):
        if not isinstance(run, dict):
            continue
        for key in ("candidates", "graph_candidates", "semantic_candidates"):
            for node in list(run.get(key, []) or []):
                nodes.add(node)
        for node in list(run.get("seeds", set()) or []):
            nodes.add(node)
    return nodes


def _compute_stagewise_loss_funnel(
    sample,
    g,
    reduced_graph,
    anchors,
    proposal_nodes,
    shortlisted_runs,
    selected_text_unit_ids,
    selected_text_map,
    graph_mode,
):
    gold_unit_ids = set(_gold_text_unit_ids_from_sample(sample, graph_mode))
    gold_entity_tokens = set(_gold_entity_tokens_from_sample(sample))

    anchor_tokens = set()
    for anchor in list(anchors or []):
        raw = str(anchor or "").strip().lower()
        if not raw:
            continue
        anchor_tokens.add(raw)
        anchor_tokens.update(content_tokens(raw.replace("_", " ")))

    anchor_hits = sorted(anchor_tokens.intersection(gold_entity_tokens))
    proposal_unit_ids, proposal_text_map = _candidate_text_unit_ids(g, proposal_nodes)
    proposal_match = supporting_fact_match_details(
        sample=sample,
        unit_ids=sorted(proposal_unit_ids),
        unit_texts=[proposal_text_map.get(sid, "") for sid in sorted(proposal_unit_ids)],
        graph_mode=graph_mode,
    )
    proposal_hits = sorted(set(proposal_match.get("matched_gold_sentence_ids", []) or []))

    phase1_nodes = _phase1_candidate_nodes(shortlisted_runs)
    phase1_unit_ids, phase1_text_map = _candidate_text_unit_ids(reduced_graph if reduced_graph is not None else g, phase1_nodes)
    phase1_match = supporting_fact_match_details(
        sample=sample,
        unit_ids=sorted(phase1_unit_ids),
        unit_texts=[phase1_text_map.get(sid, "") for sid in sorted(phase1_unit_ids)],
        graph_mode=graph_mode,
    )
    phase1_hits = sorted(set(phase1_match.get("matched_gold_sentence_ids", []) or []))

    final_unit_ids = [str(x) for x in list(selected_text_unit_ids or []) if str(x)]
    final_match = supporting_fact_match_details(
        sample=sample,
        unit_ids=final_unit_ids,
        unit_texts=[str((selected_text_map or {}).get(sid, "") or "") for sid in final_unit_ids],
        graph_mode=graph_mode,
    )
    final_hits = sorted(set(final_match.get("matched_gold_sentence_ids", []) or []))

    return {
        "enabled": True,
        "graph_mode": str(graph_mode or "current_entity_graph"),
        "gold_unit_type": "support_sentence",
        "gold_text_unit_total": int(len(gold_unit_ids)),
        "gold_entity_total": int(len(gold_entity_tokens)),
        "gold_text_unit_ids": sorted(gold_unit_ids),
        "gold_entity_tokens": sorted(gold_entity_tokens)[:128],
        "anchor_hit_count": int(len(anchor_hits)),
        "anchor_hit_rate": float(_safe_ratio(len(anchor_hits), len(gold_entity_tokens))),
        "anchor_hit_entities": list(anchor_hits),
        "proposal_hit_count": int(len(proposal_hits)),
        "proposal_hit_rate": float(_safe_ratio(len(proposal_hits), len(gold_unit_ids))),
        "proposal_hit_unit_ids": list(proposal_hits),
        "phase1_hit_count": int(len(phase1_hits)),
        "phase1_hit_rate": float(_safe_ratio(len(phase1_hits), len(gold_unit_ids))),
        "phase1_hit_unit_ids": list(phase1_hits),
        "final_chunk_hit_count": int(len(final_hits)),
        "final_chunk_hit_rate": float(_safe_ratio(len(final_hits), len(gold_unit_ids))),
        "final_chunk_hit_unit_ids": list(final_hits),
        # Filled in render stage (render_context) after context packaging.
        "rendered_hit_count": 0,
        "rendered_hit_rate": 0.0,
        "rendered_retention": 0.0,
    }


def _build_single_corridor_payload(g, anchor, seed, corridor_id, corridor_score, node_scores, fallback_text_cap=4):
    fallback_cap = max(1, int(fallback_text_cap))
    ordered_nodes = [node for node, _ in node_scores]
    local_nodes = set(ordered_nodes) | {anchor, seed}
    local_graph = g.subgraph(local_nodes).copy()

    path_nodes = []
    if anchor in local_graph and seed in local_graph:
        try:
            path_nodes = nx.shortest_path(local_graph, anchor, seed)
        except Exception:
            path_nodes = []

    path_found = bool(path_nodes)
    main_text_nodes = [node for node in path_nodes if _text_unit_id(g, node) is not None]
    fallback_reason = ""
    if not main_text_nodes:
        fallback_reason = "no_shortest_path" if not path_found else "path_without_text_units"
        scored_text_nodes = [node for node, _ in node_scores if _text_unit_id(g, node) is not None]
        anchor_local = []
        seed_local = []
        if anchor in local_graph:
            anchor_local = [node for node in local_graph.neighbors(anchor) if _text_unit_id(g, node) is not None]
        if seed in local_graph:
            seed_local = [node for node in local_graph.neighbors(seed) if _text_unit_id(g, node) is not None]
        seed_local_set = set(seed_local)
        shared_local = [node for node in anchor_local if node in seed_local_set]
        fallback_pool = _ordered_unique(shared_local + anchor_local + seed_local + scored_text_nodes)
        main_text_nodes = fallback_pool[:fallback_cap]

    connector_nodes = {node for node in path_nodes if _text_unit_id(g, node) is None}
    main_text_set = set(main_text_nodes)
    candidate_support_nodes = [
        node
        for node, _ in node_scores
        if _text_unit_id(g, node) is not None and node not in main_text_set
    ]

    adjacent_support_nodes = []
    for node in candidate_support_nodes:
        nbrs = set(g.neighbors(node))
        if nbrs.intersection(main_text_set) or nbrs.intersection(connector_nodes):
            adjacent_support_nodes.append(node)

    support_text_nodes = _ordered_unique(adjacent_support_nodes + candidate_support_nodes)

    unit_score_map = {}
    main_unit_types = []
    for node, score in node_scores:
        sid = _text_unit_id(g, node)
        if sid is None:
            continue
        unit_score_map[sid] = max(float(score), unit_score_map.get(sid, 0.0))

    main_unit_ids = _ordered_unique([_text_unit_id(g, node) for node in main_text_nodes if node in g])
    main_unit_ids = [sid for sid in main_unit_ids if sid]
    main_unit_type = "mixed"
    if main_text_nodes:
        main_unit_types = [_text_unit_type_for_node(g, node) for node in main_text_nodes if node in g]
        uniq = {u for u in main_unit_types if u != "non_text"}
        if len(uniq) == 1:
            main_unit_type = list(uniq)[0]

    support_unit_ids = _ordered_unique([_text_unit_id(g, node) for node in support_text_nodes if node in g])
    support_unit_ids = [sid for sid in support_unit_ids if sid and sid not in set(main_unit_ids)]
    connector_adjacent_unit_ids = _ordered_unique([_text_unit_id(g, node) for node in adjacent_support_nodes if node in g])
    connector_adjacent_unit_ids = [sid for sid in connector_adjacent_unit_ids if sid and sid not in set(main_unit_ids)]

    return {
        "corridor_id": corridor_id,
        "corridor_score": float(corridor_score),
        "anchors": [str(anchor), str(seed)],
        "main_path_unit_ids": main_unit_ids,
        "support_unit_ids": support_unit_ids,
        "connector_adjacent_unit_ids": connector_adjacent_unit_ids,
        "unit_score_map": unit_score_map,
        "unit_type": main_unit_type,
        # legacy aliases
        "main_path_sentence_ids": list(main_unit_ids),
        "support_sentence_ids": list(support_unit_ids),
        "connector_adjacent_sentence_ids": list(connector_adjacent_unit_ids),
        "sentence_score_map": dict(unit_score_map),
        "path_node_ids": [str(node) for node in path_nodes],
        "path_found": bool(path_found),
        "fallback_reason": str(fallback_reason),
    }


def _build_corridor(
    g,
    anchors,
    seeds,
    anchor_scores_by_node,
    cfg,
    ppr_engine,
    pair_top_lp,
    corridor_top_bc,
    show_progress=False,
):
    if not anchors or not seeds:
        return g.subgraph(anchors + list(seeds)).copy(), [], {}, []

    seed_score_map = _compute_ppr_batch(
        g=g,
        sources=list(seeds),
        cfg=cfg,
        engine=ppr_engine,
        run_seed=911,
        show_progress=show_progress,
        desc="Corridor seed PPR",
    )

    pair_stats = []
    for anchor in tqdm(
        anchors,
        total=len(anchors),
        desc="Corridor pair score",
        leave=False,
        disable=not show_progress,
    ):
        a_scores = anchor_scores_by_node.get(anchor, {})
        for seed in seeds:
            z_scores = seed_score_map.get(seed, {})
            support_sum = 0.0
            for node in g.nodes:
                support_sum += _pair_support(a_scores, z_scores, node)
            pair_stats.append(((anchor, seed), support_sum))

    pair_stats.sort(key=lambda x: x[1], reverse=True)
    retained_pairs = [pair for pair, _ in pair_stats[:pair_top_lp]]
    pair_score_map = {pair: score for pair, score in pair_stats}

    corridor_nodes = set()
    sentence_scores = {}
    corridor_payloads = []
    fallback_text_cap = max(1, int(getattr(cfg, "corridor_fallback_text_cap", 4)))

    for idx, (anchor, seed) in enumerate(
        tqdm(
            retained_pairs,
            total=len(retained_pairs),
            desc="Corridor compose",
            leave=False,
            disable=not show_progress,
        ),
        start=1,
    ):
        a_scores = anchor_scores_by_node.get(anchor, {})
        z_scores = seed_score_map.get(seed, {})
        node_scores = []

        for node in g.nodes:
            s = _pair_support(a_scores, z_scores, node)
            node_scores.append((node, s))

        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_node_scores = node_scores[:corridor_top_bc]
        for node, score in top_node_scores:
            corridor_nodes.add(node)
            sentence_scores[node] = max(sentence_scores.get(node, 0.0), score)

        corridor_nodes.add(anchor)
        corridor_nodes.add(seed)

        corridor_payloads.append(
            _build_single_corridor_payload(
                g=g,
                anchor=anchor,
                seed=seed,
                corridor_id=f"c{idx:02d}",
                corridor_score=pair_score_map.get((anchor, seed), 0.0),
                node_scores=top_node_scores,
                fallback_text_cap=fallback_text_cap,
            )
        )

    h = g.subgraph(corridor_nodes).copy()
    return h, retained_pairs, sentence_scores, corridor_payloads


def _can_remove_node(h, node, anchors, min_seed_keep, seeds):
    if node in anchors:
        return False

    trial = h.copy()
    trial.remove_node(node)

    if not anchors.issubset(trial.nodes()):
        return False

    components = list(nx.connected_components(trial))
    if not any(anchors.issubset(comp) for comp in components):
        return False

    kept_seed_count = len(seeds.intersection(set(trial.nodes())))
    return kept_seed_count >= min_seed_keep


def _greedy_trim(h, anchors, seeds, rho):
    anchors_set = set(anchors)
    min_seed_keep = max(1, int(round(rho * len(seeds)))) if seeds else 0

    changed = True
    while changed:
        changed = False

        leaf_like = [n for n in list(h.nodes()) if h.degree(n) <= 1 and n not in anchors_set]
        for node in leaf_like:
            if node not in h:
                continue
            if _can_remove_node(h, node, anchors_set, min_seed_keep, seeds):
                h.remove_node(node)
                changed = True

        for comp in list(nx.connected_components(h)):
            if not anchors_set.intersection(comp):
                h.remove_nodes_from(comp)
                changed = True

    return h


def _extract_sentence_payload(g, selected_nodes, sentence_scores):
    sentences = []
    for node in selected_nodes:
        sid = _sentence_node_to_id(g, node)
        if sid is None:
            continue
        text = str(g.nodes[node].get("text", ""))
        score = sentence_scores.get(node, 0.0)
        sentences.append((sid, text, score))

    sentences.sort(key=lambda x: x[2], reverse=True)
    sentence_ids = [sid for sid, _, _ in sentences]
    sentence_text = [text for _, text, _ in sentences]
    sentence_score_map = {sid: float(score) for sid, _, score in sentences}
    return sentence_ids, sentence_text, sentence_score_map


def _filter_corridor_payloads(corridors, selected_unit_ids):
    selected = set(selected_unit_ids)
    filtered = []

    for corridor in corridors:
        main_ids = [
            sid
            for sid in corridor.get("main_path_unit_ids", corridor.get("main_path_sentence_ids", []))
            if sid in selected
        ]
        main_set = set(main_ids)
        support_ids = [
            sid
            for sid in corridor.get("support_unit_ids", corridor.get("support_sentence_ids", []))
            if sid in selected and sid not in main_set
        ]
        connector_adjacent_ids = [
            sid
            for sid in corridor.get("connector_adjacent_unit_ids", corridor.get("connector_adjacent_sentence_ids", []))
            if sid in selected and sid not in main_set
        ]

        if not main_ids and not support_ids:
            continue

        payload = dict(corridor)
        payload["main_path_unit_ids"] = list(main_ids)
        payload["support_unit_ids"] = list(support_ids)
        payload["connector_adjacent_unit_ids"] = list(connector_adjacent_ids)
        payload["main_path_sentence_ids"] = list(main_ids)
        payload["support_sentence_ids"] = list(support_ids)
        payload["connector_adjacent_sentence_ids"] = list(connector_adjacent_ids)
        filtered.append(payload)

    return filtered


def _build_sentence_feature_table(sample, selected_sentence_ids, sentence_texts, sentence_score_map, corridors):
    question_tokens = set(content_tokens(sample.question))
    sentence_text_map = {sid: text for sid, text in zip(selected_sentence_ids, sentence_texts)}

    ranked_corridors = sorted(corridors or [], key=lambda c: float(c.get("corridor_score", 0.0)), reverse=True)
    corridor_rank = {str(c.get("corridor_id", f"c{idx:02d}")): idx for idx, c in enumerate(ranked_corridors, start=1)}

    feature_table = {}
    for global_idx, sid in enumerate(selected_sentence_ids):
        feature_table[sid] = {
            "corridor_ids": [],
            "best_corridor_rank": None,
            "best_corridor_score": 0.0,
            "is_main_candidate": False,
            "is_support_candidate": False,
            "is_connector_adjacent": False,
            "query_overlap_score": 0.0,
            "locality_score": 1.0 / float(global_idx + 1),
            "base_retrieval_score": float(sentence_score_map.get(sid, 0.0)),
        }

    for idx, corridor in enumerate(ranked_corridors, start=1):
        cid = str(corridor.get("corridor_id", f"c{idx:02d}"))
        cscore = float(corridor.get("corridor_score", 0.0))
        main_ids = [
            sid
            for sid in corridor.get("main_path_unit_ids", corridor.get("main_path_sentence_ids", []))
            if sid in feature_table
        ]
        main_set = set(main_ids)
        support_ids = [
            sid
            for sid in corridor.get("support_unit_ids", corridor.get("support_sentence_ids", []))
            if sid in feature_table and sid not in main_set
        ]
        connector_ids = [
            sid
            for sid in corridor.get("connector_adjacent_unit_ids", corridor.get("connector_adjacent_sentence_ids", []))
            if sid in feature_table and sid not in main_set
        ]

        for mpos, sid in enumerate(main_ids):
            item = feature_table.get(sid)
            if item is None:
                continue
            if cid not in item["corridor_ids"]:
                item["corridor_ids"].append(cid)
            item["is_main_candidate"] = True
            item["best_corridor_score"] = max(item["best_corridor_score"], cscore)
            if item["best_corridor_rank"] is None or idx < item["best_corridor_rank"]:
                item["best_corridor_rank"] = idx
            item["locality_score"] = max(item["locality_score"], 1.0 / float((idx) * (mpos + 1)))

        for spos, sid in enumerate(support_ids):
            item = feature_table.get(sid)
            if item is None:
                continue
            if cid not in item["corridor_ids"]:
                item["corridor_ids"].append(cid)
            item["is_support_candidate"] = True
            item["best_corridor_score"] = max(item["best_corridor_score"], cscore)
            if item["best_corridor_rank"] is None or idx < item["best_corridor_rank"]:
                item["best_corridor_rank"] = idx
            item["locality_score"] = max(item["locality_score"], 0.75 / float((idx) * (spos + 1)))

        for sid in connector_ids:
            item = feature_table.get(sid)
            if item is None:
                continue
            item["is_connector_adjacent"] = True

    for sid, item in feature_table.items():
        sent = sentence_text_map.get(sid, "")
        sent_tokens = set(content_tokens(sent))
        item["query_overlap_score"] = float(len(question_tokens.intersection(sent_tokens)))
        item["corridor_ids"] = sorted(item["corridor_ids"], key=lambda cid: corridor_rank.get(cid, 10**9))
        if item["best_corridor_rank"] is None and item["corridor_ids"]:
            item["best_corridor_rank"] = corridor_rank.get(item["corridor_ids"][0])
        if item["best_corridor_rank"] is None:
            item["best_corridor_rank"] = 10**9

    return feature_table


def _shortest_distance_with_cap(g, src, dst, cap):
    max_hops = max(1, int(cap))
    if src not in g or dst not in g:
        return max_hops
    try:
        d = nx.shortest_path_length(g, src, dst)
        return min(max_hops, int(d))
    except Exception:
        return max_hops


def _anchor_graph_reserve_candidates(g, anchor, topn, max_hops, precomputed_dmap=None):
    k = max(0, int(topn))
    if k <= 0 or anchor not in g:
        return []

    cutoff = max(1, int(max_hops))
    if isinstance(precomputed_dmap, dict):
        dmap = precomputed_dmap
    else:
        try:
            dmap = nx.single_source_shortest_path_length(g, anchor, cutoff=cutoff)
        except Exception:
            dmap = {anchor: 0}

    scores = []
    for node, dist in dmap.items():
        if node == anchor:
            continue
        ntype = g.nodes[node].get("node_type")
        if (ntype != "entity") and (not _is_chunk_like_node(g, node)):
            continue
        dist_term = 1.0 / float(1 + int(dist))
        degree_term = min(1.0, float(g.degree(node)) / 20.0)
        scores.append((node, 0.80 * dist_term + 0.20 * degree_term))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [node for node, _ in scores[:k]]


def _anchor_cache_reserve_candidates(g, anchor, topn, semantic_state):
    k = max(0, int(topn))
    if k <= 0 or anchor not in g:
        return [], {}

    scores = {}
    if isinstance(semantic_state, dict):
        ent2chk = semantic_state.get("entity_topk_chunks_cache", {}) if isinstance(semantic_state.get("entity_topk_chunks_cache", {}), dict) else {}
        chk2ent = semantic_state.get("chunk_topk_entities_cache", {}) if isinstance(semantic_state.get("chunk_topk_entities_cache", {}), dict) else {}

        anchor_key = str(anchor)
        linked_chunks = list(ent2chk.get(anchor_key, []) or [])
        for rank, node in enumerate(linked_chunks[: max(k * 2, 8)]):
            node_id = str(node)
            if node_id not in g:
                continue
            scores[node_id] = max(float(scores.get(node_id, -1.0e9)), 1.0 - 0.04 * float(rank))
            linked_entities = list(chk2ent.get(node_id, []) or [])
            for erank, ent in enumerate(linked_entities[:2]):
                ent_id = str(ent)
                if ent_id == anchor_key or ent_id not in g:
                    continue
                scores[ent_id] = max(float(scores.get(ent_id, -1.0e9)), 0.82 - 0.06 * float(rank) - 0.04 * float(erank))

        linked_entities = list(chk2ent.get(anchor_key, []) or [])
        for rank, node in enumerate(linked_entities[: max(k * 2, 8)]):
            node_id = str(node)
            if node_id not in g:
                continue
            scores[node_id] = max(float(scores.get(node_id, -1.0e9)), 1.0 - 0.04 * float(rank))
            linked_chunks = list(ent2chk.get(node_id, []) or [])
            for crank, chunk in enumerate(linked_chunks[:2]):
                chunk_id = str(chunk)
                if chunk_id == anchor_key or chunk_id not in g:
                    continue
                scores[chunk_id] = max(float(scores.get(chunk_id, -1.0e9)), 0.82 - 0.06 * float(rank) - 0.04 * float(crank))

    # Cheap graph-only reserve for diversity without multi-hop BFS.
    for nbr in g.neighbors(anchor):
        if nbr == anchor:
            continue
        ntype = g.nodes[nbr].get("node_type")
        if (ntype != "entity") and (not _is_chunk_like_node(g, nbr)):
            continue
        degree_term = min(1.0, float(g.degree(nbr)) / 20.0)
        scores[nbr] = max(float(scores.get(nbr, -1.0e9)), 0.35 + 0.15 * degree_term)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [node for node, _ in ranked[:k]]
    selected_scores = {node: float(score) for node, score in ranked[:k]}
    return selected, selected_scores


def _resolve_proposal_union_experiment_mode(cfg):
    mode = str(getattr(cfg, "proposal_union_experiment_mode", "off") or "off").strip().lower()
    if mode not in {"off", "mild", "medium", "aggressive"}:
        mode = "off"
    return mode


def _proposal_union_level_settings(mode, topn_entity, topn_chunk, reserve_topn):
    settings = {
        "entity_cap": int(topn_entity),
        "chunk_cap": int(topn_chunk),
        "reserve_cap": int(reserve_topn),
        "entity_pool_cap": int(max(topn_entity, topn_entity * 2)),
        "chunk_pool_cap": int(max(topn_chunk, topn_chunk * 2)),
        "support_per_entity": 4,
        "graph_mix_topn": int(max(1, min(reserve_topn, 3))),
        "local_cap_scale": 1.0,
        "dist_cap_max": None,
    }
    if mode == "mild":
        settings.update(
            {
                "entity_cap": max(8, int(round(float(topn_entity) * 0.95))),
                "chunk_cap": max(6, int(round(float(topn_chunk) * 0.95))),
                "reserve_cap": max(6, int(round(float(reserve_topn) * 0.90))),
                "support_per_entity": 4,
                "graph_mix_topn": max(2, int(round(float(reserve_topn) * 0.6))),
                "local_cap_scale": 0.90,
                "dist_cap_max": 3,
            }
        )
    elif mode == "medium":
        settings.update(
            {
                "entity_cap": max(6, int(round(float(topn_entity) * 0.85))),
                "chunk_cap": max(5, int(round(float(topn_chunk) * 0.80))),
                "reserve_cap": max(5, int(round(float(reserve_topn) * 0.75))),
                "support_per_entity": 3,
                "graph_mix_topn": max(2, int(round(float(reserve_topn) * 0.5))),
                "local_cap_scale": 0.75,
                "dist_cap_max": 3,
            }
        )
    elif mode == "aggressive":
        settings.update(
            {
                "entity_cap": max(5, int(round(float(topn_entity) * 0.70))),
                "chunk_cap": max(4, int(round(float(topn_chunk) * 0.65))),
                "reserve_cap": max(4, int(round(float(reserve_topn) * 0.60))),
                "support_per_entity": 2,
                "graph_mix_topn": max(1, int(round(float(reserve_topn) * 0.35))),
                "local_cap_scale": 0.60,
                "dist_cap_max": 2,
            }
        )
    return settings


def _build_anchor_proposals_lazy_experimental(
    g,
    anchors,
    semantic_entity_scores,
    semantic_chunk_scores,
    cfg,
    semantic_state=None,
    mode="mild",
):
    topn_entity = max(0, int(getattr(cfg, "semantic_topn_entity", getattr(cfg, "semantic_topn", 50))))
    topn_chunk = max(0, int(getattr(cfg, "semantic_topn_chunk", max(1, topn_entity // 2))))
    reserve_topn = max(0, int(getattr(cfg, "graph_reserve_topn", 15)))
    dist_cap = max(1, int(getattr(cfg, "tau", 4)))
    reserve_hops = max(2, int(getattr(cfg, "proposal_reserve_hops", dist_cap)))
    reserve_bfs_fallback = bool(getattr(cfg, "proposal_reserve_bfs_fallback", False))
    anchor_distance_bonus = bool(getattr(cfg, "proposal_anchor_distance_bonus", True))

    level = _proposal_union_level_settings(mode, topn_entity=topn_entity, topn_chunk=topn_chunk, reserve_topn=reserve_topn)
    entity_cap = int(level["entity_cap"])
    chunk_cap = int(level["chunk_cap"])
    reserve_cap = int(level["reserve_cap"])
    entity_pool_cap = int(level["entity_pool_cap"])
    chunk_pool_cap = int(level["chunk_pool_cap"])
    support_per_entity = int(level["support_per_entity"])
    graph_mix_topn = int(level["graph_mix_topn"])
    local_cap_base = max(8, int(getattr(cfg, "proposal_anchor_local_topn", max(32, topn_entity + topn_chunk + reserve_topn))))
    local_cap = max(8, int(round(float(local_cap_base) * float(level["local_cap_scale"]))))
    shared_high_conf_topn = max(0, int(getattr(cfg, "proposal_shared_high_conf_topn", 24)))
    global_fallback_topn = max(0, int(getattr(cfg, "proposal_global_fallback_topn", 16)))
    dist_cap_effective = int(dist_cap)
    if level["dist_cap_max"] is not None:
        dist_cap_effective = max(1, min(int(dist_cap), int(level["dist_cap_max"])))
    reserve_hops_effective = max(1, min(int(reserve_hops), int(dist_cap_effective + 1)))

    proposal_by_anchor = {}
    proposal_scores = {}
    graph_reserve_union = set()
    semantic_scores = {}
    semantic_scores.update({str(k): float(v) for k, v in (semantic_entity_scores or {}).items()})
    semantic_scores.update({str(k): float(v) for k, v in (semantic_chunk_scores or {}).items()})

    ent2chk = semantic_state.get("entity_topk_chunks_cache", {}) if isinstance(semantic_state, dict) else {}
    support_map = semantic_state.get("entity_to_chunks", {}) if isinstance(semantic_state, dict) else {}
    ent2chk = ent2chk if isinstance(ent2chk, dict) else {}
    support_map = support_map if isinstance(support_map, dict) else {}

    semantic_entities = sorted(
        [(str(node), float(score)) for node, score in (semantic_entity_scores or {}).items()],
        key=lambda x: x[1],
        reverse=True,
    )[:entity_pool_cap]
    semantic_chunks = sorted(
        [(str(node), float(score)) for node, score in (semantic_chunk_scores or {}).items()],
        key=lambda x: x[1],
        reverse=True,
    )[:chunk_pool_cap]

    high_conf_votes = {}
    high_conf_score = {}

    for anchor in anchors:
        dmap = {}
        if anchor in g and anchor_distance_bonus:
            try:
                dmap = nx.single_source_shortest_path_length(g, anchor, cutoff=dist_cap_effective)
            except Exception:
                dmap = {}

        cache_reserve_nodes, cache_reserve_scores = _anchor_cache_reserve_candidates(
            g=g,
            anchor=anchor,
            topn=reserve_cap,
            semantic_state=semantic_state,
        )
        graph_reserve_nodes = _anchor_graph_reserve_candidates(
            g=g,
            anchor=anchor,
            topn=graph_mix_topn,
            max_hops=reserve_hops_effective,
            precomputed_dmap=dmap if dmap else None,
        )
        reserve_nodes = _ordered_unique(list(cache_reserve_nodes or []) + list(graph_reserve_nodes or []))
        if not reserve_nodes and reserve_bfs_fallback:
            reserve_nodes = list(cache_reserve_nodes or [])
        reserve_nodes = reserve_nodes[:reserve_cap]

        ent_ranked = []
        for node, raw_score in semantic_entities:
            if node not in g:
                continue
            bonus = 0.0
            if anchor_distance_bonus and node in dmap:
                bonus = 1.0 - (float(min(dist_cap_effective, int(dmap[node]))) / float(max(1, dist_cap_effective)))
            ent_ranked.append((node, 0.90 * float(raw_score) + 0.10 * bonus))
        ent_ranked.sort(key=lambda x: x[1], reverse=True)
        top_entities = [node for node, _ in ent_ranked[:entity_cap]]

        chunk_ranked = []
        for node, raw_score in semantic_chunks:
            if node not in g:
                continue
            bonus = 0.0
            if anchor_distance_bonus and node in dmap:
                bonus = 1.0 - (float(min(dist_cap_effective, int(dmap[node]))) / float(max(1, dist_cap_effective)))
            chunk_ranked.append((node, 0.90 * float(raw_score) + 0.10 * bonus))
        chunk_ranked.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [node for node, _ in chunk_ranked[:chunk_cap]]

        high_conf_nodes = []
        for ent in top_entities[: max(3, entity_cap // 2)]:
            linked = []
            linked.extend(list(ent2chk.get(str(ent), []) or []))
            linked.extend(list(support_map.get(str(ent), []) or []))
            added = 0
            for node in linked:
                node_id = str(node)
                if node_id not in g:
                    continue
                high_conf_nodes.append(node_id)
                added += 1
                if added >= support_per_entity:
                    break
        for node in reserve_nodes:
            high_conf_nodes.append(str(node))

        high_conf_nodes = _ordered_unique(high_conf_nodes)[: max(2, reserve_cap)]
        for node in high_conf_nodes:
            high_conf_votes[node] = int(high_conf_votes.get(node, 0)) + 1
            high_conf_score[node] = max(
                float(high_conf_score.get(node, -1.0e9)),
                float(cache_reserve_scores.get(node, semantic_scores.get(node, 0.0))),
            )

        semantic_nodes = _ordered_unique(top_entities + top_chunks)
        graph_reserve_union.update([node for node in reserve_nodes if node in g])
        merged = _ordered_unique([anchor] + high_conf_nodes + semantic_nodes + reserve_nodes)[:local_cap]
        proposal_by_anchor[anchor] = merged

        for rank, node in enumerate(merged):
            base = float(semantic_scores.get(node, 0.0))
            if node in high_conf_nodes:
                base = max(base, 0.10)
            elif node in reserve_nodes:
                base = max(base, 0.05)
            rank_decay = 1.0 / float(rank + 1)
            proposal_scores[node] = max(float(proposal_scores.get(node, -1.0e9)), base + 0.02 * rank_decay)

    shared_ranked = sorted(
        [n for n in high_conf_votes.keys() if int(high_conf_votes.get(n, 0)) >= 2 and n in g],
        key=lambda n: (int(high_conf_votes.get(n, 0)), float(high_conf_score.get(n, 0.0))),
        reverse=True,
    )
    if not shared_ranked:
        shared_ranked = sorted(
            [n for n in high_conf_votes.keys() if n in g],
            key=lambda n: float(high_conf_score.get(n, 0.0)),
            reverse=True,
        )
    shared_high_conf = [str(n) for n in shared_ranked[:shared_high_conf_topn]]
    shared_set = set(shared_high_conf)

    global_pool = []
    for node, score in semantic_entities:
        if node in g and node not in shared_set:
            global_pool.append((node, float(score)))
    for node, score in semantic_chunks:
        if node in g and node not in shared_set:
            global_pool.append((node, float(score)))
    global_pool.sort(key=lambda x: x[1], reverse=True)
    global_fallback = [str(node) for node, _ in global_pool[:global_fallback_topn]]

    proposal_nodes = set()
    for vals in proposal_by_anchor.values():
        proposal_nodes.update(vals)
    proposal_nodes.update(shared_high_conf)
    proposal_nodes.update(global_fallback)
    proposal_nodes = {node for node in proposal_nodes if node in g}

    proposal_diag = {
        "proposal_entity_count": int(sum(1 for node in semantic_entity_scores.keys() if node in g)),
        "proposal_chunk_count": int(sum(1 for node in semantic_chunk_scores.keys() if node in g)),
        "graph_reserve_count": int(len(graph_reserve_union)),
        "high_confidence_count": int(len(shared_high_conf)),
        "global_fallback_count": int(len(global_fallback)),
        "union_candidate_count": int(len(proposal_nodes)),
    }
    proposal_partitions = {
        "shared_high_conf": [str(n) for n in shared_high_conf],
        "global_fallback": [str(n) for n in global_fallback],
    }
    return proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores, proposal_diag, proposal_partitions


def _build_anchor_proposals(g, anchors, semantic_entity_scores, semantic_chunk_scores, cfg, semantic_state=None):
    proposal_mode = _resolve_proposal_union_experiment_mode(cfg)
    if proposal_mode != "off":
        return _build_anchor_proposals_lazy_experimental(
            g=g,
            anchors=anchors,
            semantic_entity_scores=semantic_entity_scores,
            semantic_chunk_scores=semantic_chunk_scores,
            cfg=cfg,
            semantic_state=semantic_state,
            mode=proposal_mode,
        )

    topn_entity = max(0, int(getattr(cfg, "semantic_topn_entity", getattr(cfg, "semantic_topn", 50))))
    topn_chunk = max(0, int(getattr(cfg, "semantic_topn_chunk", max(1, topn_entity // 2))))
    reserve_topn = max(0, int(getattr(cfg, "graph_reserve_topn", 15)))
    dist_cap = max(1, int(getattr(cfg, "tau", 4)))
    reserve_hops = max(2, int(getattr(cfg, "proposal_reserve_hops", dist_cap)))
    reserve_bfs_fallback = bool(getattr(cfg, "proposal_reserve_bfs_fallback", False))
    anchor_distance_bonus = bool(getattr(cfg, "proposal_anchor_distance_bonus", True))
    proposal_by_anchor = {}
    proposal_scores = {}
    graph_reserve_union = set()
    semantic_scores = {}
    semantic_scores.update({str(k): float(v) for k, v in (semantic_entity_scores or {}).items()})
    semantic_scores.update({str(k): float(v) for k, v in (semantic_chunk_scores or {}).items()})

    semantic_entities = list((semantic_entity_scores or {}).items())
    semantic_chunks = list((semantic_chunk_scores or {}).items())

    for anchor in anchors:
        dmap = {}
        if anchor in g and anchor_distance_bonus:
            try:
                dmap = nx.single_source_shortest_path_length(g, anchor, cutoff=dist_cap)
            except Exception:
                dmap = {}

        reserve_nodes = _anchor_graph_reserve_candidates(
            g=g,
            anchor=anchor,
            topn=reserve_topn,
            max_hops=reserve_hops,
            precomputed_dmap=dmap if dmap else None,
        )
        if not reserve_nodes:
            cache_reserve_nodes, _ = _anchor_cache_reserve_candidates(
                g=g,
                anchor=anchor,
                topn=reserve_topn,
                semantic_state=semantic_state,
            )
            reserve_nodes = list(cache_reserve_nodes[:reserve_topn]) if cache_reserve_nodes else []
        elif reserve_bfs_fallback and len(reserve_nodes) < max(2, reserve_topn // 2):
            cache_reserve_nodes, _ = _anchor_cache_reserve_candidates(
                g=g,
                anchor=anchor,
                topn=reserve_topn,
                semantic_state=semantic_state,
            )
            for node in cache_reserve_nodes:
                if node in reserve_nodes:
                    continue
                reserve_nodes.append(node)
                if len(reserve_nodes) >= reserve_topn:
                    break

        ent_ranked = []
        for node, raw_score in semantic_entities:
            if node not in g:
                continue
            bonus = 0.0
            if anchor_distance_bonus and node in dmap:
                bonus = 1.0 - (float(min(dist_cap, int(dmap[node]))) / float(dist_cap))
            ent_ranked.append((node, 0.90 * float(raw_score) + 0.10 * bonus))

        chunk_ranked = []
        for node, raw_score in semantic_chunks:
            if node not in g:
                continue
            bonus = 0.0
            if anchor_distance_bonus and node in dmap:
                bonus = 1.0 - (float(min(dist_cap, int(dmap[node]))) / float(dist_cap))
            chunk_ranked.append((node, 0.90 * float(raw_score) + 0.10 * bonus))

        ent_ranked.sort(key=lambda x: x[1], reverse=True)
        chunk_ranked.sort(key=lambda x: x[1], reverse=True)
        semantic_nodes = [n for n, _ in ent_ranked[:topn_entity]] + [n for n, _ in chunk_ranked[:topn_chunk]]
        graph_reserve_union.update([node for node in reserve_nodes if node in g])
        merged = _ordered_unique([anchor] + semantic_nodes + reserve_nodes)
        proposal_by_anchor[anchor] = merged

        for rank, node in enumerate(merged):
            base = float(semantic_scores.get(node, 0.0))
            if node in reserve_nodes:
                base = max(base, 0.05)
            rank_decay = 1.0 / float(rank + 1)
            proposal_scores[node] = max(float(proposal_scores.get(node, -1.0e9)), base + 0.02 * rank_decay)

    proposal_nodes = set()
    for vals in proposal_by_anchor.values():
        proposal_nodes.update(vals)
    proposal_nodes = {node for node in proposal_nodes if node in g}
    proposal_diag = {
        "proposal_entity_count": int(sum(1 for node in semantic_entity_scores.keys() if node in g)),
        "proposal_chunk_count": int(sum(1 for node in semantic_chunk_scores.keys() if node in g)),
        "graph_reserve_count": int(len(graph_reserve_union)),
        "high_confidence_count": 0,
        "global_fallback_count": 0,
        "union_candidate_count": int(len(proposal_nodes)),
    }
    proposal_partitions = {
        "shared_high_conf": [],
        "global_fallback": [],
    }
    return proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores, proposal_diag, proposal_partitions


def _build_reduced_subgraph_from_proposals(g, anchors, proposal_by_anchor, proposal_scores, cfg):
    if g is None or g.number_of_nodes() <= 0:
        return g, {"applied": False, "reason": "empty_graph"}

    reduced_max_nodes = max(200, int(getattr(cfg, "ppr_subgraph_max_nodes", 30000)))
    path_hops = max(2, int(getattr(cfg, "tau", 4)) + 1)
    path_links_per_anchor = max(1, int(getattr(cfg, "graph_reserve_topn", 15)) // 2)

    keep = set()
    for anchor in anchors:
        keep.add(anchor)
        anchor_props = list(proposal_by_anchor.get(anchor, []) or [])
        keep.update(anchor_props)
        for node in anchor_props[:path_links_per_anchor]:
            if node == anchor or anchor not in g or node not in g:
                continue
            dist = _shortest_distance_with_cap(g, anchor, node, path_hops)
            if dist >= path_hops:
                continue
            try:
                path = nx.shortest_path(g, anchor, node)
            except Exception:
                path = []
            if path:
                keep.update(path)

    # Keep lightweight connector coverage for relation nodes.
    connector = set()
    for node in list(keep):
        if node not in g:
            continue
        for nbr in g.neighbors(node):
            if g.nodes[nbr].get("node_type") == "relation":
                connector.add(nbr)
    keep.update(connector)
    keep = {node for node in keep if node in g}

    if len(keep) > reduced_max_nodes:
        mandatory = {a for a in anchors if a in g}
        budget = max(0, reduced_max_nodes - len(mandatory))
        ranked = []
        for node in keep:
            if node in mandatory:
                continue
            base = float(proposal_scores.get(node, 0.0))
            degree_term = min(1.0, float(g.degree(node)) / 25.0)
            ranked.append((node, base + 0.02 * degree_term))
        ranked.sort(key=lambda x: x[1], reverse=True)
        keep = set(mandatory)
        keep.update([node for node, _ in ranked[:budget]])

    if not keep:
        return g, {"applied": False, "reason": "empty_keep_set"}
    if len(keep) >= g.number_of_nodes():
        return g, {"applied": False, "reason": "no_reduction"}

    reduced = g.subgraph(sorted(keep)).copy()
    return reduced, {
        "applied": True,
        "nodes_before": int(g.number_of_nodes()),
        "edges_before": int(g.number_of_edges()),
        "nodes_after": int(reduced.number_of_nodes()),
        "edges_after": int(reduced.number_of_edges()),
        "max_nodes": int(reduced_max_nodes),
    }


def _lazy_union_candidates_for_run(
    anchor_scores,
    anchors,
    proposal_by_anchor,
    cfg,
    shared_high_conf=None,
    global_fallback=None,
):
    mode = _resolve_proposal_union_experiment_mode(cfg)
    topk = max(0, int(getattr(cfg, "proposal_lazy_union_topk", 64)))
    shared_cap = max(2, int(getattr(cfg, "proposal_shared_high_conf_topn", 24)))
    fallback_cap = max(2, int(getattr(cfg, "proposal_global_fallback_topn", 16)))
    anchor_local_cap = max(4, int(getattr(cfg, "proposal_anchor_local_topn", 48)))

    if mode == "mild":
        topk = max(8, int(round(float(topk) * 0.90)))
        shared_cap = max(4, int(round(float(shared_cap) * 0.90)))
        fallback_cap = max(4, int(round(float(fallback_cap) * 0.85)))
        anchor_local_cap = max(6, int(round(float(anchor_local_cap) * 0.90)))
    elif mode == "medium":
        topk = max(8, int(round(float(topk) * 0.75)))
        shared_cap = max(4, int(round(float(shared_cap) * 0.80)))
        fallback_cap = max(3, int(round(float(fallback_cap) * 0.70)))
        anchor_local_cap = max(6, int(round(float(anchor_local_cap) * 0.75)))
    elif mode == "aggressive":
        topk = max(8, int(round(float(topk) * 0.60)))
        shared_cap = max(3, int(round(float(shared_cap) * 0.65)))
        fallback_cap = max(2, int(round(float(fallback_cap) * 0.55)))
        anchor_local_cap = max(4, int(round(float(anchor_local_cap) * 0.60)))

    if topk <= 0:
        merged = []
        for anchor in anchors:
            merged.extend(list(proposal_by_anchor.get(anchor, []) or []))
        return _ordered_unique(merged)

    anchor_priority = []
    for anchor in anchors:
        a_scores = (anchor_scores or {}).get(anchor, {}) or {}
        best = max([float(v) for v in a_scores.values()], default=0.0)
        anchor_priority.append((anchor, best))
    anchor_priority.sort(key=lambda x: x[1], reverse=True)

    merged = []
    seen = set()

    def _append_nodes(nodes, cap=None):
        added = 0
        for node in nodes:
            node_id = str(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            merged.append(node_id)
            added += 1
            if len(merged) >= topk:
                break
            if cap is not None and added >= int(cap):
                break

    per_anchor_cap = max(1, min(anchor_local_cap, int(topk // max(len(anchor_priority), 1))))
    shared_first = mode in {"medium", "aggressive"}
    if shared_first:
        _append_nodes(list(shared_high_conf or [])[:shared_cap], cap=shared_cap)

    for anchor, _ in anchor_priority:
        _append_nodes(list(proposal_by_anchor.get(anchor, []) or []), cap=per_anchor_cap)
        if len(merged) >= topk:
            break

    if len(merged) < topk:
        _append_nodes(list(shared_high_conf or [])[:shared_cap], cap=shared_cap)

    if len(merged) < topk:
        _append_nodes(list(global_fallback or [])[:fallback_cap], cap=fallback_cap)

    if len(merged) < topk:
        for anchor in anchors:
            _append_nodes(list(proposal_by_anchor.get(anchor, []) or []), cap=None)
            if len(merged) >= topk:
                break

    return merged[:topk]


def _weighted_sum_active_components(components, weights, disabled_keys=None):
    disabled = {str(k) for k in (disabled_keys or set())}
    ordered_keys = [str(k) for k in weights.keys()]
    original_weight_sum = sum(float(weights[k]) for k in ordered_keys)
    active_keys = [k for k in ordered_keys if k not in disabled]
    if not active_keys:
        raise ValueError("All score components are disabled; cannot compute weighted sum.")
    active_weight_sum = sum(float(weights[k]) for k in active_keys)
    if active_weight_sum <= 0.0:
        raise ValueError("Active score component weight sum must be positive.")

    scale = float(original_weight_sum) / float(active_weight_sum)
    score = 0.0
    effective_weights = {}
    for key in active_keys:
        eff_w = float(weights[key]) * float(scale)
        effective_weights[key] = float(eff_w)
        score += float(eff_w) * float(components.get(key, 0.0))
    return float(score), effective_weights


def _phase2_pair_shortlist(shortlisted_runs, anchors, g, query_sim_map, support_sim_map, cfg):
    topb = max(1, int(getattr(cfg, "pair_shortlist_topb", 6)))
    cap = max(2, int(getattr(cfg, "tau", 4)) + 1)
    pair_scores = {}
    trace_enabled = bool(getattr(cfg, "score_component_trace_enabled", False))
    ablate_pair_semantic = bool(getattr(cfg, "ablation_no_pair_semantic", False))
    ablate_pair_bridge = bool(getattr(cfg, "ablation_no_pair_bridge", False))
    pair_component_weights = {
        "anchor_align": 0.30,
        "seed_strength": 0.25,
        "semantic_rel": 0.20,
        "dist_score": 0.15,
        "bridge_potential": 0.10,
    }

    for run in shortlisted_runs:
        run_id = int(run.get("run_id", 0))
        seeds = list(run.get("seeds", []) or [])
        if not seeds:
            continue
        seed_scores = run.get("seed_score_map", {}) or {}
        for anchor in anchors:
            anchor_scores = (run.get("anchor_scores", {}) or {}).get(anchor, {}) or {}
            for seed in seeds:
                if anchor not in g or seed not in g:
                    continue
                dist = _shortest_distance_with_cap(g, anchor, seed, cap)
                dist_score = 1.0 - (float(dist) / float(cap))
                anchor_align = float(anchor_scores.get(seed, 0.0))
                seed_strength = float(seed_scores.get(seed, 0.0))
                semantic_rel = 0.5 * (
                    max(float(query_sim_map.get(seed, 0.0)), float(support_sim_map.get(seed, 0.0))) + 1.0
                )

                bridge_hits = 0
                for other_anchor in anchors:
                    if other_anchor == anchor:
                        continue
                    d_other = _shortest_distance_with_cap(g, other_anchor, seed, cap)
                    if d_other < cap:
                        bridge_hits += 1
                bridge_potential = float(bridge_hits) / float(max(1, len(anchors) - 1))

                components = {
                    "anchor_align": float(anchor_align),
                    "seed_strength": float(seed_strength),
                    "semantic_rel": float(semantic_rel),
                    "dist_score": float(dist_score),
                    "bridge_potential": float(bridge_potential),
                }
                if not (ablate_pair_semantic or ablate_pair_bridge):
                    # Keep the default path formula exactly as-is for behavior preservation.
                    score = (
                        0.30 * float(anchor_align)
                        + 0.25 * float(seed_strength)
                        + 0.20 * float(semantic_rel)
                        + 0.15 * float(dist_score)
                        + 0.10 * float(bridge_potential)
                    )
                    effective_weights = None
                    disabled = set()
                else:
                    disabled = set()
                    if ablate_pair_semantic:
                        disabled.add("semantic_rel")
                    if ablate_pair_bridge:
                        disabled.add("bridge_potential")
                    score, effective_weights = _weighted_sum_active_components(
                        components=components,
                        weights=pair_component_weights,
                        disabled_keys=disabled,
                    )
                pair = (anchor, seed)
                prev = pair_scores.get(pair)
                payload = {
                    "anchor": anchor,
                    "seed": seed,
                    "run_id": run_id,
                    "pair_proxy_score": float(score),
                    "distance": int(dist),
                    "anchor_alignment": float(anchor_align),
                    "seed_strength": float(seed_strength),
                    "semantic_relevance": float(semantic_rel),
                    "bridge_potential": float(bridge_potential),
                }
                if trace_enabled:
                    payload["pair_score_components"] = dict(components)
                    payload["pair_score_effective_weights"] = (
                        dict(effective_weights)
                        if effective_weights is not None
                        else dict(pair_component_weights)
                    )
                    payload["pair_score_disabled_components"] = sorted([str(x) for x in disabled])
                if prev is None or float(payload["pair_proxy_score"]) > float(prev["pair_proxy_score"]):
                    pair_scores[pair] = payload

    ranked = sorted(pair_scores.values(), key=lambda x: x["pair_proxy_score"], reverse=True)
    return ranked[:topb]


def _apply_final_text_rerank(
    *,
    question,
    selected_sentence_ids,
    selected_sentences,
    selected_sentence_score_map,
    cfg,
):
    sentence_rerank_enabled = bool(getattr(cfg, "sentence_rerank_enabled", True))
    ablation_no_final_text_rerank = bool(getattr(cfg, "ablation_no_final_text_rerank", False))
    embedding_diag = {
        "enabled": bool(getattr(cfg, "embedding_enabled", False) and sentence_rerank_enabled),
        "sentence_rerank_enabled": bool(sentence_rerank_enabled),
        "applied": False,
        "error": "",
        "model_name": str(getattr(cfg, "embedding_model_name", "") or ""),
        "weight": float(getattr(cfg, "embedding_weight", 0.35)),
        "rerank_topn": int(getattr(cfg, "embedding_rerank_topn", 80)),
        "batch_size": int(getattr(cfg, "embedding_batch_size", 16)),
        "max_length": int(getattr(cfg, "embedding_max_length", 192)),
        "max_chars": int(getattr(cfg, "embedding_text_max_chars", 600)),
        "head_size": 0,
    }
    sentence_rerank_semantic_calls = 0
    sentence_rerank_ms = 0.0

    if ablation_no_final_text_rerank:
        embedding_diag["enabled"] = False
        embedding_diag["error"] = "ablation_no_final_text_rerank"
    elif embedding_diag["enabled"] and selected_sentence_ids:
        sentence_rerank_start = time.perf_counter()
        sentence_rerank_semantic_calls += 1
        rerank = rerank_sentences_by_embedding(
            question=question,
            sentence_ids=selected_sentence_ids,
            sentence_texts=selected_sentences,
            base_score_map=selected_sentence_score_map,
            model_name=embedding_diag["model_name"],
            weight=embedding_diag["weight"],
            rerank_topn=embedding_diag["rerank_topn"],
            batch_size=embedding_diag["batch_size"],
            max_length=embedding_diag["max_length"],
            max_chars=embedding_diag["max_chars"],
        )
        selected_sentence_ids = list(rerank.get("ranked_sentence_ids", selected_sentence_ids))
        selected_sentences = list(rerank.get("ranked_sentence_texts", selected_sentences))
        embedding_diag["applied"] = bool(rerank.get("applied", False))
        embedding_diag["error"] = str(rerank.get("error", "") or "")
        embedding_diag["head_size"] = int(rerank.get("rerank_topn", 0))
        embedding_diag["similarity_by_sentence_id"] = rerank.get("similarity_by_sentence_id", {})
        embedding_diag["fused_score_by_sentence_id"] = rerank.get("fused_score_by_sentence_id", {})
        sentence_rerank_ms = float((time.perf_counter() - sentence_rerank_start) * 1000.0)
    elif not bool(sentence_rerank_enabled):
        embedding_diag["error"] = "sentence_rerank_disabled_by_config"

    return (
        list(selected_sentence_ids),
        list(selected_sentences),
        embedding_diag,
        int(sentence_rerank_semantic_calls),
        float(sentence_rerank_ms),
    )


def _phase2_refine_pair_bounded_local(g, pair_item, run_by_id, query_sim_map, support_sim_map, cfg):
    anchor = pair_item["anchor"]
    seed = pair_item["seed"]
    run_id = int(pair_item.get("run_id", 0))
    run = run_by_id.get(run_id, {})
    anchor_scores = (run.get("anchor_scores", {}) or {}).get(anchor, {}) or {}

    local_hops = max(2, int(getattr(cfg, "tau", 4)) + 1)
    max_local_nodes = max(80, int(getattr(cfg, "corridor_top_bc", 20)) * 8)
    corridor_top_bc = max(1, int(getattr(cfg, "corridor_top_bc", 20)))

    a_nodes = {}
    z_nodes = {}
    try:
        a_nodes = nx.single_source_shortest_path_length(g, anchor, cutoff=local_hops)
    except Exception:
        a_nodes = {anchor: 0}
    try:
        z_nodes = nx.single_source_shortest_path_length(g, seed, cutoff=local_hops)
    except Exception:
        z_nodes = {seed: 0}

    local_nodes = set(a_nodes.keys()).union(z_nodes.keys())
    if anchor in g and seed in g:
        try:
            sp = nx.shortest_path(g, anchor, seed)
        except Exception:
            sp = []
        local_nodes.update(sp)
    local_nodes.add(anchor)
    local_nodes.add(seed)

    if len(local_nodes) > max_local_nodes:
        rank = []
        for node in local_nodes:
            da = a_nodes.get(node, local_hops)
            dz = z_nodes.get(node, local_hops)
            dist_term = 1.0 / float(1 + min(da, dz))
            sem_term = 0.5 * (
                max(float(query_sim_map.get(node, 0.0)), float(support_sim_map.get(node, 0.0))) + 1.0
            )
            rank.append((node, 0.70 * dist_term + 0.30 * sem_term))
        rank.sort(key=lambda x: x[1], reverse=True)
        kept = [node for node, _ in rank[: max_local_nodes]]
        local_nodes = set(kept)
        local_nodes.update([anchor, seed])

    local_nodes = {node for node in local_nodes if node in g}
    local_subgraph = g.subgraph(sorted(local_nodes)).copy()
    if local_subgraph.number_of_nodes() <= 0:
        return None

    local_ppr = _personalized_pagerank(
        local_subgraph,
        source=anchor,
        alpha=float(getattr(cfg, "ppr_alpha", 0.15)),
        max_iter=min(80, int(getattr(cfg, "ppr_power_max_iter", 100))),
        tol=max(1.0e-6, float(getattr(cfg, "ppr_power_tol", 1.0e-6))),
        min_score=max(0.0, float(getattr(cfg, "ppr_min_score", 0.0))),
    )
    local_ppr_norm = _normalize_map(local_ppr)
    anchor_graph_norm = _normalize_map(anchor_scores)

    reverse_raw = {}
    for node in local_subgraph.nodes:
        d = _shortest_distance_with_cap(local_subgraph, node, seed, local_hops)
        reverse_raw[node] = 1.0 - (float(d) / float(local_hops))
    reverse_norm = _normalize_map(reverse_raw)

    semantic_raw = {}
    for node in local_subgraph.nodes:
        semantic_raw[node] = 0.5 * (
            max(float(query_sim_map.get(node, 0.0)), float(support_sim_map.get(node, 0.0))) + 1.0
        )
    semantic_norm = _normalize_map(semantic_raw)

    node_scores = {}
    for node in local_subgraph.nodes:
        node_scores[node] = (
            0.40 * float(local_ppr_norm.get(node, 0.0))
            + 0.25 * float(reverse_norm.get(node, 0.0))
            + 0.20 * float(anchor_graph_norm.get(node, 0.0))
            + 0.15 * float(semantic_norm.get(node, 0.0))
        )

    ranked = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    top_node_scores = ranked[:corridor_top_bc]
    if not top_node_scores:
        return None

    payload = _build_single_corridor_payload(
        g=g,
        anchor=anchor,
        seed=seed,
        corridor_id=f"c{int(run_id):02d}_{anchor}_{seed}",
        corridor_score=float(pair_item.get("pair_proxy_score", 0.0)),
        node_scores=top_node_scores,
        fallback_text_cap=max(1, int(getattr(cfg, "corridor_fallback_text_cap", 4))),
    )
    payload["pair_proxy_score"] = float(pair_item.get("pair_proxy_score", 0.0))
    payload["anchor_alignment"] = float(pair_item.get("anchor_alignment", 0.0))
    payload["seed_strength"] = float(pair_item.get("seed_strength", 0.0))
    payload["semantic_relevance"] = float(pair_item.get("semantic_relevance", 0.0))
    payload["bridge_potential"] = float(pair_item.get("bridge_potential", 0.0))
    payload["distance"] = int(pair_item.get("distance", max(1, int(getattr(cfg, "tau", 4)) + 1)))
    payload["refine_mode"] = str(getattr(cfg, "phase2_refine_mode", "bounded_local") or "bounded_local")
    return {
        "pair": (anchor, seed),
        "payload": payload,
        "node_scores": top_node_scores,
    }


def _phase2_local_refinement(g, shortlisted_pairs, shortlisted_runs, query_sim_map, support_sim_map, cfg):
    run_by_id = {int(r.get("run_id", 0)): r for r in (shortlisted_runs or [])}
    sentence_scores = {}
    corridor_nodes = set()
    corridor_payloads = []
    retained_pairs = []

    for pair_item in shortlisted_pairs:
        refined = _phase2_refine_pair_bounded_local(
            g=g,
            pair_item=pair_item,
            run_by_id=run_by_id,
            query_sim_map=query_sim_map,
            support_sim_map=support_sim_map,
            cfg=cfg,
        )
        if refined is None:
            continue

        anchor, seed = refined["pair"]
        retained_pairs.append((anchor, seed))
        corridor_payloads.append(refined["payload"])
        corridor_nodes.update([anchor, seed])
        for node, score in refined["node_scores"]:
            corridor_nodes.add(node)
            if g.nodes[node].get("node_type") == "sentence":
                sentence_scores[node] = max(float(sentence_scores.get(node, 0.0)), float(score))

    if not corridor_nodes:
        corridor_nodes = set()
        for pair_item in shortlisted_pairs:
            corridor_nodes.add(pair_item["anchor"])
            corridor_nodes.add(pair_item["seed"])
        corridor_nodes = {node for node in corridor_nodes if node in g}

    if corridor_nodes:
        corridor_graph = g.subgraph(sorted(corridor_nodes)).copy()
    else:
        corridor_graph = g.subgraph([]).copy()
    return corridor_graph, retained_pairs, sentence_scores, corridor_payloads


def _rerank_corridors_hybrid(
    corridors,
    g,
    query_sim_map,
    support_sim_map,
    cfg,
    coverage_enabled_override=None,
):
    if not corridors:
        return []

    sid_to_node = {}
    for node in g.nodes:
        if g.nodes[node].get("node_type") != "sentence":
            continue
        sid = str(g.nodes[node].get("sentence_id", node))
        sid_to_node[sid] = node

    reuse_semantic = bool(getattr(cfg, "reuse_semantic_scores_in_final", True))
    redundancy_w = max(0.0, float(getattr(cfg, "run_score_redundancy_weight", 0.10)))
    coverage_enabled = (
        bool(coverage_enabled_override)
        if coverage_enabled_override is not None
        else bool(getattr(cfg, "coverage_selection_enabled", False))
    )
    coverage_role_w = max(0.0, float(getattr(cfg, "coverage_selection_role_weight", 0.35)))
    coverage_red_w = max(0.0, float(getattr(cfg, "coverage_selection_redundancy_weight", 0.10)))
    compact_enabled = bool(getattr(cfg, "corridor_compact_shaping_enabled", False))
    answer_preserve_enabled = bool(getattr(cfg, "corridor_answer_preserve_enabled", False))
    bridge_purity_enabled = bool(getattr(cfg, "corridor_bridge_purity_shaping_enabled", False))
    path_preserve_enabled = bool(getattr(cfg, "corridor_path_preserve_enabled", False))
    path_preserve_compact_enabled = bool(getattr(cfg, "corridor_path_preserve_compact_enabled", False))
    path_preserve_guarded_enabled = bool(getattr(cfg, "corridor_path_preserve_guarded_enabled", False))
    path_preserve_compact_lite_enabled = bool(getattr(cfg, "corridor_path_preserve_compact_lite_enabled", False))
    answer_preserve_guarded_hotpot_enabled = bool(
        getattr(cfg, "corridor_answer_preserve_guarded_hotpot_enabled", False)
    )
    answer_preserve_confidence_gated_enabled = bool(
        getattr(cfg, "corridor_answer_preserve_confidence_gated_enabled", False)
    )
    compact_w = max(0.0, float(getattr(cfg, "corridor_compact_weight", 0.0)))
    answer_preserve_w = max(0.0, float(getattr(cfg, "corridor_answer_preserve_weight", 0.0)))
    bridge_purity_w = max(0.0, float(getattr(cfg, "corridor_bridge_purity_weight", 0.0)))
    path_preserve_w = max(0.0, float(getattr(cfg, "corridor_path_preserve_weight", 0.0)))
    path_preserve_incomplete_penalty_w = max(
        0.0, float(getattr(cfg, "corridor_path_preserve_incomplete_penalty_weight", 0.0))
    )
    path_preserve_compact_w = max(0.0, float(getattr(cfg, "corridor_path_preserve_compact_weight", 0.0)))
    path_anchor_threshold = max(0.0, min(1.0, float(getattr(cfg, "corridor_path_preserve_anchor_threshold", 0.35))))
    path_bridge_threshold = max(0.0, min(1.0, float(getattr(cfg, "corridor_path_preserve_bridge_threshold", 0.35))))
    path_answer_threshold = max(0.0, min(1.0, float(getattr(cfg, "corridor_path_preserve_answer_threshold", 0.55))))
    path_pair_threshold = max(0.0, min(1.0, float(getattr(cfg, "corridor_path_preserve_pair_threshold", 0.45))))
    path_guard_scale = max(0.0, min(1.0, float(getattr(cfg, "corridor_path_preserve_guard_scale", 0.70))))
    path_guard_min_complete = max(
        0.0, min(1.0, float(getattr(cfg, "corridor_path_preserve_guard_min_complete_score", 0.70)))
    )
    path_guard_min_pair = max(
        0.0, min(1.0, float(getattr(cfg, "corridor_path_preserve_guard_min_pair_retention", 0.55)))
    )
    path_guard_min_answer_density = max(
        0.0, min(1.0, float(getattr(cfg, "corridor_path_preserve_guard_min_answer_density", 0.62)))
    )
    path_compact_lite_scale = max(
        0.0, min(1.0, float(getattr(cfg, "corridor_path_preserve_compact_lite_scale", 0.65)))
    )
    path_compact_lite_overflow_coeff = max(
        0.0, float(getattr(cfg, "corridor_path_preserve_compact_lite_overflow_coeff", 0.015))
    )
    path_compact_lite_overflow_cap = max(
        0.0, float(getattr(cfg, "corridor_path_preserve_compact_lite_overflow_cap", 0.08))
    )
    answer_density_min = max(0.0, min(1.0, float(getattr(cfg, "corridor_answer_preserve_min_density", 0.30))))
    answer_preserve_light_boost_scale = max(
        0.0, min(1.0, float(getattr(cfg, "corridor_answer_preserve_light_boost_scale", 0.65)))
    )
    answer_preserve_bridge_consistency_threshold = max(
        0.0, min(1.0, float(getattr(cfg, "corridor_answer_preserve_bridge_consistency_threshold", 0.55)))
    )
    answer_preserve_confidence_threshold = max(
        0.0, min(1.0, float(getattr(cfg, "corridor_answer_preserve_confidence_threshold", 0.62)))
    )
    hotpot_anchor_bridge_sufficient = max(
        0.0, min(1.0, float(getattr(cfg, "corridor_answer_preserve_hotpot_anchor_bridge_sufficient", 0.70)))
    )
    hotpot_answer_density_cap = max(
        0.0, min(1.0, float(getattr(cfg, "corridor_answer_preserve_hotpot_answer_density_cap", 0.72)))
    )
    hotpot_quota_cap = max(0.0, min(1.0, float(getattr(cfg, "corridor_answer_preserve_hotpot_quota_cap", 0.40))))
    hotpot_relax_factor = max(0.0, min(1.0, float(getattr(cfg, "corridor_answer_preserve_hotpot_relax_factor", 0.35))))
    dataset_name = str(getattr(cfg, "dataset", "") or "").strip().lower()

    base_rows = []
    for corridor in corridors:
        main_ids = list(corridor.get("main_path_sentence_ids", []) or [])
        support_ids = list(corridor.get("support_sentence_ids", []) or [])
        connector_ids = list(
            corridor.get("connector_adjacent_unit_ids", corridor.get("connector_adjacent_sentence_ids", [])) or []
        )
        sent_ids = _ordered_unique(main_ids + support_ids + connector_ids)

        structural = 0.0
        if main_ids:
            structural += 0.6
        structural += 0.2 * min(1.0, float(len(main_ids)) / 2.0)
        structural += 0.2 * min(1.0, float(len(support_ids)) / 2.0)

        sem_vals = []
        support_vals = []
        answer_like_by_sid = {}
        for sid in sent_ids:
            node = sid_to_node.get(sid)
            if node is None:
                continue
            if reuse_semantic:
                sem = 0.5 * (float(query_sim_map.get(node, 0.0)) + 1.0)
            else:
                sem = 0.0
            sup = 0.5 * (float(support_sim_map.get(node, 0.0)) + 1.0)
            sem_vals.append(sem)
            support_vals.append(sup)
            answer_like_by_sid[sid] = max(0.0, min(1.0, max(float(sem), float(sup))))
        semantic_rel = float(sum(sem_vals) / max(len(sem_vals), 1)) if sem_vals else 0.0
        answer_support = float(sum(support_vals) / max(len(support_vals), 1)) if support_vals else 0.0
        support_density = float(len(support_ids)) / float(max(1, len(sent_ids)))
        anchor_alignment = float(corridor.get("anchor_alignment", 0.0) or 0.0)
        bridge_utility = float(corridor.get("bridge_potential", 0.0) or 0.0)
        answer_alignment = max(float(answer_support), float(semantic_rel), float(anchor_alignment))
        answer_side_ids = _ordered_unique(connector_ids + support_ids)
        if not answer_side_ids:
            answer_side_ids = list(sent_ids)
        answer_side_vals = [float(answer_like_by_sid.get(sid, 0.0)) for sid in answer_side_ids]
        answer_side_density = float(sum(answer_side_vals) / max(len(answer_side_vals), 1)) if answer_side_vals else 0.0

        bridge_focus_ids = _ordered_unique(connector_ids + support_ids)
        bridge_purity = max(0.0, min(1.0, float(bridge_utility)))
        if bridge_focus_ids:
            useful = 0
            for sid in bridge_focus_ids:
                aval = float(answer_like_by_sid.get(sid, 0.0))
                if sid in set(connector_ids):
                    useful += 1 if aval >= 0.35 else 0
                else:
                    useful += 1 if aval >= 0.45 else 0
            bridge_purity = float(useful) / float(max(len(bridge_focus_ids), 1))
        bridge_noise_ratio = max(0.0, min(1.0, 1.0 - float(bridge_purity)))
        support_span = len(bridge_focus_ids)
        support_set_compactness = 1.0 / float(1.0 + max(0, int(support_span) - 1))
        answer_candidate_ids = [
            sid for sid in sent_ids if float(answer_like_by_sid.get(sid, 0.0)) >= float(path_answer_threshold)
        ]
        bridge_candidate_ids = [
            sid for sid in bridge_focus_ids if float(answer_like_by_sid.get(sid, 0.0)) >= float(path_pair_threshold)
        ]
        bridge_answer_pair_retention = float(
            _safe_ratio(
                len(set(answer_candidate_ids).intersection(set(bridge_candidate_ids))),
                max(1, len(answer_candidate_ids)),
            )
        )
        anchor_chain_score = max(float(anchor_alignment), (1.0 if main_ids else 0.0))
        bridge_chain_score = max(
            float(bridge_utility),
            float(bridge_purity),
            float(_safe_ratio(len(bridge_focus_ids), max(1, len(sent_ids)))),
        )
        answer_chain_score = max(float(answer_alignment), float(answer_side_density))
        anchor_path_ok = bool(anchor_chain_score >= float(path_anchor_threshold))
        bridge_path_ok = bool(bridge_chain_score >= float(path_bridge_threshold))
        answer_path_ok = bool(answer_chain_score >= float(path_answer_threshold))
        path_complete_score = max(
            0.0,
            min(
                1.0,
                min(float(anchor_chain_score), float(bridge_chain_score), float(answer_chain_score)),
            ),
        )
        path_complete = bool(anchor_path_ok and bridge_path_ok and answer_path_ok and bridge_answer_pair_retention > 0.0)
        incomplete_path_indicator = 1.0 if (answer_path_ok and not path_complete) else 0.0
        role_gap = abs(float(answer_chain_score) - float(bridge_chain_score))
        chain_compactness = float(
            max(
                0.0,
                min(
                    1.0,
                    float(support_set_compactness)
                    * (0.5 + 0.5 * float(bridge_answer_pair_retention))
                    * (1.0 - 0.4 * min(1.0, float(role_gap))),
                ),
            )
        )

        pair_proxy = float(corridor.get("pair_proxy_score", corridor.get("corridor_score", 0.0)) or 0.0)
        base_score = 0.45 * structural + 0.25 * semantic_rel + 0.20 * answer_support + 0.10 * pair_proxy
        chunk_support_w = max(0.0, float(getattr(cfg, "corridor_score_chunk_support_weight", 0.0)))
        answer_align_w = max(0.0, float(getattr(cfg, "corridor_score_answer_alignment_weight", 0.0)))
        if chunk_support_w > 0.0 or answer_align_w > 0.0:
            base_score += chunk_support_w * float(support_density) + answer_align_w * float(answer_alignment)
        shaping_bonus = 0.0
        if compact_enabled and compact_w > 0.0:
            shaping_bonus += float(compact_w) * float(support_set_compactness)
        path_gate_reasons = []
        path_preserve_applied = False
        path_guard_passed = True
        if path_preserve_enabled and path_preserve_w > 0.0:
            if path_preserve_guarded_enabled:
                path_guard_passed = True
                if path_complete_score < float(path_guard_min_complete):
                    path_guard_passed = False
                    path_gate_reasons.append("guard_min_complete_not_met")
                if bridge_answer_pair_retention < float(path_guard_min_pair):
                    path_guard_passed = False
                    path_gate_reasons.append("guard_min_pair_not_met")
                if answer_side_density < float(path_guard_min_answer_density):
                    path_guard_passed = False
                    path_gate_reasons.append("guard_min_answer_density_not_met")
            if path_complete:
                if path_guard_passed:
                    path_bonus = float(path_preserve_w) * float(path_complete_score) * (
                        0.6 + 0.4 * float(bridge_answer_pair_retention)
                    )
                    if path_preserve_guarded_enabled:
                        path_bonus *= float(path_guard_scale)
                        path_gate_reasons.append("path_guard_scale_applied")
                    shaping_bonus += float(path_bonus)
                    path_preserve_applied = True
                    path_gate_reasons.append("path_complete_preserve")
                else:
                    path_gate_reasons.append("path_complete_preserve_suppressed")
            else:
                path_penalty = (
                    float(path_preserve_incomplete_penalty_w)
                    * float(incomplete_path_indicator)
                    * float(max(0.0, min(1.0, answer_chain_score)))
                )
                if path_penalty > 0.0:
                    shaping_bonus -= float(path_penalty)
                    path_gate_reasons.append("incomplete_path_penalty")
            if path_preserve_compact_enabled and path_preserve_compact_w > 0.0:
                compact_adjust = float(path_preserve_compact_w) * (float(chain_compactness) - 0.5)
                if path_preserve_compact_lite_enabled:
                    compact_adjust *= float(path_compact_lite_scale)
                    path_gate_reasons.append("path_compact_lite_scale")
                shaping_bonus += float(compact_adjust)
                branch_overflow = max(0, int(support_span) - 3)
                if branch_overflow > 0:
                    overflow_penalty = min(0.20, 0.03 * float(branch_overflow))
                    if path_preserve_compact_lite_enabled:
                        overflow_penalty = min(
                            float(path_compact_lite_overflow_cap),
                            float(path_compact_lite_overflow_coeff) * float(branch_overflow),
                        )
                    shaping_bonus -= float(overflow_penalty)
                path_gate_reasons.append("path_compact_adjust")
        preserve_gate_reasons = []
        preserve_scale = 1.0
        preserve_applied = False
        preserve_candidate_count = int(len(answer_side_ids))
        preserve_confident_count = int(sum(1 for v in answer_side_vals if float(v) >= answer_preserve_confidence_threshold))
        preserve_bridge_consistent_count = int(
            sum(
                1
                for sid in bridge_focus_ids
                if float(answer_like_by_sid.get(sid, 0.0)) >= float(answer_preserve_bridge_consistency_threshold)
            )
        )
        preserve_bridge_consistent_ratio = float(
            _safe_ratio(preserve_bridge_consistent_count, max(1, len(bridge_focus_ids)))
        )
        preserve_usefulness = float(_safe_ratio(preserve_confident_count, max(1, preserve_candidate_count)))
        if answer_preserve_enabled and answer_preserve_w > 0.0:
            preserve_gate_ok = True
            if answer_preserve_confidence_gated_enabled:
                if preserve_bridge_consistent_ratio < float(answer_preserve_bridge_consistency_threshold):
                    preserve_gate_ok = False
                    preserve_gate_reasons.append("confidence_gate_bridge_not_met")
                if answer_side_density < float(answer_preserve_confidence_threshold):
                    preserve_gate_ok = False
                    preserve_gate_reasons.append("confidence_gate_answer_not_met")
                if preserve_gate_ok:
                    preserve_scale *= float(answer_preserve_light_boost_scale)
                    preserve_gate_reasons.append("confidence_gate_light_boost")
            if answer_preserve_guarded_hotpot_enabled and dataset_name.startswith("hotpot"):
                anchor_bridge_strength = 0.5 * (float(anchor_alignment) + float(bridge_utility))
                if anchor_bridge_strength >= float(hotpot_anchor_bridge_sufficient):
                    preserve_scale *= float(hotpot_relax_factor)
                    preserve_gate_reasons.append("guarded_hotpot_anchor_bridge_sufficient")
                if answer_side_density >= float(hotpot_answer_density_cap):
                    preserve_gate_ok = False
                    preserve_gate_reasons.append("guarded_hotpot_density_cap")
                if support_density >= float(hotpot_quota_cap):
                    preserve_scale *= float(hotpot_relax_factor)
                    preserve_gate_reasons.append("guarded_hotpot_quota_cap")
            density_gap = max(0.0, float(answer_density_min) - float(answer_side_density))
            preserve_bonus = float(answer_preserve_w) * float(preserve_scale) * float(answer_side_density)
            preserve_bonus -= 0.5 * float(answer_preserve_w) * float(density_gap)
            if preserve_gate_ok and preserve_bonus > 0.0:
                shaping_bonus += float(preserve_bonus)
                preserve_applied = True
                preserve_gate_reasons.append("preserve_applied")
            else:
                preserve_gate_reasons.append("preserve_suppressed")
        if bridge_purity_enabled and bridge_purity_w > 0.0:
            shaping_bonus += float(bridge_purity_w) * float(bridge_purity)
            shaping_bonus -= 0.5 * float(bridge_purity_w) * float(bridge_noise_ratio)
        base_score += float(shaping_bonus)
        base_rows.append(
            {
                "corridor": dict(corridor),
                "base_score": float(base_score),
                "sentence_set": set(sent_ids),
                "roles": {
                    "anchor": max(0.0, min(1.0, float(anchor_alignment))),
                    "bridge": max(0.0, min(1.0, float(bridge_utility))),
                    "answer": max(0.0, min(1.0, float(answer_alignment))),
                },
                "components": {
                    "structural_connectivity": float(structural),
                    "semantic_relevance": float(semantic_rel),
                    "answer_support": float(answer_support),
                    "support_density": float(support_density),
                    "pair_proxy_score": float(pair_proxy),
                    "anchor_alignment": float(anchor_alignment),
                    "answer_alignment": float(answer_alignment),
                    "bridge_utility": float(bridge_utility),
                    "bridge_purity": float(bridge_purity),
                    "answer_side_density": float(answer_side_density),
                    "support_set_compactness": float(support_set_compactness),
                    "bridge_noise_ratio": float(bridge_noise_ratio),
                    "path_complete_score": float(path_complete_score),
                    "path_complete": bool(path_complete),
                    "bridge_answer_pair_retention": float(bridge_answer_pair_retention),
                    "incomplete_path_indicator": float(incomplete_path_indicator),
                    "chain_compactness": float(chain_compactness),
                    "path_preserve_applied": bool(path_preserve_applied),
                    "path_preserve_guard_passed": bool(path_guard_passed),
                    "path_preserve_gate_reasons": list(path_gate_reasons),
                    "corridor_compact_shaping_enabled": bool(compact_enabled),
                    "corridor_answer_preserve_enabled": bool(answer_preserve_enabled),
                    "corridor_bridge_purity_shaping_enabled": bool(bridge_purity_enabled),
                    "corridor_path_preserve_enabled": bool(path_preserve_enabled),
                    "corridor_path_preserve_compact_enabled": bool(path_preserve_compact_enabled),
                    "corridor_path_preserve_guarded_enabled": bool(path_preserve_guarded_enabled),
                    "corridor_path_preserve_compact_lite_enabled": bool(path_preserve_compact_lite_enabled),
                    "corridor_answer_preserve_guarded_hotpot_enabled": bool(answer_preserve_guarded_hotpot_enabled),
                    "corridor_answer_preserve_confidence_gated_enabled": bool(answer_preserve_confidence_gated_enabled),
                    "corridor_compact_weight": float(compact_w),
                    "corridor_answer_preserve_weight": float(answer_preserve_w),
                    "corridor_bridge_purity_weight": float(bridge_purity_w),
                    "corridor_path_preserve_weight": float(path_preserve_w),
                    "corridor_path_preserve_incomplete_penalty_weight": float(path_preserve_incomplete_penalty_w),
                    "corridor_path_preserve_compact_weight": float(path_preserve_compact_w),
                    "corridor_path_preserve_anchor_threshold": float(path_anchor_threshold),
                    "corridor_path_preserve_bridge_threshold": float(path_bridge_threshold),
                    "corridor_path_preserve_answer_threshold": float(path_answer_threshold),
                    "corridor_path_preserve_pair_threshold": float(path_pair_threshold),
                    "corridor_path_preserve_guard_scale": float(path_guard_scale),
                    "corridor_path_preserve_guard_min_complete_score": float(path_guard_min_complete),
                    "corridor_path_preserve_guard_min_pair_retention": float(path_guard_min_pair),
                    "corridor_path_preserve_guard_min_answer_density": float(path_guard_min_answer_density),
                    "corridor_path_preserve_compact_lite_scale": float(path_compact_lite_scale),
                    "corridor_path_preserve_compact_lite_overflow_coeff": float(path_compact_lite_overflow_coeff),
                    "corridor_path_preserve_compact_lite_overflow_cap": float(path_compact_lite_overflow_cap),
                    "corridor_answer_preserve_min_density": float(answer_density_min),
                    "corridor_answer_preserve_light_boost_scale": float(answer_preserve_light_boost_scale),
                    "corridor_answer_preserve_bridge_consistency_threshold": float(
                        answer_preserve_bridge_consistency_threshold
                    ),
                    "corridor_answer_preserve_confidence_threshold": float(answer_preserve_confidence_threshold),
                    "answer_preserve_candidate_count": int(preserve_candidate_count),
                    "answer_preserve_confident_count": int(preserve_confident_count),
                    "answer_preserve_bridge_consistent_count": int(preserve_bridge_consistent_count),
                    "answer_preserve_bridge_consistent_ratio": float(preserve_bridge_consistent_ratio),
                    "answer_preserve_usefulness": float(preserve_usefulness),
                    "answer_preserve_scale": float(preserve_scale),
                    "answer_preserve_applied": bool(preserve_applied),
                    "answer_preserve_gate_reasons": list(preserve_gate_reasons),
                    "corridor_shaping_bonus": float(shaping_bonus),
                    "chunk_support_bonus_weight": float(chunk_support_w),
                    "answer_alignment_bonus_weight": float(answer_align_w),
                },
            }
        )

    base_rows.sort(key=lambda x: x["base_score"], reverse=True)

    if coverage_enabled:
        pool = list(base_rows)
        selected_sets = []
        coverage_state = {"anchor": 0.0, "bridge": 0.0, "answer": 0.0}
        reranked = []
        while pool:
            best_idx = 0
            best_score = float("-inf")
            best_overlap = 0.0
            best_gain = {"anchor": 0.0, "bridge": 0.0, "answer": 0.0}
            for idx, item in enumerate(pool):
                overlap_penalty = 0.0
                for prev in selected_sets:
                    inter = len(item["sentence_set"].intersection(prev))
                    union = len(item["sentence_set"].union(prev))
                    if union > 0:
                        overlap_penalty = max(overlap_penalty, float(inter) / float(union))
                role = dict(item.get("roles", {}) or {})
                gain_anchor = max(0.0, float(role.get("anchor", 0.0)) - float(coverage_state["anchor"]))
                gain_bridge = max(0.0, float(role.get("bridge", 0.0)) - float(coverage_state["bridge"]))
                gain_answer = max(0.0, float(role.get("answer", 0.0)) - float(coverage_state["answer"]))
                coverage_gain = (gain_anchor + gain_bridge + gain_answer) / 3.0
                score = (
                    float(item["base_score"])
                    + float(coverage_role_w) * float(coverage_gain)
                    - (float(redundancy_w) + float(coverage_red_w)) * float(overlap_penalty)
                )
                if score > best_score:
                    best_idx = int(idx)
                    best_score = float(score)
                    best_overlap = float(overlap_penalty)
                    best_gain = {
                        "anchor": float(gain_anchor),
                        "bridge": float(gain_bridge),
                        "answer": float(gain_answer),
                    }

            picked = pool.pop(best_idx)
            before_state = dict(coverage_state)
            role = dict(picked.get("roles", {}) or {})
            coverage_state["anchor"] = max(float(coverage_state["anchor"]), float(role.get("anchor", 0.0)))
            coverage_state["bridge"] = max(float(coverage_state["bridge"]), float(role.get("bridge", 0.0)))
            coverage_state["answer"] = max(float(coverage_state["answer"]), float(role.get("answer", 0.0)))

            payload = dict(picked["corridor"])
            payload["corridor_score"] = float(best_score)
            payload["final_score_components"] = {
                "base_score": float(picked["base_score"]),
                "base_final_score": float(best_score),
                "redundancy_penalty": float(best_overlap),
                "coverage_selection_enabled": True,
                "coverage_gain_anchor": float(best_gain["anchor"]),
                "coverage_gain_bridge": float(best_gain["bridge"]),
                "coverage_gain_answer": float(best_gain["answer"]),
                "coverage_state_before": before_state,
                "coverage_state_after": dict(coverage_state),
                "coverage_role_weight": float(coverage_role_w),
                "coverage_redundancy_weight": float(coverage_red_w),
                **(picked.get("components", {}) or {}),
            }
            reranked.append(payload)
            selected_sets.append(set(picked["sentence_set"]))

        return reranked

    selected_sets = []
    reranked = []
    for item in base_rows:
        overlap_penalty = 0.0
        for prev in selected_sets:
            inter = len(item["sentence_set"].intersection(prev))
            union = len(item["sentence_set"].union(prev))
            if union > 0:
                overlap_penalty = max(overlap_penalty, float(inter) / float(union))
        final_score = float(item["base_score"]) - float(redundancy_w) * float(overlap_penalty)
        payload = dict(item["corridor"])
        payload["corridor_score"] = float(final_score)
        payload["final_score_components"] = {
            "base_score": float(item["base_score"]),
            "base_final_score": float(final_score),
            "redundancy_penalty": float(overlap_penalty),
            "coverage_selection_enabled": False,
            **(item.get("components", {}) or {}),
        }
        reranked.append(payload)
        selected_sets.append(set(item["sentence_set"]))

    reranked.sort(key=lambda x: float(x.get("corridor_score", 0.0)), reverse=True)
    return reranked


def _normalize_small_vector(values):
    arr = np.asarray(list(values or []), dtype=np.float32)
    if arr.size <= 0:
        return []
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if (vmax - vmin) <= 1.0e-12:
        if vmax <= 0.0:
            return [0.0 for _ in arr]
        return [1.0 for _ in arr]
    out = (arr - vmin) / (vmax - vmin)
    return [float(x) for x in out.tolist()]


def _apply_lightweight_run_rerank(ranked_runs, cfg):
    topk = max(0, int(getattr(cfg, "run_light_rerank_topk", 0)))
    enabled_flag = bool(getattr(cfg, "run_light_rerank_enabled", False))
    diag = {
        "enabled": bool(enabled_flag and topk > 1),
        "applied": False,
        "enabled_flag": bool(enabled_flag),
        "topk": int(topk),
        "head_size": 0,
        "reordered": 0,
    }
    if (not enabled_flag) or topk <= 1 or (not ranked_runs) or len(ranked_runs) <= 1:
        return ranked_runs, diag

    head_size = min(len(ranked_runs), topk)
    if head_size <= 1:
        diag["head_size"] = int(head_size)
        return ranked_runs, diag

    head = list(ranked_runs[:head_size])
    tail = list(ranked_runs[head_size:])
    before = [int((item or {}).get("run_id", idx)) for idx, item in enumerate(head)]

    base_vals = []
    bridge_vals = []
    anchor_cov_vals = []
    grounding_vals = []
    for item in head:
        comp = ((item or {}).get("run_score_components", {}) or {})
        cheap = (comp.get("cheap_pre_components", {}) or {}) if isinstance(comp, dict) else {}
        base_vals.append(float((item or {}).get("hybrid_run_score", 0.0) or 0.0))
        bridge_vals.append(
            float(
                comp.get(
                    "bridge_path_completeness",
                    comp.get("bridge_utility", cheap.get("bridge_proxy", 0.0)),
                )
                or 0.0
            )
        )
        anchor_cov_vals.append(float(comp.get("pair_coverage_score", cheap.get("anchor_coverage", 0.0)) or 0.0))
        grounding_vals.append(float(comp.get("entity_chunk_grounding_score", 0.0) or 0.0))

    base_norm = _normalize_small_vector(base_vals)
    bridge_norm = _normalize_small_vector(bridge_vals)
    anchor_cov_norm = _normalize_small_vector(anchor_cov_vals)
    grounding_norm = _normalize_small_vector(grounding_vals)

    w_base = float(getattr(cfg, "run_light_rerank_weight_base", 0.72))
    w_bridge = float(getattr(cfg, "run_light_rerank_weight_bridge_completeness", 0.12))
    w_anchor_cov = float(getattr(cfg, "run_light_rerank_weight_anchor_coverage", 0.08))
    w_ground = float(getattr(cfg, "run_light_rerank_weight_grounding", 0.08))

    diag["head_size"] = int(head_size)
    diag["weights"] = {
        "base": float(w_base),
        "bridge_completeness": float(w_bridge),
        "anchor_coverage": float(w_anchor_cov),
        "grounding": float(w_ground),
    }

    for idx, item in enumerate(head):
        light_score = (
            w_base * float(base_norm[idx])
            + w_bridge * float(bridge_norm[idx])
            + w_anchor_cov * float(anchor_cov_norm[idx])
            + w_ground * float(grounding_norm[idx])
        )
        item["light_rerank_score"] = float(light_score)
        item["light_rerank_components"] = {
            "base_norm": float(base_norm[idx]),
            "bridge_completeness_norm": float(bridge_norm[idx]),
            "anchor_coverage_norm": float(anchor_cov_norm[idx]),
            "grounding_norm": float(grounding_norm[idx]),
        }

    head.sort(
        key=lambda x: (
            float((x or {}).get("light_rerank_score", 0.0) or 0.0),
            float((x or {}).get("hybrid_run_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    after = [int((item or {}).get("run_id", idx)) for idx, item in enumerate(head)]
    diag["reordered"] = int(sum(1 for i in range(head_size) if before[i] != after[i]))
    diag["applied"] = bool(diag["reordered"] > 0)
    return head + tail, diag


def _apply_lightweight_corridor_top1_correction(reranked_corridors, cfg):
    enabled = bool(getattr(cfg, "top1_correction_enabled", False))
    topk = max(1, int(getattr(cfg, "top1_correction_topk", 3)))
    diag = {
        "enabled": bool(enabled),
        "applied": False,
        "head_size": 0,
    }
    if (not enabled) or (not reranked_corridors) or len(reranked_corridors) <= 1:
        return reranked_corridors, diag

    head_size = min(len(reranked_corridors), topk)
    if head_size <= 1:
        diag["head_size"] = int(head_size)
        return reranked_corridors, diag

    head = [dict(item) for item in reranked_corridors[:head_size]]
    tail = list(reranked_corridors[head_size:])
    diag["head_size"] = int(head_size)

    base_vals = []
    anchor_vals = []
    support_vals = []
    bridge_vals = []
    semantic_vals = []
    redundancy_vals = []
    for payload in head:
        comp = (payload.get("final_score_components", {}) or {})
        base_vals.append(float(payload.get("corridor_score", comp.get("base_final_score", 0.0)) or 0.0))
        anchor_vals.append(float(comp.get("anchor_alignment", payload.get("anchor_alignment", 0.0)) or 0.0))
        support_vals.append(float(comp.get("support_density", 0.0) or 0.0))
        bridge_vals.append(float(comp.get("bridge_utility", payload.get("bridge_potential", 0.0)) or 0.0))
        semantic_vals.append(float(comp.get("semantic_relevance", 0.0) or 0.0))
        redundancy_vals.append(float(comp.get("redundancy_penalty", 0.0) or 0.0))

    base_norm = _normalize_small_vector(base_vals)
    anchor_norm = _normalize_small_vector(anchor_vals)
    support_norm = _normalize_small_vector(support_vals)
    bridge_norm = _normalize_small_vector(bridge_vals)
    semantic_norm = _normalize_small_vector(semantic_vals)
    redundancy_norm = _normalize_small_vector(redundancy_vals)

    w_base = float(getattr(cfg, "top1_correction_corridor_weight_base", 0.75))
    w_anchor = float(getattr(cfg, "top1_correction_corridor_weight_anchor", 0.10))
    w_support = float(getattr(cfg, "top1_correction_corridor_weight_support", 0.06))
    w_bridge = float(getattr(cfg, "top1_correction_corridor_weight_bridge", 0.05))
    w_sem = float(getattr(cfg, "top1_correction_corridor_weight_semantic", 0.04))
    w_red = max(0.0, float(getattr(cfg, "top1_correction_corridor_weight_redundancy", 0.05)))

    for idx, payload in enumerate(head):
        corrected = (
            w_base * float(base_norm[idx])
            + w_anchor * float(anchor_norm[idx])
            + w_support * float(support_norm[idx])
            + w_bridge * float(bridge_norm[idx])
            + w_sem * float(semantic_norm[idx])
            - w_red * float(redundancy_norm[idx])
        )
        payload["corridor_score"] = float(corrected)
        comp = dict(payload.get("final_score_components", {}) or {})
        comp["top1_corr_corrected_score"] = float(corrected)
        payload["final_score_components"] = comp

    head.sort(key=lambda x: float(x.get("corridor_score", 0.0)), reverse=True)
    diag["applied"] = True
    return head + tail, diag


def _apply_lightweight_sentence_top1_correction(
    selected_sentence_ids,
    selected_sentences,
    sentence_feature_table,
    cfg,
):
    enabled = bool(getattr(cfg, "top1_correction_enabled", False))
    topk = max(1, int(getattr(cfg, "top1_correction_topk", 3)))
    diag = {
        "enabled": bool(enabled),
        "applied": False,
        "head_size": 0,
        "reordered": 0,
    }
    if (not enabled) or (not selected_sentence_ids) or len(selected_sentence_ids) <= 1:
        return selected_sentence_ids, selected_sentences, diag

    head_size = min(len(selected_sentence_ids), topk)
    if head_size <= 1:
        diag["head_size"] = int(head_size)
        return selected_sentence_ids, selected_sentences, diag

    head_ids = list(selected_sentence_ids[:head_size])
    tail_ids = list(selected_sentence_ids[head_size:])
    order_map = {sid: idx for idx, sid in enumerate(head_ids)}
    diag["head_size"] = int(head_size)

    base_vals = []
    corridor_vals = []
    query_vals = []
    locality_vals = []
    main_vals = []
    support_vals = []
    for sid in head_ids:
        feat = (sentence_feature_table.get(sid, {}) or {})
        base_vals.append(float(feat.get("base_retrieval_score", 0.0) or 0.0))
        corridor_vals.append(float(feat.get("best_corridor_score", 0.0) or 0.0))
        query_vals.append(float(feat.get("query_overlap_score", 0.0) or 0.0))
        locality_vals.append(float(feat.get("locality_score", 0.0) or 0.0))
        main_vals.append(1.0 if bool(feat.get("is_main_candidate", False)) else 0.0)
        support_vals.append(1.0 if bool(feat.get("is_support_candidate", False)) else 0.0)

    base_norm = _normalize_small_vector(base_vals)
    corridor_norm = _normalize_small_vector(corridor_vals)
    query_norm = _normalize_small_vector(query_vals)
    locality_norm = _normalize_small_vector(locality_vals)

    w_base = float(getattr(cfg, "top1_correction_sentence_weight_base", 0.60))
    w_corr = float(getattr(cfg, "top1_correction_sentence_weight_corridor", 0.20))
    w_main = float(getattr(cfg, "top1_correction_sentence_weight_main", 0.08))
    w_support = float(getattr(cfg, "top1_correction_sentence_weight_support", 0.04))
    w_query = float(getattr(cfg, "top1_correction_sentence_weight_query", 0.04))
    w_locality = float(getattr(cfg, "top1_correction_sentence_weight_locality", 0.04))
    w_redundancy = max(0.0, float(getattr(cfg, "top1_correction_sentence_weight_redundancy", 0.04)))

    pre_score_map = {}
    for idx, sid in enumerate(head_ids):
        pre_score_map[sid] = (
            w_base * float(base_norm[idx])
            + w_corr * float(corridor_norm[idx])
            + w_main * float(main_vals[idx])
            + w_support * float(support_vals[idx])
            + w_query * float(query_norm[idx])
            + w_locality * float(locality_norm[idx])
        )

    text_map = {sid: text for sid, text in zip(selected_sentence_ids, selected_sentences)}
    token_map = {sid: set(content_tokens(text_map.get(sid, ""))) for sid in head_ids}

    reordered_head = []
    remaining = list(head_ids)
    while remaining:
        best_sid = None
        best_score = None
        for sid in remaining:
            redundancy = 0.0
            sid_tokens = token_map.get(sid, set())
            if reordered_head:
                for prev_sid in reordered_head:
                    prev_tokens = token_map.get(prev_sid, set())
                    union = len(sid_tokens.union(prev_tokens))
                    if union <= 0:
                        continue
                    inter = len(sid_tokens.intersection(prev_tokens))
                    redundancy = max(redundancy, float(inter) / float(union))
            score = float(pre_score_map.get(sid, 0.0)) - w_redundancy * float(redundancy)
            if (best_score is None) or (score > best_score) or (
                abs(score - best_score) <= 1.0e-12
                and order_map.get(sid, 10**9) < order_map.get(best_sid, 10**9)
            ):
                best_sid = sid
                best_score = score
        reordered_head.append(best_sid)
        remaining.remove(best_sid)

    reordered_ids = reordered_head + tail_ids
    if reordered_ids == list(selected_sentence_ids):
        return selected_sentence_ids, selected_sentences, diag

    reordered_texts = [text_map.get(sid, "") for sid in reordered_ids]
    diag["applied"] = True
    diag["reordered"] = int(sum(1 for i, sid in enumerate(head_ids) if reordered_head[i] != sid))
    return reordered_ids, reordered_texts, diag


def _is_chunk_like_node(g, node):
    if node not in g:
        return False
    ntype = str((g.nodes[node] or {}).get("node_type", "") or "").strip().lower()
    return ntype in {"sentence", "chunk", "passage", "document"}


def _resolve_entity_chunk_support_maps(g, semantic_state):
    ent2chk = {}
    chk2ent = {}
    if isinstance(semantic_state, dict):
        raw_e2c = semantic_state.get("entity_to_chunks", {})
        raw_c2e = semantic_state.get("chunk_to_entities", {})
        if isinstance(raw_e2c, dict) and isinstance(raw_c2e, dict):
            for ent, chunks in raw_e2c.items():
                ent_id = str(ent)
                vals = [str(c) for c in (chunks or []) if str(c)]
                if vals:
                    ent2chk[ent_id] = vals
            for chk, ents in raw_c2e.items():
                chk_id = str(chk)
                vals = [str(e) for e in (ents or []) if str(e)]
                if vals:
                    chk2ent[chk_id] = vals

    if ent2chk and chk2ent:
        return ent2chk, chk2ent

    has_explicit_chunk_nodes = any(_is_chunk_like_node(g, node) for node in g.nodes)

    for u, v, data in g.edges(data=True):
        u_type = str((g.nodes[u] or {}).get("node_type", "") or "").strip().lower()
        v_type = str((g.nodes[v] or {}).get("node_type", "") or "").strip().lower()
        support_layer = str((data or {}).get("support_layer", "") or "").strip().lower()

        edge_is_entity_chunk = False
        pair_is_entity_chunk_like = (
            (u_type == "entity" and _is_chunk_like_node(g, v))
            or (v_type == "entity" and _is_chunk_like_node(g, u))
        )
        pair_is_entity_sentence = {u_type, v_type} == {"entity", "sentence"}
        if has_explicit_chunk_nodes:
            edge_is_entity_chunk = pair_is_entity_chunk_like and (support_layer == "entity_chunk" or support_layer == "")
        else:
            edge_is_entity_chunk = (
                pair_is_entity_chunk_like
                or pair_is_entity_sentence
                or (support_layer == "entity_chunk")
            )
        if not edge_is_entity_chunk:
            continue

        if u_type == "entity":
            ent, chk = str(u), str(v)
        else:
            ent, chk = str(v), str(u)
        ent2chk.setdefault(ent, []).append(chk)
        chk2ent.setdefault(chk, []).append(ent)

    for key, vals in list(ent2chk.items()):
        seen = set()
        out = []
        for v in vals:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        ent2chk[key] = out
    for key, vals in list(chk2ent.items()):
        seen = set()
        out = []
        for v in vals:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        chk2ent[key] = out
    return ent2chk, chk2ent


def _prepare_diffusion_graph_for_mode(reduced_graph, anchors, cfg):
    mode = str(getattr(cfg, "graph_mode", "current_entity_graph") or "current_entity_graph").strip().lower()
    if mode != "entity_chunk_graph":
        return reduced_graph, list(anchors), {"graph_mode": mode, "applied": False, "reason": "frozen_default"}

    include_chunks = bool(getattr(cfg, "chunk_node_enabled_in_diffusion", False))
    if include_chunks:
        phase1_anchors = [a for a in anchors if a in reduced_graph]
        return reduced_graph, phase1_anchors, {
            "graph_mode": mode,
            "applied": True,
            "chunk_node_enabled_in_diffusion": True,
            "reason": "mixed_diffusion",
        }

    nodes = [n for n in reduced_graph.nodes if not _is_chunk_like_node(reduced_graph, n)]
    for anchor in anchors:
        if anchor in reduced_graph and anchor not in nodes:
            nodes.append(anchor)
    nodes = list(dict.fromkeys(nodes))
    if not nodes:
        phase1_anchors = [a for a in anchors if a in reduced_graph]
        return reduced_graph, phase1_anchors, {
            "graph_mode": mode,
            "applied": False,
            "chunk_node_enabled_in_diffusion": False,
            "reason": "entity_only_empty_fallback",
        }

    diffusion_graph = reduced_graph.subgraph(nodes).copy()
    phase1_anchors = [a for a in anchors if a in diffusion_graph]
    if not phase1_anchors:
        ents = [n for n in diffusion_graph.nodes if str((diffusion_graph.nodes[n] or {}).get("node_type", "")).lower() == "entity"]
        ents.sort(key=lambda n: diffusion_graph.degree(n), reverse=True)
        phase1_anchors = ents[: max(1, int(getattr(cfg, "max_anchors", 5)))]
    return diffusion_graph, phase1_anchors, {
        "graph_mode": mode,
        "applied": True,
        "chunk_node_enabled_in_diffusion": False,
        "reason": "entity_seed_chunk_lift",
        "diffusion_graph_nodes": int(diffusion_graph.number_of_nodes()),
        "diffusion_graph_edges": int(diffusion_graph.number_of_edges()),
    }


def _entity_chunk_graph_chunk_scoring(
    reduced_graph,
    final_graph,
    chosen_run,
    shortlisted_runs,
    semantic_state,
    query_sim_map,
    support_sim_map,
    anchors,
    cfg,
    role_aware_enabled_override=None,
):
    mode = str(getattr(cfg, "chunk_scoring_mode", "entity_aggregate") or "entity_aggregate").strip().lower()
    topk = max(1, int(getattr(cfg, "chunk_score_topk", 20)))
    transition_w = max(0.0, min(1.0, float(getattr(cfg, "entity_chunk_transition_weight", 0.35))))
    ent2chk, chk2ent = _resolve_entity_chunk_support_maps(reduced_graph, semantic_state)

    seed_score_map = dict((chosen_run or {}).get("seed_score_map", {}) or {})
    agg_scores = dict((chosen_run or {}).get("agg_scores", {}) or {})
    chunk_scores = {}
    entity_scores = {}

    for node, score in seed_score_map.items():
        node_id = str(node)
        if node_id in reduced_graph and str((reduced_graph.nodes[node_id] or {}).get("node_type", "")).lower() == "entity":
            entity_scores[node_id] = max(float(entity_scores.get(node_id, -1.0e9)), float(score))
    for node, score in sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)[: max(topk * 8, 64)]:
        node_id = str(node)
        if node_id not in reduced_graph:
            continue
        ntype = str((reduced_graph.nodes[node_id] or {}).get("node_type", "")).lower()
        if ntype == "entity":
            entity_scores[node_id] = max(float(entity_scores.get(node_id, -1.0e9)), float(score))
        elif _is_chunk_like_node(reduced_graph, node_id):
            chunk_scores[node_id] = max(float(chunk_scores.get(node_id, -1.0e9)), float(score))

    if mode == "entity_aggregate":
        for ent, e_score in entity_scores.items():
            linked = list(ent2chk.get(ent, []) or [])
            for rank, chunk in enumerate(linked[: max(topk * 2, 16)]):
                if chunk not in reduced_graph or (not _is_chunk_like_node(reduced_graph, chunk)):
                    continue
                decay = max(0.20, 1.0 - 0.06 * float(rank))
                lift = float(e_score) * decay
                prev = float(chunk_scores.get(chunk, -1.0e9))
                chunk_scores[chunk] = max(prev, lift if prev < -1.0e8 else (1.0 - transition_w) * prev + transition_w * lift)
    else:
        for chunk, c_score in list(chunk_scores.items()):
            ents = list(chk2ent.get(str(chunk), []) or [])
            bonus = 0.0
            for rank, ent in enumerate(ents[:6]):
                bonus += max(0.0, float(entity_scores.get(str(ent), 0.0))) * max(0.10, 0.45 - 0.05 * float(rank))
            chunk_scores[chunk] = float(c_score) + transition_w * float(bonus)

    # Corridor-local bonus: keep chunks touched by selected nodes/bridge paths.
    if final_graph is not None:
        selected_nodes = set(final_graph.nodes())
        for node in selected_nodes:
            node_id = str(node)
            if node_id in chunk_scores:
                chunk_scores[node_id] = float(chunk_scores[node_id]) + 0.15
            if node_id in ent2chk:
                for rank, chunk in enumerate(list(ent2chk.get(node_id, []) or [])[:4]):
                    if chunk in reduced_graph and _is_chunk_like_node(reduced_graph, chunk):
                        chunk_scores[chunk] = max(float(chunk_scores.get(chunk, -1.0e9)), 0.25 - 0.03 * float(rank))

    role_aware_enabled = (
        bool(role_aware_enabled_override)
        if role_aware_enabled_override is not None
        else bool(getattr(cfg, "role_aware_chunk_scoring_enabled", False))
    )
    role_components = {}
    if role_aware_enabled and chunk_scores:
        tau = max(2, int(getattr(cfg, "tau", 4)))
        a_list = [str(a) for a in list(anchors or []) if str(a) in reduced_graph]
        anchor_maps = {
            a: _get_distance_map(reduced_graph, a, tau, distance_map_cache={})
            for a in a_list
        }
        selected_nodes = set(final_graph.nodes()) if final_graph is not None else set()
        w_anchor = max(0.0, float(getattr(cfg, "role_chunk_weight_anchor_match", 0.30)))
        w_bridge = max(0.0, float(getattr(cfg, "role_chunk_weight_bridge_support", 0.30)))
        w_answer = max(0.0, float(getattr(cfg, "role_chunk_weight_answer_likelihood", 0.25)))
        w_path = max(0.0, float(getattr(cfg, "role_chunk_weight_path_consistency", 0.15)))
        w_sum = w_anchor + w_bridge + w_answer + w_path
        if w_sum <= 0.0:
            w_anchor, w_bridge, w_answer, w_path, w_sum = 0.30, 0.30, 0.25, 0.15, 1.0
        w_anchor /= w_sum
        w_bridge /= w_sum
        w_answer /= w_sum
        w_path /= w_sum

        base_norm = _normalize_map(chunk_scores)
        for chunk in list(chunk_scores.keys()):
            if chunk not in reduced_graph:
                continue
            hits = 0
            for a in a_list:
                dmap = anchor_maps.get(a, {})
                if int(dmap.get(chunk, tau + 1)) <= tau:
                    hits += 1
            anchor_match = float(hits) / float(max(len(a_list), 1))
            bridge_support = 1.0 if hits >= 2 else anchor_match
            qv = float((query_sim_map or {}).get(chunk, 0.0))
            sv = float((support_sim_map or {}).get(chunk, 0.0))
            answer_like = max(0.0, min(1.0, 0.5 * (max(qv, sv) + 1.0)))
            if chunk in selected_nodes:
                path_consistency = 1.0
            else:
                neigh = set(reduced_graph.neighbors(chunk))
                path_consistency = 1.0 if neigh.intersection(selected_nodes) else 0.0
            role_score = (
                w_anchor * float(anchor_match)
                + w_bridge * float(bridge_support)
                + w_answer * float(answer_like)
                + w_path * float(path_consistency)
            )
            base_score = float(base_norm.get(chunk, 0.0))
            final_score = 0.55 * base_score + 0.45 * role_score
            chunk_scores[chunk] = float(final_score)
            role_components[chunk] = {
                "anchor_match": float(anchor_match),
                "bridge_support": float(bridge_support),
                "answer_likelihood": float(answer_like),
                "path_consistency": float(path_consistency),
                "role_score": float(role_score),
                "base_score_norm": float(base_score),
                "final_score": float(final_score),
            }

    ranked = sorted(
        [(str(chunk), float(score)) for chunk, score in chunk_scores.items() if chunk in reduced_graph and _is_chunk_like_node(reduced_graph, chunk)],
        key=lambda x: x[1],
        reverse=True,
    )[:topk]
    diag = {
        "applied": bool(ranked),
        "chunk_scoring_mode": mode,
        "chunk_score_topk": int(topk),
        "entity_chunk_transition_weight": float(transition_w),
        "entity_score_count": int(len(entity_scores)),
        "chunk_score_count": int(len(chunk_scores)),
        "selected_chunk_count": int(len(ranked)),
        "shortlisted_run_count": int(len(shortlisted_runs or [])),
        "role_aware_chunk_scoring_enabled": bool(role_aware_enabled),
        "role_aware_chunk_components": {str(k): dict(v) for k, v in role_components.items()} if role_components else {},
    }
    return ranked, diag


def run_graphrag_core(
    sample,
    cfg,
    method_name,
    stable_seed_selection,
    enable_trim,
):
    start = time.perf_counter()
    stage_ms = {
        # detailed timings
        "query_embed_ms": 0.0,
        "semantic_lookup_entity_ms": 0.0,
        "semantic_lookup_chunk_ms": 0.0,
        "proposal_union_ms": 0.0,
        "proposal_subgraph_build_ms": 0.0,
        "phase1_ppr_ms": 0.0,
        "phase1_run_scoring_ms": 0.0,
        "phase2_pair_shortlist_ms": 0.0,
        "phase2_refine_ms": 0.0,
        "top1_correction_ms": 0.0,
        "sentence_rerank_ms": 0.0,
        "render_ms": 0.0,
        # backward-compatible aliases
        "proposal_time_ms": 0.0,
        "phase1_ppr_time_ms": 0.0,
        "run_scoring_time_ms": 0.0,
        "phase2_refinement_time_ms": 0.0,
        "final_render_time_ms": 0.0,
    }
    shared_budget_diag = _apply_shared_budget_profile_once(cfg)
    objective_flags = _resolve_retrieval_objective_flags(cfg)
    objective_flags, objective_profile_diag = _apply_connector_objective_profile(
        cfg=cfg,
        objective_flags=objective_flags,
    )
    hybrid_anchor_diag = {
        "applied": False,
        "reason": "disabled",
        "added": [],
        "final_anchors": [],
    }
    bridge_induction_diag = {
        "applied": False,
        "reason": "disabled",
        "candidate_count": 0,
        "pair_count": 0,
        "added_count": 0,
        "added_nodes": [],
    }
    candidate_recall_boost_diag = {
        "enabled": False,
        "applied": False,
        "reason": "disabled",
    }
    evidence_density_rerank_diag = {
        "enabled": False,
        "applied": False,
        "reason": "disabled",
    }
    gl_rcedr_diag = {
        "enabled": False,
        "applied": False,
        "reason": "disabled",
    }
    answerability_selection_diag = {
        "enabled": False,
        "applied": False,
        "reason": "disabled",
    }
    graph_scope = "query_context"
    global_index_meta = {}
    g = None

    has_global_corpus = bool(str(getattr(cfg, "global_corpus_path", "") or "").strip())
    has_prebuilt_igraph = bool(str(getattr(cfg, "prebuilt_igraph_path", "") or "").strip())
    if has_global_corpus or has_prebuilt_igraph:
        g, global_index_meta = _get_global_graph(cfg)
        graph_scope = "prebuilt_igraph" if has_prebuilt_igraph else "global_corpus"
    else:
        artifacts = build_document_entity_graph(sample)
        g = artifacts.graph

    anchors = select_lexical_anchors(sample, g, int(getattr(cfg, "max_anchors", 5)), cfg=cfg)
    if not anchors:
        entities = [n for n in g.nodes if g.nodes[n].get("node_type") == "entity"]
        entities.sort(key=lambda n: g.degree(n), reverse=True)
        anchors = entities[: int(getattr(cfg, "max_anchors", 5))]

    rng = random.Random(int(getattr(cfg, "random_seed", 42)))
    show_inner_progress = bool(getattr(cfg, "show_inner_progress", True))
    if int(getattr(cfg, "num_workers", 1)) > 1:
        show_inner_progress = False

    ppr_engine = _resolve_ppr_engine(cfg, graph_scope=graph_scope, graph=g)
    proposal_start = time.perf_counter()

    semantic_diag = {
        "enabled": bool(getattr(cfg, "embedding_enabled", False)),
        "applied": False,
        "error": "",
        "model_name": str(getattr(cfg, "embedding_model_name", "") or ""),
        "semantic_topn_entity": int(
            getattr(cfg, "semantic_topn_entity", getattr(cfg, "semantic_topn", 50))
        ),
        "semantic_topn_chunk": int(
            getattr(
                cfg,
                "semantic_topn_chunk",
                max(1, int(getattr(cfg, "semantic_topn", 50)) // 2),
            )
        ),
        "graph_reserve_topn": int(getattr(cfg, "graph_reserve_topn", 15)),
        "index_available": False,
        "query_embedding_dim": 0,
        "semantic_entity_candidates": [],
        "semantic_chunk_candidates": [],
        "semantic_entity_scores": {},
        "semantic_chunk_scores": {},
        "proposal_by_anchor": {},
        "proposal_candidate_count": 0,
        "candidate_vector_count": 0,
        "retrieval_objective_mode": str(objective_flags.get("mode", "baseline")),
    }

    query_vec = None
    semantic_state = None
    semantic_entity_scores = {}
    semantic_chunk_scores = {}
    semantic_entity_lookup_mode = "disabled"
    semantic_chunk_lookup_mode = "disabled"
    query_embedding_recomputed = False
    query_embedding_cache_hit = False
    candidate_similarity_recomputed_count = 0

    if semantic_diag["enabled"]:
        query_embed_start = time.perf_counter()
        query_vec, qerr, query_embedding_recomputed, query_embedding_cache_hit = _query_embedding(sample.question, cfg)
        stage_ms["query_embed_ms"] = float((time.perf_counter() - query_embed_start) * 1000.0)
        if query_vec is None:
            semantic_diag["error"] = qerr
            semantic_entity_lookup_mode = "error"
            semantic_chunk_lookup_mode = "error"
        else:
            semantic_diag["query_embedding_dim"] = int(query_vec.shape[0])
            if graph_scope in {"global_corpus", "prebuilt_igraph"}:
                semantic_state = _get_global_semantic_state(global_index_meta)
                semantic_diag["index_available"] = semantic_state is not None

            if semantic_state is not None:
                semantic_entity_lookup_mode = "cache"
                semantic_chunk_lookup_mode = "cache"
                semantic_entity_scores, semantic_chunk_scores, semantic_lookup_diag = _semantic_topk_candidates_by_type(
                    query_vec=query_vec,
                    semantic_state=semantic_state,
                    cfg=cfg,
                    g=g,
                    anchors=anchors,
                    query_text=sample.question,
                )
                stage_ms["semantic_lookup_entity_ms"] = float(semantic_lookup_diag.get("entity_ms", 0.0))
                stage_ms["semantic_lookup_chunk_ms"] = float(semantic_lookup_diag.get("chunk_ms", 0.0))
                semantic_entity_lookup_mode = str(semantic_lookup_diag.get("entity_mode", semantic_entity_lookup_mode))
                semantic_chunk_lookup_mode = str(semantic_lookup_diag.get("chunk_mode", semantic_chunk_lookup_mode))
                semantic_diag["entity_lookup_tier1_candidate_count"] = int(
                    semantic_lookup_diag.get("entity_tier1_candidate_count", 0)
                )
                semantic_diag["entity_lookup_tier1_shortlist_count"] = int(
                    semantic_lookup_diag.get("entity_tier1_shortlist_count", 0)
                )
                semantic_diag["entity_lookup_tier1_dense_count"] = int(
                    semantic_lookup_diag.get("entity_tier1_dense_count", 0)
                )
                # Reuse offline support cache to cheaply expand entity proposals into chunk proposals.
                support_cache = semantic_state.get("entity_topk_chunks_cache", {}) if isinstance(semantic_state, dict) else {}
                if isinstance(support_cache, dict):
                    for ent_node, ent_score in list(semantic_entity_scores.items()):
                        linked_chunks = list(support_cache.get(str(ent_node), []) or [])
                        for rank, chunk_node in enumerate(linked_chunks[: int(getattr(cfg, "semantic_topn_chunk", 15))]):
                            bonus = float(ent_score) * (0.85 - 0.03 * float(rank))
                            if chunk_node in g:
                                semantic_chunk_scores[chunk_node] = max(
                                    float(semantic_chunk_scores.get(chunk_node, -1.0)),
                                    float(bonus),
                                )
                semantic_diag["applied"] = True
            elif g.number_of_nodes() <= 5000:
                semantic_entity_lookup_mode = "bruteforce"
                semantic_chunk_lookup_mode = "bruteforce"
                local_lookup_start = time.perf_counter()
                local_nodes = [
                    n
                    for n in g.nodes
                    if (g.nodes[n].get("node_type") == "entity") or _is_chunk_like_node(g, n)
                ]
                local_vectors, local_err = _load_candidate_vectors(local_nodes, g, cfg, semantic_state=None)
                if local_err:
                    semantic_diag["error"] = local_err
                    semantic_entity_lookup_mode = "error"
                    semantic_chunk_lookup_mode = "error"
                else:
                    local_scores = {n: cosine_similarity(query_vec, vec) for n, vec in local_vectors.items()}
                    ent = [(n, s) for n, s in local_scores.items() if g.nodes[n].get("node_type") == "entity"]
                    chk = [(n, s) for n, s in local_scores.items() if _is_chunk_like_node(g, n)]
                    ent.sort(key=lambda x: x[1], reverse=True)
                    chk.sort(key=lambda x: x[1], reverse=True)
                    t_ent = max(0, int(getattr(cfg, "semantic_topn_entity", getattr(cfg, "semantic_topn", 50))))
                    t_chk = max(0, int(getattr(cfg, "semantic_topn_chunk", max(1, t_ent // 2))))
                    semantic_entity_scores = {n: float(s) for n, s in ent[:t_ent]}
                    semantic_chunk_scores = {n: float(s) for n, s in chk[:t_chk]}
                    semantic_diag["applied"] = True
                local_ms = float((time.perf_counter() - local_lookup_start) * 1000.0)
                stage_ms["semantic_lookup_entity_ms"] = float(local_ms * 0.5)
                stage_ms["semantic_lookup_chunk_ms"] = float(local_ms * 0.5)
            else:
                semantic_diag["error"] = "semantic_index_unavailable_for_large_graph"
                semantic_entity_lookup_mode = "unavailable"
                semantic_chunk_lookup_mode = "unavailable"

    if bool(objective_flags.get("hybrid_anchor_recall", False)):
        anchors, hybrid_anchor_diag = _augment_anchors_with_semantic(
            anchors=anchors,
            semantic_entity_scores=semantic_entity_scores,
            sample=sample,
            g=g,
            cfg=cfg,
        )
    else:
        hybrid_anchor_diag["final_anchors"] = [str(a) for a in anchors]

    proposal_union_start = time.perf_counter()
    (
        proposal_by_anchor,
        proposal_nodes,
        proposal_scores,
        semantic_scores,
        proposal_diag,
        proposal_partitions,
    ) = _build_anchor_proposals(
        g=g,
        anchors=anchors,
        semantic_entity_scores=semantic_entity_scores,
        semantic_chunk_scores=semantic_chunk_scores,
        cfg=cfg,
        semantic_state=semantic_state,
    )

    if bool(objective_flags.get("bridge_candidate_induction", False)):
        bridge_nodes, bridge_scores, bridge_induction_diag = _induce_bridge_candidates(
            g=g,
            anchors=anchors,
            semantic_scores=semantic_scores,
            cfg=cfg,
        )
        added_nodes = []
        if bridge_nodes:
            local_cap = max(8, int(getattr(cfg, "proposal_anchor_local_topn", 48)))
            bridge_ordered = [str(n) for n in sorted(list(bridge_nodes), key=lambda x: float(bridge_scores.get(x, 0.0)), reverse=True)]
            for anchor in list(proposal_by_anchor.keys()):
                merged = _ordered_unique(list(proposal_by_anchor.get(anchor, []) or []) + bridge_ordered)
                proposal_by_anchor[anchor] = merged[:local_cap]
            for node in bridge_ordered:
                if node not in g:
                    continue
                added_nodes.append(str(node))
                proposal_nodes.add(str(node))
                bscore = float(bridge_scores.get(str(node), 0.0))
                proposal_scores[str(node)] = max(float(proposal_scores.get(str(node), -1.0e9)), float(bscore))
                semantic_scores[str(node)] = max(float(semantic_scores.get(str(node), -1.0e9)), float(bscore))
            bridge_induction_diag["added_nodes"] = list(added_nodes)
            bridge_induction_diag["added_count"] = int(len(added_nodes))
            proposal_diag["bridge_candidate_count"] = int(len(added_nodes))
            proposal_diag["union_candidate_count"] = int(len(proposal_nodes))
        else:
            bridge_induction_diag["added_nodes"] = []
            bridge_induction_diag["added_count"] = 0
            proposal_diag["bridge_candidate_count"] = 0
    else:
        proposal_diag["bridge_candidate_count"] = 0

    (
        proposal_by_anchor,
        proposal_nodes,
        proposal_scores,
        semantic_scores,
        candidate_recall_boost_diag,
    ) = apply_candidate_recall_boost(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_nodes=proposal_nodes,
        proposal_scores=proposal_scores,
        semantic_scores=semantic_scores,
        cfg=cfg,
    )
    proposal_diag["candidate_recall_boost_count"] = int(
        (candidate_recall_boost_diag or {}).get("selected_boost_count", 0)
    )
    proposal_diag["union_candidate_count"] = int(len(proposal_nodes))

    stage_ms["proposal_union_ms"] = float((time.perf_counter() - proposal_union_start) * 1000.0)
    proposal_subgraph_start = time.perf_counter()
    reduced_graph, reduced_diag = _build_reduced_subgraph_from_proposals(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_scores=proposal_scores,
        cfg=cfg,
    )
    stage_ms["proposal_subgraph_build_ms"] = float((time.perf_counter() - proposal_subgraph_start) * 1000.0)
    if reduced_graph is None or reduced_graph.number_of_nodes() <= 0:
        reduced_graph = g
        reduced_diag = {"applied": False, "reason": "fallback_to_full_graph"}

    anchors = [a for a in anchors if a in reduced_graph]
    if not anchors:
        entities = [n for n in reduced_graph.nodes if reduced_graph.nodes[n].get("node_type") == "entity"]
        entities.sort(key=lambda n: reduced_graph.degree(n), reverse=True)
        anchors = entities[: int(getattr(cfg, "max_anchors", 5))]
    diffusion_graph, phase1_anchors, diffusion_diag = _prepare_diffusion_graph_for_mode(
        reduced_graph=reduced_graph,
        anchors=anchors,
        cfg=cfg,
    )
    if not phase1_anchors:
        phase1_anchors = list(anchors)

    semantic_diag["semantic_entity_candidates"] = [str(n) for n in semantic_entity_scores.keys()]
    semantic_diag["semantic_chunk_candidates"] = [str(n) for n in semantic_chunk_scores.keys()]
    semantic_diag["semantic_entity_scores"] = {str(k): float(v) for k, v in semantic_entity_scores.items()}
    semantic_diag["semantic_chunk_scores"] = {str(k): float(v) for k, v in semantic_chunk_scores.items()}
    semantic_diag["proposal_by_anchor"] = {
        str(anchor): [str(n) for n in list(nodes or [])]
        for anchor, nodes in proposal_by_anchor.items()
    }
    semantic_diag["proposal_candidate_count"] = int(len(proposal_nodes))
    semantic_diag["proposal_entity_count"] = int(proposal_diag.get("proposal_entity_count", 0))
    semantic_diag["proposal_chunk_count"] = int(proposal_diag.get("proposal_chunk_count", 0))
    semantic_diag["graph_reserve_count"] = int(proposal_diag.get("graph_reserve_count", 0))
    semantic_diag["proposal_high_confidence_count"] = int(proposal_diag.get("high_confidence_count", 0))
    semantic_diag["proposal_global_fallback_count"] = int(proposal_diag.get("global_fallback_count", 0))
    semantic_diag["proposal_bridge_candidate_count"] = int(proposal_diag.get("bridge_candidate_count", 0))
    semantic_diag["proposal_candidate_recall_boost_count"] = int(proposal_diag.get("candidate_recall_boost_count", 0))
    semantic_diag["union_candidate_count"] = int(proposal_diag.get("union_candidate_count", len(proposal_nodes)))
    semantic_diag["hybrid_anchor_recall"] = dict(hybrid_anchor_diag or {})
    semantic_diag["bridge_candidate_induction"] = dict(bridge_induction_diag or {})
    semantic_diag["candidate_recall_boost"] = dict(candidate_recall_boost_diag or {})
    semantic_diag["query_embedding_recomputed"] = bool(query_embedding_recomputed)
    semantic_diag["query_embedding_cache_hit"] = bool(query_embedding_cache_hit)
    semantic_diag["semantic_entity_lookup_mode"] = str(semantic_entity_lookup_mode)
    semantic_diag["semantic_chunk_lookup_mode"] = str(semantic_chunk_lookup_mode)
    stage_ms["proposal_time_ms"] = float((time.perf_counter() - proposal_start) * 1000.0)

    phase1_start = time.perf_counter()
    per_anchor_runs = {a: [] for a in phase1_anchors}
    n_runs = max(1, int(getattr(cfg, "samples_per_anchor", 3)))
    phase1_parallel_enabled = bool(getattr(cfg, "phase1_parallel_ppr", True))
    original_ppr_parallel_workers = int(getattr(cfg, "ppr_parallel_workers", 1))
    if not phase1_parallel_enabled:
        setattr(cfg, "ppr_parallel_workers", 1)
    try:
        for _ in tqdm(
            range(n_runs),
            total=n_runs,
            desc="Phase1 stochastic PPR",
            leave=False,
            disable=not show_inner_progress,
        ):
            run_seed = int(rng.random() * 10**9)
            if ppr_engine == "power":
                h = _stochastic_perturb_graph(diffusion_graph, rng, float(getattr(cfg, "edge_drop_prob", 0.1)))
                run_map = _compute_ppr_batch(
                    g=h,
                    sources=phase1_anchors,
                    cfg=cfg,
                    engine="power",
                    run_seed=run_seed,
                    show_progress=show_inner_progress,
                    desc="Phase1 Anchor PPR",
                )
            else:
                run_map = _compute_ppr_batch(
                    g=diffusion_graph,
                    sources=phase1_anchors,
                    cfg=cfg,
                    engine="mc",
                    run_seed=run_seed,
                    show_progress=show_inner_progress,
                    desc="Phase1 Anchor PPR",
                )
            for anchor in phase1_anchors:
                per_anchor_runs[anchor].append(run_map.get(anchor, {}))
    finally:
        if not phase1_parallel_enabled:
            setattr(cfg, "ppr_parallel_workers", original_ppr_parallel_workers)
    stage_ms["phase1_ppr_ms"] = float((time.perf_counter() - phase1_start) * 1000.0)
    stage_ms["phase1_ppr_time_ms"] = float(stage_ms["phase1_ppr_ms"])

    run_score_start = time.perf_counter()
    run_base = []
    for run_id in range(n_runs):
        agg_scores = {}
        anchor_scores = {}
        for anchor in phase1_anchors:
            scores = per_anchor_runs.get(anchor, [{}])[run_id]
            anchor_scores[anchor] = scores
            for node, score in scores.items():
                agg_scores[node] = agg_scores.get(node, 0.0) + float(score)
        graph_candidates = [n for n in _topk_nodes(agg_scores, int(getattr(cfg, "candidate_top_t", 20))) if n in reduced_graph]
        run_base.append(
            {
                "run_id": run_id,
                "anchor_scores": anchor_scores,
                "agg_scores": agg_scores,
                "graph_candidates": graph_candidates,
            }
        )

    candidate_universe = set(phase1_anchors).union(set(proposal_nodes))
    for run in run_base:
        candidate_universe.update(run.get("graph_candidates", []))
    candidate_universe = {node for node in candidate_universe if node in reduced_graph}

    node_vectors = {}
    if semantic_diag["enabled"] and query_vec is not None and candidate_universe:
        node_vectors, vector_err = _load_candidate_vectors(
            nodes=sorted(candidate_universe),
            g=reduced_graph,
            cfg=cfg,
            semantic_state=semantic_state,
        )
        if vector_err and not semantic_diag["error"]:
            semantic_diag["error"] = vector_err
    semantic_diag["candidate_vector_count"] = int(len(node_vectors))

    query_sim_map = {}
    if query_vec is not None and node_vectors:
        vector_nodes = list(node_vectors.keys())
        vec_mat = np.stack([node_vectors[node] for node in vector_nodes], axis=0).astype(np.float32, copy=False)
        qvec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        sims = np.matmul(vec_mat, qvec)
        for node, sim in zip(vector_nodes, sims.tolist()):
            query_sim_map[node] = float(sim)
        candidate_similarity_recomputed_count += int(len(vector_nodes))
    for node, score in semantic_scores.items():
        query_sim_map[node] = max(float(score), float(query_sim_map.get(node, -1.0)))

    anchor_vecs = {anchor: node_vectors[anchor] for anchor in anchors if anchor in node_vectors}
    anchor_sim_map = {node: 0.0 for node in candidate_universe}
    if anchor_vecs and candidate_universe:
        anchor_nodes = list(anchor_vecs.keys())
        anchor_mat = np.stack([anchor_vecs[node] for node in anchor_nodes], axis=0).astype(np.float32, copy=False)
        sim_nodes = [node for node in candidate_universe if node in node_vectors]
        if sim_nodes:
            cand_mat = np.stack([node_vectors[node] for node in sim_nodes], axis=0).astype(np.float32, copy=False)
            sim_mat = np.matmul(cand_mat, anchor_mat.T)
            max_sims = np.max(sim_mat, axis=1)
            for node, sim in zip(sim_nodes, max_sims.tolist()):
                anchor_sim_map[node] = float(sim)
            candidate_similarity_recomputed_count += int(len(sim_nodes) * len(anchor_nodes))

    support_sim_map = {}
    for node in candidate_universe:
        support_sim_map[node] = _entity_support_similarity(reduced_graph, node, query_sim_map)

    tau_cap = max(1, int(getattr(cfg, "tau", 4)))
    anchor_distance_maps = {
        anchor: _get_distance_map(reduced_graph, anchor, tau_cap, distance_map_cache={})
        for anchor in anchors
        if anchor in reduced_graph
    }
    seed_bridge_bonus_global = _seed_bridge_bonus_map(
        candidates=sorted(candidate_universe),
        anchors=anchors,
        anchor_distance_maps=anchor_distance_maps,
        tau=tau_cap,
    )
    seed_chunk_grounding_bonus_global = _seed_chunk_grounding_bonus_map(
        candidates=sorted(candidate_universe),
        g=reduced_graph,
        semantic_state=semantic_state,
        query_sim_map=query_sim_map,
        support_sim_map=support_sim_map,
    )

    semantic_union_enabled = bool(getattr(cfg, "semantic_candidate_union", True))
    distance_map_cache = {}
    run_results = []
    for run in run_base:
        graph_candidates = list(run.get("graph_candidates", []))
        semantic_candidates = []
        if semantic_union_enabled:
            semantic_candidates = _lazy_union_candidates_for_run(
                anchor_scores=run.get("anchor_scores", {}) or {},
                anchors=anchors,
                proposal_by_anchor=proposal_by_anchor,
                cfg=cfg,
                shared_high_conf=(proposal_partitions or {}).get("shared_high_conf", []),
                global_fallback=(proposal_partitions or {}).get("global_fallback", []),
            )
            candidates = _ordered_unique(graph_candidates + semantic_candidates)
        else:
            candidates = list(graph_candidates)
        candidates = [node for node in candidates if node in reduced_graph]
        if not candidates:
            candidates = list(anchors)

        if semantic_diag["enabled"] and query_vec is not None:
            bridge_bonus_map = {node: float(seed_bridge_bonus_global.get(node, 0.0)) for node in candidates}
            chunk_grounding_bonus_map = {
                node: float(seed_chunk_grounding_bonus_global.get(node, 0.0)) for node in candidates
            }
            seed_score_map, graph_norm, semantic_norm, anchor_norm, bridge_norm, grounding_norm = _seed_hybrid_scores(
                candidates=candidates,
                agg_scores=run.get("agg_scores", {}),
                query_sim=query_sim_map,
                anchor_sim=anchor_sim_map,
                support_sim=support_sim_map,
                cfg=cfg,
                bridge_bonus_map=bridge_bonus_map,
                chunk_grounding_bonus_map=chunk_grounding_bonus_map,
            )
        else:
            seed_score_map = {node: float(run["agg_scores"].get(node, 0.0)) for node in candidates}
            graph_norm = _normalize_map(seed_score_map)
            semantic_norm = {node: 0.0 for node in candidates}
            anchor_norm = {node: 0.0 for node in candidates}
            bridge_norm = {node: 0.0 for node in candidates}
            grounding_norm = {node: 0.0 for node in candidates}

        objective_weights = _seed_selection_objective_weights(
            candidates=candidates,
            seed_score_map=seed_score_map,
            bridge_bonus_map=bridge_bonus_map if semantic_diag["enabled"] and query_vec is not None else {},
            chunk_grounding_bonus_map=chunk_grounding_bonus_map if semantic_diag["enabled"] and query_vec is not None else {},
            anchors=anchors,
            anchor_distance_maps=anchor_distance_maps,
            cfg=cfg,
            tau=int(getattr(cfg, "tau", 4)),
        )
        if max(objective_weights.values(), default=0.0) <= 0.0:
            objective_weights = {node: float(run["agg_scores"].get(node, 0.0)) for node in candidates}
        seeds = _greedy_seed_set(
            reduced_graph,
            candidates,
            objective_weights,
            int(getattr(cfg, "seed_k", 4)),
            int(getattr(cfg, "tau", 4)),
            distance_map_cache=distance_map_cache,
        )
        if not seeds and candidates:
            fallback = sorted(candidates, key=lambda n: float(seed_score_map.get(n, 0.0)), reverse=True)
            seeds = set(fallback[: max(1, int(getattr(cfg, "seed_k", 4)))])

        run_results.append(
            {
                "run_id": int(run["run_id"]),
                "anchor_scores": run["anchor_scores"],
                "agg_scores": run["agg_scores"],
                "graph_candidates": graph_candidates,
                "semantic_candidates": semantic_candidates,
                "candidates": candidates,
                "seeds": set(seeds),
                "seed_score_map": seed_score_map,
                "seed_score_components": {
                    "graph": graph_norm,
                    "semantic": semantic_norm,
                    "anchor": anchor_norm,
                    "bridge": bridge_norm,
                    "grounding": grounding_norm,
                },
            }
        )

    run_score_sparse_topk = max(12, int(getattr(cfg, "run_score_sparse_topk", 64)))
    for run in run_results:
        full_nodes = _ordered_unique(list(run.get("candidates", [])) + list(run.get("seeds", [])))
        if not full_nodes:
            full_nodes = list(run.get("seeds", [])) or list(anchors)
        base_weights = {
            node: float(run.get("seed_score_map", {}).get(node, run.get("agg_scores", {}).get(node, 0.0)))
            for node in full_nodes
        }
        score_nodes = list(full_nodes)
        if len(score_nodes) > run_score_sparse_topk:
            ranked_nodes = sorted(score_nodes, key=lambda n: float(base_weights.get(n, 0.0)), reverse=True)
            score_nodes = _ordered_unique(ranked_nodes[:run_score_sparse_topk] + list(run.get("seeds", [])))
        node_weights = {node: float(base_weights.get(node, 0.0)) for node in score_nodes}
        w = np.asarray([max(0.0, float(node_weights.get(node, 0.0))) for node in score_nodes], dtype=np.float32)
        if w.size <= 0:
            w = np.ones((1,), dtype=np.float32)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            w = np.ones_like(w, dtype=np.float32)
            w_sum = float(np.sum(w))

        sem_vec = np.asarray([float(query_sim_map.get(node, 0.0)) for node in score_nodes], dtype=np.float32)
        anc_vec = np.asarray([float(anchor_sim_map.get(node, 0.0)) for node in score_nodes], dtype=np.float32)
        semantic_raw = float(np.dot(w, sem_vec) / max(w_sum, 1.0e-8))
        anchor_raw = float(np.dot(w, anc_vec) / max(w_sum, 1.0e-8))

        run["score_nodes"] = score_nodes
        run["score_node_weights"] = node_weights
        run["semantic_coverage_pre"] = max(0.0, min(1.0, 0.5 * (semantic_raw + 1.0)))
        run["anchor_alignment_pre"] = max(0.0, min(1.0, 0.5 * (anchor_raw + 1.0)))

        # cheap pre-score for two-stage run scoring
        seeds = list(run.get("seeds", set()) or [])
        if seeds:
            seed_mass_vals = np.asarray(
                [float(run.get("seed_score_map", {}).get(seed, 0.0)) for seed in seeds],
                dtype=np.float32,
            )
            seed_mass = float(np.mean(seed_mass_vals)) if seed_mass_vals.size > 0 else 0.0
            seed_sem_vals = [
                0.5
                * (
                    max(float(query_sim_map.get(seed, 0.0)), float(support_sim_map.get(seed, 0.0)))
                    + 1.0
                )
                for seed in seeds
            ]
            max_seed_semantic = float(max(seed_sem_vals)) if seed_sem_vals else 0.0
        else:
            seed_mass = 0.0
            max_seed_semantic = 0.0

        seed_mass_norm = max(0.0, min(1.0, seed_mass))
        anchor_hit = 0
        if anchors:
            for anchor in anchors:
                a_scores = (run.get("anchor_scores", {}) or {}).get(anchor, {}) or {}
                if any(float(a_scores.get(seed, 0.0)) > 0.0 for seed in seeds):
                    anchor_hit += 1
            anchor_coverage = float(anchor_hit / max(len(anchors), 1))
        else:
            anchor_coverage = 0.0

        bridge_hit = 0
        if seeds and len(anchors) >= 2:
            for seed in seeds:
                connected = 0
                for anchor in anchors:
                    dmap = anchor_distance_maps.get(anchor, {})
                    if int(dmap.get(seed, tau_cap + 1)) <= tau_cap:
                        connected += 1
                if connected >= 2:
                    bridge_hit += 1
            bridge_proxy = float(bridge_hit / max(len(seeds), 1))
        else:
            bridge_proxy = 0.0

        pre_score = (
            0.40 * float(seed_mass_norm)
            + 0.25 * float(anchor_coverage)
            + 0.20 * float(max_seed_semantic)
            + 0.15 * float(bridge_proxy)
        )
        run["cheap_pre_score"] = float(pre_score)
        run["cheap_pre_components"] = {
            "graph_seed_mass": float(seed_mass_norm),
            "anchor_coverage": float(anchor_coverage),
            "max_semantic_seed": float(max_seed_semantic),
            "bridge_proxy": float(bridge_proxy),
        }

    preshortlist_topm = max(
        1,
        int(
            getattr(
                cfg,
                "phase1_run_preshortlist_topm",
                max(1, int(getattr(cfg, "phase1_run_shortlist_topk", 2))),
            )
        ),
    )
    pre_ranked_runs = sorted(run_results, key=lambda r: float(r.get("cheap_pre_score", 0.0)), reverse=True)
    full_eval_runs = list(pre_ranked_runs[: min(len(pre_ranked_runs), preshortlist_topm)])
    full_eval_run_ids = {int(r.get("run_id", -1)) for r in full_eval_runs}
    for run in run_results:
        run["full_eval_selected"] = bool(int(run.get("run_id", -1)) in full_eval_run_ids)
        run["hybrid_run_score"] = float(run.get("cheap_pre_score", 0.0))
        run["surrogate_loss"] = float(-run.get("hybrid_run_score", 0.0))
        run["global_loss"] = 0.0
        run["run_score_components"] = {
            "cheap_pre_score": float(run.get("cheap_pre_score", 0.0)),
            "cheap_pre_components": dict(run.get("cheap_pre_components", {}) or {}),
            "full_eval_selected": bool(run.get("full_eval_selected", False)),
        }

    if stable_seed_selection and full_eval_runs and len(full_eval_runs) > 1:
        surrogate_universe = set(anchors)
        for run in full_eval_runs:
            surrogate_universe.update(run.get("candidates", []))

        surrogate_weights = {}
        for node in surrogate_universe:
            vals = [r["seed_score_map"].get(node, r["agg_scores"].get(node, 0.0)) for r in full_eval_runs]
            surrogate_weights[node] = sum(float(v) for v in vals) / max(len(vals), 1)
        if max(surrogate_weights.values(), default=0.0) <= 0.0:
            for node in surrogate_universe:
                vals = [r["agg_scores"].get(node, 0.0) for r in full_eval_runs]
                surrogate_weights[node] = sum(float(v) for v in vals) / max(len(vals), 1)

        surrogate_focus_topk = max(24, int(getattr(cfg, "run_score_surrogate_topk", 128)))
        surrogate_focus = sorted(surrogate_weights.items(), key=lambda kv: float(kv[1]), reverse=True)
        surrogate_focus_nodes = [node for node, _ in surrogate_focus[:surrogate_focus_topk]]
        if not surrogate_focus_nodes:
            surrogate_focus_nodes = list(surrogate_universe)

        canonical_mode = is_canonical_copy_span_mode(
            str(getattr(cfg, "retrieval_objective_mode", "") or "").strip().lower()
        )
        if canonical_mode:
            run_weights = canonical_run_weight_bundle(cfg)
            w_sem = float(run_weights["semantic"])
            w_anchor = float(run_weights["anchor"])
            w_struct = float(run_weights["structure"])
            w_bridge = float(run_weights["bridge"])
            w_redundancy = float(run_weights["redundancy"])
            w_pair_cov = 0.0
            w_bridge_complete = 0.0
            w_grounding = 0.0
            w_anchor_disp = 0.0
        else:
            w_sem = max(0.0, float(getattr(cfg, "run_score_semantic_weight", 0.30)))
            w_anchor = max(0.0, float(getattr(cfg, "run_score_anchor_weight", 0.20)))
            w_struct = max(0.0, float(getattr(cfg, "run_score_structure_weight", 0.25)))
            w_bridge = max(0.0, float(getattr(cfg, "run_score_bridge_weight", 0.15)))
            w_redundancy = max(0.0, float(getattr(cfg, "run_score_redundancy_weight", 0.10)))
            w_pair_cov = max(0.0, float(getattr(cfg, "run_score_pair_coverage_weight", 0.0)))
            w_bridge_complete = max(0.0, float(getattr(cfg, "run_score_bridge_completeness_weight", 0.0)))
            w_grounding = max(0.0, float(getattr(cfg, "run_score_entity_chunk_grounding_weight", 0.0)))
            w_anchor_disp = max(0.0, float(getattr(cfg, "run_score_anchor_dispersion_penalty", 0.0)))
            total = (
                w_sem
                + w_anchor
                + w_struct
                + w_bridge
                + w_redundancy
                + w_pair_cov
                + w_bridge_complete
                + w_grounding
                + w_anchor_disp
            )
            if total <= 0.0:
                w_sem, w_anchor, w_struct, w_bridge, w_redundancy = 0.30, 0.20, 0.25, 0.15, 0.10
                w_pair_cov, w_bridge_complete, w_grounding, w_anchor_disp = 0.0, 0.0, 0.0, 0.0
                total = 1.0
            w_sem /= total
            w_anchor /= total
            w_struct /= total
            w_bridge /= total
            w_redundancy /= total
            w_pair_cov /= total
            w_bridge_complete /= total
            w_grounding /= total
            w_anchor_disp /= total

        best = None
        best_score = float("-inf")
        structural_cache = {}
        redundancy_cache = {}
        bridge_cache = {}
        pair_cov_cache = {}
        bridge_complete_cache = {}
        grounding_cache = {}
        anchor_disp_cache = {}
        for run in full_eval_runs:
            run_nodes = list(run.get("score_nodes", []) or [])
            semantic_cov = float(run.get("semantic_coverage_pre", 0.0))
            anchor_align = float(run.get("anchor_alignment_pre", 0.0))
            seeds_key = tuple(sorted(run.get("seeds", set())))
            if seeds_key not in structural_cache:
                structural_cache[seeds_key] = _run_structural_connectivity(
                    reduced_graph,
                    surrogate_universe=surrogate_focus_nodes,
                    surrogate_weights=surrogate_weights,
                    seeds=run.get("seeds", set()),
                    tau=int(getattr(cfg, "tau", 4)),
                    distance_map_cache=distance_map_cache,
                )
            structural_conn, dispersion_penalty, structural_loss = structural_cache[seeds_key]

            bridge_key = tuple(run_nodes)
            if bridge_key not in bridge_cache:
                bridge_cache[bridge_key] = _bridge_utility(
                    reduced_graph,
                    anchors=anchors,
                    nodes=run_nodes,
                    tau=int(getattr(cfg, "tau", 4)),
                    distance_map_cache=distance_map_cache,
                )
            bridge = float(bridge_cache[bridge_key])

            if seeds_key not in redundancy_cache:
                redundancy_cache[seeds_key] = _semantic_redundancy_penalty(
                    run.get("seeds", set()),
                    node_vectors=node_vectors,
                )
            redundancy = float(redundancy_cache[seeds_key])

            pair_coverage = 0.0
            bridge_completeness = 0.0
            grounding_score = 0.0
            anchor_dispersion = 0.0
            if not canonical_mode:
                if seeds_key not in pair_cov_cache:
                    pair_cov_cache[seeds_key] = _run_pair_coverage_score(
                        anchors=anchors,
                        seeds=run.get("seeds", set()),
                        anchor_distance_maps=anchor_distance_maps,
                        tau=int(getattr(cfg, "tau", 4)),
                    )
                pair_coverage = float(pair_cov_cache[seeds_key])

                bridge_complete_key = tuple(run_nodes)
                if bridge_complete_key not in bridge_complete_cache:
                    bridge_complete_cache[bridge_complete_key] = _run_bridge_path_completeness(
                        anchors=anchors,
                        run_nodes=run_nodes,
                        anchor_distance_maps=anchor_distance_maps,
                        tau=int(getattr(cfg, "tau", 4)),
                    )
                bridge_completeness = float(bridge_complete_cache[bridge_complete_key])

                grounding_key = (seeds_key, bridge_complete_key)
                if grounding_key not in grounding_cache:
                    grounding_cache[grounding_key] = _run_entity_chunk_grounding_score(
                        seeds=run.get("seeds", set()),
                        run_nodes=run_nodes,
                        semantic_state=semantic_state,
                    )
                grounding_score = float(grounding_cache[grounding_key])

                if seeds_key not in anchor_disp_cache:
                    anchor_disp_cache[seeds_key] = _run_anchor_dispersion_penalty(
                        anchors=anchors,
                        seeds=run.get("seeds", set()),
                        anchor_distance_maps=anchor_distance_maps,
                        tau=int(getattr(cfg, "tau", 4)),
                    )
                anchor_dispersion = float(anchor_disp_cache[seeds_key])

            if canonical_mode:
                run_score = canonical_run_score(
                    semantic=semantic_cov,
                    anchor=anchor_align,
                    structure=structural_conn,
                    bridge=bridge,
                    redundancy=redundancy,
                    cfg=cfg,
                )
                surrogate_loss = canonical_run_surrogate_loss(
                    semantic=semantic_cov,
                    anchor=anchor_align,
                    structure=structural_conn,
                    bridge=bridge,
                    redundancy=redundancy,
                    cfg=cfg,
                    dispersion_penalty=dispersion_penalty,
                )
            else:
                run_score = (
                    w_sem * semantic_cov
                    + w_anchor * anchor_align
                    + w_struct * structural_conn
                    + w_bridge * bridge
                    + w_pair_cov * pair_coverage
                    + w_bridge_complete * bridge_completeness
                    + w_grounding * grounding_score
                    - w_anchor_disp * anchor_dispersion
                    - w_redundancy * redundancy
                )
                surrogate_loss = (
                    w_sem * (1.0 - semantic_cov)
                    + w_anchor * (1.0 - anchor_align)
                    + w_struct * (1.0 - structural_conn)
                    + w_bridge * (1.0 - bridge)
                    + w_pair_cov * (1.0 - pair_coverage)
                    + w_bridge_complete * (1.0 - bridge_completeness)
                    + w_grounding * (1.0 - grounding_score)
                    + w_anchor_disp * anchor_dispersion
                    + w_redundancy * redundancy
                    + 0.25 * dispersion_penalty
                )
            run["run_score_components"] = {
                "semantic_coverage": float(semantic_cov),
                "anchor_alignment": float(anchor_align),
                "structural_connectivity": float(structural_conn),
                "bridge_utility": float(bridge),
                "pair_coverage_score": float(pair_coverage),
                "bridge_path_completeness": float(bridge_completeness),
                "entity_chunk_grounding_score": float(grounding_score),
                "anchor_dispersion_penalty": float(anchor_dispersion),
                "redundancy_penalty": float(redundancy),
                "dispersion_penalty": float(dispersion_penalty),
                "cheap_pre_score": float(run.get("cheap_pre_score", 0.0)),
                "cheap_pre_components": dict(run.get("cheap_pre_components", {}) or {}),
                "full_eval_selected": True,
                "canonical_core_scoring": bool(canonical_mode),
            }
            run["hybrid_run_score"] = float(run_score)
            run["surrogate_loss"] = float(surrogate_loss)
            run["global_loss"] = float(structural_loss)
            if run_score > best_score:
                best_score = run_score
                best = run
        chosen = best if best is not None else full_eval_runs[0]
    else:
        chosen = pre_ranked_runs[0] if pre_ranked_runs else {
            "run_id": 0,
            "anchor_scores": {},
            "graph_candidates": [],
            "semantic_candidates": [],
            "candidates": [],
            "seeds": set(),
            "agg_scores": {},
            "seed_score_map": {},
            "seed_score_components": {"graph": {}, "semantic": {}, "anchor": {}, "bridge": {}, "grounding": {}},
            "run_score_components": {},
            "hybrid_run_score": 0.0,
            "surrogate_loss": 0.0,
            "global_loss": 0.0,
        }

    shortlist_k = max(1, int(getattr(cfg, "phase1_run_shortlist_topk", 2)))
    full_score_topk = max(
        shortlist_k,
        int(
            getattr(
                cfg,
                "phase1_full_run_score_topk",
                max(shortlist_k, int(getattr(cfg, "phase1_run_preshortlist_topm", shortlist_k))),
            )
        ),
    )
    ranked_runs = sorted(run_results, key=lambda r: float(r.get("hybrid_run_score", 0.0)), reverse=True)
    ranked_runs = ranked_runs[: min(len(ranked_runs), full_score_topk)]
    ranked_runs, run_light_rerank_diag = _apply_lightweight_run_rerank(
        ranked_runs=ranked_runs,
        cfg=cfg,
    )
    shortlisted_runs = ranked_runs[:shortlist_k] if ranked_runs else [chosen]
    chosen = shortlisted_runs[0]
    stage_ms["phase1_run_scoring_ms"] = float((time.perf_counter() - run_score_start) * 1000.0)
    stage_ms["run_scoring_time_ms"] = float(stage_ms["phase1_run_scoring_ms"])

    phase2_start = time.perf_counter()
    pair_shortlist_start = time.perf_counter()
    pair_shortlist = _phase2_pair_shortlist(
        shortlisted_runs=shortlisted_runs,
        anchors=anchors,
        g=reduced_graph,
        query_sim_map=query_sim_map,
        support_sim_map=support_sim_map,
        cfg=cfg,
    )
    if not pair_shortlist and chosen.get("seeds"):
        fallback = []
        for anchor in anchors:
            for seed in list(chosen.get("seeds", [])):
                fallback.append(
                    {
                        "anchor": anchor,
                        "seed": seed,
                        "run_id": int(chosen.get("run_id", 0)),
                        "pair_proxy_score": float(chosen.get("seed_score_map", {}).get(seed, 0.0)),
                        "distance": int(_shortest_distance_with_cap(reduced_graph, anchor, seed, int(getattr(cfg, "tau", 4)) + 1)),
                        "anchor_alignment": float((chosen.get("anchor_scores", {}).get(anchor, {}) or {}).get(seed, 0.0)),
                        "seed_strength": float(chosen.get("seed_score_map", {}).get(seed, 0.0)),
                        "semantic_relevance": float(
                            0.5
                            * (
                                max(float(query_sim_map.get(seed, 0.0)), float(support_sim_map.get(seed, 0.0)))
                                + 1.0
                            )
                        ),
                        "bridge_potential": 0.0,
                    }
                )
        fallback.sort(key=lambda x: x["pair_proxy_score"], reverse=True)
        pair_shortlist = fallback[: max(1, int(getattr(cfg, "pair_shortlist_topb", 6)))]
    stage_ms["phase2_pair_shortlist_ms"] = float((time.perf_counter() - pair_shortlist_start) * 1000.0)

    phase2_refine_start = time.perf_counter()
    corridor, retained_pairs, sentence_scores, corridor_payloads = _phase2_local_refinement(
        g=reduced_graph,
        shortlisted_pairs=pair_shortlist,
        shortlisted_runs=shortlisted_runs,
        query_sim_map=query_sim_map,
        support_sim_map=support_sim_map,
        cfg=cfg,
    )
    stage_ms["phase2_refine_ms"] = float((time.perf_counter() - phase2_refine_start) * 1000.0)
    stage_ms["phase2_refinement_time_ms"] = float((time.perf_counter() - phase2_start) * 1000.0)

    final_start = time.perf_counter()
    corridor_payloads = _rerank_corridors_hybrid(
        corridors=corridor_payloads,
        g=reduced_graph,
        query_sim_map=query_sim_map,
        support_sim_map=support_sim_map,
        cfg=cfg,
        coverage_enabled_override=bool(objective_flags.get("coverage_selection", False)),
    )
    top1_corridor_start = time.perf_counter()
    corridor_payloads, top1_corridor_diag = _apply_lightweight_corridor_top1_correction(
        reranked_corridors=corridor_payloads,
        cfg=cfg,
    )
    stage_ms["top1_correction_ms"] += float((time.perf_counter() - top1_corridor_start) * 1000.0)

    final_graph = corridor
    if enable_trim and bool(getattr(cfg, "trim_on", True)):
        final_graph = _greedy_trim(
            corridor.copy(),
            anchors=anchors,
            seeds=set(chosen.get("seeds", set())),
            rho=float(getattr(cfg, "trim_rho", 0.6)),
        )

    entity_chunk_diag = {
        "enabled": str(getattr(cfg, "graph_mode", "current_entity_graph") or "current_entity_graph").strip().lower()
        == "entity_chunk_graph",
        "applied": False,
        "chunk_scoring_mode": str(getattr(cfg, "chunk_scoring_mode", "entity_aggregate") or "entity_aggregate"),
        "chunk_score_topk": int(getattr(cfg, "chunk_score_topk", 20)),
        "chunk_package_enabled": bool(getattr(cfg, "chunk_package_enabled", False)),
        "selected_chunks": [],
    }
    if entity_chunk_diag["enabled"]:
        ranked_chunks, chunk_graph_diag = _entity_chunk_graph_chunk_scoring(
            reduced_graph=reduced_graph,
            final_graph=final_graph,
            chosen_run=chosen,
            shortlisted_runs=shortlisted_runs,
            semantic_state=semantic_state,
            query_sim_map=query_sim_map,
            support_sim_map=support_sim_map,
            anchors=anchors,
            cfg=cfg,
            role_aware_enabled_override=bool(objective_flags.get("role_aware_chunk_scoring", False)),
        )
        entity_chunk_diag.update(chunk_graph_diag or {})
        if bool(getattr(cfg, "chunk_package_enabled", False)) and ranked_chunks:
            for chunk_node, c_score in ranked_chunks:
                sentence_scores[chunk_node] = max(float(sentence_scores.get(chunk_node, 0.0)), float(c_score))
            entity_chunk_diag["applied"] = True
            entity_chunk_diag["selected_chunks"] = [str(c) for c, _ in ranked_chunks]
            entity_chunk_diag["selected_chunk_scores"] = {str(c): float(s) for c, s in ranked_chunks}

    selected_nodes = set(final_graph.nodes())
    if entity_chunk_diag.get("applied", False):
        selected_nodes.update(entity_chunk_diag.get("selected_chunks", []))
    selected_sentence_ids, selected_sentences, selected_sentence_score_map = _extract_sentence_payload(
        g,
        selected_nodes,
        sentence_scores,
    )
    selected_text_map = {sid: text for sid, text in zip(selected_sentence_ids, selected_sentences) if sid and text}

    sentence_rerank_enabled = bool(getattr(cfg, "sentence_rerank_enabled", True))
    ablation_no_final_text_rerank = bool(getattr(cfg, "ablation_no_final_text_rerank", False))
    trace_enabled = bool(getattr(cfg, "score_component_trace_enabled", False))
    (
        selected_sentence_ids,
        selected_sentences,
        embedding_diag,
        sentence_rerank_semantic_calls,
        sentence_rerank_ms,
    ) = _apply_final_text_rerank(
        question=sample.question,
        selected_sentence_ids=selected_sentence_ids,
        selected_sentences=selected_sentences,
        selected_sentence_score_map=selected_sentence_score_map,
        cfg=cfg,
    )
    stage_ms["sentence_rerank_ms"] = float(sentence_rerank_ms)

    filtered_corridors = _filter_corridor_payloads(corridor_payloads, selected_sentence_ids)
    corridor_count_before_trim = int(len(corridor_payloads))
    corridor_count_after_trim = int(len(filtered_corridors))
    sentence_feature_table = _build_sentence_feature_table(
        sample=sample,
        selected_sentence_ids=selected_sentence_ids,
        sentence_texts=selected_sentences,
        sentence_score_map=selected_sentence_score_map,
        corridors=filtered_corridors,
    )
    selected_sentence_ids, selected_sentences, gl_rcedr_diag = apply_gl_rcedr(
        question_text=sample.question,
        selected_sentence_ids=selected_sentence_ids,
        selected_sentences=selected_sentences,
        sentence_feature_table=sentence_feature_table,
        cfg=cfg,
    )
    if bool(gl_rcedr_diag.get("applied", False)):
        sentence_feature_table = _build_sentence_feature_table(
            sample=sample,
            selected_sentence_ids=selected_sentence_ids,
            sentence_texts=selected_sentences,
            sentence_score_map=selected_sentence_score_map,
            corridors=filtered_corridors,
        )
    selected_sentence_ids, selected_sentences, evidence_density_rerank_diag = rerank_evidence_density(
        question_text=sample.question,
        selected_sentence_ids=selected_sentence_ids,
        selected_sentences=selected_sentences,
        sentence_feature_table=sentence_feature_table,
        cfg=cfg,
    )
    if bool(evidence_density_rerank_diag.get("applied", False)):
        sentence_feature_table = _build_sentence_feature_table(
            sample=sample,
            selected_sentence_ids=selected_sentence_ids,
            sentence_texts=selected_sentences,
            sentence_score_map=selected_sentence_score_map,
            corridors=filtered_corridors,
        )
    top1_sentence_start = time.perf_counter()
    selected_sentence_ids, selected_sentences, top1_sentence_diag = _apply_lightweight_sentence_top1_correction(
        selected_sentence_ids=selected_sentence_ids,
        selected_sentences=selected_sentences,
        sentence_feature_table=sentence_feature_table,
        cfg=cfg,
    )
    stage_ms["top1_correction_ms"] += float((time.perf_counter() - top1_sentence_start) * 1000.0)
    if bool(top1_sentence_diag.get("applied", False)):
        sentence_feature_table = _build_sentence_feature_table(
            sample=sample,
            selected_sentence_ids=selected_sentence_ids,
            sentence_texts=selected_sentences,
            sentence_score_map=selected_sentence_score_map,
            corridors=filtered_corridors,
        )
    selected_sentence_ids, selected_sentences, answerability_selection_diag = apply_answerability_constrained_selection(
        question_text=sample.question,
        selected_sentence_ids=selected_sentence_ids,
        selected_sentences=selected_sentences,
        sentence_feature_table=sentence_feature_table,
        cfg=cfg,
    )
    if bool(answerability_selection_diag.get("applied", False)):
        sentence_feature_table = _build_sentence_feature_table(
            sample=sample,
            selected_sentence_ids=selected_sentence_ids,
            sentence_texts=selected_sentences,
            sentence_score_map=selected_sentence_score_map,
            corridors=filtered_corridors,
        )
    selected_text_map = {
        sid: text for sid, text in zip(selected_sentence_ids, selected_sentences) if sid and text
    }
    final_total_ms = float((time.perf_counter() - final_start) * 1000.0)
    stage_ms["render_ms"] = max(
        0.0,
        float(final_total_ms)
        - float(stage_ms.get("sentence_rerank_ms", 0.0))
        - float(stage_ms.get("top1_correction_ms", 0.0)),
    )
    stage_ms["final_render_time_ms"] = float(stage_ms["render_ms"])

    graph_mode_diag = str(getattr(cfg, "graph_mode", "current_entity_graph") or "current_entity_graph")
    stagewise_loss_funnel = {"enabled": False}
    if bool(getattr(cfg, "stagewise_loss_funnel_enabled", True)):
        stagewise_loss_funnel = _compute_stagewise_loss_funnel(
            sample=sample,
            g=g,
            reduced_graph=reduced_graph,
            anchors=anchors,
            proposal_nodes=proposal_nodes,
            shortlisted_runs=shortlisted_runs,
            selected_text_unit_ids=selected_sentence_ids,
            selected_text_map=selected_text_map,
            graph_mode=graph_mode_diag,
        )
    connector_metrics = _compute_connector_retrieval_metrics(
        g=reduced_graph if reduced_graph is not None else g,
        anchors=anchors,
        chosen_run=chosen,
        filtered_corridors=filtered_corridors,
        anchor_distance_maps=anchor_distance_maps,
        query_sim_map=query_sim_map,
        support_sim_map=support_sim_map,
        node_vectors=node_vectors,
        tau=int(getattr(cfg, "tau", 4)),
        cfg=cfg,
    )

    anchor_results = []
    diag_topn = max(1, int(getattr(cfg, "anchor_diag_topn", 10)))
    diag_full = bool(getattr(cfg, "anchor_diag_store_full_scores", False))
    for anchor in anchors:
        for sample_idx, scores in enumerate(per_anchor_runs.get(anchor, [])):
            top_candidates = _topk_nodes(scores, diag_topn)
            if diag_full:
                score_payload = {k: float(v) for k, v in scores.items() if v > 0.0}
            else:
                score_payload = {k: float(scores.get(k, 0.0)) for k in top_candidates}
            anchor_results.append(
                AnchorResult(
                    anchor=anchor,
                    scores=score_payload,
                    top_candidates=top_candidates,
                    sample_index=sample_idx,
                    metadata={"topn": diag_topn, "full_scores": diag_full},
                )
            )

    score_component_trace = None
    if trace_enabled:
        trace_topn = max(1, int(getattr(cfg, "score_component_trace_topn", 20)))
        pair_score_examples = []
        for item in (pair_shortlist or [])[:trace_topn]:
            pair_score_examples.append(
                {
                    "anchor": str(item.get("anchor", "")),
                    "seed": str(item.get("seed", "")),
                    "run_id": int(item.get("run_id", 0)),
                    "pair_proxy_score": float(item.get("pair_proxy_score", 0.0) or 0.0),
                    "pair_score_components": dict(item.get("pair_score_components", {}) or {}),
                    "pair_score_effective_weights": dict(item.get("pair_score_effective_weights", {}) or {}),
                    "pair_score_disabled_components": list(item.get("pair_score_disabled_components", []) or []),
                }
            )

        if ablation_no_final_text_rerank:
            final_text_rerank_trace = {
                "enabled": False,
                "reason": "ablation_no_final_text_rerank",
            }
        elif not bool(sentence_rerank_enabled):
            final_text_rerank_trace = {
                "enabled": False,
                "reason": "sentence_rerank_disabled_by_config",
            }
        else:
            final_text_rerank_trace = {
                "enabled": bool(embedding_diag.get("enabled", False)),
                "applied": bool(embedding_diag.get("applied", False)),
                "error": str(embedding_diag.get("error", "") or ""),
                "reuse_candidate_available": False,
                "used_precomputed_embeddings": False,
                "reuse_deferred_to": "future_efficiency_phase",
            }

        score_component_trace = {
            "enabled": True,
            "semantic_usage": {
                "proposal": bool((semantic_diag or {}).get("enabled", False)),
                "seed_score": abs(float(getattr(cfg, "seed_score_semantic_weight", 0.0))) > 0.0,
                "run_pre_score": abs(float(getattr(cfg, "run_score_semantic_weight", 0.0))) > 0.0,
                "run_full_score": abs(float(getattr(cfg, "run_score_semantic_weight", 0.0))) > 0.0,
                "pair_score": not bool(getattr(cfg, "ablation_no_pair_semantic", False)),
                "local_refinement": True,
                "final_text_rerank": (
                    (not ablation_no_final_text_rerank) and bool(embedding_diag.get("enabled", False))
                ),
            },
            "ablation_flags": {
                "ablation_no_pair_semantic": bool(getattr(cfg, "ablation_no_pair_semantic", False)),
                "ablation_no_pair_bridge": bool(getattr(cfg, "ablation_no_pair_bridge", False)),
                "ablation_no_final_text_rerank": bool(getattr(cfg, "ablation_no_final_text_rerank", False)),
            },
            "pair_score_examples": pair_score_examples,
            "final_text_rerank": final_text_rerank_trace,
        }

    candidate_nodes_for_audit = _ordered_unique(
        list((chosen or {}).get("candidates", []) or []) + list((chosen or {}).get("seeds", set()) or [])
    )
    candidate_unit_ids, candidate_text_map = _candidate_text_unit_ids(
        reduced_graph if reduced_graph is not None else g,
        candidate_nodes_for_audit,
    )
    candidate_sentence_ids = sorted({str(x) for x in list(candidate_unit_ids or []) if str(x)})
    candidate_sentences = [str((candidate_text_map or {}).get(sid, "") or "") for sid in candidate_sentence_ids]

    latency_ms = (time.perf_counter() - start) * 1000.0
    return RetrievalResult(
        sample_id=sample.qid,
        method=method_name,
        anchors=anchors,
        seeds=sorted(chosen.get("seeds", set())),
        selected_nodes=sorted(selected_nodes),
        selected_sentence_ids=selected_sentence_ids,
        selected_sentences=selected_sentences,
        candidate_sentence_ids=candidate_sentence_ids,
        candidate_sentences=candidate_sentences,
        corridors=filtered_corridors,
        anchor_results=anchor_results,
        diagnostics={
            "graph_scope": graph_scope,
            "global_index": global_index_meta if graph_scope in {"global_corpus", "prebuilt_igraph"} else {},
            "ppr_engine_requested": str(getattr(cfg, "ppr_engine", "auto")),
            "ppr_engine_effective": ppr_engine,
            "phase1_parallel_ppr": bool(getattr(cfg, "phase1_parallel_ppr", True)),
            "phase2_refine_mode": str(getattr(cfg, "phase2_refine_mode", "bounded_local") or "bounded_local"),
            "phase2_bidirectional_full_ppr": bool(getattr(cfg, "phase2_bidirectional_full_ppr", False)),
            "reuse_semantic_scores_in_final": bool(getattr(cfg, "reuse_semantic_scores_in_final", True)),
            "graph_mode": str(getattr(cfg, "graph_mode", "current_entity_graph") or "current_entity_graph"),
            "chunk_scoring_mode": str(getattr(cfg, "chunk_scoring_mode", "entity_aggregate") or "entity_aggregate"),
            "entity_chunk_transition_weight": float(getattr(cfg, "entity_chunk_transition_weight", 0.35)),
            "chunk_node_enabled_in_diffusion": bool(getattr(cfg, "chunk_node_enabled_in_diffusion", False)),
            "chunk_score_topk": int(getattr(cfg, "chunk_score_topk", 20)),
            "chunk_package_enabled": bool(getattr(cfg, "chunk_package_enabled", False)),
            "corridor_fallback_text_cap": int(getattr(cfg, "corridor_fallback_text_cap", 4)),
            "canonical_variant_name": str(getattr(cfg, "canonical_variant_name", "a1_baseline_3_2") or "a1_baseline_3_2"),
            "retrieval_objective_mode": str(objective_flags.get("mode", "baseline")),
            "retrieval_objective_flags": dict(objective_flags or {}),
            "connector_objective_profile": dict(objective_profile_diag or {}),
            "hybrid_anchor_recall_diag": dict(hybrid_anchor_diag or {}),
            "bridge_candidate_induction_diag": dict(bridge_induction_diag or {}),
            "candidate_recall_boost_diag": dict(candidate_recall_boost_diag or {}),
            "gl_rcedr_diag": dict(gl_rcedr_diag or {}),
            "answerability_selection_diag": dict(answerability_selection_diag or {}),
            "evidence_density_rerank_diag": dict(evidence_density_rerank_diag or {}),
            "shared_budget_profile": str((shared_budget_diag or {}).get("resolved_profile", "off") or "off"),
            "shared_budget_proposal_step": int((shared_budget_diag or {}).get("proposal_step", 0)),
            "shared_budget_exploration_step": int((shared_budget_diag or {}).get("exploration_step", 0)),
            "shared_budget_diagnostics": dict(shared_budget_diag or {}),
            "selected_unit_type": ("chunk" if str(getattr(cfg, "graph_mode", "current_entity_graph") or "current_entity_graph").strip().lower() == "entity_chunk_graph" else "sentence"),
            "selected_text_map": selected_text_map,
            "selected_text_unit_ids": list(selected_sentence_ids),
            "selected_texts": list(selected_sentences),
            "candidate_text_unit_ids": list(candidate_sentence_ids),
            "candidate_texts": list(candidate_sentences),
            "stagewise_loss_funnel": stagewise_loss_funnel,
            "anchor_hit_rate": float((stagewise_loss_funnel or {}).get("anchor_hit_rate", 0.0)),
            "proposal_hit_rate": float((stagewise_loss_funnel or {}).get("proposal_hit_rate", 0.0)),
            "phase1_hit_rate": float((stagewise_loss_funnel or {}).get("phase1_hit_rate", 0.0)),
            "final_chunk_hit_rate": float((stagewise_loss_funnel or {}).get("final_chunk_hit_rate", 0.0)),
            "seed_anchor_recall": float((connector_metrics or {}).get("seed_anchor_recall", 0.0)),
            "seed_bridge_recall": float((connector_metrics or {}).get("seed_bridge_recall", 0.0)),
            "seed_answer_recall": float((connector_metrics or {}).get("seed_answer_recall", 0.0)),
            "seed_diversity": float((connector_metrics or {}).get("seed_diversity", 0.0)),
            "seed_redundancy": float((connector_metrics or {}).get("seed_redundancy", 0.0)),
            "run_bridge_coverage": float((connector_metrics or {}).get("run_bridge_coverage", 0.0)),
            "corridor_role_coverage": float((connector_metrics or {}).get("corridor_role_coverage", 0.0)),
            "answer_preserve_activation_rate": float((connector_metrics or {}).get("answer_preserve_activation_rate", 0.0)),
            "preserved_answer_usefulness": float((connector_metrics or {}).get("preserved_answer_usefulness", 0.0)),
            "path_complete_rate": float((connector_metrics or {}).get("path_complete_rate", 0.0)),
            "strict_path_complete_rate": float((connector_metrics or {}).get("strict_path_complete_rate", 0.0)),
            "answer_bearing_path_hit": float((connector_metrics or {}).get("answer_bearing_path_hit", 0.0)),
            "false_path_rate": float((connector_metrics or {}).get("false_path_rate", 0.0)),
            "equivalent_evidence_coverage": float((connector_metrics or {}).get("equivalent_evidence_coverage", 0.0)),
            "bridge_answer_pair_retention": float((connector_metrics or {}).get("bridge_answer_pair_retention", 0.0)),
            "incomplete_path_rate": float((connector_metrics or {}).get("incomplete_path_rate", 0.0)),
            "chain_compactness": float((connector_metrics or {}).get("chain_compactness", 0.0)),
            "path_preserve_activation_rate": float((connector_metrics or {}).get("path_preserve_activation_rate", 0.0)),
            "anchor_bridge_balance": float((connector_metrics or {}).get("anchor_bridge_balance", 0.0)),
            "anchor_role_share": float((connector_metrics or {}).get("anchor_role_share", 0.0)),
            "bridge_role_share": float((connector_metrics or {}).get("bridge_role_share", 0.0)),
            "answer_role_share": float((connector_metrics or {}).get("answer_role_share", 0.0)),
            "bridge_purity": float((connector_metrics or {}).get("bridge_purity", 0.0)),
            "answer_side_density": float((connector_metrics or {}).get("answer_side_density", 0.0)),
            "support_set_compactness": float((connector_metrics or {}).get("support_set_compactness", 0.0)),
            "bridge_noise_ratio": float((connector_metrics or {}).get("bridge_noise_ratio", 0.0)),
            "useful_bridge_rate": float((connector_metrics or {}).get("useful_bridge_rate", 0.0)),
            "bridge_to_answer_path_hit": float((connector_metrics or {}).get("bridge_to_answer_path_hit", 0.0)),
            "conversion_after_bridge": float((connector_metrics or {}).get("conversion_after_bridge_proxy", 0.0)),
            "conversion_after_bridge_proxy": float((connector_metrics or {}).get("conversion_after_bridge_proxy", 0.0)),
            "conversion_after_preserve": float((connector_metrics or {}).get("conversion_after_preserve_proxy", 0.0)),
            "conversion_after_preserve_proxy": float((connector_metrics or {}).get("conversion_after_preserve_proxy", 0.0)),
            "conversion_after_path_preserve": float((connector_metrics or {}).get("conversion_after_path_preserve_proxy", 0.0)),
            "conversion_after_path_preserve_proxy": float((connector_metrics or {}).get("conversion_after_path_preserve_proxy", 0.0)),
            "hotpot_overpreserve_rate": float((connector_metrics or {}).get("hotpot_overpreserve_rate", 0.0)),
            "redundancy_rate": float((connector_metrics or {}).get("redundancy_rate", 0.0)),
            "connector_quality": float((connector_metrics or {}).get("connector_quality", 0.0)),
            "connector_metrics": dict(connector_metrics or {}),
            "ppr_graph_nodes": int(diffusion_graph.number_of_nodes()),
            "ppr_graph_edges": int(diffusion_graph.number_of_edges()),
            "entity_chunk_graph": entity_chunk_diag,
            "diffusion_graph_diag": diffusion_diag,
            "proposal_reduced_subgraph": reduced_diag,
            "proposal_entity_count": int(proposal_diag.get("proposal_entity_count", 0)),
            "proposal_chunk_count": int(proposal_diag.get("proposal_chunk_count", 0)),
            "graph_reserve_count": int(proposal_diag.get("graph_reserve_count", 0)),
            "proposal_high_confidence_count": int(proposal_diag.get("high_confidence_count", 0)),
            "proposal_global_fallback_count": int(proposal_diag.get("global_fallback_count", 0)),
            "proposal_bridge_candidate_count": int(proposal_diag.get("bridge_candidate_count", 0)),
            "union_candidate_count": int(proposal_diag.get("union_candidate_count", len(proposal_nodes))),
            "proposal_subgraph_nodes": int(reduced_graph.number_of_nodes()),
            "proposal_subgraph_edges": int(reduced_graph.number_of_edges()),
            "phase1_run_count": int(n_runs),
            "phase1_run_preshortlist_topm": int(preshortlist_topm),
            "phase1_full_run_score_topk": int(full_score_topk),
            "phase1_full_eval_run_count": int(len(full_eval_runs)),
            "selected_run_count": int(len(shortlisted_runs)),
            "phase2_refined_pair_count": int(len(retained_pairs)),
            "best_run_id": int(chosen.get("run_id", 0)),
            "best_run_score": float(chosen.get("hybrid_run_score", 0.0) or 0.0),
            "best_run_score_components": chosen.get("run_score_components", {}) or {},
            "best_seed_score_components": chosen.get("seed_score_components", {}) or {},
            "num_seeds": int(len(chosen.get("seeds", set()))),
            "num_candidates_graph": int(len(chosen.get("graph_candidates", []) or [])),
            "num_candidates_semantic": int(len(chosen.get("semantic_candidates", []) or [])),
            "num_candidates_union": int(len(chosen.get("candidates", []) or [])),
            "retained_pairs": [[a, z] for a, z in retained_pairs],
            "pair_shortlist": pair_shortlist,
            "num_corridors": int(len(filtered_corridors)),
            "corridor_count_before_trim": int(corridor_count_before_trim),
            "corridor_count_after_trim": int(corridor_count_after_trim),
            "corridor_nodes_before_trim": int(corridor.number_of_nodes()),
            "corridor_nodes_after_trim": int(final_graph.number_of_nodes()),
            "sentence_scores": sentence_scores,
            "sentence_feature_table": sentence_feature_table,
            "text_unit_feature_table": sentence_feature_table,
            "semantic_selection": semantic_diag,
            "query_embedding_recomputed": bool(query_embedding_recomputed),
            "query_embedding_cache_hit": bool(query_embedding_cache_hit),
            "semantic_entity_lookup_mode": str(semantic_entity_lookup_mode),
            "semantic_chunk_lookup_mode": str(semantic_chunk_lookup_mode),
            "candidate_similarity_recomputed_count": int(candidate_similarity_recomputed_count),
            "semantic_scores_reused_in_final": bool(getattr(cfg, "reuse_semantic_scores_in_final", True)),
            "sentence_rerank_semantic_calls": int(sentence_rerank_semantic_calls),
            "top1_correction": {
                "enabled": bool(getattr(cfg, "top1_correction_enabled", False)),
                "topk": int(getattr(cfg, "top1_correction_topk", 3)),
                "corridor": top1_corridor_diag,
                "sentence": top1_sentence_diag,
            },
            "run_light_rerank": run_light_rerank_diag,
            "run_selection_mode": (
                "semantic_hybrid"
                if (stable_seed_selection and semantic_diag["enabled"] and query_vec is not None)
                else ("legacy_structural" if stable_seed_selection else "first_run")
            ),
            "run_pool": [
                {
                    "run_id": int(r.get("run_id", 0)),
                    "hybrid_run_score": float(r.get("hybrid_run_score", 0.0) or 0.0),
                    "surrogate_loss": float(r.get("surrogate_loss", 0.0) or 0.0),
                    "global_loss": float(r.get("global_loss", 0.0) or 0.0),
                    "num_graph_candidates": int(len(r.get("graph_candidates", []) or [])),
                    "num_semantic_candidates": int(len(r.get("semantic_candidates", []) or [])),
                    "num_candidates": int(len(r.get("candidates", []) or [])),
                    "num_seeds": int(len(r.get("seeds", []) or [])),
                    "run_score_components": r.get("run_score_components", {}) or {},
                }
                for r in run_results
            ],
            "shortlisted_run_ids": [int(r.get("run_id", 0)) for r in shortlisted_runs],
            "embedding_rerank": embedding_diag,
            **({"score_component_trace": score_component_trace} if score_component_trace is not None else {}),
            "stagewise_loss_funnel_enabled": bool(getattr(cfg, "stagewise_loss_funnel_enabled", True)),
            "final_top_slice_reorder_enabled": bool(getattr(cfg, "final_top_slice_reorder_enabled", False)),
            "final_top_slice_reorder_topk": int(getattr(cfg, "final_top_slice_reorder_topk", 4)),
            "answer_support_pinning_enabled": bool(getattr(cfg, "answer_support_pinning_enabled", False)),
            "answer_support_pinning_min": int(getattr(cfg, "answer_support_pinning_min", 1)),
            "oracle_support_injection_enabled": bool(getattr(cfg, "oracle_support_injection_enabled", False)),
            "stable_seed_selection": stable_seed_selection,
            "trim_enabled": bool(enable_trim and getattr(cfg, "trim_on", True)),
            "anchor_diag_topn": int(diag_topn),
            "anchor_diag_store_full_scores": bool(diag_full),
            "latency_breakdown_ms": stage_ms,
        },
        latency_ms=latency_ms,
    )


@register_method("effirag")
def run_effirag(sample, cfg):
    return run_graphrag_core(
        sample=sample,
        cfg=cfg,
        method_name="effirag",
        stable_seed_selection=True,
        enable_trim=True,
    )
