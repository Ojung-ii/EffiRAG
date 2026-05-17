from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from effirag.canonical_objective import resolve_unified_copy_span_mode as _resolve_unified_copy_span_mode
from effirag.grouped_profiles import INSTRUCTION_INTERFACE_CONTRACT, REFERENCE_LABEL
from effirag.profiles import (
    COPY_SPAN_DISABLED_MODULES,
    COPY_SPAN_RENDER_WEIGHTS,
    COPY_SPAN_RETRIEVAL_WEIGHTS,
)


DATASET_ORDER = ("hotpotqa", "2wikimultihopqa", "musique", "popqa")
STRICT_UNIFIED_VARIANT_NAME = "phase6_strict_unified_v1"
STRICT_UNIFIED_PROFILE_FAMILY = "copy_span_instruction_unified"

COPY_SPAN_INSTRUCTION_UNIFIED_LARGE = "copy_span_instruction_unified_large"
COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM = "copy_span_instruction_unified_medium"
COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT = "copy_span_instruction_unified_compact"
COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP = "copy_span_instruction_unified_large_lightsep"
COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL = "copy_span_instruction_unified_large_reorder_all"
COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL = "copy_span_instruction_unified_large_lightsep_reorder_all"
COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL = "copy_span_instruction_unified_medium_lightsep_reorder_all"
COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V1 = "copy_span_instruction_unified_dynamic_compact_v1"
COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V2 = "copy_span_instruction_unified_dynamic_compact_v2"
COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_CONTEXTUAL_COMPACT_V3 = "copy_span_instruction_unified_dynamic_contextual_compact_v3"
COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_V1 = "copy_span_instruction_unified_candidate_recall_boost_v1"
COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DYNAMIC_V1 = (
    "copy_span_instruction_unified_candidate_recall_boost_dynamic_v1"
)
COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DENSITY_RERANK_V1 = (
    "copy_span_instruction_unified_candidate_recall_boost_density_rerank_v1"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1 = "copy_span_instruction_unified_gl_rcedr_v1"
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_DYNAMIC_CONTROL = (
    "copy_span_instruction_unified_gl_rcedr_no_dynamic_control"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_STABILITY = "copy_span_instruction_unified_gl_rcedr_no_stability"
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_BRIDGE_PATH = "copy_span_instruction_unified_gl_rcedr_no_bridge_path"
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_DENSITY_FIRST_ABLATION = (
    "copy_span_instruction_unified_gl_rcedr_density_first_ablation"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT = (
    "copy_span_instruction_unified_gl_rcedr_v1_sentence_contract"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP = (
    "copy_span_instruction_unified_gl_rcedr_v1_sentence_contract_no_item_cap"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON = (
    "copy_span_instruction_unified_gl_rcedr_v1_sentence_contract_metadata_on"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40 = (
    "copy_span_instruction_unified_gl_rcedr_v1_sentence_contract_span40"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT = (
    "copy_span_instruction_unified_gl_rcedr_v1_support_span_contract"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40 = (
    "copy_span_instruction_unified_gl_rcedr_v1_support_span_contract_span40"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP = (
    "copy_span_instruction_unified_gl_rcedr_v1_support_span_contract_no_cap"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL = (
    "copy_span_instruction_unified_gl_rcedr_v1_support_span_contract_no_bridge_signal"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL = (
    "copy_span_instruction_unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN = (
    "copy_span_instruction_unified_gl_rcedr_v1_adaptive_support_span"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE = (
    "copy_span_instruction_unified_gl_rcedr_v1_adaptive_support_span_weak_bridge"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE = (
    "copy_span_instruction_unified_gl_rcedr_v1_adaptive_support_span_no_bridge"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY = (
    "copy_span_instruction_unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40 = (
    "copy_span_instruction_unified_gl_rcedr_v1_adaptive_support_span_span40"
)
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1 = "copy_span_instruction_unified_acr_v1"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ANSWERABILITY = "copy_span_instruction_unified_acr_v1_no_answerability"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_STRUCTURE = "copy_span_instruction_unified_acr_v1_no_structure"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_NOISE_PENALTY = "copy_span_instruction_unified_acr_v1_no_noise_penalty"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_COST = "copy_span_instruction_unified_acr_v1_no_cost"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ORDERING = "copy_span_instruction_unified_acr_v1_no_ordering"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_BEAM3 = "copy_span_instruction_unified_acr_v1_beam3"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11 = "copy_span_instruction_unified_acr_v11"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_STRUCTURE = "copy_span_instruction_unified_acr_v11_no_structure"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_COST = "copy_span_instruction_unified_acr_v11_no_cost"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_ORDERED = "copy_span_instruction_unified_acr_v11_ordered"
COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_BEAM3 = "copy_span_instruction_unified_acr_v11_beam3"
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2 = "copy_span_instruction_unified_gl_rcedr_v2"
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_ADAPTIVE_BRIDGE = (
    "copy_span_instruction_unified_gl_rcedr_v2_no_adaptive_bridge"
)
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_COST = "copy_span_instruction_unified_gl_rcedr_v2_no_cost"
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_REDUNDANCY = "copy_span_instruction_unified_gl_rcedr_v2_no_redundancy"
COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_DENSITY_FIRST = "copy_span_instruction_unified_gl_rcedr_v2_density_first"

UNIFIED_CLEAN_PROFILE_NAMES = (
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM,
    COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT,
)

UNIFIED_ENHANCED_PROFILE_NAMES = (
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DENSITY_RERANK_V1,
)

UNIFIED_DYNAMIC_PROFILE_NAMES = (
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V2,
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_CONTEXTUAL_COMPACT_V3,
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DYNAMIC_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ANSWERABILITY,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_STRUCTURE,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_NOISE_PENALTY,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_COST,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ORDERING,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_BEAM3,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_STRUCTURE,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_COST,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_ORDERED,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_BEAM3,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_DYNAMIC_CONTROL,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_STABILITY,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_BRIDGE_PATH,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_DENSITY_FIRST_ABLATION,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_ADAPTIVE_BRIDGE,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_COST,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_REDUNDANCY,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_DENSITY_FIRST,
)

UNIFIED_PROFILE_NAMES = UNIFIED_CLEAN_PROFILE_NAMES + UNIFIED_ENHANCED_PROFILE_NAMES + UNIFIED_DYNAMIC_PROFILE_NAMES

UNIFIED_PROFILE_ALIASES = {
    "large": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE,
    "unified_large": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE: COPY_SPAN_INSTRUCTION_UNIFIED_LARGE,
    "medium": COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM,
    "unified_medium": COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM: COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM,
    "compact": COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT,
    "unified_compact": COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT,
    COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT: COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT,
    "large_lightsep": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP,
    "unified_large_lightsep": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP: COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP,
    "large_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL,
    "unified_large_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL: COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL,
    "large_lightsep_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL,
    "unified_large_lightsep_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL: COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL,
    "medium_lightsep_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL,
    "unified_medium_lightsep_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL: COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL,
    "dynamic_compact_v1": COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V1,
    "unified_dynamic_compact_v1": COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V1: COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V1,
    "dynamic_compact_v2": COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V2,
    "unified_dynamic_compact_v2": COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V2,
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V2: COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V2,
    "dynamic_contextual_compact_v3": COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_CONTEXTUAL_COMPACT_V3,
    "unified_dynamic_contextual_compact_v3": COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_CONTEXTUAL_COMPACT_V3,
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_CONTEXTUAL_COMPACT_V3: COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_CONTEXTUAL_COMPACT_V3,
    "candidate_recall_boost_v1": COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_V1,
    "unified_candidate_recall_boost_v1": COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_V1: COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_V1,
    "candidate_recall_boost_dynamic_v1": COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DYNAMIC_V1,
    "unified_candidate_recall_boost_dynamic_v1": COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DYNAMIC_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DYNAMIC_V1: COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DYNAMIC_V1,
    "candidate_recall_boost_density_rerank_v1": COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DENSITY_RERANK_V1,
    "unified_candidate_recall_boost_density_rerank_v1": COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DENSITY_RERANK_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DENSITY_RERANK_V1: COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DENSITY_RERANK_V1,
    "gl_rcedr_v1": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1,
    "unified_gl_rcedr_v1": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1,
    "gl_rcedr_v1_sentence_contract": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT,
    "unified_gl_rcedr_v1_sentence_contract": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT,
    "gl_rcedr_v1_sentence_contract_no_item_cap": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP,
    "unified_gl_rcedr_v1_sentence_contract_no_item_cap": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP,
    "gl_rcedr_v1_sentence_contract_metadata_on": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON,
    "unified_gl_rcedr_v1_sentence_contract_metadata_on": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON,
    "gl_rcedr_v1_sentence_contract_span40": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40,
    "unified_gl_rcedr_v1_sentence_contract_span40": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40,
    "gl_rcedr_v1_support_span_contract": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT,
    "unified_gl_rcedr_v1_support_span_contract": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT,
    "gl_rcedr_v1_support_span_contract_span40": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40,
    "unified_gl_rcedr_v1_support_span_contract_span40": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40,
    "gl_rcedr_v1_support_span_contract_no_cap": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP,
    "unified_gl_rcedr_v1_support_span_contract_no_cap": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP,
    "gl_rcedr_v1_support_span_contract_no_bridge_signal": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL,
    "unified_gl_rcedr_v1_support_span_contract_no_bridge_signal": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL,
    "gl_rcedr_v1_support_span_contract_no_query_entity_signal": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL,
    "unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL,
    "gl_rcedr_v1_adaptive_support_span": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN,
    "unified_gl_rcedr_v1_adaptive_support_span": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN,
    "gl_rcedr_v1_adaptive_support_span_weak_bridge": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE,
    "unified_gl_rcedr_v1_adaptive_support_span_weak_bridge": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE,
    "gl_rcedr_v1_adaptive_support_span_no_bridge": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE,
    "unified_gl_rcedr_v1_adaptive_support_span_no_bridge": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE,
    "gl_rcedr_v1_adaptive_support_span_no_length_penalty": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY,
    "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY,
    "gl_rcedr_v1_adaptive_support_span_span40": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40,
    "unified_gl_rcedr_v1_adaptive_support_span_span40": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40,
    "acr_v1": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1,
    "unified_acr_v1": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1,
    "acr_v1_no_answerability": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ANSWERABILITY,
    "unified_acr_v1_no_answerability": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ANSWERABILITY,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ANSWERABILITY: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ANSWERABILITY,
    "acr_v1_no_structure": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_STRUCTURE,
    "unified_acr_v1_no_structure": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_STRUCTURE,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_STRUCTURE: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_STRUCTURE,
    "acr_v1_no_noise_penalty": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_NOISE_PENALTY,
    "unified_acr_v1_no_noise_penalty": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_NOISE_PENALTY,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_NOISE_PENALTY: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_NOISE_PENALTY,
    "acr_v1_no_cost": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_COST,
    "unified_acr_v1_no_cost": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_COST,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_COST: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_COST,
    "acr_v1_no_ordering": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ORDERING,
    "unified_acr_v1_no_ordering": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ORDERING,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ORDERING: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ORDERING,
    "acr_v1_beam3": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_BEAM3,
    "unified_acr_v1_beam3": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_BEAM3,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_BEAM3: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_BEAM3,
    "acr_v11": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11,
    "unified_acr_v11": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11,
    "acr_v11_no_structure": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_STRUCTURE,
    "unified_acr_v11_no_structure": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_STRUCTURE,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_STRUCTURE: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_STRUCTURE,
    "acr_v11_no_cost": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_COST,
    "unified_acr_v11_no_cost": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_COST,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_COST: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_COST,
    "acr_v11_ordered": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_ORDERED,
    "unified_acr_v11_ordered": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_ORDERED,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_ORDERED: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_ORDERED,
    "acr_v11_beam3": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_BEAM3,
    "unified_acr_v11_beam3": COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_BEAM3,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_BEAM3: COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_BEAM3,
    "gl_rcedr_no_dynamic_control": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_DYNAMIC_CONTROL,
    "unified_gl_rcedr_no_dynamic_control": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_DYNAMIC_CONTROL,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_DYNAMIC_CONTROL: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_DYNAMIC_CONTROL,
    "gl_rcedr_no_stability": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_STABILITY,
    "unified_gl_rcedr_no_stability": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_STABILITY,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_STABILITY: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_STABILITY,
    "gl_rcedr_no_bridge_path": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_BRIDGE_PATH,
    "unified_gl_rcedr_no_bridge_path": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_BRIDGE_PATH,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_BRIDGE_PATH: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_BRIDGE_PATH,
    "gl_rcedr_density_first_ablation": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_DENSITY_FIRST_ABLATION,
    "unified_gl_rcedr_density_first_ablation": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_DENSITY_FIRST_ABLATION,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_DENSITY_FIRST_ABLATION: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_DENSITY_FIRST_ABLATION,
    "gl_rcedr_v2": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2,
    "unified_gl_rcedr_v2": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2,
    "gl_rcedr_v2_no_adaptive_bridge": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_ADAPTIVE_BRIDGE,
    "unified_gl_rcedr_v2_no_adaptive_bridge": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_ADAPTIVE_BRIDGE,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_ADAPTIVE_BRIDGE: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_ADAPTIVE_BRIDGE,
    "gl_rcedr_v2_no_cost": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_COST,
    "unified_gl_rcedr_v2_no_cost": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_COST,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_COST: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_COST,
    "gl_rcedr_v2_no_redundancy": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_REDUNDANCY,
    "unified_gl_rcedr_v2_no_redundancy": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_REDUNDANCY,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_REDUNDANCY: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_REDUNDANCY,
    "gl_rcedr_v2_density_first": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_DENSITY_FIRST,
    "unified_gl_rcedr_v2_density_first": COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_DENSITY_FIRST,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_DENSITY_FIRST: COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_DENSITY_FIRST,
}

UNIFIED_PROFILE_DIRS = {
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE: "unified_large",
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM: "unified_medium",
    COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT: "unified_compact",
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP: "unified_large_lightsep",
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL: "unified_large_reorder_all",
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL: "unified_large_lightsep_reorder_all",
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL: "unified_medium_lightsep_reorder_all",
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V1: "unified_dynamic_compact_v1",
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V2: "unified_dynamic_compact_v2",
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_CONTEXTUAL_COMPACT_V3: "unified_dynamic_contextual_compact_v3",
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_V1: "unified_candidate_recall_boost_v1",
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DYNAMIC_V1: "unified_candidate_recall_boost_dynamic_v1",
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DENSITY_RERANK_V1: "unified_candidate_recall_boost_density_rerank_v1",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1: "unified_gl_rcedr_v1",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT: "unified_gl_rcedr_v1_sentence_contract",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP: "unified_gl_rcedr_v1_sentence_contract_no_item_cap",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON: "unified_gl_rcedr_v1_sentence_contract_metadata_on",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40: "unified_gl_rcedr_v1_sentence_contract_span40",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT: "unified_gl_rcedr_v1_support_span_contract",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40: "unified_gl_rcedr_v1_support_span_contract_span40",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP: "unified_gl_rcedr_v1_support_span_contract_no_cap",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL: "unified_gl_rcedr_v1_support_span_contract_no_bridge_signal",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL: "unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN: "unified_gl_rcedr_v1_adaptive_support_span",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE: "unified_gl_rcedr_v1_adaptive_support_span_weak_bridge",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE: "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY: "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40: "unified_gl_rcedr_v1_adaptive_support_span_span40",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1: "unified_acr_v1",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ANSWERABILITY: "unified_acr_v1_no_answerability",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_STRUCTURE: "unified_acr_v1_no_structure",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_NOISE_PENALTY: "unified_acr_v1_no_noise_penalty",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_COST: "unified_acr_v1_no_cost",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ORDERING: "unified_acr_v1_no_ordering",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_BEAM3: "unified_acr_v1_beam3",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11: "unified_acr_v11",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_STRUCTURE: "unified_acr_v11_no_structure",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_COST: "unified_acr_v11_no_cost",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_ORDERED: "unified_acr_v11_ordered",
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_BEAM3: "unified_acr_v11_beam3",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_DYNAMIC_CONTROL: "unified_gl_rcedr_no_dynamic_control",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_STABILITY: "unified_gl_rcedr_no_stability",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_BRIDGE_PATH: "unified_gl_rcedr_no_bridge_path",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_DENSITY_FIRST_ABLATION: "unified_gl_rcedr_density_first_ablation",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2: "unified_gl_rcedr_v2",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_ADAPTIVE_BRIDGE: "unified_gl_rcedr_v2_no_adaptive_bridge",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_COST: "unified_gl_rcedr_v2_no_cost",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_REDUNDANCY: "unified_gl_rcedr_v2_no_redundancy",
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_DENSITY_FIRST: "unified_gl_rcedr_v2_density_first",
}

UNIFIED_BUDGETS = {
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM: {
        "seed_topn": 36,
        "run_topn": 18,
        "render_topn": 18,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT: {
        "seed_topn": 30,
        "run_topn": 15,
        "render_topn": 15,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL: {
        "seed_topn": 36,
        "run_topn": 18,
        "render_topn": 18,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V1: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V2: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_CONTEXTUAL_COMPACT_V3: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_V1: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DYNAMIC_V1: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DENSITY_RERANK_V1: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ANSWERABILITY: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_STRUCTURE: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_NOISE_PENALTY: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_COST: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ORDERING: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_BEAM3: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_STRUCTURE: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_COST: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_ORDERED: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_BEAM3: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_DYNAMIC_CONTROL: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_STABILITY: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_BRIDGE_PATH: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_DENSITY_FIRST_ABLATION: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_ADAPTIVE_BRIDGE: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_COST: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_REDUNDANCY: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_DENSITY_FIRST: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
}

UNIFIED_BUDGET_CONFIG_KEYS = (
    "semantic_topn_entity",
    "semantic_topn_chunk",
    "graph_reserve_topn",
)

UNIFIED_RENDERING_LOCK = {
    "answer_support_pinning_enabled": False,
    "answer_support_pinning_min": 1,
    "final_top_slice_reorder_enabled": False,
    "final_top_slice_reorder_topk": 4,
    "corridor_answer_preserve_guarded_hotpot_enabled": False,
    "oracle_support_injection_enabled": False,
}

UNIFIED_DEFAULT_INTERFACE_LOCK = {
    "order_strategy": "score",
    "prompt_variant": "default",
    "prompt_variant_label": "default",
    "render_variant": "score_ordered_corridor_aware_flat",
    "added_instruction": "none",
}

UNIFIED_LIGHTSEP_INTERFACE_LOCK = dict(INSTRUCTION_INTERFACE_CONTRACT)

UNIFIED_METHOD_LOCK = {
    "retrieval_objective_mode": "baseline",
    "canonical_variant_name": STRICT_UNIFIED_VARIANT_NAME,
    "shared_budget_profile": "off",
    "max_anchors": 4,
    "samples_per_anchor": 4,
    "hybrid_anchor_recall_enabled": False,
    "bridge_candidate_induction_enabled": False,
    "path_candidate_expansion_enabled": False,
    "anchor_expansion_enabled": False,
    "candidate_diversity_enabled": False,
    "entity_diversity_enabled": False,
    "source_diversity_enabled": False,
    "max_bridge_candidates": 24,
    "max_path_candidates": 24,
    "max_anchor_expansion_hops": 1,
    "max_expanded_candidates": 64,
    "candidate_dedup_enabled": True,
    "dataset_specific_branch_enabled": False,
    "role_aware_chunk_scoring_enabled": False,
    "coverage_selection_enabled": False,
    "corridor_compact_shaping_enabled": False,
    "corridor_answer_preserve_enabled": False,
    "corridor_bridge_purity_shaping_enabled": False,
    "corridor_path_preserve_enabled": False,
    "corridor_path_preserve_compact_enabled": False,
    "corridor_path_preserve_guarded_enabled": False,
    "corridor_path_preserve_compact_lite_enabled": False,
    "corridor_answer_preserve_confidence_gated_enabled": False,
    "top1_correction_enabled": False,
    "run_light_rerank_enabled": False,
    "chunk_grounding_enabled": False,
    "dynamic_compact_selection_enabled": False,
    "coverage_gain_enabled": False,
    "redundancy_penalty_enabled": False,
    "bridge_preserve_enabled": False,
    "path_preserve_enabled": False,
    "adaptive_stop_enabled": False,
    "max_render_topn": 24,
    "min_render_topn": 6,
    "target_prompt_tokens": 600,
    "max_prompt_tokens": 700,
    "coverage_gain_threshold": 0.05,
    "bridge_score_threshold": 0.35,
    "redundancy_threshold": 0.62,
    "marginal_gain_threshold": 0.08,
    "evidence_density_rerank_enabled": False,
    "bridge_path_utility_enabled": False,
    "token_cost_penalty_enabled": False,
    "density_budget_awareness_enabled": False,
    "density_semantic_weight": 0.42,
    "density_bridge_path_weight": 0.24,
    "density_coverage_weight": 0.24,
    "density_redundancy_weight": 0.18,
    "density_token_cost_weight": 0.22,
    "max_selected_candidates": 24,
    "max_rendered_candidates": 24,
    "max_rendered_tokens": 340,
    "gl_rcedr_enabled": False,
    "gl_rcedr_dynamic_control_enabled": True,
    "gl_rcedr_stability_enabled": True,
    "gl_rcedr_bridge_path_enabled": True,
    "gl_rcedr_density_first_ablation_enabled": False,
    "gl_rcedr_view_count": 6,
    "gl_rcedr_seed_topk": 6,
    "gl_rcedr_recall_weight": 0.34,
    "gl_rcedr_bridge_path_weight": 0.22,
    "gl_rcedr_diversity_weight": 0.16,
    "gl_rcedr_stability_weight": 0.12,
    "gl_rcedr_redundancy_weight": 0.16,
    "gl_rcedr_cost_weight": 0.12,
    "gl_rcedr_max_selected_candidates": 24,
    "gl_rcedr_max_rendered_candidates": 24,
    "gl_rcedr_max_rendered_tokens": 360,
    "gl_rcedr_core_preserve_threshold": 0.56,
    "gl_rcedr_bridge_preserve_threshold": 0.46,
    "gl_rcedr_coverage_preserve_threshold": 0.38,
    "unified_marginal_utility_enabled": False,
    "gl_rcedr_v2_adaptive_bridge_enabled": True,
    "gl_rcedr_v2_cost_enabled": True,
    "gl_rcedr_v2_redundancy_enabled": True,
    "gl_rcedr_v2_density_first_enabled": False,
    "gl_rcedr_v2_evidence_gain_weight": 0.52,
    "gl_rcedr_v2_bridge_gain_weight": 0.28,
    "gl_rcedr_v2_redundancy_weight": 0.22,
    "gl_rcedr_v2_cost_weight": 0.14,
    "gl_rcedr_v2_seed_stability_weight": 0.15,
    "gl_rcedr_v2_seed_diversity_weight": 0.15,
    "gl_rcedr_v2_bridge_lambda_base": 1.0,
    "gl_rcedr_v2_bridge_lambda_alpha": 0.30,
    "gl_rcedr_v2_bridge_lambda_min": 0.85,
    "gl_rcedr_v2_bridge_lambda_max": 1.35,
    "gl_rcedr_v2_max_selected_candidates": 24,
    "gl_rcedr_v2_max_rendered_candidates": 24,
    "gl_rcedr_v2_max_rendered_tokens": 360,
    "gl_rcedr_v2_core_preserve_threshold": 0.56,
    "gl_rcedr_v2_bridge_preserve_threshold": 0.46,
    "selector_aware_render_enabled": False,
    "render_selected_only": False,
    "render_include_neighbor_sentences": False,
    "render_include_corridor_headers": True,
    "render_include_source_titles": "full",
    "render_include_metadata": "full",
    "render_deduplicate_selected_text": False,
    "render_deduplicate_context_text": False,
    "render_enforce_actual_prompt_budget": False,
    "render_selected_centered": False,
    "render_contextual_expansion_enabled": False,
    "render_conditional_neighbor_sentences": False,
    "render_bridge_context_enabled": False,
    "render_path_context_enabled": False,
    "max_neighbors_per_selected": 1,
    "max_context_sentences_per_selected": 1,
    "max_bridge_context_sentences": 2,
    "max_path_context_sentences": 2,
    "sentence_contract_render_enabled": False,
    "sentence_contract_max_item_tokens": None,
    "sentence_contract_metadata_pruning": True,
    "sentence_contract_chunk_expansion_allowed": False,
    "sentence_contract_preserve_selected_items": True,
    "sentence_contract_minimal_span_fallback": True,
    "sentence_contract_log_diagnostics": True,
    "support_span_contract_enabled": False,
    "support_span_max_item_tokens": 48,
    "support_span_metadata_pruning": True,
    "support_span_use_query_entity_signal": True,
    "support_span_use_anchor_entity_signal": True,
    "support_span_use_bridge_signal": True,
    "support_span_use_view_stability_signal": True,
    "support_span_length_penalty_enabled": True,
    "support_span_adjacent_sentence_enabled": True,
    "support_span_adjacent_sentence_max_count": 1,
    "support_span_chunk_expansion_allowed": False,
    "support_span_preserve_selected_items": True,
    "support_span_log_diagnostics": True,
    "adaptive_support_span_enabled": False,
    "adaptive_support_span_hard_cap_enabled": False,
    "adaptive_support_span_max_item_tokens": 40,
    "adaptive_support_span_soft_length_penalty_enabled": True,
    "adaptive_support_span_use_query_entity_signal": True,
    "adaptive_support_span_use_anchor_entity_signal": True,
    "adaptive_support_span_use_bridge_signal": True,
    "adaptive_support_span_bridge_mode": "conditional",
    "adaptive_support_span_metadata_pruning": True,
    "adaptive_support_span_preserve_selected_items": True,
    "adaptive_support_span_log_diagnostics": True,
    "answerability_selection_enabled": False,
    "answerability_selection_mode": "greedy",
    "answerability_selection_beam_size": 3,
    "answerability_selection_max_atoms": 8,
    "answerability_selection_max_tokens": 220,
    "answerability_selection_hard_token_budget_enabled": True,
    "answerability_selection_use_answerability": True,
    "answerability_selection_use_structure": True,
    "answerability_selection_use_noise_penalty": True,
    "answerability_selection_use_cost_penalty": True,
    "answerability_selection_ordering_enabled": True,
    "answerability_selection_atom_span_max_sentences": 2,
    "answerability_selection_structure_weight": 0.65,
    "answerability_selection_noise_weight": 0.55,
    "answerability_selection_cost_weight": 0.35,
    "answerability_selection_length_penalty_weight": 0.04,
    "answerability_selection_log_diagnostics": True,
    **UNIFIED_RENDERING_LOCK,
}

UNIFIED_PROFILE_INTERFACE_LOCKS = {
    # Preserve the generated Phase-6 clean profile behavior that already uses
    # the copy-span light-separator interface inherited from the SOTA source pack.
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL: UNIFIED_DEFAULT_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V1: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V2: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_CONTEXTUAL_COMPACT_V3: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_V1: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DYNAMIC_V1: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DENSITY_RERANK_V1: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ANSWERABILITY: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_STRUCTURE: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_NOISE_PENALTY: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_COST: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ORDERING: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_BEAM3: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_STRUCTURE: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_COST: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_ORDERED: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_BEAM3: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_DYNAMIC_CONTROL: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_STABILITY: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_BRIDGE_PATH: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_DENSITY_FIRST_ABLATION: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_ADAPTIVE_BRIDGE: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_COST: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_REDUNDANCY: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_DENSITY_FIRST: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
}

UNIFIED_PROFILE_METHOD_OVERRIDES = {
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V1: {
        "dynamic_compact_selection_enabled": True,
        "coverage_gain_enabled": True,
        "redundancy_penalty_enabled": True,
        "bridge_preserve_enabled": True,
        "path_preserve_enabled": True,
        "adaptive_stop_enabled": True,
        "max_render_topn": 24,
        "min_render_topn": 6,
        "target_prompt_tokens": 600,
        "max_prompt_tokens": 700,
        "coverage_gain_threshold": 0.05,
        "bridge_score_threshold": 0.35,
        "redundancy_threshold": 0.62,
        "marginal_gain_threshold": 0.08,
        # Let the dynamic selector see the large retrieved candidate pool.
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_COMPACT_V2: {
        "dynamic_compact_selection_enabled": True,
        "coverage_gain_enabled": True,
        "redundancy_penalty_enabled": True,
        "bridge_preserve_enabled": True,
        "path_preserve_enabled": True,
        "adaptive_stop_enabled": True,
        "max_render_topn": 24,
        "min_render_topn": 6,
        "target_prompt_tokens": 550,
        "max_prompt_tokens": 650,
        "coverage_gain_threshold": 0.05,
        "bridge_score_threshold": 0.35,
        "redundancy_threshold": 0.62,
        "marginal_gain_threshold": 0.08,
        "selector_aware_render_enabled": True,
        "render_selected_only": True,
        "render_include_neighbor_sentences": False,
        "render_include_corridor_headers": False,
        "render_include_source_titles": "minimal",
        "render_include_metadata": "minimal",
        "render_deduplicate_selected_text": True,
        "render_enforce_actual_prompt_budget": True,
        # Keep retrieval/selector candidate exposure identical to v1.
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_DYNAMIC_CONTEXTUAL_COMPACT_V3: {
        "dynamic_compact_selection_enabled": True,
        "coverage_gain_enabled": True,
        "redundancy_penalty_enabled": True,
        "bridge_preserve_enabled": True,
        "path_preserve_enabled": True,
        "adaptive_stop_enabled": True,
        "max_render_topn": 24,
        "min_render_topn": 8,
        "target_prompt_tokens": 750,
        "max_prompt_tokens": 850,
        "coverage_gain_threshold": 0.05,
        "bridge_score_threshold": 0.35,
        "redundancy_threshold": 0.62,
        "marginal_gain_threshold": 0.08,
        "selector_aware_render_enabled": True,
        "render_selected_only": False,
        "render_selected_centered": True,
        "render_contextual_expansion_enabled": True,
        "render_conditional_neighbor_sentences": True,
        "render_bridge_context_enabled": True,
        "render_path_context_enabled": True,
        "render_include_neighbor_sentences": True,
        "render_include_corridor_headers": False,
        "render_include_source_titles": "minimal",
        "render_include_metadata": "minimal",
        "render_deduplicate_selected_text": True,
        "render_deduplicate_context_text": True,
        "render_enforce_actual_prompt_budget": True,
        "max_neighbors_per_selected": 1,
        "max_context_sentences_per_selected": 1,
        "max_bridge_context_sentences": 2,
        "max_path_context_sentences": 2,
        # Keep retrieval/selector candidate exposure fixed.
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_V1: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DYNAMIC_V1: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "dynamic_compact_selection_enabled": True,
        "coverage_gain_enabled": True,
        "redundancy_penalty_enabled": True,
        "bridge_preserve_enabled": True,
        "path_preserve_enabled": True,
        "adaptive_stop_enabled": True,
        "max_render_topn": 24,
        "min_render_topn": 6,
        "target_prompt_tokens": 600,
        "max_prompt_tokens": 700,
        "coverage_gain_threshold": 0.05,
        "bridge_score_threshold": 0.35,
        "redundancy_threshold": 0.62,
        "marginal_gain_threshold": 0.08,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_CANDIDATE_RECALL_BOOST_DENSITY_RERANK_V1: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "evidence_density_rerank_enabled": True,
        "coverage_gain_enabled": True,
        "redundancy_penalty_enabled": True,
        "bridge_path_utility_enabled": True,
        "token_cost_penalty_enabled": True,
        "density_budget_awareness_enabled": True,
        "density_semantic_weight": 0.42,
        "density_bridge_path_weight": 0.24,
        "density_coverage_weight": 0.24,
        "density_redundancy_weight": 0.18,
        "density_token_cost_weight": 0.22,
        "max_selected_candidates": 24,
        "max_rendered_candidates": 24,
        "max_rendered_tokens": 340,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "gl_rcedr_enabled": True,
        "gl_rcedr_dynamic_control_enabled": True,
        "gl_rcedr_stability_enabled": True,
        "gl_rcedr_bridge_path_enabled": True,
        "gl_rcedr_density_first_ablation_enabled": False,
        "gl_rcedr_view_count": 6,
        "gl_rcedr_seed_topk": 6,
        "gl_rcedr_recall_weight": 0.34,
        "gl_rcedr_bridge_path_weight": 0.22,
        "gl_rcedr_diversity_weight": 0.16,
        "gl_rcedr_stability_weight": 0.12,
        "gl_rcedr_redundancy_weight": 0.16,
        "gl_rcedr_cost_weight": 0.12,
        "gl_rcedr_max_selected_candidates": 24,
        "gl_rcedr_max_rendered_candidates": 24,
        "gl_rcedr_max_rendered_tokens": 360,
        "gl_rcedr_core_preserve_threshold": 0.56,
        "gl_rcedr_bridge_preserve_threshold": 0.46,
        "gl_rcedr_coverage_preserve_threshold": 0.38,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_DYNAMIC_CONTROL: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "gl_rcedr_enabled": True,
        "gl_rcedr_dynamic_control_enabled": False,
        "gl_rcedr_stability_enabled": True,
        "gl_rcedr_bridge_path_enabled": True,
        "gl_rcedr_density_first_ablation_enabled": False,
        "gl_rcedr_view_count": 6,
        "gl_rcedr_seed_topk": 6,
        "gl_rcedr_recall_weight": 0.34,
        "gl_rcedr_bridge_path_weight": 0.22,
        "gl_rcedr_diversity_weight": 0.16,
        "gl_rcedr_stability_weight": 0.12,
        "gl_rcedr_redundancy_weight": 0.16,
        "gl_rcedr_cost_weight": 0.12,
        "gl_rcedr_max_selected_candidates": 24,
        "gl_rcedr_max_rendered_candidates": 24,
        "gl_rcedr_max_rendered_tokens": 360,
        "gl_rcedr_core_preserve_threshold": 0.56,
        "gl_rcedr_bridge_preserve_threshold": 0.46,
        "gl_rcedr_coverage_preserve_threshold": 0.38,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_STABILITY: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "gl_rcedr_enabled": True,
        "gl_rcedr_dynamic_control_enabled": True,
        "gl_rcedr_stability_enabled": False,
        "gl_rcedr_bridge_path_enabled": True,
        "gl_rcedr_density_first_ablation_enabled": False,
        "gl_rcedr_view_count": 6,
        "gl_rcedr_seed_topk": 6,
        "gl_rcedr_recall_weight": 0.34,
        "gl_rcedr_bridge_path_weight": 0.22,
        "gl_rcedr_diversity_weight": 0.16,
        "gl_rcedr_stability_weight": 0.12,
        "gl_rcedr_redundancy_weight": 0.16,
        "gl_rcedr_cost_weight": 0.12,
        "gl_rcedr_max_selected_candidates": 24,
        "gl_rcedr_max_rendered_candidates": 24,
        "gl_rcedr_max_rendered_tokens": 360,
        "gl_rcedr_core_preserve_threshold": 0.56,
        "gl_rcedr_bridge_preserve_threshold": 0.46,
        "gl_rcedr_coverage_preserve_threshold": 0.38,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_NO_BRIDGE_PATH: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "gl_rcedr_enabled": True,
        "gl_rcedr_dynamic_control_enabled": True,
        "gl_rcedr_stability_enabled": True,
        "gl_rcedr_bridge_path_enabled": False,
        "gl_rcedr_density_first_ablation_enabled": False,
        "gl_rcedr_view_count": 6,
        "gl_rcedr_seed_topk": 6,
        "gl_rcedr_recall_weight": 0.34,
        "gl_rcedr_bridge_path_weight": 0.22,
        "gl_rcedr_diversity_weight": 0.16,
        "gl_rcedr_stability_weight": 0.12,
        "gl_rcedr_redundancy_weight": 0.16,
        "gl_rcedr_cost_weight": 0.12,
        "gl_rcedr_max_selected_candidates": 24,
        "gl_rcedr_max_rendered_candidates": 24,
        "gl_rcedr_max_rendered_tokens": 360,
        "gl_rcedr_core_preserve_threshold": 0.56,
        "gl_rcedr_bridge_preserve_threshold": 0.46,
        "gl_rcedr_coverage_preserve_threshold": 0.38,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_DENSITY_FIRST_ABLATION: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "gl_rcedr_enabled": True,
        "gl_rcedr_dynamic_control_enabled": True,
        "gl_rcedr_stability_enabled": True,
        "gl_rcedr_bridge_path_enabled": True,
        "gl_rcedr_density_first_ablation_enabled": True,
        "gl_rcedr_view_count": 6,
        "gl_rcedr_seed_topk": 6,
        "gl_rcedr_recall_weight": 0.34,
        "gl_rcedr_bridge_path_weight": 0.22,
        "gl_rcedr_diversity_weight": 0.16,
        "gl_rcedr_stability_weight": 0.12,
        "gl_rcedr_redundancy_weight": 0.16,
        "gl_rcedr_cost_weight": 0.12,
        "gl_rcedr_max_selected_candidates": 24,
        "gl_rcedr_max_rendered_candidates": 24,
        "gl_rcedr_max_rendered_tokens": 360,
        "gl_rcedr_core_preserve_threshold": 0.56,
        "gl_rcedr_bridge_preserve_threshold": 0.46,
        "gl_rcedr_coverage_preserve_threshold": 0.38,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "gl_rcedr_enabled": True,
        "unified_marginal_utility_enabled": True,
        "gl_rcedr_v2_adaptive_bridge_enabled": True,
        "gl_rcedr_v2_cost_enabled": True,
        "gl_rcedr_v2_redundancy_enabled": True,
        "gl_rcedr_v2_density_first_enabled": False,
        "gl_rcedr_v2_evidence_gain_weight": 0.52,
        "gl_rcedr_v2_bridge_gain_weight": 0.28,
        "gl_rcedr_v2_redundancy_weight": 0.22,
        "gl_rcedr_v2_cost_weight": 0.14,
        "gl_rcedr_v2_seed_stability_weight": 0.15,
        "gl_rcedr_v2_seed_diversity_weight": 0.15,
        "gl_rcedr_v2_bridge_lambda_base": 1.0,
        "gl_rcedr_v2_bridge_lambda_alpha": 0.30,
        "gl_rcedr_v2_bridge_lambda_min": 0.85,
        "gl_rcedr_v2_bridge_lambda_max": 1.35,
        "gl_rcedr_v2_max_selected_candidates": 24,
        "gl_rcedr_v2_max_rendered_candidates": 24,
        "gl_rcedr_v2_max_rendered_tokens": 360,
        "gl_rcedr_v2_core_preserve_threshold": 0.56,
        "gl_rcedr_v2_bridge_preserve_threshold": 0.46,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_ADAPTIVE_BRIDGE: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "gl_rcedr_enabled": True,
        "unified_marginal_utility_enabled": True,
        "gl_rcedr_v2_adaptive_bridge_enabled": False,
        "gl_rcedr_v2_cost_enabled": True,
        "gl_rcedr_v2_redundancy_enabled": True,
        "gl_rcedr_v2_density_first_enabled": False,
        "gl_rcedr_v2_evidence_gain_weight": 0.52,
        "gl_rcedr_v2_bridge_gain_weight": 0.28,
        "gl_rcedr_v2_redundancy_weight": 0.22,
        "gl_rcedr_v2_cost_weight": 0.14,
        "gl_rcedr_v2_seed_stability_weight": 0.15,
        "gl_rcedr_v2_seed_diversity_weight": 0.15,
        "gl_rcedr_v2_bridge_lambda_base": 1.0,
        "gl_rcedr_v2_bridge_lambda_alpha": 0.30,
        "gl_rcedr_v2_bridge_lambda_min": 0.85,
        "gl_rcedr_v2_bridge_lambda_max": 1.35,
        "gl_rcedr_v2_max_selected_candidates": 24,
        "gl_rcedr_v2_max_rendered_candidates": 24,
        "gl_rcedr_v2_max_rendered_tokens": 360,
        "gl_rcedr_v2_core_preserve_threshold": 0.56,
        "gl_rcedr_v2_bridge_preserve_threshold": 0.46,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_COST: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "gl_rcedr_enabled": True,
        "unified_marginal_utility_enabled": True,
        "gl_rcedr_v2_adaptive_bridge_enabled": True,
        "gl_rcedr_v2_cost_enabled": False,
        "gl_rcedr_v2_redundancy_enabled": True,
        "gl_rcedr_v2_density_first_enabled": False,
        "gl_rcedr_v2_evidence_gain_weight": 0.52,
        "gl_rcedr_v2_bridge_gain_weight": 0.28,
        "gl_rcedr_v2_redundancy_weight": 0.22,
        "gl_rcedr_v2_cost_weight": 0.14,
        "gl_rcedr_v2_seed_stability_weight": 0.15,
        "gl_rcedr_v2_seed_diversity_weight": 0.15,
        "gl_rcedr_v2_bridge_lambda_base": 1.0,
        "gl_rcedr_v2_bridge_lambda_alpha": 0.30,
        "gl_rcedr_v2_bridge_lambda_min": 0.85,
        "gl_rcedr_v2_bridge_lambda_max": 1.35,
        "gl_rcedr_v2_max_selected_candidates": 24,
        "gl_rcedr_v2_max_rendered_candidates": 24,
        "gl_rcedr_v2_max_rendered_tokens": 360,
        "gl_rcedr_v2_core_preserve_threshold": 0.56,
        "gl_rcedr_v2_bridge_preserve_threshold": 0.46,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_NO_REDUNDANCY: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "gl_rcedr_enabled": True,
        "unified_marginal_utility_enabled": True,
        "gl_rcedr_v2_adaptive_bridge_enabled": True,
        "gl_rcedr_v2_cost_enabled": True,
        "gl_rcedr_v2_redundancy_enabled": False,
        "gl_rcedr_v2_density_first_enabled": False,
        "gl_rcedr_v2_evidence_gain_weight": 0.52,
        "gl_rcedr_v2_bridge_gain_weight": 0.28,
        "gl_rcedr_v2_redundancy_weight": 0.22,
        "gl_rcedr_v2_cost_weight": 0.14,
        "gl_rcedr_v2_seed_stability_weight": 0.15,
        "gl_rcedr_v2_seed_diversity_weight": 0.15,
        "gl_rcedr_v2_bridge_lambda_base": 1.0,
        "gl_rcedr_v2_bridge_lambda_alpha": 0.30,
        "gl_rcedr_v2_bridge_lambda_min": 0.85,
        "gl_rcedr_v2_bridge_lambda_max": 1.35,
        "gl_rcedr_v2_max_selected_candidates": 24,
        "gl_rcedr_v2_max_rendered_candidates": 24,
        "gl_rcedr_v2_max_rendered_tokens": 360,
        "gl_rcedr_v2_core_preserve_threshold": 0.56,
        "gl_rcedr_v2_bridge_preserve_threshold": 0.46,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V2_DENSITY_FIRST: {
        "bridge_candidate_induction_enabled": True,
        "path_candidate_expansion_enabled": True,
        "anchor_expansion_enabled": True,
        "candidate_diversity_enabled": True,
        "entity_diversity_enabled": True,
        "source_diversity_enabled": True,
        "max_bridge_candidates": 24,
        "max_path_candidates": 24,
        "max_anchor_expansion_hops": 1,
        "max_expanded_candidates": 64,
        "candidate_dedup_enabled": True,
        "dataset_specific_branch_enabled": False,
        "gl_rcedr_enabled": True,
        "unified_marginal_utility_enabled": True,
        "gl_rcedr_v2_adaptive_bridge_enabled": True,
        "gl_rcedr_v2_cost_enabled": True,
        "gl_rcedr_v2_redundancy_enabled": True,
        "gl_rcedr_v2_density_first_enabled": True,
        "gl_rcedr_v2_evidence_gain_weight": 0.52,
        "gl_rcedr_v2_bridge_gain_weight": 0.28,
        "gl_rcedr_v2_redundancy_weight": 0.22,
        "gl_rcedr_v2_cost_weight": 0.14,
        "gl_rcedr_v2_seed_stability_weight": 0.15,
        "gl_rcedr_v2_seed_diversity_weight": 0.15,
        "gl_rcedr_v2_bridge_lambda_base": 1.0,
        "gl_rcedr_v2_bridge_lambda_alpha": 0.30,
        "gl_rcedr_v2_bridge_lambda_min": 0.85,
        "gl_rcedr_v2_bridge_lambda_max": 1.35,
        "gl_rcedr_v2_max_selected_candidates": 24,
        "gl_rcedr_v2_max_rendered_candidates": 24,
        "gl_rcedr_v2_max_rendered_tokens": 360,
        "gl_rcedr_v2_core_preserve_threshold": 0.56,
        "gl_rcedr_v2_bridge_preserve_threshold": 0.46,
        "max_context_sentences": 24,
        "max_sentences": 24,
        "top_corridors": 0,
    },
}

_GL_RCEDR_V1_SENTENCE_CONTRACT_BASE = dict(
    UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1]
)
_GL_RCEDR_V1_SENTENCE_CONTRACT_BASE.update(
    {
        "sentence_contract_render_enabled": True,
        "sentence_contract_max_item_tokens": 32,
        "sentence_contract_metadata_pruning": True,
        "sentence_contract_chunk_expansion_allowed": False,
        "sentence_contract_preserve_selected_items": True,
        "sentence_contract_minimal_span_fallback": True,
        "sentence_contract_log_diagnostics": True,
    }
)
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT] = dict(
    _GL_RCEDR_V1_SENTENCE_CONTRACT_BASE
)

_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP = dict(_GL_RCEDR_V1_SENTENCE_CONTRACT_BASE)
_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP["sentence_contract_max_item_tokens"] = None
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP
] = _GL_RCEDR_V1_SENTENCE_CONTRACT_NO_ITEM_CAP

_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON = dict(_GL_RCEDR_V1_SENTENCE_CONTRACT_BASE)
_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON["sentence_contract_metadata_pruning"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON
] = _GL_RCEDR_V1_SENTENCE_CONTRACT_METADATA_ON

_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40 = dict(_GL_RCEDR_V1_SENTENCE_CONTRACT_BASE)
_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40["sentence_contract_max_item_tokens"] = 40
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40
] = _GL_RCEDR_V1_SENTENCE_CONTRACT_SPAN40

_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_BASE = dict(
    UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1]
)
_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_BASE.update(
    {
        "sentence_contract_render_enabled": True,
        "sentence_contract_max_item_tokens": None,
        "sentence_contract_metadata_pruning": True,
        "sentence_contract_chunk_expansion_allowed": False,
        "sentence_contract_preserve_selected_items": True,
        "sentence_contract_minimal_span_fallback": True,
        "sentence_contract_log_diagnostics": True,
        "support_span_contract_enabled": True,
        "support_span_max_item_tokens": 48,
        "support_span_metadata_pruning": True,
        "support_span_use_query_entity_signal": True,
        "support_span_use_anchor_entity_signal": True,
        "support_span_use_bridge_signal": True,
        "support_span_use_view_stability_signal": True,
        "support_span_length_penalty_enabled": True,
        "support_span_adjacent_sentence_enabled": True,
        "support_span_adjacent_sentence_max_count": 1,
        "support_span_chunk_expansion_allowed": False,
        "support_span_preserve_selected_items": True,
        "support_span_log_diagnostics": True,
    }
)
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT] = dict(
    _GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_BASE
)

_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40 = dict(_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_BASE)
_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40["support_span_max_item_tokens"] = 40
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40
] = _GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_SPAN40

_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP = dict(_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_BASE)
_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP["support_span_max_item_tokens"] = None
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP
] = _GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_CAP

_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL = dict(_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_BASE)
_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL["support_span_use_bridge_signal"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL
] = _GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_BRIDGE_SIGNAL

_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL = dict(_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_BASE)
_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL["support_span_use_query_entity_signal"] = False
_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL["support_span_use_anchor_entity_signal"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL
] = _GL_RCEDR_V1_SUPPORT_SPAN_CONTRACT_NO_QUERY_ENTITY_SIGNAL

_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_BASE = dict(
    UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1]
)
_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_BASE.update(
    {
        "sentence_contract_render_enabled": True,
        "sentence_contract_max_item_tokens": None,
        "sentence_contract_metadata_pruning": True,
        "sentence_contract_chunk_expansion_allowed": False,
        "sentence_contract_preserve_selected_items": True,
        "sentence_contract_minimal_span_fallback": True,
        "sentence_contract_log_diagnostics": True,
        "support_span_contract_enabled": False,
        "adaptive_support_span_enabled": True,
        "adaptive_support_span_hard_cap_enabled": False,
        "adaptive_support_span_max_item_tokens": 40,
        "adaptive_support_span_soft_length_penalty_enabled": True,
        "adaptive_support_span_use_query_entity_signal": True,
        "adaptive_support_span_use_anchor_entity_signal": True,
        "adaptive_support_span_use_bridge_signal": True,
        "adaptive_support_span_bridge_mode": "conditional",
        "adaptive_support_span_metadata_pruning": True,
        "adaptive_support_span_preserve_selected_items": True,
        "adaptive_support_span_log_diagnostics": True,
    }
)
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN] = dict(
    _GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_BASE
)

_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE = dict(_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_BASE)
_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE["adaptive_support_span_bridge_mode"] = "weak"
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE
] = _GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_WEAK_BRIDGE

_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE = dict(_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_BASE)
_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE["adaptive_support_span_use_bridge_signal"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE
] = _GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_BRIDGE

_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY = dict(_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_BASE)
_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY["adaptive_support_span_soft_length_penalty_enabled"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY
] = _GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_NO_LENGTH_PENALTY

_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40 = dict(_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_BASE)
_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40["adaptive_support_span_hard_cap_enabled"] = True
_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40["adaptive_support_span_max_item_tokens"] = 40
UNIFIED_PROFILE_METHOD_OVERRIDES[
    COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40
] = _GL_RCEDR_V1_ADAPTIVE_SUPPORT_SPAN_SPAN40

_ACR_V1_BASE = dict(UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_GL_RCEDR_V1])
_ACR_V1_BASE.update(
    {
        "sentence_contract_render_enabled": False,
        "support_span_contract_enabled": False,
        "adaptive_support_span_enabled": False,
        "answerability_selection_enabled": True,
        "answerability_selection_mode": "greedy",
        "answerability_selection_beam_size": 3,
        "answerability_selection_max_atoms": 8,
        "answerability_selection_max_tokens": 220,
        "answerability_selection_hard_token_budget_enabled": True,
        "answerability_selection_use_answerability": True,
        "answerability_selection_use_structure": True,
        "answerability_selection_use_noise_penalty": True,
        "answerability_selection_use_cost_penalty": True,
        "answerability_selection_ordering_enabled": True,
        "answerability_selection_atom_span_max_sentences": 2,
        "answerability_selection_structure_weight": 0.65,
        "answerability_selection_noise_weight": 0.55,
        "answerability_selection_cost_weight": 0.35,
        "answerability_selection_length_penalty_weight": 0.04,
        "answerability_selection_log_diagnostics": True,
    }
)
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1] = dict(_ACR_V1_BASE)

_ACR_V1_NO_ANSWERABILITY = dict(_ACR_V1_BASE)
_ACR_V1_NO_ANSWERABILITY["answerability_selection_use_answerability"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ANSWERABILITY] = _ACR_V1_NO_ANSWERABILITY

_ACR_V1_NO_STRUCTURE = dict(_ACR_V1_BASE)
_ACR_V1_NO_STRUCTURE["answerability_selection_use_structure"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_STRUCTURE] = _ACR_V1_NO_STRUCTURE

_ACR_V1_NO_NOISE = dict(_ACR_V1_BASE)
_ACR_V1_NO_NOISE["answerability_selection_use_noise_penalty"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_NOISE_PENALTY] = _ACR_V1_NO_NOISE

_ACR_V1_NO_COST = dict(_ACR_V1_BASE)
_ACR_V1_NO_COST["answerability_selection_use_cost_penalty"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_COST] = _ACR_V1_NO_COST

_ACR_V1_NO_ORDERING = dict(_ACR_V1_BASE)
_ACR_V1_NO_ORDERING["answerability_selection_ordering_enabled"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_NO_ORDERING] = _ACR_V1_NO_ORDERING

_ACR_V1_BEAM3 = dict(_ACR_V1_BASE)
_ACR_V1_BEAM3["answerability_selection_mode"] = "beam3"
_ACR_V1_BEAM3["answerability_selection_beam_size"] = 3
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V1_BEAM3] = _ACR_V1_BEAM3

_ACR_V11_BASE = dict(_ACR_V1_BASE)
_ACR_V11_BASE.update(
    {
        "answerability_selection_use_noise_penalty": False,
        "answerability_selection_ordering_enabled": False,
    }
)
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11] = dict(_ACR_V11_BASE)

_ACR_V11_NO_STRUCTURE = dict(_ACR_V11_BASE)
_ACR_V11_NO_STRUCTURE["answerability_selection_use_structure"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_STRUCTURE] = _ACR_V11_NO_STRUCTURE

_ACR_V11_NO_COST = dict(_ACR_V11_BASE)
_ACR_V11_NO_COST["answerability_selection_use_cost_penalty"] = False
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_NO_COST] = _ACR_V11_NO_COST

_ACR_V11_ORDERED = dict(_ACR_V11_BASE)
_ACR_V11_ORDERED["answerability_selection_ordering_enabled"] = True
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_ORDERED] = _ACR_V11_ORDERED

_ACR_V11_BEAM3 = dict(_ACR_V11_BASE)
_ACR_V11_BEAM3["answerability_selection_mode"] = "beam3"
_ACR_V11_BEAM3["answerability_selection_beam_size"] = 3
UNIFIED_PROFILE_METHOD_OVERRIDES[COPY_SPAN_INSTRUCTION_UNIFIED_ACR_V11_BEAM3] = _ACR_V11_BEAM3

UNIFIED_PROFILE_RENDERING_OVERRIDES = {
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL: {
        "final_top_slice_reorder_enabled": True,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL: {
        "final_top_slice_reorder_enabled": True,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL: {
        "final_top_slice_reorder_enabled": True,
    },
}

UNIFIED_RUNTIME_DEFAULTS = {
    "precomputed_retrieval_path": "",
    "precomputed_retrieval_strict": False,
}

UNIFIED_INTERFACE_SETTING_KEYS = (
    "order_strategy",
    "prompt_variant",
)

UNIFIED_RENDER_SELECTION_KEYS = (
    "max_context_sentences",
    "max_sentences",
    "top_corridors",
)

UNIFIED_METHOD_SETTING_KEYS = tuple(UNIFIED_METHOD_LOCK.keys()) + UNIFIED_INTERFACE_SETTING_KEYS + UNIFIED_RENDER_SELECTION_KEYS


def normalize_dataset_name(dataset_name: str | None) -> str:
    raw = str(dataset_name or "").strip().lower()
    aliases = {
        "hotpot": "hotpotqa",
        "hotpotqa": "hotpotqa",
        "2wiki": "2wikimultihopqa",
        "2wikimultihopqa": "2wikimultihopqa",
        "wikimultihopqa": "2wikimultihopqa",
        "musique": "musique",
        "popqa": "popqa",
    }
    return aliases.get(raw, raw)


def normalize_unified_profile_name(profile_name: str | None) -> str:
    raw = str(profile_name or COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM).strip().lower()
    normalized = UNIFIED_PROFILE_ALIASES.get(raw)
    if not normalized:
        supported = ", ".join(sorted(UNIFIED_PROFILE_ALIASES))
        raise ValueError(f"Unsupported unified copy-span profile '{profile_name}'. Supported: {supported}")
    return normalized


def unified_profile_dir(profile_name: str | None) -> str:
    return UNIFIED_PROFILE_DIRS[normalize_unified_profile_name(profile_name)]


def resolve_unified_copy_span_mode(dataset_name: str | None = None) -> str:
    return _resolve_unified_copy_span_mode(dataset_name)


def unified_budget(profile_name: str | None) -> Dict[str, int]:
    normalized = normalize_unified_profile_name(profile_name)
    return dict(UNIFIED_BUDGETS[normalized])


def unified_interface_config(profile_name: str | None) -> Dict[str, Any]:
    normalized = normalize_unified_profile_name(profile_name)
    return dict(UNIFIED_PROFILE_INTERFACE_LOCKS[normalized])


def unified_rendering_overrides(profile_name: str | None) -> Dict[str, Any]:
    normalized = normalize_unified_profile_name(profile_name)
    return dict(UNIFIED_PROFILE_RENDERING_OVERRIDES.get(normalized, {}))


def unified_method_overrides(profile_name: str | None) -> Dict[str, Any]:
    normalized = normalize_unified_profile_name(profile_name)
    return dict(UNIFIED_PROFILE_METHOD_OVERRIDES.get(normalized, {}))


def unified_budget_config(profile_name: str | None) -> Dict[str, int]:
    budget = unified_budget(profile_name)
    return {
        "semantic_topn_entity": int(budget["seed_topn"]),
        "semantic_topn_chunk": int(budget["run_topn"]),
        "graph_reserve_topn": int(budget["render_topn"]),
    }


def apply_unified_profile(
    cfg: Mapping[str, Any] | None,
    dataset_name: str | None,
    profile_name: str | None = COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM,
) -> Dict[str, Any]:
    ds = normalize_dataset_name(dataset_name)
    if ds not in DATASET_ORDER:
        supported = ", ".join(DATASET_ORDER)
        raise ValueError(f"Unsupported strict unified dataset '{dataset_name}'. Supported: {supported}")

    normalized_profile = normalize_unified_profile_name(profile_name)
    out = deepcopy(dict(cfg or {}))
    out["dataset"] = ds
    out.update(COPY_SPAN_RETRIEVAL_WEIGHTS)
    out.update(COPY_SPAN_RENDER_WEIGHTS)
    out.update(COPY_SPAN_DISABLED_MODULES)
    out.update(UNIFIED_METHOD_LOCK)
    out.update(unified_method_overrides(normalized_profile))
    out["retrieval_objective_mode"] = resolve_unified_copy_span_mode(ds)
    out.update(unified_interface_config(normalized_profile))
    out.update(unified_rendering_overrides(normalized_profile))
    out.update(unified_budget_config(normalized_profile))
    out.update(UNIFIED_RUNTIME_DEFAULTS)
    out.update(
        {
            "method_name": "copy-span",
            "method_profile": normalized_profile,
            "profile_family": STRICT_UNIFIED_PROFILE_FAMILY,
            "reference_label": REFERENCE_LABEL,
            "sota_reference": False,
            "strict_unified_main": True,
            "dataset_independent_method": True,
        }
    )
    return out


def unified_method_signature(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: cfg.get(key) for key in UNIFIED_METHOD_SETTING_KEYS}


def unified_budget_signature(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: cfg.get(key) for key in UNIFIED_BUDGET_CONFIG_KEYS}


def audit_unified_configs(configs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    datasets = sorted(configs)
    errors = []
    method_signatures = {dataset: unified_method_signature(cfg) for dataset, cfg in configs.items()}
    budget_signatures = {dataset: unified_budget_signature(cfg) for dataset, cfg in configs.items()}

    if datasets:
        first_method = method_signatures[datasets[0]]
        first_budget = budget_signatures[datasets[0]]
        for dataset in datasets[1:]:
            if method_signatures[dataset] != first_method:
                errors.append(f"method setting drift for {dataset}")
            if budget_signatures[dataset] != first_budget:
                errors.append(f"budget setting drift for {dataset}")

    for dataset, cfg in configs.items():
        if cfg.get("retrieval_objective_mode") not in {"baseline", "canonical_copy_span_unified"}:
            errors.append(f"unsupported unified objective for {dataset}: {cfg.get('retrieval_objective_mode')}")
        for key, expected in UNIFIED_RENDERING_LOCK.items():
            if key == "final_top_slice_reorder_enabled":
                # Reorder-all profiles may enable this globally. The signature
                # comparison above ensures it cannot drift by dataset.
                continue
            if bool(cfg.get(key)) is not bool(expected):
                errors.append(f"unified rendering lock violated for {dataset}.{key}")

    return {
        "datasets": datasets,
        "method_signatures": method_signatures,
        "budget_signatures": budget_signatures,
        "errors": errors,
        "ok": not errors,
    }
