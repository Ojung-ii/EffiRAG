from dataclasses import dataclass, fields
from typing import Any, Dict

from .utils import parse_bool


@dataclass
class RetrievalConfig:
    dataset: str = "hotpotqa"
    data_path: str = None
    split: str = "validation"
    limit: int = None
    method: str = "effirag"
    output_dir: str = "outputs/retrieval"
    timestamp_output: bool = True
    global_corpus_path: str = ""
    graph_cache_dir: str = "outputs/index_cache"
    force_rebuild_graph_index: bool = False
    prebuilt_igraph_path: str = ""
    prebuilt_igraph_format: str = "hipporag_pickle"
    prebuilt_entity_token_limit: int = 6
    openie_mode: str = "llm"
    openie_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    openie_text_max_chars: int = 2200
    openie_max_new_tokens: int = 256
    openie_local_files_only: bool = True
    openie_retry_attempts: int = 3
    openie_retry_backoff_sec: float = 0.2
    openie_error_sample_limit: int = 20
    openie_api_base_url: str = ""
    openie_api_key: str = ""
    openie_api_timeout_sec: float = 120.0
    openie_parallel_workers: int = 6
    openie_log_every: int = 200
    embedding_enabled: bool = True
    sentence_rerank_enabled: bool = True
    embedding_model_name: str = "nvidia/NV-Embed-v2"
    embedding_text_max_chars: int = 600
    embedding_weight: float = 0.35
    embedding_rerank_topn: int = 80
    embedding_batch_size: int = 16
    embedding_max_length: int = 192
    candidate_embedding_fallback_enabled: bool = False
    query_embedding_max_length: int = 96
    query_embedding_text_max_chars: int = 384
    query_embedding_cache_enabled: bool = True
    query_embedding_cache_dir: str = "outputs/query_embedding_cache"
    semantic_topn_entity: int = 30
    semantic_topn_chunk: int = 15
    graph_reserve_topn: int = 15
    semantic_topn: int = 50
    # auto: current_entity_graph->sentence, entity_chunk_graph->passage
    # sentence: legacy sentence node as chunk
    # passage: HippoRAG2-like document/passage chunk node
    index_chunk_unit: str = "auto"
    # current_entity_graph (frozen path) | entity_chunk_graph (next-method-design experiment path)
    graph_mode: str = "current_entity_graph"
    # entity_aggregate: seed/entity scores lifted to chunk via support map
    # direct_chunk: direct chunk-centric score aggregation
    chunk_scoring_mode: str = "entity_aggregate"
    entity_chunk_transition_weight: float = 0.35
    chunk_node_enabled_in_diffusion: bool = False
    chunk_score_topk: int = 20
    chunk_package_enabled: bool = False
    entity_lookup_use_two_tier: bool = True
    entity_lookup_tier1_topk: int = 256
    entity_lookup_alias_token_limit: int = 12
    entity_lookup_global_fallback_topn: int = 8
    semantic_candidate_union: bool = True
    semantic_chunk_lookup_strategy: str = "adaptive"
    semantic_chunk_cache_min_ratio: float = 0.8
    chunk_lookup_anchor_cache_topn: int = 8
    chunk_lookup_tier1_topk: int = 256
    chunk_lookup_dense_topk: int = 96
    chunk_lookup_global_fallback_topn: int = 6
    semantic_chunk_support_expand_per_entity: int = 4
    semantic_chunk_support_expand_total: int = 48
    semantic_scan_batch_size: int = 8192
    run_score_semantic_weight: float = 0.30
    run_score_anchor_weight: float = 0.20
    run_score_structure_weight: float = 0.25
    run_score_bridge_weight: float = 0.15
    run_score_redundancy_weight: float = 0.10
    # Optional next-method-design run-level objective terms (default off).
    run_score_pair_coverage_weight: float = 0.0
    run_score_bridge_completeness_weight: float = 0.0
    run_score_entity_chunk_grounding_weight: float = 0.0
    run_score_anchor_dispersion_penalty: float = 0.0
    seed_score_semantic_weight: float = 0.35
    seed_score_graph_weight: float = 0.45
    seed_score_anchor_weight: float = 0.20
    # Optional seed-level bridge/grounding boosts (default off).
    seed_score_bridge_weight: float = 0.0
    seed_score_chunk_grounding_weight: float = 0.0
    # Optional final corridor grounding boosts (default off).
    corridor_score_chunk_support_weight: float = 0.0
    corridor_score_answer_alignment_weight: float = 0.0
    top1_correction_enabled: bool = False
    top1_correction_topk: int = 3
    top1_correction_corridor_weight_base: float = 0.75
    top1_correction_corridor_weight_anchor: float = 0.10
    top1_correction_corridor_weight_support: float = 0.06
    top1_correction_corridor_weight_bridge: float = 0.05
    top1_correction_corridor_weight_semantic: float = 0.04
    top1_correction_corridor_weight_redundancy: float = 0.05
    top1_correction_sentence_weight_base: float = 0.60
    top1_correction_sentence_weight_corridor: float = 0.20
    top1_correction_sentence_weight_main: float = 0.08
    top1_correction_sentence_weight_support: float = 0.04
    top1_correction_sentence_weight_query: float = 0.04
    top1_correction_sentence_weight_locality: float = 0.04
    top1_correction_sentence_weight_redundancy: float = 0.04

    max_anchors: int = 5
    samples_per_anchor: int = 3
    num_workers: int = 1
    candidate_top_t: int = 20
    seed_k: int = 4
    pair_top_lp: int = 3
    corridor_top_bc: int = 20
    phase1_parallel_ppr: bool = True
    phase1_run_shortlist_topk: int = 2
    phase1_run_preshortlist_topm: int = 2
    phase1_full_run_score_topk: int = 2
    run_score_sparse_topk: int = 64
    run_score_surrogate_topk: int = 128
    proposal_lazy_union_topk: int = 64
    proposal_anchor_local_topn: int = 48
    proposal_shared_high_conf_topn: int = 24
    proposal_global_fallback_topn: int = 16
    # off (guardrail baseline) | mild | medium | aggressive (speed-optimized, may lower Recall@1)
    proposal_union_experiment_mode: str = "off"
    proposal_adaptive_budget: bool = True
    proposal_chunk_rank_cap: int = 48
    proposal_reserve_hops: int = 3
    proposal_reserve_graph_mix_topn: int = 3
    proposal_reserve_bfs_fallback: bool = False
    proposal_anchor_distance_bonus: bool = True
    proposal_sparse_subgraph_build: bool = False
    proposal_subgraph_anchor_neighbor_cap: int = 12
    proposal_subgraph_support_cap: int = 4
    proposal_subgraph_connector_cap: int = 8
    pair_shortlist_topb: int = 6
    phase2_refine_mode: str = "bounded_local"
    phase2_bidirectional_full_ppr: bool = False
    reuse_semantic_scores_in_final: bool = True
    trim_on: bool = True
    trim_rho: float = 0.6

    ppr_alpha: float = 0.15
    tau: int = 4
    edge_drop_prob: float = 0.1
    random_seed: int = 42
    ppr_engine: str = "auto"  # auto | power | mc
    ppr_power_max_iter: int = 100
    ppr_power_tol: float = 1.0e-6
    ppr_min_score: float = 0.0
    ppr_mc_walks: int = 512
    ppr_mc_max_steps: int = 24
    ppr_parallel_workers: int = 1
    ppr_subgraph_enable: bool = True
    ppr_subgraph_hops: int = 2
    ppr_subgraph_max_nodes: int = 30000
    anchor_diag_topn: int = 10
    anchor_diag_store_full_scores: bool = False


@dataclass
class RagConfig(RetrievalConfig):
    output_dir: str = "outputs/rag"
    run_qa: bool = True
    evaluator_mode: str = "legacy"  # legacy | hipporag2_parity
    generator: str = "heuristic"
    model_name: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_timeout_sec: float = 120.0
    llm_max_new_tokens: int = 64
    max_context_sentences: int = 14
    render_mode: str = ""
    max_corridors_in_context: int = 3
    max_main_sentences_per_corridor: int = 3
    max_support_per_corridor: int = 2
    max_total_sentences: int = 14
    # sentence_compressed (legacy/frozen) | chunk_package_basic | chunk_grounded_bridge | chunk_package_grounded_support
    delivery_mode: str = "sentence_compressed"
    max_chunk_packages: int = 4
    max_excerpt_sentences_per_package: int = 3
    # Chunk-grounded context packaging (default off; frozen behavior preserved).
    chunk_grounding_enabled: bool = False
    chunk_grounding_mode: str = "sentence_backfill"  # sentence_backfill | corridor_lift | package_score
    chunk_excerpt_max_per_corridor: int = 1
    chunk_excerpt_window_sentences_before: int = 1
    chunk_excerpt_window_sentences_after: int = 1
    chunk_excerpt_max_total_sentences: int = 4
    chunk_excerpt_dedup_enabled: bool = True
    chunk_grounding_top_corridor_chunks: int = 2
    chunk_grounding_top_k_packages: int = 4
    package_score_answer_weight: float = 0.50
    package_score_bridge_weight: float = 0.20
    package_score_support_weight: float = 0.15
    package_score_chunk_grounding_weight: float = 0.15
    package_score_redundancy_weight: float = 0.10
    alpha: float = 1.0
    beta: float = 0.35
    gamma_main: float = 0.45
    delta_support: float = 0.20
    eta_connector: float = 0.20
    zeta_query: float = 0.12
    xi_locality: float = 0.08
    lambda_redundancy: float = 0.25
    top_corridors: int = 3
    max_sentences: int = 14
    reserve_top_corridor: bool = False
    order_strategy: str = "score"
    measure_gpu_peak: bool = False
    measure_cpu_ram: bool = False
    profile_stages: bool = False
    profile_output: str = ""
    profile_limit: int = 0
    profile_query_indices: str = ""
    retrieval_only: bool = False


def dataclass_from_dict(cls, values):
    valid = {f.name for f in fields(cls)}
    payload = {k: v for k, v in values.items() if k in valid}
    return cls(**payload)


def apply_cli_overrides(config_dict, args_namespace):
    args = vars(args_namespace)
    merged = dict(config_dict)
    for k, v in args.items():
        if v is None:
            continue
        merged[k] = v

    if "trim_on" in merged:
        merged["trim_on"] = parse_bool(merged["trim_on"])
    if "run_qa" in merged:
        merged["run_qa"] = parse_bool(merged["run_qa"])
    if "measure_gpu_peak" in merged:
        merged["measure_gpu_peak"] = parse_bool(merged["measure_gpu_peak"])
    if "measure_cpu_ram" in merged:
        merged["measure_cpu_ram"] = parse_bool(merged["measure_cpu_ram"])
    if "reserve_top_corridor" in merged:
        merged["reserve_top_corridor"] = parse_bool(merged["reserve_top_corridor"])
    if "force_rebuild_graph_index" in merged:
        merged["force_rebuild_graph_index"] = parse_bool(merged["force_rebuild_graph_index"])
    if "timestamp_output" in merged:
        merged["timestamp_output"] = parse_bool(merged["timestamp_output"])
    if "openie_local_files_only" in merged:
        merged["openie_local_files_only"] = parse_bool(merged["openie_local_files_only"])
    if "embedding_enabled" in merged:
        merged["embedding_enabled"] = parse_bool(merged["embedding_enabled"])
    if "sentence_rerank_enabled" in merged:
        merged["sentence_rerank_enabled"] = parse_bool(merged["sentence_rerank_enabled"])
    if "top1_correction_enabled" in merged:
        merged["top1_correction_enabled"] = parse_bool(merged["top1_correction_enabled"])
    if "query_embedding_cache_enabled" in merged:
        merged["query_embedding_cache_enabled"] = parse_bool(merged["query_embedding_cache_enabled"])
    if "candidate_embedding_fallback_enabled" in merged:
        merged["candidate_embedding_fallback_enabled"] = parse_bool(merged["candidate_embedding_fallback_enabled"])
    if "semantic_candidate_union" in merged:
        merged["semantic_candidate_union"] = parse_bool(merged["semantic_candidate_union"])
    if "chunk_node_enabled_in_diffusion" in merged:
        merged["chunk_node_enabled_in_diffusion"] = parse_bool(merged["chunk_node_enabled_in_diffusion"])
    if "chunk_package_enabled" in merged:
        merged["chunk_package_enabled"] = parse_bool(merged["chunk_package_enabled"])
    if "entity_lookup_use_two_tier" in merged:
        merged["entity_lookup_use_two_tier"] = parse_bool(merged["entity_lookup_use_two_tier"])
    if "proposal_reserve_bfs_fallback" in merged:
        merged["proposal_reserve_bfs_fallback"] = parse_bool(merged["proposal_reserve_bfs_fallback"])
    if "proposal_adaptive_budget" in merged:
        merged["proposal_adaptive_budget"] = parse_bool(merged["proposal_adaptive_budget"])
    if "proposal_anchor_distance_bonus" in merged:
        merged["proposal_anchor_distance_bonus"] = parse_bool(merged["proposal_anchor_distance_bonus"])
    if "proposal_sparse_subgraph_build" in merged:
        merged["proposal_sparse_subgraph_build"] = parse_bool(merged["proposal_sparse_subgraph_build"])
    if "proposal_union_experiment_mode" in merged:
        merged["proposal_union_experiment_mode"] = str(merged["proposal_union_experiment_mode"]).strip().lower()
    if "index_chunk_unit" in merged:
        merged["index_chunk_unit"] = str(merged["index_chunk_unit"]).strip().lower()
    if "phase1_parallel_ppr" in merged:
        merged["phase1_parallel_ppr"] = parse_bool(merged["phase1_parallel_ppr"])
    if "phase2_bidirectional_full_ppr" in merged:
        merged["phase2_bidirectional_full_ppr"] = parse_bool(merged["phase2_bidirectional_full_ppr"])
    if "reuse_semantic_scores_in_final" in merged:
        merged["reuse_semantic_scores_in_final"] = parse_bool(merged["reuse_semantic_scores_in_final"])
    if "ppr_subgraph_enable" in merged:
        merged["ppr_subgraph_enable"] = parse_bool(merged["ppr_subgraph_enable"])
    if "anchor_diag_store_full_scores" in merged:
        merged["anchor_diag_store_full_scores"] = parse_bool(merged["anchor_diag_store_full_scores"])
    if "profile_stages" in merged:
        merged["profile_stages"] = parse_bool(merged["profile_stages"])
    if "retrieval_only" in merged:
        merged["retrieval_only"] = parse_bool(merged["retrieval_only"])
    if "chunk_grounding_enabled" in merged:
        merged["chunk_grounding_enabled"] = parse_bool(merged["chunk_grounding_enabled"])
    if "chunk_excerpt_dedup_enabled" in merged:
        merged["chunk_excerpt_dedup_enabled"] = parse_bool(merged["chunk_excerpt_dedup_enabled"])
    return merged
