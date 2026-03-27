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
    openie_parallel_workers: int = 4
    openie_log_every: int = 200
    embedding_enabled: bool = True
    sentence_rerank_enabled: bool = True
    embedding_model_name: str = "nvidia/NV-Embed-v2"
    embedding_text_max_chars: int = 600
    embedding_weight: float = 0.35
    embedding_rerank_topn: int = 80
    embedding_batch_size: int = 16
    embedding_max_length: int = 192
    semantic_topn_entity: int = 30
    semantic_topn_chunk: int = 15
    graph_reserve_topn: int = 15
    semantic_topn: int = 50
    semantic_candidate_union: bool = True
    semantic_scan_batch_size: int = 8192
    run_score_semantic_weight: float = 0.30
    run_score_anchor_weight: float = 0.20
    run_score_structure_weight: float = 0.25
    run_score_bridge_weight: float = 0.15
    run_score_redundancy_weight: float = 0.10
    seed_score_semantic_weight: float = 0.35
    seed_score_graph_weight: float = 0.45
    seed_score_anchor_weight: float = 0.20

    max_anchors: int = 5
    samples_per_anchor: int = 3
    num_workers: int = 1
    candidate_top_t: int = 20
    seed_k: int = 4
    pair_top_lp: int = 3
    corridor_top_bc: int = 20
    phase1_parallel_ppr: bool = True
    phase1_run_shortlist_topk: int = 2
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
    if "semantic_candidate_union" in merged:
        merged["semantic_candidate_union"] = parse_bool(merged["semantic_candidate_union"])
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
    return merged
