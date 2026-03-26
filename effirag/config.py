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
    embedding_enabled: bool = False
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_weight: float = 0.35
    embedding_rerank_topn: int = 80
    embedding_batch_size: int = 16
    embedding_max_length: int = 256

    max_anchors: int = 6
    samples_per_anchor: int = 8
    num_workers: int = 1
    candidate_top_t: int = 20
    seed_k: int = 4
    pair_top_lp: int = 3
    corridor_top_bc: int = 20
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
    max_context_sentences: int = 10
    render_mode: str = ""
    max_corridors_in_context: int = 2
    max_main_sentences_per_corridor: int = 2
    max_support_per_corridor: int = 1
    max_total_sentences: int = 12
    alpha: float = 1.0
    beta: float = 0.35
    gamma_main: float = 0.45
    delta_support: float = 0.20
    eta_connector: float = 0.20
    zeta_query: float = 0.12
    xi_locality: float = 0.08
    lambda_redundancy: float = 0.25
    top_corridors: int = 3
    max_sentences: int = 10
    reserve_top_corridor: bool = False
    order_strategy: str = "score"
    measure_gpu_peak: bool = False
    measure_cpu_ram: bool = False


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
    if "openie_local_files_only" in merged:
        merged["openie_local_files_only"] = parse_bool(merged["openie_local_files_only"])
    if "embedding_enabled" in merged:
        merged["embedding_enabled"] = parse_bool(merged["embedding_enabled"])
    if "ppr_subgraph_enable" in merged:
        merged["ppr_subgraph_enable"] = parse_bool(merged["ppr_subgraph_enable"])
    if "anchor_diag_store_full_scores" in merged:
        merged["anchor_diag_store_full_scores"] = parse_bool(merged["anchor_diag_store_full_scores"])
    return merged
