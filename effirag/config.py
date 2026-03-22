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


@dataclass
class RagConfig(RetrievalConfig):
    output_dir: str = "outputs/rag"
    run_qa: bool = True
    generator: str = "heuristic"
    model_name: str = ""
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
    return merged
