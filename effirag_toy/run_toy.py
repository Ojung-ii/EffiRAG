import statistics
import networkx as nx

from types import ToyConfig, RetrievalResult
from toy_graph import build_toy_instance
from seed_selection import stable_seed_selection
from corridor_trimming import build_corridor, greedy_trim
from baselines import ppr_union_baseline, shortest_path_baseline, khop_baseline
from metrics import compute_metrics


def canonical_edge(u: int, v: int):
    return (u, v) if u < v else (v, u)


def run_effirag_toy(instance, cfg: ToyConfig) -> RetrievalResult:
    G = nx.Graph()
    G.add_edges_from(instance.graph_edges)

    seed_result = stable_seed_selection(
        G=G,
        anchors=instance.query.anchors,
        M=cfg.M,
        T=cfg.T,
        K=cfg.K,
        tau=cfg.tau,
        alpha=cfg.ppr_alpha,
        base_seed=cfg.base_seed,
    )

    corridor, retained_pairs = build_corridor(
        G=G,
        anchors=instance.query.anchors,
        seeds=seed_result["seeds"],
        best_anchor_scores=seed_result["anchor_scores"],
        alpha=cfg.ppr_alpha,
        Lp=cfg.Lp,
        Bc=cfg.Bc,
    )

    trimmed = greedy_trim(
        H=corridor,
        anchors=instance.query.anchors,
        seeds=seed_result["seeds"],
        rho=cfg.rho,
    )

    return RetrievalResult(
        method="effirag",
        selected_nodes=set(trimmed.nodes()),
        selected_edges={canonical_edge(u, v) for u, v in trimmed.edges()},
        diagnostics={
            "best_run_id": seed_result["run_id"],
            "num_seeds": len(seed_result["seeds"]),
            "retained_pairs": len(retained_pairs),
            "corridor_nodes_before_trim": corridor.number_of_nodes(),
            "corridor_nodes_after_trim": trimmed.number_of_nodes(),
        },
    )


def summarize(records):
    by_method = {}
    for method, metrics in records:
        by_method.setdefault(method, []).append(metrics)

    for method, vals in by_method.items():
        print(f"\n=== {method} ===")
        for key in vals[0].keys():
            mean_val = statistics.mean(v[key] for v in vals)
            print(f"{key:20s}: {mean_val:.4f}")


def main():
    cfg = ToyConfig()
    records = []

    for i in range(cfg.num_graphs):
        instance = build_toy_instance(
            seed=cfg.base_seed + i,
            path_len=cfg.path_len,
            branch_len=cfg.branch_len,
            num_anchor_noise=cfg.num_anchor_noise,
            num_connector_noise=cfg.num_connector_noise,
            extra_random_edges=cfg.extra_random_edges,
        )

        G = nx.Graph()
        G.add_edges_from(instance.graph_edges)

        methods = [
            run_effirag_toy(instance, cfg),
            ppr_union_baseline(G, instance.query.anchors, alpha=cfg.ppr_alpha, topk=cfg.k_top_baseline),
            shortest_path_baseline(G, instance.query.anchors),
            khop_baseline(G, instance.query.anchors, k=cfg.khop_k),
        ]

        for result in methods:
            m = compute_metrics(instance, result)
            records.append((result.method, m))

    summarize(records)


if __name__ == "__main__":
    main()