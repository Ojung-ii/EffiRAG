import argparse
import json
from pathlib import Path

from .utils import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Audit EffiRAG YAML configs for naming/strategy drift.")
    parser.add_argument("config_dir", nargs="?", default="configs")
    args = parser.parse_args()

    root = Path(args.config_dir)
    rows = []
    for path in sorted(root.rglob("*.yaml")):
        payload = load_yaml(path)
        audit = payload.get("_config_audit", {}) or {}
        rows.append({
            "config": str(path),
            "graph_mode": payload.get("graph_mode", "current_entity_graph"),
            "chunk_node_enabled_in_diffusion": payload.get("chunk_node_enabled_in_diffusion", False),
            "samples_per_anchor": payload.get("samples_per_anchor", None),
            "phase1_run_preshortlist_topm": payload.get("phase1_run_preshortlist_topm", None),
            "phase1_full_run_score_topk": payload.get("phase1_full_run_score_topk", None),
            "warning_count": audit.get("warning_count", 0),
            "warnings": audit.get("warnings", []),
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
