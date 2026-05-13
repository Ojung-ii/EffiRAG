import argparse
import json
from pathlib import Path

from .unified_copy_span_policy import audit_unified_configs
from .utils import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Audit EffiRAG YAML configs for naming/strategy drift.")
    parser.add_argument("config_dir", nargs="?", default="configs")
    args = parser.parse_args()

    root = Path(args.config_dir)
    rows = []
    unified_groups = {}
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
        normalized = str(path).replace("\\", "/")
        if "/copy_span_instruction_unified/" in normalized:
            profile_name = path.parent.name
            unified_groups.setdefault(profile_name, {})[path.stem] = dict(payload)

    result = {"configs": rows}
    if unified_groups:
        unified_audit = {
            profile_name: audit_unified_configs(configs)
            for profile_name, configs in sorted(unified_groups.items())
        }
        result["unified_profile_invariance"] = unified_audit
        print(json.dumps(result, ensure_ascii=False, indent=2))
        errors = [
            error
            for audit in unified_audit.values()
            for error in audit.get("errors", [])
        ]
        if errors:
            raise SystemExit(1)
        return

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
