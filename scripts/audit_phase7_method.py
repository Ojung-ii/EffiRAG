#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from effirag.utils import load_yaml


FORBIDDEN = {
    "candidate_recall_boost",
    "dynamic_compact_selector",
    "gl_rcedr",
    "unified_acr_rcedr_selector",
    "legacy_sota_policy",
    "top1_correction",
    "answer_support_pinning",
}

ACTIVE_CFG_DIR = Path("configs/main_config/phase7_evidence_flow")
INTENT_PATH = Path("effirag/phase7_query_intent.py")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _collect_import_leaves(py_path: Path) -> Set[str]:
    tree = ast.parse(_read(py_path), filename=str(py_path))
    leaves: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                leaves.add(str(alias.name or "").split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            leaves.add(str(node.module or "").split(".")[-1])
    return leaves


def _function_source_block(py_path: Path, fn_name: str) -> str:
    text = _read(py_path)
    tree = ast.parse(text, filename=str(py_path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and str(node.name) == fn_name:
            start = int(getattr(node, "lineno", 1))
            end = int(getattr(node, "end_lineno", start))
            lines = text.splitlines()
            return "\n".join(lines[start - 1 : end])
    return ""


def _count_selected_append_calls(py_path: Path, fn_name: str = "run_phase7_evidence_flow") -> int:
    tree = ast.parse(_read(py_path), filename=str(py_path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and str(node.name) == fn_name:
            count = 0
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                    if isinstance(n.func.value, ast.Name) and str(n.func.value.id) == "selected" and str(n.func.attr) == "append":
                        count += 1
            return int(count)
    return 0


def _active_cfg_paths() -> List[Path]:
    if not ACTIVE_CFG_DIR.exists():
        return []
    return sorted([p for p in ACTIVE_CFG_DIR.glob("*.yaml") if p.is_file()])


def _cfg_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    v = cfg.get(key, default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _cfg_str(cfg: Dict[str, Any], key: str, default: str) -> str:
    return str(cfg.get(key, default) or default).strip().lower()


def main() -> int:
    errors: List[str] = []
    warnings: List[str] = []

    # 1) Config invariants
    cfg_paths = _active_cfg_paths()
    if not cfg_paths:
        errors.append(f"No active configs found in {ACTIVE_CFG_DIR}")
    objective_tuples = set()
    for path in cfg_paths:
        cfg = dict(load_yaml(str(path)) or {})
        method = _cfg_str(cfg, "method", "")
        if method != "phase7_evidence_flow":
            continue
        if _cfg_bool(cfg, "phase7_enable_phase2_refinement", True):
            errors.append(f"{path}: phase7_enable_phase2_refinement must be false")
        if _cfg_bool(cfg, "phase7_corridor_enabled", False):
            errors.append(f"{path}: phase7_corridor_enabled must be false for active mainline configs")
        if _cfg_str(cfg, "phase7_atom_unit", "sentence") != "sentence":
            errors.append(f"{path}: phase7_atom_unit must be sentence")
        if not _cfg_bool(cfg, "phase7_require_sentence_layer", True):
            errors.append(f"{path}: phase7_require_sentence_layer must be true")
        if not _cfg_bool(cfg, "phase7_build_sentence_layer", True):
            errors.append(f"{path}: phase7_build_sentence_layer must be true")
        objective_tuples.add(
            (
                _cfg_str(cfg, "phase7_objective_mode", "normalized_equal_weight"),
                float(cfg.get("phase7_lambda_bridge", 1.0) or 1.0),
                float(cfg.get("phase7_mu_redundancy", 1.0) or 1.0),
            )
        )
    if len(objective_tuples) > 1:
        errors.append("Active Phase7 configs contain dataset-specific objective settings (mode/lambda/mu differ).")

    # 2) Active path forbidden imports
    p7_path = Path("effirag/phase7_evidence_flow.py")
    reg_path = Path("effirag/registry.py")
    render_path = Path("effirag/render.py")
    if not p7_path.exists():
        errors.append("Missing effirag/phase7_evidence_flow.py")
    else:
        p7_imports = _collect_import_leaves(p7_path)
        hit = sorted([x for x in p7_imports if x in FORBIDDEN])
        if hit:
            errors.append(f"phase7_evidence_flow imports forbidden modules: {hit}")

        # Sentence-only final atoms (static contract check)
        p7_text = _read(p7_path)
        if 'node_type", "") or "").strip().lower() == "sentence"' not in p7_text:
            errors.append("Phase7 candidate extraction no longer explicitly enforces sentence-node candidates.")
        if _count_selected_append_calls(p7_path, "run_phase7_evidence_flow") != 1:
            errors.append("Hard corridor pinning suspicion: selected.append is used outside single marginal selection flow.")

        # Query-type controller guardrail.
        query_type_tokens = [
            "question_type",
            "if_comparison",
            "if_bridge",
            "comparison_question",
            "bridge_question",
            "query_type ==",
            "if query_type",
        ]
        for tok in query_type_tokens:
            if tok in p7_text:
                errors.append(f"Query-type hard controller token detected in active Phase7 path: {tok}")

        # Corridor helper must not reference gold labels/supporting facts.
        for fn_name in ("_corridor_anchor_rows", "_corridor_seed_rows", "_extract_anchor_seed_corridors"):
            src = _function_source_block(p7_path, fn_name)
            if src and ("supporting_facts" in src or "gold" in src):
                errors.append(f"{fn_name} appears to reference gold/support labels.")

        # Objective should not use gold labels.
        obj_src = _function_source_block(p7_path, "_objective_delta")
        if obj_src and "gold" in obj_src:
            errors.append("Objective function references gold labels; must remain eval-only.")
        if "answer_support_pinning" in p7_text:
            errors.append("Phase7 active path must not call answer_support_pinning.")

    if INTENT_PATH.exists():
        intent_text = _read(INTENT_PATH)
        forbidden_intent_tokens = [
            "if query_type",
            "query_type ==",
            "dataset ==",
            "hotpotqa",
            "2wikimultihopqa",
            "supporting_facts",
            "gold",
        ]
        for tok in forbidden_intent_tokens:
            if tok in intent_text:
                errors.append(f"Query-intent module contains forbidden token: {tok}")
    else:
        warnings.append("Query-intent module not found: effirag/phase7_query_intent.py")

    if reg_path.exists():
        reg_text = _read(reg_path)
        if "phase7_evidence_flow" not in reg_text:
            errors.append("Registry does not expose phase7_evidence_flow active path.")
    else:
        errors.append("Missing effirag/registry.py")

    # 3) Rendering contract check (phase7 flat path should not rerank/semantic-score)
    if not render_path.exists():
        errors.append("Missing effirag/render.py")
    else:
        phase7_flat_src = _function_source_block(render_path, "render_phase7_flat_context")
        if not phase7_flat_src:
            errors.append("render_phase7_flat_context not found.")
        else:
            forbidden_tokens = [
                "canonical_sentence_score(",
                "select_dynamic_compact_evidence(",
                "_apply_top_slice_reorder(",
                "_apply_answer_support_pinning(",
                "top1_correction",
            ]
            for tok in forbidden_tokens:
                if tok in phase7_flat_src:
                    errors.append(f"render_phase7_flat_context should not use {tok}")

    status = "PASS" if not errors else "FAIL"
    payload = {
        "status": status,
        "errors": list(errors),
        "warnings": list(warnings),
        "checked_config_count": len(cfg_paths),
        "checks": {
            "no_phase2_pruning_default": "PASS" if not any("phase7_enable_phase2_refinement" in e for e in errors) else "FAIL",
            "sentence_atoms_only_final": "PASS" if not any("sentence-node" in e or "phase7_atom_unit" in e for e in errors) else "FAIL",
            "no_hard_corridor_pinning": "PASS" if not any("Hard corridor pinning" in e for e in errors) else "FAIL",
            "no_dataset_specific_objective_weights": "PASS" if not any("dataset-specific objective settings" in e for e in errors) else "FAIL",
            "no_query_type_controller": "PASS" if not any("Query-type hard controller token" in e for e in errors) else "FAIL",
            "no_candidate_boost": "PASS" if not any("candidate_recall_boost" in e for e in errors) else "FAIL",
            "no_top1_correction": "PASS" if not any("top1_correction" in e for e in errors) else "FAIL",
            "corridor_uses_no_gold_labels": "PASS" if not any("gold/support" in e or "Objective function references gold" in e for e in errors) else "FAIL",
            "gold_diagnostics_eval_only": "PASS" if not any("Objective function references gold" in e for e in errors) else "FAIL",
            "no_query_type_hard_controller": "PASS" if not any("Query-type hard controller token" in e or "forbidden token: if query_type" in e for e in errors) else "FAIL",
            "no_dataset_specific_intent_behavior": "PASS" if not any("forbidden token: hotpotqa" in e or "forbidden token: 2wikimultihopqa" in e or "forbidden token: dataset ==" in e for e in errors) else "FAIL",
            "no_gold_use_in_retrieval": "PASS" if not any("forbidden token: supporting_facts" in e or "forbidden token: gold" in e for e in errors) else "FAIL",
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
