#!/usr/bin/env python3
import ast
import importlib
import sys
from pathlib import Path


FORBIDDEN = {
    "candidate_recall_boost",
    "dynamic_compact_selector",
    "gl_rcedr",
    "unified_acr_rcedr_selector",
    "legacy_sota_policy",
}


def _audit_phase7_module_imports(repo_root: Path) -> list[str]:
    target = repo_root / "effirag" / "phase7_evidence_flow.py"
    text = target.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(target))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = str(alias.name or "")
                leaf = mod.split(".")[-1]
                if leaf in FORBIDDEN:
                    bad.append(f"phase7_evidence_flow.py imports forbidden module: {mod}")
        elif isinstance(node, ast.ImportFrom):
            mod = str(node.module or "")
            leaf = mod.split(".")[-1]
            if leaf in FORBIDDEN:
                bad.append(f"phase7_evidence_flow.py imports forbidden module: {mod}")
    return bad


def _audit_registry_phase7_mode() -> list[str]:
    bad = []
    fq_forbidden = {f"effirag.{name}" for name in FORBIDDEN}
    for mod in list(fq_forbidden):
        sys.modules.pop(mod, None)

    reg = importlib.import_module("effirag.registry")
    before = set(sys.modules.keys())
    reg.register_defaults("phase7_evidence_flow")
    after = set(sys.modules.keys())
    newly_loaded = sorted((after - before).intersection(fq_forbidden))
    if newly_loaded:
        bad.append(
            "register_defaults('phase7_evidence_flow') loaded forbidden modules: "
            + ", ".join(newly_loaded)
        )
    if "phase7_evidence_flow" not in set(reg.METHOD_REGISTRY.keys()):
        bad.append("phase7_evidence_flow method was not registered.")
    return bad


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    errors = []
    errors.extend(_audit_phase7_module_imports(repo_root))
    errors.extend(_audit_registry_phase7_mode())
    if errors:
        print("PHASE7_ACTIVE_PATH_AUDIT: FAIL")
        for err in errors:
            print(f"- {err}")
        return 1
    print("PHASE7_ACTIVE_PATH_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

