#!/usr/bin/env python3
"""
Build strict unified copy-span-instruction configs for 4 datasets.

The generated configs intentionally keep dataset-specific data paths while
locking method-level behavior to one profile-selected setting across datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "configs" / "SOTA_config" / "copy_span_instruction_4ds" / "on_the_fly"
DEFAULT_PROFILE_ROOT = REPO_ROOT / "configs" / "main_config" / "copy_span_instruction_unified"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.config import RagConfig
from effirag.unified_copy_span_policy import (
    DATASET_ORDER,
    STRICT_UNIFIED_PROFILE_FAMILY,
    STRICT_UNIFIED_VARIANT_NAME,
    UNIFIED_DYNAMIC_PROFILE_NAMES,
    UNIFIED_ENHANCED_PROFILE_NAMES,
    UNIFIED_PROFILE_NAMES,
    UNIFIED_PROFILE_DIRS,
    apply_unified_profile,
    audit_unified_configs,
    normalize_unified_profile_name,
    unified_budget,
    unified_budget_signature,
    unified_method_signature,
)


PROFILE_ORDER = UNIFIED_PROFILE_NAMES


def _rag_config_field_names() -> set[str]:
    return {f.name for f in fields(RagConfig)}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return dict(payload)


def _filter_rag_config(raw_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = _rag_config_field_names()
    return {k: v for k, v in raw_cfg.items() if k in allowed}


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False)
    path.write_text(text, encoding="utf-8")


def _build_dataset_cfg(source_root: Path, profile_name: str, dataset: str) -> Dict[str, Any]:
    source_path = source_root / f"{dataset}.yaml"
    if not source_path.exists():
        raise FileNotFoundError(f"source config missing for {dataset}: {source_path}")
    base_cfg = _filter_rag_config(_load_yaml(source_path))
    cfg = apply_unified_profile(base_cfg, dataset, profile_name)
    cfg["output_dir"] = (
        f"outputs/main_config_runs/copy_span_instruction_unified/"
        f"{UNIFIED_PROFILE_DIRS[normalize_unified_profile_name(profile_name)]}/{dataset}"
    )
    cfg["timestamp_output"] = True
    return _filter_rag_config(cfg)


def build_unified_config_pack(profile_root: Path, source_root: Path) -> Path:
    profile_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "profile_family": STRICT_UNIFIED_PROFILE_FAMILY,
        "canonical_variant_name": STRICT_UNIFIED_VARIANT_NAME,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_root": str(source_root),
        "dataset_order": list(DATASET_ORDER),
        "strict_unified_contract": {
            "retrieval_objective_mode": "baseline",
            "answer_support_pinning_enabled": False,
            "corridor_answer_preserve_guarded_hotpot_enabled": False,
            "oracle_support_injection_enabled": False,
            "precomputed_retrieval_path": "",
            "precomputed_retrieval_strict": False,
            "dataset_specific_method_toggles_allowed": False,
        },
        "enhanced_profiles": [
            UNIFIED_PROFILE_DIRS[profile]
            for profile in UNIFIED_ENHANCED_PROFILE_NAMES
        ],
        "dynamic_profiles": [
            UNIFIED_PROFILE_DIRS[profile]
            for profile in UNIFIED_DYNAMIC_PROFILE_NAMES
        ],
        "profiles": {},
    }

    all_audits: Dict[str, Any] = {}
    for profile_name in PROFILE_ORDER:
        profile_dir = profile_root / UNIFIED_PROFILE_DIRS[profile_name]
        profile_dir.mkdir(parents=True, exist_ok=True)

        configs: Dict[str, Dict[str, Any]] = {}
        for dataset in DATASET_ORDER:
            cfg = _build_dataset_cfg(source_root, profile_name, dataset)
            cfg_path = profile_dir / f"{dataset}.yaml"
            _write_yaml(cfg_path, cfg)
            configs[dataset] = cfg

        audit = audit_unified_configs(configs)
        all_audits[profile_name] = audit
        if not audit["ok"]:
            joined = "; ".join(audit["errors"])
            raise AssertionError(f"Unified config audit failed for {profile_name}: {joined}")

        first_dataset = DATASET_ORDER[0]
        manifest["profiles"][UNIFIED_PROFILE_DIRS[profile_name]] = {
            "profile_name": profile_name,
            "budget": unified_budget(profile_name),
            "budget_config": unified_budget_signature(configs[first_dataset]),
            "method_signature": unified_method_signature(configs[first_dataset]),
            "configs": {
                dataset: str(profile_dir / f"{dataset}.yaml")
                for dataset in DATASET_ORDER
            },
            "invariance_audit": {
                "ok": bool(audit["ok"]),
                "errors": list(audit["errors"]),
            },
        }

    manifest["invariance_audit"] = {
        profile: {
            "ok": bool(audit["ok"]),
            "errors": list(audit["errors"]),
        }
        for profile, audit in all_audits.items()
    }

    manifest_path = profile_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Update strict unified copy-span config pack for 4 datasets.")
    parser.add_argument("--profile-root", default=str(DEFAULT_PROFILE_ROOT))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    args = parser.parse_args()

    profile_root = Path(args.profile_root).resolve()
    source_root = Path(args.source_root).resolve()
    manifest_path = build_unified_config_pack(profile_root, source_root)
    print(str(manifest_path))


if __name__ == "__main__":
    main()
