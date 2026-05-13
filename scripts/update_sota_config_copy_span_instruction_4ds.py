#!/usr/bin/env python3
"""
Build and refresh copy-span-instruction SOTA configs for 4 datasets.

This script creates:
  - configs/SOTA_config/copy_span_instruction_4ds/locked_precomputed/*.yaml
  - configs/SOTA_config/copy_span_instruction_4ds/on_the_fly/*.yaml
  - configs/SOTA_config/copy_span_instruction_4ds/manifest.json

The locked profile preserves historical precomputed retrieval paths for
stable quality reproduction. The on_the_fly profile clears precomputed
retrieval so retrieval timing is measured in the current run.
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
SOTA_ROOT = REPO_ROOT / "configs" / "SOTA_config" / "copy_span_instruction_4ds"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.config import RagConfig

DATASET_ORDER = [
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "popqa",
]

RETRIEVAL_CONTRACT_KEYS = [
    "retrieval_objective_mode",
    "retrieval_reconstruct_mode",
    "run_selection_mode",
    "semantic_topn_entity",
    "semantic_topn_chunk",
    "graph_reserve_topn",
    "sentence_rerank_enabled",
    "sentence_rerank_topn",
    "sentence_rerank_weight",
    "final_top_slice_reorder_enabled",
    "final_top_slice_reorder_topk",
    "answer_support_pinning_enabled",
    "answer_support_pinning_min",
    "oracle_support_injection_enabled",
    "max_context_sentences",
    "render_mode",
    "max_corridors_in_context",
    "corridor_sentence_budget",
    "corridor_support_sentence_budget",
    "corridor_connector_sentence_budget",
    "corridor_bridge_cap",
    "order_strategy",
    "canonical_variant_name",
    "prompt_variant",
]

QA_CONTRACT_KEYS = [
    "generator",
    "model_name",
    "evaluator_mode",
    "prompt_variant",
    "order_strategy",
    "max_context_sentences",
    "render_mode",
    "max_corridors_in_context",
    "max_main_sentences_per_corridor",
    "max_support_per_corridor",
    "max_total_sentences",
    "delivery_mode",
    "max_chunk_packages",
    "max_excerpt_sentences_per_package",
    "chunk_grounding_enabled",
    "chunk_grounding_mode",
    "chunk_excerpt_max_per_corridor",
    "chunk_excerpt_window_sentences_before",
    "chunk_excerpt_window_sentences_after",
    "chunk_excerpt_max_total_sentences",
    "chunk_excerpt_dedup_enabled",
    "run_qa",
    "retrieval_only",
    "llm_timeout_sec",
    "llm_max_new_tokens",
]

# Best precomputed-locked contract-matched run per dataset
# (prompt_variant=light_separator_copy_span_instruction,
#  order_strategy=score+light_separator_render).
SOURCE_RUNS: Dict[str, Dict[str, str]] = {
    "hotpotqa": {
        "summary_json": "/home/ojungii/EffiRAG/outputs/prompt_interface_bridge_resolution_round/20260428_120903/runs/s2/light_separator_copy_span_instruction/rag/hotpotqa/20260428_031310_987698/rag_summary.json",
        "config_json": "/home/ojungii/EffiRAG/outputs/prompt_interface_bridge_resolution_round/20260428_120903/runs/s2/light_separator_copy_span_instruction/rag/hotpotqa/20260428_031310_987698/logs/config_20260428_031310_987698.json",
    },
    "2wikimultihopqa": {
        "summary_json": "/home/ojungii/EffiRAG/outputs/copy_span_instruction_grouped_round/20260508_154702/runs/stage_b_full/light_separator_copy_span_instruction/rag/2wikimultihopqa/20260508_065715_511649/rag_summary.json",
        "config_json": "/home/ojungii/EffiRAG/outputs/copy_span_instruction_grouped_round/20260508_154702/runs/stage_b_full/light_separator_copy_span_instruction/rag/2wikimultihopqa/20260508_065715_511649/logs/config_20260508_065715_511649.json",
    },
    "musique": {
        "summary_json": "/home/ojungii/EffiRAG/outputs/prompt_interface_bridge_resolution_round/20260501_000216/runs/s0/light_separator_copy_span_instruction/rag/musique/20260430_150219_480752/rag_summary.json",
        "config_json": "/home/ojungii/EffiRAG/outputs/prompt_interface_bridge_resolution_round/20260501_000216/runs/s0/light_separator_copy_span_instruction/rag/musique/20260430_150219_480752/logs/config_20260430_150219_480752.json",
    },
    "popqa": {
        "summary_json": "/home/ojungii/EffiRAG/outputs/prompt_interface_bridge_resolution_round/20260501_000216/runs/s0/light_separator_copy_span_instruction/rag/popqa/20260430_150458_889780/rag_summary.json",
        "config_json": "/home/ojungii/EffiRAG/outputs/prompt_interface_bridge_resolution_round/20260501_000216/runs/s0/light_separator_copy_span_instruction/rag/popqa/20260430_150458_889780/logs/config_20260430_150458_889780.json",
    },
}


def _to_bool_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    vv = str(value or "").strip().lower()
    if vv in {"1", "true", "t", "yes", "y", "on"}:
        return "true"
    if vv in {"0", "false", "f", "no", "n", "off"}:
        return "false"
    return "false"


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    return _load_json(path)


def _rag_config_field_names() -> set[str]:
    return {f.name for f in fields(RagConfig)}


def _filter_rag_config(raw_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = _rag_config_field_names()
    return {k: v for k, v in raw_cfg.items() if k in allowed}


def _normalize_contract_fields(cfg: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    out = dict(cfg)
    out["dataset"] = dataset
    out["method"] = "effirag"
    out["prompt_variant"] = "light_separator_copy_span_instruction"
    out["order_strategy"] = "score+light_separator_render"
    out["timestamp_output"] = True
    out["output_dir"] = f"outputs/sota_config_runs/copy_span_instruction_4ds/{dataset}"
    return out


def _build_locked_cfg(raw_cfg: Mapping[str, Any], dataset: str) -> Dict[str, Any]:
    out = _normalize_contract_fields(_filter_rag_config(raw_cfg), dataset)
    precomputed_path = str(out.get("precomputed_retrieval_path", "") or "").strip()
    out["precomputed_retrieval_path"] = precomputed_path
    out["precomputed_retrieval_strict"] = "true" if precomputed_path else "false"
    return out


def _build_on_the_fly_cfg(raw_cfg: Mapping[str, Any], dataset: str) -> Dict[str, Any]:
    out = _normalize_contract_fields(_filter_rag_config(raw_cfg), dataset)
    out["precomputed_retrieval_path"] = ""
    out["precomputed_retrieval_strict"] = "false"
    return out


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_parent(path)
    text = yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False)
    path.write_text(text, encoding="utf-8")


def _expected_metrics(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "f1": float(summary.get("f1", 0.0) or 0.0),
        "em": float(summary.get("em", 0.0) or 0.0),
        "supporting_fact_recall": float(summary.get("supporting_fact_recall", 0.0) or 0.0),
        "supporting_fact_precision": float(summary.get("supporting_fact_precision", 0.0) or 0.0),
        "supporting_fact_f1": float(summary.get("supporting_fact_f1", 0.0) or 0.0),
        "prompt_tokens_avg": float(summary.get("prompt_tokens_avg", 0.0) or 0.0),
        "retrieval_latency_ms": float(summary.get("retrieval_latency_ms", 0.0) or 0.0),
        "total_latency_ms": float(summary.get("total_latency_ms", 0.0) or 0.0),
        "precomputed_retrieval_used": bool(summary.get("precomputed_retrieval_used", False)),
    }


def _extract_retrieval_contract(config_obj: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in RETRIEVAL_CONTRACT_KEYS:
        if key in config_obj:
            out[key] = config_obj.get(key)
    return out


def _extract_qa_contract(config_obj: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in QA_CONTRACT_KEYS:
        if key in config_obj:
            out[key] = config_obj.get(key)
    return out


def _build_retrieval_provenance(precomputed_query_path: str, max_hops: int = 8) -> Dict[str, Any]:
    start = str(precomputed_query_path or "").strip()
    if not start:
        return {
            "locked_precomputed_query_results": "",
            "chain_depth": 0,
            "chain": [],
            "terminal_on_the_fly_query_results": "",
            "terminal_retrieval_contract": {},
        }

    chain = []
    seen: set[str] = set()
    current = start

    for hop in range(1, max_hops + 1):
        if not current or current in seen:
            break
        seen.add(current)

        query_path = Path(current)
        run_dir = query_path.parent
        summary_path = run_dir / "rag_summary.json"
        logs_dir = run_dir / "logs"
        config_candidates = sorted(logs_dir.glob("config_*.json"))
        config_path = config_candidates[-1] if config_candidates else Path()

        summary_obj = _load_json_if_exists(summary_path)
        config_doc = _load_json_if_exists(config_path)
        cfg = dict(config_doc.get("config", {}) or {})
        next_precomputed = str(cfg.get("precomputed_retrieval_path", "") or "").strip()

        chain.append(
            {
                "hop": hop,
                "run_dir": str(run_dir),
                "query_results_jsonl": str(query_path),
                "summary_json": str(summary_path),
                "config_json": str(config_path) if config_path else "",
                "query_exists": query_path.exists(),
                "summary_exists": summary_path.exists(),
                "config_exists": bool(config_path and config_path.exists()),
                "precomputed_retrieval_used": (
                    bool(summary_obj.get("precomputed_retrieval_used", False)) if summary_obj else None
                ),
                "retrieval_source_breakdown": summary_obj.get("retrieval_source_breakdown", {}) if summary_obj else {},
                "canonical_variant_name": cfg.get("canonical_variant_name"),
                "prompt_variant": cfg.get("prompt_variant"),
                "retrieval_contract": _extract_retrieval_contract(cfg),
                "next_precomputed_retrieval_path": next_precomputed,
            }
        )

        if not next_precomputed:
            break
        current = next_precomputed

    terminal_on_the_fly = ""
    terminal_contract: Dict[str, Any] = {}
    if chain:
        last = chain[-1]
        if not str(last.get("next_precomputed_retrieval_path", "") or "").strip():
            terminal_on_the_fly = str(last.get("query_results_jsonl", ""))
            terminal_contract = dict(last.get("retrieval_contract", {}) or {})

    return {
        "locked_precomputed_query_results": start,
        "chain_depth": len(chain),
        "chain": chain,
        "terminal_on_the_fly_query_results": terminal_on_the_fly,
        "terminal_retrieval_contract": terminal_contract,
    }


def _build_qa_from_replayed_root_cfg(raw_cfg: Mapping[str, Any], dataset: str) -> Dict[str, Any]:
    out = _normalize_contract_fields(_filter_rag_config(raw_cfg), dataset)
    out["run_qa"] = True
    out["retrieval_only"] = False
    out["precomputed_retrieval_path"] = ""
    out["precomputed_retrieval_strict"] = "false"
    return out


def build_sota_config(profile_root: Path) -> Path:
    profile_root.mkdir(parents=True, exist_ok=True)
    locked_root = profile_root / "locked_precomputed"
    live_root = profile_root / "on_the_fly"
    qa_repro_root = profile_root / "qa_from_replayed_root"
    locked_root.mkdir(parents=True, exist_ok=True)
    live_root.mkdir(parents=True, exist_ok=True)
    qa_repro_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "profile_name": "copy_span_instruction_4ds",
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contract": {
            "prompt_variant": "light_separator_copy_span_instruction",
            "order_strategy": "score+light_separator_render",
            "added_instruction_semantics": "copy_span_only (via prompt_variant template)",
        },
        "datasets": {},
    }

    for dataset in DATASET_ORDER:
        src = SOURCE_RUNS[dataset]
        config_path = Path(src["config_json"])
        summary_path = Path(src["summary_json"])
        config_doc = _load_json(config_path)
        summary = _load_json(summary_path)
        raw_cfg = dict(config_doc.get("config", {}) or {})

        locked_cfg = _build_locked_cfg(raw_cfg, dataset)
        live_cfg = _build_on_the_fly_cfg(raw_cfg, dataset)
        qa_repro_cfg = _build_qa_from_replayed_root_cfg(raw_cfg, dataset)

        locked_cfg_path = locked_root / f"{dataset}.yaml"
        live_cfg_path = live_root / f"{dataset}.yaml"
        qa_repro_cfg_path = qa_repro_root / f"{dataset}.yaml"
        _write_yaml(locked_cfg_path, locked_cfg)
        _write_yaml(live_cfg_path, live_cfg)
        _write_yaml(qa_repro_cfg_path, qa_repro_cfg)
        retrieval_provenance = _build_retrieval_provenance(str(locked_cfg.get("precomputed_retrieval_path", "")))
        qa_contract = _extract_qa_contract(qa_repro_cfg)

        manifest["datasets"][dataset] = {
            "source": {
                "config_json": str(config_path),
                "summary_json": str(summary_path),
            },
            "expected_metrics_locked": _expected_metrics(summary),
            "retrieval_provenance": retrieval_provenance,
            "qa_contract": qa_contract,
            "locked_precomputed": {
                "config_yaml": str(locked_cfg_path),
                "precomputed_retrieval_path": str(locked_cfg.get("precomputed_retrieval_path", "")),
                "precomputed_retrieval_strict": _to_bool_str(locked_cfg.get("precomputed_retrieval_strict", "false")),
                "retrieval_objective_mode": str(locked_cfg.get("retrieval_objective_mode", "")),
            },
            "on_the_fly": {
                "config_yaml": str(live_cfg_path),
                "precomputed_retrieval_path": "",
                "precomputed_retrieval_strict": "false",
                "retrieval_objective_mode": str(live_cfg.get("retrieval_objective_mode", "")),
            },
            "qa_from_replayed_root": {
                "config_yaml": str(qa_repro_cfg_path),
                "precomputed_retrieval_path": "__SET_TO_REPLAYED_ROOT_QUERY_RESULTS__",
                "precomputed_retrieval_strict": "true",
            },
        }

    manifest_path = profile_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Update copy-span-instruction SOTA config pack for 4 datasets.")
    parser.add_argument(
        "--profile-root",
        default=str(SOTA_ROOT),
        help="Output directory for SOTA config pack.",
    )
    args = parser.parse_args()

    profile_root = Path(args.profile_root).resolve()
    manifest_path = build_sota_config(profile_root)
    print(str(manifest_path))


if __name__ == "__main__":
    main()
