import json
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .phase7_config import Phase7Config
from .utils import timestamp_iso_utc


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _git_field(args: List[str], default: str = "") -> str:
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
        return str(out or default)
    except Exception:
        return str(default)


def _git_branch() -> str:
    return _git_field(["git", "branch", "--show-current"], "")


def _git_commit() -> str:
    return _git_field(["git", "rev-parse", "HEAD"], "")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Phase7RunLogger:
    def __init__(self, output_dir: Path, dataset: str, limit: int, config_path: str, cfg: Phase7Config):
        self.output_dir = Path(output_dir)
        self.dataset = str(dataset or "")
        self.limit = int(limit or 0)
        self.config_path = str(config_path or "")
        self.cfg = cfg
        self.run_id = f"{self.dataset}:{uuid.uuid4().hex[:16]}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.output_dir / "phase7_run_manifest.json"
        self.query_trace_path = self.output_dir / "phase7_query_trace.jsonl"
        self.stage_timing_path = self.output_dir / "phase7_stage_timing.jsonl"
        self.stage_events_path = self.output_dir / "phase7_stage_events.jsonl"
        self.oracle_gap_trace_path = self.output_dir / "phase7_oracle_gap_trace.jsonl"
        self.index_summary_path = self.output_dir / "phase7_index_summary.json"

        if not self.query_trace_path.exists():
            self.query_trace_path.write_text("", encoding="utf-8")
        if not self.stage_timing_path.exists():
            self.stage_timing_path.write_text("", encoding="utf-8")
        if not self.stage_events_path.exists():
            self.stage_events_path.write_text("", encoding="utf-8")
        if not self.oracle_gap_trace_path.exists():
            self.oracle_gap_trace_path.write_text("", encoding="utf-8")

        if not self.manifest_path.exists():
            self._write_manifest(started=True, completed=False)

    def _phase7_cfg_payload(self) -> Dict[str, Any]:
        payload = asdict(self.cfg)
        return {
            "enabled": bool(payload.get("enabled", True)),
            "variant": str(payload.get("variant", "full") or "full"),
            "enable_phase2_refinement": bool(payload.get("enable_phase2_refinement", False)),
            "objective_mode": str(payload.get("objective_mode", "normalized_equal_weight") or "normalized_equal_weight"),
            "lambda_bridge": float(payload.get("lambda_bridge", 1.0) or 1.0),
            "lambda_decay": float(payload.get("lambda_decay", 1.0) or 1.0),
            "lambda_bq": float(payload.get("lambda_bq", 1.0) or 1.0),
            "mu_redundancy": float(payload.get("mu_redundancy", 1.0) or 1.0),
            "conditional_redundancy_enabled": bool(payload.get("conditional_redundancy_enabled", False)),
            "require_sentence_layer": bool(payload.get("require_sentence_layer", True)),
            "build_sentence_layer": bool(payload.get("build_sentence_layer", True)),
            "build_carrier_layer": bool(payload.get("build_carrier_layer", True)),
            "link_sentence_to_carrier": bool(payload.get("link_sentence_to_carrier", True)),
            "link_sentence_to_entities": bool(payload.get("link_sentence_to_entities", True)),
            "atom_unit": payload.get("atom_unit", "sentence"),
            "carrier_unit": payload.get("carrier_unit", "sentence_window"),
            "carrier_chunk_size_sentences": int(payload.get("carrier_chunk_size_sentences", 3)),
            "carrier_chunk_stride_sentences": int(payload.get("carrier_chunk_stride_sentences", 1)),
            "semantic_anchor_top_t": int(payload.get("semantic_anchor_top_t", 128)),
            "candidate_top_m": int(payload.get("candidate_top_m", 96)),
            "flow_alpha": float(payload.get("flow_alpha", 0.15)),
            "max_flow_iterations": int(payload.get("max_flow_iterations", 30)),
            "flow_tolerance": float(payload.get("flow_tolerance", 1.0e-6)),
            "max_selected_atoms": int(payload.get("max_selected_atoms", 8)),
            "max_context_tokens": int(payload.get("max_context_tokens", 520)),
            "allow_empty_candidates_for_debug": bool(payload.get("allow_empty_candidates_for_debug", False)),
            "diagnostics_enabled": bool(payload.get("diagnostics_enabled", False)),
            "diagnostics_max_examples_to_dump": int(payload.get("diagnostics_max_examples_to_dump", 100)),
            "diagnostics_dump_text": bool(payload.get("diagnostics_dump_text", True)),
            "diagnostics_dump_context": bool(payload.get("diagnostics_dump_context", True)),
            "diagnostics_dump_scores": bool(payload.get("diagnostics_dump_scores", True)),
            "diagnostics_fail_on_unit_mismatch": bool(payload.get("diagnostics_fail_on_unit_mismatch", False)),
            "corridor_enabled": bool(payload.get("corridor_enabled", False)),
            "corridor_max_anchors": int(payload.get("corridor_max_anchors", 4)),
            "corridor_max_seeds": int(payload.get("corridor_max_seeds", 16)),
            "corridor_max_hops": int(payload.get("corridor_max_hops", 3)),
            "corridor_max_paths_per_pair": int(payload.get("corridor_max_paths_per_pair", 1)),
            "corridor_degree_cap": int(payload.get("corridor_degree_cap", 100)),
            "corridor_max_pairs": int(payload.get("corridor_max_pairs", 64)),
            "anchor_decay_enabled": bool(payload.get("anchor_decay_enabled", False)),
            "anchor_decay_gamma": float(payload.get("anchor_decay_gamma", 0.7) or 0.7),
            "anchor_decay_max_hops": int(payload.get("anchor_decay_max_hops", 4)),
            "source_balanced_proposal_enabled": bool(payload.get("source_balanced_proposal_enabled", False)),
            "source_balanced_candidate_top_m": int(payload.get("source_balanced_candidate_top_m", 128)),
            "source_balanced_fill_remaining": bool(payload.get("source_balanced_fill_remaining", True)),
            "source_quota_semantic": int(payload.get("source_quota_semantic", 48)),
            "source_quota_entity_title": int(payload.get("source_quota_entity_title", 32)),
            "source_quota_graph_flow": int(payload.get("source_quota_graph_flow", 32)),
            "source_quota_anchor_neighborhood": int(payload.get("source_quota_anchor_neighborhood", 16)),
            "chain_unit_enabled": bool(payload.get("chain_unit_enabled", False)),
            "chain_unit_max_pair_units": int(payload.get("chain_unit_max_pair_units", 256)),
            "chain_unit_use_same_title": bool(payload.get("chain_unit_use_same_title", True)),
            "chain_unit_use_explicit_transition": bool(payload.get("chain_unit_use_explicit_transition", True)),
            "chain_unit_use_same_carrier": bool(payload.get("chain_unit_use_same_carrier", True)),
            "chain_unit_use_shared_entity": bool(payload.get("chain_unit_use_shared_entity", False)),
            "chain_unit_max_unit_size": int(payload.get("chain_unit_max_unit_size", 2)),
        }

    def _write_manifest(self, started: bool, completed: bool) -> None:
        existing = {}
        if self.manifest_path.exists():
            try:
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        now = timestamp_iso_utc()
        manifest = {
            "method": "phase7_evidence_flow",
            "git_branch": str(existing.get("git_branch") or _git_branch()),
            "git_commit": str(existing.get("git_commit") or _git_commit()),
            "dataset": self.dataset,
            "limit": self.limit,
            "config_path": self.config_path,
            "output_dir": str(self.output_dir),
            "run_id": str(self.run_id),
            "started_at": str(existing.get("started_at") or (now if started else "")),
            "completed_at": str(now if completed else existing.get("completed_at", "")),
            "phase7_config": self._phase7_cfg_payload(),
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def log_query(
        self,
        query_trace: Dict[str, Any],
        stage_rows: Iterable[Dict[str, Any]],
        stage_events: Iterable[Dict[str, Any]] | None = None,
        oracle_gap_trace: Dict[str, Any] | None = None,
    ) -> None:
        _append_jsonl(self.query_trace_path, dict(query_trace or {}))
        for row in list(stage_rows or []):
            _append_jsonl(self.stage_timing_path, dict(row or {}))
        for row in list(stage_events or []):
            _append_jsonl(self.stage_events_path, dict(row or {}))
        if oracle_gap_trace is not None:
            _append_jsonl(self.oracle_gap_trace_path, dict(oracle_gap_trace or {}))

    def log_index_summary(self, summary: Dict[str, Any]) -> None:
        payload = dict(summary or {})
        payload["logged_at"] = timestamp_iso_utc()
        self.index_summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def finalize(self) -> None:
        self._write_manifest(started=True, completed=True)


_LOGGER_BY_OUTDIR: Dict[str, Phase7RunLogger] = {}


def get_phase7_logger(output_dir: str, dataset: str, limit: int, config_path: str, cfg: Phase7Config) -> Phase7RunLogger:
    key = str(Path(output_dir).resolve())
    logger = _LOGGER_BY_OUTDIR.get(key)
    if logger is None:
        logger = Phase7RunLogger(
            output_dir=Path(output_dir),
            dataset=dataset,
            limit=limit,
            config_path=config_path,
            cfg=cfg,
        )
        _LOGGER_BY_OUTDIR[key] = logger
    return logger


def append_phase7_runtime_stages(output_dir: str, query_id: str, generation_ms: float, total_query_ms: float) -> None:
    out_dir = Path(str(output_dir or "")).resolve()
    path = out_dir / "phase7_stage_timing.jsonl"
    event_path = out_dir / "phase7_stage_events.jsonl"
    if not path.exists():
        return
    rows = [
        {
            "query_id": str(query_id or ""),
            "stage": "generation",
            "elapsed_ms": _safe_float(generation_ms, 0.0),
            "num_nodes": 0,
            "num_edges": 0,
            "num_candidates_in": 0,
            "num_candidates_out": 0,
        },
        {
            "query_id": str(query_id or ""),
            "stage": "total_query",
            "elapsed_ms": _safe_float(total_query_ms, 0.0),
            "num_nodes": 0,
            "num_edges": 0,
            "num_candidates_in": 0,
            "num_candidates_out": 0,
        },
    ]
    for row in rows:
        _append_jsonl(path, row)
    if event_path.exists():
        for row in rows:
            _append_jsonl(
                event_path,
                {
                    "run_id": "",
                    "dataset": "",
                    "query_id": str(query_id or ""),
                    "stage": str(row.get("stage", "") or ""),
                    "event": "end",
                    "wall_time_utc": timestamp_iso_utc(),
                    "perf_counter_ns": 0,
                    "elapsed_ms": _safe_float(row.get("elapsed_ms", 0.0), 0.0),
                    "metadata": {
                        "source": "append_phase7_runtime_stages",
                    },
                },
            )


def finalize_phase7_manifest(output_dir: str) -> None:
    out_dir = Path(str(output_dir or "")).resolve()
    key = str(out_dir)
    logger = _LOGGER_BY_OUTDIR.get(key)
    if logger is None:
        manifest = out_dir / "phase7_run_manifest.json"
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            payload["completed_at"] = timestamp_iso_utc()
            manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    logger.finalize()
