from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def perf_counter_ns_now() -> int:
    return int(time.perf_counter_ns())


@dataclass
class _StageFrame:
    stage: str
    start_perf_ns: int
    start_wall_utc: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Phase7StageTracer:
    def __init__(self, run_id: str, dataset: str, query_id: str):
        self.run_id = str(run_id or "")
        self.dataset = str(dataset or "")
        self.query_id = str(query_id or "")
        self._frames: Dict[str, _StageFrame] = {}
        self.stage_events: List[Dict[str, Any]] = []
        self.stage_spans: Dict[str, Dict[str, Any]] = {}

    def start(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        name = str(stage or "").strip()
        if not name:
            return
        frame = _StageFrame(
            stage=name,
            start_perf_ns=perf_counter_ns_now(),
            start_wall_utc=iso_utc_now(),
            metadata=dict(metadata or {}),
        )
        self._frames[name] = frame
        self.stage_events.append(
            {
                "run_id": self.run_id,
                "dataset": self.dataset,
                "query_id": self.query_id,
                "stage": name,
                "event": "start",
                "wall_time_utc": str(frame.start_wall_utc),
                "perf_counter_ns": int(frame.start_perf_ns),
                "elapsed_ms": 0.0,
                "metadata": dict(frame.metadata),
            }
        )

    def end(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        name = str(stage or "").strip()
        if not name:
            return 0.0
        end_perf = perf_counter_ns_now()
        end_wall = iso_utc_now()
        frame = self._frames.pop(name, None)
        if frame is None:
            elapsed_ms = 0.0
            start_perf = end_perf
            start_wall = end_wall
            start_meta: Dict[str, Any] = {}
        else:
            elapsed_ms = float(max(0, end_perf - int(frame.start_perf_ns)) / 1_000_000.0)
            start_perf = int(frame.start_perf_ns)
            start_wall = str(frame.start_wall_utc)
            start_meta = dict(frame.metadata or {})
        end_meta = dict(start_meta)
        end_meta.update(dict(metadata or {}))

        self.stage_events.append(
            {
                "run_id": self.run_id,
                "dataset": self.dataset,
                "query_id": self.query_id,
                "stage": name,
                "event": "end",
                "wall_time_utc": str(end_wall),
                "perf_counter_ns": int(end_perf),
                "elapsed_ms": float(elapsed_ms),
                "metadata": dict(end_meta),
            }
        )
        self.stage_spans[name] = {
            "stage": name,
            "start_time_utc": str(start_wall),
            "end_time_utc": str(end_wall),
            "start_perf_counter_ns": int(start_perf),
            "end_perf_counter_ns": int(end_perf),
            "elapsed_ms": float(elapsed_ms),
            "metadata": dict(end_meta),
            "skipped": False,
            "skip_reason": None,
        }
        return float(elapsed_ms)

    def skip(self, stage: str, skip_reason: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        name = str(stage or "").strip()
        if not name:
            return
        now_perf = perf_counter_ns_now()
        now_wall = iso_utc_now()
        payload = dict(metadata or {})
        payload["skip_reason"] = str(skip_reason or "not_applicable")
        self.stage_events.append(
            {
                "run_id": self.run_id,
                "dataset": self.dataset,
                "query_id": self.query_id,
                "stage": name,
                "event": "skip",
                "wall_time_utc": str(now_wall),
                "perf_counter_ns": int(now_perf),
                "elapsed_ms": 0.0,
                "metadata": dict(payload),
            }
        )
        self.stage_spans[name] = {
            "stage": name,
            "start_time_utc": str(now_wall),
            "end_time_utc": str(now_wall),
            "start_perf_counter_ns": int(now_perf),
            "end_perf_counter_ns": int(now_perf),
            "elapsed_ms": 0.0,
            "metadata": dict(payload),
            "skipped": True,
            "skip_reason": str(skip_reason or "not_applicable"),
        }

