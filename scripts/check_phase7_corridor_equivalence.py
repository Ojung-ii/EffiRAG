#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sha(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, 95))


def _resolve_dataset_dir(root: Path, dataset: str) -> Path:
    direct = root / dataset
    if (direct / "rag_query_results.jsonl").exists():
        return direct
    if (root / "rag_query_results.jsonl").exists():
        return root
    raise FileNotFoundError(f"Could not resolve dataset dir: root={root}, dataset={dataset}")


def _load_query_rows(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(run_dir / "rag_query_results.jsonl"):
        qid = str(row.get("sample_id", "") or "")
        if not qid:
            continue
        retrieval = dict((row.get("retrieval", {}) or {}))
        rendered = dict((row.get("rendered", {}) or {}))
        out[qid] = {
            "selected_sentence_ids": [str(x) for x in list(retrieval.get("selected_sentence_ids", []) or [])],
            "rendered_sentence_ids": [str(x) for x in list(rendered.get("sentence_ids", []) or row.get("rendered_sentence_ids", []) or [])],
            "rendered_hash": _sha(str(rendered.get("text", "") or "")),
            "candidate_count": int(len(list(retrieval.get("candidate_sentence_ids", []) or []))),
            "selected_count": int(len(list(retrieval.get("selected_sentence_ids", []) or []))),
        }
    return out


def _load_diag_rows(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(run_dir / "phase7_diagnostics.jsonl"):
        qid = str(row.get("qid", "") or row.get("query_id", "") or "")
        if not qid:
            continue
        candidate_atoms = list(row.get("phase7_candidate_atoms", []) or [])
        bq_by_source: Dict[str, float] = {}
        path_cnt_by_source: Dict[str, int] = {}
        for atom in candidate_atoms:
            sid = str(atom.get("source_id", "") or "")
            if not sid:
                continue
            bq_by_source[sid] = _safe_float(atom.get("corridor_score", 0.0), 0.0)
            path_cnt_by_source[sid] = int(atom.get("corridor_path_count", 0) or 0)
        out[qid] = {
            "bq_by_source": bq_by_source,
            "path_cnt_by_source": path_cnt_by_source,
            "selected_ids": [str(x) for x in list(row.get("selected_evidence_ids", []) or [])],
            "corridor_diag": dict(row.get("phase7_corridor_diagnostics", {}) or {}),
        }
    return out


def _load_corridor_ms(run_dir: Path) -> List[float]:
    trace_rows = _read_jsonl(run_dir / "phase7_query_trace.jsonl")
    ms = []
    for row in trace_rows:
        stage = dict((row.get("stage_ms", {}) or {}))
        if "corridor_extraction" in stage:
            ms.append(_safe_float(stage.get("corridor_extraction", 0.0), 0.0))
    if ms:
        return ms

    timing_rows = _read_jsonl(run_dir / "phase7_stage_timing.jsonl")
    for row in timing_rows:
        if str(row.get("stage", "")) != "corridor_extraction":
            continue
        ms.append(_safe_float(row.get("elapsed_ms", 0.0), 0.0))
    return ms


def _fmt4(x: float) -> str:
    return f"{float(x):.4f}"


def _fmt2(x: float) -> str:
    return f"{float(x):.2f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Phase7 corridor optimization equivalence before/after.")
    ap.add_argument("--old-root", type=str, required=True)
    ap.add_argument("--new-root", type=str, required=True)
    ap.add_argument("--datasets", type=str, default="hotpotqa,2wikimultihopqa")
    ap.add_argument(
        "--out",
        type=str,
        default="outputs/phase7_evidence_flow/optimization/corridor_equivalence_report.md",
    )
    args = ap.parse_args()

    old_root = Path(args.old_root).resolve()
    new_root = Path(args.new_root).resolve()
    datasets = [str(x).strip() for x in str(args.datasets).split(",") if str(x).strip()]

    rows_md: List[List[str]] = []
    details: Dict[str, Any] = {}

    for ds in datasets:
        old_dir = _resolve_dataset_dir(old_root, ds)
        new_dir = _resolve_dataset_dir(new_root, ds)

        old_rows = _load_query_rows(old_dir)
        new_rows = _load_query_rows(new_dir)
        old_diag = _load_diag_rows(old_dir)
        new_diag = _load_diag_rows(new_dir)

        common_ids = sorted(set(old_rows.keys()).intersection(set(new_rows.keys())))
        n = int(len(common_ids))

        selected_match = 0
        rendered_hash_match = 0
        candidate_count_equal = 0
        selected_count_equal = 0
        candidate_diff_abs: List[float] = []
        selected_diff_abs: List[float] = []
        mismatches: List[Dict[str, Any]] = []

        for qid in common_ids:
            o = old_rows[qid]
            nrow = new_rows[qid]
            if list(o["selected_sentence_ids"]) == list(nrow["selected_sentence_ids"]):
                selected_match += 1
            else:
                old_bq_map = dict((old_diag.get(qid, {}) or {}).get("bq_by_source", {}) or {})
                new_bq_map = dict((new_diag.get(qid, {}) or {}).get("bq_by_source", {}) or {})
                old_path_cnt_map = dict((old_diag.get(qid, {}) or {}).get("path_cnt_by_source", {}) or {})
                new_path_cnt_map = dict((new_diag.get(qid, {}) or {}).get("path_cnt_by_source", {}) or {})
                old_sel = list(o["selected_sentence_ids"])
                new_sel = list(nrow["selected_sentence_ids"])
                mismatches.append(
                    {
                        "query_id": qid,
                        "old_selected_ids": old_sel,
                        "new_selected_ids": new_sel,
                        "old_bq_scores": {sid: _safe_float(old_bq_map.get(sid, 0.0), 0.0) for sid in old_sel},
                        "new_bq_scores": {sid: _safe_float(new_bq_map.get(sid, 0.0), 0.0) for sid in new_sel},
                        "old_corridor_path_counts": {sid: int(old_path_cnt_map.get(sid, 0) or 0) for sid in old_sel},
                        "new_corridor_path_counts": {sid: int(new_path_cnt_map.get(sid, 0) or 0) for sid in new_sel},
                        "old_paths_found": int(((old_diag.get(qid, {}) or {}).get("corridor_diag", {}) or {}).get("num_paths_found", 0) or 0),
                        "new_paths_found": int(((new_diag.get(qid, {}) or {}).get("corridor_diag", {}) or {}).get("num_paths_found", 0) or 0),
                    }
                )
            if str(o["rendered_hash"]) == str(nrow["rendered_hash"]):
                rendered_hash_match += 1
            if int(o["candidate_count"]) == int(nrow["candidate_count"]):
                candidate_count_equal += 1
            if int(o["selected_count"]) == int(nrow["selected_count"]):
                selected_count_equal += 1
            candidate_diff_abs.append(abs(float(o["candidate_count"]) - float(nrow["candidate_count"])))
            selected_diff_abs.append(abs(float(o["selected_count"]) - float(nrow["selected_count"])))

        old_p95 = _p95(_load_corridor_ms(old_dir))
        new_p95 = _p95(_load_corridor_ms(new_dir))

        selected_rate = float(selected_match / n) if n > 0 else 0.0
        rendered_rate = float(rendered_hash_match / n) if n > 0 else 0.0
        candidate_eq_rate = float(candidate_count_equal / n) if n > 0 else 0.0
        selected_eq_rate = float(selected_count_equal / n) if n > 0 else 0.0

        rows_md.append(
            [
                ds,
                str(n),
                _fmt4(candidate_eq_rate),
                _fmt4(selected_eq_rate),
                _fmt4(selected_rate),
                _fmt4(rendered_rate),
                _fmt2(old_p95),
                _fmt2(new_p95),
            ]
        )

        details[ds] = {
            "query_count": n,
            "selected_id_match_rate": selected_rate,
            "rendered_hash_match_rate": rendered_rate,
            "candidate_count_match_rate": candidate_eq_rate,
            "selected_count_match_rate": selected_eq_rate,
            "avg_abs_candidate_count_diff": float(mean(candidate_diff_abs)) if candidate_diff_abs else 0.0,
            "avg_abs_selected_count_diff": float(mean(selected_diff_abs)) if selected_diff_abs else 0.0,
            "old_corridor_extraction_p95_ms": float(old_p95),
            "new_corridor_extraction_p95_ms": float(new_p95),
            "old_dir": str(old_dir),
            "new_dir": str(new_dir),
            "mismatches_top": mismatches[:20],
        }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md: List[str] = []
    md.append("# Phase7 Corridor Equivalence Report")
    md.append("")
    md.append(f"- old_root: `{old_root}`")
    md.append(f"- new_root: `{new_root}`")
    md.append("")
    md.append(
        _table(
            [
                "dataset",
                "n",
                "candidate_count_match_rate",
                "selected_count_match_rate",
                "selected_id_match_rate",
                "rendered_hash_match_rate",
                "old_p95_ms",
                "new_p95_ms",
            ],
            rows_md,
        )
    )
    md.append("")
    md.append("## Notes")
    md.append("- selected_id_match_rate compares exact selected sentence-id sequence per query.")
    md.append("- rendered_hash_match_rate compares SHA256 of rendered context text per query.")
    md.append("- old/new p95 use `stage_ms.corridor_extraction` when available.")
    md.append("- If selected_id_match_rate < 0.95, see `corridor_equivalence_report.json` mismatches_top for query-level details.")
    out_path.write_text("\n".join(md), encoding="utf-8")

    (out_path.parent / "corridor_equivalence_report.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(details, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
