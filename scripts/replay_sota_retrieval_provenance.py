#!/usr/bin/env python3
"""
Replay retrieval provenance chains from SOTA manifest.

Purpose:
- Do not consume historical precomputed retrieval directly.
- Re-run the retrieval-producing configs hop-by-hop (terminal -> root).
- Compare replayed retrieval outputs against historical references.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "SOTA_config" / "copy_span_instruction_4ds" / "manifest.json"
DATASET_ORDER = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False)
    path.write_text(text, encoding="utf-8")


def _slug(text: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return "unknown"
    out = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return out or "unknown"


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    return float(inter) / float(union) if union > 0 else 0.0


def _extract_selected_sentence_ids(path: Path, limit: int = 0) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    max_rows = int(limit or 0)
    for i, row in enumerate(_iter_jsonl(path)):
        if max_rows > 0 and i >= max_rows:
            break
        sample_id = str(row.get("sample_id", "") or "")
        retrieval = dict(row.get("retrieval", {}) or {})
        selected = retrieval.get("selected_sentence_ids", []) or []
        out[sample_id] = [str(x) for x in selected]
    return out


def _compare_retrieval_outputs(reference_query: Path, replay_query: Path, limit: int = 0) -> Dict[str, object]:
    ref = _extract_selected_sentence_ids(reference_query, limit=limit)
    new = _extract_selected_sentence_ids(replay_query, limit=limit)

    ref_ids = set(ref.keys())
    new_ids = set(new.keys())
    shared = sorted(ref_ids.intersection(new_ids))

    exact = 0
    jaccards: List[float] = []
    len_deltas: List[int] = []
    for sid in shared:
        ref_list = ref.get(sid, [])
        new_list = new.get(sid, [])
        if ref_list == new_list:
            exact += 1
        jaccards.append(_jaccard(ref_list, new_list))
        len_deltas.append(len(new_list) - len(ref_list))

    exact_rate = float(exact) / float(len(shared)) if shared else 0.0
    mean_jaccard = float(sum(jaccards) / float(len(jaccards))) if jaccards else 0.0
    mean_len_delta = float(sum(len_deltas) / float(len(len_deltas))) if len_deltas else 0.0

    return {
        "reference_query_path": str(reference_query),
        "replay_query_path": str(replay_query),
        "reference_samples": int(len(ref)),
        "replay_samples": int(len(new)),
        "shared_samples": int(len(shared)),
        "sample_coverage_rate_vs_reference": (float(len(shared)) / float(len(ref)) if ref else 0.0),
        "exact_selected_sentence_ids_match_rate": exact_rate,
        "mean_selected_sentence_jaccard": mean_jaccard,
        "mean_selected_sentence_count_delta": mean_len_delta,
    }


def _find_latest_summary(base_dir: Path) -> Path:
    summaries = sorted(base_dir.rglob("rag_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No rag_summary.json found under {base_dir}")
    return summaries[-1]


def _build_replay_cfg(
    base_cfg: Mapping[str, object],
    output_dir: str,
    precomputed_path: str,
    retrieval_only: bool,
) -> Dict[str, object]:
    cfg = dict(base_cfg)
    cfg["output_dir"] = output_dir
    cfg["timestamp_output"] = True
    cfg["precomputed_retrieval_path"] = str(precomputed_path or "")
    cfg["precomputed_retrieval_strict"] = bool(precomputed_path)
    if retrieval_only:
        cfg["run_qa"] = False
        cfg["retrieval_only"] = True
    else:
        cfg["run_qa"] = True
        cfg["retrieval_only"] = False
    return cfg


def _parse_bool(text: str, default: bool = True) -> bool:
    token = str(text or "").strip().lower()
    if not token:
        return bool(default)
    if token in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def main() -> None:
    p = argparse.ArgumentParser(description="Replay SOTA retrieval provenance chains without using historical precomputed files.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--datasets", default="hotpotqa,2wikimultihopqa,musique,popqa")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--limit", type=int, default=0, help="When >0, limit samples for quick reproducibility checks.")
    p.add_argument(
        "--retrieval-only",
        default="true",
        help="true/false. true skips generation and focuses on retrieval reproducibility.",
    )
    p.add_argument(
        "--run-qa-from-replayed-root",
        default="true",
        help="true/false. After retrieval replay, run QA with SOTA QA config and replayed root retrieval.",
    )
    p.add_argument("--output-root", default="")
    args = p.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = _load_json(manifest_path)

    requested = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    datasets = [d for d in DATASET_ORDER if d in requested]
    if not datasets:
        raise ValueError(f"No valid datasets requested: {requested}")

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        output_root = (REPO_ROOT / "outputs" / "sota_retrieval_replay" / f"{_ts()}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    retrieval_only = _parse_bool(str(args.retrieval_only), default=True)
    run_qa_from_replayed_root = _parse_bool(str(args.run_qa_from_replayed_root), default=True)
    records: List[Dict[str, object]] = []

    datasets_manifest = dict(manifest.get("datasets", {}) or {})
    for dataset in datasets:
        ds_payload = dict(datasets_manifest.get(dataset, {}) or {})
        rp = dict(ds_payload.get("retrieval_provenance", {}) or {})
        chain = list(rp.get("chain", []) or [])
        if not chain:
            raise RuntimeError(f"Missing retrieval provenance chain for dataset={dataset}")

        replay_map: Dict[str, str] = {}
        hop_records: List[Dict[str, object]] = []
        ds_root = output_root / dataset
        ds_root.mkdir(parents=True, exist_ok=True)

        for hop in reversed(chain):
            hop_id = int(hop.get("hop", 0) or 0)
            ref_query = Path(str(hop.get("query_results_jsonl", "") or "")).resolve()
            cfg_json = Path(str(hop.get("config_json", "") or "")).resolve()
            if not cfg_json.exists():
                raise FileNotFoundError(f"Missing hop config_json: {cfg_json}")
            cfg_doc = _load_json(cfg_json)
            base_cfg = dict(cfg_doc.get("config", {}) or {})

            next_ref_query = str(hop.get("next_precomputed_retrieval_path", "") or "").strip()
            replay_precomputed = ""
            if next_ref_query:
                replay_precomputed = replay_map.get(next_ref_query, "")
                if not replay_precomputed:
                    raise RuntimeError(
                        "Replay chain is broken: next hop replay artifact not found "
                        f"(dataset={dataset}, hop={hop_id}, next_ref={next_ref_query})"
                    )

            canonical = _slug(str(hop.get("canonical_variant_name", "") or "unknown"))
            hop_base_out = ds_root / f"hop{hop_id}_{canonical}"
            cfg_out = output_root / "configs" / dataset / f"hop{hop_id}_{canonical}.yaml"

            replay_cfg = _build_replay_cfg(
                base_cfg=base_cfg,
                output_dir=str(hop_base_out),
                precomputed_path=str(replay_precomputed),
                retrieval_only=retrieval_only,
            )
            _write_yaml(cfg_out, replay_cfg)

            cmd = [
                args.python_bin,
                "-m",
                "effirag.run_rag",
                "--config",
                str(cfg_out),
                "--output-dir",
                str(hop_base_out),
                "--timestamp-output",
                "true",
            ]
            if int(args.limit or 0) > 0:
                cmd.extend(["--limit", str(int(args.limit))])

            print(f"\n=== [replay] dataset={dataset} hop={hop_id} canonical={canonical}")
            print(" ".join(cmd))
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
            if int(proc.returncode) != 0:
                raise RuntimeError(
                    "Replay hop run failed "
                    f"(dataset={dataset}, hop={hop_id}, exit_code={int(proc.returncode)}). "
                    f"config={cfg_out}"
                )

            summary_path = _find_latest_summary(hop_base_out)
            replay_query = summary_path.with_name("rag_query_results.jsonl")
            if not replay_query.exists():
                raise FileNotFoundError(f"Missing replay query file: {replay_query}")

            replay_map[str(ref_query)] = str(replay_query)
            compare = _compare_retrieval_outputs(ref_query, replay_query, limit=int(args.limit or 0))
            replay_summary = dict(_load_json(summary_path))

            hop_record = {
                "dataset": dataset,
                "hop": hop_id,
                "canonical_variant_name": str(hop.get("canonical_variant_name", "")),
                "reference_query_results_jsonl": str(ref_query),
                "reference_config_json": str(cfg_json),
                "replay_config_yaml": str(cfg_out),
                "replay_output_dir": str(summary_path.parent),
                "replay_summary_json": str(summary_path),
                "replay_query_results_jsonl": str(replay_query),
                "replay_used_precomputed_query_results": str(replay_precomputed),
                "exit_code": int(proc.returncode),
                "replay_summary_metrics": {
                    "supporting_fact_recall": _safe_float(replay_summary.get("supporting_fact_recall", 0.0), 0.0),
                    "supporting_fact_precision": _safe_float(replay_summary.get("supporting_fact_precision", 0.0), 0.0),
                    "f1": _safe_float(replay_summary.get("f1", 0.0), 0.0),
                    "em": _safe_float(replay_summary.get("em", 0.0), 0.0),
                    "precomputed_retrieval_used": bool(replay_summary.get("precomputed_retrieval_used", False)),
                    "retrieval_source_breakdown": replay_summary.get("retrieval_source_breakdown", {}),
                },
                "retrieval_match": compare,
            }
            hop_records.append(hop_record)

        root_ref = str(rp.get("locked_precomputed_query_results", "") or "").strip()
        root_replay = replay_map.get(root_ref, "")
        root_compare = (
            _compare_retrieval_outputs(Path(root_ref), Path(root_replay), limit=int(args.limit or 0))
            if root_ref and root_replay
            else {}
        )

        qa_replay: Dict[str, object] = {}
        if run_qa_from_replayed_root:
            if not root_replay:
                raise RuntimeError(f"Missing replayed root retrieval for QA replay (dataset={dataset})")

            qa_section = dict(ds_payload.get("qa_from_replayed_root", {}) or {})
            qa_cfg_path_raw = str(qa_section.get("config_yaml", "") or "").strip()
            if not qa_cfg_path_raw:
                raise RuntimeError(
                    f"Missing qa_from_replayed_root.config_yaml in manifest (dataset={dataset})"
                )
            qa_cfg_path = Path(qa_cfg_path_raw).resolve()
            if not qa_cfg_path.exists():
                raise FileNotFoundError(f"QA replay config not found: {qa_cfg_path}")

            qa_out_base = ds_root / "qa_from_replayed_root"
            qa_cmd = [
                args.python_bin,
                "-m",
                "effirag.run_rag",
                "--config",
                str(qa_cfg_path),
                "--precomputed-retrieval-path",
                str(root_replay),
                "--precomputed-retrieval-strict",
                "true",
                "--output-dir",
                str(qa_out_base),
                "--timestamp-output",
                "true",
            ]
            if int(args.limit or 0) > 0:
                qa_cmd.extend(["--limit", str(int(args.limit))])

            print(f"\n=== [qa-replay] dataset={dataset} from_replayed_root")
            print(" ".join(qa_cmd))
            qa_proc = subprocess.run(qa_cmd, cwd=str(REPO_ROOT), check=False)
            if int(qa_proc.returncode) != 0:
                raise RuntimeError(
                    "QA replay run failed "
                    f"(dataset={dataset}, exit_code={int(qa_proc.returncode)}). config={qa_cfg_path}"
                )

            qa_summary_path = _find_latest_summary(qa_out_base)
            qa_summary = dict(_load_json(qa_summary_path))
            expected = dict(ds_payload.get("expected_metrics_locked", {}) or {})

            obs_f1 = _safe_float(qa_summary.get("f1", 0.0), 0.0)
            obs_em = _safe_float(qa_summary.get("em", 0.0), 0.0)
            obs_sf = _safe_float(qa_summary.get("supporting_fact_recall", 0.0), 0.0)
            exp_f1 = _safe_float(expected.get("f1", 0.0), 0.0)
            exp_em = _safe_float(expected.get("em", 0.0), 0.0)
            exp_sf = _safe_float(expected.get("supporting_fact_recall", 0.0), 0.0)

            qa_replay = {
                "enabled": True,
                "config_yaml": str(qa_cfg_path),
                "output_dir": str(qa_summary_path.parent),
                "summary_json": str(qa_summary_path),
                "used_replayed_root_query_results": str(root_replay),
                "metrics": {
                    "f1": obs_f1,
                    "em": obs_em,
                    "supporting_fact_recall": obs_sf,
                    "precomputed_retrieval_used": bool(qa_summary.get("precomputed_retrieval_used", False)),
                    "retrieval_source_breakdown": qa_summary.get("retrieval_source_breakdown", {}),
                },
                "expected_locked_metrics": {
                    "f1": exp_f1,
                    "em": exp_em,
                    "supporting_fact_recall": exp_sf,
                },
                "delta_vs_expected_locked": {
                    "f1": obs_f1 - exp_f1,
                    "em": obs_em - exp_em,
                    "supporting_fact_recall": obs_sf - exp_sf,
                },
            }
        else:
            qa_replay = {"enabled": False}

        records.append(
            {
                "dataset": dataset,
                "retrieval_only": bool(retrieval_only),
                "run_qa_from_replayed_root": bool(run_qa_from_replayed_root),
                "limit": int(args.limit or 0),
                "chain_depth": int(rp.get("chain_depth", 0) or 0),
                "root_reference_query_results": root_ref,
                "root_replay_query_results": root_replay,
                "root_retrieval_match": root_compare,
                "qa_replay": qa_replay,
                "hops": sorted(hop_records, key=lambda x: int(x.get("hop", 0))),
            }
        )

    out_manifest = {
        "task": "sota_retrieval_provenance_replay",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_source": str(manifest_path),
        "datasets": datasets,
        "retrieval_only": bool(retrieval_only),
        "run_qa_from_replayed_root": bool(run_qa_from_replayed_root),
        "limit": int(args.limit or 0),
        "output_root": str(output_root),
        "records": records,
    }
    out_path = output_root / "replay_manifest.json"
    _write_json(out_path, out_manifest)
    print(f"\nReplay manifest: {out_path}")


if __name__ == "__main__":
    main()
