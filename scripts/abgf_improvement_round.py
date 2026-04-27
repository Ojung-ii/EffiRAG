#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from effirag.eval_metrics import write_json


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _strip(v: Any) -> str:
    return str(v or "").strip()


def _parse_csv(raw: str) -> List[str]:
    out = []
    for x in str(raw or "").split(","):
        xx = x.strip()
        if xx:
            out.append(xx)
    return out


def _bool(v: Any) -> bool:
    t = str(v or "").strip().lower()
    return t in {"1", "true", "yes", "y", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cmd(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(args), text=True).strip()
    except Exception:
        return ""


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        rr = csv.DictReader(f, delimiter="\t")
        for row in rr:
            rows.append({str(k): str(v or "") for k, v in dict(row).items()})
    return rows


def _write_tsv(path: Path, headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        ww = csv.DictWriter(f, fieldnames=list(headers), delimiter="\t")
        ww.writeheader()
        for row in list(rows or []):
            ww.writerow({h: row.get(h, "") for h in headers})


def _discover_round_root(root: Path) -> Optional[Path]:
    if (root / "run_records.tsv").exists():
        return root
    if not root.exists():
        return None
    cands = sorted([p for p in root.iterdir() if p.is_dir() and (p / "run_records.tsv").exists()])
    if not cands:
        return None
    return cands[-1]


def _find_champion_run(round_root: Path, dataset: str) -> Optional[Dict[str, str]]:
    rows = _read_tsv(round_root / "run_records.tsv")
    hits = []
    for row in rows:
        if _strip(row.get("dataset")) != dataset:
            continue
        if _strip(row.get("variant")) != "champion_config":
            continue
        if _strip(row.get("status")) not in {"ok", "skipped"}:
            continue
        sp = Path(_strip(row.get("summary_path")))
        cp = Path(_strip(row.get("config_path")))
        if not sp.exists() or not cp.exists():
            continue
        hits.append(dict(row))
    if not hits:
        return None
    hits.sort(key=lambda x: (_strip(x.get("end_ts")), _strip(x.get("start_ts"))))
    return hits[-1]


def _find_dataset_path(qa_root: Path, dataset: str) -> Path:
    p = qa_root / f"{dataset}.json"
    if p.exists():
        return p
    fallback = REPO_ROOT / "data" / f"{dataset}.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"QA path not found for dataset={dataset}")


def _find_corpus_path(corpus_root: Path, dataset: str) -> Path:
    p = corpus_root / f"{dataset}_corpus.json"
    if p.exists():
        return p
    fallback = REPO_ROOT / "data" / f"{dataset}_corpus.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Corpus path not found for dataset={dataset}")


def _append_flags(order_strategy: str, flags: Sequence[str]) -> str:
    parts = [_strip(x) for x in str(order_strategy or "score").split("+") if _strip(x)]
    seen = set(parts)
    for f in list(flags or []):
        ff = _strip(f)
        if not ff or ff in seen:
            continue
        parts.append(ff)
        seen.add(ff)
    return "+".join(parts) if parts else "score"


def _load_yaml(path: Path) -> Dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _dump_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload or {}), sort_keys=False, allow_unicode=False), encoding="utf-8")


def _retrieval_lock_keys() -> List[str]:
    return [
        "retrieval_objective_mode",
        "max_anchors",
        "samples_per_anchor",
        "candidate_top_t",
        "phase2_refine_mode",
        "semantic_topn_chunk",
        "graph_reserve_topn",
        "corridor_top_bc",
    ]


def _assert_retrieval_locked(champion_cfg: Mapping[str, Any], variant_cfg: Mapping[str, Any], dataset: str, variant: str) -> None:
    for key in _retrieval_lock_keys():
        if key not in champion_cfg:
            continue
        if champion_cfg.get(key) != variant_cfg.get(key):
            raise RuntimeError(
                f"retrieval lock violated for dataset={dataset} variant={variant} key={key}: "
                f"champion={champion_cfg.get(key)} variant={variant_cfg.get(key)}"
            )


def _variant_flags(variant: str) -> List[str]:
    vv = _strip(variant)
    if vv == "champion_reconfirm":
        return []
    if vv == "answer_surface_normalization":
        return ["qa_answer_surface_normalization"]
    if vv == "answer_type_aware_extraction":
        return ["qa_answer_type_aware_extraction"]
    if vv == "answer_cue_highlight":
        return ["answer_cue_highlight", "gen_evidence_focus_light"]
    if vv == "evidence_supported_verification":
        return ["qa_evidence_supported_verification", "gen_evidence_verify_light"]
    raise ValueError(f"unsupported variant: {variant}")


def _find_latest_summary(run_out_dir: Path, dataset: str) -> Optional[Path]:
    ds_dir = run_out_dir / dataset
    if not ds_dir.exists():
        return None
    cands = sorted(ds_dir.glob("*/rag_summary.json"))
    if not cands:
        return None
    return cands[-1]


def _is_complete(summary_path: Path, n_samples: int) -> bool:
    if not summary_path.exists():
        return False
    qpath = summary_path.with_name("rag_query_results.jsonl")
    if not qpath.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        n = _safe_int(summary.get("n_samples", 0), 0)
        if n_samples > 0 and n not in {n_samples, 0}:
            return False
    except Exception:
        return False
    # Fast count via line iteration.
    line_count = 0
    with qpath.open("r", encoding="utf-8") as f:
        for _ in f:
            line_count += 1
    if n_samples > 0 and line_count < n_samples:
        return False
    return line_count > 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    p.add_argument(
        "--variants",
        default="champion_reconfirm,answer_surface_normalization,answer_type_aware_extraction,answer_cue_highlight,evidence_supported_verification",
    )
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--qa-root", default="data/qa")
    p.add_argument("--corpus-root", default="/home/ojungii/HippoRAG2/dataset")
    p.add_argument("--effirag-output-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--graph-cache-dir", default="outputs/index_cache")
    p.add_argument("--llm-base-url", default="http://localhost:8011/v1")
    p.add_argument("--llm-api-key", default="EMPTY")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--print-tables", default="true")
    p.add_argument("--skip-completed", default="true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    datasets = _parse_csv(args.datasets)
    variants = _parse_csv(args.variants)
    if not datasets:
        raise SystemExit("empty --datasets")
    if not variants:
        raise SystemExit("empty --variants")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    round_root = Path(args.output_root).resolve() / run_stamp
    cfg_root = round_root / "configs"
    run_root = round_root / "runs"
    log_root = round_root / "logs"
    round_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    run_records_path = round_root / "run_records.tsv"
    failures_path = round_root / "failures.json"
    env_snapshot_path = round_root / "environment_snapshot.json"
    champion_manifest_path = round_root / "champion_manifest.json"

    headers = [
        "stage",
        "variant",
        "dataset",
        "profile",
        "kind",
        "qa_path",
        "corpus_path",
        "summary_path",
        "config_path",
        "log_path",
        "status",
        "start_ts",
        "end_ts",
        "elapsed_s",
        "n_samples",
        "error_message",
    ]

    # Preflight checks.
    py_files = [str(p) for p in sorted((REPO_ROOT / "effirag").glob("*.py"))]
    if py_files:
        subprocess.run([sys.executable, "-m", "py_compile", *py_files], check=True)
    subprocess.run([sys.executable, "-m", "effirag.config_audit", "configs/canonical"], check=True)

    champion_root = _discover_round_root(Path(args.effirag_output_root).resolve())
    if champion_root is None:
        raise SystemExit(f"champion run_records.tsv not found under {args.effirag_output_root}")

    qa_root = Path(args.qa_root).resolve()
    corpus_root = Path(args.corpus_root).resolve()

    champion_by_dataset: Dict[str, Dict[str, str]] = {}
    for ds in datasets:
        hit = _find_champion_run(champion_root, ds)
        if hit is None:
            raise SystemExit(f"champion_config run not found for dataset={ds} under {champion_root}")
        champion_by_dataset[ds] = hit

    write_json(
        str(champion_manifest_path),
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "champion_root": str(champion_root),
            "datasets": datasets,
            "champions": champion_by_dataset,
        },
    )

    env_snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _cmd(["git", "rev-parse", "HEAD"]),
        "active_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "cwd": str(Path.cwd().resolve()),
        "round_root": str(round_root),
        "champion_source_root": str(champion_root),
        "datasets": datasets,
        "variants": variants,
        "n_samples": int(args.n_samples),
        "qa_root": str(qa_root),
        "corpus_root": str(corpus_root),
        "graph_cache_dir": str(Path(args.graph_cache_dir).resolve()),
        "llm_base_url": str(args.llm_base_url),
        "llm_api_key": "EMPTY" if str(args.llm_api_key) == "EMPTY" else "***",
        "model_name": str(args.model_name),
        "preflight": {
            "py_compile": "run",
            "config_audit": "run",
        },
    }
    write_json(str(env_snapshot_path), env_snapshot)

    run_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for ds in datasets:
        champion_row = champion_by_dataset[ds]
        champion_cfg_path = Path(_strip(champion_row.get("config_path"))).resolve()
        champion_summary_path = Path(_strip(champion_row.get("summary_path"))).resolve()

        qa_path = _find_dataset_path(qa_root, ds)
        corpus_path = _find_corpus_path(corpus_root, ds)

        champion_cfg = _load_yaml(champion_cfg_path)

        for vv in variants:
            stage = "s0" if vv == "champion_reconfirm" else "s1"
            profile = "champion_reconfirm" if vv == "champion_reconfirm" else "abgf_improvement"
            cfg_path = cfg_root / stage / vv / ds / "rag.yaml"
            out_dir = run_root / stage / vv / "rag"
            log_path = log_root / f"{stage}__{vv}__{ds}.log"
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Build variant config from champion.
            cfg = dict(champion_cfg or {})
            cfg["dataset"] = ds
            cfg["data_path"] = str(qa_path)
            cfg["global_corpus_path"] = str(corpus_path)
            cfg["graph_cache_dir"] = str(Path(args.graph_cache_dir).resolve())
            cfg["output_dir"] = str(out_dir)
            cfg["timestamp_output"] = True
            cfg["limit"] = int(args.n_samples)
            cfg["run_qa"] = True
            cfg["retrieval_only"] = False
            cfg["generator"] = "vllm"
            cfg["model_name"] = str(args.model_name)
            cfg["openie_mode"] = "llm"
            cfg["openie_model_name"] = str(args.model_name)
            cfg["openie_local_files_only"] = True
            cfg["openie_api_base_url"] = str(args.llm_base_url)
            cfg["openie_api_key"] = str(args.llm_api_key)
            cfg["llm_base_url"] = str(args.llm_base_url)
            cfg["llm_api_key"] = str(args.llm_api_key)
            cfg["evaluator_mode"] = "hipporag2_parity"
            cfg["num_workers"] = 1
            cfg["force_rebuild_graph_index"] = False
            cfg["canonical_variant_name"] = str(vv)
            cfg["stagewise_loss_funnel_enabled"] = True
            cfg["oracle_support_injection_enabled"] = False

            flags = _variant_flags(vv)
            cfg["order_strategy"] = _append_flags(str(cfg.get("order_strategy", "score") or "score"), flags)

            _assert_retrieval_locked(champion_cfg, cfg, ds, vv)
            _dump_yaml(cfg_path, cfg)

            existing_summary = _find_latest_summary(out_dir, ds)
            if _bool(args.skip_completed) and existing_summary is not None and _is_complete(existing_summary, int(args.n_samples)):
                run_records.append(
                    {
                        "stage": stage,
                        "variant": vv,
                        "dataset": ds,
                        "profile": profile,
                        "kind": "rag",
                        "qa_path": str(qa_path),
                        "corpus_path": str(corpus_path),
                        "summary_path": str(existing_summary),
                        "config_path": str(cfg_path),
                        "log_path": str(log_path),
                        "status": "skipped",
                        "start_ts": "",
                        "end_ts": "",
                        "elapsed_s": "0",
                        "n_samples": str(args.n_samples),
                        "error_message": "",
                    }
                )
                continue

            cmd = [
                sys.executable,
                "-m",
                "effirag.run_rag",
                "--config",
                str(cfg_path),
                "--dataset",
                ds,
                "--data-path",
                str(qa_path),
                "--global-corpus-path",
                str(corpus_path),
                "--graph-cache-dir",
                str(Path(args.graph_cache_dir).resolve()),
                "--force-rebuild-graph-index",
                "false",
                "--num-workers",
                "1",
                "--openie-mode",
                "llm",
                "--openie-model-name",
                str(args.model_name),
                "--openie-local-files-only",
                "true",
                "--openie-api-base-url",
                str(args.llm_base_url),
                "--openie-api-key",
                str(args.llm_api_key),
                "--generator",
                "vllm",
                "--model-name",
                str(args.model_name),
                "--llm-base-url",
                str(args.llm_base_url),
                "--llm-api-key",
                str(args.llm_api_key),
                "--llm-max-new-tokens",
                "64",
                "--run-qa",
                "true",
                "--retrieval-only",
                "false",
                "--evaluator-mode",
                "hipporag2_parity",
                "--limit",
                str(args.n_samples),
                "--output-dir",
                str(out_dir),
                "--timestamp-output",
                "true",
            ]

            start_ts = _utc_now()
            t0 = time.time()
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(f"[START] {start_ts}\n")
                lf.write("[COMMAND] " + " ".join(cmd) + "\n")
                lf.write(f"[CHAMPION_SUMMARY] {champion_summary_path}\n")
                lf.write(f"[FLAGS] {flags}\n")
                lf.flush()
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False).returncode
            elapsed = int(max(0, round(time.time() - t0)))
            end_ts = _utc_now()

            if rc != 0:
                msg = f"exit_code_{rc}"
                failures.append({"dataset": ds, "variant": vv, "error": msg, "log_path": str(log_path)})
                run_records.append(
                    {
                        "stage": stage,
                        "variant": vv,
                        "dataset": ds,
                        "profile": profile,
                        "kind": "rag",
                        "qa_path": str(qa_path),
                        "corpus_path": str(corpus_path),
                        "summary_path": "",
                        "config_path": str(cfg_path),
                        "log_path": str(log_path),
                        "status": "failed",
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "elapsed_s": str(elapsed),
                        "n_samples": str(args.n_samples),
                        "error_message": msg,
                    }
                )
                continue

            latest_summary = _find_latest_summary(out_dir, ds)
            if latest_summary is None or not latest_summary.exists():
                msg = "missing_summary"
                failures.append({"dataset": ds, "variant": vv, "error": msg, "log_path": str(log_path)})
                run_records.append(
                    {
                        "stage": stage,
                        "variant": vv,
                        "dataset": ds,
                        "profile": profile,
                        "kind": "rag",
                        "qa_path": str(qa_path),
                        "corpus_path": str(corpus_path),
                        "summary_path": "",
                        "config_path": str(cfg_path),
                        "log_path": str(log_path),
                        "status": "failed",
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "elapsed_s": str(elapsed),
                        "n_samples": str(args.n_samples),
                        "error_message": msg,
                    }
                )
                continue

            run_records.append(
                {
                    "stage": stage,
                    "variant": vv,
                    "dataset": ds,
                    "profile": profile,
                    "kind": "rag",
                    "qa_path": str(qa_path),
                    "corpus_path": str(corpus_path),
                    "summary_path": str(latest_summary),
                    "config_path": str(cfg_path),
                    "log_path": str(log_path),
                    "status": "ok",
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "elapsed_s": str(elapsed),
                    "n_samples": str(args.n_samples),
                    "error_message": "",
                }
            )

    _write_tsv(run_records_path, headers, run_records)
    write_json(str(failures_path), {"count": len(failures), "items": failures})

    agg_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "aggregate_abgf_improvement.py"),
        "--round-root",
        str(round_root),
        "--run-records",
        str(run_records_path),
        "--model-name",
        str(args.model_name),
        "--print-tables",
        str(args.print_tables),
    ]
    agg_rc = subprocess.run(agg_cmd, check=False).returncode
    if agg_rc != 0:
        raise SystemExit(f"aggregate_abgf_improvement failed with exit code {agg_rc}")

    print(str(round_root))


if __name__ == "__main__":
    main()
