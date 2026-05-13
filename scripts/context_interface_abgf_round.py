#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import pty
import select
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


def _run_with_live_tee(cmd: Sequence[str], log_bin_file: Any) -> int:
    """Run command with live terminal output while also appending full bytes to log."""
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        text=False,
    )
    os.close(slave_fd)
    try:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if master_fd in ready:
                try:
                    data = os.read(master_fd, 65536)
                except OSError as exc:
                    # PTY returns EIO on EOF for some platforms.
                    if exc.errno == errno.EIO:
                        data = b""
                    else:
                        raise
                if data:
                    log_bin_file.write(data)
                    log_bin_file.flush()
                    try:
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    except Exception:
                        pass
            if proc.poll() is not None:
                # Drain remaining buffered bytes once process exits.
                while True:
                    try:
                        data = os.read(master_fd, 65536)
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            data = b""
                        else:
                            raise
                    if not data:
                        break
                    log_bin_file.write(data)
                    log_bin_file.flush()
                    try:
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    except Exception:
                        pass
                break
        return int(proc.wait())
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass


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


def _resolve_global_corpus_from_champion(champion_cfg: Mapping[str, Any]) -> str:
    raw = _strip(champion_cfg.get("global_corpus_path"))
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def _normalize_global_corpus_path(v: Any) -> str:
    raw = _strip(v)
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


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
        "global_corpus_path",
    ]


def _assert_retrieval_locked(champion_cfg: Mapping[str, Any], variant_cfg: Mapping[str, Any], dataset: str, variant: str) -> None:
    for key in _retrieval_lock_keys():
        if key not in champion_cfg:
            continue
        lhs = champion_cfg.get(key)
        rhs = variant_cfg.get(key)
        if key == "global_corpus_path":
            lhs = _normalize_global_corpus_path(lhs)
            rhs = _normalize_global_corpus_path(rhs)
        if lhs != rhs:
            raise RuntimeError(
                f"retrieval lock violated for dataset={dataset} variant={variant} key={key}: "
                f"champion={champion_cfg.get(key)} variant={variant_cfg.get(key)}"
            )


def _sanitize_order_strategy(order_strategy: str) -> str:
    raw_tokens = [_strip(x).lower() for x in str(order_strategy or "score").split("+") if _strip(x)]
    if not raw_tokens:
        raw_tokens = ["score"]
    base = raw_tokens[0]
    allowed_bases = {"score", "retrieval", "corridor_rank", "query_bridge_answer", "qba"}
    if base not in allowed_bases:
        base = "score"
    disallowed_prefixes = ("qa_", "gen_")
    disallowed_exact = {
        "answer_cue_highlight",
    }
    kept = [base]
    for tok in raw_tokens[1:]:
        if tok in disallowed_exact:
            continue
        if any(tok.startswith(pref) for pref in disallowed_prefixes):
            continue
        if tok not in kept:
            kept.append(tok)
    return "+".join(kept) if kept else "score"


def _variant_plan(variant: str) -> Dict[str, Any]:
    vv = _strip(variant)
    if vv == "champion_reconfirm":
        return {
            "render_mode": None,
            "prompt_variant": "default",
            "order_flags": [],
            "render_variant": "champion_reconfirm",
            "prompt_variant_label": "default",
        }
    if vv == "corridor_grouped_flat":
        return {
            "render_mode": "corridor_aware_flat",
            "prompt_variant": "default",
            "order_flags": ["corridor_grouped_flat"],
            "render_variant": "corridor_grouped_flat",
            "prompt_variant_label": "default",
        }
    if vv == "path_structured_render":
        return {
            "render_mode": "path_bundle",
            "prompt_variant": "default",
            "order_flags": ["path_bundle_compactlite"],
            "render_variant": "path_structured_render",
            "prompt_variant_label": "default",
        }
    if vv == "role_tagged_compact_render":
        return {
            "render_mode": "corridor_aware_flat",
            "prompt_variant": "default",
            "order_flags": ["role_tagged_compact"],
            "render_variant": "role_tagged_compact_render",
            "prompt_variant_label": "default",
        }
    if vv == "evidence_first_prompt":
        return {
            "render_mode": None,
            "prompt_variant": "evidence_first",
            "order_flags": [],
            "render_variant": "champion_reconfirm",
            "prompt_variant_label": "evidence_first",
        }
    if vv == "path_structured_evidence_first":
        return {
            "render_mode": "path_bundle",
            "prompt_variant": "evidence_first",
            "order_flags": ["path_bundle_compactlite"],
            "render_variant": "path_structured_render",
            "prompt_variant_label": "evidence_first",
        }
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
        default="champion_reconfirm,corridor_grouped_flat,path_structured_render,role_tagged_compact_render,evidence_first_prompt,path_structured_evidence_first",
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

        champion_cfg = _load_yaml(champion_cfg_path)
        qa_path = _find_dataset_path(qa_root, ds)
        corpus_path = _find_corpus_path(corpus_root, ds)
        champion_global_corpus_path = _resolve_global_corpus_from_champion(champion_cfg)

        for vv in variants:
            stage = "s0" if vv == "champion_reconfirm" else "s1"
            profile = "champion_reconfirm" if vv == "champion_reconfirm" else "context_interface_abgf"
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
            cfg["global_corpus_path"] = str(champion_global_corpus_path)
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
            cfg["prompt_variant"] = "default"
            cfg["qa_utilization_variant"] = ""

            base_order = _sanitize_order_strategy(str(cfg.get("order_strategy", "score") or "score"))
            plan = _variant_plan(vv)
            flags = list(plan.get("order_flags", []) or [])
            cfg["order_strategy"] = _append_flags(base_order, flags)
            cfg["prompt_variant"] = str(plan.get("prompt_variant", "default") or "default")
            if plan.get("render_mode"):
                cfg["render_mode"] = str(plan["render_mode"])
            cfg["render_variant"] = str(plan.get("render_variant", vv) or vv)
            cfg["prompt_variant_label"] = str(plan.get("prompt_variant_label", cfg["prompt_variant"]) or cfg["prompt_variant"])

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
            if _strip(cfg.get("global_corpus_path")):
                cmd.extend(["--global-corpus-path", str(cfg["global_corpus_path"])])

            start_ts = _utc_now()
            t0 = time.time()
            with log_path.open("ab") as lf:
                header = (
                    f"[START] {start_ts}\n"
                    f"[COMMAND] {' '.join(cmd)}\n"
                    f"[CHAMPION_SUMMARY] {champion_summary_path}\n"
                    f"[FLAGS] {flags}\n"
                    f"[RENDER_MODE] {cfg.get('render_mode', '')}\n"
                    f"[PROMPT_VARIANT] {cfg.get('prompt_variant', 'default')}\n"
                )
                lf.write(header.encode("utf-8", errors="replace"))
                lf.flush()
                rc = _run_with_live_tee(cmd, lf)
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
        str(REPO_ROOT / "scripts" / "aggregate_context_interface_round.py"),
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
        raise SystemExit(f"aggregate_context_interface_round failed with exit code {agg_rc}")

    print(str(round_root))


if __name__ == "__main__":
    main()
