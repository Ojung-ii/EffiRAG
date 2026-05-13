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


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    cnt = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if str(line or "").strip():
                cnt += 1
    return int(cnt)


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


def _find_source_run(round_root: Path, dataset: str) -> Optional[Dict[str, str]]:
    rows = _read_tsv(round_root / "run_records.tsv")
    variant_priority = {
        "light_separator_render": 0,
        "champion_reconfirm": 1,
        "champion_config": 2,
    }
    hits = []
    for row in rows:
        if _strip(row.get("dataset")) != dataset:
            continue
        vv = _strip(row.get("variant"))
        if vv not in variant_priority:
            continue
        if _strip(row.get("status")) not in {"ok", "skipped"}:
            continue
        sp = Path(_strip(row.get("summary_path")))
        cp = Path(_strip(row.get("config_path")))
        if not sp.exists() or not cp.exists():
            continue
        hits.append((variant_priority[vv], dict(row)))
    if not hits:
        return None

    best_pri = min(pri for pri, _ in hits)
    best = [row for pri, row in hits if pri == best_pri]
    best.sort(key=lambda x: (_strip(x.get("end_ts")), _strip(x.get("start_ts"))))
    return best[-1]


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


def _resolve_global_corpus_path(cfg: Mapping[str, Any]) -> str:
    raw = _strip(cfg.get("global_corpus_path"))
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
            if not _strip(lhs):
                continue
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
    disallowed_exact = {"answer_cue_highlight"}
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
    if vv == "light_separator_render":
        return {
            "render_mode": "corridor_aware_flat",
            "order_flags": ["light_separator_render"],
            "prompt_variant": "default",
            "render_variant": "light_separator_render",
            "added_instruction": "none",
            "one_shot_enabled": False,
        }
    if vv == "light_separator_bridge_instruction":
        return {
            "render_mode": "corridor_aware_flat",
            "order_flags": ["light_separator_render"],
            "prompt_variant": "light_separator_bridge_instruction",
            "render_variant": "light_separator_render",
            "added_instruction": "bridge_and_copy_span",
            "one_shot_enabled": False,
        }
    if vv == "light_separator_copy_span_instruction":
        return {
            "render_mode": "corridor_aware_flat",
            "order_flags": ["light_separator_render"],
            "prompt_variant": "light_separator_copy_span_instruction",
            "render_variant": "light_separator_render",
            "added_instruction": "copy_span_only",
            "one_shot_enabled": False,
        }
    if vv == "light_separator_title_grounded_format":
        return {
            "render_mode": "corridor_aware_flat",
            "order_flags": ["title_grounded_format"],
            "prompt_variant": "default",
            "render_variant": "light_separator_title_grounded_format",
            "added_instruction": "none",
            "one_shot_enabled": False,
        }
    if vv == "light_separator_final_answer_oneshot":
        return {
            "render_mode": "corridor_aware_flat",
            "order_flags": ["light_separator_render"],
            "prompt_variant": "light_separator_final_answer_oneshot",
            "render_variant": "light_separator_render",
            "added_instruction": "final_answer_oneshot",
            "one_shot_enabled": True,
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
        default=(
            "light_separator_render,light_separator_bridge_instruction,"
            "light_separator_copy_span_instruction,light_separator_title_grounded_format,"
            "light_separator_final_answer_oneshot"
        ),
    )
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--qa-root", default="data/qa")
    p.add_argument("--corpus-root", default="/home/ojungii/HippoRAG2/dataset")
    p.add_argument("--effirag-champion-output-root", required=True)
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
    retrieval_fixed_manifest_path = round_root / "retrieval_fixed_manifest.json"

    headers = [
        "stage",
        "variant",
        "dataset",
        "profile",
        "kind",
        "qa_path",
        "corpus_path",
        "source_available",
        "source_variant",
        "source_summary_path",
        "source_config_path",
        "retrieval_fixed_expected",
        "precomputed_retrieval_path",
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

    py_files = [str(p) for p in sorted((REPO_ROOT / "effirag").glob("*.py"))]
    if py_files:
        subprocess.run([sys.executable, "-m", "py_compile", *py_files], check=True)
    subprocess.run([sys.executable, "-m", "effirag.config_audit", "configs/canonical"], check=True)

    source_root = _discover_round_root(Path(args.effirag_champion_output_root).resolve())
    if source_root is None:
        raise SystemExit(f"source run_records.tsv not found under {args.effirag_champion_output_root}")

    qa_root = Path(args.qa_root).resolve()
    corpus_root = Path(args.corpus_root).resolve()

    canonical_cfg_path = REPO_ROOT / "configs" / "canonical" / "rag_entity_first_chunk_grounded.yaml"
    if not canonical_cfg_path.exists():
        raise SystemExit(f"canonical config not found: {canonical_cfg_path}")
    canonical_cfg = _load_yaml(canonical_cfg_path)

    source_by_dataset: Dict[str, Optional[Dict[str, str]]] = {}
    for ds in datasets:
        source_by_dataset[ds] = _find_source_run(source_root, ds)

    retrieval_fixed_manifest: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_name": "prompt_interface_bridge_resolution_round",
        "source_root": str(source_root),
        "datasets": [],
    }
    for ds in datasets:
        row = source_by_dataset.get(ds)
        if row is None:
            retrieval_fixed_manifest["datasets"].append(
                {
                    "dataset": ds,
                    "source_available": False,
                    "retrieval_fixed_available": False,
                    "reason": "source_variant_missing_in_provided_root",
                }
            )
            continue
        summary_path = Path(_strip(row.get("summary_path"))).resolve()
        query_path = summary_path.with_name("rag_query_results.jsonl")
        if query_path.exists():
            retrieval_fixed_manifest["datasets"].append(
                {
                    "dataset": ds,
                    "source_available": True,
                    "source_variant": _strip(row.get("variant")),
                    "retrieval_fixed_available": True,
                    "source_summary_path": str(summary_path),
                    "source_query_path": str(query_path),
                    "source_query_count": int(_count_jsonl_lines(query_path)),
                    "source_config_path": str(Path(_strip(row.get("config_path"))).resolve()),
                }
            )
        else:
            retrieval_fixed_manifest["datasets"].append(
                {
                    "dataset": ds,
                    "source_available": True,
                    "source_variant": _strip(row.get("variant")),
                    "retrieval_fixed_available": False,
                    "reason": "source_query_artifact_missing",
                    "source_summary_path": str(summary_path),
                    "source_config_path": str(Path(_strip(row.get("config_path"))).resolve()),
                }
            )
    write_json(str(retrieval_fixed_manifest_path), retrieval_fixed_manifest)

    env_snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_name": "prompt_interface_bridge_resolution_round",
        "git_commit_hash": _cmd(["git", "rev-parse", "HEAD"]),
        "active_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "cwd": str(Path.cwd().resolve()),
        "round_root": str(round_root),
        "source_root": str(source_root),
        "retrieval_fixed_manifest_path": str(retrieval_fixed_manifest_path),
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
        qa_path = _find_dataset_path(qa_root, ds)
        corpus_path = _find_corpus_path(corpus_root, ds)

        source_row = source_by_dataset.get(ds)
        source_available = bool(source_row is not None)

        if source_available:
            source_cfg_path = Path(_strip(source_row.get("config_path"))).resolve()
            source_summary_path = Path(_strip(source_row.get("summary_path"))).resolve()
            source_variant = _strip(source_row.get("variant"))
            source_cfg = _load_yaml(source_cfg_path)
            base_cfg = dict(source_cfg)
            precomputed_retrieval_path = source_summary_path.with_name("rag_query_results.jsonl")
            retrieval_fixed_expected = bool(precomputed_retrieval_path.exists())
        else:
            source_cfg_path = Path()
            source_summary_path = Path()
            source_variant = ""
            source_cfg = None
            base_cfg = dict(canonical_cfg)
            precomputed_retrieval_path = Path()
            retrieval_fixed_expected = False

        if source_available:
            global_corpus_path = _resolve_global_corpus_path(base_cfg)
            if not global_corpus_path:
                global_corpus_path = str(corpus_path)
        else:
            global_corpus_path = str(corpus_path)

        for v_idx, vv in enumerate(variants):
            plan = _variant_plan(vv)
            stage = f"s{int(v_idx)}"
            profile = "prompt_interface_bridge"

            cfg_path = cfg_root / stage / vv / ds / "rag.yaml"
            out_dir = run_root / stage / vv / "rag"
            log_path = log_root / f"{stage}__{vv}__{ds}.log"
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            cfg = dict(base_cfg or {})
            cfg["dataset"] = ds
            cfg["data_path"] = str(qa_path)
            cfg["global_corpus_path"] = str(global_corpus_path)
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
            cfg["qa_utilization_variant"] = ""
            cfg["precomputed_retrieval_path"] = str(precomputed_retrieval_path) if retrieval_fixed_expected else ""
            cfg["precomputed_retrieval_strict"] = bool(retrieval_fixed_expected)

            base_order = _sanitize_order_strategy(str(cfg.get("order_strategy", "score") or "score"))
            flags = list(plan.get("order_flags", []) or [])
            cfg["order_strategy"] = _append_flags(base_order, flags)
            cfg["prompt_variant"] = str(plan.get("prompt_variant", "default") or "default")
            cfg["prompt_variant_label"] = str(cfg["prompt_variant"])
            cfg["render_mode"] = str(plan.get("render_mode", "corridor_aware_flat") or "corridor_aware_flat")
            cfg["render_variant"] = str(plan.get("render_variant", vv) or vv)
            cfg["added_instruction"] = str(plan.get("added_instruction", "none") or "none")
            cfg["one_shot_enabled"] = bool(plan.get("one_shot_enabled", False))

            if source_cfg is not None:
                _assert_retrieval_locked(source_cfg, cfg, ds, vv)
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
                        "source_available": str(source_available).lower(),
                        "source_variant": source_variant,
                        "source_summary_path": str(source_summary_path) if source_available else "",
                        "source_config_path": str(source_cfg_path) if source_available else "",
                        "retrieval_fixed_expected": str(retrieval_fixed_expected).lower(),
                        "precomputed_retrieval_path": str(precomputed_retrieval_path) if retrieval_fixed_expected else "",
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
            if retrieval_fixed_expected:
                cmd.extend(["--precomputed-retrieval-path", str(precomputed_retrieval_path), "--precomputed-retrieval-strict", "true"])

            start_ts = _utc_now()
            t0 = time.time()
            with log_path.open("ab") as lf:
                header = (
                    f"[START] {start_ts}\n"
                    f"[COMMAND] {' '.join(cmd)}\n"
                    f"[SOURCE_AVAILABLE] {str(source_available).lower()}\n"
                    f"[SOURCE_VARIANT] {source_variant}\n"
                    f"[SOURCE_SUMMARY] {str(source_summary_path) if source_available else ''}\n"
                    f"[RETRIEVAL_FIXED_EXPECTED] {str(retrieval_fixed_expected).lower()}\n"
                    f"[PRECOMPUTED_RETRIEVAL] {str(precomputed_retrieval_path) if retrieval_fixed_expected else ''}\n"
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
                        "source_available": str(source_available).lower(),
                        "source_variant": source_variant,
                        "source_summary_path": str(source_summary_path) if source_available else "",
                        "source_config_path": str(source_cfg_path) if source_available else "",
                        "retrieval_fixed_expected": str(retrieval_fixed_expected).lower(),
                        "precomputed_retrieval_path": str(precomputed_retrieval_path) if retrieval_fixed_expected else "",
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
                        "source_available": str(source_available).lower(),
                        "source_variant": source_variant,
                        "source_summary_path": str(source_summary_path) if source_available else "",
                        "source_config_path": str(source_cfg_path) if source_available else "",
                        "retrieval_fixed_expected": str(retrieval_fixed_expected).lower(),
                        "precomputed_retrieval_path": str(precomputed_retrieval_path) if retrieval_fixed_expected else "",
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
                    "source_available": str(source_available).lower(),
                    "source_variant": source_variant,
                    "source_summary_path": str(source_summary_path) if source_available else "",
                    "source_config_path": str(source_cfg_path) if source_available else "",
                    "retrieval_fixed_expected": str(retrieval_fixed_expected).lower(),
                    "precomputed_retrieval_path": str(precomputed_retrieval_path) if retrieval_fixed_expected else "",
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
        str(REPO_ROOT / "scripts" / "aggregate_prompt_interface_bridge_resolution_round.py"),
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
        raise SystemExit(f"aggregate_prompt_interface_bridge_resolution_round failed with exit code {agg_rc}")

    print(str(round_root))


if __name__ == "__main__":
    main()
