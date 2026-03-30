#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
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


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _to_list(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, (list, tuple, set)):
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out
    text = str(raw).strip()
    return [text] if text else []


def _parse_possible_answers(rec: dict) -> List[str]:
    answers = []
    for key in ("answer", "obj"):
        answers.extend(_to_list(rec.get(key)))
    for key in ("possible_answers", "answers", "answer_aliases", "o_aliases"):
        answers.extend(_to_list(rec.get(key)))
    dedup = []
    seen = set()
    for a in answers:
        if a in seen:
            continue
        seen.add(a)
        dedup.append(a)
    return dedup


def _qid_from_record(rec: dict) -> str:
    for key in ("_id", "id", "qid"):
        if key in rec and str(rec.get(key) or "").strip():
            return str(rec.get(key)).strip()
    return ""


def _load_dataset_gold_map(dataset: str, repo_root: Path) -> Dict[str, List[str]]:
    candidates = [
        repo_root / "data" / "qa" / f"{dataset}.json",
        repo_root.parent / "HippoRAG2" / "dataset" / f"{dataset}.json",
    ]
    src_path = None
    for p in candidates:
        if p.exists():
            src_path = p
            break
    if src_path is None:
        return {}

    data = _load_json(src_path)
    if isinstance(data, dict):
        records = data.get("data", data.get("examples", []))
    else:
        records = data
    if not isinstance(records, list):
        return {}

    out = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        qid = _qid_from_record(rec)
        if not qid:
            continue
        answers = _parse_possible_answers(rec)
        if answers:
            out[qid] = answers
    return out


def _variant_from_summary(summary_path: Path, summary: dict) -> str:
    path_str = str(summary_path)
    if "/rag_experiments/" in path_str:
        parts = summary_path.parts
        try:
            i = parts.index("rag_experiments")
            if i + 2 < len(parts):
                return f"{parts[i+1]}::{parts[i+2]}"
        except Exception:
            pass

    params = (summary.get("retrieval_params", {}) or {})
    mode = str(params.get("proposal_union_experiment_mode", "") or "")
    top1 = bool(params.get("top1_correction_enabled", False))
    if mode == "aggressive" and not top1:
        return "frozen::speed_default"
    if mode == "aggressive" and top1:
        return "frozen::quality_variant"
    if mode == "off":
        return "frozen::ablation_off"
    return f"rag::{summary_path.parent.name}"


def _is_frozen_variant(summary: dict) -> bool:
    params = (summary.get("retrieval_params", {}) or {})
    mode = str(params.get("proposal_union_experiment_mode", "") or "")
    top1 = bool(params.get("top1_correction_enabled", False))
    return (mode == "off") or (mode == "aggressive")


def _discover_summary_paths(repo_root: Path, selection: str, explicit: List[str], globs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for p in explicit:
        pp = Path(p).expanduser()
        if not pp.is_absolute():
            pp = (repo_root / pp).resolve()
        if pp.exists():
            paths.append(pp)
    for pattern in globs:
        for p in repo_root.glob(pattern):
            if p.exists():
                paths.append(p.resolve())

    if not paths:
        default_globs = [
            "outputs/rag/*/*/rag_summary.json",
            "outputs/rag_experiments/*/*/*/*/rag_summary.json",
            "outputs/final_eval/*/*/rag_summary.json",
            "outputs/rag_quality_track/*/*/rag_summary.json",
            "outputs/rag_quality_track_probe/*/*/rag_summary.json",
        ]
        for pattern in default_globs:
            for p in repo_root.glob(pattern):
                paths.append(p.resolve())

    uniq = []
    seen = set()
    for p in paths:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)

    if selection == "frozen":
        out = []
        for p in uniq:
            pstr = str(p)
            # Frozen operating-point recomputation should stay on canonical run trees,
            # not on exploratory rag_experiments branches.
            if "/outputs/rag_experiments/" in pstr:
                continue
            try:
                s = _load_json(p)
            except Exception:
                continue
            if _is_frozen_variant(s):
                out.append(p)
        return out
    return uniq


def _load_hipporag_metrics(repo_root: Path):
    try:
        hipporag_src = repo_root.parent / "HippoRAG2" / "src" / "hipporag"

        def _load_module(name: str, file_path: Path):
            spec = importlib.util.spec_from_file_location(name, str(file_path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(mod)
            return mod

        if "hipporag" not in sys.modules:
            pkg = types.ModuleType("hipporag")
            pkg.__path__ = [str(hipporag_src)]
            sys.modules["hipporag"] = pkg
        if "hipporag.utils" not in sys.modules:
            pkg = types.ModuleType("hipporag.utils")
            pkg.__path__ = [str(hipporag_src / "utils")]
            sys.modules["hipporag.utils"] = pkg
        if "hipporag.evaluation" not in sys.modules:
            pkg = types.ModuleType("hipporag.evaluation")
            pkg.__path__ = [str(hipporag_src / "evaluation")]
            sys.modules["hipporag.evaluation"] = pkg

        _load_module("hipporag.utils.logging_utils", hipporag_src / "utils" / "logging_utils.py")
        _load_module("hipporag.utils.config_utils", hipporag_src / "utils" / "config_utils.py")
        _load_module("hipporag.utils.eval_utils", hipporag_src / "utils" / "eval_utils.py")
        _load_module("hipporag.evaluation.base", hipporag_src / "evaluation" / "base.py")
        qa_eval_mod = _load_module("hipporag.evaluation.qa_eval", hipporag_src / "evaluation" / "qa_eval.py")

        return qa_eval_mod.QAExactMatch(), qa_eval_mod.QAF1Score()
    except Exception:
        return None, None


def _compute_scores(
    rows: List[dict],
    dataset: str,
    gold_map: Dict[str, List[str]],
):
    from effirag.eval.evaluator import QAEvaluator

    legacy = QAEvaluator(mode="legacy")
    parity = QAEvaluator(mode="hipporag2_parity")

    em_legacy = []
    f1_legacy = []
    em_parity = []
    f1_parity = []

    preds = []
    gold_batch = []
    for row in rows:
        if not bool(row.get("qa_executed", True)):
            continue
        pred = str(row.get("prediction", "") or "")
        sid = str(row.get("sample_id", "") or "")
        if sid in gold_map and gold_map[sid]:
            golds = list(gold_map[sid])
        else:
            golds = _to_list(row.get("gold_answers"))
            if not golds:
                golds = _to_list(row.get("answer"))
            if not golds:
                golds = [""]

        legacy_em = legacy.evaluate_batch([pred], [[golds[0]]]).get("ExactMatch", 0.0)
        legacy_f1 = legacy.evaluate_batch([pred], [[golds[0]]]).get("F1", 0.0)
        parity_res = parity.evaluate_batch([pred], [golds])

        em_legacy.append(float(legacy_em))
        f1_legacy.append(float(legacy_f1))
        em_parity.append(float(parity_res.get("ExactMatch", 0.0)))
        f1_parity.append(float(parity_res.get("F1", 0.0)))

        preds.append(pred)
        gold_batch.append(golds)

    return {
        "legacy_em": _mean(em_legacy),
        "legacy_f1": _mean(f1_legacy),
        "parity_em": _mean(em_parity),
        "parity_f1": _mean(f1_parity),
        "qa_count": len(preds),
        "preds": preds,
        "gold_batch": gold_batch,
    }


def _extract_from_md_summary_paths(md_path: Path) -> List[Path]:
    if not md_path.exists():
        return []
    text = md_path.read_text(encoding="utf-8")
    paths = []
    for match in re.finditer(r"(/home/ojungii/EffiRAG/[^\s|`]+rag_summary\.json)", text):
        p = Path(match.group(1))
        if p.exists():
            paths.append(p.resolve())
    return paths


def _default_key_variant_paths(repo_root: Path) -> List[Path]:
    md_files = [
        repo_root / "outputs/profiling/f1_boost/f1_boost_qa_summary_latest.md",
        repo_root / "outputs/profiling/semantic_selection/semantic_selection_qa_summary_latest.md",
        repo_root / "outputs/profiling/2wiki_bridge_seed_q1000/2wiki_bridge_seed_qa_summary_q1000_20260329_051308.md",
        repo_root / "outputs/profiling/next_method_design/2wiki_run_objective/nextdesign_2wiki_run_objective_qa_summary_latest.md",
        repo_root / "outputs/profiling/next_method_design/2wiki_grounding/nextdesign_2wiki_grounding_qa_summary_latest.md",
    ]
    out = []
    for md in md_files:
        out.extend(_extract_from_md_summary_paths(md))
    uniq = []
    seen = set()
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def _git_info(repo_root: Path) -> Dict[str, str]:
    out = {"branch": "", "commit": "", "tag": ""}
    try:
        out["branch"] = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root), text=True)
            .strip()
        )
    except Exception:
        pass
    try:
        out["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True)
            .strip()
        )
    except Exception:
        pass
    try:
        out["tag"] = (
            subprocess.check_output(["git", "describe", "--tags", "--exact-match"], cwd=str(repo_root), text=True)
            .strip()
        )
    except Exception:
        pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute legacy vs parity EM/F1 for existing runs.")
    parser.add_argument("--selection", type=str, default="all", choices=["all", "frozen", "key_variants"])
    parser.add_argument("--summary-path", action="append", default=[])
    parser.add_argument("--summary-glob", action="append", default=[])
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--output-md", required=True, type=str)
    parser.add_argument("--experiment-family", type=str, default="eval_parity")
    parser.add_argument("--question-being-answered", type=str, default="Legacy vs HippoRAG2 parity evaluator delta on existing runs")
    parser.add_argument("--baseline-reference", type=str, default="frozen speed_default/quality_variant/off")
    parser.add_argument("--frozen-config-reference", type=str, default="speed_default=aggressive, quality_variant=top1corr_t1, off=ablation")
    parser.add_argument("--dataset-scope", type=str, default="hotpotqa,2wikimultihopqa,musique,popqa")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    hippo_src = repo_root.parent / "HippoRAG2" / "src"
    if hippo_src.exists():
        sys.path.insert(0, str(hippo_src))

    summary_paths = _discover_summary_paths(repo_root, args.selection, args.summary_path, args.summary_glob)
    git_meta = _git_info(repo_root)
    if args.selection == "key_variants" and not args.summary_path and not args.summary_glob:
        summary_paths = _default_key_variant_paths(repo_root)

    rows = []
    dataset_gold_map_cache: Dict[str, Dict[str, List[str]]] = {}

    hippo_em_metric, hippo_f1_metric = _load_hipporag_metrics(repo_root)
    hippo_loaded = (hippo_em_metric is not None) and (hippo_f1_metric is not None)

    for summary_path in sorted(summary_paths):
        try:
            summary = _load_json(summary_path)
        except Exception:
            continue

        run_dir = summary_path.parent
        query_path = run_dir / "rag_query_results.jsonl"
        if not query_path.exists():
            continue
        try:
            run_rows = _load_jsonl(query_path)
        except Exception:
            continue

        dataset = str(summary.get("dataset", "") or "")
        if dataset not in dataset_gold_map_cache:
            dataset_gold_map_cache[dataset] = _load_dataset_gold_map(dataset, repo_root)
        gold_map = dataset_gold_map_cache.get(dataset, {})

        qa_scores = _compute_scores(run_rows, dataset=dataset, gold_map=gold_map)
        legacy_em = _safe_float(qa_scores["legacy_em"], 0.0)
        legacy_f1 = _safe_float(qa_scores["legacy_f1"], 0.0)
        parity_em = _safe_float(qa_scores["parity_em"], 0.0)
        parity_f1 = _safe_float(qa_scores["parity_f1"], 0.0)

        hippo_em = None
        hippo_f1 = None
        if hippo_loaded and qa_scores["qa_count"] > 0:
            try:
                em_res, _ = hippo_em_metric.calculate_metric_scores(qa_scores["gold_batch"], qa_scores["preds"])
                f1_res, _ = hippo_f1_metric.calculate_metric_scores(qa_scores["gold_batch"], qa_scores["preds"])
                hippo_em = _safe_float(em_res.get("ExactMatch", 0.0), 0.0)
                hippo_f1 = _safe_float(f1_res.get("F1", 0.0), 0.0)
            except Exception:
                hippo_em = None
                hippo_f1 = None

        row = {
            "dataset": dataset,
            "run_name": str(run_dir.name),
            "variant": _variant_from_summary(summary_path, summary),
            "n_samples": _safe_int(summary.get("n_samples", qa_scores["qa_count"]), qa_scores["qa_count"]),
            "legacy_EM": legacy_em,
            "legacy_F1": legacy_f1,
            "parity_EM": parity_em,
            "parity_F1": parity_f1,
            "delta_EM": parity_em - legacy_em,
            "delta_F1": parity_f1 - legacy_f1,
            "hipporag2_ref_EM": hippo_em,
            "hipporag2_ref_F1": hippo_f1,
            "delta_parity_vs_hipporag2_ref_EM": (None if hippo_em is None else parity_em - hippo_em),
            "delta_parity_vs_hipporag2_ref_F1": (None if hippo_f1 is None else parity_f1 - hippo_f1),
            "Recall@1": _safe_float(summary.get("supporting_fact_recall_at_1", 0.0), 0.0),
            "Recall@5": _safe_float(summary.get("supporting_fact_recall_at_5", 0.0), 0.0),
            "Recall@20": _safe_float(summary.get("supporting_fact_recall_at_20", 0.0), 0.0),
            "supporting_fact_recall": _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0),
            "rendered_recall": _safe_float(summary.get("rendered_supporting_fact_recall", 0.0), 0.0),
            "retrieval_ms": _safe_float(summary.get("retrieval_latency_ms", 0.0), 0.0),
            "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
            "total_ms": _safe_float(summary.get("total_latency_ms", 0.0), 0.0),
            "prediction_path": str(query_path),
            "summary_path": str(summary_path),
        }
        rows.append(row)

    aggregate = {
        "n_runs": len(rows),
        "avg_legacy_EM": _mean([_safe_float(r["legacy_EM"], 0.0) for r in rows]),
        "avg_legacy_F1": _mean([_safe_float(r["legacy_F1"], 0.0) for r in rows]),
        "avg_parity_EM": _mean([_safe_float(r["parity_EM"], 0.0) for r in rows]),
        "avg_parity_F1": _mean([_safe_float(r["parity_F1"], 0.0) for r in rows]),
        "avg_delta_EM": _mean([_safe_float(r["delta_EM"], 0.0) for r in rows]),
        "avg_delta_F1": _mean([_safe_float(r["delta_F1"], 0.0) for r in rows]),
    }

    payload = {
        "metadata": {
            "experiment_family": args.experiment_family,
            "question_being_answered": args.question_being_answered,
            "baseline_reference": args.baseline_reference,
            "frozen_config_reference": args.frozen_config_reference,
            "dataset_scope": args.dataset_scope,
            "changed_components": "evaluation semantics only (legacy vs hipporag2_parity)",
            "git_branch": git_meta.get("branch", ""),
            "git_commit": git_meta.get("commit", ""),
            "git_tag": git_meta.get("tag", ""),
            "selection": args.selection,
            "hipporag2_reference_loaded": bool(hippo_loaded),
        },
        "aggregate": aggregate,
        "rows": rows,
    }

    out_json = Path(args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = []
    md_lines.append("# Recomputed Metrics (Legacy vs HippoRAG2 Parity)")
    md_lines.append("")
    md_lines.append(f"- experiment_family: `{payload['metadata'].get('experiment_family', '')}`")
    md_lines.append(f"- question_being_answered: `{payload['metadata'].get('question_being_answered', '')}`")
    md_lines.append(f"- baseline_reference: `{payload['metadata'].get('baseline_reference', '')}`")
    md_lines.append(f"- frozen_config_reference: `{payload['metadata'].get('frozen_config_reference', '')}`")
    md_lines.append(f"- dataset_scope: `{payload['metadata'].get('dataset_scope', '')}`")
    md_lines.append(f"- changed_components: `{payload['metadata'].get('changed_components', '')}`")
    md_lines.append(f"- git_branch: `{payload['metadata'].get('git_branch', '')}`")
    md_lines.append(f"- git_commit: `{payload['metadata'].get('git_commit', '')}`")
    md_lines.append(f"- git_tag: `{payload['metadata'].get('git_tag', '')}`")
    md_lines.append(f"- selection: `{args.selection}`")
    md_lines.append(f"- n_runs: {aggregate['n_runs']}")
    md_lines.append(f"- avg_delta_EM(parity-legacy): {aggregate['avg_delta_EM']:.6f}")
    md_lines.append(f"- avg_delta_F1(parity-legacy): {aggregate['avg_delta_F1']:.6f}")
    md_lines.append("")
    md_lines.append("| dataset | variant | n_samples | legacy_EM | legacy_F1 | parity_EM | parity_F1 | delta_EM | delta_F1 | Recall@1 | Recall@5 | Recall@20 | rendered_recall | retrieval_ms | total_ms | summary_path |")
    md_lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in rows:
        md_lines.append(
            f"| {r['dataset']} | {r['variant']} | {int(r['n_samples'])} | "
            f"{_safe_float(r['legacy_EM'], 0.0):.4f} | {_safe_float(r['legacy_F1'], 0.0):.4f} | "
            f"{_safe_float(r['parity_EM'], 0.0):.4f} | {_safe_float(r['parity_F1'], 0.0):.4f} | "
            f"{_safe_float(r['delta_EM'], 0.0):+.4f} | {_safe_float(r['delta_F1'], 0.0):+.4f} | "
            f"{_safe_float(r['Recall@1'], 0.0):.4f} | {_safe_float(r['Recall@5'], 0.0):.4f} | {_safe_float(r['Recall@20'], 0.0):.4f} | "
            f"{_safe_float(r['rendered_recall'], 0.0):.4f} | {_safe_float(r['retrieval_ms'], 0.0):.2f} | {_safe_float(r['total_ms'], 0.0):.2f} | "
            f"{r['summary_path']} |"
        )

    out_md = Path(args.output_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
