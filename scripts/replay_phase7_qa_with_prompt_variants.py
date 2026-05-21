#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from effirag.eval.evaluator import QAEvaluator
from effirag.registry import get_generator, register_defaults
from effirag.types import RenderedContext, Sample
from effirag.utils import content_tokens, load_yaml, timestamp_iso_utc


PROMPT_MODES = ["current_phase7", "lightrag_short", "phase7_short"]


@dataclass
class SourcePick:
    dataset: str
    run_dir: Path
    row_count: int


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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _find_source_for_dataset(dataset: str, limit: int, roots: List[Path]) -> SourcePick:
    candidates: List[Tuple[float, int, Path]] = []
    fallback: List[Tuple[int, float, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("rag_query_results.jsonl"):
            try:
                parent = p.parent
                parts = [str(x) for x in parent.parts]
                if dataset not in parts:
                    continue
                count = sum(1 for _ in p.open("r", encoding="utf-8"))
                mtime = float(p.stat().st_mtime)
                if count >= limit:
                    candidates.append((mtime, count, parent))
                fallback.append((count, mtime, parent))
            except Exception:
                continue
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        mtime, count, run_dir = candidates[0]
        return SourcePick(dataset=dataset, run_dir=run_dir, row_count=int(count))
    if fallback:
        fallback.sort(key=lambda x: (x[0], x[1]), reverse=True)
        count, _mtime, run_dir = fallback[0]
        return SourcePick(dataset=dataset, run_dir=run_dir, row_count=int(count))
    raise FileNotFoundError(f"No rag_query_results.jsonl found for dataset={dataset} in roots={roots}")


def _format_violation(prediction: str) -> bool:
    text = str(prediction or "").strip()
    if not text:
        return True
    if "\n" in text:
        return True
    low = text.lower()
    if low.startswith("answer:") or low.startswith("final answer:"):
        return True
    if low.startswith("-") or low.startswith("*"):
        return True
    return False


def _metric_bundle(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n <= 0:
        return {
            "n": 0,
            "EM": 0.0,
            "F1": 0.0,
            "avg_output_tokens": 0.0,
            "format_violation_rate": 0.0,
            "insufficient_information_rate": 0.0,
            "no_rate": 0.0,
            "yes_rate": 0.0,
            "num_predictions_equal_no": 0,
            "num_predictions_equal_insufficient_information": 0,
        }

    em = sum(_safe_float(r.get("em", 0.0), 0.0) for r in rows) / float(n)
    f1 = sum(_safe_float(r.get("f1", 0.0), 0.0) for r in rows) / float(n)
    avg_output_tokens = sum(int(r.get("output_tokens", 0) or 0) for r in rows) / float(n)

    fmt_viol = sum(1 for r in rows if bool(r.get("format_violation", False))) / float(n)
    insufficient = sum(1 for r in rows if str(r.get("prediction_norm", "")) == "insufficient information") / float(n)
    no_rate = sum(1 for r in rows if str(r.get("prediction_norm", "")) == "no") / float(n)
    yes_rate = sum(1 for r in rows if str(r.get("prediction_norm", "")) == "yes") / float(n)

    return {
        "n": int(n),
        "EM": float(em),
        "F1": float(f1),
        "avg_output_tokens": float(avg_output_tokens),
        "format_violation_rate": float(fmt_viol),
        "insufficient_information_rate": float(insufficient),
        "no_rate": float(no_rate),
        "yes_rate": float(yes_rate),
        "num_predictions_equal_no": int(sum(1 for r in rows if str(r.get("prediction_norm", "")) == "no")),
        "num_predictions_equal_insufficient_information": int(
            sum(1 for r in rows if str(r.get("prediction_norm", "")) == "insufficient information")
        ),
    }


def _report_table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def _fmt4(x: float) -> str:
    return f"{float(x):.4f}"


def _infer_profile_variant(run_dir: Path, dataset: str) -> Tuple[str, str]:
    parts = list(run_dir.resolve().parts)
    ds_idx = -1
    for i, p in enumerate(parts):
        if str(p) == str(dataset):
            ds_idx = i
            break
    if ds_idx >= 2:
        return str(parts[ds_idx - 2]), str(parts[ds_idx - 1])
    return "unknown_profile", "unknown_variant"


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay Phase7 QA with prompt variants on fixed rendered contexts.")
    ap.add_argument("--output-root", type=str, default="outputs/phase7_evidence_flow/qa_prompt_diagnosis")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--datasets", type=str, default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--prompt-modes", type=str, default=",".join(PROMPT_MODES))
    ap.add_argument("--context-source-roots", type=str, default="")
    ap.add_argument("--python-note", type=str, default="")
    args = ap.parse_args()

    register_defaults("phase7_evidence_flow")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    prompt_modes = [x.strip() for x in str(args.prompt_modes).split(",") if x.strip()]
    for mode in prompt_modes:
        if mode not in PROMPT_MODES:
            raise ValueError(f"Unsupported prompt mode: {mode}")

    if str(args.context_source_roots).strip():
        source_roots = [Path(x.strip()).resolve() for x in str(args.context_source_roots).split(",") if x.strip()]
    else:
        source_roots = [
            Path("outputs/phase7_evidence_flow/corridor_bq_followup").resolve(),
            Path("outputs/phase7_evidence_flow/legacy_budget_diagnosis").resolve(),
            Path("outputs/phase7_evidence_flow/corridor_bq_experiment").resolve(),
        ]

    picks: Dict[str, SourcePick] = {}
    for ds in datasets:
        picks[ds] = _find_source_for_dataset(dataset=ds, limit=int(args.limit), roots=source_roots)

    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, Any]] = {}

    for ds in datasets:
        pick = picks[ds]
        rows = _read_jsonl(pick.run_dir / "rag_query_results.jsonl")
        rows = rows[: max(1, int(args.limit))]

        cfg_path = Path("configs/main_config/phase7_evidence_flow") / f"{ds}.yaml"
        cfg_payload = dict(load_yaml(str(cfg_path)) or {})
        profile, variant = _infer_profile_variant(run_dir=pick.run_dir, dataset=ds)

        evaluator_mode = "legacy"
        if rows:
            evaluator_mode = str(rows[0].get("evaluator_mode", evaluator_mode) or evaluator_mode)
        evaluator = QAEvaluator(mode=evaluator_mode)

        dataset_mode_rows: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in prompt_modes}
        for row in rows:
            rendered = dict((row.get("rendered", {}) or {}))
            sample = Sample(
                qid=str(row.get("sample_id", "") or ""),
                question=str(row.get("question", "") or ""),
                answer=str(row.get("answer", "") or ""),
                contexts=[],
                supporting_facts=[],
                metadata={
                    "answers": list(row.get("gold_answers", []) or []),
                },
            )
            rendered_ctx = RenderedContext(
                sample_id=str(row.get("sample_id", "") or ""),
                method=str(row.get("method", "phase7_evidence_flow") or "phase7_evidence_flow"),
                text=str(rendered.get("text", "") or ""),
                sentences=[str(x) for x in list(rendered.get("sentences", []) or [])],
                sentence_ids=[str(x) for x in list(rendered.get("sentence_ids", []) or row.get("rendered_sentence_ids", []) or [])],
                truncated=bool(rendered.get("truncated", False)),
                render_mode=str(rendered.get("render_mode", "phase7_flat") or "phase7_flat"),
                rendered_corridor_ids=[str(x) for x in list(rendered.get("rendered_corridor_ids", []) or [])],
                truncated_corridor_count=int(rendered.get("truncated_corridor_count", 0) or 0),
                truncated_sentence_count=int(rendered.get("truncated_sentence_count", 0) or 0),
                retrieval_selected_sentence_ids=[str(x) for x in list(rendered.get("retrieval_selected_sentence_ids", []) or [])],
                metadata=dict(rendered.get("metadata", {}) or {}),
            )

            generator_name = str(row.get("generator", cfg_payload.get("generator", "openai_compat")) or "openai_compat")
            model_name = str(cfg_payload.get("model_name", row.get("generation", {}).get("model_name", "")) or "")
            generator_fn = get_generator(generator_name)

            for prompt_mode in prompt_modes:
                cfg_obj = SimpleNamespace(
                    llm_base_url=str(cfg_payload.get("llm_base_url", "") or ""),
                    llm_api_key=str(cfg_payload.get("llm_api_key", "") or ""),
                    llm_timeout_sec=float(cfg_payload.get("llm_timeout_sec", 120.0) or 120.0),
                    llm_max_new_tokens=int(cfg_payload.get("llm_max_new_tokens", 64) or 64),
                    prompt_variant=str(cfg_payload.get("prompt_variant", "default") or "default"),
                    qa_prompt_mode=str(prompt_mode),
                    user_prompt="",
                    generator=str(generator_name),
                )
                gen = generator_fn(sample, rendered_ctx, model_name, cfg_obj)
                eval_result = evaluator.evaluate(gen.prediction, sample)

                pred_norm = str(gen.prediction or "").strip().lower()
                rec = {
                    "dataset": ds,
                    "query_id": str(sample.qid),
                    "profile": str(profile),
                    "variant": str(variant),
                    "question": str(sample.question),
                    "gold_answer": str(sample.answer),
                    "prediction": str(gen.prediction or ""),
                    "prediction_norm": pred_norm,
                    "prompt_mode": prompt_mode,
                    "generator": generator_name,
                    "model_name": model_name,
                    "em": float(eval_result.em),
                    "f1": float(eval_result.f1),
                    "rendered_context_tokens": int(len(content_tokens(str(rendered_ctx.text or "")))),
                    "output_tokens": int(len(content_tokens(str(gen.prediction or "")))),
                    "is_no": bool(pred_norm == "no"),
                    "is_yes": bool(pred_norm == "yes"),
                    "is_insufficient": bool(pred_norm == "insufficient information"),
                    "format_violation": bool(_format_violation(str(gen.prediction or ""))),
                    "generation_fallback": bool((gen.metadata or {}).get("fallback_used", False)),
                    "source_run_dir": str(pick.run_dir),
                }
                dataset_mode_rows[prompt_mode].append(rec)
                all_rows.append(dict(rec))

        ds_summary: Dict[str, Any] = {}
        for mode in prompt_modes:
            ds_summary[mode] = _metric_bundle(dataset_mode_rows.get(mode, []))
        summary[ds] = {
            "source_run_dir": str(pick.run_dir),
            "source_row_count": int(pick.row_count),
            "limit": int(len(rows)),
            "profile": str(profile),
            "variant": str(variant),
            "prompt_modes": ds_summary,
        }

    out_jsonl = output_root / "qa_prompt_replay_results.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_payload = {
        "generated_at": timestamp_iso_utc(),
        "output_root": str(output_root),
        "datasets": summary,
        "prompt_modes": prompt_modes,
        "python_note": str(args.python_note or ""),
    }
    (output_root / "qa_prompt_replay_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # markdown report
    md_lines: List[str] = []
    md_lines.append("# Phase7 QA Prompt Replay Report")
    md_lines.append("")
    rows_md: List[List[str]] = []
    flat_summary_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        src = summary.get(ds, {})
        src_dir = str(src.get("source_run_dir", ""))
        src_profile = str(src.get("profile", "unknown_profile"))
        src_variant = str(src.get("variant", "unknown_variant"))
        mode_map = dict(src.get("prompt_modes", {}) or {})
        for mode in prompt_modes:
            m = dict(mode_map.get(mode, {}) or {})
            flat_summary_rows.append(
                {
                    "dataset": ds,
                    "profile": src_profile,
                    "variant": src_variant,
                    "prompt_mode": mode,
                    "EM": float(m.get("EM", 0.0) or 0.0),
                    "F1": float(m.get("F1", 0.0) or 0.0),
                    "no_rate": float(m.get("no_rate", 0.0) or 0.0),
                    "insufficient_information_rate": float(m.get("insufficient_information_rate", 0.0) or 0.0),
                    "format_violation_rate": float(m.get("format_violation_rate", 0.0) or 0.0),
                    "avg_output_tokens": float(m.get("avg_output_tokens", 0.0) or 0.0),
                }
            )
            rows_md.append(
                [
                    ds,
                    src_profile,
                    src_variant,
                    mode,
                    _fmt4(m.get("EM", 0.0)),
                    _fmt4(m.get("F1", 0.0)),
                    _fmt4(m.get("no_rate", 0.0)),
                    _fmt4(m.get("insufficient_information_rate", 0.0)),
                    _fmt4(m.get("format_violation_rate", 0.0)),
                    _fmt4(m.get("avg_output_tokens", 0.0)),
                ]
            )
    md_lines.append(
        _report_table(
            [
                "dataset",
                "profile",
                "variant",
                "prompt_mode",
                "EM",
                "F1",
                "no_rate",
                "insufficient_rate",
                "format_violation_rate",
                "avg_output_tokens",
            ],
            rows_md,
        )
    )

    # Interpretation requested in the diagnostic completion instruction.
    md_lines.append("")
    md_lines.append("## Prompt Replay Interpretation")
    by_ds_mode: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in flat_summary_rows:
        by_ds_mode[(str(row["dataset"]), str(row["prompt_mode"]))] = row
    for ds in datasets:
        cur = by_ds_mode.get((ds, "current_phase7"), {})
        lgr = by_ds_mode.get((ds, "lightrag_short"), {})
        p7s = by_ds_mode.get((ds, "phase7_short"), {})
        md_lines.append(f"- {ds}: lightrag_short vs current_phase7 F1 delta = {_fmt4(float(lgr.get('F1', 0.0)) - float(cur.get('F1', 0.0)))}.")
        md_lines.append(f"- {ds}: phase7_short vs lightrag_short F1 delta = {_fmt4(float(p7s.get('F1', 0.0)) - float(lgr.get('F1', 0.0)))}.")
        md_lines.append(f"- {ds}: no_rate delta (phase7_short - current_phase7) = {_fmt4(float(p7s.get('no_rate', 0.0)) - float(cur.get('no_rate', 0.0)))}.")
    md_lines.append("- Prompt changes are diagnostic-only; interpret with oracle context and failure taxonomy before attributing the legacy gap.")

    (output_root / "phase7_qa_prompt_replay_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
