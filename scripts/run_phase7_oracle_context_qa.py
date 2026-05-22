#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from effirag.eval.evaluator import QAEvaluator
from effirag.registry import get_dataset_loader, get_generator, register_defaults
from effirag.types import RenderedContext, Sample
from effirag.utils import content_tokens, load_yaml, timestamp_iso_utc


PROMPT_MODES = ["phase7_short", "current_phase7", "lightrag_short"]
CONTEXT_TYPES = [
    "selected_context",
    "gold_support_context",
    "phase1_oracle_context",
    "feature_ready_oracle_context",
    "selected_plus_gold_context",
    "best_candidate_oracle_context",
]


@dataclass
class SourcePick:
    dataset: str
    run_dir: Path
    row_count: int


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


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
    best: Tuple[int, float, Path] | None = None
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("rag_query_results.jsonl"):
            parent = p.parent
            if dataset not in [str(x) for x in parent.parts]:
                continue
            try:
                count = sum(1 for _ in p.open("r", encoding="utf-8"))
                mtime = float(p.stat().st_mtime)
            except Exception:
                continue
            if count < limit:
                continue
            cand = (count, mtime, parent)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
    if best is None:
        # fallback to any available
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("rag_query_results.jsonl"):
                parent = p.parent
                if dataset not in [str(x) for x in parent.parts]:
                    continue
                try:
                    count = sum(1 for _ in p.open("r", encoding="utf-8"))
                    mtime = float(p.stat().st_mtime)
                except Exception:
                    continue
                cand = (count, mtime, parent)
                if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                    best = cand
    if best is None:
        raise FileNotFoundError(f"No source run found for dataset={dataset}")
    return SourcePick(dataset=dataset, run_dir=best[2], row_count=int(best[0]))


def _sample_lookup(dataset: str, limit: int) -> Dict[str, Sample]:
    cfg_path = Path("configs/main_config/phase7_evidence_flow") / f"{dataset}.yaml"
    cfg = dict(load_yaml(str(cfg_path)) or {})
    data_path = str(cfg.get("data_path", "") or "")
    split = str(cfg.get("split", "validation") or "validation")

    loader = get_dataset_loader(dataset)
    # Load with a safe upper bound to ensure queried IDs are available.
    samples = loader(split=split, limit=max(1000, int(limit) * 20), data_path=data_path)
    out = {str(s.qid): s for s in samples}
    return out


def _sentence_lookup(sample: Sample) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for doc in list(getattr(sample, "contexts", []) or []):
        title = str(getattr(doc, "title", "") or "")
        for idx, sent in enumerate(list(getattr(doc, "sentences", []) or [])):
            sid = f"{title}::{idx}"
            out[sid] = str(sent or "")
    return out


def _title_to_sentences(sample: Sample) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    for doc in list(getattr(sample, "contexts", []) or []):
        title = str(getattr(doc, "title", "") or "")
        rows: List[Tuple[str, str]] = []
        for idx, sent in enumerate(list(getattr(doc, "sentences", []) or [])):
            sid = f"{title}::{idx}"
            rows.append((sid, str(sent or "")))
        out[title] = rows
    return out


def _lines_to_context(lines: List[Tuple[str, str]], token_cap: int = 512) -> Tuple[str, List[str], List[str]]:
    kept: List[Tuple[str, str]] = []
    tok = 0
    for sid, text in lines:
        sid = str(sid or "")
        text = str(text or "").strip()
        if not sid or not text:
            continue
        title = sid.rsplit("::", 1)[0] if "::" in sid else sid
        line = f"[{title}] {text}"
        t = len(content_tokens(line))
        if tok + t > int(token_cap):
            continue
        kept.append((sid, text))
        tok += t
    context_text = "\n".join([f"[{sid.rsplit('::', 1)[0] if '::' in sid else sid}] {text}" for sid, text in kept])
    return context_text, [sid for sid, _ in kept], [text for _, text in kept]


def _build_oracle_contexts(sample: Sample, row: Dict[str, Any], token_cap: int = 512) -> Dict[str, Dict[str, Any]]:
    rendered = dict((row.get("rendered", {}) or {}))
    retrieval = dict((row.get("retrieval", {}) or {}))
    diag = dict((retrieval.get("diagnostics", {}) or {}))

    selected_ids = [str(x) for x in list(rendered.get("sentence_ids", []) or row.get("rendered_sentence_ids", []) or [])]
    selected_texts = [str(x) for x in list(rendered.get("sentences", []) or [])]
    if len(selected_texts) != len(selected_ids):
        selected_texts = ["" for _ in selected_ids]

    selected_lines = [(sid, txt) for sid, txt in zip(selected_ids, selected_texts) if sid and str(txt).strip()]
    selected_context_text, selected_keep_ids, selected_keep_texts = _lines_to_context(selected_lines, token_cap=token_cap)

    sentence_map = _sentence_lookup(sample)
    title_map = _title_to_sentences(sample)
    gold_pairs = list(getattr(sample, "supporting_facts", []) or [])
    gold_ids = [f"{str(t)}::{int(i)}" for t, i in gold_pairs if str(t)]

    gold_lines: List[Tuple[str, str]] = []
    for sid in gold_ids:
        text = str(sentence_map.get(sid, "") or "")
        if text:
            gold_lines.append((sid, text))
    gold_context_text, gold_keep_ids, gold_keep_texts = _lines_to_context(gold_lines, token_cap=token_cap)

    phase1_ids = [str(x) for x in list(diag.get("phase1_seed_ids", []) or [])]
    phase1_texts = [str(x) for x in list(diag.get("phase1_seed_texts", []) or [])]
    phase2_ids = [str(x) for x in list(diag.get("phase2_candidate_ids", []) or [])]
    phase2_texts = [str(x) for x in list(diag.get("phase2_candidate_texts", []) or [])]
    phase2_scores = [x for x in list(diag.get("phase2_candidate_scores", []) or [])]
    phase1_title_hits = set([str(t) for t, _i in gold_pairs if str(t)])
    phase1_lines: List[Tuple[str, str]] = []
    for idx, sid in enumerate(phase1_ids):
        text = phase1_texts[idx] if idx < len(phase1_texts) else ""
        if sid in set(gold_ids):
            if str(text).strip():
                phase1_lines.append((sid, str(text)))
            elif sid in sentence_map:
                phase1_lines.append((sid, sentence_map[sid]))
            continue
        title = sid.rsplit("::", 1)[0] if "::" in sid else sid
        if title in phase1_title_hits:
            if str(text).strip():
                phase1_lines.append((sid, str(text)))
            elif sid in sentence_map:
                phase1_lines.append((sid, sentence_map[sid]))

    # If phase1 id granularity is not sentence, fallback to title-level sentence pick.
    if not phase1_lines:
        for title in sorted(phase1_title_hits):
            for sid, text in list(title_map.get(title, []) or []):
                phase1_lines.append((sid, text))

    phase1_context_text, phase1_keep_ids, phase1_keep_texts = _lines_to_context(phase1_lines, token_cap=token_cap)

    feature_ready_lines: List[Tuple[str, str]] = []
    for idx, sid in enumerate(phase2_ids):
        text = phase2_texts[idx] if idx < len(phase2_texts) else ""
        title = sid.rsplit("::", 1)[0] if "::" in sid else sid
        if sid in set(gold_ids) or title in phase1_title_hits:
            if str(text).strip():
                feature_ready_lines.append((sid, str(text)))
            elif sid in sentence_map:
                feature_ready_lines.append((sid, sentence_map[sid]))
    if not feature_ready_lines:
        feature_ready_lines = list(phase1_lines)
    feature_ready_text, feature_ready_ids, feature_ready_texts = _lines_to_context(
        feature_ready_lines,
        token_cap=token_cap,
    )

    best_rows: List[Tuple[float, str, str]] = []
    for idx, sid in enumerate(phase2_ids):
        title = sid.rsplit("::", 1)[0] if "::" in sid else sid
        text = phase2_texts[idx] if idx < len(phase2_texts) else ""
        score = _safe_float(phase2_scores[idx], 0.0) if idx < len(phase2_scores) else 0.0
        if not text and sid in sentence_map:
            text = sentence_map[sid]
        if not text:
            continue
        gold_like = 1.0 if (sid in set(gold_ids) or title in phase1_title_hits) else 0.0
        best_rows.append((gold_like * 10.0 + score, sid, text))
    best_rows.sort(key=lambda x: (float(x[0]), str(x[1])), reverse=True)
    best_lines = [(sid, text) for _score, sid, text in best_rows]
    best_text, best_ids, best_texts = _lines_to_context(best_lines, token_cap=token_cap)

    selected_plus_gold_lines = list(selected_lines)
    selected_set = set([sid for sid, _ in selected_lines])
    for sid in gold_ids:
        if sid in selected_set:
            continue
        text = str(sentence_map.get(sid, "") or "")
        if text:
            selected_plus_gold_lines.append((sid, text))
    selected_plus_gold_text, selected_plus_gold_ids, selected_plus_gold_texts = _lines_to_context(
        selected_plus_gold_lines,
        token_cap=token_cap,
    )

    return {
        "selected_context": {
            "text": selected_context_text,
            "sentence_ids": selected_keep_ids,
            "sentences": selected_keep_texts,
        },
        "gold_support_context": {
            "text": gold_context_text,
            "sentence_ids": gold_keep_ids,
            "sentences": gold_keep_texts,
        },
        "phase1_oracle_context": {
            "text": phase1_context_text,
            "sentence_ids": phase1_keep_ids,
            "sentences": phase1_keep_texts,
        },
        "selected_plus_gold_context": {
            "text": selected_plus_gold_text,
            "sentence_ids": selected_plus_gold_ids,
            "sentences": selected_plus_gold_texts,
        },
        "feature_ready_oracle_context": {
            "text": feature_ready_text,
            "sentence_ids": feature_ready_ids,
            "sentences": feature_ready_texts,
        },
        "best_candidate_oracle_context": {
            "text": best_text,
            "sentence_ids": best_ids,
            "sentences": best_texts,
        },
    }


def _fmt4(x: float) -> str:
    return f"{float(x):.4f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase7 oracle context QA diagnostics")
    ap.add_argument("--root", type=str, default="outputs/phase7_evidence_flow/qa_prompt_diagnosis")
    ap.add_argument("--out", type=str, default="outputs/phase7_evidence_flow/qa_prompt_diagnosis/oracle_context_report.md")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--datasets", type=str, default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--context-source-roots", type=str, default="")
    args = ap.parse_args()

    register_defaults("phase7_evidence_flow")

    out_root = Path(args.root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    if str(args.context_source_roots).strip():
        roots = [Path(x.strip()).resolve() for x in str(args.context_source_roots).split(",") if x.strip()]
    else:
        roots = [
            Path("outputs/phase7_evidence_flow/corridor_bq_followup").resolve(),
            Path("outputs/phase7_evidence_flow/legacy_budget_diagnosis").resolve(),
            Path("outputs/phase7_evidence_flow/corridor_bq_experiment").resolve(),
        ]

    picks = {ds: _find_source_for_dataset(ds, int(args.limit), roots) for ds in datasets}
    sample_maps = {ds: _sample_lookup(ds, int(args.limit)) for ds in datasets}

    all_query_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        pick = picks[ds]
        run_rows = _read_jsonl(pick.run_dir / "rag_query_results.jsonl")[: max(1, int(args.limit))]

        cfg_path = Path("configs/main_config/phase7_evidence_flow") / f"{ds}.yaml"
        cfg_payload = dict(load_yaml(str(cfg_path)) or {})

        generator_name = str(cfg_payload.get("generator", "openai_compat") or "openai_compat")
        generator_fn = get_generator(generator_name)
        model_name = str(cfg_payload.get("model_name", "") or "")
        evaluator_mode = str(run_rows[0].get("evaluator_mode", "legacy") if run_rows else "legacy")
        evaluator = QAEvaluator(mode=evaluator_mode)

        failed_rows = [r for r in run_rows if _safe_float((r.get("metrics", {}) or {}).get("f1", 0.0), 0.0) < 1.0]

        per_mode: Dict[str, Dict[str, List[float]]] = {
            mode: {ctype: [] for ctype in CONTEXT_TYPES} for mode in PROMPT_MODES
        }
        per_mode_stats: Dict[str, Dict[str, Dict[str, float]]] = {
            mode: {
                ctype: {
                    "n": 0.0,
                    "em_sum": 0.0,
                    "f1_sum": 0.0,
                    "no_cnt": 0.0,
                    "insufficient_cnt": 0.0,
                    "format_violation_cnt": 0.0,
                    "output_tokens_sum": 0.0,
                }
                for ctype in CONTEXT_TYPES
            }
            for mode in PROMPT_MODES
        }
        per_mode_gold_prompt_fail: Dict[str, int] = {mode: 0 for mode in PROMPT_MODES}
        per_mode_oracle_success: Dict[str, int] = {mode: 0 for mode in PROMPT_MODES}
        per_mode_count: Dict[str, int] = {mode: 0 for mode in PROMPT_MODES}

        for row in failed_rows:
            qid = str(row.get("sample_id", "") or "")
            sample = sample_maps[ds].get(qid)
            if sample is None:
                continue
            oracle_contexts = _build_oracle_contexts(sample=sample, row=row, token_cap=512)

            for mode in PROMPT_MODES:
                f1_by_context: Dict[str, float] = {}
                em_by_context: Dict[str, float] = {}
                pred_by_context: Dict[str, str] = {}
                for ctype in CONTEXT_TYPES:
                    cctx = dict(oracle_contexts.get(ctype, {}) or {})
                    rendered_ctx = RenderedContext(
                        sample_id=str(sample.qid),
                        method="phase7_evidence_flow",
                        text=str(cctx.get("text", "") or ""),
                        sentences=[str(x) for x in list(cctx.get("sentences", []) or [])],
                        sentence_ids=[str(x) for x in list(cctx.get("sentence_ids", []) or [])],
                        truncated=False,
                        render_mode="phase7_flat",
                        rendered_corridor_ids=[],
                        truncated_corridor_count=0,
                        truncated_sentence_count=0,
                        retrieval_selected_sentence_ids=[str(x) for x in list(cctx.get("sentence_ids", []) or [])],
                        metadata={"prompt_variant": str(cfg_payload.get("prompt_variant", "default") or "default")},
                    )
                    cfg_obj = SimpleNamespace(
                        llm_base_url=str(cfg_payload.get("llm_base_url", "") or ""),
                        llm_api_key=str(cfg_payload.get("llm_api_key", "") or ""),
                        llm_timeout_sec=float(cfg_payload.get("llm_timeout_sec", 120.0) or 120.0),
                        llm_max_new_tokens=int(cfg_payload.get("llm_max_new_tokens", 64) or 64),
                        prompt_variant=str(cfg_payload.get("prompt_variant", "default") or "default"),
                        qa_prompt_mode=str(mode),
                        user_prompt="",
                        generator=str(generator_name),
                    )
                    gen = generator_fn(sample, rendered_ctx, model_name, cfg_obj)
                    eval_result = evaluator.evaluate(gen.prediction, sample)
                    f1 = float(eval_result.f1)
                    em = float(eval_result.em)
                    f1_by_context[ctype] = f1
                    em_by_context[ctype] = em
                    pred_by_context[ctype] = str(gen.prediction or "")
                    per_mode[mode][ctype].append(f1)
                    pred_norm = str(gen.prediction or "").strip().lower()
                    stats = per_mode_stats[mode][ctype]
                    stats["n"] += 1.0
                    stats["em_sum"] += float(em)
                    stats["f1_sum"] += float(f1)
                    stats["no_cnt"] += 1.0 if pred_norm == "no" else 0.0
                    stats["insufficient_cnt"] += 1.0 if pred_norm == "insufficient information" else 0.0
                    stats["format_violation_cnt"] += 1.0 if _format_violation(str(gen.prediction or "")) else 0.0
                    stats["output_tokens_sum"] += float(len(content_tokens(str(gen.prediction or ""))))

                per_mode_count[mode] += 1
                if f1_by_context.get("gold_support_context", 0.0) <= 0.0:
                    per_mode_gold_prompt_fail[mode] += 1
                if max(
                    f1_by_context.get("gold_support_context", 0.0),
                    f1_by_context.get("phase1_oracle_context", 0.0),
                    f1_by_context.get("selected_plus_gold_context", 0.0),
                ) > f1_by_context.get("selected_context", 0.0):
                    per_mode_oracle_success[mode] += 1

                all_query_rows.append(
                    {
                        "dataset": ds,
                        "query_id": qid,
                        "prompt_mode": mode,
                        "question": str(sample.question),
                        "gold_answer": str(sample.answer),
                        "selected_context_prediction": pred_by_context.get("selected_context", ""),
                        "selected_context_em": 1.0 if f1_by_context.get("selected_context", 0.0) >= 1.0 else 0.0,
                        "selected_context_f1": f1_by_context.get("selected_context", 0.0),
                        "gold_support_context_prediction": pred_by_context.get("gold_support_context", ""),
                        "gold_support_context_em": 1.0 if f1_by_context.get("gold_support_context", 0.0) >= 1.0 else 0.0,
                        "gold_support_context_f1": f1_by_context.get("gold_support_context", 0.0),
                        "phase1_oracle_context_prediction": pred_by_context.get("phase1_oracle_context", ""),
                        "phase1_oracle_context_em": 1.0 if f1_by_context.get("phase1_oracle_context", 0.0) >= 1.0 else 0.0,
                        "phase1_oracle_context_f1": f1_by_context.get("phase1_oracle_context", 0.0),
                        "feature_ready_oracle_context_prediction": pred_by_context.get("feature_ready_oracle_context", ""),
                        "feature_ready_oracle_context_em": 1.0 if f1_by_context.get("feature_ready_oracle_context", 0.0) >= 1.0 else 0.0,
                        "feature_ready_oracle_context_f1": f1_by_context.get("feature_ready_oracle_context", 0.0),
                        "best_candidate_oracle_context_prediction": pred_by_context.get("best_candidate_oracle_context", ""),
                        "best_candidate_oracle_context_em": 1.0 if f1_by_context.get("best_candidate_oracle_context", 0.0) >= 1.0 else 0.0,
                        "best_candidate_oracle_context_f1": f1_by_context.get("best_candidate_oracle_context", 0.0),
                        "selected_plus_gold_context_prediction": pred_by_context.get("selected_plus_gold_context", ""),
                        "selected_plus_gold_context_em": 1.0 if f1_by_context.get("selected_plus_gold_context", 0.0) >= 1.0 else 0.0,
                        "selected_plus_gold_context_f1": f1_by_context.get("selected_plus_gold_context", 0.0),
                    }
                )

        for mode in PROMPT_MODES:
            cnt = max(1, int(per_mode_count.get(mode, 0)))
            selected_vals = per_mode[mode]["selected_context"]
            gold_vals = per_mode[mode]["gold_support_context"]
            phase1_vals = per_mode[mode]["phase1_oracle_context"]
            plus_vals = per_mode[mode]["selected_plus_gold_context"]
            feature_ready_vals = per_mode[mode]["feature_ready_oracle_context"]
            best_candidate_vals = per_mode[mode]["best_candidate_oracle_context"]

            selected_mean = float(sum(selected_vals) / len(selected_vals)) if selected_vals else 0.0
            gold_mean = float(sum(gold_vals) / len(gold_vals)) if gold_vals else 0.0
            phase1_mean = float(sum(phase1_vals) / len(phase1_vals)) if phase1_vals else 0.0
            plus_mean = float(sum(plus_vals) / len(plus_vals)) if plus_vals else 0.0
            feature_ready_mean = float(sum(feature_ready_vals) / len(feature_ready_vals)) if feature_ready_vals else 0.0
            best_candidate_mean = float(sum(best_candidate_vals) / len(best_candidate_vals)) if best_candidate_vals else 0.0

            summary_rows.append(
                {
                    "dataset": ds,
                    "prompt_mode": mode,
                    "selected_context_F1": selected_mean,
                    "gold_support_F1": gold_mean,
                    "phase1_oracle_F1": phase1_mean,
                    "feature_ready_oracle_F1": feature_ready_mean,
                    "best_candidate_oracle_F1": best_candidate_mean,
                    "selected_plus_gold_F1": plus_mean,
                    "oracle_success_rate": float(per_mode_oracle_success.get(mode, 0) / float(cnt)),
                    "selected_plus_gold_gain": float(plus_mean - selected_mean),
                    "gold_context_prompt_failure_rate": float(per_mode_gold_prompt_fail.get(mode, 0) / float(cnt)),
                    "n_failed_queries": int(per_mode_count.get(mode, 0)),
                    "source_run_dir": str(pick.run_dir),
                    "context_metrics": {
                        ctype: {
                            "EM": float((vals.get("em_sum", 0.0) / vals.get("n", 1.0)) if vals.get("n", 0.0) > 0 else 0.0),
                            "F1": float((vals.get("f1_sum", 0.0) / vals.get("n", 1.0)) if vals.get("n", 0.0) > 0 else 0.0),
                            "no_rate": float((vals.get("no_cnt", 0.0) / vals.get("n", 1.0)) if vals.get("n", 0.0) > 0 else 0.0),
                            "insufficient_information_rate": float(
                                (vals.get("insufficient_cnt", 0.0) / vals.get("n", 1.0)) if vals.get("n", 0.0) > 0 else 0.0
                            ),
                            "format_violation_rate": float(
                                (vals.get("format_violation_cnt", 0.0) / vals.get("n", 1.0)) if vals.get("n", 0.0) > 0 else 0.0
                            ),
                            "avg_output_tokens": float(
                                (vals.get("output_tokens_sum", 0.0) / vals.get("n", 1.0)) if vals.get("n", 0.0) > 0 else 0.0
                            ),
                            "n": int(vals.get("n", 0.0)),
                        }
                        for ctype, vals in per_mode_stats[mode].items()
                    },
                }
            )

    out_json = out_root / "oracle_context_results.json"
    out_jsonl = out_root / "phase7_oracle_replay_results.jsonl"
    out_payload = {
        "generated_at": timestamp_iso_utc(),
        "root": str(out_root),
        "summary_rows": summary_rows,
        "query_rows": all_query_rows,
    }
    out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in all_query_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    md_lines: List[str] = []
    md_lines.append("# Phase7 Oracle Context QA Report")
    md_lines.append("")
    table_rows: List[List[str]] = []
    for row in summary_rows:
        table_rows.append(
            [
                str(row.get("dataset", "")),
                str(row.get("prompt_mode", "")),
                _fmt4(row.get("selected_context_F1", 0.0)),
                _fmt4(row.get("gold_support_F1", 0.0)),
                _fmt4(row.get("phase1_oracle_F1", 0.0)),
                _fmt4(row.get("feature_ready_oracle_F1", 0.0)),
                _fmt4(row.get("best_candidate_oracle_F1", 0.0)),
                _fmt4(row.get("selected_plus_gold_F1", 0.0)),
                _fmt4(row.get("oracle_success_rate", 0.0)),
                _fmt4(row.get("selected_plus_gold_gain", 0.0)),
                _fmt4(row.get("gold_context_prompt_failure_rate", 0.0)),
                str(int(row.get("n_failed_queries", 0))),
            ]
        )
    md_lines.append(
        _table(
            [
                "dataset",
                "prompt_mode",
                "selected_context_F1",
                "gold_support_F1",
                "phase1_oracle_F1",
                "feature_ready_oracle_F1",
                "best_candidate_oracle_F1",
                "selected_plus_gold_F1",
                "oracle_success_rate",
                "selected_plus_gold_gain",
                "gold_context_prompt_failure_rate",
                "n_failed",
            ],
            table_rows,
        )
    )

    out_md = Path(args.out).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(out_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
