#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.eval_adapters import (  # noqa: E402
    adapt_effirag_from_summary,
    discover_effirag_summary_path,
    load_qa_index,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _normalize_space(text: Any) -> str:
    return " ".join(_strip(text).split())


def _parse_csv(raw: str) -> List[str]:
    out: List[str] = []
    for item in str(raw or "").split(","):
        v = item.strip()
        if v:
            out.append(v)
    return out


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = _strip(line)
            if not text:
                continue
            rows.append(dict(json.loads(text) or {}))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            n += 1
    return int(n)


def _write_tsv(path: Path, headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def _resolve_qa_item(qa_index: Any, qid: str, question: str, sample_index: int) -> Optional[Dict[str, Any]]:
    q = _strip(question)
    sid = _strip(qid)
    if sid and sid in qa_index.by_qid:
        return dict(qa_index.by_qid[sid] or {})
    if q:
        cands = list(qa_index.by_question.get(q, []) or [])
        if len(cands) == 1:
            return dict(cands[0] or {})
        if cands:
            return dict(cands[0] or {})
    if 0 <= sample_index < len(qa_index.by_index):
        return dict(qa_index.by_index[sample_index] or {})
    return None


def _parse_effirag_line(line: str) -> Tuple[str, str]:
    raw = _strip(line)
    if raw.startswith("- "):
        raw = raw[2:].strip()
    if raw.startswith("• "):
        raw = raw[2:].strip()
    if ": " in raw:
        title, text = raw.split(": ", 1)
        t = _strip(title)
        x = _strip(text)
        if t and x:
            return t, x
    return "", raw


def _parse_effirag_evidence(rendered_context: str) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for line in str(rendered_context or "").splitlines():
        line = _strip(line)
        if not line:
            continue
        title, text = _parse_effirag_line(line)
        evidence.append(
            {
                "id": len(evidence) + 1,
                "title": title,
                "text": text,
                "rank": len(evidence) + 1,
            }
        )
    return evidence


def _parse_hipporag_doc_block(doc: str, rank: int) -> Dict[str, Any]:
    content = _strip(doc)
    title = ""
    text = content

    if "\n" in content:
        first, rest = content.split("\n", 1)
        first = _strip(first)
        rest = _strip(rest)
        first_norm = first
        if first_norm.lower().startswith("wikipedia title:"):
            first_norm = _strip(first_norm.split(":", 1)[1])
        if first_norm:
            title = first_norm
        if rest:
            text = rest
    elif content.lower().startswith("wikipedia title:"):
        title = _strip(content.split(":", 1)[1])
        text = content

    return {
        "id": int(rank),
        "title": title,
        "text": text,
        "rank": int(rank),
    }


def _sentence_split(text: str) -> List[str]:
    raw = _strip(text)
    if not raw:
        return []
    parts = re.split(r"(?<=[.!?])\s+", raw)
    out: List[str] = []
    for p in parts:
        s = _strip(p)
        if s:
            out.append(s)
    return out


def _build_hipporag_sentence_evidence(docs: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in list(docs or []):
        block = _parse_hipporag_doc_block(doc, rank=0)
        title = _strip(block.get("title"))
        text = _strip(block.get("text"))
        sents = _sentence_split(text)
        if not sents:
            sents = [text] if text else []
        for sent in sents:
            out.append(
                {
                    "id": len(out) + 1,
                    "title": title,
                    "text": sent,
                    "rank": len(out) + 1,
                }
            )
    return out


def _as_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _find_dataset_path(qa_root: Path, dataset: str) -> Path:
    p = qa_root / f"{dataset}.json"
    if p.exists():
        return p
    alt = REPO_ROOT / "data" / "qa" / f"{dataset}.json"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"QA file not found for dataset={dataset} (checked {p} and {alt})")


def _discover_hipporag_metric_path(hippo_root: Path, dataset: str) -> Optional[Path]:
    base_roots = [hippo_root]
    abgf_dir = hippo_root / "abgf_calibration"
    if abgf_dir.exists():
        base_roots.append(abgf_dir)

    rel_candidates = [
        f"{dataset}/hipporag2_abgf_calibration_metrics.json",
        f"{dataset}_full/hipporag2_abgf_calibration_metrics.json",
        f"{dataset}_full_verify/hipporag2_abgf_calibration_metrics.json",
        f"{dataset}_smoke/hipporag2_abgf_calibration_metrics.json",
        f"batch_verify/{dataset}/hipporag2_abgf_calibration_metrics.json",
    ]

    for base in base_roots:
        for rel in rel_candidates:
            cand = (base / rel).resolve()
            if cand.exists():
                return cand
    return None


def _load_hipporag_records(
    *,
    hipporag_root: Path,
    dataset: str,
    qa_index: Any,
    max_records: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    warnings: List[str] = []
    metric_path = _discover_hipporag_metric_path(hipporag_root, dataset)
    if metric_path is None:
        raise FileNotFoundError(
            f"HippoRAG2 calibration metrics not found for dataset={dataset} under {hipporag_root}"
        )

    metric_payload = dict(_read_json(metric_path) or {})
    query_path = Path(_strip(metric_payload.get("paths", {}).get("query_solutions_path"))).resolve()
    if not query_path.exists():
        raise FileNotFoundError(f"query_solutions_path does not exist: {query_path}")

    rows = list(_read_json(query_path) or [])
    if max_records > 0:
        rows = rows[:max_records]

    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        sample = dict(row or {})
        question = _strip(sample.get("question"))
        qa_item = _resolve_qa_item(qa_index, qid="", question=question, sample_index=i)
        if qa_item is None:
            warnings.append(f"qa lookup miss for hipporag2 dataset={dataset} sample_index={i}")
        qid = _strip((qa_item or {}).get("qid")) or f"{dataset}-hp-{i:06d}"
        gold = _strip((qa_item or {}).get("answer"))
        if not gold:
            gold_answers = list(sample.get("gold_answers", []) or [])
            gold = _strip(gold_answers[0] if gold_answers else "")

        docs = [str(x) for x in list(sample.get("docs", []) or []) if _strip(x)]
        evidence_blocks = [_parse_hipporag_doc_block(doc, rank=j + 1) for j, doc in enumerate(docs)]
        evidence_sentences = _build_hipporag_sentence_evidence(docs)

        records.append(
            {
                "dataset": dataset,
                "qid": qid,
                "question": question or _strip((qa_item or {}).get("question")),
                "gold_answer": gold,
                "evidence_block": evidence_blocks,
                "evidence_sentence": evidence_sentences,
                "rendered_context": "\n\n".join(docs),
                "source_sample_index": i,
                "source_path": str(query_path),
            }
        )

    source_meta = {
        "dataset_metric_path": str(metric_path),
        "query_solutions_path": str(query_path),
    }
    return records, source_meta, warnings


@dataclass(frozen=True)
class SystemSpec:
    system_name: str
    kind: str
    variant: str


def _parse_systems(raw_csv: str) -> List[SystemSpec]:
    out: List[SystemSpec] = []
    for token in _parse_csv(raw_csv):
        t = _strip(token)
        tl = t.lower()
        if tl.startswith("effirag_"):
            variant = t[len("effirag_") :]
            if not variant:
                raise ValueError(f"invalid effirag system token: {token}")
            out.append(SystemSpec(system_name=t, kind="effirag", variant=variant))
            continue
        if tl in {"hipporag2", "hipporag"}:
            out.append(SystemSpec(system_name=t, kind="hipporag2", variant="hipporag2"))
            continue
        raise ValueError(f"unsupported system token: {token}")
    if not out:
        raise ValueError("no systems parsed from --systems")
    return out


def _build_labels(n: int) -> List[str]:
    if n <= 0:
        return []
    labels: List[str] = []
    alphabet = [chr(ord("A") + i) for i in range(26)]
    for i in range(n):
        if i < 26:
            labels.append(alphabet[i])
            continue
        quotient = i // 26
        remainder = i % 26
        labels.append(f"{alphabet[quotient - 1]}{alphabet[remainder]}")
    return labels


def _chunked(seq: Sequence[Mapping[str, Any]], size: int) -> Iterable[Tuple[int, Sequence[Mapping[str, Any]]]]:
    chunk_size = max(1, int(size))
    for i in range(0, len(seq), chunk_size):
        yield i // chunk_size, seq[i : i + chunk_size]


def _render_sample_preview_md(rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    lines: List[str] = ["# Sample 20 Judge Records", ""]
    for row in rows:
        evidence = list(row.get("evidence", []) or [])
        ev_lines: List[str] = []
        for item in evidence[:3]:
            title = _strip(item.get("title"))
            text = _strip(item.get("text"))
            if title:
                ev_lines.append(f"- {title}: {text[:160]}")
            else:
                ev_lines.append(f"- {text[:160]}")
        lines.extend(
            [
                f"## {row.get('judge_id')}",
                f"- dataset: {row.get('dataset')}",
                f"- qid: {row.get('qid')}",
                f"- system_id: {row.get('system_id')}",
                f"- question: {row.get('question')}",
                f"- gold_answer: {row.get('gold_answer')}",
                "- evidence (first 3):",
                *ev_lines,
                "",
            ]
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_prompts(prompt_dir: Path) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)

    rubric = """You are an evidence sufficiency judge.

Given a question, a gold answer, and retrieved evidence lines, determine whether the evidence is sufficient to justify the gold answer.

Use only the provided evidence. Do not use outside knowledge.

Scoring:
0 = No relevant evidence.
1 = Partial evidence only; the evidence is related but insufficient to justify the gold answer.
2 = The answer surface appears, but the relation needed to answer the question is not supported.
3 = Sufficient equivalent evidence; the evidence justifies the gold answer even if it does not exactly match annotated support facts.
4 = Minimal sufficient evidence; the evidence contains a compact set of facts sufficient to justify the gold answer.

Tasks:
1. Decide whether the evidence is sufficient.
2. Select the smallest evidence IDs sufficient to justify the gold answer.
3. Indicate whether the answer surface appears in the evidence.
4. Indicate whether the evidence is equivalent to the gold support, even if wording differs.
5. Return JSON only.

Output schema:
{
  "judge_id": "...",
  "sufficient": true,
  "score": 4,
  "minimal_evidence_ids": [1, 2],
  "answer_surface_present": true,
  "equivalent_evidence": true,
  "insufficient_reason": null
}
"""
    (prompt_dir / "llm_judge_rubric.md").write_text(rubric, encoding="utf-8")

    operational = """Read the provided judge input JSONL file.

For each record:
1. Apply the judge rubric exactly.
2. Evaluate each record independently.
3. Use only the question, gold_answer, and evidence fields.
4. Do not use outside knowledge.
5. Do not infer the true system behind system_id.
6. Return one JSON object per input record.
7. Do not skip any record.
8. If a record cannot be judged, return a JSON object with "error": "...".
9. Do not modify the input file.
10. Write the results to the specified output JSONL file.

Required output fields:
- judge_id
- sufficient
- score
- minimal_evidence_ids
- answer_surface_present
- equivalent_evidence
- insufficient_reason
- error
"""
    (prompt_dir / "operational_instruction_for_chatgpt_or_codex.md").write_text(operational, encoding="utf-8")


def _write_expected_output_schema(path: Path) -> None:
    schema = {
        "type": "object",
        "required": [
            "judge_id",
            "sufficient",
            "score",
            "minimal_evidence_ids",
            "answer_surface_present",
            "equivalent_evidence",
            "insufficient_reason",
            "error",
        ],
        "properties": {
            "judge_id": {"type": "string"},
            "sufficient": {"type": ["boolean", "null"]},
            "score": {"type": ["integer", "number", "null"], "minimum": 0, "maximum": 4},
            "minimal_evidence_ids": {"type": "array", "items": {"type": "integer"}},
            "answer_surface_present": {"type": ["boolean", "null"]},
            "equivalent_evidence": {"type": ["boolean", "null"]},
            "insufficient_reason": {"type": ["string", "null"]},
            "error": {"type": ["string", "null"]},
        },
    }
    _write_json(path, schema)


def _copy_aggregation_scripts(round_dir: Path) -> None:
    agg_dir = round_dir / "aggregation"
    agg_dir.mkdir(parents=True, exist_ok=True)
    src_agg = REPO_ROOT / "scripts" / "aggregate_llm_judge_outputs_template.py"
    src_validate = REPO_ROOT / "scripts" / "validate_judge_outputs.py"
    if src_agg.exists():
        shutil.copy2(src_agg, agg_dir / "aggregate_judge_outputs_template.py")
    if src_validate.exists():
        shutil.copy2(src_validate, agg_dir / "validate_judge_outputs.py")
    _write_expected_output_schema(agg_dir / "expected_judge_output_schema.json")


def _run_validation(round_dir: Path) -> Tuple[int, str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_llm_judge_inputs.py"),
        "--round-dir",
        str(round_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_text = "\n".join([_strip(proc.stdout), _strip(proc.stderr)]).strip()
    return int(proc.returncode), log_text


def _build_readme(
    round_dir: Path,
    *,
    source_paths: Mapping[str, Any],
    datasets: Sequence[str],
    systems: Sequence[str],
    chunk_size: int,
) -> None:
    text = f"""# LLM Judge Input Round

- Generated at (UTC): {_utc_now_iso()}
- Output directory: {round_dir}
- Datasets: {", ".join(datasets)}
- Systems: {", ".join(systems)}
- Blind mapping policy: per_query_randomized
- Chunk size: {int(chunk_size)}

## Source Paths
```json
{json.dumps(dict(source_paths or {}), ensure_ascii=False, indent=2)}
```

## Prompt Files
- Rubric: `prompts/llm_judge_rubric.md`
- Operational instruction: `prompts/operational_instruction_for_chatgpt_or_codex.md`

## Validation
- JSON report: `validation/input_validation_report.json`
- Markdown report: `validation/input_validation_report.md`

## Next Step
1. Open `prompts/llm_judge_rubric.md`.
2. Open `prompts/operational_instruction_for_chatgpt_or_codex.md`.
3. For each chunk in `judge_chunks/`, apply the judge rubric.
4. Save outputs as `judge_outputs/<chunk_name>_judged.jsonl`.
5. Run `aggregation/validate_judge_outputs.py`.
6. Run `aggregation/aggregate_judge_outputs_template.py`.
"""
    (round_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build LLM-judged evidence sufficiency input files.")
    ap.add_argument("--datasets", required=True, help="CSV datasets (e.g., hotpotqa,2wikimultihopqa)")
    ap.add_argument("--systems", required=True, help="CSV systems")
    ap.add_argument("--effirag-output-root", required=True, help="EffiRAG round output root")
    ap.add_argument("--hipporag2-output-root", required=True, help="HippoRAG2 output root or abgf_calibration root")
    ap.add_argument("--qa-root", required=True, help="QA dataset root containing <dataset>.json")
    ap.add_argument("--output-root", required=True, help="Parent output root")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-records-per-dataset", type=int, default=1000)
    ap.add_argument("--timestamp", default="", help="Optional fixed timestamp dir name")
    ap.add_argument("--print-summary", default="true", help="Print summary to stdout")
    args = ap.parse_args()

    datasets = _parse_csv(args.datasets)
    systems = _parse_systems(args.systems)
    chunk_size = max(1, int(args.chunk_size))
    max_records = max(1, int(args.max_records_per_dataset))

    output_parent = Path(args.output_root).resolve()
    round_dir = output_parent / (_strip(args.timestamp) or _utc_stamp())
    round_dir.mkdir(parents=True, exist_ok=True)

    judge_inputs_dir = round_dir / "judge_inputs"
    judge_chunks_dir = round_dir / "judge_chunks"
    hidden_mapping_dir = round_dir / "hidden_mapping"
    prompts_dir = round_dir / "prompts"
    validation_dir = round_dir / "validation"
    previews_dir = round_dir / "previews"
    aggregation_dir = round_dir / "aggregation"
    for d in [
        judge_inputs_dir,
        judge_chunks_dir,
        hidden_mapping_dir,
        prompts_dir,
        validation_dir,
        previews_dir,
        aggregation_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    source_paths: Dict[str, Any] = {
        "effirag_output_root": str(Path(args.effirag_output_root).resolve()),
        "hipporag2_output_root": str(Path(args.hipporag2_output_root).resolve()),
        "qa_root": str(Path(args.qa_root).resolve()),
    }

    failures: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    all_judge_records: List[Dict[str, Any]] = []
    all_mapping_rows: List[Dict[str, Any]] = []
    dataset_records: Dict[str, List[Dict[str, Any]]] = {}
    dataset_counts: Dict[str, Dict[str, int]] = {}
    hippo_block_records: List[Dict[str, Any]] = []
    hippo_sentence_records: List[Dict[str, Any]] = []
    labels = _build_labels(len(systems))

    for dataset in datasets:
        try:
            qa_path = _find_dataset_path(Path(args.qa_root), dataset)
            qa_index = load_qa_index(str(qa_path))
        except Exception as exc:
            failures.append({"dataset": dataset, "stage": "qa_load", "error": str(exc)})
            continue

        per_system_rows: Dict[str, List[Dict[str, Any]]] = {}
        per_system_source_path: Dict[str, str] = {}
        per_system_variant: Dict[str, str] = {}

        for ss in systems:
            try:
                if ss.kind == "effirag":
                    summary_path, disc_meta = discover_effirag_summary_path(
                        effirag_output_root=args.effirag_output_root,
                        dataset=dataset,
                        variant=ss.variant,
                    )
                    if summary_path is None:
                        raise FileNotFoundError(
                            f"summary not found for dataset={dataset}, variant={ss.variant}; warnings={disc_meta.get('warnings')}"
                        )
                    adapted = adapt_effirag_from_summary(
                        summary_path=str(summary_path),
                        dataset=dataset,
                        variant=ss.variant,
                        qa_index=qa_index,
                        n_samples=max_records,
                    )
                    rows: List[Dict[str, Any]] = []
                    for r in list(adapted.get("records", []) or []):
                        rendered_context = _strip(r.get("rendered_context"))
                        evidence = _parse_effirag_evidence(rendered_context)
                        if not evidence:
                            fallback_items = [str(x) for x in list(r.get("retrieved_items", []) or []) if _strip(x)]
                            evidence = [
                                {"id": j + 1, "title": "", "text": txt, "rank": j + 1}
                                for j, txt in enumerate(fallback_items)
                            ]
                        rows.append(
                            {
                                "dataset": dataset,
                                "qid": _strip(r.get("qid")),
                                "question": _strip(r.get("question")),
                                "gold_answer": _strip(r.get("gold_answer")),
                                "evidence_block": evidence,
                                "evidence_sentence": evidence,
                                "rendered_context": rendered_context,
                                "source_path": str(adapted.get("query_path", "")),
                            }
                        )
                    per_system_rows[ss.system_name] = rows
                    per_system_source_path[ss.system_name] = str(adapted.get("query_path", ""))
                    per_system_variant[ss.system_name] = ss.variant

                    run_rows.append(
                        {
                            "dataset": dataset,
                            "system": ss.system_name,
                            "source": "effirag",
                            "variant": ss.variant,
                            "source_path": str(adapted.get("query_path", "")),
                            "raw_records": len(rows),
                            "common_records": "",
                            "status": "ok",
                            "note": "",
                        }
                    )
                elif ss.kind == "hipporag2":
                    rows, src_meta, warns = _load_hipporag_records(
                        hipporag_root=Path(args.hipporag2_output_root).resolve(),
                        dataset=dataset,
                        qa_index=qa_index,
                        max_records=max_records,
                    )
                    per_system_rows[ss.system_name] = rows
                    per_system_source_path[ss.system_name] = str(src_meta.get("query_solutions_path", ""))
                    per_system_variant[ss.system_name] = ss.variant
                    for w in warns:
                        failures.append({"dataset": dataset, "system": ss.system_name, "stage": "warning", "error": w})

                    for rec in rows:
                        hippo_block_records.append(
                            {
                                "dataset": dataset,
                                "qid": rec.get("qid"),
                                "question": rec.get("question"),
                                "gold_answer": rec.get("gold_answer"),
                                "evidence": rec.get("evidence_block", []),
                            }
                        )
                        hippo_sentence_records.append(
                            {
                                "dataset": dataset,
                                "qid": rec.get("qid"),
                                "question": rec.get("question"),
                                "gold_answer": rec.get("gold_answer"),
                                "evidence": rec.get("evidence_sentence", []),
                            }
                        )

                    run_rows.append(
                        {
                            "dataset": dataset,
                            "system": ss.system_name,
                            "source": "hipporag2",
                            "variant": ss.variant,
                            "source_path": str(src_meta.get("query_solutions_path", "")),
                            "raw_records": len(rows),
                            "common_records": "",
                            "status": "ok",
                            "note": "",
                        }
                    )
                else:
                    raise ValueError(f"Unsupported system kind: {ss.kind}")
            except Exception as exc:
                failures.append(
                    {
                        "dataset": dataset,
                        "system": ss.system_name,
                        "stage": "load_system_records",
                        "error": str(exc),
                    }
                )
                run_rows.append(
                    {
                        "dataset": dataset,
                        "system": ss.system_name,
                        "source": ss.kind,
                        "variant": ss.variant,
                        "source_path": "",
                        "raw_records": 0,
                        "common_records": 0,
                        "status": "error",
                        "note": str(exc),
                    }
                )

        available_systems = [ss.system_name for ss in systems if ss.system_name in per_system_rows]
        if len(available_systems) != len(systems):
            failures.append(
                {
                    "dataset": dataset,
                    "stage": "dataset_skip",
                    "error": f"missing systems for dataset: expected={len(systems)} got={len(available_systems)}",
                }
            )
            continue

        by_system_qid: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for sys_name in available_systems:
            qmap: Dict[str, Dict[str, Any]] = {}
            for rec in list(per_system_rows.get(sys_name, []) or []):
                qid = _strip(rec.get("qid"))
                if not qid:
                    continue
                if qid not in qmap:
                    qmap[qid] = dict(rec)
            by_system_qid[sys_name] = qmap

        common_qids: Optional[set[str]] = None
        for sys_name in available_systems:
            qids = set(by_system_qid[sys_name].keys())
            common_qids = qids if common_qids is None else common_qids.intersection(qids)
        common_qids = common_qids or set()

        ordered_qids: List[str] = []
        qa_order = [str(item.get("qid")) for item in list(qa_index.by_index or [])]
        for qid in qa_order:
            if qid in common_qids:
                ordered_qids.append(qid)
        for qid in sorted(common_qids):
            if qid not in ordered_qids:
                ordered_qids.append(qid)

        if max_records > 0:
            ordered_qids = ordered_qids[:max_records]

        dataset_records[dataset] = []
        dataset_counts[dataset] = {}

        for rr in run_rows:
            if rr.get("dataset") == dataset and rr.get("status") == "ok":
                rr["common_records"] = len(ordered_qids)

        for qidx, qid in enumerate(ordered_qids, start=1):
            rng = random.Random(f"{args.seed}|{dataset}|{qid}")
            sys_perm = list(available_systems)
            rng.shuffle(sys_perm)
            blind_map = {sys_perm[i]: labels[i] for i in range(len(sys_perm))}
            # Sort records by label so A/B/C order is stable for judges.
            sorted_sys = sorted(sys_perm, key=lambda s: blind_map[s])

            for sys_name in sorted_sys:
                src = dict(by_system_qid[sys_name][qid] or {})
                sid = blind_map[sys_name]
                judge_id = f"{dataset}_{qidx:06d}_{sid}"
                evidence = list(src.get("evidence_block", []) or [])
                judge_record = {
                    "judge_id": judge_id,
                    "dataset": dataset,
                    "qid": qid,
                    "system_id": sid,
                    "question": _strip(src.get("question")),
                    "gold_answer": _strip(src.get("gold_answer")),
                    "evidence": evidence,
                    "metadata": {
                        "n_evidence": len(evidence),
                        "source_variant_hidden": True,
                        "include_prediction": False,
                        "has_gold_supports_hidden": True,
                    },
                }
                dataset_records[dataset].append(judge_record)
                all_judge_records.append(judge_record)

                true_system = "effirag" if sys_name.lower().startswith("effirag_") else "hipporag2"
                true_variant = per_system_variant.get(sys_name, "")
                all_mapping_rows.append(
                    {
                        "judge_id": judge_id,
                        "dataset": dataset,
                        "qid": qid,
                        "system_id": sid,
                        "true_system": true_system,
                        "true_variant": true_variant,
                        "source_path": per_system_source_path.get(sys_name, ""),
                        "source_system_name": sys_name,
                    }
                )
                dataset_counts[dataset][sys_name] = int(dataset_counts[dataset].get(sys_name, 0) + 1)

        _write_jsonl(judge_inputs_dir / f"{dataset}_records.jsonl", dataset_records[dataset])
        for idx, chunk_rows in _chunked(dataset_records[dataset], chunk_size):
            _write_jsonl(judge_chunks_dir / f"{dataset}_chunk_{idx:03d}.jsonl", chunk_rows)

    all_judge_records.sort(key=lambda r: (_strip(r.get("dataset")), _strip(r.get("judge_id"))))
    all_mapping_rows.sort(key=lambda r: (_strip(r.get("dataset")), _strip(r.get("judge_id"))))

    _write_jsonl(judge_inputs_dir / "all_records.jsonl", all_judge_records)
    _write_jsonl(hidden_mapping_dir / "record_mapping.jsonl", all_mapping_rows)

    # Optional explicit HippoRAG2 granularity exports.
    _write_jsonl(judge_inputs_dir / "hipporag2_block_records.jsonl", hippo_block_records)
    _write_jsonl(judge_inputs_dir / "hipporag2_sentence_records.jsonl", hippo_sentence_records)

    system_blind_mapping = {
        "mapping_policy": "per_query_randomized",
        "seed": int(args.seed),
        "systems": [ss.system_name for ss in systems],
        "labels": labels,
        "note": "system_id labels are randomized per query to reduce order/system bias.",
    }
    _write_json(hidden_mapping_dir / "system_blind_mapping.json", system_blind_mapping)

    _write_prompts(prompts_dir)
    _copy_aggregation_scripts(round_dir)

    sample_preview_rows = all_judge_records[:20]
    _write_jsonl(previews_dir / "sample_20_records.jsonl", sample_preview_rows)
    _render_sample_preview_md(sample_preview_rows, previews_dir / "sample_20_records.md")

    manifest = {
        "generated_at_utc": _utc_now_iso(),
        "round_name": "llm_judge_input_round",
        "round_dir": str(round_dir),
        "datasets": datasets,
        "systems": [ss.system_name for ss in systems],
        "chunk_size": chunk_size,
        "max_records_per_dataset": max_records,
        "seed": int(args.seed),
        "source_paths": source_paths,
        "counts": {
            "all_records": len(all_judge_records),
            "all_mapping_rows": len(all_mapping_rows),
            "dataset_records": {ds: len(dataset_records.get(ds, [])) for ds in datasets},
            "dataset_system_counts": dataset_counts,
            "chunk_files": len(list(judge_chunks_dir.glob("*.jsonl"))),
        },
        "integrity_caution": {
            ds: (
                "integrity caution: optional dataset; retrieval-fixed consistency may differ from hotpotqa/2wikimultihopqa"
                if ds in {"musique", "popqa"}
                else ""
            )
            for ds in datasets
        },
    }
    _write_json(round_dir / "manifest.json", manifest)

    failures_payload = {"count": len(failures), "items": failures}
    _write_json(round_dir / "failures.json", failures_payload)

    run_headers = [
        "dataset",
        "system",
        "source",
        "variant",
        "source_path",
        "raw_records",
        "common_records",
        "status",
        "note",
    ]
    _write_tsv(round_dir / "run_records.tsv", run_headers, run_rows)

    _build_readme(
        round_dir,
        source_paths=source_paths,
        datasets=datasets,
        systems=[ss.system_name for ss in systems],
        chunk_size=chunk_size,
    )

    # Validation and report files.
    rc, validation_log = _run_validation(round_dir)
    if validation_log:
        (validation_dir / "validator_stdout.log").write_text(validation_log + "\n", encoding="utf-8")

    if _as_bool(args.print_summary):
        print(json.dumps({"round_dir": str(round_dir), "validation_rc": rc, "failures": len(failures)}, indent=2))
        print(f"[DONE] {round_dir}")

    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
