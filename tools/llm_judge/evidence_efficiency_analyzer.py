#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - handled by runtime fallback
    AutoTokenizer = None  # type: ignore[assignment]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _strip(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or _strip(value) == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or _strip(value) == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _strip(value).lower() in {"1", "true", "yes", "y", "on"}


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    xs = sorted(float(v) for v in values)
    q_clamped = max(0.0, min(1.0, float(q)))
    idx = (len(xs) - 1) * q_clamped
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(xs[lo])
    frac = idx - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = _strip(line)
            if not text:
                continue
            rows.append(dict(json.loads(text)))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _render_markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals: List[str] = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            elif value is None:
                vals.append("-")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _resolve_input_root(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir() and (path / "judge_chunks").exists():
        return (path / "judge_chunks").resolve()
    return path


def _resolve_output_root(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir() and (path / "judge_outputs").exists():
        return (path / "judge_outputs").resolve()
    return path


def _infer_round_root_from_input(input_root: Path) -> Optional[Path]:
    input_root = input_root.resolve()
    if input_root.name == "judge_chunks":
        return input_root.parent
    if (input_root / "judge_chunks").exists():
        return input_root
    return None


class TokenCounter:
    def __init__(self, tokenizer_name: str) -> None:
        self.tokenizer_name = _strip(tokenizer_name)
        self.mode = "whitespace"
        self.warning: Optional[str] = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        if AutoTokenizer is None:
            self.warning = "transformers not available; using whitespace token fallback"
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=True,
            )
            self.mode = "hf_tokenizer"
        except Exception as exc:
            self.warning = (
                f"failed to load tokenizer `{self.tokenizer_name}` from local cache; "
                f"using whitespace token fallback: {type(exc).__name__}: {exc}"
            )
            self._tokenizer = None
            self.mode = "whitespace"

    def count(self, text: str) -> int:
        text = str(text or "")
        if not text.strip():
            return 0
        if self._tokenizer is not None:
            try:
                return int(len(self._tokenizer.encode(text, add_special_tokens=False)))
            except Exception:
                pass
        return len(re.findall(r"\S+", text))


def _normalize_variant_aliases(value: str) -> List[str]:
    value = _strip(value)
    if not value:
        return [""]
    aliases = {value}
    lower = value.lower()
    if lower.startswith("hipporag"):
        aliases.add("hipporag2")
        aliases.add("hipporag__tangent__tempna")
    if lower == "hipporag2":
        aliases.add("hipporag__tangent__tempNA")
    return sorted(aliases)


def _normalize_system_aliases(value: str) -> List[str]:
    value = _strip(value)
    if not value:
        return [""]
    aliases = {value}
    lower = value.lower()
    if lower.startswith("hipporag"):
        aliases.add("hipporag2")
    return sorted(aliases)


def _render_evidence_item_text(item: Mapping[str, Any]) -> str:
    title = _strip(item.get("title"))
    text = _strip(item.get("text"))
    if title:
        return f"[{title}] {text}".strip()
    return text


def _load_input_records(input_root: Path, max_records: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    duplicate_ids: List[str] = []
    seen: set[str] = set()
    for fp in sorted(input_root.glob("*.jsonl")):
        for row in _read_jsonl(fp):
            judge_id = _strip(row.get("judge_id"))
            if not judge_id:
                continue
            if judge_id in seen:
                duplicate_ids.append(judge_id)
                continue
            seen.add(judge_id)
            rr = dict(row)
            rr["_input_file"] = fp.name
            rows.append(rr)
            if max_records > 0 and len(rows) >= max_records:
                return rows, sorted(set(duplicate_ids))
    return rows, sorted(set(duplicate_ids))


def _load_output_rows(
    output_roots: Sequence[Path],
    duplicate_policy: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    replaced: List[str] = []
    parse_errors: List[str] = []
    output_files: List[str] = []

    for root in output_roots:
        for fp in sorted(root.glob("*_judged.jsonl")):
            output_files.append(str(fp))
            try:
                rows = _read_jsonl(fp)
            except Exception as exc:
                parse_errors.append(f"{fp}: {type(exc).__name__}: {exc}")
                continue
            for row in rows:
                judge_id = _strip(row.get("judge_id"))
                if not judge_id:
                    continue
                rr = dict(row)
                rr["_output_file"] = str(fp)
                rr["_output_root"] = str(root)
                if judge_id in by_id:
                    duplicates.append(judge_id)
                    if duplicate_policy == "prefer-last":
                        replaced.append(judge_id)
                        by_id[judge_id] = rr
                    continue
                by_id[judge_id] = rr

    report = {
        "duplicate_output_judge_ids": sorted(set(duplicates)),
        "replaced_output_judge_ids": sorted(set(replaced)),
        "parse_errors": parse_errors,
        "output_file_count": len(output_files),
    }
    return by_id, report


def _load_mapping(mapping_root: Optional[Path]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if mapping_root is None:
        return {}, warnings
    mapping_root = mapping_root.resolve()
    mapping_path = mapping_root / "record_mapping.jsonl"
    if not mapping_path.exists():
        warnings.append(f"mapping file not found: {mapping_path}")
        return {}, warnings
    rows = _read_jsonl(mapping_path)
    return {_strip(r.get("judge_id")): dict(r) for r in rows if _strip(r.get("judge_id"))}, warnings


def _load_input_run_records(round_root: Optional[Path]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    if round_root is None:
        return {}
    tsv_path = round_root / "run_records.tsv"
    if not tsv_path.exists():
        return {}
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dataset = _strip(row.get("dataset"))
            source = _strip(row.get("source"))
            variant = _strip(row.get("variant"))
            if not dataset or not source:
                continue
            out[(dataset, source, variant)] = dict(row)
    return out


def _iter_qa_metric_files(paths: Sequence[Path]) -> Iterable[Path]:
    for raw in paths:
        path = raw.resolve()
        if path.is_file():
            yield path
            continue
        if path.is_dir():
            for fp in sorted(path.rglob("*.json")):
                yield fp
            for fp in sorted(path.rglob("*.jsonl")):
                yield fp


def _maybe_add_qa_row(
    rows: List[Dict[str, Any]],
    *,
    dataset: Any,
    system: Any,
    variant: Any,
    em: Any,
    f1: Any,
    source_file: Path,
) -> None:
    ds = _strip(dataset)
    if not ds:
        return
    em_val = None
    f1_val = None
    if em is not None and _strip(em) != "":
        em_val = _safe_float(em)
    if f1 is not None and _strip(f1) != "":
        f1_val = _safe_float(f1)
    if em_val is None and f1_val is None:
        return

    sys_val = _strip(system)
    var_val = _strip(variant)
    lower_path = str(source_file).lower()
    if not sys_val and ("hipporag2" in lower_path or "hipporag" in lower_path):
        sys_val = "hipporag2"
    if not var_val:
        if sys_val == "hipporag2":
            var_val = "hipporag2"
    rows.append(
        {
            "dataset": ds,
            "system": sys_val,
            "variant": var_val,
            "EM": em_val,
            "F1": f1_val,
            "source_file": str(source_file),
        }
    )


def _extract_qa_rows_from_object(obj: Any, source_file: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("records"), list):
            for rec in obj.get("records", []):
                if not isinstance(rec, dict):
                    continue
                _maybe_add_qa_row(
                    out,
                    dataset=rec.get("dataset", obj.get("dataset")),
                    system=rec.get("system", rec.get("true_system", obj.get("system"))),
                    variant=rec.get("variant", rec.get("true_variant", obj.get("variant"))),
                    em=rec.get("em", rec.get("EM")),
                    f1=rec.get("f1", rec.get("F1")),
                    source_file=source_file,
                )
        elif isinstance(obj.get("rows"), list):
            for rec in obj.get("rows", []):
                if not isinstance(rec, dict):
                    continue
                _maybe_add_qa_row(
                    out,
                    dataset=rec.get("dataset", obj.get("dataset")),
                    system=rec.get("system", rec.get("true_system", obj.get("system"))),
                    variant=rec.get("variant", rec.get("true_variant", obj.get("variant"))),
                    em=rec.get("em", rec.get("EM")),
                    f1=rec.get("f1", rec.get("F1")),
                    source_file=source_file,
                )
        elif isinstance(obj.get("metrics"), dict):
            metrics = dict(obj.get("metrics") or {})
            _maybe_add_qa_row(
                out,
                dataset=obj.get("dataset"),
                system=obj.get("system"),
                variant=obj.get("variant"),
                em=metrics.get("EM", obj.get("EM")),
                f1=metrics.get("F1", obj.get("F1")),
                source_file=source_file,
            )
        else:
            _maybe_add_qa_row(
                out,
                dataset=obj.get("dataset"),
                system=obj.get("system"),
                variant=obj.get("variant"),
                em=obj.get("EM", obj.get("em")),
                f1=obj.get("F1", obj.get("f1")),
                source_file=source_file,
            )
    elif isinstance(obj, list):
        for rec in obj:
            if not isinstance(rec, dict):
                continue
            _maybe_add_qa_row(
                out,
                dataset=rec.get("dataset"),
                system=rec.get("system", rec.get("true_system")),
                variant=rec.get("variant", rec.get("true_variant")),
                em=rec.get("EM", rec.get("em")),
                f1=rec.get("F1", rec.get("f1")),
                source_file=source_file,
            )
    return out


def _load_qa_metrics(paths: Sequence[Path]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    warnings: List[str] = []
    loaded_files: List[str] = []
    duplicate_keys: List[str] = []
    qa_rows_by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_exact_keys: set[Tuple[str, str, str]] = set()

    for fp in _iter_qa_metric_files(paths):
        try:
            if fp.suffix == ".jsonl":
                obj = _read_jsonl(fp)
            else:
                obj = _read_json(fp)
        except Exception as exc:
            warnings.append(f"failed to parse QA metrics file {fp}: {type(exc).__name__}: {exc}")
            continue
        rows = _extract_qa_rows_from_object(obj, fp)
        if not rows:
            continue
        loaded_files.append(str(fp))
        for row in rows:
            dataset = _strip(row.get("dataset"))
            system = _strip(row.get("system"))
            variant = _strip(row.get("variant"))
            exact_key = (dataset, system, variant)
            if exact_key in seen_exact_keys:
                duplicate_keys.append("|".join(exact_key))
            seen_exact_keys.add(exact_key)
            qa_rows_by_dataset[dataset].append(row)

    report = {
        "loaded_files": loaded_files,
        "warnings": warnings,
        "duplicate_exact_keys": sorted(set(duplicate_keys)),
        "row_count": int(sum(len(v) for v in qa_rows_by_dataset.values())),
    }
    return dict(qa_rows_by_dataset), report


def _resolve_qa_metric_row(
    qa_rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    dataset: str,
    system: str,
    variant: str,
) -> Optional[Dict[str, Any]]:
    rows = list(qa_rows_by_dataset.get(dataset, []))
    if not rows:
        return None
    system_aliases = set(_normalize_system_aliases(system))
    variant_aliases = set(_normalize_variant_aliases(variant))

    best: Optional[Dict[str, Any]] = None
    best_score = -1
    for row in rows:
        row_system = _strip(row.get("system"))
        row_variant = _strip(row.get("variant"))
        row_system_aliases = set(_normalize_system_aliases(row_system))
        row_variant_aliases = set(_normalize_variant_aliases(row_variant))
        system_match = bool(system_aliases.intersection(row_system_aliases)) if row_system else True
        variant_match = bool(variant_aliases.intersection(row_variant_aliases)) if row_variant else False
        if not variant_match:
            continue
        score = 0
        if row_system and _strip(row_system) == system:
            score += 100
        elif row_system and system_match:
            score += 80
        elif not row_system:
            score += 40
        if row_variant and _strip(row_variant) == variant:
            score += 100
        elif row_variant and variant_match:
            score += 80
        if score > best_score:
            best_score = score
            best = dict(row)
    return best


def _prepare_record_rows(
    input_rows: Sequence[Mapping[str, Any]],
    output_rows_by_id: Mapping[str, Mapping[str, Any]],
    mapping_by_id: Mapping[str, Mapping[str, Any]],
    input_run_rows: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    token_counter: TokenCounter,
    duplicate_output_policy: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    record_rows: List[Dict[str, Any]] = []
    validation: Dict[str, Any] = {
        "missing_output_judge_ids": [],
        "invalid_output_score_judge_ids": [],
        "invalid_minimal_evidence_id_records": [],
        "empty_evidence_judge_ids": [],
        "zero_token_judge_ids": [],
        "sufficient_but_empty_minimal_evidence_ids": [],
        "score_lt_2_but_nonempty_minimal_evidence_ids": [],
        "mapping_missing_judge_ids": [],
        "unknown_output_judge_ids": [],
    }

    input_ids = {_strip(r.get("judge_id")) for r in input_rows if _strip(r.get("judge_id"))}
    for output_jid in sorted(set(output_rows_by_id) - input_ids):
        validation["unknown_output_judge_ids"].append(output_jid)

    source_meta_by_group: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, row in input_run_rows.items():
        dataset, source, variant = key
        source_meta_by_group[(dataset, source, variant)] = dict(row)

    for row in input_rows:
        judge_id = _strip(row.get("judge_id"))
        dataset = _strip(row.get("dataset"))
        system_id = _strip(row.get("system_id"))
        qid = _strip(row.get("qid"))
        question = _strip(row.get("question"))
        gold_answer = _strip(row.get("gold_answer"))
        evidence = list(row.get("evidence", []) or [])

        item_token_counts: List[int] = []
        token_count_by_evidence_id: Dict[int, int] = {}
        valid_evidence_ids: List[int] = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            evidence_id = _safe_int(item.get("id"), default=-1)
            if evidence_id < 0:
                continue
            tok = token_counter.count(_render_evidence_item_text(item))
            token_count_by_evidence_id[evidence_id] = tok
            item_token_counts.append(tok)
            valid_evidence_ids.append(evidence_id)

        total_context_tokens = int(sum(item_token_counts))
        n_evidence_items = int(len(token_count_by_evidence_id))
        if n_evidence_items <= 0:
            validation["empty_evidence_judge_ids"].append(judge_id)
        if total_context_tokens <= 0:
            validation["zero_token_judge_ids"].append(judge_id)

        out = output_rows_by_id.get(judge_id)
        output_missing = out is None
        if output_missing:
            validation["missing_output_judge_ids"].append(judge_id)
            out = {}

        score_raw = out.get("score")
        score: Optional[int]
        if score_raw is None or _strip(score_raw) == "":
            score = None
        else:
            try:
                score = int(score_raw)
            except Exception:
                score = None
        if score is None or score not in {0, 1, 2, 3}:
            if not output_missing:
                validation["invalid_output_score_judge_ids"].append(judge_id)

        sufficient = bool(score is not None and score >= 2)
        minimal_evidence_ids_raw = list(out.get("minimal_evidence_ids", []) or [])
        valid_selected_ids: List[int] = []
        invalid_selected_ids: List[Any] = []
        for value in minimal_evidence_ids_raw:
            if isinstance(value, int) and value in token_count_by_evidence_id:
                valid_selected_ids.append(value)
            else:
                invalid_selected_ids.append(value)
        if invalid_selected_ids:
            validation["invalid_minimal_evidence_id_records"].append(
                {
                    "judge_id": judge_id,
                    "invalid_ids": invalid_selected_ids,
                    "output_file": _strip(out.get("_output_file")),
                }
            )

        score_lt_2_nonempty = bool((score is None or score < 2) and len(minimal_evidence_ids_raw) > 0)
        sufficient_but_empty = bool(sufficient and len(valid_selected_ids) <= 0)
        if score_lt_2_nonempty:
            validation["score_lt_2_but_nonempty_minimal_evidence_ids"].append(judge_id)
        if sufficient_but_empty:
            validation["sufficient_but_empty_minimal_evidence_ids"].append(judge_id)

        selected_minimal_tokens = (
            int(sum(token_count_by_evidence_id[x] for x in valid_selected_ids))
            if sufficient and valid_selected_ids
            else 0
        )
        minimal_evidence_density = (
            float(selected_minimal_tokens / total_context_tokens)
            if sufficient and selected_minimal_tokens > 0 and total_context_tokens > 0
            else None
        )
        redundancy_ratio = (
            float(total_context_tokens / selected_minimal_tokens)
            if sufficient and selected_minimal_tokens > 0
            else None
        )

        mapping = mapping_by_id.get(judge_id)
        true_system = _strip(mapping.get("true_system")) if mapping else ""
        true_variant = _strip(mapping.get("true_variant")) if mapping else ""
        if mapping is None and mapping_by_id:
            validation["mapping_missing_judge_ids"].append(judge_id)

        source_meta = source_meta_by_group.get((dataset, true_system, true_variant))
        avg_tokens_per_item_record = float(total_context_tokens / n_evidence_items) if n_evidence_items > 0 else 0.0

        record_rows.append(
            {
                "judge_id": judge_id,
                "dataset": dataset,
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "system_id": system_id,
                "true_system": true_system or None,
                "true_variant": true_variant or None,
                "group_system": true_system or system_id,
                "group_variant": true_variant or "",
                "source_path": _strip((source_meta or {}).get("source_path")) or None,
                "input_file": _strip(row.get("_input_file")),
                "output_file": _strip(out.get("_output_file")) or None,
                "output_root": _strip(out.get("_output_root")) or None,
                "output_missing": output_missing,
                "score": score,
                "sufficient": sufficient,
                "answer_surface_present": _as_bool(out.get("answer_surface_present")),
                "equivalent_evidence": _as_bool(out.get("equivalent_evidence")),
                "judge_error": bool(_strip(out.get("error"))),
                "error": _strip(out.get("error")) or None,
                "n_evidence_items": n_evidence_items,
                "total_context_tokens": total_context_tokens,
                "avg_tokens_per_item_record": avg_tokens_per_item_record,
                "evidence_item_token_counts": item_token_counts,
                "minimal_evidence_ids": list(valid_selected_ids),
                "minimal_evidence_ids_raw": minimal_evidence_ids_raw,
                "invalid_minimal_evidence_ids": invalid_selected_ids,
                "selected_minimal_evidence_tokens": selected_minimal_tokens,
                "minimal_evidence_density": minimal_evidence_density,
                "redundancy_ratio": redundancy_ratio,
                "score_lt_2_but_nonempty_minimal_evidence_ids": score_lt_2_nonempty,
                "sufficient_but_empty_minimal_evidence_ids": sufficient_but_empty,
                "empty_evidence": n_evidence_items <= 0,
                "zero_token_context": total_context_tokens <= 0,
                "mapping_joined": bool(mapping),
                "valid_evidence_ids": valid_evidence_ids,
            }
        )

    validation["counts"] = {
        "input_record_count": len(input_rows),
        "matched_output_record_count": int(len(input_rows) - len(validation["missing_output_judge_ids"])),
        "missing_output_count": len(validation["missing_output_judge_ids"]),
        "invalid_output_score_count": len(validation["invalid_output_score_judge_ids"]),
        "invalid_minimal_evidence_id_record_count": len(validation["invalid_minimal_evidence_id_records"]),
        "empty_evidence_record_count": len(validation["empty_evidence_judge_ids"]),
        "zero_token_record_count": len(validation["zero_token_judge_ids"]),
        "sufficient_but_empty_minimal_evidence_count": len(validation["sufficient_but_empty_minimal_evidence_ids"]),
        "score_lt_2_but_nonempty_minimal_evidence_count": len(validation["score_lt_2_but_nonempty_minimal_evidence_ids"]),
        "mapping_missing_count": len(validation["mapping_missing_judge_ids"]),
        "unknown_output_count": len(validation["unknown_output_judge_ids"]),
    }
    return record_rows, validation


def _compute_group_rows(
    record_rows: Sequence[Mapping[str, Any]],
    *,
    group_mode: str,
    qa_rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in record_rows:
        dataset = _strip(row.get("dataset"))
        if group_mode == "unblinded":
            system = _strip(row.get("true_system"))
            variant = _strip(row.get("true_variant"))
            if not system:
                continue
        else:
            system = _strip(row.get("system_id"))
            variant = ""
        groups[(dataset, system, variant)].append(row)

    summary_rows: List[Dict[str, Any]] = []
    qa_joined_groups = 0
    qa_total_groups = 0
    qa_joined_records = 0

    for (dataset, system, variant), rows in sorted(groups.items()):
        qa_total_groups += 1
        n = len(rows)
        evidence_items = [float(_safe_int(r.get("n_evidence_items"))) for r in rows]
        total_context_tokens = [float(_safe_int(r.get("total_context_tokens"))) for r in rows]
        all_item_token_counts: List[float] = []
        sufficient_rows = [r for r in rows if _as_bool(r.get("sufficient"))]
        selected_min_tokens = [
            float(_safe_int(r.get("selected_minimal_evidence_tokens")))
            for r in sufficient_rows
            if _safe_int(r.get("selected_minimal_evidence_tokens")) > 0
        ]
        densities = [
            float(_safe_float(r.get("minimal_evidence_density")))
            for r in sufficient_rows
            if r.get("minimal_evidence_density") is not None
        ]
        redundancy = [
            float(_safe_float(r.get("redundancy_ratio")))
            for r in sufficient_rows
            if r.get("redundancy_ratio") is not None
        ]
        scores = [
            float(_safe_int(r.get("score")))
            for r in rows
            if r.get("score") is not None and _safe_int(r.get("score")) in {0, 1, 2, 3}
        ]
        sufficient_rate = _mean([1.0 if _as_bool(r.get("sufficient")) else 0.0 for r in rows])
        answer_surface_rate = _mean([1.0 if _as_bool(r.get("answer_surface_present")) else 0.0 for r in rows])
        equivalent_rate = _mean([1.0 if _as_bool(r.get("equivalent_evidence")) else 0.0 for r in rows])
        judge_error_rate = _mean([1.0 if _as_bool(r.get("judge_error")) else 0.0 for r in rows])

        for r in rows:
            token_counts = list(r.get("evidence_item_token_counts", []) or [])
            if token_counts:
                all_item_token_counts.extend(float(_safe_int(tok)) for tok in token_counts)

        avg_total_context_tokens = _mean(total_context_tokens)
        qa_row = _resolve_qa_metric_row(
            qa_rows_by_dataset,
            dataset=dataset,
            system=system,
            variant=variant,
        )
        em = qa_row.get("EM") if qa_row else None
        f1 = qa_row.get("F1") if qa_row else None
        if qa_row is not None:
            qa_joined_groups += 1
            qa_joined_records += n

        def _per_token(metric_value: Optional[float], factor: float) -> Optional[float]:
            if metric_value is None or avg_total_context_tokens <= 0:
                return None
            return float(metric_value / avg_total_context_tokens * factor)

        summary_rows.append(
            {
                "group_mode": group_mode,
                "dataset": dataset,
                "system": system,
                "variant": variant,
                "n": n,
                "avg_evidence_items": _mean(evidence_items),
                "median_evidence_items": _median(evidence_items),
                "avg_total_context_tokens": avg_total_context_tokens,
                "median_total_context_tokens": _median(total_context_tokens),
                "p90_total_context_tokens": _percentile(total_context_tokens, 0.90),
                "avg_tokens_per_item": _mean(all_item_token_counts),
                "p90_tokens_per_item": _percentile(all_item_token_counts, 0.90),
                "avg_score": _mean(scores),
                "llm_sufficient_rate": sufficient_rate,
                "answer_surface_rate": answer_surface_rate,
                "equivalent_evidence_rate": equivalent_rate,
                "avg_selected_minimal_tokens_sufficient_only": _mean(selected_min_tokens),
                "median_selected_minimal_tokens_sufficient_only": _median(selected_min_tokens),
                "minimal_evidence_density_sufficient_only": _mean(densities),
                "redundancy_ratio_sufficient_only": _mean(redundancy),
                "sufficiency_per_100_tokens": _per_token(sufficient_rate, 100.0),
                "sufficiency_per_1k_tokens": _per_token(sufficient_rate, 1000.0),
                "answer_surface_per_100_tokens": _per_token(answer_surface_rate, 100.0),
                "answer_surface_per_1k_tokens": _per_token(answer_surface_rate, 1000.0),
                "EM": em,
                "F1": f1,
                "EM_per_100_tokens": _per_token(em, 100.0),
                "F1_per_100_tokens": _per_token(f1, 100.0),
                "EM_per_1k_tokens": _per_token(em, 1000.0),
                "F1_per_1k_tokens": _per_token(f1, 1000.0),
                "judge_error_rate": judge_error_rate,
                "qa_metrics_source_file": qa_row.get("source_file") if qa_row else None,
            }
        )

    qa_report = {
        "qa_group_joined_count": qa_joined_groups,
        "qa_group_total_count": qa_total_groups,
        "qa_group_join_success_rate": float(qa_joined_groups / qa_total_groups) if qa_total_groups else 0.0,
        "qa_record_joined_count": qa_joined_records,
        "qa_record_total_count": int(len(record_rows)),
        "qa_record_join_success_rate": float(qa_joined_records / len(record_rows)) if record_rows else 0.0,
    }
    return summary_rows, qa_report


def _build_summary_payload(
    *,
    record_rows: Sequence[Mapping[str, Any]],
    blind_rows: Sequence[Mapping[str, Any]],
    unblinded_rows: Sequence[Mapping[str, Any]],
    primary_mode: str,
    qa_report: Mapping[str, Any],
) -> Dict[str, Any]:
    primary_rows = list(unblinded_rows if primary_mode == "unblinded" else blind_rows)
    return {
        "generated_at_utc": _utc_now_iso(),
        "primary_group_mode": primary_mode,
        "primary_grouping": ["dataset", "system", "variant"] if primary_mode == "unblinded" else ["dataset", "system"],
        "record_count": len(record_rows),
        "primary_rows": primary_rows,
        "blind_rows": list(blind_rows),
        "unblinded_rows": list(unblinded_rows),
        "qa_join": dict(qa_report),
    }


def _write_summary_tables(
    summary_dir: Path,
    payload: Mapping[str, Any],
) -> None:
    primary_rows = list(payload.get("primary_rows", []) or [])
    context_cols = [
        "dataset",
        "system",
        "variant",
        "n",
        "avg_evidence_items",
        "median_evidence_items",
        "avg_total_context_tokens",
        "median_total_context_tokens",
        "p90_total_context_tokens",
        "avg_tokens_per_item",
        "p90_tokens_per_item",
    ]
    coverage_cols = [
        "dataset",
        "system",
        "variant",
        "llm_sufficient_rate",
        "avg_total_context_tokens",
        "sufficiency_per_100_tokens",
        "sufficiency_per_1k_tokens",
        "answer_surface_rate",
        "answer_surface_per_100_tokens",
        "answer_surface_per_1k_tokens",
        "equivalent_evidence_rate",
    ]
    density_cols = [
        "dataset",
        "system",
        "variant",
        "avg_selected_minimal_tokens_sufficient_only",
        "median_selected_minimal_tokens_sufficient_only",
        "minimal_evidence_density_sufficient_only",
        "redundancy_ratio_sufficient_only",
    ]
    redundancy_cols = [
        "dataset",
        "system",
        "variant",
        "avg_total_context_tokens",
        "avg_selected_minimal_tokens_sufficient_only",
        "minimal_evidence_density_sufficient_only",
        "redundancy_ratio_sufficient_only",
    ]
    qa_cols = [
        "dataset",
        "system",
        "variant",
        "EM",
        "F1",
        "avg_total_context_tokens",
        "EM_per_100_tokens",
        "F1_per_100_tokens",
        "EM_per_1k_tokens",
        "F1_per_1k_tokens",
    ]
    final_cols = [
        "dataset",
        "system",
        "variant",
        "llm_sufficient_rate",
        "avg_total_context_tokens",
        "sufficiency_per_100_tokens",
        "minimal_evidence_density_sufficient_only",
        "redundancy_ratio_sufficient_only",
        "F1",
        "F1_per_100_tokens",
    ]

    _write_json(summary_dir / "context_quantity_summary.json", payload)
    _write_json(summary_dir / "coverage_vs_efficiency_summary.json", payload)
    (summary_dir / "context_quantity_table.md").write_text(
        "# Context Quantity Table\n\n" + _render_markdown_table(primary_rows, context_cols),
        encoding="utf-8",
    )
    (summary_dir / "coverage_vs_efficiency_table.md").write_text(
        "# Coverage vs Efficiency Table\n\n" + _render_markdown_table(primary_rows, coverage_cols),
        encoding="utf-8",
    )
    (summary_dir / "token_density_table.md").write_text(
        "# Token Density Table\n\n" + _render_markdown_table(primary_rows, density_cols),
        encoding="utf-8",
    )
    (summary_dir / "redundancy_table.md").write_text(
        "# Redundancy Table\n\n" + _render_markdown_table(primary_rows, redundancy_cols),
        encoding="utf-8",
    )
    (summary_dir / "qa_utility_per_token_table.md").write_text(
        "# QA Utility per Token Table\n\n" + _render_markdown_table(primary_rows, qa_cols),
        encoding="utf-8",
    )
    (summary_dir / "final_analysis_table.md").write_text(
        "# Final Analysis Table\n\n" + _render_markdown_table(primary_rows, final_cols),
        encoding="utf-8",
    )


def _build_validation_payload(
    *,
    record_rows: Sequence[Mapping[str, Any]],
    input_duplicate_ids: Sequence[str],
    output_report: Mapping[str, Any],
    record_validation: Mapping[str, Any],
    tokenizer: TokenCounter,
    mapping_used: bool,
    qa_report: Mapping[str, Any],
    qa_metric_report: Mapping[str, Any],
    duplicate_output_policy: str,
) -> Dict[str, Any]:
    counts = dict(record_validation.get("counts", {}) or {})
    input_record_count = int(counts.get("input_record_count", 0))
    matched_output_record_count = int(counts.get("matched_output_record_count", 0))

    mapping_joined = sum(1 for r in record_rows if _as_bool(r.get("mapping_joined")))
    qa_group_join_rate = _safe_float(qa_report.get("qa_group_join_success_rate"), 0.0)
    qa_record_join_rate = _safe_float(qa_report.get("qa_record_join_success_rate"), 0.0)
    mapping_join_rate = float(mapping_joined / len(record_rows)) if record_rows else 0.0

    pass_errors = []
    if input_duplicate_ids:
        pass_errors.append("duplicate_input_judge_ids")
    if output_report.get("parse_errors"):
        pass_errors.append("output_parse_errors")
    if output_report.get("duplicate_output_judge_ids") and duplicate_output_policy != "prefer-last":
        pass_errors.append("duplicate_output_judge_ids")
    if record_validation.get("missing_output_judge_ids"):
        pass_errors.append("missing_output_judge_ids")
    if record_validation.get("invalid_output_score_judge_ids"):
        pass_errors.append("invalid_output_score_judge_ids")
    if record_validation.get("invalid_minimal_evidence_id_records"):
        pass_errors.append("invalid_minimal_evidence_ids")
    if record_validation.get("sufficient_but_empty_minimal_evidence_ids"):
        pass_errors.append("sufficient_but_empty_minimal_evidence_ids")

    warnings: List[str] = []
    if tokenizer.warning:
        warnings.append(tokenizer.warning)
    warnings.extend(list(qa_metric_report.get("warnings", []) or []))
    if output_report.get("duplicate_output_judge_ids") and duplicate_output_policy == "prefer-last":
        warnings.append(
            "duplicate output judge IDs were resolved with prefer-last policy: "
            + str(len(output_report.get("duplicate_output_judge_ids", [])))
        )
    if record_validation.get("score_lt_2_but_nonempty_minimal_evidence_ids"):
        warnings.append(
            "score<2 but non-empty minimal_evidence_ids: "
            + str(len(record_validation.get("score_lt_2_but_nonempty_minimal_evidence_ids", [])))
        )
    if record_validation.get("empty_evidence_judge_ids"):
        warnings.append(
            "empty evidence records: "
            + str(len(record_validation.get("empty_evidence_judge_ids", [])))
        )
    if record_validation.get("zero_token_judge_ids"):
        warnings.append(
            "zero token records: " + str(len(record_validation.get("zero_token_judge_ids", [])))
        )
    if mapping_used and mapping_join_rate < 1.0:
        warnings.append(f"mapping join success rate below 1.0: {mapping_join_rate:.6f}")
    if qa_metric_report.get("row_count", 0) > 0 and qa_group_join_rate < 1.0:
        warnings.append(f"QA group join success rate below 1.0: {qa_group_join_rate:.6f}")

    checks = [
        {
            "name": "input_output_judge_id_join_success_rate",
            "pass": matched_output_record_count == input_record_count,
            "details": f"matched={matched_output_record_count}, input={input_record_count}",
        },
        {
            "name": "missing_output_judge_ids",
            "pass": len(record_validation.get("missing_output_judge_ids", [])) == 0,
            "details": f"missing={len(record_validation.get('missing_output_judge_ids', []))}",
        },
        {
            "name": "duplicate_output_judge_ids",
            "pass": (len(output_report.get("duplicate_output_judge_ids", [])) == 0) or duplicate_output_policy == "prefer-last",
            "details": (
                f"duplicates={len(output_report.get('duplicate_output_judge_ids', []))}, "
                f"policy={duplicate_output_policy}"
            ),
        },
        {
            "name": "invalid_evidence_id_in_minimal_evidence_ids",
            "pass": len(record_validation.get("invalid_minimal_evidence_id_records", [])) == 0,
            "details": f"violations={len(record_validation.get('invalid_minimal_evidence_id_records', []))}",
        },
        {
            "name": "empty_evidence_records",
            "pass": len(record_validation.get("empty_evidence_judge_ids", [])) == 0,
            "details": f"count={len(record_validation.get('empty_evidence_judge_ids', []))}",
        },
        {
            "name": "zero_token_records",
            "pass": len(record_validation.get("zero_token_judge_ids", [])) == 0,
            "details": f"count={len(record_validation.get('zero_token_judge_ids', []))}",
        },
        {
            "name": "sufficient_true_but_empty_minimal_evidence_ids",
            "pass": len(record_validation.get("sufficient_but_empty_minimal_evidence_ids", [])) == 0,
            "details": f"count={len(record_validation.get('sufficient_but_empty_minimal_evidence_ids', []))}",
        },
        {
            "name": "score_lt_2_but_nonempty_minimal_evidence_ids",
            "pass": True,
            "details": f"count={len(record_validation.get('score_lt_2_but_nonempty_minimal_evidence_ids', []))}",
        },
        {
            "name": "tokenizer_fallback_used",
            "pass": tokenizer.mode == "hf_tokenizer",
            "details": tokenizer.mode,
        },
        {
            "name": "mapping_join_success_rate",
            "pass": (not mapping_used) or math.isclose(mapping_join_rate, 1.0),
            "details": f"{mapping_join_rate:.6f}",
        },
        {
            "name": "qa_metric_join_success_rate",
            "pass": (qa_metric_report.get("row_count", 0) == 0) or math.isclose(qa_group_join_rate, 1.0),
            "details": f"group={qa_group_join_rate:.6f}, record={qa_record_join_rate:.6f}",
        },
    ]

    return {
        "generated_at_utc": _utc_now_iso(),
        "pass": len(pass_errors) == 0,
        "errors": pass_errors,
        "warnings": warnings,
        "tokenizer_mode": tokenizer.mode,
        "tokenizer_fallback_used": tokenizer.mode != "hf_tokenizer",
        "checks": checks,
        "counts": {
            **counts,
            "input_output_join_success_rate": float(matched_output_record_count / input_record_count) if input_record_count else 0.0,
            "mapping_joined_record_count": mapping_joined,
            "mapping_join_success_rate": mapping_join_rate,
            "qa_group_join_success_rate": qa_group_join_rate,
            "qa_record_join_success_rate": qa_record_join_rate,
            "input_duplicate_count": len(input_duplicate_ids),
            "output_duplicate_count": len(output_report.get("duplicate_output_judge_ids", [])),
        },
        "input_duplicate_judge_ids": list(input_duplicate_ids),
        "missing_output_judge_ids": list(record_validation.get("missing_output_judge_ids", [])),
        "duplicate_output_judge_ids": list(output_report.get("duplicate_output_judge_ids", [])),
        "invalid_output_score_judge_ids": list(record_validation.get("invalid_output_score_judge_ids", [])),
        "invalid_minimal_evidence_id_records": list(record_validation.get("invalid_minimal_evidence_id_records", [])),
        "sufficient_but_empty_minimal_evidence_ids": list(record_validation.get("sufficient_but_empty_minimal_evidence_ids", [])),
        "score_lt_2_but_nonempty_minimal_evidence_ids": list(record_validation.get("score_lt_2_but_nonempty_minimal_evidence_ids", [])),
        "empty_evidence_judge_ids": list(record_validation.get("empty_evidence_judge_ids", [])),
        "zero_token_judge_ids": list(record_validation.get("zero_token_judge_ids", [])),
        "unknown_output_judge_ids": list(record_validation.get("unknown_output_judge_ids", [])),
        "output_parse_errors": list(output_report.get("parse_errors", [])),
        "qa_metric_report": dict(qa_metric_report),
    }


def _write_validation_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Evidence Efficiency Validation Report",
        "",
        f"- generated_at_utc: {_strip(payload.get('generated_at_utc'))}",
        f"- pass: {payload.get('pass')}",
        f"- tokenizer_mode: {_strip(payload.get('tokenizer_mode'))}",
        f"- tokenizer_fallback_used: {payload.get('tokenizer_fallback_used')}",
        "",
        "## Counts",
        "```json",
        json.dumps(dict(payload.get("counts", {}) or {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Checks",
        "| name | pass | details |",
        "| --- | --- | --- |",
    ]
    for check in payload.get("checks", []) or []:
        lines.append(f"| {_strip(check.get('name'))} | {check.get('pass')} | {_strip(check.get('details'))} |")
    lines.extend(["", "## Warnings"])
    warnings = list(payload.get("warnings", []) or [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.extend(["", "## Errors"])
    errors = list(payload.get("errors", []) or [])
    if errors:
        for err in errors:
            lines.append(f"- {err}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_readme(
    output_dir: Path,
    manifest: Mapping[str, Any],
    primary_rows: Sequence[Mapping[str, Any]],
    validation_payload: Mapping[str, Any],
) -> None:
    best_coverage = None
    best_efficiency = None
    if primary_rows:
        best_coverage = max(primary_rows, key=lambda r: _safe_float(r.get("llm_sufficient_rate"), 0.0))
        best_efficiency = max(primary_rows, key=lambda r: _safe_float(r.get("sufficiency_per_100_tokens"), 0.0))

    lines = [
        "# Token-Normalized Evidence Efficiency Analysis",
        "",
        f"- generated_at_utc: {_strip(manifest.get('created_at'))}",
        f"- judge input root: {_strip(manifest.get('judge_input_root'))}",
        f"- judge output roots: {', '.join(list(manifest.get('judge_output_roots', []) or []))}",
        f"- tokenizer: {_strip(manifest.get('tokenizer_name'))}",
        f"- tokenizer_mode: {_strip(manifest.get('tokenizer_mode'))}",
        f"- mapping used: {bool(manifest.get('mapping_used'))}",
        f"- QA metrics joined: {bool(manifest.get('qa_metrics_joined'))}",
        f"- score interpretation: {_strip(manifest.get('score_interpretation'))}",
        "",
        "## Context Quantity Summary",
    ]
    if best_coverage is not None:
        lines.extend(
            [
                f"- highest coverage group: {best_coverage.get('dataset')} / {best_coverage.get('system')} / {best_coverage.get('variant')}",
                f"  suff_rate={_safe_float(best_coverage.get('llm_sufficient_rate')):.4f}, "
                f"avg_ctx_tok={_safe_float(best_coverage.get('avg_total_context_tokens')):.2f}",
                f"- highest sufficiency-per-token group: {best_efficiency.get('dataset')} / {best_efficiency.get('system')} / {best_efficiency.get('variant')}",
                f"  suff/100tok={_safe_float(best_efficiency.get('sufficiency_per_100_tokens')):.6f}, "
                f"density={_safe_float(best_efficiency.get('minimal_evidence_density_sufficient_only')):.4f}, "
                f"redundancy={_safe_float(best_efficiency.get('redundancy_ratio_sufficient_only')):.4f}",
            ]
        )
    else:
        lines.append("- no primary rows available")
    lines.extend(
        [
            "",
            "## Notes",
            "- The LLM judge is interpreted as a binary sufficiency audit with `score >= 2` treated as sufficient.",
            "- Compactness is not inferred from judge score; it is computed deterministically from token accounting over judge-selected minimal evidence IDs.",
            "- Minimal evidence count alone is not treated as a fair compactness metric because evidence item granularity differs across systems.",
            "- Coverage-oriented metrics can favor broader passage retrieval, so this analysis complements coverage with token-normalized efficiency.",
            "",
            "## Validation",
            f"- validation_pass: {validation_payload.get('pass')}",
            f"- tokenizer_fallback_used: {validation_payload.get('tokenizer_fallback_used')}",
            f"- mapping_join_success_rate: {_safe_float((validation_payload.get('counts') or {}).get('mapping_join_success_rate')):.6f}",
            f"- qa_metric_join_success_rate: {_safe_float((validation_payload.get('counts') or {}).get('qa_group_join_success_rate')):.6f}",
            "",
            "## Next Step",
            "- If large context-length gaps remain while coverage still favors broader retrieval, run a same-token-budget rejudge next.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute token-normalized evidence efficiency metrics from canonical judge input/output artifacts."
    )
    ap.add_argument("--judge-input-root", required=True)
    ap.add_argument("--judge-output-root", required=True, action="append")
    ap.add_argument("--mapping-root", default="")
    ap.add_argument("--qa-metrics-path", action="append", default=[])
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--duplicate-output-policy", choices=["error", "prefer-last"], default="error")
    args = ap.parse_args()

    input_root = _resolve_input_root(Path(args.judge_input_root))
    output_roots = [_resolve_output_root(Path(p)) for p in list(args.judge_output_root or [])]
    mapping_root = Path(args.mapping_root).resolve() if _strip(args.mapping_root) else None
    qa_metric_paths = [Path(p).resolve() for p in list(args.qa_metrics_path or [])]
    base_output_root = Path(args.output_root).resolve()
    analysis_dir = base_output_root / _timestamp_slug()
    record_dir = analysis_dir / "record_level"
    summary_dir = analysis_dir / "summary"
    validation_dir = analysis_dir / "validation"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    input_rows, input_duplicate_ids = _load_input_records(input_root, max_records=int(args.max_records or 0))
    output_rows_by_id, output_report = _load_output_rows(output_roots, duplicate_policy=args.duplicate_output_policy)
    mapping_by_id, mapping_warnings = _load_mapping(mapping_root)
    input_round_root = _infer_round_root_from_input(input_root)
    input_run_rows = _load_input_run_records(input_round_root)

    token_counter = TokenCounter(_strip(args.tokenizer_name))
    qa_rows_by_dataset, qa_metric_report = _load_qa_metrics(qa_metric_paths)

    record_rows, record_validation = _prepare_record_rows(
        input_rows=input_rows,
        output_rows_by_id=output_rows_by_id,
        mapping_by_id=mapping_by_id,
        input_run_rows=input_run_rows,
        token_counter=token_counter,
        duplicate_output_policy=args.duplicate_output_policy,
    )
    _write_jsonl(record_dir / "record_level_efficiency.jsonl", record_rows)

    blind_rows, blind_qa_report = _compute_group_rows(
        record_rows,
        group_mode="blind",
        qa_rows_by_dataset={},
    )
    unblinded_rows, unblinded_qa_report = _compute_group_rows(
        record_rows,
        group_mode="unblinded",
        qa_rows_by_dataset=qa_rows_by_dataset,
    )
    primary_mode = "unblinded" if unblinded_rows else "blind"
    primary_qa_report = unblinded_qa_report if primary_mode == "unblinded" else blind_qa_report

    summary_payload = _build_summary_payload(
        record_rows=record_rows,
        blind_rows=blind_rows,
        unblinded_rows=unblinded_rows,
        primary_mode=primary_mode,
        qa_report=primary_qa_report,
    )
    _write_summary_tables(summary_dir, summary_payload)

    validation_payload = _build_validation_payload(
        record_rows=record_rows,
        input_duplicate_ids=input_duplicate_ids,
        output_report=output_report,
        record_validation=record_validation,
        tokenizer=token_counter,
        mapping_used=bool(mapping_by_id),
        qa_report=primary_qa_report,
        qa_metric_report={
            **qa_metric_report,
            "warnings": list(qa_metric_report.get("warnings", [])) + mapping_warnings,
        },
        duplicate_output_policy=args.duplicate_output_policy,
    )
    _write_json(validation_dir / "efficiency_validation_report.json", validation_payload)
    _write_validation_md(validation_dir / "efficiency_validation_report.md", validation_payload)

    manifest = {
        "analysis_name": "token_normalized_evidence_efficiency",
        "judge_input_root": str(input_root),
        "judge_output_roots": [str(p) for p in output_roots],
        "mapping_root": str(mapping_root) if mapping_root else None,
        "qa_metrics_paths": [str(p) for p in qa_metric_paths],
        "tokenizer_name": _strip(args.tokenizer_name),
        "tokenizer_mode": token_counter.mode,
        "score_interpretation": "binary_sufficiency_score_ge_2",
        "compactness_source": "token_accounting_over_judge_selected_minimal_evidence_ids",
        "duplicate_output_policy": args.duplicate_output_policy,
        "mapping_used": bool(mapping_by_id),
        "qa_metrics_joined": bool(_safe_float(primary_qa_report.get("qa_group_join_success_rate"), 0.0) > 0.0),
        "max_records": int(args.max_records or 0),
        "created_at": _utc_now_iso(),
        "counts": {
            "input_records": len(input_rows),
            "record_level_rows": len(record_rows),
            "blind_group_rows": len(blind_rows),
            "unblinded_group_rows": len(unblinded_rows),
        },
        "warnings": [w for w in [token_counter.warning, *mapping_warnings, *list(qa_metric_report.get("warnings", []))] if w],
    }
    _write_json(analysis_dir / "manifest.json", manifest)
    _build_readme(analysis_dir, manifest, summary_payload.get("primary_rows", []), validation_payload)

    print(
        json.dumps(
            {
                "output_dir": str(analysis_dir),
                "validation_pass": validation_payload.get("pass"),
                "tokenizer_mode": token_counter.mode,
                "primary_group_mode": primary_mode,
                "record_count": len(record_rows),
                "group_count": len(summary_payload.get("primary_rows", [])),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
