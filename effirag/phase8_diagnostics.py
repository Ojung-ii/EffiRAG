from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


TRACE_FILES = [
    "rag_summary.json",
    "rag_query_results.jsonl",
    "phase7_query_trace.jsonl",
    "phase7_diagnostics.jsonl",
    "phase8_query_trace.jsonl",
    "phase8_seed_trace.jsonl",
    "phase8_evidence_trace.jsonl",
    "phase8_stage_timing.jsonl",
]


def normalize_title(x: str) -> str:
    text = html.unescape(str(x or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def canonical_support_key(title: str, sent_idx: int | str) -> str:
    idx = str(sent_idx).strip()
    return f"{normalize_title(title)}::{idx}"


def normalize_evidence_id(x: str) -> str:
    text = html.unescape(str(x or ""))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    if "::" in text and not text.startswith(("s::", "c::", "e::")):
        title, idx = text.rsplit("::", 1)
        return canonical_support_key(title, idx)
    return text.casefold()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(vals: Iterable[Any]) -> float:
    xs = [safe_float(v) for v in vals]
    return float(sum(xs) / len(xs)) if xs else 0.0


def percentile(vals: Iterable[Any], q: float) -> float:
    xs = sorted(safe_float(v) for v in vals)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * float(q)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def stats(vals: Iterable[Any]) -> Dict[str, float]:
    xs = [safe_float(v) for v in vals]
    return {
        "mean": mean(xs),
        "p50": float(median(xs)) if xs else 0.0,
        "p95": percentile(xs, 0.95),
        "max": max(xs) if xs else 0.0,
    }


def md_table(rows: List[Mapping[str, Any]], keys: List[str]) -> str:
    out = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for row in rows:
        vals = []
        for key in keys:
            val = row.get(key, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def query_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("query_id")
        or row.get("qid")
        or row.get("sample_id")
        or row.get("id")
        or ""
    )


def run_dirs(root: Path) -> List[Path]:
    found = set()
    for name in TRACE_FILES:
        for path in root.glob(f"*/*/*/{name}"):
            found.add(path.parent)
    return sorted(found)


def run_parts(root: Path, run_dir: Path) -> Dict[str, str]:
    parts = list(run_dir.relative_to(root).parts)
    return {
        "profile": parts[0] if len(parts) > 0 else "",
        "variant": parts[1] if len(parts) > 1 else "",
        "dataset": parts[2] if len(parts) > 2 else "",
    }


def rows_by_query(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = query_id(row)
        if qid and qid not in out:
            out[qid] = dict(row)
    return out


def candidate_chain(row: Mapping[str, Any]) -> Dict[str, Any]:
    direct = row.get("candidate_chain_feasibility")
    if isinstance(direct, dict):
        return direct
    diagnostics = row.get("diagnostics")
    if isinstance(diagnostics, dict):
        nested = diagnostics.get("candidate_chain_feasibility")
        if isinstance(nested, dict):
            return nested
    return {}


def chain_value(row: Mapping[str, Any], key: str, default: Any = 0.0) -> Any:
    chain = candidate_chain(row)
    if key in chain:
        return chain.get(key, default)
    return row.get(key, default)


def phase8_diagnostics(row: Mapping[str, Any]) -> Dict[str, Any]:
    diag = row.get("phase8_diagnostics")
    if isinstance(diag, dict):
        return diag
    diagnostics = row.get("diagnostics")
    if isinstance(diagnostics, dict):
        diag = diagnostics.get("phase8_diagnostics")
        if isinstance(diag, dict):
            return diag
    retrieval = row.get("retrieval")
    if isinstance(retrieval, dict):
        diagnostics = retrieval.get("diagnostics")
        if isinstance(diagnostics, dict):
            diag = diagnostics.get("phase8_diagnostics")
            if isinstance(diag, dict):
                return diag
    return {}


def diag_section(row: Mapping[str, Any], name: str) -> Dict[str, Any]:
    section = phase8_diagnostics(row).get(name)
    return dict(section) if isinstance(section, dict) else {}


def phase8_enabled(row: Mapping[str, Any]) -> bool:
    if "phase8_pamae_enabled" in row:
        return bool(row.get("phase8_pamae_enabled"))
    diagnostics = row.get("diagnostics")
    if isinstance(diagnostics, dict) and "phase8_pamae_enabled" in diagnostics:
        return bool(diagnostics.get("phase8_pamae_enabled"))
    return bool(phase8_diagnostics(row))


def support_keys_from_gold(gold: Any) -> set[str]:
    keys: set[str] = set()
    if not isinstance(gold, list):
        return keys
    for item in gold:
        if isinstance(item, dict):
            title = item.get("title", "")
            idx = item.get("sent_idx", item.get("sentence_id", item.get("sent_id", "")))
            if title != "" and idx != "":
                keys.add(canonical_support_key(str(title), idx))
            unit_id = item.get("unit_id")
            if unit_id:
                keys.add(normalize_evidence_id(str(unit_id)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            keys.add(canonical_support_key(str(item[0]), item[1]))
        elif isinstance(item, str):
            keys.add(normalize_evidence_id(item))
    return keys


def normalized_id_set(ids: Any) -> set[str]:
    if not isinstance(ids, list):
        return set()
    return {normalize_evidence_id(str(x)) for x in ids if str(x or "")}


def _id_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x or "")]
    return []


def atom_id_set(ids: Any) -> set[str]:
    if not isinstance(ids, list):
        return set()
    return {str(x) for x in ids if str(x or "").startswith("s::")}


def selected_atom_ids(row: Mapping[str, Any]) -> List[str]:
    ids = row.get("selected_atom_ids")
    if isinstance(ids, list):
        return [str(x) for x in ids]
    breakdown = row.get("selected_evidence_feature_breakdown")
    if isinstance(breakdown, list):
        atoms = []
        for item in breakdown:
            if isinstance(item, dict) and item.get("atom_id"):
                atoms.append(str(item.get("atom_id")))
        if atoms:
            return atoms
    ids = row.get("selected_evidence_ids")
    return [str(x) for x in ids] if isinstance(ids, list) else []


def is_phase8_variant(variant: str) -> bool:
    return str(variant or "").startswith("pamae_seed")


def is_source_balanced_variant(variant: str) -> bool:
    return str(variant or "") == "source_balanced_128"


def f1_from_counts(matched: int, predicted: int, gold: int) -> Optional[float]:
    if gold <= 0:
        return None
    if predicted <= 0:
        return 0.0
    precision = matched / predicted
    recall = matched / gold
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def oracle_f1_from_recall(recall: Optional[float]) -> Optional[float]:
    if recall is None:
        return None
    if recall <= 0:
        return 0.0
    return float(2.0 * recall / (1.0 + recall))


def _gold_support_keys(query: Mapping[str, Any]) -> set[str]:
    diag = query.get("phase7_diag")
    if isinstance(diag, dict):
        gold = support_keys_from_gold(diag.get("gold_supporting_facts", []))
        if gold:
            return gold
    result = query.get("result")
    if isinstance(result, dict):
        rendered = result.get("rendered")
        if isinstance(rendered, dict):
            gold = support_keys_from_gold(rendered.get("gold_supporting_facts", []))
            if gold:
                return gold
    return set()


def _candidate_ids(query: Mapping[str, Any]) -> List[str]:
    diag = query.get("phase7_diag")
    if isinstance(diag, dict):
        ids = _id_list(diag.get("phase2_candidate_ids"))
        if ids:
            return ids
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        ids = _id_list(p7.get("phase2_candidate_ids"))
        if ids:
            return ids
    result = query.get("result")
    if isinstance(result, dict):
        retrieval = result.get("retrieval")
        if isinstance(retrieval, dict):
            ids = _id_list(retrieval.get("candidate_sentence_ids"))
            if ids:
                return ids
    return []


def _selected_ids(query: Mapping[str, Any]) -> List[str]:
    diag = query.get("phase7_diag")
    if isinstance(diag, dict):
        ids = _id_list(diag.get("selected_evidence_ids"))
        if ids:
            return ids
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        ids = _id_list(p7.get("selected_atom_ids"))
        if ids:
            return ids
    result = query.get("result")
    if isinstance(result, dict):
        ids = _id_list(result.get("retrieval_selected_sentence_ids"))
        if ids:
            return ids
    return []


def _rendered_ids(query: Mapping[str, Any]) -> List[str]:
    diag = query.get("phase7_diag")
    if isinstance(diag, dict):
        ids = _id_list(diag.get("rendered_evidence_ids"))
        if ids:
            return ids
    result = query.get("result")
    if isinstance(result, dict):
        ids = _id_list(result.get("rendered_sentence_ids"))
        if ids:
            return ids
        rendered = result.get("rendered")
        if isinstance(rendered, dict):
            ids = _id_list(rendered.get("sentence_ids"))
            if ids:
                return ids
    return []


def _normalized_overlap(gold: set[str], ids: List[str]) -> Dict[str, Any]:
    norm_ids = normalized_id_set(ids)
    if not gold:
        return {
            "partial": None,
            "full": None,
            "recall": None,
            "matched": 0,
            "gold_count": 0,
            "predicted_count": len(norm_ids),
            "null_reason": "MISSING_GOLD_SUPPORT",
        }
    matched = len(gold & norm_ids)
    recall = matched / len(gold)
    return {
        "partial": matched > 0,
        "full": matched == len(gold),
        "recall": float(recall),
        "matched": matched,
        "gold_count": len(gold),
        "predicted_count": len(norm_ids),
        "null_reason": None,
    }


def normalized_query_diagnostics(query: Mapping[str, Any]) -> Dict[str, Any]:
    gold = _gold_support_keys(query)
    candidate = _normalized_overlap(gold, _candidate_ids(query))
    selected = _normalized_overlap(gold, _selected_ids(query))
    rendered = _normalized_overlap(gold, _rendered_ids(query))
    selected_f1 = f1_from_counts(
        int(selected["matched"]),
        int(selected["predicted_count"]),
        int(selected["gold_count"]),
    )
    candidate_oracle = oracle_f1_from_recall(candidate["recall"])
    p7 = query.get("phase7")
    chain = candidate_chain(p7 if isinstance(p7, dict) else {})
    chain_feasible = chain.get("chain_unit_oracle_feasible")
    if chain_feasible is None and isinstance(p7, dict):
        chain_feasible = p7.get("chain_unit_oracle_feasible")
    return {
        "query_id": str(query.get("query_id", "")),
        "normalized_candidate_gold_partial": candidate["partial"],
        "normalized_candidate_gold_full": candidate["full"],
        "normalized_candidate_gold_recall": candidate["recall"],
        "normalized_selected_gold_partial": selected["partial"],
        "normalized_selected_gold_full": selected["full"],
        "normalized_selected_gold_recall": selected["recall"],
        "normalized_rendered_gold_partial": rendered["partial"],
        "normalized_rendered_gold_full": rendered["full"],
        "normalized_rendered_gold_recall": rendered["recall"],
        "candidate_oracle_F1": candidate_oracle,
        "selected_context_F1": selected_f1,
        "chain_unit_oracle_feasible": bool(chain_feasible) if chain_feasible is not None else None,
        "candidate_null_reason": candidate["null_reason"],
        "selected_null_reason": selected["null_reason"],
        "rendered_null_reason": rendered["null_reason"],
        "gold_support_count": candidate["gold_count"],
        "candidate_id_count": candidate["predicted_count"],
        "selected_id_count": selected["predicted_count"],
        "rendered_id_count": rendered["predicted_count"],
    }


def source_tag_counter(selected_ids: Iterable[str], evidence_tags: Mapping[str, Any]) -> Counter:
    counter: Counter = Counter()
    for sid in selected_ids:
        tags = evidence_tags.get(sid)
        if not tags and "::" in sid:
            tags = evidence_tags.get(normalize_evidence_id(sid))
        if isinstance(tags, str):
            counter[tags] += 1
        elif isinstance(tags, list):
            for tag in tags:
                counter[str(tag)] += 1
    return counter


def load_run(root: Path, run_dir: Path) -> Dict[str, Any]:
    p7 = rows_by_query(read_jsonl(run_dir / "phase7_query_trace.jsonl"))
    p7_diag = rows_by_query(read_jsonl(run_dir / "phase7_diagnostics.jsonl"))
    p8q = rows_by_query(read_jsonl(run_dir / "phase8_query_trace.jsonl"))
    seeds = rows_by_query(read_jsonl(run_dir / "phase8_seed_trace.jsonl"))
    evidence = rows_by_query(read_jsonl(run_dir / "phase8_evidence_trace.jsonl"))
    timing = rows_by_query(read_jsonl(run_dir / "phase8_stage_timing.jsonl"))
    results = rows_by_query(read_jsonl(run_dir / "rag_query_results.jsonl"))
    qids = set()
    for mapping in [p7, p7_diag, p8q, seeds, evidence, timing, results]:
        qids.update(mapping.keys())
    queries = []
    for qid in sorted(qids):
        queries.append(
            {
                "query_id": qid,
                "phase7": p7.get(qid, {}),
                "phase7_diag": p7_diag.get(qid, {}),
                "phase8_query": p8q.get(qid, {}),
                "seed": seeds.get(qid, {}),
                "evidence": evidence.get(qid, {}),
                "timing": timing.get(qid, {}),
                "result": results.get(qid, {}),
            }
        )
    files = {
        name: (run_dir / name).exists()
        for name in TRACE_FILES
    }
    summary = read_json(run_dir / "rag_summary.json")
    expected = safe_int(summary.get("num_queries", summary.get("count", summary.get("samples", 100))), 100)
    actual = len(queries)
    if not any(files.values()):
        status = "MISSING"
    elif not files.get("rag_summary.json") and actual == 0:
        status = "MISSING"
    elif actual == 0:
        status = "MISSING"
    elif actual < min(expected, 100):
        status = "PARTIAL"
    elif files.get("rag_summary.json") and safe_int(summary.get("num_queries", summary.get("count", summary.get("samples", actual))), actual) != actual:
        status = "SUMMARY_MISMATCH"
    else:
        status = "COMPLETE"
    return {
        **run_parts(root, run_dir),
        "run_dir": str(run_dir),
        "files": files,
        "summary": summary,
        "expected_num_queries": expected,
        "actual_num_queries": actual,
        "status": status,
        "queries": queries,
    }


def load_runs(root: Path) -> List[Dict[str, Any]]:
    root = root.resolve()
    return [load_run(root, run_dir) for run_dir in run_dirs(root)]


def group_runs(runs: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (str(run.get("dataset", "")), str(run.get("profile", "")), str(run.get("variant", "")))
        groups[key].append(run)
    return dict(groups)


def iter_queries(group: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for run in group:
        for query in run.get("queries", []):
            yield query


def question_for(query: Mapping[str, Any]) -> str:
    for key in ["phase7", "phase8_query", "phase7_diag", "result"]:
        row = query.get(key)
        if isinstance(row, dict) and row.get("question"):
            return str(row.get("question"))
    return ""


def metric_from_result(query: Mapping[str, Any], key: str) -> float:
    result = query.get("result")
    if not isinstance(result, dict):
        return 0.0
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return 0.0
    aliases = {
        "sf_recall": "supporting_fact_recall",
        "sf_precision": "supporting_fact_precision",
        "sf_f1": "supporting_fact_f1",
    }
    return safe_float(metrics.get(aliases.get(key, key), 0.0), 0.0)


def failure_label_repaired(variant: str, query: Mapping[str, Any]) -> str:
    diag = normalized_query_diagnostics(query)
    candidate_recall = safe_float(diag.get("normalized_candidate_gold_recall"), 0.0)
    selected_recall = safe_float(diag.get("normalized_selected_gold_recall"), 0.0)
    rendered_recall = safe_float(diag.get("normalized_rendered_gold_recall"), 0.0)
    f1 = metric_from_result(query, "f1")
    sf_recall = metric_from_result(query, "sf_recall")
    p7 = query.get("phase7")
    p7row = p7 if isinstance(p7, dict) else {}
    if is_source_balanced_variant(variant):
        if candidate_recall <= 0.0:
            return "PHASE1_NOT_FOUND"
        if selected_recall + 1e-9 < candidate_recall:
            return "FINAL_SELECTION_FAILED"
        if rendered_recall < 1.0 and sf_recall < 1.0:
            return "SELECTED_NOT_SUFFICIENT"
        if rendered_recall >= 1.0 and f1 < 0.5:
            return "QA_PROMPT_FAILED_WITH_GOLD_CONTEXT"
        if rendered_recall > 0.0 and f1 < 0.5:
            return "QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT"
        return "OTHER"

    if is_phase8_variant(variant):
        universe = entity_universe(query)
        best = seed_best(query)
        refine = seed_refinement(query)
        proposal = evidence_proposal(query)
        if safe_float(universe.get("num_entities", 0.0), 0.0) <= 0.0:
            return "UQ_ENTITY_MISS"
        before = float(bool(refine.get("before_seed_evidence_gold_hit_eval_only", False)))
        after = float(bool(refine.get("after_seed_evidence_gold_hit_eval_only", False)))
        changed = safe_float(refine.get("num_changed_seeds", 0.0), 0.0) > 0.0
        if changed and after < before:
            return "REFINEMENT_DRIFT"
        if not bool(best.get("seed_gold_hit_eval_only", False)) and candidate_recall <= 0.10:
            return "SEED_SELECTION_MISS"
        if safe_float(proposal.get("num_final_evidence_candidates", 0.0), 0.0) > 0.0 and candidate_recall <= 0.10:
            return "SEED_TO_EVIDENCE_MISS"
        feasible = diag.get("chain_unit_oracle_feasible")
        if feasible is False and candidate_recall > 0.0:
            return "CANDIDATE_CHAIN_INFEASIBLE"
        if selected_recall + 1e-9 < candidate_recall:
            return "FINAL_SELECTION_FAILED"
        if rendered_recall < 1.0 and sf_recall < 1.0:
            return "SELECTED_NOT_SUFFICIENT"
        if rendered_recall > 0.0 and f1 < 0.5:
            return "QA_PROMPT_FAILED"
        return "OTHER"

    return "OTHER"


def evidence_proposal(query: Mapping[str, Any]) -> Dict[str, Any]:
    ev = query.get("evidence")
    if isinstance(ev, dict):
        proposal = ev.get("evidence_proposal")
        if isinstance(proposal, dict):
            return proposal
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        section = diag_section(p7, "evidence_proposal")
        if section:
            return section
    return {}


def evidence_tags(query: Mapping[str, Any]) -> Dict[str, Any]:
    ev = query.get("evidence")
    if isinstance(ev, dict):
        tags = ev.get("evidence_source_tags")
        if isinstance(tags, dict):
            return tags
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        tags = phase8_diagnostics(p7).get("evidence_source_tags")
        if isinstance(tags, dict):
            return tags
    return {}


def seed_best(query: Mapping[str, Any]) -> Dict[str, Any]:
    seed = query.get("seed")
    if isinstance(seed, dict):
        best = seed.get("best_seed_selection")
        if isinstance(best, dict):
            return best
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        return diag_section(p7, "best_seed_selection")
    return {}


def seed_refinement(query: Mapping[str, Any]) -> Dict[str, Any]:
    seed = query.get("seed")
    if isinstance(seed, dict):
        ref = seed.get("refinement")
        if isinstance(ref, dict):
            return ref
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        return diag_section(p7, "refinement")
    return {}


def entity_universe(query: Mapping[str, Any]) -> Dict[str, Any]:
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        section = diag_section(p7, "entity_universe")
        if section:
            return section
    p8q = query.get("phase8_query")
    if isinstance(p8q, dict) and p8q.get("entity_universe_size"):
        return {"num_entities": p8q.get("entity_universe_size")}
    return {}


def sampling_diag(query: Mapping[str, Any]) -> Dict[str, Any]:
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        section = diag_section(p7, "pamae_sampling")
        if section:
            return section
    p8q = query.get("phase8_query")
    if isinstance(p8q, dict):
        return {
            "k": p8q.get("k", 0),
            "sample_size": p8q.get("sample_size", 0),
            "num_samples": p8q.get("num_samples", 0),
        }
    return {}


def phase7_candidate_recall(query: Mapping[str, Any]) -> float:
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        return safe_float(chain_value(p7, "candidate_gold_recall", 0.0), 0.0)
    return 0.0


def phase7_candidate_full(query: Mapping[str, Any]) -> bool:
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        return bool(chain_value(p7, "candidate_gold_full", False))
    return False


def phase7_chain_oracle(query: Mapping[str, Any]) -> float:
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        return 1.0 if bool(chain_value(p7, "chain_unit_oracle_feasible", False)) else 0.0
    return 0.0


def missing_files_for(run: Mapping[str, Any]) -> List[str]:
    files = run.get("files")
    if not isinstance(files, dict):
        return TRACE_FILES
    return [name for name, exists in files.items() if not exists]
