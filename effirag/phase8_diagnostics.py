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
    return {
        **run_parts(root, run_dir),
        "run_dir": str(run_dir),
        "files": files,
        "summary": read_json(run_dir / "rag_summary.json"),
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
