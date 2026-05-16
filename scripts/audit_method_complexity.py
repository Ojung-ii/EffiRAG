#!/usr/bin/env python3
"""Phase-6O method complexity and dataset-heuristic risk audit."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


DATASET_NAME_TOKENS = ("hotpotqa", "2wikimultihopqa", "2wiki", "musique", "popqa")
SCAN_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    ("dataset ==", re.compile(r"\bdataset(?:_name)?\s*==")),
    ("dataset !=", re.compile(r"\bdataset(?:_name)?\s*!=")),
    ("dataset in", re.compile(r"\bdataset(?:_name)?\s+in\b")),
    ("hotpotqa", re.compile(r"\bhotpotqa\b", re.IGNORECASE)),
    ("2wikimultihopqa", re.compile(r"\b2wikimultihopqa\b", re.IGNORECASE)),
    ("2wiki", re.compile(r"\b2wiki\b", re.IGNORECASE)),
    ("musique", re.compile(r"\bmusique\b", re.IGNORECASE)),
    ("popqa", re.compile(r"\bpopqa\b", re.IGNORECASE)),
)

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".toml",
    ".ini",
    ".cfg",
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _iter_text_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    skip_parts = {".git", ".pytest_cache", "__pycache__", "outputs", "logs"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def _is_path_like_line(line_lc: str) -> bool:
    if "output_dir" in line_lc or "outputs/" in line_lc or "logs/" in line_lc:
        return True
    if "config_path" in line_lc or "summary_path" in line_lc:
        return True
    return False


def classify_dataset_branch_risk(*, rel_path: str, pattern: str, line_text: str) -> Tuple[str, str]:
    lower = line_text.lower()
    has_dataset_token = any(tok in lower for tok in DATASET_NAME_TOKENS)
    has_branch_keyword = bool(re.search(r"\b(if|elif|else|case|match|when|switch)\b", lower))

    if pattern in {"dataset ==", "dataset !=", "dataset in"}:
        return "high", "method behavior changes by dataset variable condition"
    if has_dataset_token and has_branch_keyword:
        return "high", "dataset name appears in branching logic"
    if _is_path_like_line(lower):
        return "low", "path/logging/output naming only"
    if rel_path.startswith("configs/main_config/copy_span_instruction_unified/"):
        return "medium", "config-level dataset mention; behavior should be reviewed"
    return "medium", "dataset mention found; check if it affects method behavior"


def scan_dataset_branches(*, code_roots: Sequence[Path], config_root: Path, repo_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scan_roots = [p.resolve() for p in list(code_roots) + [config_root]]
    for root in scan_roots:
        for path in _iter_text_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel_path = str(path.resolve().relative_to(repo_root))
            for idx, line in enumerate(text.splitlines(), start=1):
                for pattern_name, pattern_re in SCAN_PATTERNS:
                    if not pattern_re.search(line):
                        continue
                    risk_level, note = classify_dataset_branch_risk(
                        rel_path=rel_path,
                        pattern=pattern_name,
                        line_text=line,
                    )
                    rows.append(
                        {
                            "file": rel_path,
                            "line_no": idx,
                            "pattern": pattern_name,
                            "line_text": line.strip(),
                            "risk_level": risk_level,
                            "note": note,
                        }
                    )
    rows.sort(key=lambda x: (str(x.get("file", "")), _safe_int(x.get("line_no", 0), 0), str(x.get("pattern", ""))))
    return rows


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(obj or {}) if isinstance(obj, Mapping) else {}
    except Exception:
        return {}


def _count_budget_fields(cfg: Mapping[str, Any]) -> int:
    count = 0
    for key in cfg.keys():
        k = str(key).lower()
        if any(
            token in k
            for token in (
                "budget",
                "token",
                "topn",
                "topk",
                "topm",
                "max_",
                "min_",
            )
        ):
            count += 1
    return count


def _count_threshold_fields(cfg: Mapping[str, Any]) -> int:
    count = 0
    for key in cfg.keys():
        k = str(key).lower()
        if "threshold" in k or k.endswith("_min") or k.endswith("_max"):
            count += 1
    return count


def _choose_profile_yaml(profile_dir: Path) -> Path | None:
    preferred = [
        profile_dir / "hotpotqa.yaml",
        profile_dir / "2wikimultihopqa.yaml",
        profile_dir / "musique.yaml",
        profile_dir / "popqa.yaml",
    ]
    for path in preferred:
        if path.exists():
            return path
    yamls = sorted(list(profile_dir.glob("*.yaml")) + list(profile_dir.glob("*.yml")))
    return yamls[0] if yamls else None


def _risk_note_for_profile(
    *,
    num_enabled_flags: int,
    uses_sentence_contract: bool,
    uses_support_span: bool,
    uses_adaptive_span: bool,
    uses_gl_rcedr: bool,
) -> str:
    complexity_axes = int(uses_sentence_contract) + int(uses_support_span) + int(uses_adaptive_span)
    if num_enabled_flags >= 55 or complexity_axes >= 2:
        return "high complexity risk; keep single main profile + minimal ablations in paper text"
    if num_enabled_flags >= 40 or uses_gl_rcedr:
        return "medium complexity; describe as query-adaptive generic method, not dataset heuristic"
    return "low complexity"


def collect_profile_complexity(config_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not config_root.exists():
        return rows
    for profile_dir in sorted(p for p in config_root.iterdir() if p.is_dir()):
        config_path = _choose_profile_yaml(profile_dir)
        if config_path is None:
            continue
        cfg = _read_yaml(config_path)
        bool_items = {str(k): v for k, v in cfg.items() if isinstance(v, bool)}
        num_enabled = int(sum(1 for v in bool_items.values() if bool(v)))
        num_disabled = int(sum(1 for v in bool_items.values() if not bool(v)))
        uses_sentence_contract = bool(cfg.get("sentence_contract_render_enabled", False))
        uses_support_span = bool(cfg.get("support_span_contract_enabled", False))
        uses_adaptive_span = bool(cfg.get("adaptive_support_span_enabled", False))
        uses_gl_rcedr = bool(cfg.get("gl_rcedr_enabled", False))
        rows.append(
            {
                "profile": profile_dir.name,
                "num_enabled_flags": num_enabled,
                "num_disabled_flags": num_disabled,
                "num_budget_fields": _count_budget_fields(cfg),
                "num_threshold_fields": _count_threshold_fields(cfg),
                "uses_sentence_contract": uses_sentence_contract,
                "uses_support_span": uses_support_span,
                "uses_adaptive_span": uses_adaptive_span,
                "uses_gl_rcedr": uses_gl_rcedr,
                "risk_note": _risk_note_for_profile(
                    num_enabled_flags=num_enabled,
                    uses_sentence_contract=uses_sentence_contract,
                    uses_support_span=uses_support_span,
                    uses_adaptive_span=uses_adaptive_span,
                    uses_gl_rcedr=uses_gl_rcedr,
                ),
                "sample_config_path": str(config_path),
            }
        )
    rows.sort(key=lambda x: str(x.get("profile", "")))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row or {}))


def _build_markdown(
    *,
    dataset_scan_rows: Sequence[Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
) -> str:
    high_count = sum(1 for r in dataset_scan_rows if str(r.get("risk_level", "")) == "high")
    medium_count = sum(1 for r in dataset_scan_rows if str(r.get("risk_level", "")) == "medium")
    low_count = sum(1 for r in dataset_scan_rows if str(r.get("risk_level", "")) == "low")
    profile_count = len(profile_rows)
    high_complexity = sum(1 for r in profile_rows if "high complexity" in str(r.get("risk_note", "")).lower())

    lines: List[str] = []
    lines.append("# Method Complexity and Heuristic Risk Audit")
    lines.append("")
    lines.append("## 1. Dataset-specific branch scan")
    lines.append("")
    lines.append(f"- scanned matches: {len(dataset_scan_rows)}")
    lines.append(f"- high risk: {high_count}")
    lines.append(f"- medium risk: {medium_count}")
    lines.append(f"- low risk: {low_count}")
    lines.append("")
    lines.append("## 2. Profile complexity")
    lines.append("")
    lines.append(f"- unified profile directories scanned: {profile_count}")
    lines.append(f"- high complexity profiles: {high_complexity}")
    lines.append("")
    lines.append("## 3. Main method framing risk")
    lines.append("")
    lines.append(
        "- Risk grows when many similarly named variants are treated as primary claims. Keep one main profile and move others to ablation/appendix."
    )
    lines.append("")
    lines.append("## 4. Reviewer-risk assessment")
    lines.append("")
    if high_count > 0:
        lines.append("- Dataset-conditional behavior patterns exist and should be explicitly justified or removed.")
    else:
        lines.append("- No direct dataset-conditional method branch (`dataset ==`, `dataset in`, `dataset !=`) was detected.")
    if high_complexity > 0:
        lines.append("- Several profiles carry high flag complexity; paper framing should avoid appearing as heuristic search.")
    else:
        lines.append("- Profile complexity is moderate and can be presented as controlled ablation.")
    lines.append("")
    lines.append("## 5. Recommendation")
    lines.append("")
    lines.append("- Treat `legacy_sota` as teacher/reference, not a direct optimization target.")
    lines.append("- Evaluate primary success against `unified_large` QA-token Pareto and additional external baseline comparisons.")
    lines.append("- Preserve dataset-agnostic query/evidence-adaptive framing.")
    lines.append("- Restrict main text to 1-2 primary candidates and report the rest as ablations.")
    lines.append("- Use QA-aware Pareto evidence for final method choice, not retrieval-only metrics alone.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit method complexity and dataset-specific heuristic risk.")
    parser.add_argument(
        "--config-root",
        default="configs/main_config/copy_span_instruction_unified",
        help="Unified config root.",
    )
    parser.add_argument(
        "--code-roots",
        nargs="+",
        default=["effirag", "scripts"],
        help="Code roots to scan for dataset branching patterns.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for audit CSV/MD files.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_root = (repo_root / args.config_root).resolve()
    code_roots = [(repo_root / p).resolve() for p in list(args.code_roots or [])]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_scan_rows = scan_dataset_branches(code_roots=code_roots, config_root=config_root, repo_root=repo_root)
    profile_rows = collect_profile_complexity(config_root)

    scan_csv = output_dir / "dataset_branch_scan.csv"
    profile_csv = output_dir / "profile_complexity.csv"
    md_path = output_dir / "METHOD_COMPLEXITY_AUDIT.md"

    _write_csv(
        scan_csv,
        dataset_scan_rows,
        ["file", "line_no", "pattern", "line_text", "risk_level", "note"],
    )
    _write_csv(
        profile_csv,
        profile_rows,
        [
            "profile",
            "num_enabled_flags",
            "num_disabled_flags",
            "num_budget_fields",
            "num_threshold_fields",
            "uses_sentence_contract",
            "uses_support_span",
            "uses_adaptive_span",
            "uses_gl_rcedr",
            "risk_note",
            "sample_config_path",
        ],
    )
    md_path.write_text(
        _build_markdown(dataset_scan_rows=dataset_scan_rows, profile_rows=profile_rows) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "config_root": str(config_root),
        "code_roots": [str(p) for p in code_roots],
        "output_dir": str(output_dir),
        "dataset_scan_rows": len(dataset_scan_rows),
        "profile_rows": len(profile_rows),
        "dataset_branch_scan_csv": str(scan_csv),
        "profile_complexity_csv": str(profile_csv),
        "method_complexity_md": str(md_path),
    }
    (output_dir / "phase6o_method_complexity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()
