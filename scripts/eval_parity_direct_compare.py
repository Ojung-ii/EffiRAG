#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import List, Tuple


def _load_rows(path: Path):
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


def _to_gold_list(row: dict) -> List[str]:
    vals = row.get("gold_answers")
    if isinstance(vals, list):
        out = [str(x).strip() for x in vals if str(x).strip()]
        if out:
            return out
    answer = str(row.get("answer", "") or "").strip()
    return [answer]


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _resolve_reference_eval_src(repo_root: Path) -> Path:
    candidate = repo_root / "third_party" / "HippoRAG2" / "src" / "hipporag"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("hipporag source not found in local third_party/HippoRAG2.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct legacy/parity/HippoRAG2 evaluator comparison.")
    parser.add_argument("--input-jsonl", required=True, type=str)
    parser.add_argument("--output-md", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from effirag.eval.evaluator import QAEvaluator

    input_path = Path(args.input_jsonl).resolve()
    rows = _load_rows(input_path)
    preds = [str((r.get("prediction", "") or "")) for r in rows]
    gold_batch = [_to_gold_list(r) for r in rows]

    legacy_eval = QAEvaluator(mode="legacy").evaluate_batch(preds, gold_batch)
    parity_eval = QAEvaluator(mode="hipporag2_parity").evaluate_batch(preds, gold_batch)

    reference_loaded = False
    reference_scores = {"ExactMatch": None, "F1": None}
    try:
        reference_eval_src = _resolve_reference_eval_src(repo_root)

        def _load_module(name: str, file_path: Path):
            spec = importlib.util.spec_from_file_location(name, str(file_path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(mod)
            return mod

        # Build lightweight package skeleton for relative imports.
        if "hipporag" not in sys.modules:
            pkg = types.ModuleType("hipporag")
            pkg.__path__ = [str(reference_eval_src)]
            sys.modules["hipporag"] = pkg
        if "hipporag.utils" not in sys.modules:
            pkg = types.ModuleType("hipporag.utils")
            pkg.__path__ = [str(reference_eval_src / "utils")]
            sys.modules["hipporag.utils"] = pkg
        if "hipporag.evaluation" not in sys.modules:
            pkg = types.ModuleType("hipporag.evaluation")
            pkg.__path__ = [str(reference_eval_src / "evaluation")]
            sys.modules["hipporag.evaluation"] = pkg

        # Load only evaluator-related modules without importing hipporag/__init__.py.
        _load_module("hipporag.utils.logging_utils", reference_eval_src / "utils" / "logging_utils.py")
        _load_module("hipporag.utils.config_utils", reference_eval_src / "utils" / "config_utils.py")
        _load_module("hipporag.utils.eval_utils", reference_eval_src / "utils" / "eval_utils.py")
        _load_module("hipporag.evaluation.base", reference_eval_src / "evaluation" / "base.py")
        qa_eval_mod = _load_module("hipporag.evaluation.qa_eval", reference_eval_src / "evaluation" / "qa_eval.py")

        QAExactMatch = qa_eval_mod.QAExactMatch
        QAF1Score = qa_eval_mod.QAF1Score

        em_metric = QAExactMatch()
        f1_metric = QAF1Score()
        ref_em, _ = em_metric.calculate_metric_scores(gold_batch, preds)
        ref_f1, _ = f1_metric.calculate_metric_scores(gold_batch, preds)
        reference_scores = {
            "ExactMatch": float(ref_em.get("ExactMatch", 0.0)),
            "F1": float(ref_f1.get("F1", 0.0)),
        }
        reference_loaded = True
    except Exception:
        reference_loaded = False

    report = {
        "input_jsonl": str(input_path),
        "n_samples": len(rows),
        "legacy": {
            "ExactMatch": float(legacy_eval.get("ExactMatch", 0.0)),
            "F1": float(legacy_eval.get("F1", 0.0)),
        },
        "hipporag2_parity": {
            "ExactMatch": float(parity_eval.get("ExactMatch", 0.0)),
            "F1": float(parity_eval.get("F1", 0.0)),
        },
        "hipporag2_reference": reference_scores,
        "hipporag2_reference_loaded": bool(reference_loaded),
        "delta_parity_minus_legacy": {
            "ExactMatch": float(parity_eval.get("ExactMatch", 0.0) - legacy_eval.get("ExactMatch", 0.0)),
            "F1": float(parity_eval.get("F1", 0.0) - legacy_eval.get("F1", 0.0)),
        },
    }
    if reference_loaded:
        report["delta_parity_minus_hipporag2_ref"] = {
            "ExactMatch": float(report["hipporag2_parity"]["ExactMatch"] - reference_scores["ExactMatch"]),
            "F1": float(report["hipporag2_parity"]["F1"] - reference_scores["F1"]),
        }

    out_json = Path(args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Evaluator Direct Compare Report")
    lines.append("")
    lines.append(f"- input_jsonl: `{input_path}`")
    lines.append(f"- n_samples: {len(rows)}")
    lines.append(f"- hipporag2_reference_loaded: {reference_loaded}")
    lines.append("")
    lines.append("| evaluator | EM | F1 |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| legacy | {report['legacy']['ExactMatch']:.6f} | {report['legacy']['F1']:.6f} |")
    lines.append(f"| hipporag2_parity | {report['hipporag2_parity']['ExactMatch']:.6f} | {report['hipporag2_parity']['F1']:.6f} |")
    if reference_loaded:
        lines.append(f"| hipporag2_reference | {reference_scores['ExactMatch']:.6f} | {reference_scores['F1']:.6f} |")
    lines.append("")
    lines.append("| delta | value |")
    lines.append("| --- | ---: |")
    lines.append(f"| parity - legacy (EM) | {report['delta_parity_minus_legacy']['ExactMatch']:.6f} |")
    lines.append(f"| parity - legacy (F1) | {report['delta_parity_minus_legacy']['F1']:.6f} |")
    if reference_loaded:
        lines.append(f"| parity - hipporag2_ref (EM) | {report['delta_parity_minus_hipporag2_ref']['ExactMatch']:.6f} |")
        lines.append(f"| parity - hipporag2_ref (F1) | {report['delta_parity_minus_hipporag2_ref']['F1']:.6f} |")

    out_md = Path(args.output_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
