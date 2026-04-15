#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _parse_run_arg(item: str):
    token = str(item or "").strip()
    if not token:
        raise ValueError("empty --run value")
    if "=" in token:
        label, path = token.split("=", 1)
        return label.strip(), path.strip()
    path = token
    label = Path(path).parent.name
    return label, path


def _metric(payload: dict, key: str, fallback: float = 0.0) -> float:
    val = payload.get(key, fallback)
    try:
        return float(val)
    except Exception:
        return float(fallback)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple run summary json files into a single markdown/json table."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in form label=/abs/path/to/rag_summary.json (or just /abs/path). Repeatable.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/profiling/final_compare_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="outputs/profiling/final_compare_summary.md",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Run Comparison",
    )
    args = parser.parse_args()

    rows = []
    for item in args.run:
        label, path = _parse_run_arg(item)
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "label": label,
                "path": str(p.resolve()),
                "supporting_fact_recall": _metric(payload, "supporting_fact_recall", 0.0),
                "recall_at_1": _metric(payload, "supporting_fact_recall_at_1", 0.0),
                "recall_at_5": _metric(payload, "supporting_fact_recall_at_5", 0.0),
                "recall_at_20": _metric(payload, "supporting_fact_recall_at_20", 0.0),
                "rendered_recall": _metric(
                    payload,
                    "rendered_supporting_fact_recall",
                    _metric(payload, "supporting_fact_recall", 0.0),
                ),
                "em": _metric(payload, "em", 0.0),
                "f1": _metric(payload, "f1", 0.0),
                "retrieval_ms": _metric(payload, "retrieval_latency_ms", 0.0),
                "total_ms": _metric(payload, "total_latency_ms", _metric(payload, "retrieval_latency_ms", 0.0)),
            }
        )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "title": args.title,
        "n_runs": int(len(rows)),
        "runs": rows,
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [f"# {args.title}", ""]
    lines.append("| profile | Recall@1 | Recall@5 | Recall@20 | rendered recall | EM | F1 | retrieval_ms | total_ms | summary_path |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            "| {label} | {r1:.4f} | {r5:.4f} | {r20:.4f} | {rr:.4f} | {em:.4f} | {f1:.4f} | {ret:.2f} | {tot:.2f} | {path} |".format(
                label=r["label"],
                r1=r["recall_at_1"],
                r5=r["recall_at_5"],
                r20=r["recall_at_20"],
                rr=r["rendered_recall"],
                em=r["em"],
                f1=r["f1"],
                ret=r["retrieval_ms"],
                tot=r["total_ms"],
                path=r["path"],
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
