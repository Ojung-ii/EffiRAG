#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Select top-k fusion variants from stage-1 summary")
    p.add_argument("--summary-json", required=True)
    p.add_argument("--baseline-variant", required=True)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--max-recall-drop", type=float, default=0.01)
    p.add_argument("--output-json", default="")
    args = p.parse_args()

    payload = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []) or [])

    candidates = [r for r in rows if str(r.get("variant", "")) != args.baseline_variant]

    guarded = [
        r
        for r in candidates
        if float(r.get("delta_recall_at_20", 0.0)) >= -abs(float(args.max_recall_drop))
    ]
    pool = guarded if guarded else candidates

    def key(r):
        return (
            float(r.get("delta_f1", -1e9)),
            float(r.get("delta_em", -1e9)),
            float(r.get("delta_rendered_recall", -1e9)),
            -float(r.get("delta_total_ms", 0.0)),
        )

    ranked = sorted(pool, key=key, reverse=True)
    selected = ranked[: max(1, int(args.topk))]

    out = {
        "summary_json": str(Path(args.summary_json).resolve()),
        "baseline_variant": args.baseline_variant,
        "max_recall_drop": float(args.max_recall_drop),
        "topk": int(args.topk),
        "selected_variants": [str(r.get("variant", "")) for r in selected],
        "ranked": [
            {
                "variant": str(r.get("variant", "")),
                "delta_f1": float(r.get("delta_f1", 0.0)),
                "delta_em": float(r.get("delta_em", 0.0)),
                "delta_recall_at_20": float(r.get("delta_recall_at_20", 0.0)),
                "delta_rendered_recall": float(r.get("delta_rendered_recall", 0.0)),
                "delta_total_ms": float(r.get("delta_total_ms", 0.0)),
            }
            for r in ranked
        ],
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(",".join(out["selected_variants"]))


if __name__ == "__main__":
    main()
