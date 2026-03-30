#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _load(path: str):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _pick_best(rows, baseline_variant: str, max_recall_drop: float = 0.01):
    non_base = [r for r in rows if str(r.get('variant', '')) != baseline_variant]
    guarded = [r for r in non_base if float(r.get('delta_recall_at_20', 0.0)) >= -abs(max_recall_drop)]
    pool = guarded if guarded else non_base
    if not pool:
        return None
    return max(
        pool,
        key=lambda r: (
            float(r.get('delta_f1', -1e9)),
            float(r.get('delta_em', -1e9)),
            float(r.get('delta_rendered_recall', -1e9)),
            -float(r.get('delta_total_ms', 0.0)),
        ),
    )


def _fmt(v, d=4):
    if v is None:
        return 'N/A'
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return 'N/A'


def main():
    ap = argparse.ArgumentParser(description='Summarize entity-chunk graph final interpretation.')
    ap.add_argument('--hotpot-summary', required=True)
    ap.add_argument('--wiki-summary', required=True)
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--output-md', required=True)
    ap.add_argument('--git-branch', default='')
    ap.add_argument('--git-commit', default='')
    ap.add_argument('--git-tag', default='')
    args = ap.parse_args()

    hot = _load(args.hotpot_summary)
    wiki = _load(args.wiki_summary)

    hot_base = str(hot.get('baseline_variant', 'hotpot_baseline'))
    wiki_base = str(wiki.get('baseline_variant', '2wiki_baseline'))

    hot_rows = list(hot.get('rows', []) or [])
    wiki_rows = list(wiki.get('rows', []) or [])

    hot_best = _pick_best(hot_rows, hot_base, max_recall_drop=0.01)
    wiki_best = _pick_best(wiki_rows, wiki_base, max_recall_drop=0.01)

    payload = {
        'experiment_family': 'next_method_design',
        'question_being_answered': 'Does entity-chunk graph provide incremental gain over current-best fusion?',
        'baseline_reference': 'frozen baseline + current-best retrieval-delivery fusion',
        'frozen_config_reference': 'configs/rag_speed_profile.yaml + configs/rag_quality_profile.yaml',
        'dataset_scope': ['hotpotqa', '2wikimultihopqa'],
        'changed_components': ['entity_chunk_graph', 'retrieval_delivery_fusion_inheritance'],
        'git_branch': args.git_branch,
        'git_commit': args.git_commit,
        'git_tag': args.git_tag,
        'hotpot': {
            'baseline_variant': hot_base,
            'best_variant': (hot_best or {}).get('variant', ''),
            'best_row': hot_best,
            'summary_path': str(Path(args.hotpot_summary).resolve()),
        },
        'wiki': {
            'baseline_variant': wiki_base,
            'best_variant': (wiki_best or {}).get('variant', ''),
            'best_row': wiki_best,
            'summary_path': str(Path(args.wiki_summary).resolve()),
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# Entity-Chunk Graph Final Interpretation',
        '',
        '- experiment_family: `next_method_design`',
        '- baseline_reference: `frozen baseline + current-best retrieval-delivery fusion`',
        f'- git_branch: {args.git_branch}',
        f'- git_commit: {args.git_commit}',
        f'- git_tag: {args.git_tag}',
        '',
        '## HotpotQA',
    ]

    if hot_best:
        lines.append(
            f"- best candidate: `{hot_best.get('variant', '')}` "
            f"(ΔR20={_fmt(hot_best.get('delta_recall_at_20'), 4)}, "
            f"Δrendered={_fmt(hot_best.get('delta_rendered_recall'), 4)}, "
            f"ΔEM={_fmt(hot_best.get('delta_em'), 4)}, "
            f"ΔF1={_fmt(hot_best.get('delta_f1'), 4)}, "
            f"Δtotal_ms={_fmt(hot_best.get('delta_total_ms'), 2)})."
        )
    else:
        lines.append('- no non-baseline candidate found.')

    lines.extend(['', '## 2Wiki'])
    if wiki_best:
        lines.append(
            f"- best candidate: `{wiki_best.get('variant', '')}` "
            f"(ΔR20={_fmt(wiki_best.get('delta_recall_at_20'), 4)}, "
            f"Δrendered={_fmt(wiki_best.get('delta_rendered_recall'), 4)}, "
            f"ΔEM={_fmt(wiki_best.get('delta_em'), 4)}, "
            f"ΔF1={_fmt(wiki_best.get('delta_f1'), 4)}, "
            f"Δtotal_ms={_fmt(wiki_best.get('delta_total_ms'), 2)})."
        )
    else:
        lines.append('- no non-baseline candidate found.')

    lines.extend([
        '',
        '## Decision Rules',
        '- If both datasets beat current-best fusion with stable Recall@20 and F1, full replacement can be considered.',
        '- If only one dataset gains, prefer hybrid migration (dataset-specific).',
        '- If gains are marginal/unstable, keep current-best fusion and treat entity-chunk graph as follow-up track.',
        '',
    ])

    out_md.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
