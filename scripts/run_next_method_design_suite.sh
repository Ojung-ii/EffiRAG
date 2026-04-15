#!/bin/bash
set -euo pipefail

# Orchestrates next_method_design experiments in the requested order:
# 1) 2Wiki run-objective redesign
# 2) 2Wiki grounding ablation
# 3) MuSiQue failure diagnosis
# 4) Strategy + final interpretation report generation

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

LIMIT="${LIMIT:-200}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

RUN_2WIKI_RUN_OBJECTIVE="${RUN_2WIKI_RUN_OBJECTIVE:-true}"
RUN_2WIKI_GROUNDING="${RUN_2WIKI_GROUNDING:-true}"
RUN_MUSIQUE_DIAG="${RUN_MUSIQUE_DIAG:-true}"

export CUDA_VISIBLE_DEVICES
export PYTHON_BIN

echo "[nextdesign_suite] branch=$(git -C "${REPO_ROOT}" branch --show-current) commit=$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
echo "[nextdesign_suite] LIMIT=${LIMIT}"

if [ "${RUN_2WIKI_RUN_OBJECTIVE}" = "true" ]; then
  LIMIT="${LIMIT}" bash scripts/run_2wiki_run_objective_ablation.sh
fi

if [ "${RUN_2WIKI_GROUNDING}" = "true" ]; then
  LIMIT="${LIMIT}" bash scripts/run_2wiki_grounding_ablation.sh
fi

if [ "${RUN_MUSIQUE_DIAG}" = "true" ]; then
  MODE=qa LIMIT="${LIMIT}" bash scripts/run_musique_failure_diagnosis.sh
fi

bash scripts/generate_next_phase_operating_strategy.sh

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

root = Path('outputs/profiling/next_method_design')
root.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def load_latest(preferred: Path, fallback: Path):
    data = load_json(preferred)
    if data is not None:
        return data
    return load_json(fallback)

run_obj = load_latest(
    root / '2wiki_run_objective' / 'nextdesign_2wiki_run_objective_qa_summary_latest.json',
    root / '2wiki_run_objective' / 'nextdesign_2wiki_run_objective_retrieval_only_summary_latest.json',
)
ground = load_latest(
    root / '2wiki_grounding' / 'nextdesign_2wiki_grounding_qa_summary_latest.json',
    root / '2wiki_grounding' / 'nextdesign_2wiki_grounding_retrieval_only_summary_latest.json',
)
musique = load_latest(
    root / 'musique_failure_diagnosis' / 'nextdesign_musique_failure_qa_summary_latest.json',
    root / 'musique_failure_diagnosis' / 'nextdesign_musique_failure_retrieval_only_summary_latest.json',
)

lines = []
lines.append('# Next Method Design Final Interpretation')
lines.append('')

if run_obj:
    rows = run_obj.get('rows', [])
    best_f1 = None
    for r in rows:
        if r.get('variant') == 'baseline':
            continue
        if best_f1 is None or float(r.get('delta_f1', 0.0)) > float(best_f1.get('delta_f1', 0.0)):
            best_f1 = r
    lines.append('## 2Wiki Run-Objective Redesign')
    if best_f1 is not None:
        lines.append(
            f"- best variant by ΔF1: `{best_f1.get('variant')}` "
            f"(ΔF1={float(best_f1.get('delta_f1', 0.0)):+.4f}, "
            f"ΔRecall@20={float(best_f1.get('delta_recall_at_20', 0.0)):+.4f}, "
            f"Δtotal_ms={float(best_f1.get('delta_total_ms', 0.0)):+.2f})."
        )
    else:
        lines.append('- No non-baseline rows found.')
    lines.append('')

if ground:
    rows = ground.get('rows', [])
    best_f1 = None
    for r in rows:
        if r.get('variant') == 'baseline':
            continue
        if best_f1 is None or float(r.get('delta_f1', 0.0)) > float(best_f1.get('delta_f1', 0.0)):
            best_f1 = r
    lines.append('## 2Wiki Grounding Ablation')
    if best_f1 is not None:
        lines.append(
            f"- best variant by ΔF1: `{best_f1.get('variant')}` "
            f"(ΔF1={float(best_f1.get('delta_f1', 0.0)):+.4f}, "
            f"ΔRecall@20={float(best_f1.get('delta_recall_at_20', 0.0)):+.4f}, "
            f"Δtotal_ms={float(best_f1.get('delta_total_ms', 0.0)):+.2f})."
        )
    else:
        lines.append('- No non-baseline rows found.')
    lines.append('')

if musique:
    lines.append('## MuSiQue Failure Diagnosis')
    lines.append(
        f"- proposal_stage_drop_rate={float(musique.get('proposal_stage_drop_rate', 0.0)):.4f}, "
        f"run_selection_stage_drop_rate={float(musique.get('run_selection_stage_drop_rate', 0.0)):.4f}, "
        f"pair_refinement_stage_drop_rate={float(musique.get('pair_refinement_stage_drop_rate', 0.0)):.4f}, "
        f"final_selection_stage_drop_rate={float(musique.get('final_selection_stage_drop_rate', 0.0)):.4f}."
    )
    lines.append(
        f"- supporting_fact_recall={float(musique.get('supporting_fact_recall', 0.0)):.4f}, "
        f"rendered_recall={float(musique.get('rendered_recall', 0.0)):.4f}, "
        f"F1={float(musique.get('f1', 0.0)):.4f}."
    )
    lines.append('')

lines.append('## Answers To Core Questions')
lines.append('- Q1. Is run-level objective redesign more effective than seed-only correction on 2Wiki?')
lines.append('  Answer from latest run-objective summary: check best ΔF1/ΔRecall@20 variant above.')
lines.append('- Q2. Does entity-chunk grounding improve 2Wiki quality?')
lines.append('  Answer from latest grounding summary: check best grounding variant above.')
lines.append('- Q3. What is MuSiQue primary bottleneck?')
lines.append('  Answer from stage-drop rates above; highest drop stage is the current bottleneck candidate.')
lines.append('- Q4. What should be the next primary improvement axis?')
lines.append('  Use the stronger of (run-objective redesign vs grounding) under Recall@20/rendered-recall guardrails.')
lines.append('- Q5. How to integrate into operating strategy while preserving frozen defaults?')
lines.append('  Keep frozen defaults unchanged; promote only branch-isolated variants after stable q200+ verification.')
lines.append('')

out = root / 'nextdesign_final_interpretation.md'
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'[nextdesign_suite] wrote {out}')
PY

echo "[nextdesign_suite] done"
