#!/bin/bash
set -euo pipefail

# Generate next-phase operating strategy report for next_method_design family.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_PATH="${OUT_PATH:-outputs/profiling/next_method_design/next_phase_operating_strategy.md}"
mkdir -p "$(dirname "${OUT_PATH}")"

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

latest_or_na() {
  local pattern="$1"
  local hit
  hit="$(find outputs/profiling/next_method_design -type f -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}')"
  if [ -z "${hit}" ]; then
    echo "N/A"
  else
    echo "${hit}"
  fi
}

RUN_OBJECTIVE_LATEST="$(latest_or_na 'nextdesign_2wiki_run_objective_*_summary_latest.md')"
GROUNDING_LATEST="$(latest_or_na 'nextdesign_2wiki_grounding_*_summary_latest.md')"
MUSIQUE_LATEST="$(latest_or_na 'nextdesign_musique_failure_*_summary_latest.md')"

cat > "${OUT_PATH}" <<MD
# Next Phase Operating Strategy (Next Method Design)

## Metadata
- experiment_family: next_method_design
- git_branch: ${GIT_BRANCH}
- git_commit: ${GIT_COMMIT}
- git_tag: ${GIT_TAG}

## Why Frozen Defaults Stay Intact
1. Frozen operating points preserve reproducibility for system optimization and operating-point selection tracks.
2. New experiments in this branch target methodology changes, not production default replacement.
3. Config/script/output namespaces are isolated to avoid contaminating frozen baselines.

## Current Frozen Position (Reference)
- speed_default: aggressive
- quality_variant: top1corr_t1
- off: ablation/reference

## Dataset-Specific Current References
- HotpotQA quality candidate: semantic_union
- 2Wiki quality candidate: baseline
- 2Wiki retrieval-speed candidate: bridge_seed_score_up
- MuSiQue: failure-diagnosis-first strategy

## Excluded Axes (For This Phase)
- semantic top-N expansion reruns
- support/main sentence expansion reruns
- Hotpot additional sweeps

## Next Primary Axes
1. 2Wiki run-level objective redesign
2. 2Wiki entity-chunk evidence grounding
3. MuSiQue failure-mode diagnosis

## Latest Artifacts
- 2Wiki run-objective summary: ${RUN_OBJECTIVE_LATEST}
- 2Wiki grounding summary: ${GROUNDING_LATEST}
- MuSiQue diagnosis summary: ${MUSIQUE_LATEST}

## Integration Rule Back To Operations
1. Keep frozen defaults unchanged until next-method-design variants show stable gains over baseline on q200+.
2. Promote only variants that keep Recall@20/rendered recall stable or higher while improving EM/F1 with acceptable latency.
3. Separate conclusions by dataset; do not force a single global variant if behaviors diverge.
MD

echo "[nextdesign] strategy report generated: ${OUT_PATH}"
