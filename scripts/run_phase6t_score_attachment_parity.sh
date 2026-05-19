#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper kept for command compatibility in Phase-6T parity-repair runs.
cd /home/ojungii/EffiRAG
exec bash scripts/run_phase6t_candidate_score_parity_gpu1.sh "$@"
