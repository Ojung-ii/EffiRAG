#!/usr/bin/env bash
set -euo pipefail

python3 -m effirag.run_retrieval --config configs/retrieval.yaml "$@"
