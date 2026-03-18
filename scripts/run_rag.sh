#!/usr/bin/env bash
set -euo pipefail

python3 -m effirag.run_rag --config configs/rag.yaml "$@"
