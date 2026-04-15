#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/profiling/eval_parity}"
REPORT_MD="${REPORT_MD:-${OUT_DIR}/unit_test_report.md}"
LOG_TXT="${LOG_TXT:-${OUT_DIR}/unit_test_report.log}"

mkdir -p "${OUT_DIR}"

START_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
set +e
(
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" -m pytest tests/test_eval_parity.py tests/test_metrics.py -q
) >"${LOG_TXT}" 2>&1
RC=$?
set -e
END_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

{
  echo "# Eval Parity Unit Test Report"
  echo
  echo "- started_utc: ${START_UTC}"
  echo "- ended_utc: ${END_UTC}"
  echo "- command: \`${PYTHON_BIN} -m pytest tests/test_eval_parity.py tests/test_metrics.py -q\`"
  echo "- exit_code: ${RC}"
  echo "- log_path: \`${LOG_TXT}\`"
  echo
  echo "## Result"
  if [ "${RC}" -eq 0 ]; then
    echo "- status: PASS"
  else
    echo "- status: FAIL"
  fi
  echo
  echo "## Tail Log"
  echo '```text'
  tail -n 80 "${LOG_TXT}" || true
  echo '```'
} >"${REPORT_MD}"

if [ "${RC}" -ne 0 ]; then
  echo "[FAIL] Unit tests failed. See ${REPORT_MD}" >&2
  exit "${RC}"
fi

echo "[PASS] Unit tests passed. Report: ${REPORT_MD}"
