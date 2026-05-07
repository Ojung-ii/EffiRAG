#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash stop_vllm.sh --port 8011
#   bash stop_vllm.sh --port 8012
#   PORT=8011 bash stop_vllm.sh
#   PORT=8012 bash stop_vllm.sh
# Optional:
#   bash stop_vllm.sh --port 8011 --timeout 12

PORT="${PORT:-8011}"
TIMEOUT="${TIMEOUT:-10}"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage: bash stop_vllm.sh [options]

Options:
  --port PORT       vLLM API port to stop (default: $PORT or 8011)
  --timeout SEC     Graceful TERM wait seconds before KILL (default: 10)
  --dry-run         Print targets only, do not kill
  -h, --help        Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[stop_vllm] unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if ! [[ "${PORT}" =~ ^[0-9]+$ ]]; then
  echo "[stop_vllm] invalid port: ${PORT}" >&2
  exit 1
fi
if ! [[ "${TIMEOUT}" =~ ^[0-9]+$ ]]; then
  echo "[stop_vllm] invalid timeout: ${TIMEOUT}" >&2
  exit 1
fi

log() {
  echo "[stop_vllm] $*"
}

cmdline_of() {
  local pid="$1"
  ps -p "${pid}" -o args= 2>/dev/null || true
}

owner_of() {
  local pid="$1"
  ps -p "${pid}" -o user= 2>/dev/null | awk '{print $1}' || true
}

declare -A PID_SET
declare -A DESC_VISITED

add_pid() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then
    return
  fi
  if ! [[ "${pid}" =~ ^[0-9]+$ ]]; then
    return
  fi
  PID_SET["${pid}"]=1
}

collect_descendants() {
  local parent="$1"
  if [[ -z "${parent}" ]]; then
    return
  fi
  if ! [[ "${parent}" =~ ^[0-9]+$ ]]; then
    return
  fi
  if [[ -n "${DESC_VISITED[${parent}]:-}" ]]; then
    return
  fi
  DESC_VISITED["${parent}"]=1
  local kids=()
  mapfile -t kids < <(pgrep -P "${parent}" 2>/dev/null || true)
  for kid in "${kids[@]:-}"; do
    add_pid "${kid}"
    collect_descendants "${kid}"
  done
}

has_tracked_ancestor() {
  local pid="${1:-}"
  local hops=0
  while [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ && "${pid}" != "1" && ${hops} -lt 40 ]]; do
    if [[ -n "${PID_SET[${pid}]:-}" ]]; then
      return 0
    fi
    pid="$(ps -p "${pid}" -o ppid= 2>/dev/null | awk '{print $1}' || true)"
    hops=$((hops + 1))
  done
  return 1
}

listener_pids=()
if command -v lsof >/dev/null 2>&1; then
  mapfile -t listener_pids < <(lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)
fi
for pid in "${listener_pids[@]:-}"; do
  add_pid "${pid}"
done

candidate_pids=()
mapfile -t candidate_pids < <(
  pgrep -f "vllm( |$)|vllm\\.entrypoints\\.openai\\.api_server|python.*vllm" 2>/dev/null || true
)
for pid in "${candidate_pids[@]:-}"; do
  cmd="$(cmdline_of "${pid}")"
  [[ -z "${cmd}" ]] && continue
  if [[ "${cmd}" == *"--port ${PORT}"* ]] || [[ "${cmd}" == *"--port=${PORT}"* ]] || [[ "${cmd}" == *":${PORT}"* ]]; then
    add_pid "${pid}"
  fi
done

initial_targets=("${!PID_SET[@]}")
for pid in "${initial_targets[@]:-}"; do
  collect_descendants "${pid}"
done

# Safety: include ray/stale python processes only when they descend from tracked vLLM targets.
ray_candidate_pids=()
mapfile -t ray_candidate_pids < <(
  pgrep -f "ray::|raylet|gcs_server|python.*ray" 2>/dev/null || true
)
for pid in "${ray_candidate_pids[@]:-}"; do
  if has_tracked_ancestor "${pid}"; then
    add_pid "${pid}"
    collect_descendants "${pid}"
  fi
done

targets=()
for pid in "${!PID_SET[@]}"; do
  owner="$(owner_of "${pid}")"
  [[ -z "${owner}" ]] && continue
  if [[ "${owner}" != "${USER}" ]]; then
    continue
  fi
  targets+=("${pid}")
done

if [[ ${#targets[@]} -eq 0 ]]; then
  log "no vLLM targets found for port ${PORT}"
else
  log "target port=${PORT}, timeout=${TIMEOUT}s"
  for pid in "${targets[@]}"; do
    log "target pid=${pid} cmd=$(cmdline_of "${pid}")"
  done
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  log "dry-run enabled; no process will be killed."
else
  if [[ ${#targets[@]} -gt 0 ]]; then
    for pid in "${targets[@]}"; do
      kill -TERM "${pid}" 2>/dev/null || true
    done

    deadline=$((SECONDS + TIMEOUT))
    while (( SECONDS < deadline )); do
      alive=()
      for pid in "${targets[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          alive+=("${pid}")
        fi
      done
      if [[ ${#alive[@]} -eq 0 ]]; then
        break
      fi
      sleep 1
    done

    for pid in "${targets[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    done
  fi
fi

log "post-check: lsof -i :${PORT}"
if command -v lsof >/dev/null 2>&1; then
  lsof -i :"${PORT}" 2>/dev/null || true
else
  log "lsof not found"
fi

log "post-check: ps aux | grep -E 'vllm|ray::|raylet|gcs_server'"
ps aux \
  | grep -E "vllm( |$)|vllm\\.entrypoints\\.openai\\.api_server|ray::|raylet|gcs_server" \
  | grep -v grep \
  | grep -v "stop_vllm.sh" \
  || true

log "post-check: nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  if command -v timeout >/dev/null 2>&1; then
    timeout 5 nvidia-smi || true
  else
    nvidia-smi || true
  fi
else
  log "nvidia-smi not found"
fi

log "done"
