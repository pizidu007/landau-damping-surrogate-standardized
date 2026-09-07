#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${PROJECT_ROOT}/results/continuum_v1_macrostep_round10/formal"
CONTROL_DIR="${RESULT_ROOT}/_automation"
DEVICE="${1:-cuda:1}"
PYTHON_BIN="/wangx/home/duxinxu/miniconda3/envs/landau-pic-surrogate/bin/python"
SCREEN_SESSION="landau_round10_auto"

mkdir -p "${CONTROL_DIR}"
screen -wipe >/dev/null 2>&1 || true
SCREEN_LIST="$(screen -ls 2>&1 || true)"
if grep -q "[.]${SCREEN_SESSION}[[:space:]]" <<<"${SCREEN_LIST}"; then
  echo "Round 10 automation is already running in screen session ${SCREEN_SESSION}" >&2
  exit 1
fi

(
  cd "${CONTROL_DIR}"
  screen -L -DmS "${SCREEN_SESSION}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_round10_automatic.py" \
    --device "${DEVICE}" </dev/null >/dev/null 2>&1 &
)

SCREEN_PID=""
for _ in {1..20}; do
  SCREEN_LIST="$(screen -ls 2>&1 || true)"
  SCREEN_PID="$(awk -v name=".${SCREEN_SESSION}" '$1 ~ name {split($1, value, "."); print value[1]; exit}' <<<"${SCREEN_LIST}")"
  [[ -n "${SCREEN_PID}" ]] && break
  sleep 0.25
done
if [[ -z "${SCREEN_PID}" ]]; then
  echo "Failed to create screen session ${SCREEN_SESSION}" >&2
  exit 1
fi
echo "${SCREEN_PID}" >"${CONTROL_DIR}/launcher.pid"
echo "Round 10 automation started in screen session ${SCREEN_SESSION} (PID ${SCREEN_PID})"
echo "Status: ${CONTROL_DIR}/status.json"
echo "Runner log: ${CONTROL_DIR}/screenlog.0"
echo "Task logs: ${CONTROL_DIR}/logs"
