#!/usr/bin/env bash
# =============================================================================
# Turnkey openpi (pi0.5) CycleVLA evaluation for ONE LIBERO task suite on ONE
# GPU. Brings up the policy server, runs the full two-stage eval, then tears the
# server down.
#
#   experiments/robot/libero/eval_openpi_suite.sh <gpu_id> <port> <task_suite>
#
# Example:
#   experiments/robot/libero/eval_openpi_suite.sh 0 8000 libero_spatial
#
# What it does:
#   1. Launches the openpi policy server (pi0.5) on GPU <gpu_id>, port <port>,
#      in the background (its own process group, so teardown is clean).
#   2. Waits for the server to be ready (or aborts if it dies early).
#   3. Stage 1 -- transit-only baseline eval.
#   4. Stage 2 -- full-method eval (re-runs only the episodes Stage 1 failed).
#   5. Kills the server (via an EXIT trap, so it dies even on Ctrl-C / error).
#
# The server runs the JAX model (heavy GPU work); the eval client only renders
# the LIBERO sim. Both are pinned to <gpu_id>.
#
# Run several of these on different GPUs/ports (in the background) to evaluate
# multiple task suites in parallel.
# =============================================================================
set -euo pipefail

GPU="${1:?usage: $0 <gpu_id> <port> <task_suite>}"
PORT="${2:?usage: $0 <gpu_id> <port> <task_suite>}"
SUITE="${3:?usage: $0 <gpu_id> <port> <task_suite>}"

# Repo root: this script lives in experiments/robot/libero/.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
EVAL_PY="${REPO}/env/bin/python"            # the openvla-oft conda env
SERVE_SH="${REPO}/openpi/scripts/serve_openpi_cyclevla.sh"
SERVER_LOG="${REPO}/experiments/logs/serve_openpi_${SUITE}_port${PORT}.log"
mkdir -p "$(dirname "$SERVER_LOG")"

echo "[${SUITE}] launching openpi server on GPU ${GPU}, port ${PORT} (log: ${SERVER_LOG})"
# `setsid` puts the server in its own process group so we can kill the whole
# tree (uv + serve_policy.py) with a single `kill -- -<pgid>`.
CUDA_VISIBLE_DEVICES="${GPU}" PORT="${PORT}" setsid "${SERVE_SH}" > "${SERVER_LOG}" 2>&1 &
SERVER_PGID=$!
trap 'kill -- -"${SERVER_PGID}" 2>/dev/null || true' EXIT INT TERM

# --- Wait for the server to be ready -----------------------------------------
# The server only starts listening after the JAX model is fully loaded, so a
# successful /healthz (or websocket connect) means "ready". Bail if it dies.
if command -v curl >/dev/null 2>&1; then
  echo "[${SUITE}] waiting for server to be ready..."
  for _ in $(seq 1 240); do                 # up to ~20 min for model load
    if curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1; then
      echo "[${SUITE}] server ready"
      break
    fi
    if ! kill -0 "${SERVER_PGID}" 2>/dev/null; then
      echo "[${SUITE}] ERROR: server exited before becoming ready; see ${SERVER_LOG}" >&2
      exit 1
    fi
    sleep 5
  done
else
  echo "[${SUITE}] curl not found; the eval client will wait for the server itself"
fi

# --- Two-stage eval (client only renders the sim; pinned to the same GPU) ----
cd "${REPO}"

echo "[${SUITE}] Stage 1 -- transit baseline"
CUDA_VISIBLE_DEVICES="${GPU}" "${EVAL_PY}" \
  experiments/robot/libero/run_libero_eval_openpi_transit.py \
  --host 0.0.0.0 --port "${PORT}" --task_suite_name "${SUITE}"

echo "[${SUITE}] Stage 2 -- full method (transit/backtrack + MBR)"
CUDA_VISIBLE_DEVICES="${GPU}" "${EVAL_PY}" \
  experiments/robot/libero/run_libero_eval_openpi_cyclevla.py \
  --host 0.0.0.0 --port "${PORT}" --task_suite_name "${SUITE}"

echo "[${SUITE}] done -- tearing down server"
# server is killed by the EXIT trap
