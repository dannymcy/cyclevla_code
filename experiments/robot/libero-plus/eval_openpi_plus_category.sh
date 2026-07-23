#!/usr/bin/env bash
# =============================================================================
# Turnkey openpi (pi0.5) CycleVLA evaluation for ONE LIBERO-Plus perturbation
# category across all 4 suites, on ONE GPU. Brings up the policy server, runs
# the full two-stage eval, then tears the server down.
#
#   experiments/robot/libero-plus/eval_openpi_plus_category.sh <gpu_id> <port> <category> [eval_fraction]
#
# Examples:
#   experiments/robot/libero-plus/eval_openpi_plus_category.sh 2 8002 camera
#   experiments/robot/libero-plus/eval_openpi_plus_category.sh 3 8004 language 10
#
# What it does:
#   1. Launches the openpi policy server (pi0.5) on GPU <gpu_id>, port <port>,
#      with JAX memory settings for multi-server-per-GPU (PREALLOCATE=false,
#      MEM_FRACTION=0.45).
#   2. Waits for the server to be ready (polls /healthz, up to ~20 min).
#   3. For each of the 4 LIBERO suites (spatial, object, goal, 10):
#        Stage 1 -- transit-only baseline eval.
#        Stage 2 -- full-method eval (re-runs only the episodes Stage 1 failed).
#   4. Kills the server (via an EXIT trap, so it dies even on Ctrl-C / error).
#
# Run up to 2 of these per GPU (different ports) to evaluate multiple categories
# in parallel. Each server uses ~9GB VRAM; two fit on a 23GB A10.
#
# Prerequisites:
#   conda activate /hdd2/chenyang/openvla-oft/env-plus
#   OPENAI_API_KEY in the repo-root .env
# =============================================================================
set -euo pipefail

GPU="${1:?usage: $0 <gpu_id> <port> <category> [eval_fraction]}"
PORT="${2:?usage: $0 <gpu_id> <port> <category> [eval_fraction]}"
CATEGORY="${3:?usage: $0 <gpu_id> <port> <category> [eval_fraction]}"
EVAL_FRACTION="${4:-100}"

# LIBERO-Plus needs its own config path (separate bddl/init/asset paths).
export LIBERO_CONFIG_PATH="${HOME}/.libero-plus"

# JAX memory: allow multiple servers per GPU without preallocation.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

# Repo root: this script lives in experiments/robot/libero-plus/.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO}"

EVAL_PY="${REPO}/env-plus/bin/python"
SERVE_SH="${REPO}/openpi/scripts/serve_openpi_cyclevla.sh"
SERVER_LOG="${REPO}/experiments/logs/serve_openpi_plus_${CATEGORY}_port${PORT}.log"
mkdir -p "$(dirname "$SERVER_LOG")"

TRANSIT_DIR="./rollouts-plus/rollouts_plus_openpi_transit"
CYCLEVLA_DIR="./rollouts-plus/rollouts_plus_openpi_cyclevla"

echo "[${CATEGORY}] launching openpi server on GPU ${GPU}, port ${PORT} (log: ${SERVER_LOG})"
# `setsid` puts the server in its own process group so we can kill the whole
# tree (uv + serve_policy.py) with a single `kill -- -<pgid>`.
CUDA_VISIBLE_DEVICES="${GPU}" PORT="${PORT}" setsid "${SERVE_SH}" > "${SERVER_LOG}" 2>&1 &
SERVER_PGID=$!
trap 'kill -- -"${SERVER_PGID}" 2>/dev/null || true' EXIT INT TERM

# --- Wait for the server to be ready -----------------------------------------
if command -v curl >/dev/null 2>&1; then
  echo "[${CATEGORY}] waiting for server to be ready..."
  for _ in $(seq 1 240); do                 # up to ~20 min for model load
    if curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1; then
      echo "[${CATEGORY}] server ready on port ${PORT}"
      break
    fi
    if ! kill -0 "${SERVER_PGID}" 2>/dev/null; then
      echo "[${CATEGORY}] ERROR: server exited before becoming ready; see ${SERVER_LOG}" >&2
      exit 1
    fi
    sleep 5
  done
else
  echo "[${CATEGORY}] curl not found; the eval client will wait for the server itself"
fi

# --- Two-stage eval across all 4 suites --------------------------------------
SUITES=(libero_spatial libero_object libero_goal libero_10)

for SUITE in "${SUITES[@]}"; do
  echo "============================================================"
  echo "LIBERO-Plus openpi eval | suite=${SUITE} category=${CATEGORY} frac=${EVAL_FRACTION}% | GPU ${GPU} port ${PORT}"
  echo "============================================================"

  echo "[${SUITE}/${CATEGORY}] Stage 1 -- openpi transit baseline"
  CUDA_VISIBLE_DEVICES="${GPU}" "${EVAL_PY}" \
    experiments/robot/libero-plus/run_libero_plus_eval_openpi_transit.py \
    --host 0.0.0.0 --port "${PORT}" \
    --task_suite_name "${SUITE}" \
    --category "${CATEGORY}" \
    --eval_fraction "${EVAL_FRACTION}" \
    --video_save_dir "${TRANSIT_DIR}"

  echo "[${SUITE}/${CATEGORY}] Stage 2 -- openpi full method (transit/backtrack + MBR)"
  CUDA_VISIBLE_DEVICES="${GPU}" "${EVAL_PY}" \
    experiments/robot/libero-plus/run_libero_plus_eval_openpi_cyclevla.py \
    --host 0.0.0.0 --port "${PORT}" \
    --task_suite_name "${SUITE}" \
    --category "${CATEGORY}" \
    --eval_fraction "${EVAL_FRACTION}" \
    --video_save_dir "${CYCLEVLA_DIR}" \
    --video_base_dir "${TRANSIT_DIR}"

  echo "[${SUITE}/${CATEGORY}] done"
done

# Refresh aggregate logs
echo "[aggregate] refreshing AGGREGATE-*.txt per-category logs across suites"
"${EVAL_PY}" experiments/robot/libero-plus/aggregate_plus_logs.py \
  --log_dir ./rollouts-plus/logs_plus_openpi_transit --stage transit || true
"${EVAL_PY}" experiments/robot/libero-plus/aggregate_plus_logs.py \
  --log_dir ./rollouts-plus/logs_plus_openpi_cyclevla --stage cyclevla || true

echo "[${CATEGORY}] ALL DONE -- tearing down server"
# server is killed by the EXIT trap
