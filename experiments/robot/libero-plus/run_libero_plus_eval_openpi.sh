#!/usr/bin/env bash
# =============================================================================
# Turnkey LIBERO-Plus CycleVLA evaluation (openpi / pi0.5 backbone) for ONE
# perturbation category on ONE GPU.
#
#   experiments/robot/libero-plus/run_libero_plus_eval_openpi.sh <gpu_id> <task_suite|all> <category> [eval_fraction] [host] [port]
#
# <task_suite> is a single suite (libero_spatial/libero_object/libero_goal/libero_10)
# OR the literal `all` -- `all` loops the 4 standard suites sequentially on the GPU.
#
# <category> is one of the 7 perturbation categories, or `all`:
#   camera | robot | language | light | background | noise | layout | all
#
# [host] and [port] default to 0.0.0.0 and 8000.
#
# Examples:
#   # full Camera slice of libero_spatial on GPU 0, server on localhost:8000
#   experiments/robot/libero-plus/run_libero_plus_eval_openpi.sh 0 libero_spatial camera
#   # 10%-subsampled Language category across ALL 4 suites, server on port 8001
#   experiments/robot/libero-plus/run_libero_plus_eval_openpi.sh 0 all language 10 0.0.0.0 8001
#
# What it does, per suite (mirrors the two-stage LIBERO workflow):
#   Stage 1 -- transit-only baseline; writes per-episode videos to $TRANSIT_DIR.
#   Stage 2 -- full method (VLM transit/backtrack + MBR); re-runs ONLY the
#              episodes Stage 1 failed (it scans $TRANSIT_DIR).
# Both stages are given the SAME --category / --eval_fraction / --seed so their
# episode numbering lines up for the rerun mechanism.
#
# This is a client-only script -- the openpi policy server must already be
# running (see `openpi/scripts/serve_openpi_cyclevla.sh`).
#
# Prerequisites:
#   conda activate /hdd2/chenyang/openvla-oft/env-plus
#   # Start server in another terminal:
#   CUDA_VISIBLE_DEVICES=0 PORT=8000 openpi/scripts/serve_openpi_cyclevla.sh
#   # OPENAI_API_KEY in .env for the VLM detector + Language matcher
# =============================================================================
set -euo pipefail

# LIBERO-Plus and stock LIBERO share the same Python namespace but need
# different bddl/init/asset paths.  Point at the Plus-specific config so both
# envs can run simultaneously.
export LIBERO_CONFIG_PATH="${HOME}/.libero-plus"

GPU="${1:?usage: $0 <gpu_id> <task_suite|all> <category> [eval_fraction] [host] [port]}"
SUITE_ARG="${2:?usage: $0 <gpu_id> <task_suite|all> <category> [eval_fraction] [host] [port]}"
CATEGORY="${3:?usage: $0 <gpu_id> <task_suite|all> <category> [eval_fraction] [host] [port]}"
EVAL_FRACTION="${4:-100}"
HOST="${5:-0.0.0.0}"
PORT="${6:-8000}"

# Repo root: this script lives in experiments/robot/libero-plus/.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO}"

# Stage-1 transit videos live here; Stage 2 scans this dir for failed episodes.
TRANSIT_DIR="./rollouts-plus/rollouts_plus_openpi_transit"
CYCLEVLA_DIR="./rollouts-plus/rollouts_plus_openpi_cyclevla"

# Resolve <task_suite>: `all` -> the 4 standard suites; otherwise the single named suite.
if [ "${SUITE_ARG}" = "all" ]; then
  SUITES=(libero_spatial libero_object libero_goal libero_10)
else
  SUITES=("${SUITE_ARG}")
fi

for SUITE in "${SUITES[@]}"; do
  echo "============================================================"
  echo "LIBERO-Plus openpi eval | suite=${SUITE} category=${CATEGORY} frac=${EVAL_FRACTION}% | GPU ${GPU}"
  echo "============================================================"

  echo "[${SUITE}/${CATEGORY}] Stage 1 -- openpi transit baseline"
  CUDA_VISIBLE_DEVICES="${GPU}" python \
    experiments/robot/libero-plus/run_libero_plus_eval_openpi_transit.py \
    --host "${HOST}" --port "${PORT}" \
    --task_suite_name "${SUITE}" \
    --category "${CATEGORY}" \
    --eval_fraction "${EVAL_FRACTION}" \
    --video_save_dir "${TRANSIT_DIR}"

  echo "[${SUITE}/${CATEGORY}] Stage 2 -- openpi full method (transit/backtrack + MBR)"
  CUDA_VISIBLE_DEVICES="${GPU}" python \
    experiments/robot/libero-plus/run_libero_plus_eval_openpi_cyclevla.py \
    --host "${HOST}" --port "${PORT}" \
    --task_suite_name "${SUITE}" \
    --category "${CATEGORY}" \
    --eval_fraction "${EVAL_FRACTION}" \
    --video_save_dir "${CYCLEVLA_DIR}" \
    --video_base_dir "${TRANSIT_DIR}"

  echo "[${SUITE}/${CATEGORY}] done"
done

# Refresh the per-category aggregate logs across suites.
echo "[aggregate] refreshing AGGREGATE-*.txt per-category logs across suites"
python experiments/robot/libero-plus/aggregate_plus_logs.py \
  --log_dir ./rollouts-plus/logs_plus_openpi_transit --stage transit || true
python experiments/robot/libero-plus/aggregate_plus_logs.py \
  --log_dir ./rollouts-plus/logs_plus_openpi_cyclevla --stage cyclevla || true
