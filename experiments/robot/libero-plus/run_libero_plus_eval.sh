#!/usr/bin/env bash
# =============================================================================
# Turnkey LIBERO-Plus CycleVLA evaluation (OpenVLA-OFT backbone) for ONE
# perturbation category on ONE GPU.
#
#   experiments/robot/libero-plus/run_libero_plus_eval.sh <gpu_id> <task_suite> <category> [eval_fraction]
#
# <task_suite> is a single suite (libero_spatial/libero_object/libero_goal/libero_10)
# OR the literal `all` -- `all` loops the 4 standard suites sequentially on the GPU.
#
# <category> is one of the 7 perturbation categories, or `all`:
#   camera | robot | language | light | background | noise | layout | all
#
# Examples:
#   # full Camera slice of libero_spatial on GPU 0
#   experiments/robot/libero-plus/run_libero_plus_eval.sh 0 libero_spatial camera
#   # 10%-subsampled Language category across ALL 4 suites on GPU 1
#   experiments/robot/libero-plus/run_libero_plus_eval.sh 0 all camera 10
#   experiments/robot/libero-plus/run_libero_plus_eval.sh 2 all language 10
#
# What it does, per suite (mirrors the two-stage LIBERO workflow in OPENVLA.md):
#   Stage 1 -- transit-only baseline; writes per-episode videos to $TRANSIT_DIR.
#   Stage 2 -- full method (VLM transit/backtrack + MBR); re-runs ONLY the
#              episodes Stage 1 failed (it scans $TRANSIT_DIR).
#              [currently commented out below -- transit-only for now]
# Both stages are given the SAME --category / --eval_fraction / --seed so their
# episode numbering lines up for the rerun mechanism.
#
# Run several of these on different GPUs (in the background) to evaluate
# multiple categories in parallel.
#
# Prerequisites:
#   conda activate openvla-oft-plus      # the LIBERO-Plus conda env (see LIBERO.md)
#   conda activate /hdd2/kai/openvla-oft/env-plus
#   OPENAI_API_KEY in the repo-root .env # used by the VLM detector + Language matcher
#                                        # (loaded via python-dotenv; see SETUP.md)
# =============================================================================
set -euo pipefail

GPU="${1:?usage: $0 <gpu_id> <task_suite|all> <category> [eval_fraction]}"
SUITE_ARG="${2:?usage: $0 <gpu_id> <task_suite|all> <category> [eval_fraction]}"
CATEGORY="${3:?usage: $0 <gpu_id> <task_suite|all> <category> [eval_fraction]}"
EVAL_FRACTION="${4:-100}"

# Repo root: this script lives in experiments/robot/libero-plus/.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO}"

# Trained CycleVLA checkpoint (same checkpoint as the LIBERO eval). Override with
# `CKPT=/path/to/other_chkpt experiments/robot/libero-plus/run_libero_plus_eval.sh ...`.
CKPT="${CKPT:-/hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--500000_chkpt}"

# Stage-1 transit videos live here; Stage 2 scans this dir for failed episodes.
TRANSIT_DIR="./rollouts-plus/rollouts_plus_decomposed_progress_transit"
MBR_DIR="./rollouts-plus/rollouts_plus_decomposed_progress_mbr"

# Resolve <task_suite>: `all` -> the 4 standard suites; otherwise the single named suite.
if [ "${SUITE_ARG}" = "all" ]; then
  SUITES=(libero_spatial libero_object libero_goal libero_10)
else
  SUITES=("${SUITE_ARG}")
fi

for SUITE in "${SUITES[@]}"; do
  echo "============================================================"
  echo "LIBERO-Plus eval | suite=${SUITE} category=${CATEGORY} frac=${EVAL_FRACTION}% | GPU ${GPU}"
  echo "============================================================"

  echo "[${SUITE}/${CATEGORY}] Stage 1 -- transit baseline"
  CUDA_VISIBLE_DEVICES="${GPU}" python \
    experiments/robot/libero-plus/run_libero_plus_eval_decomposed_progress_transit.py \
    --pretrained_checkpoint "${CKPT}" \
    --task_suite_name "${SUITE}" \
    --category "${CATEGORY}" \
    --eval_fraction "${EVAL_FRACTION}" \
    --video_save_dir "${TRANSIT_DIR}"

  # echo "[${SUITE}/${CATEGORY}] Stage 2 -- full method (transit/backtrack + MBR)"
  # CUDA_VISIBLE_DEVICES="${GPU}" python \
  #   experiments/robot/libero-plus/run_libero_plus_eval_decomposed_progress_mbr.py \
  #   --pretrained_checkpoint "${CKPT}" \
  #   --task_suite_name "${SUITE}" \
  #   --category "${CATEGORY}" \
  #   --eval_fraction "${EVAL_FRACTION}" \
  #   --video_save_dir "${MBR_DIR}" \
  #   --video_base_dir "${TRANSIT_DIR}"

  echo "[${SUITE}/${CATEGORY}] done"
done

# Refresh the per-category aggregate logs (7 per stage) that sum across the 4 suites.
# Idempotent and cheap; `|| true` so a parse hiccup never fails the eval run.
echo "[aggregate] refreshing AGGREGATE-*.txt per-category logs across suites"
python experiments/robot/libero-plus/aggregate_plus_logs.py || true
