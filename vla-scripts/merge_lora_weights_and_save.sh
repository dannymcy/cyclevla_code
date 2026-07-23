#!/usr/bin/env bash
# conda activate /hdd2/chenyang/openvla-oft/env
# chmod +x vla-scripts/merge_lora_weights_and_save.sh
# ./vla-scripts/merge_lora_weights_and_save.sh

########################################
# User-configurable variables
########################################

# Base model checkpoint (change if you used a different base)
BASE_CHECKPOINT="openvla/openvla-7b"

# Common prefix of your checkpoint directories, i.e. everything
# BEFORE the "--50000_chkpt" suffix
CKPT_PREFIX="/hdd2/chenyang/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state"

# GPU id you want to use in CUDA_VISIBLE_DEVICES
GPU_ID=0

########################################
# Loop over checkpoints
########################################

for STEP in $(seq 500000 50000 500000); do
    CKPT_DIR="${CKPT_PREFIX}--${STEP}_chkpt"

    echo "============================================================"
    echo "Merging LoRA weights for checkpoint step ${STEP}"
    echo "Checkpoint directory: ${CKPT_DIR}"
    echo "Base checkpoint:      ${BASE_CHECKPOINT}"
    echo "GPU:                  ${GPU_ID}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint "${BASE_CHECKPOINT}" \
        --lora_finetuned_checkpoint_dir "${CKPT_DIR}"

    echo "Done merging step ${STEP}"
    echo
done

echo "All checkpoints processed."
