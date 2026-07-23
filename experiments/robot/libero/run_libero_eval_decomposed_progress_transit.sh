#!/bin/bash
# conda activate /hdd2/chenyang/openvla-oft/env
# chmod +x experiments/robot/libero/run_libero_eval_decomposed_progress_transit.sh
# ./experiments/robot/libero/run_libero_eval_decomposed_progress_transit.sh

# Base checkpoint path (without the checkpoint number suffix)
CHECKPOINT_BASE="/hdd2/chenyang/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state"

# Array of checkpoint numbers
CHECKPOINT_NUMS=(500000)

# Array of num_trials_per_task for each checkpoint (must match CHECKPOINT_NUMS length)
NUM_TRIALS_PER_TASK=(10)

# Array of task suite names
TASK_SUITES=("libero_spatial" "libero_10")
# TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# Loop through each checkpoint
for i in "${!CHECKPOINT_NUMS[@]}"; do
    CHKPT_NUM="${CHECKPOINT_NUMS[$i]}"
    NUM_TRIALS="${NUM_TRIALS_PER_TASK[$i]}"
    
    CHECKPOINT_PATH="${CHECKPOINT_BASE}--${CHKPT_NUM}_chkpt"
    LOCAL_LOG_DIR="./experiments/logs/logs_sub_decomposed_progress_transit_${CHKPT_NUM}_chkpt"
    VIDEO_SAVE_DIR="./rollouts/rollouts_sub_decomposed_progress_transit_${CHKPT_NUM}_chkpt"

    echo "============================================="
    echo "Processing checkpoint: ${CHKPT_NUM}"
    echo "Num trials per task: ${NUM_TRIALS}"
    echo "============================================="

    # Run each task sequentially
    for TASK in "${TASK_SUITES[@]}"; do
        echo "Running task: $TASK"
        CUDA_VISIBLE_DEVICES="2" python experiments/robot/libero/run_libero_eval_decomposed_progress_transit.py \
            --pretrained_checkpoint "$CHECKPOINT_PATH" \
            --task_suite_name "$TASK" \
            --local_log_dir "$LOCAL_LOG_DIR" \
            --video_save_dir "$VIDEO_SAVE_DIR" \
            --num_trials_per_task "$NUM_TRIALS"

        echo "Finished task: $TASK"
        echo "---------------------------------------------"
    done

    echo "Finished checkpoint: ${CHKPT_NUM}"
    echo ""
done

echo "All checkpoints and tasks completed."