#!/bin/bash
# chmod +x experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.sh
# ./experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.sh

# Base checkpoint path (without the checkpoint number suffix)
CHECKPOINT_BASE="/hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state"

# Array of checkpoint numbers
CHECKPOINT_NUMS=(500000)

# Array of num_trials_per_task for each checkpoint (must match CHECKPOINT_NUMS length)
NUM_TRIALS_PER_TASK=(10)

# Array of rerun episodes for each checkpoint (must match CHECKPOINT_NUMS length)
# Use empty string "" to run all episodes for that checkpoint
# Checkpoints: 200000, 350000, 500000
RERUN_EPISODES_LIST=(
    # '{"libero_spatial": [55, 75], "libero_object": [1, 14], "libero_goal": [1, 43], "libero_10": [33, 71]}'
    # '{"libero_spatial": [75, 92], "libero_object": [3, 14], "libero_goal": [8, 42], "libero_10": [22, 36]}'
    '{"libero_spatial": [75, 92], "libero_object": [22, 79], "libero_goal": [1, 43], "libero_10": [22, 75]}'
)
# RERUN_EPISODES_LIST=("")

# Array of task suite names
TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
# TASK_SUITES=("libero_10")

# Loop through each checkpoint
for i in "${!CHECKPOINT_NUMS[@]}"; do
    CHKPT_NUM="${CHECKPOINT_NUMS[$i]}"
    NUM_TRIALS="${NUM_TRIALS_PER_TASK[$i]}"
    RERUN_EPISODES="${RERUN_EPISODES_LIST[$i]}"
    
    CHECKPOINT_PATH="${CHECKPOINT_BASE}--${CHKPT_NUM}_chkpt"
    LOCAL_LOG_DIR="./experiments/logs/logs_sub_decomposed_progress_transit_seed_${CHKPT_NUM}_chkpt"
    VIDEO_SAVE_DIR="./rollouts/rollouts_sub_decomposed_progress_transit_seed_${CHKPT_NUM}_chkpt"

    echo "============================================="
    echo "Processing checkpoint: ${CHKPT_NUM}"
    echo "Num trials per task: ${NUM_TRIALS}"
    echo "Rerun episodes: ${RERUN_EPISODES:-all}"
    echo "============================================="

    # Run each task sequentially
    for TASK in "${TASK_SUITES[@]}"; do
        echo "Running task: $TASK"
        
        if [ -n "$RERUN_EPISODES" ]; then
            CUDA_VISIBLE_DEVICES="3" python experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.py \
                --pretrained_checkpoint "$CHECKPOINT_PATH" \
                --task_suite_name "$TASK" \
                --local_log_dir "$LOCAL_LOG_DIR" \
                --video_save_dir "$VIDEO_SAVE_DIR" \
                --num_trials_per_task "$NUM_TRIALS" \
                --rerun_episodes "$RERUN_EPISODES"
        else
            CUDA_VISIBLE_DEVICES="3" python experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.py \
                --pretrained_checkpoint "$CHECKPOINT_PATH" \
                --task_suite_name "$TASK" \
                --local_log_dir "$LOCAL_LOG_DIR" \
                --video_save_dir "$VIDEO_SAVE_DIR" \
                --num_trials_per_task "$NUM_TRIALS"
        fi
        
        echo "Finished task: $TASK"
        echo "---------------------------------------------"
    done

    echo "Finished checkpoint: ${CHKPT_NUM}"
    echo ""
done

echo "All checkpoints and tasks completed."