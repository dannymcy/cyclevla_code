#!/bin/bash
# chmod +x experiments/robot/libero/run_libero_eval_decomposed_rule_progress_8-dim.sh
# ./experiments/robot/libero/run_libero_eval_decomposed_rule_progress_8-dim.sh

# Set the shared checkpoint path
CHECKPOINT_PATH="/hdd2/chenyang/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_8-dim_A100/openvla-7b-oft-finetuned-libero-spatial-object-goal-10+libero_decomposed_progress+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--30000_chkpt"

# Array of task suite names
TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# Run each task sequentially
for TASK in "${TASK_SUITES[@]}"; do
    echo "Running task: $TASK"
    CUDA_VISIBLE_DEVICES="2" python experiments/robot/libero/run_libero_eval_decomposed_rule_progress_8-dim.py \
        --pretrained_checkpoint "$CHECKPOINT_PATH" \
        --task_suite_name "$TASK"

    echo "Finished task: $TASK"
    echo "---------------------------------------------"
done

echo "All tasks completed."