#!/bin/bash
# chmod +x experiments/robot/libero/run_libero_eval_original.sh
# ./experiments/robot/libero/run_libero_eval_original.sh

# Set the shared checkpoint path
CHECKPOINT_PATH="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"

# Array of task suite names
TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# Run each task sequentially
for TASK in "${TASK_SUITES[@]}"; do
    echo "Running task: $TASK"
    CUDA_VISIBLE_DEVICES="2" python experiments/robot/libero/run_libero_eval_original.py \
        --pretrained_checkpoint "$CHECKPOINT_PATH" \
        --task_suite_name "$TASK"

    echo "Finished task: $TASK"
    echo "---------------------------------------------"
done

echo "All tasks completed."