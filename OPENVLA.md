# CycleVLA in the LIBERO and LIBERO-Plus Simulation Benchmark Using OpenVLA Backbone

## Relevant Files

Evaluation
* `experiments/robot/libero/`: LIBERO eval files
  * `run_libero_eval_decomposed_progress_mbr.py`: Official CycleVLA eval pipeline — 9-dim policy + VLM `transit/backtrack` decision at ~90% subtask progress + MBR seed-sampling on backtrack.
  * `run_libero_eval_decomposed_progress_transit.py`: Baseline transit-only eval; no VLM, no MBR. Measures raw subtask-transit success of the trained 9-dim policy.
  * `run_libero_eval_decomposed_progress_transit_seed.py`: Seed-sweep harness; same loop as `_transit` plus per-seed `trajectory_analysis_*.xlsx` dumps under `rollouts/.../` for `run_mbr_analysis.py`.
  * `run_mbr_analysis.py`: Aggregates the seed-sweep xlsx files across checkpoint folders and tasks; emits Random / MBR / r-NN statistics matrices over N ∈ {4, 8, 16, 32, 64} and distance metrics ∈ {L1, L2, Chebyshev, cosine, correlation}.
  * `libero_utils.py`: LIBERO env / image / state utilities reused by every eval script above.
* `experiments/robot/`: General eval utils files
  * `openvla_utils.py`: OpenVLA-specific eval utils
  * `robot_utils.py`: Other eval utils

Training
* `vla-scripts/finetune_progress.py`: Official 9-dim finetune entrypoint (loss-metric tracking is split for the stop signal `s_t` at action index 7 and the progress signal `p_t` at index 8).
* `vla-scripts/merge_lora_weights_and_save.py`: Merge the trained LoRA adapter back into the base OpenVLA weights; run on the downstream device used for inference to avoid the train-vs-test GPU performance drop.


## Finetuning on LIBERO Datasets

Launch the finetuning script with the configuration below. This is the hyperparameter set used in the paper — we trained a single checkpoint covering all four LIBERO task suites, on a rented 4×A100 server (adjust the literal paths to match your environment). The pretrained OFT checkpoint we re-finetune from is `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`.

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_progress.py \
  --vla_path moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --data_root_dir "/home/to0space/ygy/openvla-oft/decomposed_dataset/libero_sub_progress/" \
  --dataset_name libero_decomposed_progress \
  --run_root_dir "/home/to0space/ygy/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/" \
  --use_l1_regression False \
  --use_diffusion True \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 500005 \
  --save_freq 25000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --grad_accumulation_steps 4 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "CycleVLA_libero_sub_decomposed_progress_oft_A100" \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state
```

After training, merge the LoRA adapter into the base OpenVLA weights before running eval. Point `--lora_adapter_dir` at one of the saved `*_chkpt` folders under `run_root_dir`:

```bash
python vla-scripts/merge_lora_weights_and_save.py \
  --base_vla_path moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --lora_adapter_dir /home/to0space/ygy/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b-oft-finetuned-libero-spatial-object-goal-10+libero_decomposed_progress+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--50000_chkpt
```

Please be sure to test your policy with the same device/GPU used to train it! Otherwise, performance may drop substantially. You may be able to avoid the performance drop if you merge the LoRA weights into the base model on the downstream device used for testing (e.g., if you train on H100 and then merge on A100 before testing on A100). You can see our script [vla-scripts/merge_lora_weights_and_save.py](vla-scripts/merge_lora_weights_and_save.py) for merging the LoRA adapter into the base model offline. It's okay if you already merged LoRA weights into the base OpenVLA model during finetuning; you can always redownload the base model and merge again as long as you still have the LoRA adapter (`merge_lora_weights_and_save.py` will handle this for you).


## Launching LIBERO Evaluations

Our trained CycleVLA checkpoint for LIBERO will be released here:
* Coming soon.

Each of the four task suites uses the **same** unified checkpoint; only `--task_suite_name` changes between runs. You can set the `TRANSFORMERS_CACHE` and `HF_HOME` environment variables to change where the checkpoint files get cached. The checkpoint path below is the literal one this repo evaluates against (the `50000_chkpt` step from the A100 run); swap in any other `*_chkpt` folder from the same `run_root_dir` to evaluate a different training step.

```bash
# Shared checkpoint (single policy for all four task suites)
CKPT=/hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--50000_chkpt

# LIBERO-Spatial
CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero/run_libero_eval_decomposed_progress_mbr.py \
  --pretrained_checkpoint $CKPT --task_suite_name libero_spatial

# LIBERO-Object
CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero/run_libero_eval_decomposed_progress_mbr.py \
  --pretrained_checkpoint $CKPT --task_suite_name libero_object

# LIBERO-Goal
CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero/run_libero_eval_decomposed_progress_mbr.py \
  --pretrained_checkpoint $CKPT --task_suite_name libero_goal

# LIBERO-10 (LIBERO-Long)
CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero/run_libero_eval_decomposed_progress_mbr.py \
  --pretrained_checkpoint $CKPT --task_suite_name libero_10
```

Notes:
* By default the script runs 100 rollouts per task suite (10 tasks × 10 trials each); the trial count is `num_trials_per_task` in `run_libero_eval_decomposed_progress_mbr.py`. Bump `--num_trials_per_task` for higher-confidence numbers, and change the random seed via `--seed`. There are other arguments in the script; we use the defaults that work with the CycleVLA checkpoint above.
* **NOTE: Setting `--center_crop True` is important** because we finetuned OpenVLA-OFT with random crop augmentations (we took a random crop with 90% area in every training sample, so at test time we simply take the center 90% crop). The script asserts this assumption.
* The evaluation script logs results locally. You can also log results in Weights & Biases by setting `--use_wandb True` and specifying `--wandb_project <PROJECT>` and `--wandb_entity <ENTITY>`.
* We use the custom transformers v4.40.1 fork at https://github.com/moojink/transformers-openvla-oft.git — results may vary slightly with other transformers versions.


## Conducting MBR Analysis

CycleVLA uses Minimum Bayes Risk decoding as a test-time scaling strategy on top of the stochastic VLA. For each (task, checkpoint) pair we run the 9-dim policy `N` times with different seeds, slice every rollout into action chunks of size `H = NUM_ACTIONS_CHUNK = 8`, and at each decision step we either (a) pick the chunk that minimises pairwise distance to the other `N-1` candidates (MBR) or (b) pick one at random (Random) — repeating r-NN selection alongside as a third method. Better-than-Random improvements indicate the policy's stochastic samples cluster around success modes, which is exactly the property MBR exploits.

The sweep grid in `experiments/robot/libero/run_mbr_analysis.py` is fixed at `N_VALUES = [4, 8, 16, 32, 64]` (line 33) crossed with `DISTANCE_METRICS = ['CHEBYSHEV', 'CORRELATION', 'COSINE', 'L1', 'L2']` (line 34), evaluated for `METHODS = ['MBR', 'RNN']` against the `RANDOM` baseline (line 35). Empirical takeaway: gains saturate around `N = 16` and `L2` is the strongest distance — consistent with the dense translational and sparse rotational structure of the per-chunk action signal.

The pipeline runs in two steps. Step 1 produces the per-seed rollouts; step 2 aggregates them.

```bash
# Step 1 — seed sweep per task suite. Re-run with different --pretrained_checkpoint
# values (e.g. 200k, 350k, 500k steps) to study the effect of training duration.
# Each run writes trajectory_analysis_*.xlsx files under
#   rollouts/rollouts_sub_decomposed_progress_transit_seed_{ckpt}_chkpt/{task_suite}/
CKPT=/hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--50000_chkpt

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.py \
    --pretrained_checkpoint $CKPT --task_suite_name $SUITE
done
```

```bash
# Step 2 — aggregate across all checkpoint folders in rollouts/. Produces the
# Random / MBR / r-NN statistics matrices over N x distance metric. The
# --first_n flag restricts the analysis to the first K decision steps of each
# rollout (e.g. --first_n 1 only uses step 0).
python experiments/robot/libero/run_mbr_analysis.py \
  --rollouts_dir /hdd2/kai/openvla-oft/rollouts
```

See the paper for the success-probability formulation, the full per-`N` and per-metric tables, and the runtime breakdown across A10 / A100 GPUs.
