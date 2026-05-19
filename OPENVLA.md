# CycleVLA in the LIBERO and LIBERO-Plus Simulation Benchmark Using OpenVLA Backbone

## Relevant Files

| File | Purpose |
| --- | --- |
| `vla-scripts/finetune_progress.py` | Official 9-dim finetune entrypoint; logs split L1 metrics for stop `s_t` (idx 7) and progress `p_t` (idx 8). |
| `vla-scripts/merge_lora_weights_and_save.py` | Merge the trained LoRA adapter back into base OpenVLA weights for eval. |
| `experiments/robot/libero/run_libero_eval_decomposed_progress_mbr.py` | Official CycleVLA eval — 9-dim policy + VLM `transit/backtrack` decision at ~90% subtask progress + MBR seed-sampling on backtrack. |
| `experiments/robot/libero/run_libero_eval_decomposed_progress_transit.py` | Baseline transit-only eval (no VLM, no MBR); raw subtask-transit success. |
| `experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.py` | Seed-sweep harness; same loop as `_transit` plus per-seed `trajectory_analysis_*.xlsx` dumps for MBR analysis. |
| `experiments/robot/libero/run_mbr_analysis.py` | Aggregates the seed-sweep xlsx into Random / MBR matrices over N × distance metric. |
| `experiments/robot/libero/libero_utils.py` | LIBERO env / image / state utilities reused by every eval script. |
| `experiments/robot/openvla_utils.py` | OpenVLA-specific eval utilities. |
| `experiments/robot/robot_utils.py` | Shared (non-OpenVLA) eval utilities. |

## Finetuning

Paper hyperparameters below — a single checkpoint covering all four LIBERO task suites, trained on a 4×A100 server. Finetuned from the base `openvla/openvla-7b` model. Adjust the literal paths to your environment.

```bash
# in:  decomposed_dataset/libero_sub_progress/ (Stage-3 RLDS, see LIBERO.md)
# out: checkpoints under --run_root_dir, as *_chkpt folders per --save_freq
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_progress.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "/hdd2/kai/openvla-oft/decomposed_dataset/libero_sub_progress/" \
  --dataset_name libero_decomposed_progress \
  --run_root_dir "/hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/" \
  --use_l1_regression False \
  --use_diffusion True \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 335000 \
  --max_steps 500005 \
  --save_freq 50000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --grad_accumulation_steps 8 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "CycleVLA_libero_sub_decomposed_progress_oft_A100" \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state
```

## Merge LoRA Adapter

Point `--lora_adapter_dir` at one of the saved `*_chkpt` folders under `run_root_dir`:

```bash
python vla-scripts/merge_lora_weights_and_save.py \
  --base_vla_path openvla/openvla-7b \
  --lora_adapter_dir /hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--500000_chkpt
```

Note: merge on the **same GPU type used for inference** — a train-vs-test device mismatch (e.g. train H100, test A100) drops performance substantially. The adapter is re-mergeable any time; re-download the base model and merge again if needed.

## Launching LIBERO Evaluations

Our trained CycleVLA checkpoint for LIBERO will be released here: **coming soon.**

All four task suites use the **same** unified checkpoint — only `--task_suite_name` changes. `TRANSFORMERS_CACHE` / `HF_HOME` control where checkpoint files cache. The `CKPT` below is the literal `500000_chkpt` from the A100 run; swap in any other `*_chkpt` folder to evaluate a different step.

Evaluation is a **two-stage workflow**: run the transit-only baseline first, then the full method, which re-runs **only the episodes the baseline failed** (it scans the baseline's rollout videos). Pass a shared `VIDEO_DIR` so the two scripts line up — their defaults do not match.

```bash
CKPT=/hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--500000_chkpt
VIDEO_DIR=./rollouts/rollouts_sub_decomposed_progress_transit_500000_chkpt

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  # Stage 1 — transit-only baseline; writes failed-episode videos to $VIDEO_DIR/$SUITE
  CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero/run_libero_eval_decomposed_progress_transit.py \
    --pretrained_checkpoint $CKPT --task_suite_name $SUITE --video_save_dir $VIDEO_DIR

  # Stage 2 — full method (VLM transit/backtrack + MBR); re-runs ONLY the episodes Stage 1 failed
  CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero/run_libero_eval_decomposed_progress_mbr.py \
    --pretrained_checkpoint $CKPT --task_suite_name $SUITE --video_base_dir $VIDEO_DIR
done
```

Notes:
* Stage 2's `--video_base_dir` must equal Stage 1's `--video_save_dir`; the shared `VIDEO_DIR` above ensures this. Pass `--rerun_all True` to skip the baseline and evaluate every episode fresh. Caveat: with no baseline videos and `--rerun_all` unset, every episode is counted as a pass without being run.
* Default 100 rollouts per suite (10 tasks × 10 trials). Tune via `--num_trials_per_task`; change RNG via `--seed`. Other args use the defaults matching the checkpoint above.
* **`--center_crop True` is required** — OFT was finetuned with random 90%-area crops, so eval takes the center 90% crop. The script asserts this.
* Set `--use_wandb True` with `--wandb_project` / `--wandb_entity` to also log to W&B (results are logged locally by default).
* We use the transformers v4.40.1 fork at https://github.com/moojink/transformers-openvla-oft.git — other versions may shift results slightly.

## MBR Analysis

CycleVLA uses Minimum Bayes Risk (MBR) decoding as test-time scaling on the stochastic VLA: run the policy `N` times with different seeds, slice into action chunks of size `H = NUM_ACTIONS_CHUNK = 8`, and at each decision step select one chunk from the `N` candidates. The **default** selection rule is **r-NN density** — keep the chunk in the densest r-NN neighborhood of the candidate set (`use_MBR_vanilla=False`, `distance_metric=l2`); **vanilla MBR** — keep the chunk with the smallest average pairwise distance — is the opt-in alternative (`use_MBR_vanilla=True`). `run_mbr_analysis.py` scores both variants against a `RANDOM` baseline (pick a chunk at random); better-than-Random gains show the stochastic samples cluster around success modes.

Pipeline (2 steps): Step 1 produces per-seed rollouts, Step 2 aggregates them.

```bash
# Step 1 — seed sweep per task suite. Re-run with different --pretrained_checkpoint
# values (e.g. 200k/350k/500k steps) to study training duration.
# out: trajectory_analysis_*.xlsx under
#      rollouts/rollouts_sub_decomposed_progress_transit_seed_{ckpt}_chkpt/{task_suite}/
CKPT=/hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--500000_chkpt

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.py \
    --pretrained_checkpoint $CKPT --task_suite_name $SUITE
done
```

```bash
# Step 2 — aggregate across all checkpoint folders in rollouts/.
# out: Random / MBR statistics matrices over N x distance metric.
# --first_n K restricts analysis to the first K decision steps (e.g. --first_n 1 = step 0 only).
python experiments/robot/libero/run_mbr_analysis.py \
  --rollouts_dir /hdd2/kai/openvla-oft/rollouts
```

Notes:
* Sweep grid (fixed in `run_mbr_analysis.py`): `N_VALUES = [4, 8, 16, 32, 64]` × `DISTANCE_METRICS = ['CHEBYSHEV', 'CORRELATION', 'COSINE', 'L1', 'L2']`, with `METHODS = ['MBR', 'RNN']` against the `RANDOM` baseline.
* See the paper for the success-probability formulation, full per-`N` / per-metric tables, and the A10 / A100 runtime breakdown.
