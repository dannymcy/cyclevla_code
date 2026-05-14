# CycleVLA in the LIBERO and LIBERO-Plus Simulation Benchmark Using pi0.5 Backbone

## Relevant Files

Conversion
* `openpi/examples/libero/convert_libero_data_to_lerobot_cyclevla.py`: Converts the CycleVLA Stage-3 RLDS dataset (`libero_decomposed_progress`, built by `rlds_dataset_builder/LIBERO_Decomposed_Progress/`) into LeRobot v2.0 format. Writes 9-dim `actions = [Δx, Δy, Δz, Δu, Δv, Δw, gripper, s_t, p_t]` so pi0.5's flow-matching action expert predicts the supervision dims jointly with the base 7-dim action; mirrors OFT's `libero_dataset_transform` (`prismatic/vla/datasets/rlds/oxe/transforms.py:828-854`) without the `invert_gripper_actions` step.

Normalization
* `openpi/scripts/compute_norm_stats.py`: Upstream openpi helper. Reads `state` and `actions` from the LeRobot dataset bound to the selected `TrainConfig` and writes per-dim `mean`/`std`/`q01`/`q99` to `openpi/assets/<config_name>/<repo_id>/norm_stats.json`. The training loader auto-picks the file up via `DataConfig.norm_stats`.

Training
* `openpi/src/openpi/training/config.py`: Holds the registered `pi05_libero_cyclevla` `TrainConfig` (right after `pi05_libero`). Mirrors `pi05_libero` byte-for-byte except `repo_id="cyclevla/libero_decomposed_progress"` and `weight_loader → gs://openpi-assets/checkpoints/pi05_libero/params` (starts from the openpi-released LIBERO-finetuned pi0.5 checkpoint, not `pi05_base`, so the model already knows LIBERO and only has to learn the two new supervision dims). `action_horizon=10`; the data pipeline zero-pads our 9-dim actions to `action_dim=32` via `PadStatesAndActions`.
* `openpi/scripts/train.py`: Upstream openpi JAX trainer. Generic; no code changes needed for the CycleVLA config.
* `openpi/scripts/train_pi05_libero_cyclevla.sh`: Thin bash launcher that forwards args to `uv run scripts/train.py pi05_libero_cyclevla ...`. Pins `WANDB_DIR` under `openpi/wandb/` so wandb runs always land in one place.


## Step 1 — Convert the RLDS Dataset to LeRobot Format

After the LIBERO 3-stage dataset pipeline finishes (see `LIBERO.md`), point the converter at the Stage-3 RLDS output directory:

```bash
cd /hdd2/kai/openvla-oft/openpi
uv run examples/libero/convert_libero_data_to_lerobot_cyclevla.py \
    --data_dir /hdd2/kai/openvla-oft/decomposed_dataset/libero_sub_progress
```

Output: `openpi/data/lerobot/cyclevla/libero_decomposed_progress/`.


## Step 2 — Compute Normalization Statistics

```bash
cd /hdd2/kai/openvla-oft/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_libero_cyclevla
```

Output: `openpi/assets/pi05_libero_cyclevla/cyclevla/libero_decomposed_progress/norm_stats.json`. 


## Step 3 — Launch pi0.5 Finetune

Launch the finetuning script with the configuration below. This is the hyperparameter set used in the paper — we trained on a 4-GPU server starting from the openpi-released `pi05_libero` checkpoint (so only the two new supervision dims and the subtask-level prompt distribution need to be learned, not LIBERO from scratch).

```bash
cd /hdd2/kai/openvla-oft/openpi
scripts/train_pi05_libero_cyclevla.sh \
    CycleVLA_libero_sub_decomposed_progress_pi05_A100 \
    --project-name cyclevla_openpi \
    --batch-size 128 \
    --fsdp-devices 8 \
    --no-ema-decay \
    --overwrite
```


## Launching LIBERO Evaluations

Coming soon — a separate inference script under `experiments/robot/libero/` will load the pi0.5 checkpoint, map its 9-dim output back to the LIBERO env's gripper convention, and run the same MBR-decoding eval pipeline as the OpenVLA backbone (see `OPENVLA.md`).
