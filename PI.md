# CycleVLA in the LIBERO and LIBERO-Plus Simulation Benchmark Using pi0.5 Backbone

## Relevant Files

| File | Purpose |
| --- | --- |
| `openpi/examples/libero/convert_libero_data_to_lerobot_cyclevla.py` | Converts the Stage-3 RLDS dataset (`libero_decomposed_progress`) into 9-dim LeRobot v2.0 format `actions = [Δx, Δy, Δz, Δu, Δv, Δw, gripper, s_t, p_t]`; mirrors OFT's `libero_dataset_transform` but keeps the raw gripper (no `invert_gripper_actions`). |
| `openpi/scripts/compute_norm_stats.py` | Upstream helper; writes per-dim `mean/std/q01/q99` norm stats picked up by the training loader. |
| `openpi/src/openpi/training/config.py` | Holds the `pi05_libero_cyclevla` `TrainConfig` (`repo_id=cyclevla/libero_decomposed_progress`, weight-loads from the released `pi05_libero` checkpoint, `action_horizon=10`; 9-dim actions zero-padded to 32). |
| `openpi/scripts/train.py` | Upstream JAX trainer; generic, no CycleVLA changes. |
| `openpi/scripts/train_pi05_libero_cyclevla.sh` | Bash launcher forwarding to `train.py pi05_libero_cyclevla`; pins `WANDB_DIR` under `openpi/wandb/`. |
| `openpi/src/openpi/policies/libero_policy.py` | `LiberoInputs` / `LiberoOutputs` transforms; `LiberoOutputs.action_dim=9` so stop/progress dims survive to inference. |
| `openpi/scripts/serve_policy.py` | Upstream websocket policy server; generic, no CycleVLA changes. |
| `openpi/scripts/serve_openpi_cyclevla.sh` | Helper that serves the trained cyclevla checkpoint with `--policy.config pi05_libero_cyclevla`. |
| `experiments/robot/openpi_utils.py` | `OpenPiClient` websocket wrapper + openpi-vs-OFT glue (raw gripper, raw stop/progress floats, `resize_with_pad`). |
| `experiments/robot/libero/run_libero_eval_openpi_transit.py` | Transit-only eval (advance subtask on the stop signal); openpi counterpart of `run_libero_eval_decomposed_progress_transit.py`. |
| `experiments/robot/libero/run_libero_eval_openpi_cyclevla.py` | Full-method eval (VLM `transit/backtrack` at ~90% progress + sim rewind + MBR on backtrack); openpi counterpart of `run_libero_eval_decomposed_progress_mbr.py`. |
| `experiments/robot/libero/eval_openpi_suite.sh` | Turnkey one-suite runner: launches the policy server on a given GPU/port, runs the two-stage eval (transit then full method), and tears the server down. Run several in parallel for multi-GPU evaluation. |

## Step 1 — Convert RLDS Dataset to LeRobot Format

After the LIBERO 3-stage dataset pipeline finishes (see `LIBERO.md`):

```bash
cd /hdd2/kai/openvla-oft/openpi
# in:  decomposed_dataset/libero_sub_progress/ (Stage-3 RLDS)
# out: openpi/data/lerobot/cyclevla/libero_decomposed_progress/
uv run examples/libero/convert_libero_data_to_lerobot_cyclevla.py \
    --data_dir /hdd2/kai/openvla-oft/decomposed_dataset/libero_sub_progress
```

## Step 2 — Compute Normalization Statistics

```bash
cd /hdd2/kai/openvla-oft/openpi
# out: openpi/assets/pi05_libero_cyclevla/cyclevla/libero_decomposed_progress/norm_stats.json
uv run scripts/compute_norm_stats.py --config-name pi05_libero_cyclevla
```

## Step 3 — Launch pi0.5 Finetune

Paper hyperparameters below — trained on a 4-GPU server.

```bash
cd /hdd2/kai/openvla-oft/openpi
scripts/train_pi05_libero_cyclevla.sh \
    CycleVLA_libero_sub_decomposed_progress_pi05_A100 \
    --project-name cyclevla_openpi \
    --batch-size 128 \
    --fsdp-devices 8 \
    --overwrite
```

Note: training starts from the openpi-released `pi05_libero` checkpoint (not `pi05_base`), so the model learns the two new supervision dims (`s_t`, `p_t`) and the subtask-level prompt distribution.

## Launching LIBERO Evaluations

The pi0.5 policy is evaluated **client/server**: an openpi process serves the checkpoint over a websocket; a LIBERO eval client in the openvla-oft env queries it each step. pi0.5 keeps the raw RLDS gripper, so dims 0-6 need no remap — the only openpi-vs-OFT glue lives in `experiments/robot/openpi_utils.py`.

**One-time setup** — install the websocket client into the eval env:

```bash
conda activate /hdd2/kai/openvla-oft/env
pip install -e /hdd2/kai/openvla-oft/openpi/packages/openpi-client
```

**Step A — Serve the policy.** Edit `CKPT_DIR` in the helper (or export it) to point at your trained checkpoint (the dir must contain `params/` and `assets/`):

```bash
cd /hdd2/kai/openvla-oft/openpi
CUDA_VISIBLE_DEVICES=0 scripts/serve_openpi_cyclevla.sh   # pi0.5 on GPU 0, ws://0.0.0.0:8000
```

`CUDA_VISIBLE_DEVICES` selects the GPU the pi0.5 model runs on; `PORT` and `CKPT_DIR` are env-overridable too (used below for parallel evaluation).

**Step B — Run the eval client.** In a separate shell, with the server up. All suites use the same checkpoint; only `--task_suite_name` changes.

This is a **two-stage** workflow (same as the OpenVLA-OFT `_mbr.py` eval): run the transit baseline **first**, then the full method, which re-runs **only the episodes the transit baseline failed** (it scans the baseline's rollout videos to find them). Run both stages for the same `--task_suite_name`.

```bash
conda activate /hdd2/kai/openvla-oft/env
cd /hdd2/kai/openvla-oft

# Stage 1 — transit-only baseline. Writes per-episode rollout videos to
#   rollouts/rollouts_openpi_transit/  (raw subtask-transit success).
python experiments/robot/libero/run_libero_eval_openpi_transit.py \
  --host 0.0.0.0 --port 8000 --task_suite_name libero_spatial

# Stage 2 — full method (VLM transit/backtrack + sim rewind + MBR on backtrack).
# Re-runs ONLY the episodes Stage 1 failed; reads them from --video_base_dir
# (default == Stage 1's video_save_dir, rollouts/rollouts_openpi_transit/).
python experiments/robot/libero/run_libero_eval_openpi_cyclevla.py \
  --host 0.0.0.0 --port 8000 --task_suite_name libero_spatial
```

Notes:
* Swap `--task_suite_name` for `libero_object`, `libero_goal`, or `libero_10` (run both stages for each).
* The full-method script **requires** the transit baseline to have been run first for the same suite — it aborts with a clear error if no baseline videos are found under `--video_base_dir`. Pass `--rerun_all True` to skip the baseline and evaluate every episode fresh instead. If you change the transit script's `--video_save_dir`, pass the matching path as the full-method script's `--video_base_dir`.
* The full-method script needs `OPENAI_API_KEY` (in `.env`) for the VLM detector — same `VLMDetector` prompt as the OpenVLA `_mbr.py` eval.
* The full-method script runs MBR decoding on every backtrack: it samples `--mbr_num_seeds` candidate chunks from the server (`Policy.infer` re-keys its RNG each call, so repeated queries are naturally diverse), ranks them, and executes the winner; backtracks are capped by `--max_subtask_retries`. The post-hoc seed-sweep analysis (`run_mbr_analysis.py`) is out of scope for the pi0.5 backbone.
* Logs and rollout videos: `experiments/logs/logs_openpi_*` and `rollouts/rollouts_openpi_*`. Set `--use_wandb True` to also log to W&B.
* **GPU placement:** pi0.5 inference runs on the *server* (`CUDA_VISIBLE_DEVICES` at serve time picks its GPU). The eval *client* loads no model — its only GPU use is MuJoCo offscreen rendering — so `CUDA_VISIBLE_DEVICES` on a client command is optional.

## Parallel Evaluation Across GPUs

Each server is one pi0.5 model on one GPU on one port; each client connects by `--port`. For true N× speedup, run one `eval_openpi_suite.sh` per GPU on distinct ports.

**One suite, turnkey.** `eval_openpi_suite.sh <gpu_id> <port> <task_suite>` launches the server, runs transit + full method, and tears the server down:

```bash
conda activate /hdd2/kai/openvla-oft/env
cd /hdd2/kai/openvla-oft
bash experiments/robot/libero/eval_openpi_suite.sh 0 8000 libero_spatial
```

To evaluate all four suites in parallel, launch one per GPU/port in the background:

```bash
bash experiments/robot/libero/eval_openpi_suite.sh 0 8000 libero_spatial &
bash experiments/robot/libero/eval_openpi_suite.sh 1 8001 libero_object  &
bash experiments/robot/libero/eval_openpi_suite.sh 2 8002 libero_goal    &
bash experiments/robot/libero/eval_openpi_suite.sh 3 8003 libero_10      &
wait
```

**Manual equivalent.** The turnkey scripts just automate this: launch one server per GPU on its own port, then point each suite's two-stage client at the matching port — e.g. `CUDA_VISIBLE_DEVICES=1 PORT=8001 scripts/serve_openpi_cyclevla.sh` paired with `... run_libero_eval_openpi_transit.py --port 8001 --task_suite_name libero_object`.
