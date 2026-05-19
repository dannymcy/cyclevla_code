"""
run_libero_eval_openpi_cyclevla.py

Full-method CycleVLA LIBERO eval for the **pi05 / openpi** policy.

This is the openpi counterpart of `run_libero_eval_decomposed_progress_mbr.py`.
It runs the full proactive self-correction loop:

  - Per-subtask two-phase protocol: drive the subtask until the policy reports
    ~90% progress (`to_check`), query a VLM for a `transit | backtrack`
    decision, then drive to completion on the stop signal (`to_complete`).
  - On `backtrack`, physically rewind the sim to the start of the target
    subtask, then retry using **MBR (Minimum Bayes Risk) decoding** as
    test-time scaling: sample N candidate action chunks, rank them, and execute
    the MBR-selected chunk (bounded by a per-subtask retry cap).

How MBR sampling works here: the openpi server is stochastic per `infer()`
call (`Policy.infer` splits its RNG each call), so N `client.get_action` calls
on the same observation yield N diverse chunks -- no seed control or server
change needed. Unlike the OpenVLA MBR (which replays a chosen diffusion seed),
we already hold every sampled chunk, so we cache and execute the winner
directly.

What is intentionally NOT included (per the current task scope): the seed-sweep
harness (`run_libero_eval_decomposed_progress_transit_seed.py`) and the post-hoc
`run_mbr_analysis.py` aggregator. MBR *decoding* (the retry mechanism) IS
included -- it is the core of the full method.

The VLM detector and the sim-rewind / trajectory-feature helpers are imported
from `run_libero_eval_decomposed_progress_mbr.py` so the VLM prompt and the
MuJoCo state-restore logic stay single-sourced with the OpenVLA-OFT eval.

The 9-dim policy is served remotely by openpi (see
`openpi/serve_openpi_cyclevla.sh`); all openpi-vs-OpenVLA convention
differences are handled in `experiments/robot/openpi_utils.py`.

Two-stage workflow (mirrors `run_libero_eval_decomposed_progress_mbr.py`): this
full-method eval re-runs only the episodes the transit baseline failed, so the
transit eval must be run FIRST. It scans the transit baseline's rollout videos
(`--video_base_dir`, default == the transit script's `video_save_dir`) to find
the failed episodes. Pass `--rerun_all True` to evaluate every episode fresh
instead. If no baseline videos are found, the run aborts (no silent fake-100%).

Usage (the eval env, with the policy server already running):

  conda activate /hdd2/kai/openvla-oft/env
  # 1) transit baseline first (writes rollout videos used to pick the failures)
  python experiments/robot/libero/run_libero_eval_openpi_transit.py \
      --host 0.0.0.0 --port 8000 --task_suite_name libero_spatial
  # 2) full method -- re-runs only the episodes the baseline failed
  python experiments/robot/libero/run_libero_eval_openpi_cyclevla.py \
      --host 0.0.0.0 --port 8000 --task_suite_name libero_spatial
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

import wandb

# Append repo root so the interpreter can find `experiments.robot`.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(ROOT_DIR)

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video_decomposed,
)
from experiments.robot.openpi_utils import OpenPiClient, split_openpi_action
from experiments.robot.robot_utils import DATE_TIME, set_seed_everywhere

# Reuse the VLM detector, the sim-rewind helpers, and the trajectory-feature
# extractor from the OpenVLA-OFT MBR eval so the VLM prompt, the MuJoCo
# state-restore logic, and the MBR feature layout stay single-sourced.
# Importing this module has no model-loading side effects.
from experiments.robot.libero.run_libero_eval_decomposed_progress_mbr import (
    VLMDetector,
    backtrace_robot_states,
    complex_states,
    extract_trajectory_features,
    get_failed_episodes_from_videos,
    pick_place_states,
    record_robot_state,
    restore_robot_only,
)


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # openpi policy server connection
    #################################################################################################################
    host: str = "0.0.0.0"                            # Host of the openpi websocket policy server
    port: int = 8000                                 # Port of the openpi websocket policy server

    # Number of actions to execute open-loop before requerying the policy. The
    # server returns a chunk of `action_horizon` (10) actions; this must be <= 10.
    num_open_loop_steps: int = 5

    #################################################################################################################
    # CycleVLA proactive-correction parameters
    #################################################################################################################
    progress_threshold: float = 0.90                 # Progress signal level that triggers the VLM check
    max_subtask_retries: int = 3                     # Max backtrack/retry attempts per subtask
    vlm_model: str = "gpt-5.5"                       # VLM used for the transit/backtrack decision
    vlm_temperature: float = 1.0                     # VLM sampling temperature

    # MBR (Minimum Bayes Risk) decoding on backtrack -- the test-time-scaling
    # retry mechanism. Mirrors the `mbr_config` dict in the OpenVLA MBR eval.
    mbr_num_seeds: int = 8                           # Candidate action chunks sampled per backtrack
    mbr_distance_metric: str = "l2"                  # l2 | l1 | cosine | correlation | chebyshev
    mbr_use_failed_repulsion: bool = False           # Repel candidates away from previously-failed trajectories
    mbr_r_neighborhood: Optional[int] = None         # r-NN neighborhood size (None = adaptive)
    mbr_vanilla: bool = False                        # Use plain average-distance MBR (no r-NN density)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 1024                          # Sim render res for rollout videos (policy input is 224 regardless)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs/logs_openpi_cyclevla"   # Local directory for eval logs
    video_save_dir: str = "./rollouts/rollouts_openpi_cyclevla"

    # Two-stage workflow (mirrors run_libero_eval_decomposed_progress_mbr.py):
    # the full method re-runs only the episodes the transit baseline failed.
    # `video_base_dir` must point at the transit eval's `video_save_dir` (they
    # match by default). `rerun_all=True` runs every episode fresh instead.
    video_base_dir: str = "./rollouts/rollouts_openpi_transit"
    rerun_all: bool = False

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 0                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    # The openpi server (`pi05_libero_cyclevla`, action_horizon=10) returns a
    # 10-action chunk; we cannot execute more open-loop steps than that.
    assert 1 <= cfg.num_open_loop_steps <= 10, "num_open_loop_steps must be in [1, 10]"
    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY must be set (in .env) for the VLM detector."

    # Two-stage guard. The full method re-runs only the episodes the transit
    # baseline failed (read from `video_base_dir`). If that dir has no rollout
    # videos, `get_failed_episodes_from_videos` would yield an empty failed-list
    # -> every episode silently marked success without running. Fail loudly so
    # this can never produce a fake ~100%.
    if not cfg.rerun_all:
        base = os.path.join(cfg.video_base_dir, cfg.task_suite_name)
        has_videos = os.path.isdir(base) and any(
            f.endswith(".mp4") for _, _, files in os.walk(base) for f in files
        )
        if not has_videos:
            raise FileNotFoundError(
                f"No transit-baseline rollout videos found at {base}.\n"
                f"The full-method eval re-runs only the episodes the transit baseline failed, "
                f"so run the transit eval for `{cfg.task_suite_name}` first:\n"
                f"  python experiments/robot/libero/run_libero_eval_openpi_transit.py "
                f"--task_suite_name {cfg.task_suite_name}\n"
                f"(keep its --video_save_dir equal to this script's --video_base_dir, "
                f"currently '{cfg.video_base_dir}')\n"
                f"-- or pass --rerun_all True to evaluate every episode fresh."
            )


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    run_id = f"EVAL-{cfg.task_suite_name}-openpi-cyclevla-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def sample_and_rank_chunks_mbr(
    cfg: GenerateConfig,
    client: OpenPiClient,
    obs,
    current_subtask: str,
    failed_trajectories: list,
    selection_mode: str = "rep",
    log_file=None,
):
    """Sample N candidate action chunks from the openpi server and MBR-rank them.

    openpi counterpart of `sample_and_rank_seeds_mbr` in the OpenVLA MBR eval.
    Call this AFTER a backtrack, so `obs` is the sim observation at the start of
    the subtask being retried.

    The openpi server is stochastic per `infer()` call, so N `client.get_action`
    calls on the same `obs` yield N diverse candidate chunks -- no seed control
    needed. Because we hold every sampled chunk, we return the chunks themselves
    (best-first) instead of seeds; the caller executes the winner directly.

    Args:
        failed_trajectories: first-N trajectory feature vectors from prior failed
            runs of this subtask (used only when `cfg.mbr_use_failed_repulsion`).
        selection_mode: "rep" (representative / densest pocket) or "away".

    Returns:
        list of action chunks (each a list of 9-dim np.ndarray), ranked best-first.
    """
    num_seeds = cfg.mbr_num_seeds
    distance_metric = cfg.mbr_distance_metric
    # multiply by 6: translation (xyz) + rotation (xyz) per timestep
    expected_features = cfg.num_open_loop_steps * 6

    # Starting end-effector pose (same 8-dim state layout used for the policy).
    current_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    current_euler = np.asarray(quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float64)

    sampled_chunks = []              # raw 9-dim chunks (list of np.ndarray)
    sampled_state_trajectories = []  # per-chunk feature vector for MBR

    for _ in range(num_seeds):
        # One stochastic sample from the server (diverse across calls).
        chunk = client.get_action(obs, current_subtask, cfg.num_open_loop_steps)
        sampled_chunks.append(chunk)

        # Integrate the action deltas into a predicted state trajectory, exactly
        # as the OpenVLA MBR does (cumulative position; composed rotations).
        cumulative_pos = current_pos.copy()
        cumulative_rot = R.from_euler("xyz", current_euler)
        state_features: list = []
        for action in chunk:
            robot_action, _, _ = split_openpi_action(action)
            cumulative_pos = cumulative_pos + np.asarray(robot_action[:3])
            cumulative_rot = cumulative_rot * R.from_euler("xyz", np.asarray(robot_action[3:6]))
            state_features.extend(cumulative_pos.tolist())
            state_features.extend(cumulative_rot.as_euler("xyz").tolist())

        # Pad short chunks to a fixed feature length.
        while len(state_features) < expected_features:
            state_features.extend([0, 0, 0, 0, 0, 0])
        sampled_state_trajectories.append(np.array(state_features[:expected_features]))

    # ---- MBR ranking (ported from sample_and_rank_seeds_mbr) ----------------
    X = np.stack(sampled_state_trajectories)  # (num_seeds, expected_features)
    N = X.shape[0]

    metric_map = {
        "l2": "euclidean",
        "l1": "cityblock",
        "cosine": "cosine",
        "correlation": "correlation",
        "chebyshev": "chebyshev",
    }
    dist_mat = cdist(X, X, metric=metric_map.get(distance_metric, "euclidean"))

    # Vanilla MBR: rank by average distance to all other candidates.
    if cfg.mbr_vanilla:
        avg_dist = dist_mat.mean(axis=1)
        ranked_indices = np.argsort(avg_dist)[::-1] if selection_mode == "away" else np.argsort(avg_dist)
        log_message(
            f"Vanilla MBR ranking complete (mode={selection_mode}, metric={distance_metric}); "
            f"top avg-distances: {avg_dist[ranked_indices[:3]]}",
            log_file,
        )
        return [sampled_chunks[i] for i in ranked_indices]

    # Adaptive r-NN neighborhood size.
    r = cfg.mbr_r_neighborhood if cfg.mbr_r_neighborhood is not None else max(2, min(4, int(np.sqrt(N))))
    r_eff = min(r, max(1, N - 1))

    # r-NN radius = distance to the r-th nearest neighbor; densest pocket center.
    rnn_radius = np.partition(dist_mat, r_eff, axis=1)[:, r_eff]
    center_idx = int(np.argmin(rnn_radius))
    cluster_idx = np.argsort(dist_mat[center_idx])[:r_eff]
    # Medoid inside the pocket (most representative candidate).
    intra = dist_mat[np.ix_(cluster_idx, cluster_idx)]
    medoid_local = cluster_idx[int(np.argmin(intra.mean(axis=1)))]
    d_to_medoid = dist_mat[medoid_local]

    def robust_norm(v):
        v = np.asarray(v)
        if len(v) < 2:
            return np.zeros_like(v)
        lo, hi = np.percentile(v, [10, 90])
        v_clip = np.clip(v, lo, hi)
        med = np.median(v_clip)
        iqr = (np.percentile(v_clip, 75) - np.percentile(v_clip, 25)) + 1e-8
        return (v - med) / iqr

    # Optional repulsion away from previously-failed trajectories.
    if cfg.mbr_use_failed_repulsion and failed_trajectories:
        valid_failed = [
            ft for ft in failed_trajectories
            if isinstance(ft, np.ndarray) and ft.shape[0] == expected_features
        ]
        if valid_failed:
            d_fail = cdist(X, np.stack(valid_failed), metric=metric_map.get(distance_metric, "euclidean")).min(axis=1)
        else:
            d_fail = np.full((N,), np.median(rnn_radius) if N > 0 else 1.0)
    else:
        d_fail = np.full((N,), np.median(rnn_radius) if N > 0 else 1.0)

    rnn_norm = robust_norm(rnn_radius)
    dmed_norm = robust_norm(d_to_medoid)
    dfail_norm = robust_norm(d_fail)
    repulse = 1.0 / (1.0 + np.exp(-dfail_norm))  # sigmoid-softened repulsion (tau=1)
    lambda_fail = 0.5

    if selection_mode == "away":
        final_scores = dmed_norm + lambda_fail * repulse
    else:
        final_scores = -rnn_norm + lambda_fail * repulse
    ranked_indices = np.argsort(final_scores)[::-1]  # higher is better

    log_message(
        f"MBR ranking complete (mode={selection_mode}, metric={distance_metric}); "
        f"medoid candidate index {medoid_local}; top scores: {final_scores[ranked_indices[:3]]}",
        log_file,
    )
    return [sampled_chunks[i] for i in ranked_indices]


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    client: OpenPiClient,
    initial_state=None,
    log_file=None,
):
    """Run a single episode with the proactive transit/backtrack loop."""
    # Get subtasks. The bare lowercased subtask string is the prompt the openpi
    # policy was trained with (`prompt_from_task=True`; the Stage-3 RLDS builder
    # writes `chunk['subtask'].lower()` into `language_instruction`).
    states = pick_place_states(task_description, f"{cfg.task_suite_name}_no_noops")
    states = complex_states(task_description, f"{cfg.task_suite_name}_no_noops") if states is None else states
    states = [state.lower() for state in states]

    vlm_detector = VLMDetector(model_name=cfg.vlm_model, temperature=cfg.vlm_temperature)

    # Reset environment
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Open-loop action queue, refilled from the policy server when empty.
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images, replay_wrist_images, replay_subtasks = [], [], []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    success = False
    current_state = states[0]
    subtask_hist, exe_type_hist = [current_state], ["init"]
    robot_state_hist: List[Dict[str, Any]] = []  # full sim snapshots for backtracking

    # Per-subtask phase: "to_check" -> VLM decision -> "to_complete".
    subtask_phase = "to_check"

    # Goal/long tasks use the same robust (consecutive/recurring) confirmation
    # for the 90% progress signal that all suites use for the stop signal.
    use_robust_progress_checking = cfg.task_suite_name in ["libero_goal", "libero_10"]
    if use_robust_progress_checking:
        first_progress_high_seen = False
        consecutive_progress_high_count = 0
        steps_since_last_progress_high = 0
    else:
        count_check_signals = 0

    # Stop-signal (100% completion) confirmation counters.
    first_high_seen = False
    consecutive_high_count = 0
    steps_since_last_high = 0

    # Per-subtask backtrack/retry + MBR bookkeeping.
    subtask_retry_count: Dict[str, int] = {}          # bounds the correction loop
    subtask_chunk_rankings: Dict[str, list] = {}      # MBR-ranked candidate chunks per subtask
    current_chunk_index: Dict[str, int] = {}          # which ranked chunk to use next
    subtask_failed_trajectories: Dict[str, list] = {}  # failed-run features (for repulsion)

    try:
        while t < max_steps * 1.5 + cfg.num_steps_wait:
            # Let objects settle
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action("openpi"))
                t += 1
                continue

            # Record replay frame (raw rotated images at env res)
            img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)
            replay_images.append(img)
            replay_wrist_images.append(wrist_img)
            replay_subtasks.append(f"{exe_type_hist[-1]}: {current_state}")

            # If action queue is empty, requery the openpi policy server
            if len(action_queue) == 0:
                actions = client.get_action(obs, current_state, cfg.num_open_loop_steps)
                action_queue.extend(actions)

            # Record full sim snapshot BEFORE executing the next action (used to
            # rewind on backtrack).
            robot_state_hist = record_robot_state(robot_state_hist, current_state, env, obs)

            # Split the 9-dim openpi action: dims 0-6 env-ready (raw gripper),
            # dims 7-8 raw stop/progress floats.
            robot_action, stop_signal, progress_signal = split_openpi_action(action_queue.popleft())

            # Step environment with the 7-dim robot action only
            obs, reward, done, info = env.step(robot_action.tolist())
            t += 1

            if done:
                success = True
                log_message(f"Environment signaled done for subtask: {current_state} in {t} steps", log_file)
                break

            # =========================
            # PHASE 1: Check for 90% progress (VLM decision point)
            # =========================
            if subtask_phase == "to_check":
                # Gripper subtasks are trivial; skip the VLM check for them.
                is_gripper_subtask = (
                    "close the gripper to grasp" in current_state.lower()
                    or "open the gripper to release" in current_state.lower()
                )
                if is_gripper_subtask:
                    log_message(f"Gripper subtask detected: `{current_state}` - skipping VLM check", log_file)
                    subtask_phase = "to_complete"
                    first_high_seen = False
                    consecutive_high_count = 0
                    steps_since_last_high = 0
                    continue

                vlm_check_needed = False

                if use_robust_progress_checking:
                    # Robust checking (Goal/Long tasks): consecutive or recurring high signals.
                    if progress_signal >= cfg.progress_threshold:
                        consecutive_progress_high_count += 1
                        if not first_progress_high_seen:
                            first_progress_high_seen = True
                            log_message(
                                f"90% progress signal observed at step {t} for subtask: {current_state} "
                                f"(progress={progress_signal:.2f})",
                                log_file,
                            )
                        if consecutive_progress_high_count >= 2:
                            vlm_check_needed = True
                            log_message(
                                f"90% progress confirmed at step {t} "
                                f"({consecutive_progress_high_count} consecutive signals)",
                                log_file,
                            )
                        elif first_progress_high_seen and steps_since_last_progress_high >= 2:
                            vlm_check_needed = True
                            log_message(
                                f"90% progress re-confirmed at step {t} "
                                f"(after {steps_since_last_progress_high} low steps)",
                                log_file,
                            )
                        else:
                            if first_progress_high_seen and steps_since_last_progress_high > 0:
                                log_message(
                                    f"Ignoring 90% re-signal at step {t} "
                                    f"(only {steps_since_last_progress_high} low steps since last; need 2+)",
                                    log_file,
                                )
                        steps_since_last_progress_high = 0
                    else:
                        consecutive_progress_high_count = 0
                        if first_progress_high_seen:
                            steps_since_last_progress_high += 1
                else:
                    # Simple checking (Spatial/Object tasks): 2nd high signal triggers.
                    if progress_signal >= cfg.progress_threshold:
                        count_check_signals += 1
                        log_message(
                            f"90% signal #{count_check_signals} at step {t} (progress={progress_signal:.2f})", log_file
                        )
                        if count_check_signals >= 2:
                            vlm_check_needed = True
                            log_message(f"90% progress confirmed after {count_check_signals} high signals", log_file)

                if vlm_check_needed:
                    # Hold a few identical frames in the replay for VLM context (no physics step).
                    current_img = get_libero_image(obs)
                    current_wrist_img = get_libero_wrist_image(obs)
                    for _ in range(32):
                        replay_images.append(current_img)
                        replay_wrist_images.append(current_wrist_img)
                        replay_subtasks.append(f"VLM_90%_check: {current_state}")

                    # Query the VLM for a transit/backtrack decision.
                    res = vlm_detector.detect_subtask(
                        current_state, states, subtask_hist, task_description, current_img, current_wrist_img
                    )
                    detected_state, exe_type, reason = vlm_detector.extract_res(res)
                    log_message(
                        f"VLM check at 90% for subtask `{current_state}` -> next: {detected_state}, "
                        f"type: {exe_type}, reason: {reason}",
                        log_file,
                    )

                    # Guard: a backtrack target must be an exact subtask string.
                    # If the VLM names something else, a sim rewind / index
                    # lookup would fail, so fall back to continuing.
                    if exe_type == "backtrack" and detected_state not in states:
                        log_message(
                            f"VLM backtrack target `{detected_state}` not in subtask list; treating as transit.",
                            log_file,
                        )
                        exe_type = "transit"

                    if exe_type == "backtrack":
                        subtask_retry_count[current_state] = subtask_retry_count.get(current_state, 0) + 1
                        retry_num = subtask_retry_count[current_state]

                        # Bound the correction loop: after `max_subtask_retries`
                        # backtracks, force the subtask to completion.
                        if retry_num >= cfg.max_subtask_retries:
                            log_message(
                                f"Maximum retries ({cfg.max_subtask_retries}) reached for subtask "
                                f"`{current_state}`. Forcing continuation to completion.",
                                log_file,
                            )
                            subtask_phase = "to_complete"
                            first_high_seen = False
                            consecutive_high_count = 0
                            steps_since_last_high = 0
                            continue

                        log_message(
                            f"VLM decided to backtrack from `{current_state}` to `{detected_state}` "
                            f"at 90% progress (retry #{retry_num})",
                            log_file,
                        )

                        # Record the failed run's first-N trajectory features
                        # under the FAILING subtask (feeds MBR failed-repulsion).
                        failed_feat = extract_trajectory_features(
                            robot_state_hist, current_state, start_idx=0, end_idx=cfg.num_open_loop_steps
                        )
                        subtask_failed_trajectories.setdefault(current_state, []).append(failed_feat)

                        # Switch to the target subtask and physically rewind the
                        # sim to its start.
                        current_state = detected_state
                        subtask_hist.append(current_state)
                        exe_type_hist.append(f"backtrack_90%_retry{retry_num}")
                        action_queue.clear()

                        reversed_snaps = backtrace_robot_states(robot_state_hist, current_state)
                        for snap in reversed_snaps:
                            restore_robot_only(env, snap)
                            obs, _, _, _ = env.step(get_libero_dummy_action("openpi"))
                            replay_images.append(get_libero_image(obs))
                            replay_wrist_images.append(get_libero_wrist_image(obs))
                            replay_subtasks.append(f"backtrack_from_90%_retry{retry_num}")

                        # MBR decoding: on the first backtrack to this subtask,
                        # sample N candidate chunks from the (stochastic) openpi
                        # server and rank them; on later backtracks reuse the
                        # ranking by stepping to the next-best chunk.
                        if current_state not in subtask_chunk_rankings:
                            log_message(
                                f"Sampling {cfg.mbr_num_seeds} candidate chunks for MBR decoding "
                                f"of subtask `{current_state}`...",
                                log_file,
                            )
                            subtask_chunk_rankings[current_state] = sample_and_rank_chunks_mbr(
                                cfg,
                                client,
                                obs,
                                current_state,
                                subtask_failed_trajectories.get(current_state, []),
                                log_file=log_file,
                            )
                            current_chunk_index[current_state] = 0
                        else:
                            current_chunk_index[current_state] += 1
                            if current_chunk_index[current_state] >= len(subtask_chunk_rankings[current_state]):
                                current_chunk_index[current_state] = 0  # wrap around

                        # Prime the action queue with the MBR-selected chunk so
                        # the retry executes it directly (openpi cannot replay a
                        # seed, but we already hold the ranked chunk).
                        chunk_idx = current_chunk_index[current_state]
                        action_queue.extend(subtask_chunk_rankings[current_state][chunk_idx])
                        log_message(
                            f"Retrying subtask `{current_state}` with MBR-ranked chunk "
                            f"#{chunk_idx + 1}/{len(subtask_chunk_rankings[current_state])}",
                            log_file,
                        )

                        # Reset phase tracking for the retry.
                        subtask_phase = "to_check"
                        if use_robust_progress_checking:
                            first_progress_high_seen = False
                            consecutive_progress_high_count = 0
                            steps_since_last_progress_high = 0
                        else:
                            count_check_signals = 0
                        first_high_seen = False
                        consecutive_high_count = 0
                        steps_since_last_high = 0
                    else:
                        # transit: continue current subtask toward completion.
                        log_message(
                            f"VLM decided to continue with subtask `{current_state}` "
                            f"(detected: {detected_state}, type: {exe_type})",
                            log_file,
                        )
                        subtask_phase = "to_complete"
                        if use_robust_progress_checking:
                            first_progress_high_seen = False
                            consecutive_progress_high_count = 0
                            steps_since_last_progress_high = 0
                        else:
                            count_check_signals = 0
                        first_high_seen = False
                        consecutive_high_count = 0
                        steps_since_last_high = 0

            # =========================
            # PHASE 2: Check for termination (100% completion via stop signal)
            # =========================
            elif subtask_phase == "to_complete":
                subtask_completed = False

                # openpi emits a raw stop float (~1.0 = stop); the OpenVLA
                # script's `== -1` test becomes `> 0.5` here.
                if stop_signal > 0.5:
                    consecutive_high_count += 1
                    if not first_high_seen:
                        first_high_seen = True
                        log_message(
                            f"Stop signal observed at step {t} for subtask: {current_state} "
                            f"(stop={stop_signal:.2f}, progress={progress_signal:.2f})",
                            log_file,
                        )
                    if consecutive_high_count >= 2:
                        subtask_completed = True
                        log_message(
                            f"Subtask `{current_state}` completed at step {t} "
                            f"({consecutive_high_count} consecutive stop signals)",
                            log_file,
                        )
                    elif first_high_seen and steps_since_last_high >= 2:
                        subtask_completed = True
                        log_message(
                            f"Subtask `{current_state}` completed at step {t} "
                            f"(stop re-confirmed after {steps_since_last_high} low steps)",
                            log_file,
                        )
                    else:
                        if first_high_seen and steps_since_last_high > 0:
                            log_message(
                                f"Ignoring stop re-signal at step {t} "
                                f"(only {steps_since_last_high} low steps since last; need 2+)",
                                log_file,
                            )
                    steps_since_last_high = 0
                else:
                    consecutive_high_count = 0
                    if first_high_seen:
                        steps_since_last_high += 1

                if subtask_completed:
                    prev_state = current_state
                    current_index = states.index(current_state)
                    log_message(f"Subtask `{current_state}` completed via stop signal at step {t}", log_file)

                    if current_index + 1 < len(states):
                        current_state = states[current_index + 1]
                        subtask_hist.append(current_state)
                        exe_type_hist.append("continue_termination")
                        log_message(f"Moving from `{prev_state}` to next subtask: `{current_state}`", log_file)

                        action_queue.clear()
                        subtask_phase = "to_check"
                        if use_robust_progress_checking:
                            first_progress_high_seen = False
                            consecutive_progress_high_count = 0
                            steps_since_last_progress_high = 0
                        else:
                            count_check_signals = 0
                        first_high_seen = False
                        consecutive_high_count = 0
                        steps_since_last_high = 0
                    else:
                        log_message(f"All subtasks completed at step {t}", log_file)
                        break

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    # Drain any remaining queued actions (best effort, mirrors the MBR script).
    if not success and len(action_queue) > 0:
        log_message(f"Executing {len(action_queue)} remaining actions in queue...", log_file)
        while len(action_queue) > 0:
            try:
                robot_action, _, _ = split_openpi_action(action_queue.popleft())
                obs, reward, done, info = env.step(robot_action.tolist())
            except ValueError:
                break  # Environment already terminated
            if done:
                success = True
                log_message("Environment signaled done while executing remaining actions", log_file)
                break
            replay_images.append(get_libero_image(obs))
            replay_wrist_images.append(get_libero_wrist_image(obs))
            replay_subtasks.append(f"finishing: {current_state}")

    return success, replay_images, replay_wrist_images, replay_subtasks


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    client: OpenPiClient,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, "openpi", resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0

    # Two-stage workflow: scan the transit baseline's rollout videos to find
    # which episodes it failed. The full method re-runs only those (the guard
    # in validate_config guarantees these videos exist unless --rerun_all).
    rerun_dict = get_failed_episodes_from_videos(cfg, log_file)

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        # Only re-run episodes the transit baseline failed (unless --rerun_all).
        # Episode numbering matches the transit script: the video filename uses
        # the post-increment `total_episodes`, so we test `total_episodes + 1`.
        rerun = cfg.rerun_all or (total_episodes + 1 in rerun_dict[cfg.task_suite_name])

        if rerun:
            log_message(f"Starting episode {task_episodes + 1}...", log_file)
            success, replay_images, replay_wrist_images, replay_subtasks = run_episode(
                cfg, env, task_description, client, initial_state, log_file
            )
        else:
            # Transit baseline already succeeded here -- count as success, do not re-run.
            log_message(f"Episode {task_episodes + 1} passed the transit baseline; skipping.", log_file)
            success = True

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay videos (front + wrist) only for episodes we actually ran.
        if rerun:
            save_rollout_video_decomposed(
                replay_images, total_episodes, success=success, task_description=task_description,
                video_save_dir=os.path.join(cfg.video_save_dir, cfg.task_suite_name), log_file=log_file,
                subtasks=replay_subtasks,
            )
            save_rollout_video_decomposed(
                replay_wrist_images, total_episodes, success=success, task_description=task_description,
                video_save_dir=os.path.join(cfg.video_save_dir, cfg.task_suite_name), log_file=log_file,
                subtasks=replay_subtasks, wrist=True,
            )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate the openpi CycleVLA policy on LIBERO tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Connect to the openpi policy server (blocks until the server is up)
    client = OpenPiClient(host=cfg.host, port=cfg.port)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg, task_suite, task_id, client, total_episodes, total_successes, log_file
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_success_rate, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
