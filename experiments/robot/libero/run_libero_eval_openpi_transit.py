"""
run_libero_eval_openpi_transit.py

Baseline subtask-transit-only LIBERO eval for the **pi05 / openpi** CycleVLA
policy.

This is the openpi counterpart of `run_libero_eval_decomposed_progress_transit.py`:
same transit-only protocol (no VLM, no backtrack, no MBR), but the 9-dim policy
is served remotely by openpi instead of run in-process. Subtasks are advanced
purely on the policy's stop signal.

Architecture differences vs. the OpenVLA-OFT script (all isolated in
`experiments/robot/openpi_utils.py`):
  - No local model: actions come from an openpi websocket policy server.
  - openpi emits the raw RLDS gripper and raw stop/progress floats, so there is
    no `process_action` / gripper inversion -- we threshold `stop > 0.5`.

Usage (the eval env, with the policy server already running -- see
`openpi/serve_openpi_cyclevla.sh`):

  conda activate /hdd2/kai/openvla-oft/env
  python experiments/robot/libero/run_libero_eval_openpi_transit.py \
      --host 0.0.0.0 --port 8000 --task_suite_name libero_spatial
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append repo root so the interpreter can find `experiments.robot`.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(ROOT_DIR)

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    save_rollout_video_decomposed,
)
from experiments.robot.openpi_utils import OpenPiClient, split_openpi_action
from experiments.robot.robot_utils import DATE_TIME, set_seed_everywhere

from fsm_utils.build import *
from fsm_utils.utils import *


def pick_place_states(language_instruction, task_suite):
    """Decompose a pick-and-place task into ordered subtask strings via the FSM."""
    if task_suite == "libero_spatial_no_noops":
        pick_object_full, pick_object_simple, place_object = extract_pick_place_libero_spatial(language_instruction)
        fsm, states = fsm_pick_place_libero_spatial(pick_object_full, pick_object_simple, place_object)
    elif task_suite == "libero_object_no_noops":
        pick_object, place_object = extract_pick_place_libero_object(language_instruction)
        fsm, states = fsm_pick_place_libero_object(pick_object, place_object)
    elif task_suite == "libero_goal_no_noops":
        pick_object, place_object = extract_pick_place_libero_goal(language_instruction)
        if pick_object is not None:
            fsm, states = fsm_pick_place_libero_goal(pick_object, place_object)
        else:
            return None
    elif task_suite == "libero_10_no_noops":
        pick_object, place_object = extract_pick_place_libero_10(language_instruction)
        if pick_object is not None:
            fsm, states = fsm_pick_place_libero_10(pick_object, place_object)
        else:
            return None

    return states


def complex_states(language_instruction, task_suite):
    """Fallback subtask decomposition for non-pick-place tasks."""
    fsm, states = fsm_complex_libero(language_instruction, task_suite)
    return states


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
    local_log_dir: str = "./experiments/logs/logs_openpi_transit"   # Local directory for eval logs
    video_save_dir: str = "./rollouts/rollouts_openpi_transit"

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


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    run_id = f"EVAL-{cfg.task_suite_name}-openpi-transit-{DATE_TIME}"
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


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    client: OpenPiClient,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Get subtasks. The bare lowercased subtask string is the prompt the openpi
    # policy was trained with (`prompt_from_task=True`; the Stage-3 RLDS builder
    # writes `chunk['subtask'].lower()` into `language_instruction`).
    states = pick_place_states(task_description, f"{cfg.task_suite_name}_no_noops")
    states = complex_states(task_description, f"{cfg.task_suite_name}_no_noops") if states is None else states
    states = [state.lower() for state in states]

    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Open-loop action queue, refilled from the policy server when empty.
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    replay_images, replay_subtasks = [], []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        for subtask_idx, current_state in enumerate(states):
            if success:
                break  # Exit full task loop once done is True

            t = 0
            # Stop-signal robustness counters - reset for each subtask
            first_high_seen = False
            consecutive_high_count = 0
            steps_since_last_high = 0

            # Prevent carry-over actions from previous subtask
            action_queue.clear()

            while t < max_steps // len(states) * 1.5 + cfg.num_steps_wait:
                if t == 0:
                    log_message(f"Starting subtask: {current_state}", log_file)

                # Do nothing for the first few timesteps to let objects stabilize
                if t < cfg.num_steps_wait and subtask_idx == 0:
                    obs, reward, done, info = env.step(get_libero_dummy_action("openpi"))
                    t += 1
                    continue

                # Record replay frame (raw rotated agentview image at env res)
                replay_images.append(get_libero_image(obs))
                replay_subtasks.append(current_state)

                # If action queue is empty, requery the openpi policy server
                if len(action_queue) == 0:
                    actions = client.get_action(obs, current_state, cfg.num_open_loop_steps)
                    action_queue.extend(actions)

                # Split the 9-dim openpi action: dims 0-6 are env-ready (raw
                # gripper, no inversion), dims 7-8 are the raw stop/progress.
                robot_action, stop_signal, progress_signal = split_openpi_action(action_queue.popleft())

                # Step environment with the 7-dim robot action only
                obs, reward, done, info = env.step(robot_action.tolist())
                t += 1

                if done:
                    success = True
                    log_message(f"Finished subtask: {current_state} in {t} steps", log_file)
                    break  # Exit subtask loop

                # Robust stop-signal detection. openpi emits a raw stop float
                # (~1.0 = stop); the OpenVLA script's `== -1` test becomes
                # `> 0.5` here. The confirmation logic is otherwise identical.
                if stop_signal > 0.5:
                    consecutive_high_count += 1

                    if not first_high_seen:
                        first_high_seen = True
                        log_message(
                            f"Stop signal observed at step {t} for subtask: {current_state} "
                            f"(stop={stop_signal:.2f}, progress={progress_signal:.2f}).",
                            log_file,
                        )

                    # Break conditions
                    if consecutive_high_count >= 2:
                        # Consecutive STOP confirmations
                        log_message(
                            f"Finished subtask: {current_state} at step {t} "
                            f"({consecutive_high_count} consecutive stop signals, "
                            f"latest stop={stop_signal:.2f}, progress={progress_signal:.2f}).",
                            log_file,
                        )
                        break
                    elif first_high_seen and steps_since_last_high >= 2:
                        # STOP recurs after at least 2 "low" steps
                        log_message(
                            f"Finished subtask: {current_state} at step {t} "
                            f"(stop re-confirmed after {steps_since_last_high} low steps; "
                            f"stop={stop_signal:.2f}, progress={progress_signal:.2f}).",
                            log_file,
                        )
                        break
                    else:
                        if first_high_seen and steps_since_last_high > 0:
                            log_message(
                                f"Ignoring stop re-signal at step {t} "
                                f"(only {steps_since_last_high} low steps since last; need 2+).",
                                log_file,
                            )
                        # Reset the low-signal counter on a high
                        steps_since_last_high = 0
                else:
                    # Low (no stop), maintain robustness counters
                    consecutive_high_count = 0
                    if first_high_seen:
                        steps_since_last_high += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images, replay_subtasks


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

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, replay_subtasks = run_episode(
            cfg, env, task_description, client, initial_state, log_file
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video_decomposed(
            replay_images, total_episodes, success=success, task_description=task_description,
            video_save_dir=os.path.join(cfg.video_save_dir, cfg.task_suite_name), log_file=log_file,
            subtasks=replay_subtasks,
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
