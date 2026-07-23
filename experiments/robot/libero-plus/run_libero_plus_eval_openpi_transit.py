"""
run_libero_plus_eval_openpi_transit.py

Baseline subtask-transit-only eval for the LIBERO-Plus robustness benchmark
(openpi / pi0.5 backbone). LIBERO-Plus counterpart of
`experiments/robot/libero/run_libero_eval_openpi_transit.py`.

Stage 1 of the two-stage LIBERO-Plus workflow: it drives the same 9-dim policy
as `..._openpi_cyclevla.py` but with no VLM, no backtrack, and no MBR —
subtasks advance purely on the policy's stop signal (`stop > 0.5`) — and writes
per-episode rollout videos that Stage 2 (`..._openpi_cyclevla.py`) scans to
re-run only the episodes this baseline failed.

Architecture: the pi0.5 policy is served remotely by openpi over a websocket;
all openpi-vs-OpenVLA convention differences are handled in
`experiments/robot/openpi_utils.py`:
  - No local model: actions come from an openpi websocket policy server.
  - openpi emits the raw RLDS gripper and raw stop/progress floats, so there is
    no `process_action` / gripper inversion -- we threshold `stop > 0.5`.

LIBERO-Plus specifics (see experiments/robot/libero_plus_utils.py):
  * The chosen `--task_suite_name` (libero_spatial/object/goal/10) now contains
    ~2,400 perturbed task *variants* across 7 categories; `--category` restricts
    the run to one category, `--eval_fraction` sub-samples it.
  * `num_trials_per_task = 1` (the LIBERO-Plus paper protocol).
  * The CycleVLA FSM is run on the *canonical* task instruction recovered per
    task (filename-derived for the 6 non-Language categories; GPT-matched for
    the Language category).
  * Output goes under `rollouts-plus/`.

Usage (with the policy server already running -- see
`openpi/scripts/serve_openpi_cyclevla.sh`):

  conda activate /hdd2/chenyang/openvla-oft/env-plus
  CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero-plus/run_libero_plus_eval_openpi_transit.py \
      --host 0.0.0.0 --port 8000 --task_suite_name libero_spatial --category camera
"""

import json
import logging
import os
import sys
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

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

# LIBERO-Plus-specific helpers: perturbation-category lookup, canonical-task recovery
# (incl. the Language-category GPT matcher), variant sub-sampling, and the sidecar
# progress-file helpers that let an interrupted run resume where it left off.
from experiments.robot.libero_plus_utils import (
    PLUS_CATEGORY_COUNTS,
    category_slug,
    env_render_resolution,
    load_or_init_progress,
    normalize_category,
    resolve_canonical_task,
    rollout_label,
    save_progress,
    select_plus_tasks,
)

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
    # LIBERO-Plus environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Underlying suite (libero_spatial/object/goal/10)
    category: str = "all"                            # Perturbation category: all | camera | robot | language | light | background | noise | layout
    eval_fraction: int = 100                         # Sub-sample %: 10..100 (step 10). 100 = full LIBERO-Plus protocol
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                     # LIBERO-Plus protocol: 1 deterministic rollout per variant
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 1024                          # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./rollouts-plus/logs_plus_openpi_transit"   # Local directory for eval logs
    video_save_dir: str = "./rollouts-plus/rollouts_plus_openpi_transit"
    # Directory holding the per-suite Language-category instruction->canonical-task cache.
    # Shared with the cyclevla script (same default) on purpose: the two stages reuse each
    # other's GPT-match results and resolve every rewrite to the identical canonical task.
    instruction_cache_dir: str = "./experiments/robot/libero-plus"

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
    normalize_category(cfg.category)
    assert 10 <= cfg.eval_fraction <= 100 and cfg.eval_fraction % 10 == 0, \
        f"eval_fraction must be an integer in 10..100 (step 10); got {cfg.eval_fraction}"


def _default_log_filename(cfg: GenerateConfig) -> str:
    """Build the EVAL-*.txt basename for a fresh run."""
    run_id = (f"EVAL-{cfg.task_suite_name}-{category_slug(normalize_category(cfg.category)) if cfg.category != 'all' else 'all'}"
              f"-frac{cfg.eval_fraction}-openpi-{DATE_TIME}")
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id + ".txt"


def setup_logging(cfg: GenerateConfig, log_filename: str, log_mode: str):
    """Set up logging to file and optionally to wandb.

    `log_mode` is "w" for fresh, "a" for resume.
    """
    run_id = os.path.splitext(log_filename)[0]
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, log_filename)
    log_file = open(local_log_filepath, log_mode)
    logger.info(f"Logging to local log file ({'append' if log_mode == 'a' else 'write'}): {local_log_filepath}")

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
                break

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
                    break

                # Robust stop-signal detection. openpi emits a raw stop float
                # (~1.0 = stop); the confirmation logic matches the OpenVLA
                # script's pattern, but with `> 0.5` instead of `== -1`.
                if stop_signal > 0.5:
                    consecutive_high_count += 1

                    if not first_high_seen:
                        first_high_seen = True
                        log_message(
                            f"Stop signal observed at step {t} for subtask: {current_state} "
                            f"(stop={stop_signal:.2f}, progress={progress_signal:.2f}).",
                            log_file,
                        )

                    if consecutive_high_count >= 2:
                        log_message(
                            f"Finished subtask: {current_state} at step {t} "
                            f"({consecutive_high_count} consecutive stop signals, "
                            f"latest stop={stop_signal:.2f}, progress={progress_signal:.2f}).",
                            log_file,
                        )
                        break
                    elif first_high_seen and steps_since_last_high >= 2:
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
                        steps_since_last_high = 0
                else:
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
    category: str,
    client: OpenPiClient,
    total_episodes=0,
    total_successes=0,
    category_stats=None,
    log_file=None,
):
    """Run evaluation for a single LIBERO-Plus task variant.

    `category` is the variant's perturbation category; `category_stats` is a
    {category: [episodes, successes]} dict updated in place for per-category reporting.
    """
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment. The LIBERO-Plus env applies the perturbation automatically
    # from the task name. `task.language` is filename-derived/dirty for LIBERO-Plus, so we
    # discard it and recover the canonical FSM-style instruction explicitly below.
    # The Noise category must render at 256 (its corruptions are 256-bound); the other 6
    # categories use cfg.env_img_res (default 1024).
    env, _ = get_libero_env(
        task, "openpi",
        resolution=env_render_resolution(category, cfg.env_img_res),
    )
    task_description = resolve_canonical_task(
        task.name, category, env, cfg.task_suite_name, cfg.instruction_cache_dir,
        log_fn=lambda m: log_message(m, log_file),
    )
    # Label for the saved rollout video: the original rewritten instruction for the
    # Language category (so each video shows its real input), else the canonical task.
    video_label = rollout_label(category, env, task_description)

    # Per-category output sub-directory, e.g. rollouts-plus/.../libero_spatial/Camera/
    video_dir = os.path.join(cfg.video_save_dir, cfg.task_suite_name, category_slug(category))

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

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

        success, replay_images, replay_subtasks = run_episode(
            cfg, env, task_description, client, initial_state, log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        if category_stats is not None:
            category_stats[category][0] += 1
            if success:
                category_stats[category][1] += 1

        save_rollout_video_decomposed(
            replay_images, total_episodes, success=success, task_description=video_label,
            video_save_dir=video_dir, log_file=log_file, subtasks=replay_subtasks
        )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

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
    """Main function to evaluate the openpi CycleVLA policy on LIBERO-Plus tasks."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Initialize the LIBERO-Plus task suite (re-uses the standard 4 suite names; each
    # now holds ~2,400 perturbed variants). We do this BEFORE connecting to the server
    # because `select_plus_tasks` is cheap and its length lets us short-circuit an
    # already-complete (suite, category) run without paying the connection cost.
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    # Select the variants to evaluate: filter by perturbation category and sub-sample.
    # The cyclevla script calls select_plus_tasks with the same args -> identical task order,
    # which is what makes the sidecar-progress-file resume safe (see libero_plus_utils).
    selected = select_plus_tasks(
        task_suite, cfg.task_suite_name, cfg.category, cfg.eval_fraction, cfg.seed
    )
    num_selected = len(selected)

    # Consult the progress sidecar to see if a partially-completed run exists for this
    # exact (task_suite, category, eval_fraction, seed) combo.
    default_log_fn = _default_log_filename(cfg)
    (start_index, total_episodes, total_successes, prior_category_stats,
     log_filename, log_mode) = load_or_init_progress(cfg, default_log_fn)

    log_file, local_log_filepath, run_id = setup_logging(cfg, log_filename, log_mode)

    log_message(
        f"Task suite: {cfg.task_suite_name} | category: {cfg.category} | "
        f"eval_fraction: {cfg.eval_fraction}% | evaluating {num_selected} / "
        f"{task_suite.n_tasks} variants",
        log_file,
    )
    if log_mode == "a":
        log_message(
            f"===== RESUMING at task index {start_index}/{num_selected} "
            f"(restored total_episodes={total_episodes}, total_successes={total_successes}) =====",
            log_file,
        )

    category_stats = defaultdict(lambda: [0, 0])
    for k, v in prior_category_stats.items():
        category_stats[k] = list(v)

    if start_index >= num_selected:
        log_message(
            f"Nothing to do: {start_index}/{num_selected} tasks already complete on disk.",
            log_file,
        )
    else:
        # Connect to the openpi policy server (blocks until the server is up)
        client = OpenPiClient(host=cfg.host, port=cfg.port)

        for i, (task_id, category) in enumerate(tqdm.tqdm(
            selected[start_index:], initial=start_index, total=num_selected
        )):
            total_episodes, total_successes = run_task(
                cfg, task_suite, task_id, category, client,
                total_episodes, total_successes, category_stats, log_file,
            )

            save_progress(
                cfg, start_index + i + 1, num_selected,
                total_episodes, total_successes, category_stats, log_filename,
            )

    # Final results
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    log_message("Per-category success rates:", log_file)
    for category in sorted(category_stats):
        episodes, successes = category_stats[category]
        rate = successes / episodes if episodes > 0 else 0
        full_total = PLUS_CATEGORY_COUNTS.get(cfg.task_suite_name, {}).get(category)
        suffix = f"  [full suite: {full_total} variants]" if full_total else ""
        log_message(
            f"  {category}: {successes}/{episodes} ({rate * 100:.1f}%){suffix}", log_file
        )

    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_success_rate, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
