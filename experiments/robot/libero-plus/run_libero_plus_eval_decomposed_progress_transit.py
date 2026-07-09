"""
run_libero_plus_eval_decomposed_progress_transit.py

Baseline subtask-transit-only eval for the LIBERO-Plus robustness benchmark
(OpenVLA-OFT backbone). LIBERO-Plus counterpart of
`experiments/robot/libero/run_libero_eval_decomposed_progress_transit.py`.

Stage 1 of the two-stage LIBERO-Plus workflow: it drives the same 9-dim policy
as `..._mbr.py` but with no VLM, no backtrack, and no MBR — subtasks advance
purely on the policy's stop signal (`termination_signal == -1`) — and writes
per-episode rollout videos that Stage 2 (`..._mbr.py`) scans to re-run only the
episodes this baseline failed.

LIBERO-Plus specifics (see experiments/robot/libero_plus_utils.py):
  * The chosen `--task_suite_name` (libero_spatial/object/goal/10) now contains
    ~2,400 perturbed task *variants* across 7 categories; `--category` restricts
    the run to one category, `--eval_fraction` sub-samples it.
  * `num_trials_per_task = 1` (the LIBERO-Plus paper protocol).
  * The CycleVLA FSM is run on the *canonical* task instruction recovered per
    task (filename-derived for the 6 non-Language categories; GPT-matched for
    the Language category).
  * Output goes under `rollouts-plus/`.
"""

# watch -n 1 nvidia-smi
# conda activate openvla-oft-plus

# CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero-plus/run_libero_plus_eval_decomposed_progress_transit.py   --pretrained_checkpoint <CKPT>   --task_suite_name libero_spatial   --category camera

import json
import logging
import os
import sys
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")
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
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

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
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = False                  # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = True                       # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

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
    local_log_dir: str = "./rollouts-plus/logs_plus_decomposed_progress_transit"   # Local directory for eval logs
    video_save_dir: str = "./rollouts-plus/rollouts_plus_decomposed_progress_transit"
    # Directory holding the per-suite Language-category instruction->canonical-task cache.
    # Shared with the mbr script (same default) on purpose: the two stages reuse each
    # other's GPT-match results and resolve every rewrite to the identical canonical task.
    instruction_cache_dir: str = "./experiments/robot/libero-plus"

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 0                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite (LIBERO-Plus re-uses the 4 standard suite names)
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    # Validate the LIBERO-Plus perturbation category (raises on an unknown value).
    normalize_category(cfg.category)

    # Validate the sub-sampling fraction.
    assert 10 <= cfg.eval_fraction <= 100 and cfg.eval_fraction % 10 == 0, \
        f"eval_fraction must be an integer in 10..100 (step 10); got {cfg.eval_fraction}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name
    if "decomposed" in cfg.pretrained_checkpoint:
        unnorm_key = "libero_decomposed"
    if "decomposed" in cfg.pretrained_checkpoint and "oversample" in cfg.pretrained_checkpoint:
        unnorm_key = "libero_decomposed_oversample"
    elif "decomposed" in cfg.pretrained_checkpoint and "progress" in cfg.pretrained_checkpoint:
        unnorm_key = "libero_decomposed_progress"

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def _default_log_filename(cfg: GenerateConfig) -> str:
    """Build the EVAL-*.txt basename for a fresh run. Kept as its own helper so
    `load_or_init_progress` can be passed it as the fallback filename when no sidecar
    progress file exists yet."""
    run_id = (f"EVAL-{cfg.task_suite_name}-{category_slug(normalize_category(cfg.category)) if cfg.category != 'all' else 'all'}"
              f"-frac{cfg.eval_fraction}-{cfg.model_family}-{DATE_TIME}")
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id + ".txt"


def setup_logging(cfg: GenerateConfig, log_filename: str, log_mode: str):
    """Set up logging to file and optionally to wandb.

    The caller resolves `log_filename` (either the newly-generated one from
    `_default_log_filename` for a fresh run, or the one persisted in `.progress.json`
    on resume) so the same EVAL-*.txt gets appended to across restarts — keeping
    aggregate_plus_logs.py seeing exactly one log per (suite, category). `log_mode` is
    "w" for fresh, "a" for resume.
    """
    # The wandb run name is the filename without the ".txt" suffix — matches the pre-refactor behaviour.
    run_id = os.path.splitext(log_filename)[0]

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, log_filename)
    log_file = open(local_log_filepath, log_mode)
    logger.info(f"Logging to local log file ({'append' if log_mode == 'a' else 'write'}): {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family, stop=False, progress=False):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    # print(action)
    action = normalize_gripper_action(action, binarize=True, stop=stop, progress=progress)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    # (0 = go, 1 = stop), flip it back (-1 = stop, +1 = go) following the gripper
    if model_family == "openvla":
        action = invert_gripper_action(action, stop=stop, progress=progress)

    return action.tolist()


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Get subtasks
    states = pick_place_states(task_description, f"{cfg.task_suite_name}_no_noops")
    states = complex_states(task_description, f"{cfg.task_suite_name}_no_noops") if states is None else states
    states = [state.lower() for state in states]
    subtask_list = [f"Task: {task_description}. The current subtask: {state}" for state in states]

    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images, replay_subtasks = [], []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Progress signal threshold
    progress_threshold = 0.97

    # Run episode
    success = False
    try:
        for subtask_idx, (current_subtask, current_state) in enumerate(zip(subtask_list, states)):
            if success:
                break  # Exit full task loop once done is True

            t = 0
            # Progress signal tracking - reset for each subtask
            first_high_seen = False
            consecutive_high_count = 0
            steps_since_last_high = 0

            # Prevent carry-over actions from previous subtask
            action_queue.clear()
            
            while t < max_steps // len(states) * 1.5 + cfg.num_steps_wait:
            # while t < max_steps + cfg.num_steps_wait:
                if t == 0:
                    log_message(f"Starting subtask: {current_state}", log_file)

                # Do nothing for the first few timesteps to let objects stabilize
                if t < cfg.num_steps_wait and subtask_idx == 0:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # Prepare observation
                observation, img = prepare_observation(obs, resize_size)
                replay_images.append(img)
                replay_subtasks.append(current_state)

                # If action queue is empty, requery model
                if len(action_queue) == 0:
                    actions = get_action(
                        cfg,
                        model,
                        observation,
                        current_state,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                    action_queue.extend(actions)

                # Get and process next action
                action = process_action(action_queue.popleft(), cfg.model_family, stop=True, progress=True)
                print(action)

                # Step environment
                termination_signal = action[-2]
                progress_signal = action[-1]
                obs, reward, done, info = env.step(action[:-2])
                t += 1

                if done:
                    success = True
                    log_message(f"Finished subtask: {current_state} in {t} steps", log_file)
                    break  # Exit subtask loop

                # Enhanced progress signal logic
                if termination_signal == -1:
                    consecutive_high_count += 1

                    if not first_high_seen:
                        first_high_seen = True
                        log_message(
                            f"Termination signal observed at step {t} for subtask: {current_state} "
                            f"(term={termination_signal}, progress={progress_signal:.2f}).",
                            log_file,
                        )

                    # Break conditions (re-using previous robustness pattern)
                    if consecutive_high_count >= 2:
                        # Consecutive STOP confirmations
                        log_message(
                            f"Finished subtask: {current_state} at step {t} "
                            f"({consecutive_high_count} consecutive termination signals, "
                            f"latest term={termination_signal}, progress={progress_signal:.2f}).",
                            log_file,
                        )
                        break
                    elif first_high_seen and steps_since_last_high >= 2:
                        # STOP recurs after at least 2 "low" steps
                        log_message(
                            f"Finished subtask: {current_state} at step {t} "
                            f"(termination re-confirmed after {steps_since_last_high} low steps; "
                            f"term={termination_signal}, progress={progress_signal:.2f}).",
                            log_file,
                        )
                        break
                    else:
                        if first_high_seen and steps_since_last_high > 0:
                            log_message(
                                f"Ignoring termination re-signal at step {t} "
                                f"(only {steps_since_last_high} low steps since last; need 2+).",
                                log_file,
                            )
                        # Reset the low-signal counter on a high
                        steps_since_last_high = 0
                else:
                    # Low (no termination), maintain robustness counters
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
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    category_stats=None,
    log_file=None,
):
    """Run evaluation for a single LIBERO-Plus task variant.

    `category` is the variant's perturbation category; `category_stats` is a
    {category: [episodes, successes]} dict updated in place for per-category reporting.
    """
    # Get task variant
    task = task_suite.get_task(task_id)

    # Get initial states (LIBERO-Plus's get_task_init_states strips perturbation suffixes)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment. The LIBERO-Plus env applies the perturbation automatically
    # from the task name. `task.language` is filename-derived/dirty for LIBERO-Plus, so we
    # discard it and recover the canonical FSM-style instruction explicitly below.
    # The Noise category must render at 256 (its corruptions are 256-bound); the other 6
    # categories use cfg.env_img_res (default 1024).
    env, _ = get_libero_env(
        task, cfg.model_family,
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

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, replay_subtasks = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Update per-category stats
        if category_stats is not None:
            category_stats[category][0] += 1
            if success:
                category_stats[category][1] += 1

        # Save replay video under the per-category sub-directory
        save_rollout_video_decomposed(
            replay_images, total_episodes, success=success, task_description=video_label,
            video_save_dir=video_dir, log_file=log_file, subtasks=replay_subtasks
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

    # Log to wandb if enabled
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
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize the LIBERO-Plus task suite (re-uses the standard 4 suite names; each
    # now holds ~2,400 perturbed variants). We do this BEFORE model init because
    # `select_plus_tasks` is cheap and its length lets us short-circuit an already-complete
    # (suite, category) run without paying the model-load cost.
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    # Select the variants to evaluate: filter by perturbation category and sub-sample.
    # The mbr script calls select_plus_tasks with the same args -> identical task order,
    # which is what makes the sidecar-progress-file resume safe (see libero_plus_utils).
    selected = select_plus_tasks(
        task_suite, cfg.task_suite_name, cfg.category, cfg.eval_fraction, cfg.seed
    )
    num_selected = len(selected)

    # Consult the progress sidecar to see if a partially-completed run exists for this
    # exact (task_suite, category, eval_fraction, seed) combo. On resume this hands back
    # the original log filename so the same EVAL-*.txt is appended to.
    default_log_fn = _default_log_filename(cfg)
    (start_index, total_episodes, total_successes, prior_category_stats,
     log_filename, log_mode) = load_or_init_progress(cfg, default_log_fn)

    # Setup logging (append mode on resume, write mode on fresh run)
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

    # Rebuild per-category accounting as a defaultdict (autovivify for any category we
    # haven't seen yet, e.g. when resuming into a new suite of `--category all`).
    category_stats = defaultdict(lambda: [0, 0])
    for k, v in prior_category_stats.items():
        category_stats[k] = list(v)

    if start_index >= num_selected:
        # Nothing left to do — this (suite, category) run was already complete on disk.
        # Fall through to the final-summary block so aggregate_plus_logs sees consistent
        # totals in the appended log tail even for a no-op resume.
        log_message(
            f"Nothing to do: {start_index}/{num_selected} tasks already complete on disk.",
            log_file,
        )
    else:
        # Only load the model when there's actual work — a repeated resume-into-a-complete-run
        # otherwise wastes minutes on model init just to log "nothing to do".
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        resize_size = get_image_resize_size(cfg)

        # Main loop, resumed from `start_index`. `initial`+`total` give tqdm the correct
        # global bar even though we only iterate the suffix of `selected`.
        for i, (task_id, category) in enumerate(tqdm.tqdm(
            selected[start_index:], initial=start_index, total=num_selected
        )):
            total_episodes, total_successes = run_task(
                cfg,
                task_suite,
                task_id,
                category,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                total_episodes,
                total_successes,
                category_stats,
                log_file,
            )

            # Persist progress after each completed task so a crash between iterations
            # loses at most one task. `start_index + i + 1` is the count of completed
            # items = position we'd resume from next time.
            save_progress(
                cfg, start_index + i + 1, num_selected,
                total_episodes, total_successes, category_stats, log_filename,
            )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Per-category success-rate table
    log_message("Per-category success rates:", log_file)
    for category in sorted(category_stats):
        episodes, successes = category_stats[category]
        rate = successes / episodes if episodes > 0 else 0
        full_total = PLUS_CATEGORY_COUNTS.get(cfg.task_suite_name, {}).get(category)
        suffix = f"  [full suite: {full_total} variants]" if full_total else ""
        log_message(
            f"  {category}: {successes}/{episodes} ({rate * 100:.1f}%){suffix}", log_file
        )

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()