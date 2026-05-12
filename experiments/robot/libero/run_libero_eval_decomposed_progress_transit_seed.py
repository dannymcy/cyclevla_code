"""
run_libero_eval_decomposed_progress_transit_seed.py

Seed-sweep harness that produces the trajectory data consumed by
`run_mbr_analysis.py`.

Same eval loop as `..._transit.py` (9-dim policy, stop-signal subtask
transitions, no VLM / no backtrack), plus `save_trajectory_json(...)`
(line ~348) which dumps per-seed (state, action, progress) JSONs under
`rollouts/.../`. A `--rerun_episodes` flag lets you re-roll a specific
{task_suite: [episode_ids]} set. The aggregated JSONs are then crunched
by `run_mbr_analysis.py` into Random / MBR / r-NN statistics matrices.
"""

# watch -n 1 nvidia-smi
# conda activate /hdd2/kai/openvla-oft/env

# CUDA_VISIBLE_DEVICES="2" python experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.py   --pretrained_checkpoint /hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--50000_chkpt   --task_suite_name libero_spatial
# CUDA_VISIBLE_DEVICES="2" python experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.py   --pretrained_checkpoint /hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--50000_chkpt   --task_suite_name libero_object
# CUDA_VISIBLE_DEVICES="2" python experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.py   --pretrained_checkpoint /hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--50000_chkpt   --task_suite_name libero_goal
# CUDA_VISIBLE_DEVICES="2" python experiments/robot/libero/run_libero_eval_decomposed_progress_transit_seed.py   --pretrained_checkpoint /hdd2/kai/openvla-oft/checkpoints/libero/libero_sub_decomposed_progress_A100/openvla-7b+libero_decomposed_progress+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--diffusion--3rd_person_img--wrist_img--proprio_state--50000_chkpt   --task_suite_name libero_10

import json
import logging
import os
import sys
import gc
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from collections import defaultdict
import ast

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed to register 3D plots
from matplotlib.lines import Line2D
from scipy.spatial.distance import cosine, cdist, correlation
from scipy.stats import mannwhitneyu, wasserstein_distance
from itertools import combinations
import pandas as pd
from sklearn.covariance import OAS
from scipy.spatial.transform import Rotation as R

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

    rerun_episodes: Optional[str] = None             # JSON dict of task_suite -> episode list, e.g., '{"libero_spatial": [1,2], "libero_object": [3,4]}'

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task (50)
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs/logs_sub_decomposed_progress_transit_seed_225000_chkpt"        # Local directory for eval logs
    video_save_dir: str = "./rollouts/rollouts_sub_decomposed_progress_transit_seed_225000_chkpt"

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

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


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


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

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


def save_trajectory_json(
    seed,
    task_description,
    success,
    replay_subtasks,
    robot_states,  # List of (pos, quat, gripper) tuples
    actions,  # List of action vectors
    progress_signals,
    json_dir
):
    """Save trajectory data paired with subtasks including actions."""
    
    # Identify subtask transitions
    subtask_segments = []
    current_subtask = None
    start_idx = 0
    
    for i, subtask in enumerate(replay_subtasks):
        if subtask != current_subtask:
            if current_subtask is not None:
                subtask_segments.append({
                    "name": current_subtask,
                    "start": start_idx,
                    "end": i-1,
                    "states": robot_states[start_idx:i],
                    "actions": actions[start_idx:i]  # Save actions too
                })
            current_subtask = subtask
            start_idx = i
    
    # Add final segment
    if current_subtask:
        subtask_segments.append({
            "name": current_subtask,
            "start": start_idx,
            "end": len(replay_subtasks)-1,
            "states": robot_states[start_idx:],
            "actions": actions[start_idx:]  # Save actions too
        })
    
    trajectory_data = {
        "metadata": {
            "seed": seed,
            "task": task_description,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        },
        "subtask_segments": subtask_segments,
        "progress_signals": progress_signals,
        "actions": actions,  # Save full action sequence
        "total_steps": len(robot_states)
    }
    
    json_path = os.path.join(json_dir, f"traj_seed_{seed:03d}.json")
    with open(json_path, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    return json_path


def visualize_task_trajectories(json_dir, task_name):
    """One image, 6 different 3D views (2x3 grid) for XYZ (from states).
       PLUS an additional image with the same layout for rotation actions (action[3:6]).
       - EXACT subtask names (only .strip()).
       - Per-subtask base shade index (t) stable within the task.
       - Success -> Greens(t +/- delta); Failure -> Reds(t +/- delta).
       - Positions figure: end-effector XYZ from `states`.
       - Rotations figure: action rotation components from `actions[:, 3:6]` (model units).
    """

    # ---------- helpers ----------
    def _extract_xyz(states):
        """Extract end-effector XYZ from states: [pos(3), quat(4), gripper] or dict."""
        xyz = []
        for s in states:
            if isinstance(s, dict):
                pos = s.get("pos") or s.get("eef_pos") or s.get("position")
            else:
                pos = s[0] if (isinstance(s, (list, tuple, np.ndarray)) and len(s) > 0) else None
            if pos is None:
                continue
            pos = np.asarray(pos, dtype=float)
            if pos.shape[0] >= 3:
                xyz.append(pos[:3])
        return np.asarray(xyz, dtype=float) if xyz else np.empty((0, 3), dtype=float)

    def _extract_action_rot(actions):
        """Extract the rotation components from actions: action[:, 3:6]."""
        if actions is None:
            return np.empty((0, 3), dtype=float)
        A = np.asarray(actions, dtype=float)
        if A.ndim != 2 or A.shape[1] < 6:
            return np.empty((0, 3), dtype=float)
        return A[:, 3:6]

    def _set_equal_3d(ax, X, Y, Z):
        """Equal aspect ratio for 3D axes."""
        if not (len(X) and len(Y) and len(Z)):
            return
        x_min, x_max = np.min(X), np.max(X)
        y_min, y_max = np.min(Y), np.max(Y)
        z_min, z_max = np.min(Z), np.max(Z)
        rng = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-9)
        cx, cy, cz = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2
        half = rng / 2
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)

    def _clip01(x): return float(np.clip(x, 0.0, 1.0))

    # ---------- load ----------
    json_files = glob.glob(os.path.join(json_dir, "traj_seed_*.json"))
    if not json_files:
        print(f"No trajectory files found in {json_dir}")
        return

    trajectories = []
    for jf in sorted(json_files):
        with open(jf, 'r') as f:
            trajectories.append(json.load(f))

    # ---------- collect EXACT subtasks & segments ----------
    # keys: (subtask_exact_name, outcome) where outcome in {"success","failure"}
    segments_by_key_pos = {}  # for XYZ plotting (from states)
    segments_by_key_rot = {}  # for rotation plotting (from actions[:,3:6])
    success_subtasks, failure_subtasks = set(), set()

    for traj in trajectories:
        outcome = "success" if traj["metadata"].get("success", False) else "failure"
        traj_actions_full = traj.get("actions", None)  # full action sequence (fallback if segment lacks 'actions')
        for seg in traj.get("subtask_segments", []):
            name_exact = (seg.get("name") or "").strip()  # ONLY strip whitespace
            if not name_exact:
                continue

            # positions from states
            P = _extract_xyz(seg.get("states", []))

            # rotations from actions: prefer per-segment actions; fallback to slicing full
            seg_actions = seg.get("actions", None)
            Rv = _extract_action_rot(seg_actions)
            if Rv.size == 0 and traj_actions_full is not None:
                # try slicing full actions by segment indices
                s0 = seg.get("start", None); s1 = seg.get("end", None)
                if s0 is not None and s1 is not None and isinstance(s0, int) and isinstance(s1, int):
                    slice_actions = np.asarray(traj_actions_full, dtype=float)
                    s1c = min(s1 + 1, slice_actions.shape[0])
                    if 0 <= s0 < s1c:
                        Rv = _extract_action_rot(slice_actions[s0:s1c])

            if P.size == 0 and Rv.size == 0:
                continue

            if outcome == "success":
                success_subtasks.add(name_exact)
            else:
                failure_subtasks.add(name_exact)

            key = (name_exact, outcome)
            if P.size:
                segments_by_key_pos.setdefault(key, []).append(P)
            if Rv.size:
                segments_by_key_rot.setdefault(key, []).append(Rv)

    # Universe of subtasks within THIS task dir
    all_subtasks = sorted(success_subtasks | failure_subtasks)
    K = max(1, len(all_subtasks))

    # Stable base shade index per subtask (within Greens/Reds)
    base_t_values = np.linspace(0.35, 0.9, K)
    subtask_to_t = {s: base_t_values[i] for i, s in enumerate(all_subtasks)}

    # ---------- gather for equal aspect ----------
    def _gather_all_xyz(segdict):
        all_X, all_Y, all_Z = [], [], []
        for lst in segdict.values():
            for P in lst:
                all_X.append(P[:, 0]); all_Y.append(P[:, 1]); all_Z.append(P[:, 2])
        if all_X and all_Y and all_Z:
            return np.concatenate(all_X), np.concatenate(all_Y), np.concatenate(all_Z)
        return None, None, None

    # ---------- shared plotter (multi-view 2x3) ----------
    def _plot_multiview(segments_by_key, title_prefix, outfile, axis_labels=('X', 'Y', 'Z')):
        view_angles = [
            (30, 45),    # default perspective
            (90, 0),     # top-down (XY)
            (0, 0),      # front (XZ)
            (0, 90),     # side (YZ)
            (20, 135),   # rear diagonal
            (110, 45),   # underside
        ]
        fig = plt.figure(figsize=(22, 14))  # large canvas
        axes = []
        for i, (elev, azim) in enumerate(view_angles, 1):
            ax = fig.add_subplot(2, 3, i, projection='3d')
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"View {i}: elev={elev}, azim={azim}", fontsize=11, fontweight='bold')
            ax.set_xlabel(axis_labels[0]); ax.set_ylabel(axis_labels[1]); ax.set_zlabel(axis_labels[2])
            ax.grid(True, alpha=0.25)
            axes.append(ax)

        # draw with per-subtask shade variation and outcome color family
        for subtask in all_subtasks:
            t_base = subtask_to_t[subtask]
            for outcome, cmap, alpha, lw in (
                ("success", plt.cm.Greens, 0.9, 1.9),
                ("failure", plt.cm.Reds,   0.75, 1.7),
            ):
                key = (subtask, outcome)
                segs = segments_by_key.get(key, [])
                if not segs:
                    continue
                n = len(segs)
                t_vals = np.array([t_base]) if n == 1 else np.linspace(
                    _clip01(t_base - 0.12), _clip01(t_base + 0.12), n
                )
                for P, t in zip(segs, t_vals):
                    color = cmap(t)
                    for ax in axes:
                        ax.plot(P[:, 0], P[:, 1], P[:, 2],
                                color=color, alpha=alpha, linewidth=lw, linestyle='-')

        # equal aspect for all subplots
        X, Y, Z = _gather_all_xyz(segments_by_key)
        if X is not None:
            for ax in axes:
                _set_equal_3d(ax, X, Y, Z)

        # legends
        if success_subtasks:
            succ_handles = [mpatches.Patch(color=plt.cm.Greens(subtask_to_t[s]), label=s)
                            for s in sorted(success_subtasks)]
            ncol = 2 if len(succ_handles) > 12 else 1
            leg_s = axes[0].legend(handles=succ_handles, title="Subtasks (SUCCESS, Greens)",
                                   loc='upper left', bbox_to_anchor=(-0.05, 1.25),
                                   frameon=True, fancybox=True, shadow=True,
                                   ncol=ncol, fontsize=9, title_fontsize=10)
            axes[0].add_artist(leg_s)

        if failure_subtasks:
            fail_handles = [mpatches.Patch(color=plt.cm.Reds(subtask_to_t[s]), label=s)
                            for s in sorted(failure_subtasks)]
            ncol = 2 if len(fail_handles) > 12 else 1
            axes[0].legend(handles=fail_handles, title="Subtasks (FAILURE, Reds)",
                           loc='upper right', bbox_to_anchor=(1.15, 1.25),
                           frameon=True, fancybox=True, shadow=True,
                           ncol=ncol, fontsize=9, title_fontsize=10)

        outcome_handles = [
            Line2D([0], [0], color=plt.cm.Greens(0.7), lw=2.2, ls='-', label='Success (Greens)'),
            Line2D([0], [0], color=plt.cm.Reds(0.7),   lw=2.2, ls='-', label='Failure (Reds)'),
        ]
        axes[1].legend(handles=outcome_handles, loc='upper right', fontsize=8, frameon=True)

        # title & save
        success_count = sum(1 for t in trajectories if t["metadata"].get("success", False))
        total_count = len(trajectories)
        rate = (success_count / total_count * 100.0) if total_count else 0.0
        fig.suptitle(
            f"{title_prefix} — Task: {task_name} — Seeds: {total_count} — Success: {success_count}/{total_count} ({rate:.1f}%)",
            fontsize=16, fontweight='bold', y=0.99
        )
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = os.path.join(json_dir, outfile)
        plt.savefig(out_path, dpi=240, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    # ---------- PLOT 1: Positions (XYZ from states) — unchanged ----------
    _plot_multiview(
        segments_by_key=segments_by_key_pos,
        title_prefix="End-effector displacement (XYZ)",
        outfile="trajectory_visualization_3d_multiview.png",
        axis_labels=('X', 'Y', 'Z'),
    )

    # ---------- PLOT 2: Rotation actions (action[3:6]) ----------
    # Units are in your policy's action space (often radians for axis-angle components in many setups,
    # but since this is model-dependent, we label axes generically).
    _plot_multiview(
        segments_by_key=segments_by_key_rot,
        title_prefix="End-effector rotation action (action[3:6])",
        outfile="trajectory_visualization_rot_action_3d_multiview.png",
        axis_labels=('Rot-X', 'Rot-Y', 'Rot-Z'),
    )


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
    seed=None,
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
    set_seed_everywhere(seed)

    # Track robot states, actions, and progress
    robot_states = []
    actions_executed = []
    progress_signals = []
            
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

                # Save the full action vector INCLUDING THIS
                actions_executed.append(action.tolist() if isinstance(action, np.ndarray) else action)

                # Step environment
                termination_signal = action[-2]
                progress_signal = action[-1]
                obs, reward, done, info = env.step(action[:-2])
                t += 1

                # After stepping environment, record state
                robot_states.append([
                    obs["robot0_eef_pos"].tolist(),
                    obs["robot0_eef_quat"].tolist(),
                    float(obs["robot0_gripper_qpos"][0])
                ])
                
                progress_signals.append(float(progress_signal))

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

    return success, replay_images, replay_subtasks, robot_states, actions_executed, progress_signals


def check_trajectory_exists(json_dir, seed):
    """Check if trajectory JSON already exists for this seed."""
    json_path = os.path.join(json_dir, f"traj_seed_{seed:03d}.json")
    return os.path.exists(json_path)


def get_rerun_pairs(rerun_list, original_num_trials=10):
    """Convert original total_episodes indices to (task_id, episode_idx) pairs."""
    pairs = set()
    for total_ep in rerun_list:
        task_id = (total_ep - 1) // original_num_trials
        episode_idx = (total_ep - 1) % original_num_trials
        pairs.add((task_id, episode_idx))
    return pairs


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    seed=None,
    last_seed=127
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Get json_dir to check for existing trajectories
    task_clean = task_description.replace(" ", "_")
    json_dir = os.path.join(cfg.video_save_dir, cfg.task_suite_name, task_clean)

    # Per-task counters
    task_episodes = 0
    task_successes = 0

    # Track if this task ran and its result
    task_ran = False
    task_success = False

    # Check if trajectory already exists for this seed
    if check_trajectory_exists(json_dir, seed):
        log_message(f"Trajectory already exists for seed {seed}, task {task_description}. Loading from JSON...", log_file)

        # Load the existing trajectory to get success status
        json_path = os.path.join(json_dir, f"traj_seed_{seed:03d}.json")
        with open(json_path, 'r') as f:
            existing_traj = json.load(f)
        success = existing_traj["metadata"]["success"]

        # Mark as ran and record result
        task_ran = True
        task_success = success

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        log_message(f"Loaded existing result - Success: {success}", log_file)

        # If this is the last seed, still create visualization
        if seed == last_seed:
            log_message(f"Creating visualization for task {task_description}...", log_file)
            visualize_task_trajectories(json_dir, task_description)
            log_message(f"Visualization complete for {task_description}", log_file)
    else:
        try:
            # Start episodes
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

                # Parse rerun episodes from config (if specified)
                if cfg.rerun_episodes:
                    rerun_dict = ast.literal_eval(cfg.rerun_episodes)
                    if cfg.task_suite_name in rerun_dict:
                        rerun_pairs = get_rerun_pairs(rerun_dict[cfg.task_suite_name], original_num_trials=cfg.num_trials_per_task)
                        rerun = (task_id, episode_idx) in rerun_pairs
                        
                        if not rerun:
                            continue  # Skip non-rerun episodes

                # Only count and run if this is a rerun episode
                log_message(f"Starting episode {task_episodes + 1}...", log_file)

                task_episodes += 1
                total_episodes += 1

                is_last_seed = (seed == last_seed)

                # Run the episode
                success, replay_images, replay_subtasks, robot_states, actions_executed, progress_signals = run_episode(
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
                    seed,
                )

                # Mark as ran and record result
                task_ran = True
                task_success = success

                # Save video
                _, _, json_dir = save_rollout_video_decomposed(
                    replay_images, total_episodes, success=success,
                    task_description=task_description,
                    video_save_dir=os.path.join(cfg.video_save_dir, cfg.task_suite_name),
                    log_file=log_file, subtasks=replay_subtasks, save=True, seed=seed
                )

                # Save trajectory JSON
                json_path = save_trajectory_json(
                    seed, task_description, success,
                    replay_subtasks, robot_states, actions_executed,
                    progress_signals, json_dir
                )
                log_message(f"Saved trajectory to: {json_path}", log_file)

                if is_last_seed:
                    log_message(f"Creating visualization for task {task_description}...", log_file)
                    visualize_task_trajectories(json_dir, task_description)
                    log_message(f"Visualization complete for {task_description}", log_file)

                if success:
                    task_successes += 1
                    total_successes += 1

                # Log results
                log_message(f"Success: {success}", log_file)
                log_message(f"# episodes completed so far: {total_episodes}", log_file)
                log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

        finally:
            # Clean up environment
            try:
                if 'env' in locals() and hasattr(env, 'close'):
                    env.close()
            finally:
                if 'env' in locals():
                    del env
                gc.collect()

    # Log task results (only if task ran)
    if task_ran:
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

        log_message(f"Current task success rate: {task_success_rate}", log_file)
        log_message(f"Current total success rate: {total_success_rate}", log_file)

        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": task_success_rate,
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    return total_episodes, total_successes, task_description, task_ran, task_success


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    overall_total_episodes, overall_total_successes = 0, 0

    # Per-task tracking (across all seeds)
    per_task_stats = defaultdict(lambda: {"episodes": 0, "successes": 0, "task_description": None})

    # --- Run each seed 0..127 ---
    seed_num = 128
    for seed in range(seed_num):
        cfg.seed = seed
        log_message(f"\n========== Seed {seed} ==========", log_file)

        # Reset per-seed totals
        total_episodes, total_successes = 0, 0

        for task_id in tqdm.tqdm(range(num_tasks), desc=f"seed {seed}"):
            total_episodes, total_successes, task_desc, task_ran, task_success = run_task(
                cfg,
                task_suite,
                task_id,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                total_episodes,
                total_successes,
                log_file,
                seed,
                seed_num - 1
            )

            # Track per-task stats (only if task actually ran)
            if task_ran:
                per_task_stats[task_id]["episodes"] += 1
                per_task_stats[task_id]["task_description"] = task_desc
                if task_success:
                    per_task_stats[task_id]["successes"] += 1

        seed_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        log_message(f"Seed {seed} — episodes: {total_episodes}, successes: {total_successes}, "
                    f"success rate: {seed_success_rate:.4f} ({seed_success_rate*100:.1f}%)", log_file)

        if cfg.use_wandb:
            wandb.log({
                "seed": seed,
                "success_rate/seed_total": seed_success_rate,
                "num_episodes/seed_total": total_episodes,
            })

        overall_total_episodes += total_episodes
        overall_total_successes += total_successes

    # Log per-task results
    log_message("\n========== Per-task results across all seeds ==========", log_file)
    for task_id in sorted(per_task_stats.keys()):
        stats = per_task_stats[task_id]
        task_desc = stats["task_description"] or f"Task {task_id}"
        episodes = stats["episodes"]
        successes = stats["successes"]
        rate = float(successes) / float(episodes) if episodes > 0 else 0.0
        log_message(
            f"Task {task_id} ({task_desc}): {successes}/{episodes} "
            f"({rate:.4f}, {rate*100:.1f}%)",
            log_file
        )

        if cfg.use_wandb:
            wandb.log({
                f"success_rate/task_{task_id}": rate,
                f"num_episodes/task_{task_id}": episodes,
            })

    # Final aggregated results across all seeds
    final_success_rate = (
        float(overall_total_successes) / float(overall_total_episodes)
        if overall_total_episodes > 0 else 0.0
    )

    log_message("\n========== Final aggregated results across seeds ==========", log_file)
    log_message(f"Total episodes: {overall_total_episodes}", log_file)
    log_message(f"Total successes: {overall_total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/overall": final_success_rate,
                "num_episodes/overall": overall_total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return final_success_rate


def analyze_trajectories(cfg: GenerateConfig) -> None:
    """
    Analyze trajectories by mimicking the inference selection process.
    
    Paper experiment:
    - Sample x seeds (4/8/16/32/64) for y=200 times
    - Run MBR on x seeds vs randomly choose one
    - Compare probability that chosen seed is successful
    - Draw with replacement cross trials
    - Use different distance metrics (l2, l1, cosine, correlation, chebyshev)
    """
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    all_results = {}

    # Determine valid top-k values
    valid_top_k = (1, 3)

    # For each task
    for task_id in range(num_tasks):
        excel_data = []  # For Excel export
        task = task_suite.get_task(task_id)
        env_tmp, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        try:
            pass
        finally:
            try:
                if hasattr(env_tmp, "close"):
                    env_tmp.close()
            except Exception:
                pass
            del env_tmp

        task_clean = task_description.replace(" ", "_")
        json_dir = os.path.join(cfg.video_save_dir, cfg.task_suite_name, task_clean)

        if not os.path.exists(json_dir):
            print(f"Skipping {task_description} - no data found")
            continue

        print(f"\nAnalyzing task: {task_description}")

        # Load all trajectory JSONs
        json_files = glob.glob(os.path.join(json_dir, "traj_seed_*.json"))
        if not json_files:
            continue

        trajectories = []
        for jf in sorted(json_files):
            with open(jf, 'r') as f:
                trajectories.append(json.load(f))

        # Separate success and failure trajectories
        success_trajs = [t for t in trajectories if t["metadata"]["success"]]
        failure_trajs = [t for t in trajectories if not t["metadata"]["success"]]
        print(f"  Found {len(success_trajs)} success, {len(failure_trajs)} failure trajectories")

        # Process into 8-step segments
        def segment_trajectory(traj, segment_size=8):
            """Segment trajectory into chunks of segment_size."""
            segments = []
            progress = traj.get("progress_signals", [])
            for seg in traj.get("subtask_segments", []):
                seg_states = seg.get("states", [])
                seg_actions = seg.get("actions", [])
                subtask_name = seg.get("name", "unknown")

                for i in range(0, len(seg_states), segment_size):
                    state_chunk = seg_states[i:i+segment_size]
                    action_chunk = seg_actions[i:i+segment_size] if seg_actions else []
                    progress_chunk = progress[i:i+segment_size] if i < len(progress) else []

                    if len(state_chunk) < segment_size:
                        continue

                    positions = np.array([
                        s[0] if isinstance(s, list) else s.get("pos", [0, 0, 0])
                        for s in state_chunk
                    ])[:, :3]

                    quaternions_wxyz = np.array([
                        s[1] if isinstance(s, list) else s.get("quat", [1, 0, 0, 0])
                        for s in state_chunk
                    ], dtype=float)

                    # Reorder to xyzw
                    quats_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]

                    # Normalize to unit quaternions (robust)
                    norm = np.linalg.norm(quats_xyzw, axis=1, keepdims=True)
                    norm = np.where(norm == 0, 1.0, norm)
                    quats_xyzw = quats_xyzw / norm

                    # Convert to Euler
                    rotations = R.from_quat(quats_xyzw).as_euler('xyz', degrees=True)

                    avg_progress = np.mean(progress_chunk) if progress_chunk else 0.0

                    segments.append({
                        "timestep_start": i,
                        "subtask": subtask_name,
                        "positions": positions.flatten(),
                        "rotations": rotations.flatten(),
                        "combined": np.concatenate([positions.flatten(), rotations.flatten()]),
                        "progress": avg_progress,
                    })
            return segments

        # Segment all trajectories
        success_segments_by_time = defaultdict(list)
        failure_segments_by_time = defaultdict(list)

        for traj in success_trajs:
            for seg in segment_trajectory(traj):
                success_segments_by_time[seg["timestep_start"]].append(seg)

        for traj in failure_trajs:
            for seg in segment_trajectory(traj):
                failure_segments_by_time[seg["timestep_start"]].append(seg)

        def simulate_selection_process(success_segs, failure_segs, num_trials=200):
            """
            Simulate MBR vs Random selection process.
            
            Paper setup:
            - x = sample_sizes (4/8/16/32/64 hypotheses)
            - y = num_trials (200 repetitions)
            - Draw with replacement across trials
            - Within each trial: sample WITHOUT replacement (unique trajectories
            - Compare MBR vs random selection
            """
            # Need minimum data to run analysis
            if len(failure_segs) < 3 or len(success_segs) < 5:
                return None

            # Results structure (no anchor strategies needed - MBR is anchor-free)
            results = defaultdict(lambda: defaultdict(list))

            features = ["positions", "rotations", "combined"]
            sample_sizes = [4, 8, 16, 32, 64]
            metrics = ["l2", "l1", "cosine", "correlation", "chebyshev"]
            
            metric_map = {
                "l2": "euclidean",
                "l1": "cityblock",
                "cosine": "cosine",
                "correlation": "correlation",
                "chebyshev": "chebyshev"
            }

            # Combined pool: (label, segment) where label=1 for success, 0 for failure
            combined = [(1, seg) for seg in success_segs] + [(0, seg) for seg in failure_segs]

            for trial in range(num_trials):
                if trial % 5 == 0:
                    print(f"    Trial {trial}/{num_trials}...")

                for feature in features:
                    for sample_size in sample_sizes:
                        # Sample x trajectories WITH REPLACEMENT (as per paper)
                        num_samples = min(sample_size, len(combined))
                        if num_samples < 2:
                            continue
                        sampled_idx = np.random.choice(len(combined), num_samples, replace=False)
                        sampled = [combined[i] for i in sampled_idx]

                        labels = [lbl for lbl, _ in sampled]
                        X = np.stack([seg[feature] for _, seg in sampled])
                        N = X.shape[0]

                        def any_success_top_k(order, k):
                            k_eff = min(k, len(order)) # For n=4/8, top-10 becomes top-4/top-8
                            if k_eff == 0:
                                return False
                            return any(labels[i] == 1 for i in order[:k_eff])


                        # -------- Random baseline --------
                        random_order = np.random.permutation(N)
                        for k in valid_top_k:
                            results[feature][f"random_n{sample_size}_top{k}"].append(
                                any_success_top_k(random_order, k)
                            )

                        # -------- MBR and r-NN for each metric --------
                        for mname in metrics:
                            dist_mat = cdist(X, X, metric=metric_map[mname])

                            # Standard MBR: average distance to all
                            avg_dist = dist_mat.mean(axis=1)
                            mbr_rep_order = np.argsort(avg_dist)         # most typical (min avg dist)
                            mbr_away_order = np.argsort(avg_dist)[::-1]  # most atypical (max avg dist)

                            for k in valid_top_k:
                                results[feature][f"mbr_{mname}(rep)_n{sample_size}_top{k}"].append(
                                    any_success_top_k(mbr_rep_order, k)
                                )
                                results[feature][f"mbr_{mname}(away)_n{sample_size}_top{k}"].append(
                                    any_success_top_k(mbr_away_order, k)
                                )

                            # r-NN density-based
                            r = max(2, min(4, int(np.sqrt(N))))
                            r_eff = min(r, max(1, N - 1))

                            # r-NN radius (distance to r-th nearest neighbor)
                            rnn_radius = np.partition(dist_mat, r_eff, axis=1)[:, r_eff]

                            # Find pocket center (smallest r-NN radius = densest region)
                            center_idx = int(np.argmin(rnn_radius))
                            order_center = np.argsort(dist_mat[center_idx])
                            cluster_idx = order_center[:r_eff]

                            # Find medoid inside pocket
                            intra = dist_mat[np.ix_(cluster_idx, cluster_idx)]
                            medoid_local = cluster_idx[int(np.argmin(intra.mean(axis=1)))]

                            # Distance to medoid for orderings
                            d_to_medoid = dist_mat[medoid_local]

                            # Representative: medoid first, then closest to medoid
                            rep_order = np.concatenate([[medoid_local],
                                                        np.argsort(np.where(np.arange(N) == medoid_local,
                                                                            np.inf, d_to_medoid))])
                            # Exploratory: farthest from medoid first
                            away_order = np.argsort(d_to_medoid)[::-1]

                            for k in valid_top_k:
                                results[feature][f"rnn_{mname}(rep)_n{sample_size}_top{k}"].append(
                                    any_success_top_k(rep_order, k)
                                )
                                results[feature][f"rnn_{mname}(away)_n{sample_size}_top{k}"].append(
                                    any_success_top_k(away_order, k)
                                )

            # Aggregate results
            aggregated = {}
            for feature in features:
                aggregated[feature] = {}
                for key, values in results[feature].items():
                    if values:
                        aggregated[feature][key] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "count": len(values)
                        }
            return aggregated

        # Analyze all timesteps
        timestep_analysis = {}
        timesteps = sorted(set(success_segments_by_time.keys()) | set(failure_segments_by_time.keys()))
        for t in timesteps:
            success_segs = success_segments_by_time.get(t, [])
            failure_segs = failure_segments_by_time.get(t, [])
            if success_segs and failure_segs:
                print(f"  Analyzing timestep {t} ({len(success_segs)} success, {len(failure_segs)} failure segments)...")
                simulation_results = simulate_selection_process(success_segs, failure_segs)
                if simulation_results:
                    timestep_analysis[t] = simulation_results

                # Collect data for Excel
                if simulation_results:
                    base_mets = ["l2", "l1", "cosine", "correlation", "chebyshev"]
                    sample_sizes_export = [4, 8, 16, 32, 64]

                    # Build metric keys
                    metric_keys = []
                    
                    # Random baseline (metric-independent)
                    for n in sample_sizes_export:
                        metric_keys.append(f"random_n{n}")
                    
                    # MBR and r-NN (metric-dependent)
                    for m in base_mets:
                        for n in sample_sizes_export:
                            metric_keys += [
                                f"mbr_{m}(rep)_n{n}",
                                f"mbr_{m}(away)_n{n}",
                                f"rnn_{m}(rep)_n{n}",
                                f"rnn_{m}(away)_n{n}",
                            ]

                    for feature, f_results in simulation_results.items():
                        for metric in metric_keys:
                            for k in valid_top_k:
                                key = f"{metric}_top{k}"
                                if key in f_results:
                                    excel_data.append({
                                        "Task": task_description[:30],
                                        "Success_Rate": f"{len(success_trajs)/len(trajectories)*100:.1f}%",
                                        "Timestep": t,
                                        "Feature": feature,
                                        "Metric": metric.upper(),
                                        f"Top-{k}_Prob": f"{f_results[key]['mean']*100:.1f}%",
                                        f"Top-{k}_Std": f"{f_results[key]['std']*100:.1f}%",
                                    })

        # Summaries
        def summarize_results():
            """Compute summary statistics across all timesteps."""
            summary = {}

            # Find best metric/feature combination
            best_config = {
                "metric": None,
                "feature": None,
                "timestep": None,
                "top1_prob": 0
            }

            for t, t_results in timestep_analysis.items():
                for feature, f_results in t_results.items():
                    for metric_key in list(f_results.keys()):
                        if metric_key.endswith("_top1"):
                            prob = f_results[metric_key]["mean"]
                            if prob > best_config["top1_prob"]:
                                best_config.update({
                                    "metric": metric_key.replace("_top1", ""),
                                    "feature": feature,
                                    "timestep": t,
                                    "top1_prob": prob
                                })

            # Average performance by period
            periods = {
                "early": [t for t in timesteps if t < 40],
                "mid": [t for t in timesteps if 40 <= t < 80],
                "late": [t for t in timesteps if t >= 80]
            }
            period_performance = {}
            for period_name, period_times in periods.items():
                period_probs = []
                for t in period_times:
                    if t in timestep_analysis:
                        if "combined" in timestep_analysis[t]:
                            if "mbr_l2(rep)_n8_top1" in timestep_analysis[t]["combined"]:
                                period_probs.append(
                                    timestep_analysis[t]["combined"]["mbr_l2(rep)_n8_top1"]["mean"]
                                )
                if period_probs:
                    period_performance[period_name] = {
                        "mean_top1_prob": float(np.mean(period_probs)),
                        "std_top1_prob": float(np.std(period_probs))
                    }

            summary["best_configuration"] = best_config
            summary["temporal_evolution"] = period_performance
            return summary

        # Store results
        task_results = {
            "task_name": task_description,
            "num_success": len(success_trajs),
            "num_failure": len(failure_trajs),
            "success_rate": len(success_trajs) / len(trajectories) if trajectories else 0,
            "timestep_analysis": timestep_analysis,
            "summary": summarize_results()
        }
        all_results[task_description] = task_results

        # Save task-specific analysis as JSON
        task_analysis_path = os.path.join(json_dir, "selection_simulation_analysis.json")
        with open(task_analysis_path, 'w') as f:
            json.dump(task_results, f, indent=2)
        print(f"  Saved analysis to {task_analysis_path}")

        # Save Excel file with all results
        if excel_data:
            excel_path = os.path.join(cfg.video_save_dir, cfg.task_suite_name, f"trajectory_analysis_{task_clean}.xlsx")

            # Create DataFrame and save with multiple sheets
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Raw data
                df_main = pd.DataFrame(excel_data)
                df_main.to_excel(writer, sheet_name='Raw_Data', index=False)

                # Sheet 2: Summary by Task
                task_summary = []
                for task_name, results in all_results.items():
                    if "summary" in results and "best_configuration" in results["summary"]:
                        best = results["summary"]["best_configuration"]
                        task_summary.append({
                            "Task": task_name[:40],
                            "Success_Rate": f"{results['success_rate']*100:.1f}%",
                            "Best_Feature": best['feature'],
                            "Best_Metric": best['metric'],
                            "Best_Timestep": best['timestep'],
                            "Best_Top1_Prob": f"{best['top1_prob']*100:.1f}%"
                        })
                if task_summary:
                    pd.DataFrame(task_summary).to_excel(writer, sheet_name='Task_Summary', index=False)

                # Sheet 3: Averaged by Feature-Metric combination
                if excel_data:
                    df_pivot = df_main.copy()
                    top_k_cols = [col for col in df_pivot.columns if col.startswith('Top-') and col.endswith('_Prob')]
                    for col in top_k_cols:
                        df_pivot[col] = df_pivot[col].str.rstrip('%').astype(float)
                    agg_dict = {col: 'mean' for col in top_k_cols}
                    avg_results = df_pivot.groupby(['Feature', 'Metric']).agg(agg_dict).round(1)
                    avg_results.to_excel(writer, sheet_name='Feature_Metric_Averages')

                # Sheet 4: Time-step analysis
                df_time = df_main.copy()
                top_k_cols = [col for col in df_time.columns if col.startswith('Top-') and col.endswith('_Prob')]
                for col in top_k_cols:
                    df_time[col] = df_time[col].str.rstrip('%').astype(float)
                agg_dict = {col: ['mean', 'std', 'count'] for col in top_k_cols}
                time_agg = df_time.groupby(['Task', 'Timestep', 'Feature', 'Metric']).agg(agg_dict)

                time_agg.columns = [f"{c[0]}_{c[1].capitalize()}" for c in time_agg.columns.to_flat_index()]
                time_agg = time_agg.reset_index()
                for col in time_agg.columns:
                    if col.startswith('Top-') and ('Mean' in col or 'Std' in col):
                        time_agg[col] = time_agg[col].round(1)
                time_agg.to_excel(writer, sheet_name='Time_Step_Analysis', index=False)

                # Format columns
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except Exception:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width

            print(f"\nSaved Excel analysis to: {excel_path}")

        # Print summary
        print("\n" + "="*80)
        print("SELECTION SIMULATION ANALYSIS SUMMARY")
        print("="*80)
        for task_name, results in all_results.items():
            print(f"\n{task_name} (Success rate: {results['success_rate']*100:.1f}%)")
            if "summary" in results and "best_configuration" in results["summary"]:
                best = results["summary"]["best_configuration"]
                print(f"  Best configuration:")
                print(f"    Feature: {best['feature']}")
                print(f"    Metric: {best['metric']}")
                print(f"    Timestep: {best['timestep']}")
                print(f"    Top-1 success probability: {best['top1_prob']*100:.1f}%")



if __name__ == "__main__":
    eval_libero()

    # Run trajectory analysis on saved JSONs
    @draccus.wrap()
    def run_analysis(cfg: GenerateConfig):
        analyze_trajectories(cfg)
    
    run_analysis()