"""
run_libero_plus_eval_decomposed_progress_mbr.py

Full CycleVLA evaluation pipeline for the LIBERO-Plus robustness benchmark
(OpenVLA-OFT backbone). LIBERO-Plus counterpart of
`experiments/robot/libero/run_libero_eval_decomposed_progress_mbr.py`.

Stage 2 of the two-stage LIBERO-Plus workflow. Uses the 9-dim subtask-aware
policy to step through subtasks via the stop + progress signals. At ~90%
progress per subtask, a VLM is queried with `type: transit | backtrack`; on
`backtrack` the env is physically rewound to the start of the failing subtask,
N seeds are resampled, and the best trajectory is selected via Minimum Bayes
Risk (MBR). Per-component latencies are tracked.

As in the LIBERO eval, this re-runs ONLY the episodes the transit baseline
(`..._transit.py`) failed — it scans the baseline's rollout videos under
`--video_base_dir`. Pass `--rerun_all True` to skip the baseline and evaluate
every selected variant fresh.

LIBERO-Plus specifics (see experiments/robot/libero_plus_utils.py):
  * `--task_suite_name` (libero_spatial/object/goal/10) holds ~2,400 perturbed
    variants across 7 categories; `--category` restricts the run to one,
    `--eval_fraction` sub-samples it. The transit and mbr scripts MUST be run
    with the same `--category` / `--eval_fraction` / `--seed` so episode
    numbering lines up for the rerun mechanism.
  * `num_trials_per_task = 1` (the LIBERO-Plus paper protocol).
  * The CycleVLA FSM is run on the *canonical* task instruction recovered per
    task (filename-derived for the 6 non-Language categories; GPT-matched for
    the Language category).
  * Output goes under `rollouts-plus/`.
"""

# watch -n 1 nvidia-smi
# conda activate openvla-oft-plus

# CUDA_VISIBLE_DEVICES="0" python experiments/robot/libero-plus/run_libero_plus_eval_decomposed_progress_mbr.py   --pretrained_checkpoint <CKPT>   --task_suite_name libero_spatial   --category camera   --video_base_dir <transit VIDEO_DIR>

import json
import time
import logging
import os
import sys
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

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

import re
import io
import cv2
import base64
from dotenv import load_dotenv
from openai import OpenAI
from fsm_utils.build import *
from fsm_utils.utils import *


# Load environment variables from .env file
load_dotenv()

class VLMDetector:
    def __init__(self, model_name="gpt-5.5", temperature=1):

        api_key = os.environ['OPENAI_API_KEY']
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

        # If in region that cannot access OpenAI API
        # api_key = os.environ.get('ZENMUX_OPENAI_API_KEY')
        # if not api_key:
        #     raise ValueError("ZenMux API key not found. Set ZENMUX_OPENAI_API_KEY environment variable.")
        # self.client = OpenAI(
        #     api_key=api_key,
        #     base_url="https://zenmux.ai/api/v1"  # Add this
        # )
        # self.model_name = model_name
        # self.temperature = temperature

    def encode_image(self, input_img):
        if input_img is None:
            raise ValueError("Image loading failed.")

        # because using cv2, to convert from rgb to bgr[s1_score, s2_score]
        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

        success, encoded_image = cv2.imencode('.png', input_img)
        if not success:
            raise ValueError("Image encoding failed.")
        image_bytes = io.BytesIO(encoded_image).read()
        return f'data:image/png;base64,{base64.b64encode(image_bytes).decode("utf-8")}'
    
    def extract_res(self, output_text):
        lines = output_text.strip().splitlines()
        subtask, type_str, reason = None, None, None

        for line in lines:
            if line.lower().startswith("next_subtask:"):
                subtask = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("type:"):
                type_str = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("reason:"):
                reason = line.split(":", 1)[1].strip()

        if not subtask or not type_str or not reason:
            raise ValueError("Failed to extract subtask or type from the response.")

        return subtask, type_str, reason

    def detect_subtask(self, current_subtask, subtasks, subtask_history, language_instruction, obv, obv_wrist):
        obv_encoded = self.encode_image(obv)
        wrist_encoded = self.encode_image(obv_wrist)

        output_format = f"""
        ""Write in the following format. Output nothing else:
        next_subtask: <exact subtask from subtasks list>
        type: <transit / backtrack>
        reason: <explanation>
        """

        prompt = f"""
        You are an expert robot behavior annotator. Decide what the robot should do next given it is ~90% through the current subtask.
        Your job is to FORECAST whether the current subtask will likely succeed if we continue without corrective repositioning.

        Inputs:
        1) Task instruction: {language_instruction}
        2) Subtask list: {subtasks}
        3) Current subtask: {current_subtask}
        4) Visual inputs (two synchronized views):
        - FRONT: third-person view (global alignment, object identity, spatial relations)
        - WRIST: close-up gripper view (detailed contact, local geometry, physical affordances)

        Decision rule (forecasting at ~90%):
        - Default to **transit** when success appears reasonably likely within the next few actions **without** corrective repositioning.
        - Choose **backtrack** if strong, unambiguous visual evidence indicates that the subtask will fail without repositioning.

        View-specific fusion instruction:
        - FRONT view provides **global context**: object identity, pose, global alignment, reachability, and path clearance.
        - WRIST view provides **local interaction cues**: gripper orientation, contact points, slip, stability, and detailed positioning relative to affordances.
        - Combine both views to reason about **functional success**: whether the current configuration supports the intended physical interaction (e.g., grasping, pulling, pushing).
        - FRONT dominates for global spatial reasoning and goal reachability.
        - WRIST dominates for local contact accuracy and grasp quality.

        Affordance reasoning guidance:
        - Evaluate whether the gripper's pose is **consistent with the object's intended use**:
        - For a bowl: grasping the **edge or rim** is acceptable and often intended for lifting.
        - For a drawer: alignment with the **handle** is the key indicator of readiness.
        - For a push or pull action: confirm direction and surface contact match the required motion.
        - Do not penalize partial or asymmetric contacts if they serve a valid affordance and appear stable.
        
        Wrong object or wrong subtask detection:
        In addition to misalignment, detect late-stage “silent failures” involving **wrong object engagement or wrong subtask execution**.
        If visual evidence indicates the gripper is interacting with an unintended object, target, or affordance
        (e.g., lifting or contacting a distractor, manipulating the wrong receptacle, or committing to a different subtask's goal),
        or that the intended object/site remains unaffected while another changes,
        output `type: backtrack` and set `next_subtask` to the earliest subtask that restores correct target selection
        and preconditions (typically a reach, align, or target-identification step, NOT a trivial open/close gripper).

        Backtracking target:
        - Do NOT backtrack to a trivial "open gripper" or "close gripper" subtask.
        - Backtrack to the **earliest** subtask that restores the missing precondition
        (typically a positioning or alignment step that enables correct affordance engagement).

        Output format (STRICT; keep keys exactly):
        next_subtask: <exact subtask from subtasks list>
        type: <transit / backtrack>
        reason: <explain in a concise paragraph, justifying the decision based on predicted execution success and task/subtask correctness>

        front_view_evidence:
        - <concise observable cue 1>
        - <concise observable cue 2>
        - <concise observable cue 3>
        - <concise observable cue 4>

        wrist_view_evidence:
        - <concise observable cue 1>
        - <concise observable cue 2>
        - <concise observable cue 3>
        - <concise observable cue 4>

        assessment:
        - success_likelihood: <high | medium | low>
        - key_risks: <comma-separated brief phrases>
        - view_agreement: <agree | partial | disagree>; <short phrase on which view dominates and why>
        - decision_basis: <short phrase linking likelihood + dominant cues to decision>

        Constraints:
        - Focus strictly on **observable** visual and physical evidence.
        - Keep each bullet concise (≤12 words).
        - Use **exact** strings from `subtasks` for `next_subtask`.
        - `type` must be either transit or backtrack.
        - Return only the specified fields; no extra commentary.

        Now produce the decision.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": obv_encoded, "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": wrist_encoded, "detail": "high"}},
                    {"type": "text", "text": output_format},
                ]
            }
        ]

        # Call to OpenAI API
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )

        # Process API response
        response_content = completion.choices[0].message.content.strip()
        return response_content


@dataclass
class EpisodeLatency:
    """Track latency for a single episode."""
    vlm_detector: float = 0.0
    action_inference: float = 0.0
    sampling: float = 0.0
    mbr_computation: float = 0.0
    backtracking: float = 0.0
    
    def add(self, component: str, latency: float):
        current = getattr(self, component)
        setattr(self, component, current + latency)


@dataclass
class LatencyTracker:
    """Track latency across all episodes."""
    episodes: List[EpisodeLatency] = field(default_factory=list)
    current_episode: EpisodeLatency = field(default_factory=EpisodeLatency)
    
    def add_to_current(self, component: str, latency: float):
        """Add latency to current episode."""
        self.current_episode.add(component, latency)
    
    def finish_episode(self):
        """Finish current episode and start a new one."""
        self.episodes.append(self.current_episode)
        self.current_episode = EpisodeLatency()
    
    def log_episode_summary(self, episode_num: int, log_file):
        """Log the current episode's latencies."""
        log_message(f"\n===== Episode {episode_num} Latency Breakdown =====", log_file)
        log_message(f"  VLM detector: {self.current_episode.vlm_detector:.3f}s", log_file)
        log_message(f"  Action inference: {self.current_episode.action_inference:.3f}s", log_file)
        log_message(f"  Sampling: {self.current_episode.sampling:.3f}s", log_file)
        log_message(f"  MBR computation: {self.current_episode.mbr_computation:.3f}s", log_file)
        log_message(f"  Backtracking: {self.current_episode.backtracking:.3f}s", log_file)
        total = (self.current_episode.vlm_detector + self.current_episode.action_inference + 
                self.current_episode.sampling + self.current_episode.mbr_computation + 
                self.current_episode.backtracking)
        log_message(f"  TOTAL THIS EPISODE: {total:.3f}s", log_file)
        
        # Show running averages if we have completed episodes
        if self.episodes:
            log_message(f"\n  Running averages across {len(self.episodes)} completed episodes:", log_file)
            components = ['vlm_detector', 'action_inference', 'sampling', 'mbr_computation', 'backtracking']
            for comp in components:
                total = sum(getattr(ep, comp) for ep in self.episodes)
                avg = total / len(self.episodes) if self.episodes else 0
                log_message(f"    {comp}: {avg:.3f}s/episode", log_file)
    
    def log_final_summary(self, log_file):
        """Log final summary of all episodes."""
        log_message("\n" + "="*80, log_file)
        log_message("FINAL LATENCY ANALYSIS (AVERAGED ACROSS EPISODES)", log_file)
        log_message("="*80, log_file)
        
        if not self.episodes:
            log_message("No episodes completed.", log_file)
            return
        
        components = ['vlm_detector', 'action_inference', 'sampling', 'mbr_computation', 'backtracking']
        
        log_message(f"Total episodes analyzed: {len(self.episodes)}", log_file)
        log_message("\nAverage latency PER EPISODE (only counting episodes where component was used):", log_file)
        
        grand_total = 0
        for comp in components:
            ep_values = [getattr(ep, comp) for ep in self.episodes]
            
            # Only count non-zero episodes for this component
            non_zero_values = [v for v in ep_values if v > 0]
            num_used = len(non_zero_values)
            
            if num_used > 0:
                total = sum(non_zero_values)
                avg = total / num_used
                min_val = min(non_zero_values)
                max_val = max(non_zero_values)
                
                log_message(f"\n{comp.upper().replace('_', ' ')}:", log_file)
                log_message(f"  Used in {num_used}/{len(self.episodes)} episodes", log_file)
                log_message(f"  Average (when used): {avg:.3f}s", log_file)
                log_message(f"  Min: {min_val:.3f}s", log_file)
                log_message(f"  Max: {max_val:.3f}s", log_file)
                log_message(f"  Total across all episodes: {total:.3f}s", log_file)
                grand_total += total
            else:
                log_message(f"\n{comp.upper().replace('_', ' ')}:", log_file)
                log_message(f"  Not used in any episode", log_file)
        
        log_message(f"\nTOTAL time across all episodes: {grand_total:.3f}s", log_file)
        log_message(f"Average total time per episode: {grand_total / len(self.episodes):.3f}s", log_file)
        log_message("="*80 + "\n", log_file)


def reverse_action(action) -> np.ndarray:
    action = np.array(action, dtype=np.float32)
    reversed_action = action.copy()
    reversed_action[:6] *= -1
    # reversed_action[6] = -reversed_action[6]
    return reversed_action.tolist()


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
    local_log_dir: str = "./rollouts-plus/logs_plus_decomposed_progress_mbr"        # Local directory for eval logs
    video_save_dir: str = "./rollouts-plus/rollouts_plus_decomposed_progress_mbr"
    video_base_dir: str = "./rollouts-plus/rollouts_plus_decomposed_progress_transit"  # Stage-1 transit videos to scan for failures
    # Directory holding the per-suite Language-category instruction->canonical-task cache.
    # Shared with the transit script (same default) on purpose: this mbr stage reuses the
    # transit stage's GPT-match results and resolves every rewrite to the same canonical task.
    instruction_cache_dir: str = "./experiments/robot/libero-plus"

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 0                                    # Random Seed (for reproducibility)

    rerun_all: bool = False                          # Run all failure correction

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

    return observation, img, wrist_img  # Return both processed observation and original image for replay


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


# https://github.com/Lifelong-Robot-Learning/LIBERO/issues/16
# https://github.com/Lifelong-Robot-Learning/LIBERO/issues/34
# https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/notebooks/quick_walkthrough.ipynb
def restore_robot_only(env, saved_state: np.ndarray) -> None:
    """Restore only robot portion of the state, keeping current object positions"""
    # Get current state (with current object positions)
    current_state = env.sim.get_state().flatten()
    
    # MuJoCo state structure when flattened:
    # [time, qpos, qvel, act, mocap_pos, mocap_quat, userdata]
    
    # Find dimensions
    nq = env.sim.model.nq  # Total qpos dimensions
    nv = env.sim.model.nv  # Total qvel dimensions
    
    # Typically for LIBERO:
    # - First 7-9 qpos values are robot joints
    # - Rest are object positions
    robot_qpos_dim = 9  # Adjust based on your robot
    robot_qvel_dim = 9
    
    # Copy only robot portions from saved to current
    # Time is at index 0
    # qpos starts at index 1
    # qvel starts at index 1 + nq
    
    # Replace robot qpos
    current_state[1:1+robot_qpos_dim] = saved_state[1:1+robot_qpos_dim]
    
    # Replace robot qvel
    qvel_start = 1 + nq
    current_state[qvel_start:qvel_start+robot_qvel_dim] = saved_state[qvel_start:qvel_start+robot_qvel_dim]
    
    # Also preserve mocap positions (important for OSC control)
    if env.sim.model.nmocap > 0:
        mocap_start = 1 + nq + nv + env.sim.model.na
        mocap_end = mocap_start + 7 * env.sim.model.nmocap  # 3 pos + 4 quat per mocap
        current_state[mocap_start:mocap_end] = saved_state[mocap_start:mocap_end]
    
    # Set the modified state
    env.set_state(current_state)


def record_robot_state(history: List[Dict[str, Any]], subtask: str, env, obs) -> List[Dict[str, Any]]:
    """Record state using LIBERO's functions"""
    state = env.sim.get_state().flatten()
    
    obs_data = {
        "robot0_eef_pos": obs["robot0_eef_pos"].copy(),
        "robot0_eef_quat": obs["robot0_eef_quat"].copy(),
        "robot0_gripper_qpos": obs["robot0_gripper_qpos"].copy()
    }
    
    for entry in history:
        if entry["subtask"] == subtask:
            entry["states"].append(state)
            entry["observations"].append(obs_data)
            return history
    
    history.append({
        "subtask": subtask, 
        "states": [state],
        "observations": [obs_data]
    })
    return history


def backtrace_robot_states(history: List[Dict[str, Any]], target_subtask: str) -> List[Dict[str, Any]]:
    """
    Backtrace and trim robot state history to the target subtask (inclusive).
    Returns a list of snapshots in reverse order (most recent first), then trims history.
    """
    reversed_states: List[Dict[str, Any]] = []
    for i in range(len(history) - 1, -1, -1):
        reversed_states.extend(reversed(history[i]["states"]))
        if history[i]["subtask"] == target_subtask:
            del history[i:]
            break
    return reversed_states


def extract_trajectory_features(robot_state_hist, current_state, start_idx=0, end_idx=None):
    """Extract trajectory features from stored observations for MBR comparison.
    Can extract either first n (for comparison) or full trajectory (for storage)."""
    features = []
    
    # Find the subtask in history
    for entry in robot_state_hist:
        if entry["subtask"] == current_state:
            observations = entry["observations"][start_idx:end_idx]
            
            for obs in observations:
                # Extract end-effector position
                pos = obs.get("robot0_eef_pos", [0, 0, 0])[:3]
                
                # Extract quaternion and convert to euler
                quat_wxyz = obs.get("robot0_eef_quat", [1, 0, 0, 0])
                quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                
                try:
                    euler = R.from_quat(quat_xyzw).as_euler('xyz', degrees=False)
                except:
                    euler = [0, 0, 0]
                
                features.extend(pos)
                features.extend(euler)
            
            # Pad to expected size if needed
            # multiply by 6 because translation (xyz) + rotation (xyz) have dimension of 6
            expected_size = (end_idx - start_idx) * 6 if end_idx else len(features)
            while len(features) < expected_size:
                features.extend([0, 0, 0, 0, 0, 0])
            
            break
    
    return np.array(features[:expected_size] if end_idx else features)
    

def sample_and_rank_seeds_mbr(
    cfg,
    model,
    observation,
    current_subtask,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    num_seeds,
    failed_trajectories,
    distance_metric="l2",
    use_failed_repulsion=False,
    r_neighborhood=None,  # Optional, will use adaptive by default
    selection_mode="rep",  # "rep" (representative) or "away" (diverse)
    MBR_vanilla=False,  # Use standard MBR without r-NN density
    log_file=None,
    latency_tracker=None,
):
    """Sample multiple seeds and rank them using MBR with r-NN density estimation.
    
    IMPORTANT: This should be called AFTER backtracking, so 'observation' is from
    the beginning of the subtask we're retrying.
    
    Args:
        selection_mode: "rep" for representative (dense pocket), "away" for diverse
        MBR_vanilla: If True, use standard MBR (average distance)
    """    
    # Store original seed
    original_seed = cfg.seed
    
    # Sample trajectories with different seeds
    sampled_state_trajectories = []
    seed_list = []
    
    sampling_start = time.time()
    for i in range(num_seeds):
        # Use a different seed for each sample
        seed = original_seed + i * 10
        set_seed_everywhere(seed)
        seed_list.append(seed)
        
        # Get action chunk with this seed
        actions = get_action(
            cfg,
            model,
            observation,
            current_subtask,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )
        
        # Convert actions to predicted state trajectory
        # Starting from current observation state (which is at beginning of subtask after backtrack)
        current_pos = np.array(observation["state"][:3])  # Current end-effector position
        current_euler = np.array(observation["state"][3:6])  # Current orientation as euler
        
        state_features = []
        cumulative_pos = current_pos.copy()
        
        # Convert initial euler to rotation object for proper composition
        cumulative_rot = R.from_euler('xyz', current_euler)
        
        for action in actions:
            processed = process_action(action, cfg.model_family, stop=True, progress=True)
            
            # Extract action deltas
            pos_delta = np.array(processed[:3])
            rot_delta_euler = np.array(processed[3:6])
            
            # Apply position delta
            cumulative_pos = cumulative_pos + pos_delta
            
            # Apply rotation delta properly (compose rotations)
            delta_rot = R.from_euler('xyz', rot_delta_euler)
            cumulative_rot = cumulative_rot * delta_rot
            
            # Convert back to euler for feature vector
            current_euler = cumulative_rot.as_euler('xyz')
            
            # Append to trajectory features
            state_features.extend(cumulative_pos.tolist())
            state_features.extend(current_euler.tolist())
        
        # Add cfg to know the expected size
        # multiply by 6 because translation (xyz) + rotation (xyz) have dimension of 6
        expected_features = cfg.num_open_loop_steps * 6

        # When padding:
        while len(state_features) < expected_features:
            state_features.extend([0, 0, 0, 0, 0, 0])
        
        sampled_state_trajectories.append(np.array(state_features[:expected_features]))
    
    sampling_time = time.time() - sampling_start
    if latency_tracker:
        latency_tracker.add_to_current('sampling', sampling_time)
    
    
    # Convert to numpy array (MBR computation)
    mbr_start = time.time()
    X = np.stack(sampled_state_trajectories)  # Shape: (num_seeds, cfg.num_open_loop_steps * 6)
    N = X.shape[0]
    
    # Compute pairwise distances
    metric_map = {
        "l2": "euclidean",
        "l1": "cityblock",
        "cosine": "cosine",
        "correlation": "correlation",
        "chebyshev": "chebyshev"
    }
    
    dist_mat = cdist(X, X, metric=metric_map.get(distance_metric, "euclidean"))

    # === Vanilla MBR ===
    if MBR_vanilla:
        # Standard MBR: average distance to all
        avg_dist = dist_mat.mean(axis=1)
        
        if selection_mode == "away":
            # Most atypical (max avg dist)
            ranked_indices = np.argsort(avg_dist)[::-1]
        else:
            # Most typical (min avg dist)
            ranked_indices = np.argsort(avg_dist)
        
        ranked_seeds = [seed_list[i] for i in ranked_indices]
        
        log_message(
            f"Vanilla MBR ranking complete (mode={selection_mode}, metric={distance_metric}). "
            f"Top 3 seeds: {ranked_seeds[:3]} with avg distances: {avg_dist[ranked_indices[:3]]}",
            log_file,
        )
        
        # Restore original seed and return
        set_seed_everywhere(original_seed)
        return ranked_seeds
    
    # === Adaptive r-NN neighborhood size ===
    if r_neighborhood is None:
        r = max(2, min(4, int(np.sqrt(N))))
    else:
        r = r_neighborhood
    r_eff = min(r, max(1, N - 1))
    
    # r-NN radius (distance to r-th nearest neighbor)
    rnn_radius = np.partition(dist_mat, r_eff, axis=1)[:, r_eff]
    
    # Find pocket center (smallest r-NN radius) - densest point
    center_idx = int(np.argmin(rnn_radius))
    order_center = np.argsort(dist_mat[center_idx])
    cluster_idx = order_center[:r_eff]
    
    # Find medoid inside pocket (most representative point)
    intra = dist_mat[np.ix_(cluster_idx, cluster_idx)]
    medoid_local = cluster_idx[int(np.argmin(intra.mean(axis=1)))]
    
    # Distance to medoid for orderings
    d_to_medoid = dist_mat[medoid_local]
    
    # Robust normalization function
    def robust_norm(v):
        v = np.asarray(v)
        if len(v) < 2:
            return np.zeros_like(v)
        lo, hi = np.percentile(v, [10, 90])
        v_clip = np.clip(v, lo, hi)
        med = np.median(v_clip)
        iqr = (np.percentile(v_clip, 75) - np.percentile(v_clip, 25)) + 1e-8
        return (v - med) / iqr
    
    # Compute base scores
    if selection_mode == "away":
        # Away mode: prefer points far from medoid (diverse selection)
        base_order = np.argsort(d_to_medoid)[::-1]
    else:
        # Representative mode: prefer medoid and points close to it
        base_order = np.concatenate([[medoid_local],
                                     np.argsort(np.where(np.arange(N)==medoid_local,
                                                        np.inf, d_to_medoid))])
    
    # Apply failed trajectory repulsion if enabled
    if use_failed_repulsion and failed_trajectories:
        valid_failed = []
        for ft in failed_trajectories:
            # ft should already be the first cfg.num_open_loop_steps timesteps (cfg.num_open_loop_steps * 6 dims)
            if isinstance(ft, np.ndarray) and ft.shape[0] == expected_features:
                valid_failed.append(ft)
        
        if valid_failed:
            failed_features = np.stack(valid_failed)
            
            # Compute distances from sampled to failed trajectories
            dist_to_failed = cdist(X, failed_features, metric=metric_map.get(distance_metric, "euclidean"))
            
            # Get minimum distance to any failed trajectory
            d_fail = dist_to_failed.min(axis=1)
        else:
            # No valid failures - use neutral fallback
            d_fail = np.full((N,), np.median(rnn_radius) if N > 0 else 1.0)
    else:
        # No failure repulsion - use neutral fallback
        d_fail = np.full((N,), np.median(rnn_radius) if N > 0 else 1.0)
    
    # Normalize all components robustly
    rnn_norm = robust_norm(rnn_radius)      # lower is denser (better for rep)
    dmed_norm = robust_norm(d_to_medoid)    # higher = farther from medoid
    dfail_norm = robust_norm(d_fail)        # higher = farther from failures
    
    # Soften repulsion with sigmoid
    tau = 1.0
    repulse = 1.0 / (1.0 + np.exp(-(dfail_norm / tau)))
    
    lambda_fail = 0.5  # Weight for failure repulsion
    
    # Compute final scores based on selection mode
    if selection_mode == "away":
        # Away + failure-aware: maximize distance-from-medoid + λ*repulsion
        final_scores = dmed_norm + lambda_fail * repulse
        ranked_indices = np.argsort(final_scores)[::-1]  # Higher is better
    else:
        # Representative + failure-aware: minimize density - λ*repulsion
        final_scores = -rnn_norm + lambda_fail * repulse  # Convert to "higher is better"
        ranked_indices = np.argsort(final_scores)[::-1]  # Higher is better
    
    ranked_seeds = [seed_list[i] for i in ranked_indices]
    
    log_message(
        f"MBR ranking complete (mode={selection_mode}, metric={distance_metric}). "
        f"Medoid: seed {seed_list[medoid_local]}. "
        f"Top 3 seeds: {ranked_seeds[:3]} with scores: {final_scores[ranked_indices[:3]]}",
        log_file,
    )

    mbr_time = time.time() - mbr_start
    if latency_tracker:
        latency_tracker.add_to_current('mbr_computation', mbr_time)
    
    # Restore original seed
    set_seed_everywhere(original_seed)
    
    return ranked_seeds


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
    latency_tracker=None,
):
    """Run a single episode in the environment with MBR-based retry mechanism."""
    # Get subtasks
    states = pick_place_states(task_description, f"{cfg.task_suite_name}_no_noops")
    states = complex_states(task_description, f"{cfg.task_suite_name}_no_noops") if states is None else states
    states = [state.lower() for state in states]
    subtask_list = [f"Task: {task_description}. The current subtask: {state}" for state in states]

    vlm_detector = VLMDetector(model_name="gpt-5.5", temperature=1)

    # Initialize latency tracker if not provided
    if latency_tracker is None:
        latency_tracker = LatencyTracker()

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
    replay_images, replay_wrist_images, replay_subtasks = [], [], []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    current_state = states[0]
    current_subtask = subtask_list[0]
    subtask_hist, exe_type_hist = [current_state], ["init"]
    robot_state_hist: List[Dict[str, Any]] = []

    # Progress tracking for VLM check at 90%
    subtask_phase = "to_check"  # "to_check" -> VLM check -> "to_complete"
    progress_threshold = 0.90

    # Determine checking robustness based on task suite
    use_robust_progress_checking = cfg.task_suite_name in ["libero_goal", "libero_10"]

    if use_robust_progress_checking:
        # Robust checking for complex tasks (same as termination logic)
        first_progress_high_seen = False
        consecutive_progress_high_count = 0
        steps_since_last_progress_high = 0
    else:
        # Simple checking for spatial/object
        count_check_signals = 0

    # Termination detection (for 100% completion)
    first_high_seen = False
    consecutive_high_count = 0
    steps_since_last_high = 0

    # MBR-based retry mechanism
    subtask_retry_count = {}  # Track retry count per subtask
    subtask_seed_rankings = {}  # Store seed rankings per subtask
    subtask_failed_trajectories = {}  # Store failed trajectory features per subtask
    current_seed_index = {}  # Track which seed we're using per subtask
    
    # MBR configuration (can be made configurable)
    mbr_config = {
        "num_seeds": 8,  # Number of seeds to sample
        "max_retries": 3,  # Maximum retry attempts per subtask
        "distance_metric": "l2",  # Options: "l2", "l1", "cosine", "correlation", "chebyshev"
        "use_failed_repulsion": False,  # Whether to use previously failed trajectories for repulsion
        "r_neighborhood": None,  # Size of r-NN neighborhood
        "use_MBR_vanilla": False  # Use standard MBR without r-NN density
    }

    try:
        while t < max_steps * 1.5 + cfg.num_steps_wait:
            # Let objects settle
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img, wrist_img = prepare_observation(obs, resize_size)
            replay_images.append(img)
            replay_wrist_images.append(wrist_img)
            replay_subtasks.append(f"{exe_type_hist[-1]}: {current_state}")

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                start_time = time.time()
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
                latency_tracker.add_to_current('action_inference', time.time() - start_time)
                action_queue.extend(actions)

            # === Record full sim snapshot BEFORE executing next action ===
            robot_state_hist = record_robot_state(robot_state_hist, current_state, env, obs)

            # Get and process next action (now 9-dim: 7 robot actions + termination + progress)
            action = process_action(action_queue.popleft(), cfg.model_family, stop=True, progress=True)
            print(action)

            # Extract signals
            termination_signal = action[-2]
            progress_signal = action[-1]
            
            # Step environment with the physical action dimensions ONLY (exclude stop and progress dim)
            obs, reward, done, info = env.step(action[:-2])
            t += 1

            if done:
                success = True
                log_message(f"Environment signaled done for subtask: {current_state} in {t} steps", log_file)
                break

            # =========================
            # PHASE 1: Check for 90% progress (VLM decision point)
            # =========================
            if subtask_phase == "to_check":
                # Check if this is a gripper subtask (skip VLM check for these)
                is_gripper_subtask = (
                    "close the gripper to grasp" in current_state.lower() or
                    "open the gripper to release" in current_state.lower()
                )
                
                if is_gripper_subtask:
                    # Skip VLM check and go directly to completion phase for gripper subtasks
                    log_message(f"Gripper subtask detected: `{current_state}` - skipping VLM check", log_file)
                    subtask_phase = "to_complete"
                    first_high_seen = False
                    consecutive_high_count = 0
                    steps_since_last_high = 0
                    continue
                
                vlm_check_needed = False

                if use_robust_progress_checking:
                    # ROBUST CHECKING (Goal/Long tasks) - Same logic as termination
                    if progress_signal >= progress_threshold:
                        consecutive_progress_high_count += 1
                        
                        if not first_progress_high_seen:
                            first_progress_high_seen = True
                            log_message(
                                f"90% progress signal observed at step {t} for subtask: {current_state} "
                                f"(progress={progress_signal:.2f})",
                                log_file,
                            )
                        
                        # Progress confirmation conditions (same as termination)
                        if consecutive_progress_high_count >= 2:
                            # Consecutive progress confirmations
                            vlm_check_needed = True
                            log_message(
                                f"90% progress confirmed at step {t} "
                                f"({consecutive_progress_high_count} consecutive signals)",
                                log_file,
                            )
                        elif first_progress_high_seen and steps_since_last_progress_high >= 2:
                            # Progress recurs after at least 2 "low" steps
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
                        # Reset the low-signal counter on a high
                        steps_since_last_progress_high = 0
                    else:
                        # Low progress signal
                        consecutive_progress_high_count = 0
                        if first_progress_high_seen:
                            steps_since_last_progress_high += 1
                
                else:
                    # SIMPLE CHECKING (Spatial/Object tasks) - Original logic
                    if progress_signal >= progress_threshold:
                        count_check_signals += 1
                        log_message(f"90% signal #{count_check_signals} at step {t} (progress={progress_signal:.2f})", log_file)
                        if count_check_signals >= 2:
                            vlm_check_needed = True
                            log_message(f"90% progress confirmed after {count_check_signals} high signals", log_file)

                if vlm_check_needed:
                    # Add a few identical frames for VLM context (no physics step)
                    _, current_img, current_wrist_img = prepare_observation(obs, resize_size)
                    for _ in range(32):
                        replay_images.append(current_img)
                        replay_wrist_images.append(current_wrist_img)
                        replay_subtasks.append(f"VLM_90%_check: {current_state}")

                    # Run VLM decision @90%
                    start_time = time.time()
                    res = vlm_detector.detect_subtask(
                        current_state,
                        states,
                        subtask_hist,
                        task_description,
                        current_img,
                        current_wrist_img
                    )
                    latency_tracker.add_to_current('vlm_detector', time.time() - start_time)

                    detected_state, exe_type, reason = vlm_detector.extract_res(res)
                    log_message(
                        f"VLM check at 90% for subtask `{current_state}` → next: {detected_state}, "
                        f"type: {exe_type}, reason: {reason}",
                        log_file,
                    )

                    if exe_type == "backtrack":
                        # Check retry count for this subtask
                        if current_state not in subtask_retry_count:
                            subtask_retry_count[current_state] = 0
                        
                        subtask_retry_count[current_state] += 1
                        
                        # If we've already retried 3 times, force completion
                        if subtask_retry_count[current_state] >= mbr_config["max_retries"]:
                            log_message(
                                f"Maximum retries ({mbr_config['max_retries']}) reached for subtask `{current_state}`. "
                                f"Forcing continuation to completion.",
                                log_file,
                            )
                            subtask_phase = "to_complete"
                            first_high_seen = False
                            consecutive_high_count = 0
                            steps_since_last_high = 0
                            continue
                        
                        # IMPORTANT: Store failed trajectory BEFORE backtracking
                        # Extract first n timesteps for MBR comparison
                        failed_first_n = extract_trajectory_features(
                            robot_state_hist, current_state, start_idx=0, end_idx=cfg.num_open_loop_steps
                        )
                        
                        # Store the full failed trajectory for potential future use
                        failed_full = extract_trajectory_features(
                            robot_state_hist, current_state, start_idx=0, end_idx=None
                        )
                        
                        if current_state not in subtask_failed_trajectories:
                            subtask_failed_trajectories[current_state] = {
                                "first_n": [],  # For MBR comparison
                                "full": []       # Full trajectory storage
                            }
                        
                        subtask_failed_trajectories[current_state]["first_n"].append(failed_first_n)
                        subtask_failed_trajectories[current_state]["full"].append(failed_full)
                        
                        log_message(
                            f"VLM decided to backtrack from {current_state} to {detected_state} at 90% progress "
                            f"(retry #{subtask_retry_count[current_state]})",
                            log_file,
                        )
                        
                        # === FIRST: Perform backtrack to reset state ===
                        current_state = detected_state
                        current_subtask = subtask_list[states.index(current_state)]
                        
                        # Add to history
                        subtask_hist.append(current_state)
                        exe_type_hist.append(f"backtrack_90%_retry{subtask_retry_count[current_state]}")
                        
                        # Clear pending actions
                        action_queue.clear()
                        
                        # Perform physical backtrack
                        start_time = time.time()
                        reversed_snaps = backtrace_robot_states(robot_state_hist, current_state)
                        for snap in reversed_snaps:
                            restore_robot_only(env, snap)
                            dummy_action = get_libero_dummy_action(cfg.model_family)
                            obs, _, _, _ = env.step(dummy_action)
                            observation, img, wrist_img = prepare_observation(obs, resize_size)
                            replay_images.append(img)
                            replay_wrist_images.append(wrist_img)
                            replay_subtasks.append(f"backtrack_from_90%_retry{subtask_retry_count[current_state]}")
                        latency_tracker.add_to_current('backtracking', time.time() - start_time)
                        
                        # === THEN: Sample and rank seeds with MBR (observation is now from beginning of subtask) ===
                        if current_state not in subtask_seed_rankings:
                            log_message(f"Sampling {mbr_config['num_seeds']} seeds for MBR selection...", log_file)
                            
                            # Now observation is from the beginning of the subtask after backtrack
                            seed_rankings = sample_and_rank_seeds_mbr(
                                cfg=cfg,
                                model=model,
                                observation=observation,  # This is now from beginning of subtask
                                current_subtask=current_state,
                                processor=processor,
                                action_head=action_head,
                                proprio_projector=proprio_projector,
                                noisy_action_projector=noisy_action_projector,
                                num_seeds=mbr_config["num_seeds"],
                                failed_trajectories=[ft for ft in subtask_failed_trajectories.get(current_state, {}).get("first_n", [])],
                                distance_metric=mbr_config["distance_metric"],
                                use_failed_repulsion=mbr_config["use_failed_repulsion"],
                                r_neighborhood=mbr_config["r_neighborhood"],
                                MBR_vanilla=mbr_config["use_MBR_vanilla"],
                                log_file=log_file,
                                latency_tracker=latency_tracker,
                            )
                            
                            subtask_seed_rankings[current_state] = seed_rankings
                            current_seed_index[current_state] = 0
                        else:
                            # Move to next seed in the ranking
                            current_seed_index[current_state] += 1
                            if current_seed_index[current_state] >= len(subtask_seed_rankings[current_state]):
                                current_seed_index[current_state] = 0  # Wrap around if needed
                        
                        # Set the seed for the next attempt
                        selected_seed = subtask_seed_rankings[current_state][current_seed_index[current_state]]
                        set_seed_everywhere(selected_seed)
                        log_message(
                            f"Using seed {selected_seed} (rank {current_seed_index[current_state] + 1}) "
                            f"for retry attempt",
                            log_file,
                        )
                        
                        # Reset phase tracking for the retry
                        subtask_phase = "to_check"
                        if use_robust_progress_checking:
                            # Reset robust progress tracking
                            first_progress_high_seen = False
                            consecutive_progress_high_count = 0
                            steps_since_last_progress_high = 0
                        else:
                            # Reset simple tracking
                            count_check_signals = 0
                        # Reset termination tracking
                        first_high_seen = False
                        consecutive_high_count = 0
                        steps_since_last_high = 0

                    else:
                        # VLM decided NOT to backtrack - continue with current subtask
                        log_message(
                            f"VLM decided to continue with subtask `{current_state}` "
                            f"(detected: {detected_state}, type: {exe_type})",
                            log_file,
                        )
                        # Transition to completion phase
                        subtask_phase = "to_complete"
                        if use_robust_progress_checking:
                            # Reset robust progress tracking
                            first_progress_high_seen = False
                            consecutive_progress_high_count = 0
                            steps_since_last_progress_high = 0
                        else:
                            # Reset simple tracking
                            count_check_signals = 0
                        # Reset termination tracking
                        first_high_seen = False
                        consecutive_high_count = 0
                        steps_since_last_high = 0

            # =========================
            # PHASE 2: Check for termination (100% completion via stop signal)
            # =========================
            elif subtask_phase == "to_complete":
                subtask_completed = False

                if termination_signal == -1:  # Binary termination signal
                    consecutive_high_count += 1

                    if not first_high_seen:
                        first_high_seen = True
                        log_message(
                            f"Termination signal observed at step {t} for subtask: {current_state} "
                            f"(term={termination_signal}, progress={progress_signal:.2f})",
                            log_file,
                        )

                    # Termination confirmation conditions
                    if consecutive_high_count >= 2:
                        # Consecutive STOP confirmations
                        subtask_completed = True
                        log_message(
                            f"Subtask `{current_state}` completed at step {t} "
                            f"({consecutive_high_count} consecutive termination signals)",
                            log_file,
                        )
                    elif first_high_seen and steps_since_last_high >= 2:
                        # STOP recurs after at least 2 "low" steps
                        subtask_completed = True
                        log_message(
                            f"Subtask `{current_state}` completed at step {t} "
                            f"(termination re-confirmed after {steps_since_last_high} low steps)",
                            log_file,
                        )
                    else:
                        if first_high_seen and steps_since_last_high > 0:
                            log_message(
                                f"Ignoring termination re-signal at step {t} "
                                f"(only {steps_since_last_high} low steps since last; need 2+)",
                                log_file,
                            )
                    # Reset the low-signal counter on a high
                    steps_since_last_high = 0
                else:
                    # Low (no termination)
                    consecutive_high_count = 0
                    if first_high_seen:
                        steps_since_last_high += 1

                if subtask_completed:
                    prev_state = current_state
                    current_index = states.index(current_state)

                    log_message(f"Subtask `{current_state}` completed via termination signal at step {t}", log_file)

                    # Advance to next subtask if any
                    if current_index + 1 < len(states):
                        current_state = states[current_index + 1]
                        current_subtask = subtask_list[current_index + 1]

                        subtask_hist.append(current_state)
                        exe_type_hist.append("continue_termination")

                        log_message(f"Moving from `{prev_state}` to next subtask: `{current_state}`", log_file)

                        # Clear action queue when transitioning to new subtask
                        action_queue.clear()

                        # Reset phase tracking for the new subtask
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
                        # All subtasks completed
                        log_message(f"All subtasks completed at step {t}", log_file)
                        break
                
    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    # After all subtasks completed, execute remaining actions in queue
    if not success and len(action_queue) > 0:
        log_message(f"Executing {len(action_queue)} remaining actions in queue...", log_file)
        while len(action_queue) > 0:
            try:
                action = process_action(action_queue.popleft(), cfg.model_family, stop=True, progress=True)
                obs, reward, done, info = env.step(action[:-2])
            except ValueError:
                # Environment already terminated, just break
                break

            if done:
                success = True
                log_message(f"Environment signaled done while executing remaining actions", log_file)
                break

            observation, img, wrist_img = prepare_observation(obs, resize_size)
            replay_images.append(img)
            replay_wrist_images.append(wrist_img)  # Don't forget wrist images!
            replay_subtasks.append(f"finishing: {current_state}")

    return success, replay_images, replay_wrist_images, replay_subtasks


def get_failed_episodes_from_videos(cfg: GenerateConfig, log_file=None):
    """
    Parse the Stage-1 transit video directory to find FAILED episodes.
    Returns a dictionary in the format: {task_suite_name: [list of failed episode numbers]}

    IMPORTANT (LIBERO-Plus): episode numbers are a running counter over the *selected*
    task list, so they are only consistent between the transit and mbr runs when both
    use the same `--category`. When a single category is being evaluated we therefore
    scan only that category's sub-directory; for `--category all` we scan the whole
    suite directory (numbering is global and consistent across the per-category subdirs).
    """
    # Search in the task-suite (and, for a single category, category) sub-directory
    video_dir = os.path.join(cfg.video_base_dir, cfg.task_suite_name)
    if cfg.category != "all":
        video_dir = os.path.join(video_dir, category_slug(normalize_category(cfg.category)))
    
    if not os.path.exists(video_dir):
        log_message(f"Video directory not found: {video_dir}", log_file)
        return {cfg.task_suite_name: []}
    
    failed_episodes = set()
    all_episodes = set()
    
    # Pattern to extract episode number and success from filename
    pattern = r"--episode=(\d+)--success=(True|False)--"
    
    # Walk through all folders to find video files
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                match = re.search(pattern, file)
                if match:
                    episode_num = int(match.group(1))
                    success = match.group(2) == "True"
                    
                    all_episodes.add(episode_num)
                    
                    # Add FAILED episodes to our set
                    if not success:
                        failed_episodes.add(episode_num)
    
    # Convert to sorted list
    episodes_list = sorted(list(failed_episodes))
    
    # Build the rerun_dict format
    rerun_dict = {cfg.task_suite_name: episodes_list}
    
    # Log what we found
    log_message(f"\n{'='*80}", log_file)
    log_message(f"RERUN DICT FROM VIDEO ANALYSIS", log_file)
    log_message(f"{'='*80}", log_file)
    log_message(f"Video directory: {video_dir}", log_file)
    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Total episodes found: {len(all_episodes)}", log_file)
    log_message(f"Failed episodes to rerun: {episodes_list}", log_file)
    log_message(f"Number of failed episodes: {len(episodes_list)}", log_file)
    log_message(f"Constructed rerun_dict: {rerun_dict}", log_file)
    log_message(f"{'='*80}\n", log_file)
    
    return rerun_dict


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
    rerun_dict=None,
    log_file=None,
    latency_tracker=None,
):
    """Run evaluation for a single LIBERO-Plus task variant.

    `category` is the variant's perturbation category; `category_stats` is a
    {category: [episodes, successes]} dict updated in place for per-category reporting.
    `rerun_dict` is the precomputed {task_suite_name: [failed episode numbers]} from the
    Stage-1 transit videos — passed in (computed once in eval_libero) rather than rescanned
    here per task.
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
        if cfg.rerun_all:
            rerun = True
        else:
            rerun = True if total_episodes + 1 in rerun_dict[cfg.task_suite_name] else False
        
        if rerun:
            success, replay_images, replay_wrist_images, replay_subtasks = run_episode(
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
                latency_tracker,
            )

            # LOG EPISODE LATENCY SUMMARY
            if latency_tracker:
                latency_tracker.log_episode_summary(total_episodes, log_file)
                latency_tracker.finish_episode()
        else:
            success = True

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
        if rerun:
            save_rollout_video_decomposed(
                replay_images, total_episodes, success=success, task_description=video_label,
                video_save_dir=video_dir, log_file=log_file, subtasks=replay_subtasks
            )
            save_rollout_video_decomposed(
                replay_wrist_images, total_episodes, success=success, task_description=video_label,
                video_save_dir=video_dir, log_file=log_file, subtasks=replay_subtasks, wrist=True
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
    # MUST match the transit run's --category / --eval_fraction / --seed so episode
    # numbering lines up with the Stage-1 transit videos scanned for failures — the same
    # invariant is what makes the sidecar-progress-file resume safe.
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

    # Initialize latency tracker (restart from zero on resume — running averages aren't
    # persisted, since they're diagnostic-only and would be misleading if reloaded)
    latency_tracker = LatencyTracker()

    # Rebuild per-category accounting as a defaultdict (autovivify for any category we
    # haven't seen yet, e.g. when resuming into a new suite of `--category all`).
    category_stats = defaultdict(lambda: [0, 0])
    for k, v in prior_category_stats.items():
        category_stats[k] = list(v)

    if start_index >= num_selected:
        # Nothing left to do — this (suite, category) run was already complete on disk.
        # Fall through to the final-summary block so aggregate_plus_logs sees consistent
        # totals in the appended log tail even for a no-op resume. Skip model load, the
        # rerun_dict scan, and the main loop entirely.
        log_message(
            f"Nothing to do: {start_index}/{num_selected} tasks already complete on disk.",
            log_file,
        )
    else:
        # Only load the model when there's actual work — a repeated resume-into-a-complete-run
        # otherwise wastes minutes on model init just to log "nothing to do".
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        resize_size = get_image_resize_size(cfg)

        # Scan the Stage-1 transit videos ONCE for the set of failed episodes (the result is
        # independent of task_id), then pass it into every run_task — vs. the LIBERO original,
        # which rescans inside each run_task (~2,400 redundant os.walk scans at LIBERO-Plus scale).
        # The scanned set covers ALL selected episodes (1..num_selected), so resuming at
        # start_index just skips the prefix of the same iteration order — no re-scan needed.
        rerun_dict = get_failed_episodes_from_videos(cfg, log_file)

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
                rerun_dict,
                log_file,
                latency_tracker,
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

    # Log final latency analysis (only meaningful when episodes ran this invocation —
    # otherwise reports zeros because `latency_tracker` was just constructed above.)
    latency_tracker.log_final_summary(log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()