"""
run_libero_plus_eval_openpi_cyclevla.py

Full CycleVLA evaluation pipeline for the LIBERO-Plus robustness benchmark
(openpi / pi0.5 backbone). LIBERO-Plus counterpart of
`experiments/robot/libero/run_libero_eval_openpi_cyclevla.py`.

Stage 2 of the two-stage LIBERO-Plus workflow. Uses the 9-dim subtask-aware
openpi policy to step through subtasks via the stop + progress signals. At ~90%
progress per subtask, a VLM is queried with `type: transit | backtrack`; on
`backtrack` the env is physically rewound to the start of the failing subtask,
N chunks are sampled from the stochastic openpi server, and the best one is
selected via Minimum Bayes Risk (MBR). Per-component latencies are tracked.

As in the LIBERO eval, this re-runs ONLY the episodes the transit baseline
(`..._openpi_transit.py`) failed -- it scans the baseline's rollout videos under
`--video_base_dir`. Pass `--rerun_all True` to skip the baseline and evaluate
every selected variant fresh.

Architecture: the pi0.5 policy is served remotely by openpi over a websocket;
all openpi-vs-OpenVLA convention differences are handled in
`experiments/robot/openpi_utils.py`.

How MBR sampling works here: the openpi server is stochastic per `infer()`
call (`Policy.infer` splits its RNG each call), so N `client.get_action` calls
on the same observation yield N diverse chunks -- no seed control or server
change needed. Unlike the OpenVLA MBR (which replays a chosen diffusion seed),
we already hold every sampled chunk, so we cache and execute the winner
directly.

LIBERO-Plus specifics (see experiments/robot/libero_plus_utils.py):
  * `--task_suite_name` (libero_spatial/object/goal/10) holds ~2,400 perturbed
    variants across 7 categories; `--category` restricts the run to one,
    `--eval_fraction` sub-samples it. The transit and cyclevla scripts MUST be
    run with the same `--category` / `--eval_fraction` / `--seed` so episode
    numbering lines up for the rerun mechanism.
  * `num_trials_per_task = 1` (the LIBERO-Plus paper protocol).
  * The CycleVLA FSM is run on the *canonical* task instruction recovered per
    task (filename-derived for the 6 non-Language categories; GPT-matched for
    the Language category).
  * Output goes under `rollouts-plus/`.

Usage (with the policy server already running):

  conda activate /hdd2/chenyang/openvla-oft/env-plus
  # 1) transit baseline first (writes rollout videos used to pick the failures)
  python experiments/robot/libero-plus/run_libero_plus_eval_openpi_transit.py \
      --host 0.0.0.0 --port 8000 --task_suite_name libero_spatial --category camera
  # 2) full method -- re-runs only the episodes the baseline failed
  python experiments/robot/libero-plus/run_libero_plus_eval_openpi_cyclevla.py \
      --host 0.0.0.0 --port 8000 --task_suite_name libero_spatial --category camera
"""

import json
import re
import time
import logging
import os
import sys
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Dict, List

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

import wandb

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
    pick_place_states,
    record_robot_state,
    restore_robot_only,
)

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

from dotenv import load_dotenv

# Load environment variables from .env file (OPENAI_API_KEY for VLM detector)
load_dotenv()


# ---- Task suite constants (same as every other eval script) ----

class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


# ---- Logging ----

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


# ---- Latency tracking (copied from the Plus MBR script; cannot import due to
#      the hyphen in the `libero-plus/` directory name) ----

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
        self.current_episode.add(component, latency)

    def finish_episode(self):
        self.episodes.append(self.current_episode)
        self.current_episode = EpisodeLatency()

    def log_episode_summary(self, episode_num: int, log_file):
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

        if self.episodes:
            log_message(f"\n  Running averages across {len(self.episodes)} completed episodes:", log_file)
            components = ['vlm_detector', 'action_inference', 'sampling', 'mbr_computation', 'backtracking']
            for comp in components:
                total = sum(getattr(ep, comp) for ep in self.episodes)
                avg = total / len(self.episodes) if self.episodes else 0
                log_message(f"    {comp}: {avg:.3f}s/episode", log_file)

    def log_final_summary(self, log_file):
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


# ---- Failed-episode scanner (LIBERO-Plus version: scans per-category sub-dirs) ----

def get_failed_episodes_from_videos(cfg, log_file=None):
    """Parse the Stage-1 transit video directory to find FAILED episodes.
    Returns a dictionary in the format: {task_suite_name: [list of failed episode numbers]}

    LIBERO-Plus: episode numbers are a running counter over the *selected*
    task list, so they are only consistent between the transit and cyclevla runs
    when both use the same `--category`. When a single category is being evaluated
    we scan only that category's sub-directory; for `--category all` we scan the
    whole suite directory.
    """
    video_dir = os.path.join(cfg.video_base_dir, cfg.task_suite_name)
    if cfg.category != "all":
        video_dir = os.path.join(video_dir, category_slug(normalize_category(cfg.category)))

    if not os.path.exists(video_dir):
        log_message(f"Video directory not found: {video_dir}", log_file)
        return {cfg.task_suite_name: []}

    failed_episodes = set()
    all_episodes = set()
    pattern = r"--episode=(\d+)--success=(True|False)--"

    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                match = re.search(pattern, file)
                if match:
                    episode_num = int(match.group(1))
                    success = match.group(2) == "True"
                    all_episodes.add(episode_num)
                    if not success:
                        failed_episodes.add(episode_num)

    episodes_list = sorted(list(failed_episodes))
    rerun_dict = {cfg.task_suite_name: episodes_list}

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


# ---- Config ----

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

    # MBR (Minimum Bayes Risk) decoding on backtrack
    mbr_num_seeds: int = 8                           # Candidate action chunks sampled per backtrack
    mbr_distance_metric: str = "l2"                  # l2 | l1 | cosine | correlation | chebyshev
    mbr_use_failed_repulsion: bool = False           # Repel candidates away from previously-failed trajectories
    mbr_r_neighborhood: Optional[int] = None         # r-NN neighborhood size (None = adaptive)
    mbr_vanilla: bool = False                        # Use plain average-distance MBR (no r-NN density)

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
    local_log_dir: str = "./rollouts-plus/logs_plus_openpi_cyclevla"   # Local directory for eval logs
    video_save_dir: str = "./rollouts-plus/rollouts_plus_openpi_cyclevla"
    video_base_dir: str = "./rollouts-plus/rollouts_plus_openpi_transit"  # Stage-1 transit videos to scan for failures
    # Directory holding the per-suite Language-category instruction->canonical-task cache.
    instruction_cache_dir: str = "./experiments/robot/libero-plus"

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 0                                    # Random Seed (for reproducibility)
    rerun_all: bool = False                          # Run all episodes fresh (skip the two-stage rerun mechanism)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    assert 1 <= cfg.num_open_loop_steps <= 10, "num_open_loop_steps must be in [1, 10]"
    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY must be set (in .env) for the VLM detector."
    normalize_category(cfg.category)
    assert 10 <= cfg.eval_fraction <= 100 and cfg.eval_fraction % 10 == 0, \
        f"eval_fraction must be an integer in 10..100 (step 10); got {cfg.eval_fraction}"

    # Two-stage guard: transit baseline videos must exist (unless --rerun_all).
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
                f"  python experiments/robot/libero-plus/run_libero_plus_eval_openpi_transit.py "
                f"--host {cfg.host} --port {cfg.port} --task_suite_name {cfg.task_suite_name} "
                f"--category {cfg.category} --eval_fraction {cfg.eval_fraction}\n"
                f"(keep its --video_save_dir equal to this script's --video_base_dir, "
                f"currently '{cfg.video_base_dir}')\n"
                f"-- or pass --rerun_all True to evaluate every episode fresh."
            )


def _default_log_filename(cfg: GenerateConfig) -> str:
    run_id = (f"EVAL-{cfg.task_suite_name}-{category_slug(normalize_category(cfg.category)) if cfg.category != 'all' else 'all'}"
              f"-frac{cfg.eval_fraction}-openpi-cyclevla-{DATE_TIME}")
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id + ".txt"


def setup_logging(cfg: GenerateConfig, log_filename: str, log_mode: str):
    run_id = os.path.splitext(log_filename)[0]
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, log_filename)
    log_file = open(local_log_filepath, log_mode)
    logger.info(f"Logging to local log file ({'append' if log_mode == 'a' else 'write'}): {local_log_filepath}")
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
    return log_file, local_log_filepath, run_id


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


# ---- MBR chunk sampling (openpi version: stochastic server, no seed control) ----

def sample_and_rank_chunks_mbr(
    cfg: GenerateConfig,
    client: OpenPiClient,
    obs,
    current_subtask: str,
    failed_trajectories: list,
    selection_mode: str = "rep",
    log_file=None,
    latency_tracker=None,
):
    """Sample N candidate action chunks from the openpi server and MBR-rank them.

    The openpi server is stochastic per `infer()` call, so N calls on the same
    observation yield N diverse candidate chunks -- no seed control needed.
    Returns the chunks themselves (best-first) instead of seeds.
    """
    num_seeds = cfg.mbr_num_seeds
    distance_metric = cfg.mbr_distance_metric
    expected_features = cfg.num_open_loop_steps * 6

    current_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    current_euler = np.asarray(quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float64)

    sampling_start = time.time()

    sampled_chunks = []
    sampled_state_trajectories = []

    for _ in range(num_seeds):
        chunk = client.get_action(obs, current_subtask, cfg.num_open_loop_steps)
        sampled_chunks.append(chunk)

        cumulative_pos = current_pos.copy()
        cumulative_rot = R.from_euler("xyz", current_euler)
        state_features: list = []
        for action in chunk:
            robot_action, _, _ = split_openpi_action(action)
            cumulative_pos = cumulative_pos + np.asarray(robot_action[:3])
            cumulative_rot = cumulative_rot * R.from_euler("xyz", np.asarray(robot_action[3:6]))
            state_features.extend(cumulative_pos.tolist())
            state_features.extend(cumulative_rot.as_euler("xyz").tolist())

        while len(state_features) < expected_features:
            state_features.extend([0, 0, 0, 0, 0, 0])
        sampled_state_trajectories.append(np.array(state_features[:expected_features]))

    sampling_time = time.time() - sampling_start
    if latency_tracker:
        latency_tracker.add_to_current('sampling', sampling_time)

    # ---- MBR ranking ----
    mbr_start = time.time()

    X = np.stack(sampled_state_trajectories)
    N = X.shape[0]

    metric_map = {
        "l2": "euclidean", "l1": "cityblock", "cosine": "cosine",
        "correlation": "correlation", "chebyshev": "chebyshev",
    }
    dist_mat = cdist(X, X, metric=metric_map.get(distance_metric, "euclidean"))

    if cfg.mbr_vanilla:
        avg_dist = dist_mat.mean(axis=1)
        ranked_indices = np.argsort(avg_dist)[::-1] if selection_mode == "away" else np.argsort(avg_dist)
        log_message(
            f"Vanilla MBR ranking complete (mode={selection_mode}, metric={distance_metric}); "
            f"top avg-distances: {avg_dist[ranked_indices[:3]]}",
            log_file,
        )
        mbr_time = time.time() - mbr_start
        if latency_tracker:
            latency_tracker.add_to_current('mbr_computation', mbr_time)
        return [sampled_chunks[i] for i in ranked_indices]

    r = cfg.mbr_r_neighborhood if cfg.mbr_r_neighborhood is not None else max(2, min(4, int(np.sqrt(N))))
    r_eff = min(r, max(1, N - 1))

    rnn_radius = np.partition(dist_mat, r_eff, axis=1)[:, r_eff]
    center_idx = int(np.argmin(rnn_radius))
    cluster_idx = np.argsort(dist_mat[center_idx])[:r_eff]
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
    repulse = 1.0 / (1.0 + np.exp(-dfail_norm))
    lambda_fail = 0.5

    if selection_mode == "away":
        final_scores = dmed_norm + lambda_fail * repulse
    else:
        final_scores = -rnn_norm + lambda_fail * repulse
    ranked_indices = np.argsort(final_scores)[::-1]

    log_message(
        f"MBR ranking complete (mode={selection_mode}, metric={distance_metric}); "
        f"medoid candidate index {medoid_local}; top scores: {final_scores[ranked_indices[:3]]}",
        log_file,
    )

    mbr_time = time.time() - mbr_start
    if latency_tracker:
        latency_tracker.add_to_current('mbr_computation', mbr_time)

    return [sampled_chunks[i] for i in ranked_indices]


# ---- Episode runner (two-phase: to_check -> VLM -> to_complete) ----

def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    client: OpenPiClient,
    initial_state=None,
    log_file=None,
    latency_tracker=None,
):
    """Run a single episode with the proactive transit/backtrack loop."""
    if latency_tracker is None:
        latency_tracker = LatencyTracker()

    # Get subtasks. The bare lowercased subtask string is the prompt the openpi
    # policy was trained with.
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

    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images, replay_wrist_images, replay_subtasks = [], [], []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    success = False
    current_state = states[0]
    subtask_hist, exe_type_hist = [current_state], ["init"]
    robot_state_hist: List[Dict[str, Any]] = []

    subtask_phase = "to_check"

    # Goal/long tasks use robust (consecutive/recurring) confirmation for the
    # 90% progress signal.
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
    subtask_retry_count: Dict[str, int] = {}
    subtask_chunk_rankings: Dict[str, list] = {}
    current_chunk_index: Dict[str, int] = {}
    subtask_failed_trajectories: Dict[str, list] = {}

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
                start_time = time.time()
                actions = client.get_action(obs, current_state, cfg.num_open_loop_steps)
                action_queue.extend(actions)
                latency_tracker.add_to_current('action_inference', time.time() - start_time)

            # Record full sim snapshot BEFORE executing the next action
            robot_state_hist = record_robot_state(robot_state_hist, current_state, env, obs)

            # Split the 9-dim openpi action
            robot_action, stop_signal, progress_signal = split_openpi_action(action_queue.popleft())

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
                    if progress_signal >= cfg.progress_threshold:
                        count_check_signals += 1
                        log_message(
                            f"90% signal #{count_check_signals} at step {t} (progress={progress_signal:.2f})", log_file
                        )
                        if count_check_signals >= 2:
                            vlm_check_needed = True
                            log_message(f"90% progress confirmed after {count_check_signals} high signals", log_file)

                if vlm_check_needed:
                    current_img = get_libero_image(obs)
                    current_wrist_img = get_libero_wrist_image(obs)
                    for _ in range(32):
                        replay_images.append(current_img)
                        replay_wrist_images.append(current_wrist_img)
                        replay_subtasks.append(f"VLM_90%_check: {current_state}")

                    start_time = time.time()
                    res = vlm_detector.detect_subtask(
                        current_state, states, subtask_hist, task_description, current_img, current_wrist_img
                    )
                    detected_state, exe_type, reason = vlm_detector.extract_res(res)
                    latency_tracker.add_to_current('vlm_detector', time.time() - start_time)

                    log_message(
                        f"VLM check at 90% for subtask `{current_state}` -> next: {detected_state}, "
                        f"type: {exe_type}, reason: {reason}",
                        log_file,
                    )

                    if exe_type == "backtrack" and detected_state not in states:
                        log_message(
                            f"VLM backtrack target `{detected_state}` not in subtask list; treating as transit.",
                            log_file,
                        )
                        exe_type = "transit"

                    if exe_type == "backtrack":
                        subtask_retry_count[current_state] = subtask_retry_count.get(current_state, 0) + 1
                        retry_num = subtask_retry_count[current_state]

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

                        failed_feat = extract_trajectory_features(
                            robot_state_hist, current_state, start_idx=0, end_idx=cfg.num_open_loop_steps
                        )
                        subtask_failed_trajectories.setdefault(current_state, []).append(failed_feat)

                        current_state = detected_state
                        subtask_hist.append(current_state)
                        exe_type_hist.append(f"backtrack_90%_retry{retry_num}")
                        action_queue.clear()

                        start_time = time.time()
                        reversed_snaps = backtrace_robot_states(robot_state_hist, current_state)
                        for snap in reversed_snaps:
                            restore_robot_only(env, snap)
                            obs, _, _, _ = env.step(get_libero_dummy_action("openpi"))
                            replay_images.append(get_libero_image(obs))
                            replay_wrist_images.append(get_libero_wrist_image(obs))
                            replay_subtasks.append(f"backtrack_from_90%_retry{retry_num}")
                        latency_tracker.add_to_current('backtracking', time.time() - start_time)

                        # MBR decoding: sample N chunks on first backtrack; reuse ranking later
                        if current_state not in subtask_chunk_rankings:
                            log_message(
                                f"Sampling {cfg.mbr_num_seeds} candidate chunks for MBR decoding "
                                f"of subtask `{current_state}`...",
                                log_file,
                            )
                            subtask_chunk_rankings[current_state] = sample_and_rank_chunks_mbr(
                                cfg, client, obs, current_state,
                                subtask_failed_trajectories.get(current_state, []),
                                log_file=log_file,
                                latency_tracker=latency_tracker,
                            )
                            current_chunk_index[current_state] = 0
                        else:
                            current_chunk_index[current_state] += 1
                            if current_chunk_index[current_state] >= len(subtask_chunk_rankings[current_state]):
                                current_chunk_index[current_state] = 0

                        chunk_idx = current_chunk_index[current_state]
                        action_queue.extend(subtask_chunk_rankings[current_state][chunk_idx])
                        log_message(
                            f"Retrying subtask `{current_state}` with MBR-ranked chunk "
                            f"#{chunk_idx + 1}/{len(subtask_chunk_rankings[current_state])}",
                            log_file,
                        )

                        # Reset phase tracking for the retry
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
                        # transit: continue current subtask toward completion
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

    # Drain remaining queued actions (best effort)
    if not success and len(action_queue) > 0:
        log_message(f"Executing {len(action_queue)} remaining actions in queue...", log_file)
        while len(action_queue) > 0:
            try:
                robot_action, _, _ = split_openpi_action(action_queue.popleft())
                obs, reward, done, info = env.step(robot_action.tolist())
            except ValueError:
                break
            if done:
                success = True
                log_message("Environment signaled done while executing remaining actions", log_file)
                break
            replay_images.append(get_libero_image(obs))
            replay_wrist_images.append(get_libero_wrist_image(obs))
            replay_subtasks.append(f"finishing: {current_state}")

    return success, replay_images, replay_wrist_images, replay_subtasks


# ---- Task runner (LIBERO-Plus variant: per-category sub-dirs + rerun_dict) ----

def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    category: str,
    client: OpenPiClient,
    total_episodes=0,
    total_successes=0,
    category_stats=None,
    rerun_dict=None,
    log_file=None,
    latency_tracker=None,
):
    """Run evaluation for a single LIBERO-Plus task variant."""
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    env, _ = get_libero_env(
        task, "openpi",
        resolution=env_render_resolution(category, cfg.env_img_res),
    )
    task_description = resolve_canonical_task(
        task.name, category, env, cfg.task_suite_name, cfg.instruction_cache_dir,
        log_fn=lambda m: log_message(m, log_file),
    )
    video_label = rollout_label(category, env, task_description)
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

        if cfg.rerun_all:
            rerun = True
        else:
            rerun = True if total_episodes + 1 in rerun_dict[cfg.task_suite_name] else False

        if rerun:
            success, replay_images, replay_wrist_images, replay_subtasks = run_episode(
                cfg, env, task_description, client, initial_state, log_file, latency_tracker,
            )

            if latency_tracker:
                latency_tracker.log_episode_summary(total_episodes, log_file)
                latency_tracker.finish_episode()
        else:
            success = True

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        if category_stats is not None:
            category_stats[category][0] += 1
            if success:
                category_stats[category][1] += 1

        if rerun:
            save_rollout_video_decomposed(
                replay_images, total_episodes, success=success, task_description=video_label,
                video_save_dir=video_dir, log_file=log_file, subtasks=replay_subtasks
            )
            save_rollout_video_decomposed(
                replay_wrist_images, total_episodes, success=success, task_description=video_label,
                video_save_dir=video_dir, log_file=log_file, subtasks=replay_subtasks, wrist=True
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


# ---- Main ----

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate the openpi CycleVLA policy on LIBERO-Plus tasks."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    # Select variants: filter by category and sub-sample. MUST match the transit
    # run's args so episode numbering lines up for the rerun mechanism.
    selected = select_plus_tasks(
        task_suite, cfg.task_suite_name, cfg.category, cfg.eval_fraction, cfg.seed
    )
    num_selected = len(selected)

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

    latency_tracker = LatencyTracker()

    category_stats = defaultdict(lambda: [0, 0])
    for k, v in prior_category_stats.items():
        category_stats[k] = list(v)

    if start_index >= num_selected:
        log_message(
            f"Nothing to do: {start_index}/{num_selected} tasks already complete on disk.",
            log_file,
        )
    else:
        client = OpenPiClient(host=cfg.host, port=cfg.port)

        # Scan Stage-1 transit videos ONCE for the set of failed episodes
        rerun_dict = get_failed_episodes_from_videos(cfg, log_file)

        for i, (task_id, category) in enumerate(tqdm.tqdm(
            selected[start_index:], initial=start_index, total=num_selected
        )):
            total_episodes, total_successes = run_task(
                cfg, task_suite, task_id, category, client,
                total_episodes, total_successes, category_stats,
                rerun_dict, log_file, latency_tracker,
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

    latency_tracker.log_final_summary(log_file)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
