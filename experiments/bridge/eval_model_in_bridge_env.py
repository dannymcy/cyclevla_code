"""
eval_model_in_bridge_env.py

Runs a model checkpoint in a real-world Bridge V2 environment.

Usage:
    # VLA:
    python experiments/robot/bridge/eval_model_in_bridge_env.py \
        --model.type <VLM_TYPE> \
        --pretrained_checkpoint <CHECKPOINT_PATH>

    # Octo:
    python experiments/robot/bridge/eval_model_in_bridge_env.py --model_family octo \
         --blocking True --control_frequency 2.5

    # RT-1-X:
    python experiments/robot/bridge/eval_model_in_bridge_env.py --model_family rt_1_x \
        --pretrained_checkpoint <CHECKPOINT_PATH>
"""

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import cv2
import draccus
import numpy as np

from prismatic.conf import ModelConfig, ModelRegistry

# TODO (@moojink) Hack so that the interpreter can find experiments.robot
sys.path.append("../..")
from experiments.bridge.utils import (
    draw_bboxes,
    draw_gripper,
    draw_interactive,
    get_action,
    get_image_resize_size,
    get_model,
    get_next_task_label,
    get_octo_policy_function,
    get_preprocessed_image,
    get_widowx_env,
    make_reasoning_image,
    refresh_obs,
    save_rollout_gif,
)


@dataclass
class GenerateConfig:
    # fmt: off

    # ModelConfig from `prisma/conf/models.py`; override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.REPRODUCTION_7B.model_id)
    )
    model_family: str = "llava"                                 # Base VLM model family (for prompt builder)

    # Model Parameters
    pretrained_checkpoint: Union[str, Path] = Path(             # Pretrained VLA checkpoint to load
        "/scr/moojink/checkpoints/tri/reproduction-llava-v15+mx-bridge+n1+b32+x7/checkpoints/"
        "step-077500-epoch-00-loss=0.0488.pt"
    )

    # Environment-Specific Parameters
    host_ip: str = "localhost"
    port: int = 5556

    # Note (@moojink) =>> Setting initial orientation with a 30 degree offset -- more natural!
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, 0., 0.16])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = True
    max_episodes: int = 500
    max_steps: int = 600
    control_frequency: float = 2

    # Training stage (doesn't matter here, but the loading function expects the argument)
    stage: str = "vla-finetune"

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    # Load Model --> Get Expected Image Dimensions
    model = get_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # [Octo] Create JAX JIT-compiled policy function.
    policy_fn = None
    if cfg.model_family == "octo":
        policy_fn = get_octo_policy_function(model)

    # Initialize the Widow-X Environment
    env = get_widowx_env(cfg, model)

    # === Start Evaluation ===
    task_label = ""
    episode_idx = 0
    # prev_action = np.array([0, 0, 0, 0, 0, 0, 1.0])

    while episode_idx < cfg.max_episodes:
        # Get Task Description from User
        task_label = get_next_task_label(task_label)
        rollout_images = []

        model.reset_async()

        # Reset Environment
        obs, _ = env.reset()

        # Setup
        t = 0
        zero_action_count = 0
        step_duration = 1.0 / cfg.control_frequency

        # Start Episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the Camera Image and Proprioceptive State
                    print("Taking image...", end=" ")
                    obs = refresh_obs(obs, env)
                    time.sleep(0.1)
                    obs = refresh_obs(obs, env)
                    print("done.")

                    # Save Image for Rollout GIF =>> Switch on History / No History
                    if len(obs["full_image"].shape) == 4:
                        video_image = obs["full_image"][-1]
                    else:
                        video_image = obs["full_image"]

                    # Get Preprocessed Image
                    obs["full_image"] = get_preprocessed_image(obs, resize_size)

                    # Query Model --> Get Action
                    info_dict = dict()
                    action = get_action(cfg, model, obs, task_label, policy_fn, info_dict=info_dict)

                    # Add the reasoning to the image
                    try:
                        reasoning_img, metadata = make_reasoning_image(info_dict["decoded_tokens"])
                        draw_gripper(video_image, metadata["gripper"])
                        draw_bboxes(video_image, metadata["bboxes"])
                        draw_interactive(video_image, model.use_interactive)
                        video_image = np.concatenate([video_image, reasoning_img], axis=1)
                    except ValueError:
                        print("\033[93m\033[1mWARNING:\033[0m Can't draw reasoning image.")
                        video_image = np.concatenate([video_image, np.zeros_like(video_image)], axis=1)
                    rollout_images.append(video_image)

                    if True:  # has issues with X11 display on dgx
                        bgr_img = cv2.cvtColor(video_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite("image.png", bgr_img)

                    # [OpenVLA] End episode early if the robot doesn't move at all for a few consecutive steps!
                    #   - Reason: Inference is pretty slow with a single local GPU...
                    if (
                        cfg.model_family == "llava"
                        and np.isclose(np.linalg.norm(action), 1, atol=0.01)
                        and np.linalg.norm(action[:6]) < 0.01
                    ):
                        zero_action_count += 1
                        if zero_action_count == 11:
                            print("Ending episode early due to robot inaction.")
                            break
                    else:
                        zero_action_count = 0

                    # Execute Action
                    print("action:", action)
                    t += 1
                    # TODO: If action is malformed, i.e. not all 7 elements are action tokens, repeat the step
                    obs, _, _, _, _ = env.step(action)
                    # prev_action = action

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt")
                    while True:
                        print("Press Enter to use interactive mode, or type 'exit' to terminate.")
                        request = input()
                        if request in ["exit", "continue", ""]:
                            break
                        else:
                            continue

                    if request == "":
                        model.use_interactive = True
                        continue
                    elif request == "continue":
                        continue
                else:
                    print(f"\nCaught exception: {e}")
                    traceback.print_exception(type(e), e, e.__traceback__)
                    print("")

                break

        # Save a Replay GIF of the Episode
        save_rollout_gif(rollout_images, f"{episode_idx}_{task_label.replace(' ', '-')}")

        # Redo Episode or Continue...
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()
