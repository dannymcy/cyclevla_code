"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video_decomposed(rollout_images, idx, success, task_description,
                       video_save_dir=None, log_file=None, subtasks=None, wrist=False, save=True, seed=None):
    """Saves an MP4 replay of an episode, optionally overlaying subtask text.
    Saves two versions when subtasks provided: one with subtitles (_subtitled) and one without (_raw).
    If no subtasks, saves only the raw version.
    """
    assert subtasks is None or len(subtasks) == len(rollout_images), "subtasks must match the number of images"

    # Still return the json_dir path even if not saving video
    task_clean = task_description.replace(" ", "_")
    json_dir = os.path.join(video_save_dir, task_clean)

    if save and seed is not None:
        os.makedirs(json_dir, exist_ok=True)

    if not save:
        return None, None, json_dir

    rollout_dir = f"{video_save_dir}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)

    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    
    if seed is not None:
        processed_task_description = f"{seed}_{processed_task_description}"

    if not wrist:
        base_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}"
    else:
        base_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--wrist--task={processed_task_description}"

    # Always save raw version
    mp4_path_raw = f"{base_path}_raw.mp4"
    video_writer_raw = imageio.get_writer(mp4_path_raw, fps=30)
    
    for img in rollout_images:
        video_writer_raw.append_data(img)
    video_writer_raw.close()
    
    print(f"Saved rollout MP4 (raw) at path {mp4_path_raw}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 (raw) at path {mp4_path_raw}\n")

    # Only save subtitled version if subtasks provided
    mp4_path_subtitled = None
    if subtasks:
        mp4_path_subtitled = f"{base_path}_subtitled.mp4"
        video_writer_subtitled = imageio.get_writer(mp4_path_subtitled, fps=30)

        # Get image dimensions for scaling
        img_height, img_width = rollout_images[0].shape[:2]
        scale_factor = img_width / 256.0  # Scale relative to 256x256 baseline

        # Scale font size and pixel-based layout parameters
        font_size = max(10, int(10 * scale_factor))
        max_width_px = int(150 * scale_factor)
        padding = int(5 * scale_factor)
        
        # wrap_width is CHARACTER count, NOT pixels - keep it fixed!
        wrap_width = 25

        # Load font with scaled size
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
        except:
            font = ImageFont.load_default()

        for i, img in enumerate(rollout_images):
            # Convert base image to RGBA
            pil_img = Image.fromarray(img).convert("RGBA")

            # Create transparent overlay
            overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            text = subtasks[i]
            lines = textwrap.wrap(text, width=wrap_width)

            # Text layout
            line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 2
            total_text_height = line_height * len(lines)
            x = pil_img.width - max_width_px - padding
            y = pil_img.height - total_text_height - padding

            # Semi-transparent white rectangle
            draw.rectangle(
                [x - 2, y - 2, pil_img.width - padding + 2, pil_img.height - padding + 2],
                fill=(255, 255, 255, 180)  # RGBA: semi-transparent white
            )

            # Draw each line of text in black
            for j, line in enumerate(lines):
                draw.text((x, y + j * line_height), line, fill=(0, 0, 0, 255), font=font)

            # Merge overlay with original image
            pil_img = Image.alpha_composite(pil_img, overlay).convert("RGB")
            video_writer_subtitled.append_data(np.array(pil_img))

        video_writer_subtitled.close()
        print(f"Saved rollout MP4 (subtitled) at path {mp4_path_subtitled}")
        if log_file is not None:
            log_file.write(f"Saved rollout MP4 (subtitled) at path {mp4_path_subtitled}\n")

    return mp4_path_subtitled, mp4_path_raw, json_dir


def save_rollout_video(rollout_images, idx, success, task_description, video_save_dir=None, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"{video_save_dir}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
