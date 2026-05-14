import json
import os
import re
import time
import torch
from PIL import Image
from tqdm import tqdm
import sys
import argparse
import warnings
import tensorflow as tf
import tensorflow_datasets as tfds

sys.path.append(os.getcwd())
from ecot_scripts.generate_embodied_data.primitive_movements import get_move_primitives_episode, detect_gripper_changes

# Import Libero-specific utilities
sys.path.append("../..")
sys.path.append(os.getcwd())
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from experiments.robot.robot_utils import (
    get_image_resize_size,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from libero.libero import benchmark
from vlm_utils.query import *
from decompose_utils import *



def decompose_single_traj(episode_id, episode, captions, results_path):
    gripper_states = detect_gripper_changes(episode, task_suite=task_suite)
    if task_suite == "bridge":
        thresholds = [0.03, 0.03, 0.03]
    else:
        thresholds = [0.02, 0.0075, 0.03]

    move_primitives = get_move_primitives_episode(episode, task_suite=task_suite, optimize_thresholds=task_suite!="bridge", thresholds=[0.02, 0.0075, 0.03])

    # Create episode directory
    episode_dir = os.path.join(results_path, f"episode_{episode_id}")
    os.makedirs(episode_dir, exist_ok=True)

    # Compute ft
    ft = dict()
    ft["state_3d"] = [[float(x) for x in step["observation"]["state"][:3].numpy()] for step in episode["steps"]]
    ft["move_primitive"] = [move[0] for move in move_primitives]

    # Compute mt
    if task_suite == "bridge":
        mt = {
            "episode_id_pseudo": str(episode_id),
            "episode_id": str(int(episode["episode_metadata"]["episode_id"].numpy())),
            "file_path": str(episode["episode_metadata"]["file_path"].numpy())[2:-1],
            "n_steps": len(episode["steps"]),
            "language_instruction": str(next(iter(episode["steps"]))["language_instruction"].numpy().decode()),
        }
    else:
        mt = {
            "episode_id_pseudo": str(episode_id),
            "episode_id": str(episode_id),
            "file_path": str(episode["episode_metadata"]["file_path"].numpy())[2:-1],
            "n_steps": len(episode["steps"]),
            "language_instruction": str(next(iter(episode["steps"]))["language_instruction"].numpy().decode()),
        }
    mt["caption"] = captions[mt["file_path"]][mt["episode_id"]]["caption"]


    # Find continuous chunks
    chunks = []
    current_chunk = {"value": gripper_states[0], "start": 0, "end": 0}

    for i, state in enumerate(gripper_states):
        if state == current_chunk["value"]:
            current_chunk["end"] = i
        else:
            chunks.append(current_chunk)
            current_chunk = {"value": state, "start": i, "end": i}
    chunks.append(current_chunk)  # Add last chunk

    # Save summary
    entry = {"metadata": mt, "chunks": chunks, "features": ft}
    with open(os.path.join(episode_dir, "chunks_summary.json"), "w") as f:
        json.dump(entry, f, indent=2)

    # Save images by chunk
    steps = list(episode["steps"])
    for chunk_idx, chunk in enumerate(chunks):
        val = chunk["value"]
        val_str = "close" if val == -1 else "open" if val == 1 else "idle"
        chunk_dir = os.path.join(episode_dir, f"chunk_{chunk_idx}_{val_str}")
        os.makedirs(chunk_dir, exist_ok=True)

        for step_idx in range(chunk["start"], chunk["end"] + 1):
            if step_idx < len(steps):
                step = steps[step_idx]

                # Get image from observation
                try:
                    if task_suite == "bridge":
                        image = Image.fromarray(step["observation"]["image_0"].numpy())
                    else:
                        image = Image.fromarray(step["observation"]["image"].numpy())
                except KeyError:
                    print(f"Warning: No image found in step {step_idx}")
                    continue

                image.save(os.path.join(chunk_dir, f"step_{step_idx}.png"))

    return len(chunks), len(episode["steps"])


def decompose_trajectories(ds, captions_path=None, results_path=None):
    os.makedirs(results_path, exist_ok=True)

    # Load scene descriptions
    with open(captions_path, "r") as captions_file:
        captions_dict = json.load(captions_file)

    # Load existing summary if it exists
    summary_file_path = os.path.join(results_path, "all_episodes_summary.json")
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as f:
            episodes_summary = json.load(f)
    else:
        episodes_summary = {
            "total_episodes": 0,
            "episodes": {}
        }
    
    for i, episode in enumerate(ds):
        # if i != 82: continue

        # Extact the gripper states
        n_chunks, n_steps = decompose_single_traj(i, episode, captions_dict, results_path)

        # Add to the summary
        episodes_summary["episodes"][str(i)] = {
            "num_chunks":  n_chunks,
            "total_steps": n_steps
        }
        sorted_episodes = dict(sorted(
            episodes_summary["episodes"].items(),
            key=lambda item: int(item[0])
        ))
        episodes_summary["episodes"] = sorted_episodes
        episodes_summary["total_episodes"] = len(episodes_summary["episodes"])
            
        # Save the complete summary
        with open(summary_file_path, "w") as f:
            json.dump(episodes_summary, f, indent=2)


def process_trajectories(ds, decompose_path=None, results_path=None):
    os.makedirs(results_path, exist_ok=True)

    # First load the overall summary
    with open(os.path.join(decompose_path, "all_episodes_summary.json"), "r") as f:
        episodes_summary = json.load(f)

    # Load existing processed summary if it exists
    summary_file_path = os.path.join(results_path, "all_episodes_summary.json")
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as f:
            episodes_summary_processed = json.load(f)
    else:
        episodes_summary_processed = {
            "total_episodes": 0,
            "episodes": {}
        }

    if "gpt" not in args.vlm:
        lm = vlm_init(model_name=args.vlm)
        lm.append(args.vlm)
    else:
        lm = None
    
    # Read each episode's data
    for i, episode_id in enumerate(episodes_summary["episodes"].keys()):
        # if i != 43: continue

        episode_dir = os.path.join(decompose_path, f"episode_{episode_id}")
        chunks_json_path = os.path.join(episode_dir, "chunks_summary.json")
        
        if os.path.exists(chunks_json_path):
            with open(chunks_json_path, "r") as f:
                episode_data = json.load(f)

        if "libero" in task_suite:
            episode_data = process_libero_coarse(decompose_path, results_path, episode_id, episodes_summary, episode_data, task_suite, lm)
        # elif task_suite == "bridge":

        # Add to the summary
        if episode_data is not None:
            episodes_summary_processed["episodes"][str(i)] = {
                "num_chunks": len(episode_data["chunks"]),
                "total_steps": episode_data["metadata"]["n_steps"]
            }
            sorted_episodes_processed = dict(sorted(
                episodes_summary_processed["episodes"].items(),
                key=lambda item: int(item[0])
            ))
            episodes_summary_processed["episodes"] = sorted_episodes_processed
            episodes_summary_processed["total_episodes"] = len(episodes_summary_processed["episodes"])
        
            # Save the complete summary
            with open(summary_file_path, "w") as f:
                json.dump(episodes_summary_processed, f, indent=2)


def visualize_decomposed_rlds(original_rlds_dir, task_suite):
    import numpy as np

    ds = tfds.load(task_suite, data_dir=original_rlds_dir, split="train")

    for i, episode in enumerate(ds):
        if i != 0:
            continue

        print(f"\nProcessing task suite: {task_suite}")
        print("===== EPISODE METADATA =====")
        metadata = episode.get("episode_metadata", {})
        for k, v in metadata.items():
            print(f"{k}: {v.numpy() if hasattr(v, 'numpy') else v}")

        print("\n===== STEPS =====")
        original_steps = list(episode["steps"])
        for step_idx, step in enumerate(original_steps):
            print(f"\n----- STEP {step_idx} -----")
            print("action:", np.round(step["action"].numpy(), 3))
            print("discount:", float(step["discount"].numpy()))
            print("is_first:", bool(step["is_first"].numpy()))
            print("is_last:", bool(step["is_last"].numpy()))
            print("is_terminal:", (step["is_terminal"].numpy()))
            print("language_instruction:", step["language_instruction"].numpy())

            obs = step["observation"]
            print("observation:")
            if "image" in obs:
                print("  image: shape={}, dtype={}".format(obs["image"].shape, obs["image"].dtype))
            if "wrist_image" in obs:
                print("  wrist_image: shape={}, dtype={}".format(obs["wrist_image"].shape, obs["wrist_image"].dtype))
            if "state" in obs:
                print("  state:", np.round(obs["state"].numpy(), 3))
            if "joint_state" in obs:
                print("  joint_state:", np.round(obs["joint_state"].numpy(), 3))
            print("reward:", float(step["reward"].numpy()))



parser = argparse.ArgumentParser()

parser.add_argument("--gpu",  default=0, type=int)
parser.add_argument("--save-path", default="/hdd2/kai/openvla-oft/vlm_response")
parser.add_argument("--dataset", default="libero")  # ["libero", "bridge"]
parser.add_argument("--task-suite-id", default=0, type=int)  # [0, 1, 2, 3]
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--vlm", default="gpt", type=str)  # ["gpt", "Qwen/Qwen2.5-VL-7B-Instruct"]
args = parser.parse_args()

device = f"cuda:{args.gpu}"

# Set dir
if args.dataset == "libero":
    task_suite_list = ["libero_spatial_no_noops", "libero_object_no_noops", "libero_goal_no_noops", "libero_10_no_noops"]
    task_suite = task_suite_list[args.task_suite_id]
    data_dir = "/hdd2/kai/openvla-oft/LIBERO/libero/libero/modified_libero_rlds"
elif args.dataset == "bridge":
    task_suite = "bridge"
    data_dir = "/hdd2/kai/openvla-oft"
    
# Set random seed
set_seed_everywhere(args.seed)
warnings.filterwarnings("ignore")



# watch -n 1 nvidia-smi
# conda activate /hdd2/kai/openvla-oft/env_2
# CUDA_VISIBLE_DEVICES="0,1,2" python ecot_scripts/generate_embodied_data/decompose_traj.py --task-suite-id 0 --vlm "gpt"
# CUDA_VISIBLE_DEVICES="0,1,2" python ecot_scripts/generate_embodied_data/decompose_traj.py --dataset "bridge"

if __name__ == "__main__":
    # Initialize Libero task suite
    if task_suite == "bridge":
        ds = tfds.load(
            "bridge_dataset",
            data_dir=data_dir,
            split="train",
        )
    else:
        ds = tfds.load(
            task_suite,
            data_dir=data_dir,
            split="train",
        )

    # NOTE the generator expects the captions.json file to be present in the working directory
    # The captions should be generated using the script in
    # scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py
    captions_path = os.path.join(args.save_path, "scene_description", args.dataset, task_suite, "results.json")
    decompose_path = os.path.join(args.save_path, "decompose_traj", args.dataset, task_suite)
    process_path = os.path.join(args.save_path, "process_traj", args.dataset, task_suite)

    decompose_trajectories(ds, captions_path=captions_path, results_path=decompose_path)
    process_trajectories(ds, decompose_path=decompose_path, results_path=process_path)

    # visualize_decomposed_rlds(data_dir, "libero_spatial_no_noops")
    visualize_decomposed_rlds("/hdd2/kai/openvla-oft/decomposed_dataset/libero_sub_progress", "libero_decomposed_progress")
