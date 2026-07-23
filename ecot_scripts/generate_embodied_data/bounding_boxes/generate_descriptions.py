import argparse
import json
import os
import warnings

import torch
from PIL import Image
from tqdm import tqdm
import sys
from utils import NumpyFloatValuesEncoder

import tensorflow_datasets as tfds
from prismatic import load
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


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
from vlm_utils.utils import *


# watch -n 1 nvidia-smi
# conda activate /hdd2/chenyang/openvla-oft/env
# CUDA_VISIBLE_DEVICES="0" python ecot_scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py --task-suite-id 0
# CUDA_VISIBLE_DEVICES="0" python ecot_scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py --dataset "bridge"

parser = argparse.ArgumentParser()

parser.add_argument("--gpu",  default=0, type=int)
parser.add_argument("--save-path", default="/hdd2/chenyang/openvla-oft/vlm_response/scene_description")
parser.add_argument("--dataset", default="libero")  # ["libero", "bridge"]
parser.add_argument("--task-suite-id", default=0, type=int)  # [0, 1, 2, 3]
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--vlm", default="prismatic", type=str)  # [prismatic, mllama]
args = parser.parse_args()

device = f"cuda:{args.gpu}"
hf_token = os.environ['HUGGINGFACE_TOKEN']

# Set dir
if args.dataset == "libero":
    task_suite_list = ["libero_spatial_no_noops", "libero_object_no_noops", "libero_goal_no_noops", "libero_10_no_noops"]
    task_suite = task_suite_list[args.task_suite_id]
    data_dir = "/hdd2/chenyang/openvla-oft/LIBERO/libero/libero/modified_libero_rlds"
elif args.dataset == "bridge":
    task_suite = "bridge"
    data_dir = "/hdd2/chenyang/openvla-oft"
results_path = os.path.join(args.save_path, args.dataset, task_suite)

# Set random seed
set_seed_everywhere(args.seed)
warnings.filterwarnings("ignore")


# Create results directory if it doesn't exist
os.makedirs(results_path, exist_ok=True)
results_json_path = os.path.join(results_path, f"results.json")
images_path = os.path.join(results_path, "images")
os.makedirs(images_path, exist_ok=True)


def create_user_prompt(lang_instruction):
    user_prompt = "Briefly describe the things in this scene and their spatial relations to each other."
    # user_prompt = "Briefly describe the objects in this scene."
    # user_prompt = f"Describe all objects present in this scene and provide a detailed account of their spatial relationships to one another. Ensure that no object is omitted and focus on their relative positions and arrangements."

    lang_instruction = lang_instruction.strip()
    if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
        lang_instruction = lang_instruction[:-1]
    if len(lang_instruction) > 0 and " " in lang_instruction:
        user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    return user_prompt



# Initialize VLM
if args.vlm == "prismatic":
    # Load Prismatic VLM
    vlm_model_id = "prism-dinosiglip+7b"
    print(f"Loading Prismatic VLM ({vlm_model_id})...")
    vlm = load(vlm_model_id, hf_token=hf_token)
    vlm = vlm.to(device, dtype=torch.bfloat16)
else:
    model, tokenizer, processor = vlm_init(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct")


# LIBERO RLDS can be loaded using tfds directly
# https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py

# Some differences of data structure of LIBERO RLDS compared to Bridge RLDS
# https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/features.json
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


# Process tasks
results_json = {}
i = 0
for episode_idx, episode in tqdm(enumerate(ds), desc="Processing Episodes"):
    if i > 500: break
    i += 1

    if task_suite == "bridge":
        episode_id = episode["episode_metadata"]["episode_id"].numpy()
        episode_id_pseudo = str(episode_idx) 
    else:
        episode_id = str(episode_idx) 
        episode_id_pseudo = str(episode_idx) 
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()

    for step in episode["steps"]:
        lang_instruction = step["language_instruction"].numpy().decode()
        try:
            if task_suite == "bridge":
                image = Image.fromarray(step["observation"]["image_0"].numpy())
            else:
                image = Image.fromarray(step["observation"]["image"].numpy())
        except KeyError:
            print(f"Warning: No image found in step")
            continue
        
        # Save the image
        image_filename = f"episode_{episode_id_pseudo}.png"
        image_path = os.path.join(images_path, image_filename)
        image.save(image_path)   
        
        # Create user prompt based on task description
        user_prompt = create_user_prompt(lang_instruction)
        if args.vlm == "prismatic":
            prompt_builder = vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=user_prompt)
            prompt_text = prompt_builder.get_prompt()
            
            # Generate caption
            caption = vlm.generate(
                image,
                prompt_text,
                do_sample=True,
                temperature=0.4,
                max_new_tokens=64,
                min_length=1,
            )
        else:
            caption = vlm_inference_mllama(model, tokenizer, processor, user_prompt, image=image, max_new_tokens=64, temperature=0.2)
        break
        
    # Store results
    episode_json = {
        "episode_id_pseudo": int(episode_id_pseudo),
        "episode_id": int(episode_id),
        "file_path": file_path,
        "caption": caption,
    }
    
    if file_path not in results_json:
        results_json[file_path] = {}
    
    results_json[file_path][int(episode_id)] = episode_json
    
    # Save results after each episode
    with open(results_json_path, "w") as f:
        json.dump(results_json, f, cls=NumpyFloatValuesEncoder)
