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
from ecot_scripts.generate_embodied_data.primitive_movements import get_move_primitives_episode
from prismatic import load

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


class Gemini:
    def __init__(self):
        api_key = "PUT_KEY_HERE"
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def safe_call(self, f):
        while True:
            try:
                res = f()
                return res
            except ResourceExhausted:
                time.sleep(5)

    def generate(self, prompt):
        chat = self.safe_call(lambda: self.model.start_chat(history=[]))
        response = self.safe_call(lambda: chat.send_message(prompt).text)

        for i in range(8):
            if "FINISHED" in response:
                print(f"n_retries: {i}")
                return response

            response = response + self.safe_call(lambda: chat.send_message("Truncated, please continue.").text)

        print(f"n_retries: {iter}")

        return None


def generate_with_finished_check(model, tokenizer, processor, prompt, max_attempts=3, **kwargs):
    for attempt in range(max_attempts):
        # Generate the response
        response = vlm_inference(model, tokenizer, processor, prompt, **kwargs)
        
        # Check if the response contains "FINISHED"
        if "FINISHED" in response:
            return response
        
        # If not, log the attempt and try again
        print(f"Attempt {attempt+1}/{max_attempts}: Response didn't contain 'FINISHED'. Regenerating...")
    
    # If we've exhausted all attempts, return the last response anyway
    print(f"Warning: Generated response doesn't contain 'FINISHED' after {max_attempts} attempts.")
    return response


def build_prompt(features, language_instruction, caption=None, list_only_moves=False):
    structured_features = "{\n"

    keys = list(features.keys())

    for i in range(len(features[keys[0]])):
        if list_only_moves:
            structured_features = structured_features + f'    {i}: "{features["move_primitive"][i]}"\n'
        else:
            structured_features = structured_features + f'    {i}: {"{"}\n'

            for key in keys:
                feature_value = features[key][i]
                if isinstance(feature_value, str):
                    feature_value = f'"{feature_value}"'

                structured_features = structured_features + f'        "{key}": {feature_value},\n'

            structured_features = structured_features + "    },\n"

    structured_features = structured_features + "}"

    if list_only_moves:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on the "
            "trajectory and describes the move that is about to be executed."
        )
    else:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on "
            "the trajectory. The provided features are the following:\n"
            "\n"
            '- "state_3d" are the current 3d coordinates of the robotic arm end effector; '
            "moving forward increases the first coordinate; moving left increases the second "
            "coordinate; moving up increases the third coordinate,\n"
            '- "move_primitive" describes the move that is about to be executed,\n'
            '- "gripper_position" denotes the location of the gripper in the 256x256 image observation'
        )

    if caption is None:
        caption = ""
    else:
        caption = f"""## Scene description

The robot is operating in the following environment. {caption}

"""

    break_line = ""  # for line formatting

    return f"""# Annotate the training trajectory with reasoning

## Specification of the experimental setup

You're an expert reinforcement learning researcher. You've trained an optimal policy for controlling a robotic arm. The
robot successfully completed a task specified by the instruction: "{language_instruction}". For that purpose, the
robotic arm executed a sequence of actions. Consecutive moves that were executed are the following:


```python
trajectory_features = {structured_features}
```

{features_desc}

{caption}## Your objective

I want you to annotate the given trajectory with reasoning. That is, for each step, I need to know not only {
break_line}which action should be chosen, but importantly what reasoning justifies that action choice. I want you to {
break_line}be descriptive and include all the relevant information available. The reasoning should include the task {
break_line}to complete, the remaining high-level steps, the high-level movements that should be executed and why they {
break_line}are required, the premises that allow inferring the direction of each move, including the locations of {
break_line}relevant objects, possible obstacles or difficulties to avoid, and any other relevant justification.

### Begin by describing the task

Start by giving an overview of the task. Make it more comprehensive than the simple instruction. Include the activity, {
break_line}the objects the robotic arm interacts with, and their relative locations in the environment. Then, describe {
break_line}the high-level movements that were most likely executed, based on the task that was completed and the {
break_line}primitive movements that were executed. Then, for each high-level movement write the interval of steps that {
break_line}movement consists of. Also, for each high-level movement write a justification for why it should be {
break_line}executed. Write an answer for this part using markdown and natural language. Be descriptive and highlight {
break_line}all the relevant details, but ensure that your description is consistent with the trajectory that was {
break_line}executed, specified by the features listed above in the `trajectory_features` dictionary.

### List the reasonings for each step

Finally, for each step describe the reasoning that allows to determine the correct action. For each step describe the {
break_line}remaining part of the objective, the current progress, the objects that are still relevant for determining {
break_line}the plan, and the plan for the next steps, based on the available features. Start the reasoning from a high {
break_line}level and gradually add finer features. I need you to be descriptive and very precise. Ensure that the {
break_line}reasoning is consistent with the task and the executed trajectory. Write the answer for this part as a {
break_line}Python-executable dictionary. For every step in the initial trajectory there should be exactly one separate {
break_line}item of the form <step id>:<reasoning>. Do not group the answers. The final dictionary should have exactly {
break_line}the same set of integer keys as the dictionary of features provided in the `trajectory_features` dictionary {
break_line}above. The reasoning should be a single string that describes the reasoning in natural language and {
break_line}includes all the required features.

Each reasoning string should have the following form:
- Describe the full task that remains to be completed (but only describe what remains), and place it inside a {
break_line}tag <task>.
- Describe the complete high-level plan for completing the remaining task (the list of remaining high-level steps), {
break_line}and place it inside a tag <plan>.
- Describe the high-level step that should be executed now (chosen from the list of high-level steps), and place it {
break_line}inside a tag <subtask>.
- Describe why the chosen high-level step should be executed now, which features of the current environment influence {
break_line}that decision, and how it should be done. Place it within a tag <subtask_reason>.
- Describe the current primitive movement of the arm that needs to be executed, and place it inside a tag <move>.
- Describe why the chosen movement should be executed now and which features of the current environment influence that {
break_line}decision. Place it inside a tag <move_reason>.

## Task summary

Here is a breakdown of what needs to be done:

- Describe the task.
- Describe the high-level movements that were executed, based on the completed task and the listed features.
- Describe the plan for the solution that allowed the robot to complete the task successfully.
- For each step on the trajectory, describe the reasoning that leads to determining the correct action. The reasoning {
break_line}should be descriptive and precise. You should provide exactly one reasoning string for each step on the {
break_line}trajectory specified by `trajectory_features`.
- At the very end of the response, write a single label FINISHED to indicate that the answer is complete."""


def find_task_occurrences(input_string, tags):
    pattern = r"(\d+):"
    for tag in tags:
        pattern = pattern + r"\s*<" + tag + r">([^<]*)<\/" + tag + ">"

    matches = re.findall(pattern, input_string)
    return matches


def extract_reasoning_dict(reasoning_output, tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason")):
    if reasoning_output is None:
        return dict()

    trajectory = dict()

    matches = find_task_occurrences(reasoning_output, tags)

    for match in matches:
        trajectory[int(match[0])] = dict(zip(tags, match[1:]))

    return trajectory


def get_reasoning_dict(features, metadata, lm):
    language_instruction = metadata["language_instruction"]
    caption = metadata["caption"] if "caption" in metadata.keys() else None

    prompt = build_prompt(features, language_instruction, caption=caption, list_only_moves=True)
    print("metadata:", metadata, "\nprompt:", prompt)

    if args.vlm == "prismatic":
        pass
    else:
        reasoning_output = vlm_inference(lm[0], lm[1], lm[2], prompt, image=None, max_new_tokens=256, temperature=0.2)
        # reasoning_output = generate_with_finished_check(
        #     lm[0], 
        #     lm[1],
        #     lm[2],
        #     prompt, 
        #     max_attempts=8, 
        #     max_new_tokens=512, 
        #     temperature=0.2
        # )

    print("reasoning:", reasoning_output)

    return extract_reasoning_dict(reasoning_output)


def build_single_reasoning(episode_id, builder, lm, captions):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))

    ft = dict()

    ft["state_3d"] = [list(step["observation"]["state"][:3].numpy()) for step in episode["steps"]]

    move_primitives = get_move_primitives_episode(episode)
    ft["move_primitive"] = [move[0] for move in move_primitives]

    mt = {
        "episode_id": str(episode_id),
        "file_path": str(episode["episode_metadata"]["file_path"].numpy())[2:-1],
        "n_steps": len(episode["steps"]),
        "language_instruction": str(next(iter(episode["steps"]))["language_instruction"].numpy().decode()),
    }

    mt["caption"] = captions[mt["file_path"]][mt["episode_id"]]["caption"]

    reasoning = get_reasoning_dict(ft, mt, lm)
    entry = {"reasoning": reasoning, "features": ft, "metadata": mt}

    return entry


def generate_reasonings(builder, episode_ids, captions_path=None, save_path=None):
    reasonings = dict()

    # Initialize VLM
    if args.vlm == "prismatic":
        # Load Prismatic VLM
        vlm_model_id = "prism-dinosiglip+7b"
        print(f"Loading Prismatic VLM ({vlm_model_id})...")
        vlm = load(vlm_model_id, hf_token=hf_token)
        vlm = vlm.to(device, dtype=torch.bfloat16)
    else:
        # model, tokenizer, processor = vlm_init(model_name="meta-llama/Llama-3.1-8B-Instruct")
        model, tokenizer, processor = vlm_init()
        vlm = [model, tokenizer, processor]

    if os.path.exists(save_path):
        print(save_path, "existing, loading contents")
        with open(save_path, "r") as f:
            reasonings = json.load(f)

        print("loaded reasonings:", sum([len(v) for v in reasonings.values()]), "entries")

    with open(captions_path, "r") as captions_file:
        captions_dict = json.load(captions_file)

    for i in episode_ids:
        entry = build_single_reasoning(i, builder, vlm, captions_dict)

        if entry["metadata"]["file_path"] in reasonings.keys():
            reasonings[entry["metadata"]["file_path"]][entry["metadata"]["episode_id"]] = entry
        else:
            reasonings[entry["metadata"]["file_path"]] = {entry["metadata"]["episode_id"]: entry}

        print("computed reasoning:", entry)

        with open(save_path, "w") as out_f:
            json.dump(reasonings, out_f)



parser = argparse.ArgumentParser()

parser.add_argument("--id", default=0, type=int)
parser.add_argument("--gpu",  default=0, type=int)
parser.add_argument("--results-path", default="./vlm_response/full_reasonings/libero")
parser.add_argument("--data-path", default="./LIBERO/libero/libero/modified_libero_rlds")
parser.add_argument("--task-suite", default="libero_spatial_no_noops", type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--vlm", default="mllama", type=str)  # [prismatic, mllama]
args = parser.parse_args()

device = f"cuda:{args.gpu}"
# Load HUGGINGFACE_TOKEN from a .env file at the repo root so the script
# keeps working when invoked standalone (no shell-level env export needed).
from dotenv import load_dotenv
load_dotenv()
hf_token = os.environ["HUGGINGFACE_TOKEN"]


# Set random seed
set_seed_everywhere(args.seed)
warnings.filterwarnings("ignore")

# Create results directory if it doesn't exist
os.makedirs(args.results_path, exist_ok=True)
captions_json_path = os.path.join("./vlm_response/scene_descriptions/libero", f"captions_{args.task_suite}.json")
reasonings_json_path = os.path.join(args.results_path, f"reasonings_{args.task_suite}.json")


# watch -n 1 nvidia-smi
# conda activate /hdd2/chenyang/openvla-oft/env_2
# CUDA_VISIBLE_DEVICES="0,1,2" python ecot_scripts/generate_embodied_data/full_reasonings.py  --id 0

if __name__ == "__main__":
    # Initialize Libero task suite
    builder = tfds.builder_from_directory(builder_dir=os.path.join(args.data_path, args.task_suite, "1.0.0"))
    split_info = builder.info.splits
    episode_ids = range(split_info["train"].num_examples)

    # NOTE the generator expects the captions.json file to be present in the working directory
    # The captions should be generated using the script in
    # scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py
    generate_reasonings(builder, episode_ids, captions_path=captions_json_path, save_path=reasonings_json_path)
