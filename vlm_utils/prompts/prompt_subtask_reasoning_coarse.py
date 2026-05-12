import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
import sys
from PIL import Image
sys.path.append("../..")
sys.path.append(os.getcwd())
from vlm_utils.query import *
from vlm_utils.utils import *



def reason_subtask_coarse_prompt(features, language_instruction, subtasks, caption, list_only_moves, max_step=100):
    structured_features = "{\n"
    keys = list(features.keys())
    max_len = len(features[keys[0]])

    if max_len > max_step:  # create a window so the length is not too long for LLM
        step = max(1, round(max_len / max_step))
        indices = list(range(0, max_len, step))
    else:
        indices = list(range(max_len))

    for i in indices:
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
            "Each entry in that dictionary corresponds to a single step on the trajectory and describes the move that is about to be executed."
        )
    else:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on the trajectory. The provided features are the following:\n\n"
            "- 'state_3d' are the current 3D coordinates of the robotic arm end effector; moving forward increases the first coordinate; moving left increases the second coordinate; moving up increases the third coordinate.\n"
            "- 'move_primitive' describes the move that is about to be executed.\n"
            "- 'gripper_position' denotes the location of the gripper in the 256x256 image observation."
        )

    contents = f"""
    You are an expert reinforcement learning researcher. You have trained an optimal policy to control a robotic arm, which successfully completed a task as specified in natural language. The robot executed a sequence of actions to complete this task. Each action is recorded as a step in a trajectory.

    ### Input
    1. Task: {language_instruction}
    2. Subtasks list: {subtasks}
    3. Trajectory features: {features_desc}
    ```python
    trajectory_features = {structured_features}
    ```
    4. Scene description: {caption}

    ### Instruction
    Decompose the entire trajectory into subtasks and assign a start and end step index to each subtask. The goal is to produce a mapping of the form:
    Labeled_dict = {{"subtask_1": [start_idx, end_idx], "subtask_2": [start_idx, end_idx], ...}}

    ### Rules
    1. Noisy labels. The trajectory data contains noise. Apply the following rules carefully:
        - **"stop" labels often appear even when the robot is still moving.**
        - Treat "stop" as movement if it appears between meaningful motion steps.
        - Do NOT segment a new subtask just because you see a "stop".
        - "stop" is noisy and should be grouped with adjacent movement, not isolated.
        - "open/close gripper" labels that persist for very short durations may be noise. However, gripper actions are generally more reliable than motion primitives.
    2. Robust Labeling Strategies
        - Brief, short, inconsistent movement descriptions that conflict with prior and subsequent step are likely to be noise.
        - Cross-reference other movement labels to decide whether a short-duration label is meaningful or just noise.
    3. Exhaustive Coverage
        - You must not skip any steps. Every step in the trajectory should be assigned to exactly one subtask.
        - There should be **no gaps** in the index ranges. Subtasks must be labeled sequentially and cover the **entire range** of the trajectory indices. 
        - Step indices should be contiguous and cover the entire trajectory from **0 to {max_len - 1}**.
        - There should be **{len(subtasks)} subtasks** as given in the subtasks list.

    Write in the following format. Do not output anything else (Plain Text Only):
    Labeled_dict: {{"subtask_1": [start_idx, end_idx], "subtask_2": [start_idx, end_idx], ...}}
    Reasoning: "<Explain your logic for identifying and segmenting each subtask>"
    """
    return contents


def reason_subtask_coarse(features, language_instruction, subtasks, caption, list_only_moves, output_path, existing_response=None, temperature_dict=None, model_dict=None, gpt=True, lm=None):
    user_contents_filled = reason_subtask_coarse_prompt(features, language_instruction, subtasks, caption, list_only_moves)

    if gpt:
        if existing_response is None:
            system = "You are a helpful assistant."
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / time_string
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/subtask_reasoning_coarse.json"

            print("=" * 50)
            print("=" * 20, "Reasoning Subtask Coarse", "=" * 20)
            print("=" * 50)
            
            json_data = query(system, [(user_contents_filled, [])], [], save_path, model_dict['subtask_reasoning_coarse'], temperature_dict['subtask_reasoning_coarse'], debug=False)
    
        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            loaded_response = json_data["res"]
            print(loaded_response)
            print()
    

    else:
        if existing_response is None:
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / time_string
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/subtask_reasoning_coarse.json"

            print("=" * 50)
            print("=" * 20, "Reasoning Subtask Coarse", "=" * 20)
            print("=" * 50)

            json_data = vlm_inference_qwen(save_path, lm[0], lm[1], lm[2], user_contents_filled, images=None, max_new_tokens=1024, temperature=temperature_dict['subtask_reasoning_coarse'])

        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            loaded_response = json_data["res"]
            print(loaded_response)
            print()


    return user_contents_filled, json_data["res"] 