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



def propose_subtask_coarse_prompt(language_instruction, caption):
    contents = f"""
    ### Input
    1. Task: {language_instruction}

    ### Instruction
    You are given a high-level robotic task description. Your job is to decompose this task into a minimal set of formal subtasks that a robot must perform to complete the task.

    ### Rules
    1. Minimal Subtask Decomposition
        - Focus on the **necessary and sufficient** actions to complete the task.
        - Do **not** over-decompose — prefer atomic but essential steps.
        - Ask: What does the robot **must** do to succeed, regardless of variations in execution?
    2. Object-Centric Reasoning
        - Use the presence and spatial arrangement of **objects** as key indicators.
        - Mention **relative spatial cues** (e.g., above the drawer handle, to the right of the plate) if implied in the task.
    3. Skillset Usage and Formal Language
        - Each subtask must begin with one of these actions (verbs):
            1) `"Move the gripper ..."`
            2) `"Rotate the gripper ..."`
            3) `"Open the gripper ..."`
            4) `"Close the gripper ..."`
        - Keep language **precise, formal, and robotic**.
        - As a rule of thumb:
            1) **Move/Rotate** is often followed by **Close** to grasp.
            2) Grasped objects are then **moved**, followed by **Open** to release.
    
    ### Examples (Learn from these)
    Task: "put the white mug on the left plate and put the yellow and white mug on the right plate"
    Subtasks: ["Move the gripper above the white mug.", "Close the gripper to grasp the white mug.", "Move the gripper above the left plate while holding the white mug.", "Open the gripper to release the white mug.", "Move the gripper above the yellow and white mug.", "Close the gripper to grasp the yellow and white mug.", "Move the gripper above the right plate while holding the yellow and white mug.", "Open the gripper to release the yellow and white mug."]
    ---
    Task: "open the middle drawer of the cabinet."
    Subtasks: ["Move the gripper toward the handle of the middle drawer of the cabinet.", "Close the gripper to grasp the drawer handle.", "Pull the drawer outward to open it.", "Open the gripper to release the drawer handle."]
    ---
    Task: "turn on the stove and put the moka pot on it."
    subtasks: ["Move the gripper toward the stove knob.", "Close the gripper to grasp the stove knob.", "Rotate the gripper to turn on the stove.", "Open the gripper to release the stove knob.", "Move the gripper above the moka pot.", "Close the gripper to grasp the moka pot.", "Move the gripper above the stove while holding the moka pot.", "Open the gripper to release the moka pot."]
    ---

    Write in the following format. Do not output anything else (Plain Text Only):
    Subtasks: ["<subtask_1>", "<subtask_2>", ...]
    Reasoning: "<Explain how and why you chose each subtask based on the instruction>
    """
    return contents


def propose_subtask_coarse(language_instruction, caption, output_path, existing_response=None, temperature_dict=None, model_dict=None, gpt=True, lm=None):
    user_contents_filled = propose_subtask_coarse_prompt(language_instruction, caption)

    if gpt:
        if existing_response is None:
            system = "You are a helpful assistant."
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / time_string
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/subtask_proposal_coarse.json"

            print("=" * 50)
            print("=" * 20, "Proposing Subtask Coarse", "=" * 20)
            print("=" * 50)
            
            json_data = query(system, [(user_contents_filled, [])], [], save_path, model_dict['subtask_proposal_coarse'], temperature_dict['subtask_proposal_coarse'], debug=False)
    
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
            save_path = str(save_folder) + "/subtask_proposal_coarse.json"

            print("=" * 50)
            print("=" * 20, "Proposing Subtask Coarse", "=" * 20)
            print("=" * 50)

            json_data = vlm_inference_qwen(save_path, lm[0], lm[1], lm[2], user_contents_filled, images=None, max_new_tokens=1024, temperature=temperature_dict['subtask_proposal_coarse'])

        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            loaded_response = json_data["res"]
            print(loaded_response)
            print()


    return user_contents_filled, json_data["res"] 