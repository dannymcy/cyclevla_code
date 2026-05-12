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



def decompose_subtask_prompt(gripper_state, previous_subtasks, states, language_instruction):
    contents = f"""
    You are an expert annotator helping to label robotic manipulation actions.
    A reinforcement learning researcher trained a robotic arm to perform the task: {language_instruction}.
    The robot performs this task in chunks, where each chunk represents a meaningful subtask.

    Input:
    1.  Three consecutive frames showing the robot's motion during the current chunk.
    2.  Subtask history: {previous_subtasks} (if this is the first chunk, this will be None).
    3.  Candidate subtask descriptions: {states}. Note: The candidate list may include a "No meaningful subtask" option. Choose this when there is no clear or obvious motion that contributes to a meaningful subtask — for example, if the gripper barely moves, idles, or performs subtle adjustments without clear task-related intention.

    Instructions:
    1. Select exactly one subtask from the candidate list that best describes the robot's current action.
    2. Only focus on how the gripper moves spatially (ignoring whether it opens or closes):
        - Consider motion such as approaching, placing, or staying still.
        - Do not reason based on gripper opening/closing; another system handles that.
    3. Provide your reasoning in a chain-of-thought style:
        - Step 1: State whether the gripper moves.
        - Step 2: Describe how it moves according (from where to what object?) to the visual cues.
        - Step 3: Match the movement to the most appropriate subtask from the candidate list.
    4.  The correct subtask for the current chunk may sometimes be the same as the previous one. Do not assume that every new chunk must correspond to a different subtask.

    Strict Output Format:
    Reasons_is_move: <yes/no — does the robot gripper move?>
    Reasons_how_move: <describe the gripper's motion and visual evidence — from where to what object?>
    Subtask: <exact chosen subtask>
    """
    return contents


def decompose_subtask_prompt_generalized(gripper_state, previous_subtasks, states, language_instruction):
    contents = f"""
    You are an expert annotator helping to label robotic manipulation actions.
    A reinforcement learning researcher trained a robotic arm to perform the task: {language_instruction}.
    The robot performs this task in chunks, where each chunk represents a meaningful subtask.

    Input:
    1.  Three consecutive frames showing the robot's motion during the current chunk.
    2.  Subtask history: {previous_subtasks} (if this is the first chunk, this will be None).

    Instructions:
    1. Describe the subtask that best describes the robot's current action.
    2. Only focus on how the gripper moves spatially (ignoring whether it opens or closes):
        - Consider motion such as approaching, placing, or staying still.
        - Do not reason based on gripper opening/closing; another system handles that.
    3. Provide your reasoning in a chain-of-thought style:
        - Step 1: State whether the gripper moves.
        - Step 2: Describe how it moves according (from where to what object?) to the visual cues.
        - Step 3: Describe the most appropriate subtask.
    4.  The correct subtask for the current chunk may sometimes be the same as the previous one if they similar goal or the current task resumes the last subtask. Do not assume that every new chunk must correspond to a different subtask.
    5.  The robot may stay still with no obvious changes. In that case, output the same subtask as the previous one.

    Strict Output Format:
    Reasons_frame: <what is happening in each frame?>
    Reasons_is_move: <yes/no — does the robot gripper move?>
    Reasons_how_move: <describe the gripper's motion and visual evidence — from where to what object?>
    Reasons_is_same: <yes/no — the same as last task?>
    Subtask: <describe subtask>
    """
    return contents


def extend_subtask_prompt(subtasks, language_instruction):
    contents = f"""
    You are an expert annotator helping to label robotic manipulation actions.
    A reinforcement learning researcher trained a robotic arm to perform the task: open the top drawer and put the bowl inside.
    The robot performs this task in chunks, where each chunk represents a meaningful subtask (combining motion and/or gripper state).

    Input:
    1.  The list of subtasks executed during the full trajectory, including both gripper state and motion subtasks: [approach the top drawer, open gripper, pull the top drawer open, close gripper, lift and reposition the bowl towards the drawer opening, open gripper]
    2.  Total subtasks: 6

    Task Objective:
    You goal is to refine the subtask list by:
    1. Includes all critical motions (e.g., approach, transport, reposition).
    2. Refining motion subtasks (motion + gripper are always separated):
        - You are allowed to merge missing motion information into motion subtasks ONLY.
        - You CANNOT merge missing motion information into gripper subtasks.
    3. Gripper subtasks can only be refined by specifying their interaction targets (e.g., “open gripper to open gripper to release the object”), but you are NOT allowed to add or merge motion phrases into them.
    4. You must preserve:
        - The same number of subtasks as the original.
        - The same positions and order of the gripper subtasks.
        - No extra subtasks are allowed.

    Two-Pass Chain-of-Thought (CoT) Plan:
    Pass 1: Planning — Insert Missing Motion Subtasks
        1. First, reason step-by-step about whether any motion subtask is missing (such as not approach the target object before placing the picked object in gripper).
        2. You are allowed in this stage to insert missing motion subtasks only, without worrying yet about aligning with the gripper actions or the final structure.
    Pass 2: Merge — Finalize a Compatible Subtask List
        1. When integrating missing motion subtasks, they can only be merged into the motion subtasks directly between two gripper actions, and cannot be merged backward or forward across gripper actions.
        2. In other words, you can only enrich a motion subtask with missing information if it is located between two gripper actions that it logically relates to.
        3. The task sequence should be logical and reasonable after merging.
        4. After merging, you must produce a subtask list where:
            - The number of subtasks matches the original input.
            - The gripper actions (open/close) remain exactly in their original positions, only with target refinement.
            - Motion and gripper actions stay separate.

    Strict Output Format:
    Pass_1_Reasoning: <Describe your thinking and the draft insertion (before merging)>
    Pass_2_Merging::  <Describe your thinking about merging>
    Final_Subtasks: [<subtask_1>, <subtask_2>, ..., <subtask_N>]
    """
    return contents


def decompose_subtask(gripper_state, previous_subtasks, states, language_instruction, video_dir, chunk, output_path, existing_response=None, temperature_dict=None, model_dict=None, num_frames=3, gpt=True, lm=None):
    user_contents_filled = decompose_subtask_prompt(gripper_state, previous_subtasks, states, language_instruction)

    if gpt:
        if existing_response is None:
            system = "You are a helpful assistant."
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / (time_string + "_" + chunk)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/subtask_decomposition.json"

            encoded_img_list = []
            all_files = os.listdir(video_dir)
            image_paths = [os.path.join(video_dir, f) for f in all_files if f.endswith(('.jpg', '.png'))]
            image_paths = select_uniform_frames(image_paths, num_frames=num_frames)

            print()
            print(888, image_paths)
            print()

            for img_path in image_paths:
                img_vis = cv2.imread(img_path)
                # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)  # Uncomment if needed
                encoded_img = encode_image(img_vis)
                encoded_img_list.append(encoded_img)

            print("=" * 50)
            print("=" * 20, "Decomposing Subtask", "=" * 20)
            print("=" * 50)
            
            json_data = query(system, [(user_contents_filled, encoded_img_list)], [], save_path, model_dict['subtask_decomposition'], temperature_dict['subtask_decomposition'], debug=False)
    
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
            save_folder = output_path / (time_string + "_" + chunk)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/subtask_decomposition.json"

            all_files = os.listdir(video_dir)
            image_paths = [os.path.join(video_dir, f) for f in all_files if f.endswith(('.jpg', '.png'))]
            image_paths = select_uniform_frames(image_paths, num_frames=num_frames)
            images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

            print("=" * 50)
            print("=" * 20, "Decomposing Subtask", "=" * 20)
            print("=" * 50)

            json_data = vlm_inference_qwen(save_path, lm[0], lm[1], lm[2], user_contents_filled, images=images, max_new_tokens=256, temperature=temperature_dict['subtask_decomposition'], image_size=(256, 256))

        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            loaded_response = json_data["res"]
            print(loaded_response)
            print()


    return user_contents_filled, json_data["res"] 