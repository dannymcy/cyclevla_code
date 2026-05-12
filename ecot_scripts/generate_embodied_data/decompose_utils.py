import json
import os
import re
import sys
import shutil
import pathlib
import time
sys.path.append("../..")
sys.path.append(os.getcwd())
from fsm_utils.build import *
from fsm_utils.utils import *
from vlm_utils.utils import *
# from vlm_utils.prompts.prompt_subtask_decomposition import decompose_subtask
# from vlm_utils.prompts.prompt_subtask_reasoning_ecot import reason_subtask_ecot
from vlm_utils.prompts.prompt_subtask_reasoning_coarse import reason_subtask_coarse
from vlm_utils.prompts.prompt_subtask_proposal_coarse import propose_subtask_coarse


# Communicating to ChatGPT API
temperature_dict = {
    # "subtask_decomposition": 0.2,
    # "subtask_reasoning_ecot": 0.2,
    "subtask_reasoning_coarse": 0.2,
    "subtask_proposal_coarse": 0.2,
}
# Pricing list (https://platform.openai.com/docs/pricing)
model_dict = {
    # "subtask_decomposition": "gpt-4.5-preview",
    # "subtask_reasoning_ecot": "gpt-4o",
    "subtask_reasoning_coarse": "gpt-4.1",
    "subtask_proposal_coarse": "gpt-4.1",
}


def pick_place_is_perfect(num_chunks, chunk_values, task_suite):
    if num_chunks == 4 and chunk_values == [0, -1, 0, 1]:
        return 4
    elif num_chunks == 3 and chunk_values == [0, -1, 0]:
        return 3

    if task_suite == "libero_10_no_noops":
        if num_chunks == 8 and chunk_values == [0, -1, 0, 1, 0, -1, 0, 1]:
            return 8
        elif num_chunks == 7 and chunk_values == [0, -1, 0, 1, 0, -1, 0]:
            return 7

    return False


def pick_place_to_discard(num_chunks, chunk_values, task_suite):
    if task_suite in ["libero_spatial_no_noops", "libero_object_no_noops"]:
        threshold = 7
    elif task_suite in ["libero_goal_no_noops", "libero_10_no_noops"]:
        threshold = 13
    
    any_consecutive = any(chunk_values[i] in [-1, 1] and chunk_values[i+1] in [-1, 1] for i in range(len(chunk_values)-1))
    if num_chunks >= threshold or any_consecutive:
        return True
    return False


def pick_place_to_truncate(num_chunks, chunk_values, task_suite):
    if num_chunks == 5 and chunk_values == [0, -1, 0, 1, 0]:
        return True
    
    if task_suite == "libero_10_no_noops":
        if num_chunks == 9 and chunk_values == [0, -1, 0, 1, 0, -1, 0, 1, 0]:
            return True
    
    return False


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


def has_consecutive_moves(moves, target_move, states, num_consecutive=3, tail_clip=10):
    """
    Check if there are num_consecutive moves containing the target_move substring.

    Args:
        moves (list of str): List of move descriptions.
        target_move (str): The substring to search for (e.g., "open gripper").
        num_consecutive (int): Number of consecutive occurrences required.

    Returns:
        bool: True if there is at least one occurrence of num_consecutive moves containing the target substring.
    """
    # Only check the last `tail_clip` moves for two pick and place tasks in libero_10_no_noops
    if len(states) == 8 and len(moves) > tail_clip:
        moves = moves[-tail_clip:]
    
    for i in range(len(moves) - num_consecutive + 1):
        if all(target_move in move for move in moves[i:i + num_consecutive]):
            return True
    return False


def pair_traj_to_states(episode_data, states, is_perfect):
    states = states[0:-1] if is_perfect == 3 or is_perfect == 7 else states
    for i in range(len(states)):
        episode_data["chunks"][i]["subtask"] = states[i]
    return episode_data


def pair_traj_to_states_llm_coarse(episode_data, states, labeled_dict, move_primitive):
    def correct_labeled_dict_indices_extend_end(labeled_dict: dict) -> dict:
        """
        Corrects the indices in the labeled_dict by extending the end indices 
        to ensure continuity without gaps.

        Args:
            labeled_dict (dict): Original dict with possibly disconnected indices.

        Returns:
            dict: Corrected labeled_dict with continuous indices.
        """
        subtasks = list(labeled_dict.keys())
        corrected_dict = {}
        
        for i, subtask in enumerate(subtasks):
            start_idx, end_idx = labeled_dict[subtask]

            if i < len(subtasks) - 1:
                # Extend current subtask end_idx to directly precede the next subtask start_idx
                next_start_idx = labeled_dict[subtasks[i+1]][0]
                end_idx = next_start_idx - 1
            corrected_dict[subtask] = [start_idx, end_idx]

        return corrected_dict


    def extend_gripper_indices(labeled_dict, move_primitive, subtask_idxs=[1, 3]):
        """
        Extend start and end indices of specified subtasks to fully include continuous gripper actions,
        and adjust other subtasks accordingly to avoid overlapping or gaps.

        Args:
            labeled_dict (dict): dict with subtasks indices
            move_primitive (list): list of move primitives corresponding to trajectory steps
            subtask_idxs (list): indexes of subtasks (0-based) to adjust for gripper movements
        
        Returns:
            dict: Adjusted labeled_dict
        """
        subtasks = list(labeled_dict.keys())

        for idx in subtask_idxs:
            if idx >= len(subtasks):
                continue  # Skip if the idx is beyond the available subtasks

            subtask = subtasks[idx]
            start_idx, end_idx = labeled_dict[subtask]

            # Extend backwards to include continuous gripper action
            while start_idx > 0 and ("gripper" in move_primitive[start_idx - 1]):
                start_idx -= 1

            # Extend forwards to include continuous gripper action
            while end_idx < len(move_primitive) - 1 and ("gripper" in move_primitive[end_idx + 1]):
                end_idx += 1

            # Update current subtask indices
            labeled_dict[subtask] = [start_idx, end_idx]

            # Adjust previous subtask end_idx to avoid overlap
            if idx > 0:
                prev_subtask = subtasks[idx - 1]
                labeled_dict[prev_subtask][1] = start_idx - 1

            # Adjust next subtask start_idx to avoid overlap
            if idx < len(subtasks) - 1:
                next_subtask = subtasks[idx + 1]
                labeled_dict[next_subtask][0] = end_idx + 1

        return labeled_dict


    # Step 1: Check for mismatch
    assert len(states) == len(labeled_dict), "Mismatch: number of states and labeled subtasks must match."

    # Step 2: Sort labeled dict
    sorted_subtasks = sorted(labeled_dict.items(), key=lambda x: int(x[0].split('_')[1]))
    sorted_label_dict = {subtask_key: (start_idx, end_idx) for subtask_key, (start_idx, end_idx) in sorted_subtasks}

    # Step 3: Corrects the indices in the labeled_dict by extending the end indices to ensure continuity without gaps.
    sorted_label_dict = correct_labeled_dict_indices_extend_end(sorted_label_dict)

    # Step 4: Extend indices of open/close gripper subtasks to fully include gripper action
    sorted_label_dict = extend_gripper_indices(sorted_label_dict, move_primitive, subtask_idxs=[1, 3])

    new_chunks = []
    for i, (subtask_key, (start_idx, end_idx)) in enumerate(sorted_label_dict.items()):
        new_chunks.append({
            "start": start_idx,
            "end": end_idx,
            "subtask": states[i]
        })
    episode_data["chunks"] = new_chunks

    return episode_data


# def subtask_decomposition_vlm(data_path, chunk_tuple, gripper_state, previous_subtasks, states, language_instruction, video_dir, temperature_dict, model_dict, lm=None, gpt=True, start_over=False):
#     file_idx, chunk = chunk_tuple
#     if gpt:
#         output_dir = pathlib.Path(data_path) / "gpt_response" / "subtask_decomposition"
#     else:
#         output_dir = pathlib.Path(data_path) / "qwen_response" / "subtask_decomposition"

#     os.makedirs(output_dir, exist_ok=True)
#     conversation_hist = []

#     if start_over:
#         user, res = decompose_subtask(gripper_state, previous_subtasks, states, language_instruction, video_dir, chunk, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, num_frames=2, gpt=gpt, lm=lm)
#         if gpt: time.sleep(5)
#     else:
#         user, res = decompose_subtask(gripper_state, previous_subtasks, states, language_instruction, video_dir, chunk, output_dir, existing_response=load_response("subtask_decomposition", output_dir, file_idx=file_idx), temperature_dict=temperature_dict, model_dict=model_dict, num_frames=2, gpt=gpt, lm=lm)
#     conversation_hist.append([user, res])

#     return conversation_hist


# def subtask_reasoning_ecot_llm(data_path, features, language_instruction, caption, list_only_moves, temperature_dict, model_dict, lm=None, gpt=True, start_over=False):
#     if gpt:
#         output_dir = pathlib.Path(data_path) / "gpt_response" / "subtask_reasoning_ecot"
#     else:
#         output_dir = pathlib.Path(data_path) / "qwen_response" / "subtask_reasoning_ecot"

#     os.makedirs(output_dir, exist_ok=True)
#     conversation_hist = []

#     if start_over:
#         user, res = reason_subtask_ecot(features, language_instruction, caption, list_only_moves, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, gpt=gpt, lm=lm)
#         if gpt: time.sleep(5)
#     else:
#         user, res = reason_subtask_ecot(features, language_instruction, caption, list_only_moves, output_dir, existing_response=load_response("subtask_reasoning_ecot", output_dir), temperature_dict=temperature_dict, model_dict=model_dict, gpt=gpt, lm=lm)
#     conversation_hist.append([user, res])

#     return conversation_hist


def subtask_reasoning_coarse_llm(data_path, features, language_instruction, subtasks, caption, list_only_moves, temperature_dict, model_dict, lm=None, gpt=True, start_over=False):
    if gpt:
        output_dir = pathlib.Path(data_path) / "gpt_response" / "subtask_reasoning_coarse"
    else:
        output_dir = pathlib.Path(data_path) / "qwen_response" / "subtask_reasoning_coarse"

    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []

    if start_over:
        user, res = reason_subtask_coarse(features, language_instruction, subtasks, caption, list_only_moves, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, gpt=gpt, lm=lm)
        if gpt: time.sleep(5)
    else:
        user, res = reason_subtask_coarse(features, language_instruction, subtasks, caption, list_only_moves, output_dir, existing_response=load_response("subtask_reasoning_coarse", output_dir), temperature_dict=temperature_dict, model_dict=model_dict, gpt=gpt, lm=lm)
    conversation_hist.append([user, res])

    return conversation_hist


def subtask_proposal_coarse_llm(data_path, language_instruction, caption, temperature_dict, model_dict, lm=None, gpt=True, start_over=False):
    if gpt:
        output_dir = pathlib.Path(data_path) / "gpt_response" / "subtask_proposal_coarse"
    else:
        output_dir = pathlib.Path(data_path) / "qwen_response" / "subtask_proposal_coarse"

    os.makedirs(output_dir, exist_ok=True)
    conversation_hist = []

    if start_over:
        user, res = propose_subtask_coarse(language_instruction, caption, output_dir, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, gpt=gpt, lm=lm)
        if gpt: time.sleep(5)
    else:
        user, res = propose_subtask_coarse(language_instruction, caption, output_dir, existing_response=load_response("subtask_proposal_coarse", output_dir), temperature_dict=temperature_dict, model_dict=model_dict, gpt=gpt, lm=lm)
    conversation_hist.append([user, res])

    return conversation_hist


def process_libero_coarse(decompose_path, results_path, episode_id, episodes_summary, episode_data, task_suite, lm=None):
    source_folder = os.path.join(decompose_path, f"episode_{episode_id}")
    destination_folder = os.path.join(results_path, f"episode_{episode_id}")

    num_chunks = episodes_summary["episodes"][episode_id]["num_chunks"]
    chunk_values = [chunk["value"] for chunk in episode_data["chunks"]]
    
    features = episode_data["features"]
    caption = episode_data["metadata"]["caption"]
    language_instruction = episode_data["metadata"]["language_instruction"]
    states = pick_place_states(language_instruction, task_suite)
    to_propose = states is None
    states = complex_states(language_instruction, task_suite) if states is None else states
    
    is_perfect = pick_place_is_perfect(num_chunks, chunk_values, task_suite)
    to_truncate = pick_place_to_truncate(num_chunks, chunk_values, task_suite)
    to_discard = pick_place_to_discard(num_chunks, chunk_values, task_suite)


    if to_discard:
        return None


    elif is_perfect and not to_propose:
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
        shutil.copytree(source_folder, destination_folder)

        episode_data = pair_traj_to_states(episode_data, states, is_perfect)
        with open(os.path.join(destination_folder, "chunks_summary.json"), "w") as f:
            json.dump(episode_data, f, indent=2)


    elif to_truncate and not to_propose:
        # 1) Remove last segment comprising all 0s in the episode data
        # First identify which chunk to remove (the last one with all 0s)
        if episode_data["chunks"][-1]["value"] == 0:
            # Get the start frame of the last chunk
            last_chunk_start = episode_data["chunks"][-1]["start"]
            # Update total steps
            episode_data["metadata"]["n_steps"] = last_chunk_start
            # Remove the last chunk
            episode_data["chunks"].pop()
        
        # 2) Copy the folder structure but remove the last images
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)

        # Create destination directory
        os.makedirs(destination_folder, exist_ok=True)
        
        # First copy all files at the root level (including chunks_summary.json which we'll overwrite)
        for item in os.listdir(source_folder):
            source_item = os.path.join(source_folder, item)
            dest_item = os.path.join(destination_folder, item)
            
            if not os.path.isdir(source_item):
                shutil.copy2(source_item, dest_item)
        
        # Copy all chunk folders except the last one (which we removed from the data)
        chunk_folders = [d for d in os.listdir(source_folder) if d.startswith("chunk_")]
        total_chunks = len(episode_data["chunks"])
        
        for folder in chunk_folders:
            # Extract chunk index from folder name (chunk_X_value)
            try:
                chunk_idx = int(folder.split("_")[1])
                if chunk_idx < total_chunks:  # Skip the last chunk folder
                    source_chunk = os.path.join(source_folder, folder)
                    dest_chunk = os.path.join(destination_folder, folder)
                    shutil.copytree(source_chunk, dest_chunk)
            except (IndexError, ValueError):
                # If folder name doesn't match expected format, copy it anyway
                source_chunk = os.path.join(source_folder, folder)
                dest_chunk = os.path.join(destination_folder, folder)
                shutil.copytree(source_chunk, dest_chunk)
        
        # 3) Save the updated chunk_summary.json
        episode_data = pair_traj_to_states(episode_data, states, 4)
        with open(os.path.join(destination_folder, "chunks_summary.json"), "w") as f:
            json.dump(episode_data, f, indent=2)
        

    else: 
        # 1) If simple pick place, checks for the presence of at least one sequence consecutive moves containing "open gripper". If not, it means the gripper is not opened after placing the object
        destination_folder = os.path.join(results_path, f"episode_{episode_id}_llm")
        if not to_propose and not has_consecutive_moves(features["move_primitive"], "open gripper", states, num_consecutive=3, tail_clip=10):
            states = states[0:-1]

        # 2) LLM decomposes the subtasks
        conversation_hist = subtask_reasoning_coarse_llm(destination_folder, features, language_instruction, states, caption, True, temperature_dict, model_dict, lm=lm, gpt=lm==None, start_over=False)
        labeled_dict = extract_labeled_dict_coarse(conversation_hist[-1][1])
        episode_data = pair_traj_to_states_llm_coarse(episode_data, states, labeled_dict, features["move_primitive"])

        # 3) Copy the observations according to the LLM response
        for chunk in episode_data["chunks"]:
            start_idx = chunk["start"]
            end_idx = chunk["end"]
            subtask = chunk["subtask"]
            subtask_folder = f"chunk_{episode_data['chunks'].index(chunk)}_" + "_".join(subtask.strip().lower().replace(".", "").split())
            target_dir = os.path.join(destination_folder, subtask_folder)
            os.makedirs(target_dir, exist_ok=True)

            for idx in range(start_idx, end_idx + 1):
                found = False
                for root, dirs, files in os.walk(source_folder):
                    filename = f"step_{idx}.png"
                    if filename in files:
                        src_path = os.path.join(root, filename)
                        dst_path = os.path.join(target_dir, filename)
                        shutil.copy2(src_path, dst_path)
                        found = True
                        break
                if not found:
                    print(f"[WARN] Image for step_{idx}.png not found in any source folder.")

        # 4) Save the updated chunk_summary.json
        with open(os.path.join(destination_folder, "chunks_summary.json"), "w") as f:
            json.dump(episode_data, f, indent=2)
        
    
    return episode_data