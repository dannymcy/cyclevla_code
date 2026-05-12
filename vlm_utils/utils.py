import cv2
import base64
import numpy as np
import io
import os
import json
import re
import ast
from pathlib import Path


def encode_image(input_img):
    # Check if the image is loaded properly
    if input_img is None:
        raise ValueError("The image could not be loaded. Please check the file path.")
    
    # Encode the image as a JPEG (or PNG) to a memory buffer
    img_vis = input_img.copy()
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    success, encoded_image = cv2.imencode('.png', img_vis)
    if not success:
        raise ValueError("Could not encode the image")

    # Convert the encoded image to bytes and then to a base64 string
    image_bytes = io.BytesIO(encoded_image).read()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')

    # return base64_string
    return f'data:image/png;base64, {base64_string}'


def load_response(prompt_name, prompt_path, file_idx=None, get_latest=True):
    if prompt_path.exists():
        subdirs = [d for d in os.listdir(prompt_path) if os.path.isdir(prompt_path / d)]
        subdirs.sort()
        
        if get_latest and file_idx is None:
            # Find the latest subdirectory
            latest_subdir = max(subdirs, key=lambda d: (prompt_path / d).stat().st_mtime)
            json_file_path = prompt_path / latest_subdir / f"{prompt_name}.json"
            if json_file_path.exists():
                return json_file_path
        elif file_idx is not None:
            selected_subdir = subdirs[file_idx]
            json_file_path = prompt_path / selected_subdir / f"{prompt_name}.json"
            if json_file_path.exists():
                return json_file_path
        else:
            # Process all subdirectories
            responses = []
            for subdir in subdirs:
                json_file_path = prompt_path / subdir / f"{prompt_name}.json"
                if json_file_path.exists():
                    responses.append(json_file_path)
            return responses


def select_uniform_frames(image_paths, num_frames=3):
    """
    Select uniformly sampled frames from a list of image paths, sorted numerically by step number.
    """
    def extract_step_number(path):
        match = re.search(r'step_(\d+)', path)
        return int(match.group(1)) if match else -1 

    if len(image_paths) == 0:
        return []

    # Sort by the step number extracted from the file name
    image_paths = sorted(image_paths, key=extract_step_number)

    if len(image_paths) <= num_frames:
        return image_paths

    indices = [round(i * (len(image_paths) - 1) / (num_frames - 1)) for i in range(num_frames)]
    selected_images = [image_paths[i] for i in indices]
    return selected_images


# def extract_decomposed_subtask(response_text):
#     """
#     Extracts 'Subtask' fields from a plain text response.

#     Args:
#         response_text (str): The text containing the response.

#     Returns:
#         tuple: subtask (str)
#     """
#     lines = response_text.splitlines()

#     subtask = None
#     same_as_previous = None

#     for line in lines:
#         if line.startswith("Subtask:"):
#             subtask = line.replace("Subtask:", "").strip()

#     return subtask


# def find_task_occurrences(input_string, tags):
#     pattern = r"(\d+):"
#     for tag in tags:
#         pattern = pattern + r"\s*<" + tag + r">([^<]*)<\/" + tag + ">"

#     matches = re.findall(pattern, input_string)
#     return matches


# def extract_reasoning_dict(reasoning_output, tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason")):
#     if reasoning_output is None:
#         return dict()

#     trajectory = dict()

#     matches = find_task_occurrences(reasoning_output, tags)

#     for match in matches:
#         trajectory[int(match[0])] = dict(zip(tags, match[1:]))

#     return trajectory


def extract_labeled_dict_coarse(llm_output: str) -> dict:
    """
    Extracts the Labeled_dict from LLM output using JSON-safe parsing (no ast).
    
    Args:
        llm_output (str): The full response string from the LLM.
    
    Returns:
        dict: Parsed labeled dict with subtask names and step index ranges.
    
    Raises:
        ValueError: If no valid Labeled_dict is found or parsing fails.
    """
    # Step 1: Find the dict-like substring after 'Labeled_dict:'
    match = re.search(r'Labeled_dict:\s*(\{.*?\})', llm_output, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'Labeled_dict' in the input.")

    dict_str = match.group(1)

    # Step 2: Replace single quotes with double quotes (JSON uses double quotes)
    dict_str_json = dict_str.replace("'", '"')

    try:
        labeled_dict = json.loads(dict_str_json)
        return labeled_dict
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Labeled_dict as JSON: {e}")


def extract_proposed_subtask(response_text):
    """
    Extracts the list of subtasks from a plain text response without using ast.

    Args:
        response_text (str): The full response from the LLM.

    Returns:
        list[str]: A list of extracted subtasks.
    """
    lines = response_text.splitlines()
    subtask_line = None

    for line in lines:
        if line.strip().startswith("Subtasks:"):
            subtask_line = line.strip()[len("Subtasks:"):].strip()
            break

    if not subtask_line:
        return []

    # Strip brackets and split by comma, handle quotes
    subtask_line = subtask_line.strip("[]")
    parts = subtask_line.split(",")
    subtasks = [p.strip().strip('"').strip("'") for p in parts if p.strip()]

    return subtasks