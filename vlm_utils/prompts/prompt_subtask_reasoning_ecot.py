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



def reason_subtask_ecot_prompt(features, language_instruction, caption, list_only_moves):
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

Give an overview of the task. Make it more comprehensive than the simple instruction. Include the activity, {
break_line}the objects the robotic arm interacts with, and their relative locations in the environment. Then, describe {
break_line}the high-level movements that were most likely executed, based on the task that was completed and the {
break_line}primitive movements that were executed. Then, for each high-level movement write the interval of steps that {
break_line}movement consists of. Also, for each high-level movement write a justification for why it should be {
break_line}executed. Write an answer for this part using markdown and natural language. Be descriptive and highlight {
break_line}all the relevant details, but ensure that your description is consistent with the trajectory that was {
break_line}executed, specified by the features listed above in the `trajectory_features` dictionary.

## Task summary

Here is a breakdown of what needs to be done:

- Describe the task.
- Describe the high-level movements that were executed, based on the completed task and the listed features.
- Describe the plan for the solution that allowed the robot to complete the task successfully.
- At the very end of the response, write a single label FINISHED to indicate that the answer is complete."""


def reason_subtask_ecot(features, language_instruction, caption, list_only_moves, output_path, existing_response=None, temperature_dict=None, model_dict=None, gpt=True, lm=None):
    user_contents_filled = reason_subtask_ecot_prompt(features, language_instruction, caption, list_only_moves)

    if gpt:
        if existing_response is None:
            system = "You are a helpful assistant."
            ts = time.time()
            time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
            save_folder = output_path / time_string
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = str(save_folder) + "/subtask_reasoning_ecot.json"

            print("=" * 50)
            print("=" * 20, "Reasoning Subtask ECoT", "=" * 20)
            print("=" * 50)
            
            json_data = query(system, [(user_contents_filled, [])], [], save_path, model_dict['subtask_reasoning_ecot'], temperature_dict['subtask_reasoning_ecot'], debug=False)
    
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
            save_path = str(save_folder) + "/subtask_reasoning_ecot.json"

            print("=" * 50)
            print("=" * 20, "Reasoning Subtask ECoT", "=" * 20)
            print("=" * 50)

            json_data = vlm_inference_qwen(save_path, lm[0], lm[1], lm[2], user_contents_filled, images=None, max_new_tokens=1024, temperature=temperature_dict['subtask_reasoning_ecot'])

        else:
            with open(existing_response, 'r') as f:
                json_data = json.load(f)
            loaded_response = json_data["res"]
            print(loaded_response)
            print()


    return user_contents_filled, json_data["res"] 