from transitions import Machine
from typing import List
import re


def fsm_pick_place_libero_spatial(pick_object_full: str, pick_object_simple: str, place_object: str):

    states = [f"Move the gripper above the {pick_object_full}.",
              f"Close the gripper to grasp the {pick_object_full}.", 
              f"Move the gripper above the {place_object} while holding the {pick_object_simple}.", 
              f"Open the gripper to release the {pick_object_simple}."]
    
    transitions = [
        # { 'trigger': 'melt', 'source': 'solid', 'dest': 'liquid' },
        # { 'trigger': 'evaporate', 'source': 'liquid', 'dest': 'gas' },
        # { 'trigger': 'sublimate', 'source': 'solid', 'dest': 'gas' },
        # { 'trigger': 'ionize', 'source': 'gas', 'dest': 'plasma' }
    ]

    machine = Machine(states=states, transitions=transitions, initial=states[0])

    return machine, states


def fsm_pick_place_libero_object(pick_object: str, place_object: str):

    states = [f"Move the gripper above the {pick_object}.",
              f"Close the gripper to grasp the {pick_object}.", 
              f"Move the gripper above the {place_object} while holding the {pick_object}.", 
              f"Open the gripper to release the {pick_object}."]

    transitions = [
        # { 'trigger': 'melt', 'source': 'solid', 'dest': 'liquid' },
        # { 'trigger': 'evaporate', 'source': 'liquid', 'dest': 'gas' },
        # { 'trigger': 'sublimate', 'source': 'solid', 'dest': 'gas' },
        # { 'trigger': 'ionize', 'source': 'gas', 'dest': 'plasma' }
    ]

    machine = Machine(states=states, transitions=transitions, initial=states[0])

    return machine, states


def fsm_pick_place_libero_goal(pick_object: str, place_object: str):

    states = [f"Move the gripper above the {pick_object}.",
              f"Close the gripper to grasp the {pick_object}.", 
              f"Move the gripper above the {place_object} while holding the {pick_object}.", 
              f"Open the gripper to release the {pick_object}."]
    
    transitions = [
        # { 'trigger': 'melt', 'source': 'solid', 'dest': 'liquid' },
        # { 'trigger': 'evaporate', 'source': 'liquid', 'dest': 'gas' },
        # { 'trigger': 'sublimate', 'source': 'solid', 'dest': 'gas' },
        # { 'trigger': 'ionize', 'source': 'gas', 'dest': 'plasma' }
    ]

    machine = Machine(states=states, transitions=transitions, initial=states[0])

    return machine, states


def fsm_pick_place_libero_10(pick_object: List[str], place_object: List[str]):
    """
    Build FSM for pick and place tasks where multiple pick/place pairs may be involved.

    - If len(pick_object) == len(place_object), they are mapped one-to-one.
    - If len(place_object) == 1, all picks go to the same place.
    """
    assert len(pick_object) >= len(place_object), \
        "Invalid input: length of pick list must be >= length of place list, or place list should be of length 1."

    states = []

    # One-to-one mapping
    if len(pick_object) == len(place_object):
        for pick, place in zip(pick_object, place_object):
            states.append(f"Move the gripper above the {pick}.")
            states.append(f"Close the gripper to grasp the {pick}.")
            states.append(f"Move the gripper above the {place} while holding the {pick}.")
            states.append(f"Open the gripper to release the {pick}.")

    else:
        # All picks go to a single place
        place = place_object[0]
        for pick in pick_object:
            pick_cleaned = re.sub(r"\s+on the (right|left)", "", pick)
            states.append(f"Move the gripper above the {pick}.")
            states.append(f"Close the gripper to grasp the {pick}.")
            states.append(f"Move the gripper above the {place} while holding the {pick_cleaned}.")
            states.append(f"Open the gripper to release the {pick_cleaned}.")

    transitions = []

    machine = Machine(states=states, transitions=transitions, initial=states[0])

    return machine, states


def fsm_complex_libero(language_instruction: str, task_suite: str):
    libero_goal_dict = {
        "open the middle drawer of the cabinet": [
            "Move the gripper toward the handle of the middle drawer of the cabinet.",
            "Close the gripper to grasp the drawer handle.",
            "Move the gripper to pull the drawer outward."
        ],
        "open the top drawer and put the bowl inside": [
            "Move the gripper above the handle of the top drawer and insert it into the gap.",
            "Move the gripper to pull the drawer outward.",
            "Move the gripper above the bowl.",
            "Close the gripper to grasp the bowl.",
            "Move the gripper above the open top drawer while holding the bowl.",
            "Open the gripper to release the bowl."
        ],
        "push the plate to the front of the stove": [
            "Move the gripper toward the plate and make contact.",
            "Close the gripper to secure attachment to the plate.",
            "Move the gripper forward to push the plate toward the front of the stove while maintaining contact."
        ],
        "turn on the stove": [
            "Move the gripper above the stove knob.",
            "Close the gripper to grasp the stove knob.",
            "Rotate the gripper to turn on the stove."
        ]
    }

    libero_10_dict = {
        "turn on the stove and put the moka pot on it": [
            "Move the gripper above the stove knob.",
            "Close the gripper to grasp the stove knob.",
            "Rotate the gripper to turn on the stove.",
            "Open the gripper to release the stove knob.",
            "Move the gripper above the moka pot.",
            "Close the gripper to grasp the moka pot.",
            "Move the gripper above the stove while holding the moka pot.",
            "Open the gripper to release the moka pot."
        ],
        "put the black bowl in the bottom drawer of the cabinet and close it": [
            "Move the gripper above the black bowl.",
            "Close the gripper to grasp the black bowl.",
            "Move the gripper above the bottom drawer of the cabinet while holding the black bowl.",
            "Open the gripper to release the black bowl.",
            "Move the gripper to close the bottom drawer of the cabinet."
        ],
        "put the yellow and white mug in the microwave and close it": [
            "Move the gripper above the yellow and white mug.",
            "Close the gripper to grasp the yellow and white mug.",
            "Move the gripper inside the microwave while holding the yellow and white mug.",
            "Open the gripper to release the yellow and white mug.",
            "Move the gripper to close the microwave."
        ]
    }

    if task_suite == "libero_goal_no_noops":
        states = libero_goal_dict[language_instruction]
    elif task_suite == "libero_10_no_noops":
        states = libero_10_dict[language_instruction]

    transitions = []

    machine = Machine(states=states, transitions=transitions, initial=states[0])

    return machine, states