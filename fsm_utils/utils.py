import re

def extract_pick_place_libero_spatial(task_description):
    """
    Extract the object to pick up and the location to place it from a task description.
    
    Args:
        task_description (str): Task description string
        
    Returns:
        tuple: (pick_object_full, pick_object_simple, place_object)
    """
    # Basic pattern check
    if not task_description.startswith("pick up the ") or " and place it on the " not in task_description:
        return None, None, None
    
    # Split the task description
    parts = task_description.split(" and place it on the ")
    if len(parts) != 2:
        return None, None, None
    
    # Extract the pick object (everything after "pick up the")
    pick_object_full = parts[0][len("pick up the "):]
    if " from " in pick_object_full:
        pick_object_full = pick_object_full.replace(" from ", " on ")
    
    # Extract just the object name without location descriptors
    location_words = ["between", "from", "in", "next to", "on"]
    pick_object_simple = pick_object_full
    
    for location_word in location_words:
        if f" {location_word} " in pick_object_simple:
            pick_object_simple = pick_object_simple.split(f" {location_word} ")[0].strip()
    
    # Extract the place object (everything in the second part)
    place_object = parts[1]
    
    return pick_object_full, pick_object_simple, place_object


def extract_pick_place_libero_object(task_description):
    """
    Extract the object to pick up and the location to place it from a task description.
    
    Args:
        task_description (str): Task description string
        
    Returns:
        tuple: (pick_object, place_object)
    """
    # Basic pattern check
    if not task_description.startswith("pick up the ") or " and place it in the " not in task_description:
        return None, None
    
    # Split the task description
    parts = task_description.split(" and place it in the ")
    if len(parts) != 2:
        return None, None
    
    # Extract the pick object (everything after "pick up the")
    pick_object = parts[0][len("pick up the "):]
    
    # Extract the place object (everything in the second part)
    place_object = parts[1]
    
    return pick_object, place_object


def extract_pick_place_libero_goal(task_description):
    """
    Extract the object to pick up and the location to place it from a task description.
    
    Args:
        task_description (str): Task description string
        
    Returns:
        tuple: (pick_object, place_object)
    """
    # Basic pattern check
    proposition_list = [
        ("put the ", " on the "),
        ("put the ", " on "),
        ("put the ", " in the "),
    ]

    valid_idx = -1
    for i, proposition in enumerate(proposition_list):
        if task_description.startswith(proposition[0]) and proposition[1] in task_description:
            valid_idx = i
            break

    if valid_idx == -1:
        return None, None
    
    # Split the task description
    parts = task_description.split(proposition_list[valid_idx][1])
    if len(parts) != 2:
        return None, None
    
    # Extract the pick object (everything after "pick up the")
    pick_object = parts[0][len(proposition_list[valid_idx][0]):]
    
    # Extract the place object (everything in the second part)
    place_object = parts[1]
    
    return pick_object, place_object


def extract_pick_place_libero_10(task_description):
    """
    Extract the object to pick up and the location to place it from a task description.
    
    Args:
        task_description (str): Task description string
        
    Returns:
        tuple: ([pick_objects], [place_object])
    """
    if "turn on" in task_description or "microwave" in task_description or "and close it" in task_description:
        return None, None

    # Match 'put both the X and the Y in the Z'
    both_match = re.search(r"put both the (.*?) and the (.*?) in the (.*)", task_description)
    if both_match:
        pick1 = both_match.group(1).strip()
        pick2 = both_match.group(2).strip()
        place = both_match.group(3).strip()
        return ([pick1, pick2], [place])

    # Match 'put both Xs on the Y'
    custom_both_match = re.search(r"put both (.*?)s? on the (.*)", task_description)
    if custom_both_match:
        base_object = custom_both_match.group(1).strip()
        place = custom_both_match.group(2).strip()

        # Create two pseudo-individual objects for robot instructions
        pick1 = f"{base_object} on the right"
        pick2 = f"{base_object} on the left"
        return ([pick1, pick2], [place])

    # Match 'pick up the X and place it in the Y'
    custom_single_match = re.search(r"pick up the (.*?) and place it in the (.*)", task_description)
    if custom_single_match:
        pick = custom_single_match .group(1).strip()
        place = custom_single_match .group(2).strip()
        pick_place = ([pick], [place])
        return pick_place

    # Match: put the X on the Y [and put the Z on the W]...
    single_match_pattern = r"put the (.*?) (?:on|to) the (.*?)(?: and |$)"
    single_matches = re.findall(single_match_pattern, task_description)

    if not single_matches:
        return None, None

    pick_list, place_list = [], []
    for pick, place in single_matches:
        pick_list.append(pick.strip())
        place_list.append(place.strip())

    return pick_list, place_list