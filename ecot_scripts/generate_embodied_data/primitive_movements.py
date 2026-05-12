import numpy as np
import robosuite.utils.transform_utils as T


def describe_move(move_vec, task_suite="bridge"):
    names = [
        {-1: "backward", 0: None, 1: "forward"},
        {-1: "right", 0: None, 1: "left"},
        {-1: "down", 0: None, 1: "up"},
        {-1: "tilt down", 0: None, 1: "tilt up"},
        {},
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},
        {-1: "close gripper", 0: None, 1: "open gripper"},
    ]

    xyz_move = [names[i][move_vec[i]] for i in range(0, 3)]
    xyz_move = [m for m in xyz_move if m is not None]

    if len(xyz_move) != 0:
        description = "move " + " ".join(xyz_move)
    else:
        description = ""

    if move_vec[3] == 0:
        move_vec[3] = move_vec[4]  # identify rolling and pitching

    if move_vec[3] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[3][move_vec[3]]

    if move_vec[5] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[5][move_vec[5]]

    if move_vec[6] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[6][move_vec[6]]

    if len(description) == 0:
        description = "stop"

    return description


def classify_movement(move, task_suite="bridge", thresholds=[0.03, 0.03, 0.03]):
    """
    Classify movement using 3 thresholds for translation, rotation, and gripper respectively.

    Args:
        move: A list of 7D state vectors. Only the first and last are used to compute the diff.
        thresholds: List of 3 values [translation_thresh, rotation_thresh, gripper_thresh]

    Returns:
        Tuple: (description string, 7D movement classification vector)
    """
    translation_thresh, rotation_thresh, gripper_thresh = thresholds
    diff = move[-1] - move[0]

    # Normalize translation movement (x, y, z)
    if np.sum(np.abs(diff[:3])) > 3 * translation_thresh:
        diff[:3] *= 3 * translation_thresh / np.sum(np.abs(diff[:3]))

    diff[3:6] /= 10
    move_vec = np.zeros_like(diff, dtype=float)

    # Apply thresholds
    move_vec[:3] = 1 * (diff[:3] > translation_thresh) - 1 * (diff[:3] < -translation_thresh)
    move_vec[3:6] = 1 * (diff[3:6] > rotation_thresh) - 1 * (diff[3:6] < -rotation_thresh)
    move_vec[6] = 1 * (diff[6] > gripper_thresh) - 1 * (diff[6] < -gripper_thresh)

    return describe_move(move_vec, task_suite=task_suite), move_vec


def normalize_gripper_opening(states, max_finger_dist=0.04):
    # The maximum opening of the Franka Emika Panda gripper is 80 millimeters
    """
    Normalize the gripper width (last 2 dims) to [0, 1] and replace with a single value.

    Args:
        states: np.ndarray of shape (N, 8)
        max_finger_dist: max opening for each finger in meters (default 0.04)

    Returns:
        np.ndarray of shape (N, 7) with last value as normalized gripper opening
    """
    # Average gripper width (absolute) and normalize
    left = np.abs(states[:, 6])
    right = np.abs(states[:, 7])
    avg_opening = (left + right) / 2
    normalized = np.clip(avg_opening / max_finger_dist, 0, 1)

    # Replace last two dims with one normalized dim
    new_states = np.hstack([states[:, :6], normalized[:, np.newaxis]])
    return new_states


def axisangle2euler(states):
    # https://robosuite.ai/docs/source/robosuite.utils.html
    """
    Converts axis-angle representation (states[:, 3:6]) to Euler angles (rpy) in-place.

    Args:
        states (np.ndarray): Array of shape (N, 8) where states[:, 3:6] contains axis-angle.

    Returns:
        np.ndarray: Modified states array with Euler angles (roll, pitch, yaw) in place of axis-angle.
    """
    states = states.copy()
    for i in range(states.shape[0]):
        axis_angle = states[i, 3:6]
        quat = T.axisangle2quat(axis_angle)
        rmat = T.quat2mat(quat)
        euler = T.mat2euler(rmat)  # (roll, pitch, yaw)
        states[i, 3:6] = euler
    return states


def optimize_movement_classification(trajectory_states, initial_thresholds=[0.03, 0.03, 0.03], 
                                     min_translation_thresh=0.02, max_translation_thresh=0.4, 
                                     num_steps=50):
    """
    Optimizes the translation threshold to minimize overlaps between translation and other movements
    while also minimizing the number of stops.
    
    Args:
        trajectory_states: List of state vectors for the entire trajectory
        initial_thresholds: Initial [translation, rotation, gripper] thresholds
        min_translation_thresh: Minimum translation threshold to try
        max_translation_thresh: Maximum translation threshold to try
        num_steps: Number of threshold values to try
        
    Returns:
        Optimal threshold values [translation, rotation, gripper]
    """
    def count_overlaps_and_stops(thresholds):
        # Create windows of 4 states
        move_windows = [trajectory_states[i:i+4] for i in range(len(trajectory_states)-3)]
        
        # Classify each window
        classifications = []
        for move in move_windows:
            _, move_vec = classify_movement(move, thresholds=thresholds)
            classifications.append(move_vec)
            
        # Count overlaps and stops
        overlaps = 0
        stops = 0
        
        for move_vec in classifications:
            # Check if translation is happening
            has_translation = any(move_vec[:3] != 0)
            
            # Check if rotation or gripper movement is happening
            has_rotation = any(move_vec[3:6] != 0)
            has_gripper = move_vec[6] != 0
            
            # Count overlaps (translation happening simultaneously with rotation or gripper)
            # if has_translation and (has_rotation or has_gripper):
            if has_translation and has_gripper:
                overlaps += 1
                
            # Count stops (no movement at all)
            if not has_translation and not has_rotation and not has_gripper:
                stops += 1
                
        return overlaps, stops
    
    # Try different translation thresholds
    thresh_values = np.linspace(min_translation_thresh, max_translation_thresh, num_steps)
    
    best_score = float('inf')
    best_thresh = initial_thresholds[0]
    best_results = None
    
    for trans_thresh in thresh_values:
        thresholds = [trans_thresh, initial_thresholds[1], initial_thresholds[2]]
        overlaps, stops = count_overlaps_and_stops(thresholds)
        
        # Define a score - heavily penalize overlaps, lightly penalize stops
        # each stop is considered x times worse than a overlap.
        # this number is magic and finetuned
        score = overlaps * 1 + stops * 2.5
        # print(trans_thresh, score)
        
        if score < best_score:
            best_score = score
            best_thresh = trans_thresh
            best_results = (overlaps, stops)
    
    print()
    print(f"Optimal translation threshold: {best_thresh:.4f}")
    print(f"Results: {best_results[0]} overlaps, {best_results[1]} stops")
    
    return [best_thresh, initial_thresholds[1], initial_thresholds[2]]


def get_move_primitives_episode(episode, task_suite="bridge", thresholds=[0.03, 0.03, 0.03], optimize_thresholds=True):
    steps = list(episode["steps"])

    states = np.array([step["observation"]["state"] for step in steps])
    actions = [step["action"][:3].numpy() for step in steps]

    if task_suite != "bridge":
        states = axisangle2euler(states)
        states[:, 1] *= -1 
        states[:, 3:6] *= -1 
        states = normalize_gripper_opening(states)
    
    # print()
    # print(states)

    if optimize_thresholds:
        thresholds = optimize_movement_classification(states, initial_thresholds=thresholds,
                                                      min_translation_thresh=thresholds[0] - 0.01, 
                                                      max_translation_thresh=thresholds[0] + 0.01, 
                                                      num_steps=50)

    move_trajs = [states[i : i + 4] for i in range(len(states) - 1)]
    primitives = [classify_movement(move, task_suite=task_suite, thresholds=thresholds) for move in move_trajs]
    primitives.append(primitives[-1])

    return primitives


# def connect_sparse_values(states, window_size=4):
#     """
#     Connect sparse values in a binary list by filling in isolated different values.
    
#     Args:
#         states: List of binary values (0s and 1s)
#         window_size: Size of window to consider for determining if a value is isolated
        
#     Returns:
#         Processed list with connected values
#     """
#     if len(states) <= window_size * 2:
#         return states  # List too short to process
        
#     result = states.copy()
    
#     # Handle the beginning of the list
#     if len(result) > window_size:
#         start_value = result[window_size]
#         if all(val == start_value for val in result[window_size:window_size*2]):
#             # If there's a consistent value after the start, make the beginning match
#             for i in range(window_size):
#                 result[i] = start_value
    
#     # Handle the end of the list
#     if len(result) > window_size * 2:
#         end_value = result[-window_size-1]
#         if all(val == end_value for val in result[-window_size*2:-window_size]):
#             # If there's a consistent value before the end, make the ending match
#             for i in range(len(result)-window_size, len(result)):
#                 result[i] = end_value
    
#     # Process the main part of the list
#     i = window_size
#     while i < len(result) - window_size:
#         # Get current value and surrounding window
#         current = result[i]
#         before = result[i-window_size:i]
#         after = result[i+1:i+window_size+1]
        
#         # If current value differs from both before and after
#         if (current == 1 and all(b == 0 for b in before) and all(a == 0 for a in after)) or \
#            (current == 0 and all(b == 1 for b in before) and all(a == 1 for a in after)):
#             # Change the current value to match surrounding values
#             result[i] = 1 - current
            
#         # Check for small islands of different values
#         if window_size > 1:
#             island_size = 1
#             for j in range(1, window_size):
#                 if i+j < len(result) and result[i] == result[i+j]:
#                     island_size += 1
#                 else:
#                     break
                    
#             if 1 < island_size < window_size:
#                 # Check if surrounded by the opposite value
#                 if i-1 >= 0 and i+island_size < len(result) and \
#                    result[i-1] == result[i+island_size] and \
#                    result[i-1] != result[i]:
#                     # Fill in the small island
#                     for j in range(island_size):
#                         result[i+j] = result[i-1]
                    
#         i += 1
    
#     return result


def combine_move_vectors(moves_list):
    """
    Combine multiple move vectors by averaging and rounding.

    Args:
        moves_list: list of np.ndarray of shape (D,), each in {-1, 0, 1}

    Returns:
        np.ndarray of shape (D,) with combined move vector in {-1, 0, 1}
    """
    avg = np.mean(moves_list, axis=0)
    rounded = np.round(avg).astype(int)
    return np.clip(rounded, -1, 1)  # just in case rounding gives out-of-bound


def filter_abnormal_gripper_states(states, target=-1):
    """
    Replace abnormal values (0s) that are surrounded by a longer sequence of consistent values (e.g., -1).

    Args:
        states (list or np.ndarray): list of -1, 0, 1 representing gripper actions.
        target (int): the surrounding dominant value to check for (-1 or 1).

    Returns:
        np.ndarray: cleaned states with filtered abnormal values.
    """
    states = np.array(states)
    filtered = states.copy()
    N = len(states)

    i = 0
    while i < N:
        if states[i] != 0:
            i += 1
            continue

        # Start of abnormal (0) segment
        start = i
        while i < N and states[i] == 0:
            i += 1
        end = i  # exclusive

        # Check values before and after
        left = start - 1
        right = end

        # Count consecutive `target` values on the left
        left_count = 0
        while left >= 0 and states[left] == target:
            left_count += 1
            left -= 1

        # Count consecutive `target` values on the right
        right_count = 0
        while right < N and states[right] == target:
            right_count += 1
            right += 1

        # If both sides have more target values than abnormal region, filter
        if left_count + right_count > (end - start):
            filtered[start:end] = target

    return filtered


def detect_gripper_changes(episode, task_suite="bridge", threshold_list=[0.028, 0.03, 0.032]):
    """
    Vectorized gripper change detection:
    -1 = close gripper
     0 = no gripper movement
     1 = open gripper

    Args:
        episode: RLDS episode
        task_suite: "bridge" or "libero task suite"

    Returns:
        np.ndarray of shape (N,) with values -1, 0, or 1
    """
    move_list = []
    for threshold in threshold_list:
        primitives = get_move_primitives_episode(episode, task_suite=task_suite, thresholds=[0.03, 0.03, threshold], optimize_thresholds=False)
        move_vecs = np.array([mv for _, mv in primitives])  # shape (N, 7)
        move_vecs = move_vecs[:, 6].astype(int) # just extract the 7th dim (gripper)
        move_list.append(move_vecs)
    
    move = combine_move_vectors(move_list)

    if task_suite != "bridge":
        move = filter_abnormal_gripper_states(move, target=-1)
        move = filter_abnormal_gripper_states(move, target=1)
    
    return move.tolist()