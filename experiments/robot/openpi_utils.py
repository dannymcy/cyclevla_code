"""
openpi_utils.py

Inference-side glue for evaluating a **pi05 / openpi** CycleVLA policy with the
LIBERO eval infrastructure in this repo. Sibling of `openvla_utils.py`, but
where `openvla_utils.py` loads an in-process PyTorch OpenVLA model, this module
talks to a remote **openpi websocket policy server** (started by
`openpi/serve_openpi_cyclevla.sh`).

Why a separate module is needed -- the openpi policy differs from the OpenVLA-OFT
policy in three ways that must be reconciled here, *not* in the eval scripts:

1. Architecture. openpi serves a JAX pi05 model behind a websocket; there is no
   local model object, no action head, no proprio projector. We just send an
   observation dict and receive an action chunk.

2. Gripper convention. OpenVLA-OFT inverts the gripper in its dataloader
   (`1 - clip(g, 0, 1)`) and de-inverts at eval time via
   `normalize_gripper_action` + `invert_gripper_action`. openpi (matching stock
   `pi05_libero`) trains on the *raw* RLDS gripper and its LIBERO example feeds
   the policy's gripper straight to `env.step()`. So we deliberately apply NO
   gripper normalization/inversion to an openpi action -- dims 0-6 are already
   in the env's native convention.

3. Stop/progress range. OpenVLA-OFT's `process_action` maps the stop signal to
   {-1 = STOP, +1 = GO}. openpi has no such post-processing: the model emits
   stop and progress as raw floats -- stop ~= {0.0, 1.0} (1.0 = stop, from
   `is_last`) and progress ~= [0.1, 1.0] (from `is_terminal`). The openpi eval
   scripts therefore threshold them directly (`stop > 0.5`, `progress >= ...`).

Requires the `openpi-client` package to be importable in the eval env:
    conda activate /hdd2/kai/openvla-oft/env
    pip install -e /hdd2/kai/openvla-oft/openpi/packages/openpi-client
"""

import numpy as np

# openpi-client: websocket client + image preprocessing identical to openpi
# training (`resize_with_pad` + `convert_to_uint8`). Installed separately --
# see the module docstring.
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from experiments.robot.libero.libero_utils import (
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)

# pi05 LIBERO is trained on 224x224 images (see openpi/examples/libero/main.py).
OPENPI_RESIZE_SIZE = 224

# CycleVLA action layout returned by the server (config `pi05_libero_cyclevla`
# with `action_dim=9`): [0:6] 6D EEF delta, [6] gripper, [7] stop s_t,
# [8] progress p_t.
OPENPI_ACTION_DIM = 9


def build_openpi_observation(obs, task_label, resize_size=OPENPI_RESIZE_SIZE):
    """Build the observation dict expected by the openpi LIBERO policy server.

    Mirrors `openpi/examples/libero/main.py` exactly so the inputs match how
    the policy was trained: 180-deg-rotated images resized with aspect-ratio
    padding to 224x224 uint8, and an 8-dim proprio state.

    Args:
        obs: raw robosuite/LIBERO observation dict.
        task_label: language prompt. For CycleVLA this is the bare lowercased
            subtask string -- the Stage-3 RLDS builder writes exactly that into
            `language_instruction`, and `pi05_libero_cyclevla` trains with
            `prompt_from_task=True`, so the inference prompt must match.
        resize_size: target square image size.

    Returns:
        dict with keys `observation/image`, `observation/wrist_image`,
        `observation/state`, `prompt` -- the keys `LiberoInputs` expects.
    """
    # `get_libero_image` / `get_libero_wrist_image` already rotate 180 deg,
    # which is also what openpi's LIBERO example does -- so reuse them.
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Aspect-preserving resize + uint8. We use openpi's own `resize_with_pad`
    # (not OpenVLA's `resize_image_for_policy`) so preprocessing matches the
    # openpi training pipeline.
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

    # 8-dim proprio: 3D eef pos + 3D eef axis-angle + 2D gripper qpos.
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_label),
    }


def split_openpi_action(action):
    """Split a 9-dim openpi action into (robot_action, stop, progress).

    Returns:
        robot_action: np.ndarray of shape (7,) -- [6D EEF delta, gripper],
            directly executable by `env.step()`. Deliberately NOT passed
            through `normalize_gripper_action`/`invert_gripper_action`: openpi
            is trained on the raw RLDS gripper, so dims 0-6 are already in the
            env's native convention (see module docstring, point 2).
        stop: float -- raw stop signal s_t, ~1.0 means "stop" (from `is_last`).
        progress: float -- raw progress signal p_t, ~[0.1, 1.0] (from
            `is_terminal`).
    """
    action = np.asarray(action, dtype=np.float32)
    assert action.shape[-1] == OPENPI_ACTION_DIM, (
        f"Expected a {OPENPI_ACTION_DIM}-dim openpi action "
        f"(serve with config `pi05_libero_cyclevla`, action_dim=9), got shape {action.shape}."
    )
    robot_action = action[:7]
    stop = float(action[7])
    progress = float(action[8])
    return robot_action, stop, progress


class OpenPiClient:
    """Thin wrapper around openpi's `WebsocketClientPolicy` for LIBERO eval.

    Connecting blocks until the policy server (`openpi/serve_openpi_cyclevla.sh`)
    is reachable, so construct this once at eval startup.
    """

    def __init__(self, host="0.0.0.0", port=8000, resize_size=OPENPI_RESIZE_SIZE):
        self.host = host
        self.port = port
        self.resize_size = resize_size
        # Blocks until the server responds; raises on connection failure.
        self._client = WebsocketClientPolicy(host=host, port=port)
        print(f"[openpi] Connected to policy server at ws://{host}:{port}")

    def get_action(self, obs, task_label, replan_steps):
        """Query the server and return a list of up to `replan_steps` actions.

        The server returns a chunk of `action_horizon` (10) actions. We return
        only the first `replan_steps` of them so the caller's action queue
        (`deque(maxlen=replan_steps)`) is filled exactly, in order, with no
        actions silently dropped.

        Args:
            obs: raw LIBERO observation dict.
            task_label: language prompt (bare lowercased subtask string).
            replan_steps: number of open-loop actions to execute before
                requerying the policy.

        Returns:
            list of np.ndarray, each shape (9,).
        """
        element = build_openpi_observation(obs, task_label, self.resize_size)
        action_chunk = np.asarray(self._client.infer(element)["actions"])
        n = min(replan_steps, len(action_chunk))
        return [action_chunk[i] for i in range(n)]
