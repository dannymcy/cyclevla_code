import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import draccus
import zmq

from prismatic.conf import ModelConfig, ModelRegistry

# TODO (@moojink) Hack so that the interpreter can find experiments.robot
sys.path.append("../..")
from experiments.bridge.utils import get_model


@dataclass
class GenerateConfig:
    # fmt: off

    # ModelConfig from `prisma/conf/models.py`; override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.REPRODUCTION_7B.model_id)
    )
    model_family: str = "llava"                                 # Base VLM model family (for prompt builder)

    # Model Parameters
    pretrained_checkpoint: Union[str, Path] = Path(             # Pretrained VLA checkpoint to load
        "/scr/moojink/checkpoints/tri/reproduction-llava-v15+mx-bridge+n1+b32+x7/checkpoints/"
        "step-077500-epoch-00-loss=0.0488.pt"
    )

    # Environment-Specific Parameters
    host_ip: str = "localhost"
    port: int = 5678

    # Note (@moojink) =>> Setting initial orientation with a 30 degree offset -- more natural!
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, 0., 0.16])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = False
    max_episodes: int = 50
    max_steps: int = 600
    control_frequency: float = 5

    # Training stage (doesn't matter here, but the loading function expects the argument)
    stage: str = "vla-finetune"

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def run_reasoning_server(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    # Load Model --> Get Expected Image Dimensions
    model = get_model(cfg)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5623")

    while True:
        print("Waiting for input...")
        message = socket.recv()
        inputs = pickle.loads(message)
        result = model.raw_generate(*inputs)
        print("Output generated:", result)
        socket.send(pickle.dumps(result))
        print("Output sent")


if __name__ == "__main__":
    run_reasoning_server()
