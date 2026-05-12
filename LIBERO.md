# LIBERO and LIBERO-Plus Simulation Benchmark

## Setup

Set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From cyclevla base dir
```

Note: use `mujoco==3.3.0` — newer versions (e.g. 3.4.0) cause lower inference performance due to physics simulation differences, even when using the same trained checkpoint.

## Generate Subtask-Decomposed LIBERO Dataset

**Option A:** download our pre-generated dataset (coming soon) and extract it to `decomposed_dataset/libero_sub_progress/` at the repo root, so it matches the Stage-3 output path Option B would produce.

**Option B:** regenerate from the original `modified_libero_rlds`. The pipeline runs in three stages. Adjust `--task-suite-id` (0=spatial, 1=object, 2=goal, 3=10) per suite, and run each stage four times to cover all four suites.

```bash
conda activate /hdd2/kai/openvla-oft/env
# Stage 1 — per-episode scene captions (Prismatic VLM by default)
#   in:  modified_libero_rlds (original LIBERO RLDS)
#   out: vlm_response/scene_description/libero/{task_suite}/results.json
CUDA_VISIBLE_DEVICES=0 python ecot_scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py --task-suite-id 0

# Stage 2 — gripper-state chunking + LLM subtask labels (gpt-4.1 by default)
#   in:  modified_libero_rlds + Stage-1 results.json
#   out: vlm_response/decompose_traj/libero/{task_suite}/episode_*/chunks_summary.json
#        vlm_response/process_traj/libero/{task_suite}/episode_*[_llm]/chunks_summary.json
CUDA_VISIBLE_DEVICES=0,1,2 python ecot_scripts/generate_embodied_data/decompose_traj.py --task-suite-id 0 --vlm gpt

# Stage 3 — build the official RLDS dataset (LIBERO_Decomposed_Progress, subtask-only instruction)
#   in:  modified_libero_rlds + Stage-2 process_traj/.../chunks_summary.json
#   out: decomposed_dataset/libero_sub_progress/libero_decomposed_progress/1.0.0/*.tfrecord-*
cd rlds_dataset_builder/LIBERO_Decomposed_Progress
CUDA_VISIBLE_DEVICES=-1 tfds build --data_dir=/hdd2/kai/openvla-oft/decomposed_dataset/libero_sub_progress --overwrite
```
