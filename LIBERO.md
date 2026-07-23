# LIBERO and LIBERO-Plus Simulation Benchmark

## Setup LIBERO

Set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From cyclevla base dir
```

Note: use `mujoco==3.3.0` — newer versions (e.g. 3.4.0) cause lower inference performance due to physics simulation differences, even when using the same trained checkpoint.

## Setup LIBERO-Plus

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) is a robustness benchmark that drops in for LIBERO at eval time. It is a fork of LIBERO that occupies the same `libero.libero.*` Python namespace and the same `~/.libero/config.yaml` global config, so it cannot coexist with stock LIBERO in one conda env. We use a separate env (`openvla-oft-plus`) cloned from `openvla-oft`, and decouple the per-env `LIBERO_CONFIG_PATH` so the two installs never fight over the same YAML. LIBERO-Plus reuses the original suite names (`libero_spatial / libero_object / libero_goal / libero_10 / libero_90 / libero_mix`) rather than registering new `libero_plus_*` names — perturbation variants are added as extra `.bddl` files inside each existing suite folder. Switching benchmarks is done by `conda activate` between the two envs, not by changing `--task_suite_name`.

```bash
# Clone LIBERO-Plus into the repo
git clone https://github.com/sylvestf/LIBERO-plus.git

# Clone the conda env and swap LIBERO for LIBERO-Plus
conda create --name openvla-oft-plus --clone openvla-oft
conda activate openvla-oft-plus
pip uninstall libero
cd LIBERO-plus

# Follow LIBERO-Plus's own README for the asset bundle (https://github.com/sylvestf/LIBERO-plus):
#   - download the HuggingFace asset archive (textures, perturbed scenes, init_files)
#   - extract it into ./LIBERO-plus/libero/libero/ per their instructions
#   - install any extra apt/pip deps they list
```

Then decouple the per-env libero config so the two envs don't share `~/.libero/config.yaml`. This is a one-time setup — `conda env config vars set` persists the variable in the env's metadata, so every future `conda activate` re-exports it automatically.

```bash
conda activate openvla-oft
conda env config vars set LIBERO_CONFIG_PATH=$HOME/.libero

conda activate openvla-oft-plus
conda env config vars set LIBERO_CONFIG_PATH=$HOME/.libero-plus

# Trigger config generation under openvla-oft-plus, then verify it points at LIBERO-Plus.
# On first import, LIBERO-Plus prompts to specify custom paths (dataset folder, etc.) —
# answer N to all prompts to accept defaults; defaults already resolve under the LIBERO-plus clone.
python -c "import libero.libero"
cat $HOME/.libero-plus/config.yaml  # bddl_files / init_files / assets should resolve under /hdd2/chenyang/openvla-oft/LIBERO-plus/libero/libero/
```

Note: the default LIBERO-Plus eval protocol is `--num_trials_per_task 1` (vs. 50 in stock LIBERO) because each perturbation variant is already a separate BDDL — total ~1k–3k rollouts per suite instead of 10×50.

## Generate Subtask-Decomposed LIBERO Dataset

**Option A (recommanded):** download our pre-generated dataset (coming soon) and extract it to `decomposed_dataset/libero_sub_progress/` at the repo root, so it matches the Stage-3 output path Option B would produce.

**Option B:** regenerate from the original `modified_libero_rlds`. The pipeline runs in three stages. Adjust `--task-suite-id` (0=spatial, 1=object, 2=goal, 3=10) per suite, and run stage 1 and 2 four times to cover all four suites.

```bash
conda activate openvla-oft
# Stage 1 — per-episode scene captions (Prismatic VLM by default)
#   in:  modified_libero_rlds (original LIBERO RLDS)
#   out: vlm_response/scene_description/libero/{task_suite}/results.json
CUDA_VISIBLE_DEVICES=0 python ecot_scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py --task-suite-id 0

# Stage 2 — gripper-state chunking + LLM subtask labels (gpt-4.1 by default)
#   in:  modified_libero_rlds + Stage-1 results.json
#   out: vlm_response/decompose_traj/libero/{task_suite}/episode_*/chunks_summary.json
#        vlm_response/process_traj/libero/{task_suite}/episode_*[_llm]/chunks_summary.json
CUDA_VISIBLE_DEVICES=0,1,2 python ecot_scripts/generate_embodied_data/decompose_traj.py --task-suite-id 0 --vlm gpt

# Stage 3 — build the official RLDS dataset (LIBERO_Decomposed_Progress, subtask-only instruction, run once)
#   in:  modified_libero_rlds + Stage-2 process_traj/.../chunks_summary.json
#   out: decomposed_dataset/libero_sub_progress/libero_decomposed_progress/1.0.0/*.tfrecord-*
cd rlds_dataset_builder/LIBERO_Decomposed_Progress
CUDA_VISIBLE_DEVICES=-1 tfds build --data_dir=/hdd2/chenyang/openvla-oft/decomposed_dataset/libero_sub_progress --overwrite
```
