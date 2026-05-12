# Setup Instructions

## Set Up Conda Environment for OpenVLA-OFT

Mostly mirrors the upstream OpenVLA-OFT setup (https://github.com/moojink/openvla-oft/blob/main/SETUP.md), with the additions noted below.

```bash
# Option A: named env
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Option B: path-based env in the project folder
conda create --prefix "$(pwd)/env" python=3.10 -y
conda activate "$(pwd)/env"

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
# Example (the configuration used in this repo's dev environment):
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Clone CycleVLA repo and pip install to download dependencies.
# `--recurse-submodules` pulls openpi/ and rlds_dataset_builder/ in the same step.
git clone --recurse-submodules git@github.com:dannymcy/cyclevla_code.git
cd cyclevla_code
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

# Install additional packages used by CycleVLA's eval and analysis scripts
pip install openai pyzmq scikit-learn pandas openpyxl python-dotenv transitions tensorflow_hub apache_beam plotly
```

## Set Up Conda Environment for pi0.5 (openpi)

Follow the setup in https://github.com/Physical-Intelligence/openpi (this repo vendors openpi as a submodule under `openpi/`).

## Set Up API Keys (.env)

Scripts that talk to Hugging Face or OpenAI read credentials from a `.env` file at the repo root via `python-dotenv` (`load_dotenv()` is called with no path argument, so the file must sit next to where you launch from).

```bash
cat > .env <<'EOF'
HUGGINGFACE_TOKEN=hf_xxx_replace_me
OPENAI_API_KEY=sk-xxx_replace_me
EOF
```