# JarvisEvo Training Guide

## SFT (Supervised Fine-Tuning) Training

JarvisEvo provides two SFT training approaches. You can choose either one based on your needs:

### Option 1: Using LLaMA-Factory

For training with LLaMA-Factory, please refer to the official documentation:
- **Use LLaMA-Factory for training. Refer to Qwen3-VL Examples.**
- Visit [LLaMA-Factory Official Documentation](https://github.com/hiyouga/LLaMA-Factory) for detailed environment setup and training script instructions

### Option 2: Using Repository Training Scripts

#### Environment Setup

1. Create conda environment:
```bash
conda create -n jarvisevo_sft python=3.11
conda activate jarvisevo_sft
```

2. Install dependencies:
```bash
# Install basic dependencies
pip install -r envs/requirements_sft.txt

# Navigate to training directory and install local package
cd src/sft_rft
pip install -e ".[torch,metrics]" --no-build-isolation
```

#### Training Execution

Use the provided training script:
```bash
# Ensure you are in the src/sft_rft directory
cd src/sft_rft
bash run_scripts/single_node_sft.sh
```

## SEPO Training

🚧 **Coming Soon** - SEPO training code is being organized and will be released soon!


