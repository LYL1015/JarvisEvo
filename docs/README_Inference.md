# Batch Inference Guide

This guide provides instructions on how to run the batch inference for JarvisEvo.

## Step 1: Installation
1. **Conda Environment:** Set up the conda environment with required dependencies before running the demo.
```bash
conda create -n jarvisevo_infer python=3.11
conda activate jarvisevo_infer
pip install -r envs/requirements_infer.txt
# cd src/sft_rft
# pip install -e .
```
2. **Install Adobe Lightroom:** Please download and install Adobe Lightroom on your local machine from the [official website](https://www.adobe.com/products/photoshop-lightroom.html). After installation, sign in using your Adobe account credentials.

> **Note:** Adobe Lightroom is a commercial product and may require a subscription or trial account.

## Step 2: Download Model Weights

To run the Gradio demo, you need to download the weights from Hugging Face and place them in the correct location:

1. Download the JarvisEvo weights from [Hugging Face repository](https://huggingface.co/JarvisEvo/JarvisEvo)
2. Create the weights directory (if it doesn't exist):
   ```bash
   cd JarvisEvo/
   mkdir -p ./checkpoints/pretrained/JarvisEvo/
   ```
3. Place the downloaded weight files in the `./checkpoints/pretrained/JarvisEvo` directory
4. If you've placed the model weights in a different location, remember to update the `model_name_or_path` parameter in `src/inference/config/qwen3_vl.yaml` to point to your custom model directory.


## Step 3: Running the Batch Inference
Once the environment is set up and activated, you can run the batch inference with the following command from the root directory of the project:

```bash
# Firstly, start the API server
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve ./checkpoints/pretrained/JarvisEvo --tensor-parallel-size 8 --port 8086 --api-key 0 --served-model-name qwen3_vl --max_model_len 20480 --limit-mm-per-prompt.image 5

# Next, set the environment variables for ```LIGHTROOM_RESULTS_DIR```.
cd lrc_scripts/servers
export LIGHTROOM_RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lr_caches/results"
cd ../..

# Run the batch inference with full configuration
python inference.py \
    --image_path /path/to/your/images \
    --save_base_path /path/to/save/results \
    --api_endpoint localhost \
    --api_port 8086 8085 \
    --api_key 0 \
    --model_name qwen3_vl \
    --max_threads 20 \
    --task_type lightroom \
    --prompt_file_name user_want_en.txt \
    --max_rounds 10 \
    --quality_threshold 3.0 \
    --default_timeout 180 \
    --api_timeout 30

# For AIGC mode
python inference.py \
    --image_path /path/to/your/images \
    --save_base_path /path/to/save/results \
    --task_type aigc \
    --prompt_file_name user_want_en.txt \
    --AIGC_model_pth /path/to/aigc/model \
    --AIGC_device cuda:0
```

### Parameter Explanation

**vllm serve command parameters:**
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`: Sets the multiprocessing method to 'spawn' for worker processes (required for compatibility with certain CUDA contexts)
- `./checkpoints/pretrained/JarvisEvo`: Path to the model checkpoint directory
- `--tensor-parallel-size 8`: Number of GPUs to use for tensor parallelism (distributes model across 8 GPUs)
- `--port 8086`: Port number for the API server
- `--api-key 0`: API key for authentication (set to '0' to disable authentication)
- `--dtype float32`: Data type for model weights (float32 for full precision)
- `--served-model-name qwen3_vl`: Model name identifier for the API endpoint
- `--max_seq_len 20480`: Maximum sequence length (tokens) that can be processed
- `--limit-mm-per-prompt image=5`: Maximum number of images allowed per prompt (multimodal limit)

**inference.py command parameters:**

*API Configuration:*
- `--api_endpoint`: API server address (default: `localhost`)
- `--api_port`: API server port(s). Multiple ports enable load balancing (default: `[8086, 8085]`). You can specify multiple ports like `--api_port 8086 8085`.
- `--api_key`: API authentication key (default: `0`)
- `--model_name`: AI model name for image processing (default: `qwen3_vl`)

*Processing Configuration:*
- `--max_threads`: Maximum concurrent processing threads (default: `20`)
- `--task_type`: Processing mode - `lightroom`, `aigc`, or `auto` (default: `lightroom`)

*File Paths:*
- `--image_path` (required): Directory containing input images with subdirectories. Each subdirectory should contain an image file (`before.jpg` or `before.png`) and a user prompt file.
- `--save_base_path` (required): Base directory for saving processing results
- `--prompt_file_name`: Filename of user prompt file in each image directory (default: `user_want_en.txt`)

*AIGC Configuration (only used when task_type=aigc):*
- `--AIGC_model_pth`: AIGC model path or identifier (default: `None`)
- `--AIGC_device`: AIGC device specification (default: `cuda:0`)

*Processing Parameters:*
- `--max_rounds`: Maximum number of processing rounds (default: `10`)
- `--quality_threshold`: Minimum quality score threshold for triggering reflection (default: `3.0`)
- `--default_timeout`: Default timeout for API requests in seconds (default: `180`)
- `--api_timeout`: API connection timeout in seconds (default: `30`)
