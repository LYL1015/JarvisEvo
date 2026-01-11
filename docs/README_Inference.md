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

## Step 2: Download Model Weights and Dataset

To run the batch inference, you need to download the model weights and ArtEdit-Bench-Lr dataset from Hugging Face:

### 2.1 Download JarvisEvo Model Weights

1. Create the weights directory (if it doesn't exist):
   ```bash
   cd JarvisEvo/
   mkdir -p ./checkpoints/pretrained/JarvisEvo/
   ```

2. Download the JarvisEvo weights from [Hugging Face repository](https://huggingface.co/JarvisEvo/JarvisEvo):
   ```bash
   # Using huggingface-cli (recommended)
   huggingface-cli download JarvisEvo/JarvisEvo --local-dir ./checkpoints/pretrained/JarvisEvo
   
   # Or using git-lfs
   git lfs install
   git clone https://huggingface.co/JarvisEvo/JarvisEvo ./checkpoints/pretrained/JarvisEvo
   ```

3. If you've placed the model weights in a different location, remember to update the `model_name_or_path` parameter in `src/inference/config/qwen3_vl.yaml` to point to your custom model directory.

### 2.2 Download ArtEdit-Bench-Lr Dataset

1. Create the dataset directory:
   ```bash
   mkdir -p ./datasets/ArtEdit-Bench/
   ```

2. Download the ArtEdit-Bench-Lr dataset from [Hugging Face dataset repository](https://huggingface.co/datasets/JarvisEvo/ArtEdit-Bench):
   ```bash
   # Using huggingface-cli (recommended)
   huggingface-cli download JarvisEvo/ArtEdit-Bench --repo-type dataset --local-dir ./datasets/ArtEdit-Bench
   
   # Or using git-lfs
   git lfs install
   git clone https://huggingface.co/datasets/JarvisEvo/ArtEdit-Bench ./datasets/ArtEdit-Bench
   ```

3. Extract the dataset:
   ```bash
   cd ./datasets/ArtEdit-Bench/
   unzip ArtEdit-Bench-Lr.zip
   ```

**Note:** The ArtEdit-Bench-Lr dataset is approximately 805 MB. Make sure you have sufficient disk space before downloading.


## Step 3: Start Server-Side Services

### 3.1 Start JarvisEvo API Service (Server-Side - Terminal 1)

First, start the JarvisEvo vLLM API service on the server:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve ./checkpoints/pretrained/JarvisEvo \
    --tensor-parallel-size 8 \
    --port 8086 \
    --api-key 0 \
    --served-model-name qwen3_vl \
    --max_model_len 20480 \
    --limit-mm-per-prompt.image 5
```

**Note:** This service needs to keep running. Please start it in a separate terminal window.

### 3.2 Start Lightroom Reverse Connection Server (Server-Side - Terminal 2)

Open a new terminal window on the server and start the Lightroom reverse connection server:

```bash
cd lrc_scripts/servers
bash start_reverse_server.sh
```

**Default Configuration:**
- Listen Address: `0.0.0.0`
- Listen Port: `8081`
- Upload Directory: `JarvisEvo/lrc_scripts/servers/lr_caches/uploads`
- Results Directory: `JarvisEvo/lrc_scripts/servers/lr_caches/results`

**Custom Configuration Examples:**

```bash
# Customize port and directories
bash start_reverse_server.sh --port 8082 --max-retries 10 --wait-timeout 300

# View all available parameters
bash start_reverse_server.sh --help
```

**Note:** This service needs to keep running. Please start it in a separate terminal window.

## Step 4: Start Mac Client Service

### 4.1 Configure Server Connection Information

On the Mac local machine, you first need to configure the server address to connect to. Edit the `lrc_scripts/clients/start_mac_client.sh` file and modify the following configuration:

```bash
# Change to your server IP and port (supports multiple servers, separated by commas) in start_mac_client.sh
DEFAULT_LINUX_SERVERS="YOUR_SERVER_IP:8081"
```

### 4.2 Install Agent-to-Lightroom (A2L) Plugin

**IMPORTANT:** Before starting the Mac/Windows client, you must first install the A2L plugin in Adobe Lightroom Classic. This plugin enables communication between the agent and Lightroom for automated photo processing.

**Installation Steps:**

1. Open Adobe Lightroom Classic
2. Navigate to `File` → `Plug-in Manager...`
3. Click the `Add` button in the Plugin Manager window
4. Browse and select the `lrc_scripts/clients/agent_to_lightroom/XMPlayer.lrplugin/` directory
5. Click `Done` to complete the installation

The XMP Player plugin should now appear in your plugin list and be ready to use.

**For detailed installation instructions with screenshots, please refer to:**
[Agent-to-Lightroom Plugin Documentation](../lrc_scripts/clients/agent_to_lightroom/README.md)

### 4.3 Start Mac/Windows Client

After installing the A2L plugin, run the following command on the Mac local machine to start the client:

```bash
cd lrc_scripts/clients
bash start_mac_client.sh
```

**Default Configuration:**
- Server Address: Needs to be configured in the script
- Local Lightroom API Port: `7777`
- Polling Interval: `1.0` seconds

**Custom Configuration Examples:**

```bash
# Specify server address and port
bash start_mac_client.sh --servers "192.168.1.100:8081,192.168.1.101:8081" --api-port 7777

# View all available parameters
bash start_mac_client.sh --help
```

**Prerequisites:**
- Ensure Adobe Lightroom Classic is installed and running on Mac
- Ensure the A2L plugin is properly installed (see section 4.2)
- Ensure network connectivity to server port 8081

**Note:** This service needs to keep running. The client will automatically connect to the server and maintain heartbeat.

## Step 5: Run Batch Inference

Once all services are successfully started and connected, run the batch inference on the server side:

```bash
# Full configuration for Lightroom mode
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

# AIGC mode
python inference.py \
    --image_path /path/to/your/images \
    --save_base_path /path/to/save/results \
    --task_type aigc \
    --prompt_file_name user_want_en.txt \
    --AIGC_model_pth ./checkpoints/pretrained/Qwen-Image-Edit-2511 \
    --AIGC_device cuda:0
```

**Parameter Explanation:**
- `--AIGC_model_pth`: Path to the AIGC model checkpoint directory
  - Example: `checkpoints/pretrained/Qwen-Image-Edit-2511`
  - This should point to the directory containing the pre-trained AIGC model files
  - The model is used for AI-generated content and image editing tasks
  - Make sure the model files are downloaded and placed in the specified directory before running

## Parameter Explanation

### vLLM API Service Parameters (Step 3.1)

- `VLLM_WORKER_MULTIPROC_METHOD=spawn`: Sets the multiprocessing method to 'spawn' for worker processes (required for compatibility with certain CUDA contexts)
- `./checkpoints/pretrained/JarvisEvo`: Path to the model checkpoint directory
- `--tensor-parallel-size 8`: Number of GPUs to use for tensor parallelism (distributes model across 8 GPUs)
- `--port 8086`: Port number for the API server
- `--api-key 0`: API key for authentication (set to '0' to disable authentication)
- `--served-model-name qwen3_vl`: Model name identifier for the API endpoint
- `--max_model_len 20480`: Maximum sequence length (tokens) that can be processed
- `--limit-mm-per-prompt.image 5`: Maximum number of images allowed per prompt (multimodal limit)

### Lightroom Reverse Connection Server Parameters (Step 3.2)

- `--host HOST`: Listen address (default: `0.0.0.0`)
- `--port PORT`: Listen port (default: `8081`)
- `--upload-dir DIR`: Upload file storage directory (default: `./lr_caches/uploads`)
- `--results-dir DIR`: Result file storage directory (default: `./lr_caches/results`)
- `--max-retries NUM`: Maximum retry count (default: `5`)
- `--wait-timeout SEC`: File wait timeout in seconds (default: `180.0`)
- `--retry-delay SEC`: Retry delay in seconds (default: `2.0`)
- `--backoff-factor NUM`: Backoff factor (default: `1.5`)

### Mac Client Parameters (Step 4.2)

- `--servers SERVERS`: Linux server addresses (format: `IP:PORT,IP:PORT`)
- `--api-port PORT`: Local Lightroom API port (default: `7777`)
- `--api-path PATH`: API_Lightroom project path (default: `./`)
- `--client-id ID`: Client ID (default: auto-generated)
- `--poll-interval SEC`: Polling interval in seconds (default: `1.0`)
- `--retry-delay SEC`: Connection retry delay in seconds (default: `3.0`)
- `--max-failures NUM`: Maximum consecutive failures (default: `5`)
- `--health-interval SEC`: Health check interval in seconds (default: `30.0`)
- `--max-empty-polls NUM`: Consecutive empty polls threshold (default: `50`)

### Batch Inference Parameters (Step 5)

**API Configuration:**
- `--api_endpoint`: API server address (default: `localhost`)
- `--api_port`: API server port(s). Multiple ports enable load balancing (default: `[8086, 8085]`). You can specify multiple ports like `--api_port 8086 8085`
- `--api_key`: API authentication key (default: `0`)
- `--model_name`: AI model name for image processing (default: `qwen3_vl`)

**Processing Configuration:**
- `--max_threads`: Maximum concurrent processing threads (default: `20`)
- `--task_type`: Processing mode - `lightroom`, `aigc`, or `auto` (default: `lightroom`)

**File Paths:**
- `--image_path` (required): Directory containing input images with subdirectories. Each subdirectory should contain an image file (`before.jpg` or `before.png`) and a user prompt file
- `--save_base_path` (required): Base directory for saving processing results
- `--prompt_file_name`: Filename of user prompt file in each image directory (default: `user_want_en.txt`)

**AIGC Configuration (only used when task_type=aigc):**
- `--AIGC_model_pth`: AIGC model path or identifier (default: `None`)
- `--AIGC_device`: AIGC device specification (default: `cuda:0`)

**Processing Parameters:**
- `--max_rounds`: Maximum number of processing rounds (default: `10`)
- `--quality_threshold`: Minimum quality score threshold for triggering reflection (default: `3.0`)
- `--default_timeout`: Default timeout for API requests in seconds (default: `180`)
- `--api_timeout`: API connection timeout in seconds (default: `30`)
