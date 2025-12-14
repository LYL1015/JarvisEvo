
import re
import os
import json
from typing import Optional
import asyncio
import aiohttp
import tempfile
import logging
from pathlib import Path
import time
from lua_converter import LuaConverter
from json_repair import repair_json
import threading
# Set the visible CUDA device (adjust as necessary)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 配置日志
logger = logging.getLogger(__name__)


class LightroomManager:
    """
    Simplified Lightroom manager for processing model output <tool_call></tool_call> content
    Based on pp3_reward implementation, calls remote Lightroom service through reverse_server
    
    Usage example:
        lightroom_manager = LightroomManager()
        
        # Extract tool_call content from model output
        model_output = "Please adjust image <tool_call>{'exposure': 0.5, 'contrast': 0.2}</tool_call>"
        tool_content = lightroom_manager.extract_tool_call_content(model_output)
        
        # Process image
        processed_path = lightroom_manager.process_image("/path/to/image.jpg", tool_content)
        
    Environment variable configuration:
        LIGHTROOM_SERVER_HOST: reverse_server host address (default: 127.0.0.1)
        LIGHTROOM_SERVER_PORT: reverse_server port (default: 8081)
        LIGHTROOM_MAX_RETRIES: Maximum retry attempts (default: 5)
        LIGHTROOM_REQUEST_TIMEOUT: Request timeout in seconds (default: 600.0)
        LIGHTROOM_TEMP_DIR: Temporary files directory (default: /tmp/lightroom_grpo)
        LIGHTROOM_RESULTS_DIR: Results files directory (default: /.../lightroom_results)
    """
    
    def __init__(self):
        # Configure reverse_server address and retry parameters
        self.server_host = os.getenv('LIGHTROOM_SERVER_HOST', '0.0.0.0')
        self.server_port = int(os.getenv('LIGHTROOM_SERVER_PORT', '8081'))
        self.server_url = f"http://{self.server_host}:{self.server_port}"
        
        # Retry configuration
        self.max_retries = int(os.getenv('LIGHTROOM_MAX_RETRIES', '5'))
        self.request_timeout = float(os.getenv('LIGHTROOM_REQUEST_TIMEOUT', '600.0'))
        self.retry_delay = float(os.getenv('LIGHTROOM_RETRY_DELAY', '2.0'))
        self.backoff_factor = float(os.getenv('LIGHTROOM_BACKOFF_FACTOR', '1.5'))
        
        # Temporary directory configuration
        self.temp_base_dir = os.getenv('LIGHTROOM_TEMP_DIR', '/tmp/lightroom_grpo')
        os.makedirs(self.temp_base_dir, exist_ok=True)
        
        # Results directory configuration
        self.results_base_dir = os.getenv('LIGHTROOM_RESULTS_DIR', '/apdcephfs_zwfy/share_303937731/lyl1015/projects/JarvisArt_Evo/lr_grpo_caches/lightroom_results')

        self.max_concurrent = int(os.getenv('LIGHTROOM_MAX_CONCURRENT', '20'))
        self._semaphore = threading.Semaphore(self.max_concurrent)

    def extract_tool_call_content(self, model_output: str) -> Optional[str]:
        """
        Extract content from <tool_call></tool_call> tags in model output
        
        Args:
            model_output (str): Raw output text from the model
            
        Returns:
            Optional[str]: Extracted tool call content, returns None if not found
        """
        try:
            # Use regex to extract content within <tool_call> tags
            pattern = r'<tool_call>(.*?)</tool_call>'
            matches = re.findall(pattern, model_output, re.DOTALL)
            
            if matches:
                # Return first matched content, stripped of leading/trailing whitespace
                return matches[0].strip()
            else:
                logger.warning("No <tool_call> tags found in model output")
                return None
                
        except Exception as e:
            logger.error(f"Error occurred while extracting tool_call content: {e}")
            return None
    
    def json_to_lua(self, json_content: str) -> str:
        """
        Convert JSON format Lightroom parameters to Lua script (using LuaConverter)
        
        Args:
            json_content (str): JSON format parameter string
            
        Returns:
            str: Converted Lua script content
        """
        try:
            # Use LuaConverter to convert JSON parameters to Lua format, add return statement header
            adjustments_lua = 'return ' + LuaConverter.to_lua(json_content)
            return adjustments_lua
            
        except Exception as e:
            logger.error(f"JSON to Lua conversion failed: {e}, returning empty Lua table")
            # Return empty Lua table as fallback
            return 'return {}'
    
    async def process_image_async(self, image_path: str, tool_call_content: str) -> Optional[str]:
        """
        Asynchronously process image using extracted tool_call content
        
        Args:
            image_path (str): Original image file path
            tool_call_content (str): Tool call content extracted from model output
            
        Returns:
            Optional[str]: Path to processed image, returns None on failure
        """
        try:
            # Create temporary directory and Lua configuration file
            temp_dir = Path(tempfile.mkdtemp(prefix='lightroom_grpo_', dir=self.temp_base_dir))
            lua_path = temp_dir / "config.lua"
            
            # Convert tool_call_content string to JSON first
            json_content = json.loads(repair_json(tool_call_content))

            lua_content = self.json_to_lua(json_content)
            with open(lua_path, 'w', encoding='utf-8') as f:
                f.write(lua_content)
            
            # Submit Lightroom processing task
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                result_path = await self._submit_lightroom_task_with_retry(
                    session, image_path, lua_path, "lightroom_process"
                )
                
                return result_path
                
        except Exception as e:
            logger.error(f"Lightroom image processing failed: {e}")
            return None
    
    def process_image(self, image_path: str, tool_call_content: str) -> Optional[str]:
        """
        Synchronous version of image processing method
        
        Args:
            image_path (str): Original image file path
            tool_call_content (str): Tool call content extracted from model output
            
        Returns:
            Optional[str]: Path to processed image, returns None on failure
        """
        with self._semaphore:  # Limit concurrent processing
            return asyncio.run(self.process_image_async(image_path, tool_call_content))
    
    async def _submit_lightroom_task_with_retry(self, session: aiohttp.ClientSession, 
                                              photo_path: str, lua_path: str, task_name: str) -> Optional[str]:
        """
        Submit Lightroom processing task to reverse_server and wait for result - with exponential backoff retry
        
        Args:
            session (aiohttp.ClientSession): HTTP session for requests
            photo_path (str): Path to the input photo
            lua_path (str): Path to the Lua configuration file
            task_name (str): Name of the processing task
            
        Returns:
            Optional[str]: Processing result path, returns None on failure
        """
        last_exception = None
        current_delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return await self._submit_lightroom_task(session, photo_path, lua_path, task_name)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning(f"{task_name} task failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    logger.info(f"Waiting {current_delay:.1f}s before retry...")
                    await asyncio.sleep(current_delay)
                    current_delay = min(current_delay * self.backoff_factor, 60.0)  # Maximum 60 seconds
                else:
                    logger.error(f"{task_name} task finally failed (attempted {self.max_retries + 1} times): {e}")
        
        return None
    
    async def _submit_lightroom_task(self, session: aiohttp.ClientSession, 
                                    photo_path: str, lua_path: str, task_name: str) -> str:
        """
        Submit Lightroom processing task to reverse_server and wait for result
        
        Args:
            session (aiohttp.ClientSession): HTTP session for requests
            photo_path (str): Path to the input photo
            lua_path (str): Path to the Lua configuration file
            task_name (str): Name of the processing task
            
        Returns:
            str: Processing result path
        """
        # 1. Submit task
        params = {
            "photo_path": str(photo_path),
            "xmp_path": str(lua_path)
        }
        
        async with session.post(f"{self.server_url}/api/submit_task_with_files", params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Task submission failed {response.status}: {error_text}")
            
            result = await response.json()
            task_id = result["task_id"]
            logger.info(f"{task_name} task submitted: {task_id}")
        
        # 2. Poll task status until completion
        max_wait_time = float(os.getenv('LIGHTROOM_FILE_WAIT_TIMEOUT', '300.0'))
        start_time = time.time()
        poll_interval = 2.0
        
        while time.time() - start_time < max_wait_time:
            async with session.get(f"{self.server_url}/api/task_status/{task_id}") as response:
                if response.status != 200:
                    await asyncio.sleep(poll_interval)
                    continue
                
                status_data = await response.json()
                task_status = status_data.get("status")
                
                if task_status == "completed":
                    # 3. Get processing result
                    result_path = await self._get_task_result(task_id, task_name)
                    logger.info(f"{task_name} task processing completed: {task_id} -> {result_path}")
                    return result_path
                elif task_status == "failed":
                    error_msg = status_data.get("error", "Unknown error")
                    raise Exception(f"Lightroom processing failed: {error_msg}")
                elif task_status == "pending":
                    logger.warning(f"Task {task_id} status reset to pending, continue waiting for other client processing")
                
                await asyncio.sleep(poll_interval)
        
        raise Exception(f"Task {task_id} ({task_name}) wait timeout ({max_wait_time:.1f}s)")
    
    async def _get_task_result(self, task_id: str, task_name: str) -> str:
        """
        Get task processing result - read local file directly from LIGHTROOM_RESULTS_DIR
        
        Args:
            task_id (str): Task identifier
            task_name (str): Name of the processing task
            
        Returns:
            str: Path to the result file
        """
        task_result_dir = Path(self.results_base_dir) / task_id
        
        # Grace period and polling interval
        grace_wait_seconds = float(os.getenv('LIGHTROOM_RESULT_GRACE_WAIT', '120.0'))
        poll_interval_seconds = float(os.getenv('LIGHTROOM_RESULT_POLL_INTERVAL', '1.0'))
        
        # Search for processing result files
        candidate_files = [
            task_result_dir / "processed",
            task_result_dir / "processed.jpg",
            task_result_dir / "processed.jpeg",
            task_result_dir / f"processed_{task_id}.jpg"
        ]
        
        start_time = time.time()
        while True:
            result_file = None
            for candidate in candidate_files:
                if candidate.exists() and candidate.stat().st_size > 0:
                    result_file = candidate
                    break
            
            if result_file is not None:
                logger.info(f"{task_name} found result file: {result_file}")
                return str(result_file)
            
            elapsed = time.time() - start_time
            if elapsed >= grace_wait_seconds:
                if task_result_dir.exists():
                    files = list(task_result_dir.glob('*'))
                    logger.error(f"{task_name} result directory exists but no valid files: {task_result_dir}")
                    logger.error(f"Directory contents: {[f.name for f in files]}")
                else:
                    logger.error(f"{task_name} result directory does not exist: {task_result_dir}")
                raise Exception(f"Task {task_id} result file does not exist (waited {elapsed:.1f}s)")
            
            await asyncio.sleep(poll_interval_seconds)
