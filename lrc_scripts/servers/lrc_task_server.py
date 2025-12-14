"""
Reverse Connection Server - Linux server provides task queue
Clients actively connects to fetch tasks
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import time
import json
import os
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional, Any
from uuid import uuid4
from pathlib import Path
import logging
import aiofiles
from io import BytesIO
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lightroom Reverse Connection Service", version="1.0.0")

# Task status enumeration - simple state machine eliminates all special cases
class TaskStatus(str, Enum):
    PENDING = "pending"      # Waiting to be read
    READING = "reading"      # Has been read, waiting for processing
    PROCESSING = "processing" # Processing in progress
    COMPLETED = "completed"   # Processing completed
    FAILED = "failed"        # Processing failed

# Data models
class Task(BaseModel):
    task_id: str
    photo_path: str
    xmp_path: str
    created_at: float
    status: TaskStatus = TaskStatus.PENDING
    client_id: Optional[str] = None
    read_at: Optional[float] = None        # Read timestamp
    read_timeout: float = 10.0             # Read timeout in seconds
    result: Optional[Dict[str, Any]] = None
    file_package_path: Optional[str] = None
    requires_download: bool = False

class TaskResult(BaseModel):
    task_id: str
    client_id: str
    success: bool
    elapsed_time: float
    error: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None

class ClientInfo(BaseModel):
    client_id: str
    client_type: str = "lightroom_bridge"
    capabilities: List[str] = ["photo_processing"]
    status: str = "ready"
    last_seen: float
    local_port: int = 7777

# Global storage - message center core data structures
tasks: Dict[str, Task] = {}
clients: Dict[str, ClientInfo] = {}
completed_tasks: Dict[str, Task] = {}
# Removed task_queue - replaced with state machine to eliminate special cases

# Task status lock - ensures atomic operations
task_lock = asyncio.Lock()

# File transfer storage - use environment variables to configure paths
file_packages: Dict[str, str] = {}  # task_id -> package_path
upload_dir = Path(os.getenv('LIGHTROOM_UPLOAD_DIR', './lr_caches/uploads'))
upload_dir.mkdir(parents=True, exist_ok=True)

# Processing result storage - separate folder for each task
results_dir = Path(os.getenv('LIGHTROOM_RESULTS_DIR', './lr_caches/lightroom_results'))
results_dir.mkdir(parents=True, exist_ok=True)

# Retry configuration
MAX_FILE_WAIT_RETRIES = int(os.getenv('LIGHTROOM_MAX_RETRIES', '5'))
FILE_WAIT_TIMEOUT = float(os.getenv('LIGHTROOM_FILE_WAIT_TIMEOUT', '180.0'))
RETRY_DELAY = float(os.getenv('LIGHTROOM_RETRY_DELAY', '2.0'))
BACKOFF_FACTOR = float(os.getenv('LIGHTROOM_BACKOFF_FACTOR', '1.5'))

async def wait_for_file_with_retries(file_path: str, timeout: float = None) -> bool:
    """
    File waiting function with retry mechanism
    Handles Mac Lightroom transmission delay issues
    """
    if timeout is None:
        timeout = FILE_WAIT_TIMEOUT
    
    start_time = time.time()
    retry_count = 0
    current_delay = RETRY_DELAY
    
    while time.time() - start_time < timeout and retry_count < MAX_FILE_WAIT_RETRIES:
        if Path(file_path).exists():
            # Additional wait after file exists to ensure write completion
            await asyncio.sleep(0.5)
            if Path(file_path).exists() and Path(file_path).stat().st_size > 0:
                logger.info(f"File ready: {file_path} (retry {retry_count} times)")
                return True
        
        logger.debug(f"Waiting for file {file_path}... (retry {retry_count}/{MAX_FILE_WAIT_RETRIES})")
        await asyncio.sleep(current_delay)
        
        # Exponential backoff
        current_delay = min(current_delay * BACKOFF_FACTOR, 30.0)  # Maximum 30 seconds
        retry_count += 1
    
    logger.error(f"File wait timeout: {file_path} (elapsed {time.time() - start_time:.1f}s, retry {retry_count} times)")
    return False

@app.get("/")
async def root():
    return {
        "message": "Lightroom reverse connection service is running (message center mode)",
        "version": "2.0.0",
        "active_clients": len(clients),
        "task_stats": {
            "pending": len([t for t in tasks.values() if t.status == TaskStatus.PENDING]),
            "reading": len([t for t in tasks.values() if t.status == TaskStatus.READING]),
            "processing": len([t for t in tasks.values() if t.status == TaskStatus.PROCESSING]),
            "completed": len(completed_tasks)
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "stats": {
            "active_clients": len(clients),
            "pending_tasks": len([t for t in tasks.values() if t.status == TaskStatus.PENDING]),
            "reading_tasks": len([t for t in tasks.values() if t.status == TaskStatus.READING]),
            "processing_tasks": len([t for t in tasks.values() if t.status == TaskStatus.PROCESSING]),
            "completed_tasks": len(completed_tasks)
        }
    }

@app.post("/api/register_client")
async def register_client(client: ClientInfo):
    """Register Mac client"""
    client.last_seen = time.time()
    clients[client.client_id] = client
    logger.info(f"Client registered: {client.client_id}")
    return {"message": "Registration successful", "client_id": client.client_id}

@app.get("/api/get_task/{client_id}")
async def get_task(client_id: str):
    """Mac client fetches task - atomic operation, no race conditions"""
    # Add debug logging
    logger.info(f"Received task request, client ID: {client_id}")
    logger.info(f"Currently registered clients: {list(clients.keys())}")
    
    if client_id not in clients:
        # Return more explicit error instead of default 404
        logger.warning(f"Client {client_id} not registered")
        raise HTTPException(status_code=403, detail="Client not registered, please register first")
    
    async with task_lock:
        # Update client last active time
        clients[client_id].last_seen = time.time()
        
        # Find earliest created available task - FIFO order ensures fairness
        available_task = None
        earliest_time = float('inf')
        
        for task in tasks.values():
            if task.status == TaskStatus.PENDING and task.created_at < earliest_time:
                available_task = task
                earliest_time = task.created_at
        
        if not available_task:
            return {}  # No available tasks
        
        # Atomic state transition: pending -> reading
        available_task.status = TaskStatus.READING
        available_task.client_id = client_id
        available_task.read_at = time.time()
        
        logger.info(f"Task {available_task.task_id} read by client {client_id}")
        
        return {
            "task_id": available_task.task_id,
            "photo_path": available_task.photo_path,
            "xmp_path": available_task.xmp_path,
            "created_at": available_task.created_at,
            "requires_download": available_task.requires_download,
            "file_package_path": available_task.file_package_path,
            "read_timeout": available_task.read_timeout
        }

class StartProcessingRequest(BaseModel):
    client_id: str

@app.post("/api/start_processing/{task_id}")
async def start_processing(task_id: str, request: StartProcessingRequest):
    """Client confirms start processing task - state transition reading -> processing"""
    async with task_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="Task does not exist")
        
        task = tasks[task_id]
        if task.client_id != request.client_id:
            raise HTTPException(status_code=403, detail="Task does not belong to this client")
        
        if task.status != TaskStatus.READING:
            raise HTTPException(status_code=400, detail=f"Task status error: {task.status}")
        
        # Atomic state transition: reading -> processing
        task.status = TaskStatus.PROCESSING
        logger.info(f"Task {task_id} started processing (client: {request.client_id})")
        
        return {"message": "Processing started", "task_id": task_id}

@app.post("/api/report_result")
async def report_result(result: TaskResult):
    """Mac client reports task result - atomic state transition"""
    async with task_lock:
        task_id = result.task_id
        
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="Task does not exist")
        
        task = tasks[task_id]
        if task.client_id != result.client_id:
            raise HTTPException(status_code=403, detail="Task does not belong to this client")
        
        # Atomic state transition: processing -> completed/failed
        task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        task.result = {
            "success": result.success,
            "elapsed_time": result.elapsed_time,
            "error": result.error,
            "result_data": result.result_data,
            "completed_at": time.time()
        }
        
        # Move to completed queue
        completed_tasks[task_id] = task
        del tasks[task_id]
        
        status = "✅Success" if result.success else "❌Failed"
        logger.info(f"Task {task_id} {status} (elapsed: {result.elapsed_time:.1f}s)")
        
        return {"message": "Result recorded", "task_id": task_id}

@app.post("/api/upload_result")
async def upload_result(task_id: str, processed_image: UploadFile = File(...)):
    """Receive processed result image uploaded by Mac client - with retry wait mechanism"""
    
    if task_id not in completed_tasks:
        raise HTTPException(status_code=404, detail="Task does not exist or not completed")
    
    # Create separate folder for each task
    task_result_dir = results_dir / task_id
    task_result_dir.mkdir(exist_ok=True)
    
    try:
        # Save processed image
        processed_filename = f"processed.jpg"
        processed_path = task_result_dir / processed_filename
        
        # Write to temporary file, then atomically move
        temp_path = processed_path.with_suffix('.tmp')
        
        with open(temp_path, "wb") as f:
            content = await processed_image.read()
            f.write(content)
        
        # Atomically move to final location
        temp_path.rename(processed_path)
        
        # Wait for file to be fully written and readable
        if not await wait_for_file_with_retries(str(processed_path), timeout=30.0):
            raise Exception(f"Upload file verification failed: {processed_path}")
        
        # Update task record
        task = completed_tasks[task_id]
        if not task.result:
            task.result = {}
        
        task.result.update({
            "processed_image_path": str(processed_path),
            "upload_timestamp": time.time(),
            "file_size": processed_path.stat().st_size
        })
        
        logger.info(f"Task {task_id} processing result saved: {processed_path} ({processed_path.stat().st_size} bytes)")
        
        return {
            "message": "Processing result uploaded successfully", 
            "task_id": task_id,
            "saved_path": str(processed_path),
            "result_folder": str(task_result_dir),
            "file_size": processed_path.stat().st_size
        }
        
    except Exception as e:
        # Clean up temporary file
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        
        logger.error(f"Failed to save processing result {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

@app.post("/api/submit_task_with_files")
async def submit_task_with_files(photo_path: str, xmp_path: str):
    """Submit task requiring file transfer - efficient direct transfer"""
    task_id = str(uuid4())
    
    # Check if files exist
    if not Path(photo_path).exists():
        raise HTTPException(status_code=404, detail=f"Photo file does not exist: {photo_path}")
    if not Path(xmp_path).exists():
        raise HTTPException(status_code=404, detail=f"XMP file does not exist: {xmp_path}")
    
    # Directly store source file paths, no packaging needed
    task = Task(
        task_id=task_id,
        photo_path=f"/tmp/lightroom_task_{task_id}/before.jpg",  # Mac local path
        xmp_path=f"/tmp/lightroom_task_{task_id}/config.lua",   # Mac local path
        created_at=time.time(),
        requires_download=True,
        file_package_path=None  # No packaging used
    )
    
    tasks[task_id] = task
    
    # Store source file paths for download use
    file_packages[task_id] = {
        "photo_path": photo_path,
        "xmp_path": xmp_path
    }
    
    logger.info(f"Created efficient transfer task: {task_id}")
    return {"message": "Task submitted", "task_id": task_id}

@app.get("/api/download_file/{task_id}/{file_type}")
async def download_file(task_id: str, file_type: str):
    """Download task files - efficient direct transfer"""
    if task_id not in file_packages:
        raise HTTPException(status_code=404, detail="Task files do not exist")
    
    file_paths = file_packages[task_id]
    
    if file_type == "photo":
        file_path = file_paths["photo_path"]
        filename = "before.jpg"
        media_type = "image/jpeg"
    elif file_type == "xmp":
        file_path = file_paths["xmp_path"] 
        filename = "config.lua"
        media_type = "text/plain"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type, supported: photo, xmp")
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"File does not exist: {file_path}")
    
    def iterfile(path: str):
        with open(path, mode="rb") as file_like:
            yield from file_like
    
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(iterfile(file_path), media_type=media_type, headers=headers)

# Efficient transfer, no packaging function needed

@app.post("/api/submit_task")
async def submit_task(photo_path: str, xmp_path: str):
    """Submit new task"""
    task_id = str(uuid4())
    task = Task(
        task_id=task_id,
        photo_path=photo_path,
        xmp_path=xmp_path,
        created_at=time.time()
    )
    
    tasks[task_id] = task
    
    logger.info(f"New task submitted: {task_id}")
    logger.info(f"  Photo: {Path(photo_path).name}")
    logger.info(f"  XMP: {Path(xmp_path).name}")
    
    return {
        "message": "Task submitted",
        "task_id": task_id,
        "queue_position": len([t for t in tasks.values() if t.status == TaskStatus.PENDING])
    }

@app.get("/api/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Query task status"""
    # Check in-progress tasks
    if task_id in tasks:
        task = tasks[task_id]
        return {
            "task_id": task_id,
            "status": task.status,
            "created_at": task.created_at,
            "client_id": task.client_id,
            "result": task.result
        }
    
    # Check completed tasks
    if task_id in completed_tasks:
        task = completed_tasks[task_id]
        return {
            "task_id": task_id,
            "status": task.status,
            "created_at": task.created_at,
            "client_id": task.client_id,
            "result": task.result
        }
    
    raise HTTPException(status_code=404, detail="Task does not exist")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    active_clients = [
        {
            "client_id": client_id,
            "last_seen": info.last_seen,
            "status": info.status,
            "capabilities": info.capabilities
        }
        for client_id, info in clients.items()
        if time.time() - info.last_seen < 60  # Active within 60 seconds
    ]
    
    return {
        "active_clients": active_clients,
        "queue_stats": {
            "pending": len([t for t in tasks.values() if t.status == TaskStatus.PENDING]),
            "processing": len([t for t in tasks.values() if t.status == TaskStatus.PROCESSING]),
            "completed_today": len(completed_tasks)
        },
        "recent_tasks": [
            {
                "task_id": task.task_id,
                "status": task.status,
                "created_at": task.created_at,
                "photo": Path(task.photo_path).name
            }
            for task in list(completed_tasks.values())[-10:]  # Last 10 tasks
        ]
    }

@app.get("/api/clients")
async def list_clients():
    """List all clients"""
    current_time = time.time()
    return {
        "clients": [
            {
                "client_id": client_id,
                "status": "online" if current_time - info.last_seen < 30 else "offline",
                "last_seen": info.last_seen,
                "capabilities": info.capabilities,
                "local_port": info.local_port
            }
            for client_id, info in clients.items()
        ]
    }

# Background task for cleaning up expired tasks
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_tasks())

async def cleanup_old_tasks():
    """Periodically clean up expired tasks and timeout reads - Linus-style concise logic"""
    while True:
        try:
            current_time = time.time()
            
            async with task_lock:
                # Clean up timed-out READING status tasks - reset to pending for other clients to execute
                reading_timeout_tasks = []
                for task_id, task in tasks.items():
                    if (task.status == TaskStatus.READING and 
                        task.read_at and 
                        current_time - task.read_at > task.read_timeout):
                        reading_timeout_tasks.append(task_id)
                
                for task_id in reading_timeout_tasks:
                    task = tasks[task_id]
                    logger.warning(f"Task {task_id} read timeout, reset to pending (client: {task.client_id})")
                    # Atomic state transition: reading -> pending
                    task.status = TaskStatus.PENDING
                    task.client_id = None
                    task.read_at = None
                
                # Clean up long-running PROCESSING tasks (30 minutes)
                processing_timeout_tasks = []
                for task_id, task in tasks.items():
                    if (task.status == TaskStatus.PROCESSING and 
                        current_time - task.created_at > 1800):
                        processing_timeout_tasks.append(task_id)
                
                for task_id in processing_timeout_tasks:
                    task = tasks[task_id]
                    logger.warning(f"Task {task_id} processing timeout, reset to pending (client: {task.client_id})")
                    # Atomic state transition: processing -> pending
                    task.status = TaskStatus.PENDING
                    task.client_id = None
                    task.read_at = None
            
            # Clean up offline clients (10 minutes) - no lock needed, simple dict operation
            offline_clients = [
                client_id for client_id, info in clients.items()
                if current_time - info.last_seen > 600
            ]
            
            for client_id in offline_clients:
                logger.info(f"Cleaning up offline client: {client_id}")
                del clients[client_id]
            
            await asyncio.sleep(30)  # Check every 30 seconds for more timely timeout handling
            
        except Exception as e:
            logger.error(f"Cleanup task exception: {e}")
            await asyncio.sleep(60)



@app.get("/api/download_task_result/{task_id}")
async def download_task_result(task_id: str):
    """Download task processing result image"""
    
    # First check completed tasks
    if task_id not in completed_tasks:
        raise HTTPException(status_code=404, detail="Task does not exist or not completed")
    
    task = completed_tasks[task_id]
    
    # Check if task has result
    if not task.result or "processed_image_path" not in task.result:
        raise HTTPException(status_code=404, detail="Task result does not exist")
    
    processed_path = task.result["processed_image_path"]
    
    # Check if file exists
    if not Path(processed_path).exists():
        raise HTTPException(status_code=404, detail=f"Result file does not exist: {processed_path}")
    
    # Stream file
    def iterfile(path: str):
        with open(path, mode="rb") as file_like:
            yield from file_like
    
    # Get file extension to determine media type
    file_ext = Path(processed_path).suffix.lower()
    media_type = "image/jpeg" if file_ext in ['.jpg', '.jpeg'] else "application/octet-stream"
    
    headers = {
        "Content-Disposition": f"attachment; filename=processed_{task_id}.jpg"
    }
    
    logger.info(f"Downloading task result: {task_id} -> {processed_path}")
    return StreamingResponse(iterfile(processed_path), media_type=media_type, headers=headers)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Lightroom Reverse Connection Server')
    parser.add_argument('--host', default='0.0.0.0', help='Listen address')
    parser.add_argument('--port', type=int, default=8081, help='Listen port')
    
    args = parser.parse_args()
    
    print("🚀 Starting Lightroom Reverse Connection Server")
    print("=" * 40)
    print(f"Listening: {args.host}:{args.port}")
    print("API Documentation: http://localhost:8081/docs")
    print("=" * 40)
    
    uvicorn.run(app, host=args.host, port=args.port)