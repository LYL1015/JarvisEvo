import requests
import json
from PIL import Image
import yaml
from pathlib import Path
import os
from urllib.parse import urlparse

# Load configuration
def load_config():
    config_path = Path(__file__).parent / "config" / "config.yaml"
    paths_path = Path(__file__).parent / "config" / "paths.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(paths_path, 'r') as f:
        paths = yaml.safe_load(f)
    
    return config, paths

config, paths = load_config()

def is_local_server(url):
    """Check if URL points to local server"""
    parsed = urlparse(url)
    hostname = parsed.hostname or parsed.netloc.split(':')[0]
    return hostname.lower() in ['localhost', '127.0.0.1', '::1']

def send_photo_request_local(photo_path, lua_path, url, timeout):
    """Send request with file paths (local mode)"""
    payload = {
        "photo_path": photo_path,
        "lua_path": lua_path
    }
    
    print("Sending request in local mode (file paths only)...")
    return requests.post(url, json=payload, timeout=timeout)

def send_photo_request_remote(photo_path, lua_path, url, timeout):
    """Send request with file uploads (remote mode)"""
    # Validate files exist
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"Photo file not found: {photo_path}")
    if not os.path.exists(lua_path):
        raise FileNotFoundError(f"Lua file not found: {lua_path}")
    
    # Get file sizes for progress tracking
    photo_size = os.path.getsize(photo_path)
    lua_size = os.path.getsize(lua_path)
    total_size = photo_size + lua_size
    
    print(f"Sending request in remote mode (uploading files)...")
    print(f"Photo: {os.path.basename(photo_path)} ({photo_size:,} bytes)")
    print(f"Lua: {os.path.basename(lua_path)} ({lua_size:,} bytes)")
    print(f"Total upload size: {total_size:,} bytes")
    
    # Prepare multipart form data
    files = {
        'photo_file': (os.path.basename(photo_path), open(photo_path, 'rb')),
        'lua_file': (os.path.basename(lua_path), open(lua_path, 'rb'))
    }
    
    data = {
        'mode': 'upload',
        'photo_filename': os.path.basename(photo_path),
        'lua_filename': os.path.basename(lua_path)
    }
    
    try:
        response = requests.post(url, files=files, data=data, timeout=timeout)
        return response
    finally:
        # Always close file handles
        for file_tuple in files.values():
            if hasattr(file_tuple[1], 'close'):
                file_tuple[1].close()

def send_photo_request(photo_path, lua_path, url=None, timeout=None):
    """Main function that automatically chooses local or remote mode"""
    url = url or f"http://{config['http']['host']}:{config['http']['port']}"
    timeout = timeout or config['http']['timeout']
    
    try:
        # Determine mode based on URL
        if is_local_server(url):
            response = send_photo_request_local(photo_path, lua_path, url, timeout)
        else:
            response = send_photo_request_remote(photo_path, lua_path, url, timeout)
        
        # Check response
        if response.status_code == 200:
            print("Success! Server response:", response.json())
        else:
            print(f"Error: Status code {response.status_code}")
            print("Error message:", response.text)
            
    except (FileNotFoundError, ValueError) as e:
        print(f"File error: {str(e)}")
    except requests.exceptions.RequestException as e:
        if "Connection" in str(e):
            print("Connection error: Please ensure server is running")
        else:
            print(f"Request error: {str(e)}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def main():
    # Load example paths from configuration
    photo_path = paths['examples']['photo_path']
    lua_path = paths['examples']['preset_path']
    
    # You can override URL here for testing remote mode
    # For example: send_photo_request(photo_path, lua_path, "http://192.168.1.100:7777")
    # For local testing with upload mode: send_photo_request(photo_path, lua_path, "http://127.0.0.1:7777")
    send_photo_request(photo_path, lua_path)


if __name__ == "__main__":
    main()