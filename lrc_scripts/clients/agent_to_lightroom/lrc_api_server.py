import socket
import os
import time
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

class LightroomAPI:
    def __init__(self, host="127.0.0.1", port=7878):
        self.host = host
        self.port = port
        self.socket = None
        self.response_thread = None
        self.connected = False
        self.last_output_path = None  # 存储最后处理的图片输出路径
    
    def connect(self):
        """建立与服务器的连接"""
        if self.connected:
            return True
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(None)  # 禁用超时
            # 设置 TCP keepalive
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # 在 Linux 系统上设置更多的 keepalive 参数
            if hasattr(socket, 'TCP_KEEPIDLE'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60)
            if hasattr(socket, 'TCP_KEEPCNT'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
            
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # 启动响应处理线程
            self.response_thread = threading.Thread(target=self._handle_responses)
            self.response_thread.daemon = True
            self.response_thread.start()
            
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            return False
    
    def _handle_responses(self):
        """处理服务器响应"""
        while self.connected:
            try:
                response = self.socket.recv(4096).decode().strip()
                if not response:
                    print("Empty response from server")
                    break
                
                print(f"Received response: {response}")
                status, *message = response.split('|')
                message = '|'.join(message) if message else ''
                
                if status == "success":
                    print(f"\nPhoto processed successfully!")
                    if message:
                        self.last_output_path = message  # 保存输出路径
                        print(f"Output saved to: {message}")
                elif status == "error":
                    self.last_output_path = None
                    print(f"\nError: {message}")
                elif status == "pong":
                    print("Server is alive (received pong)")
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error in response handler: {e}")
                self.connected = False
                break
    
    def send_request(self, message):
        """发送请求到服务器"""
        if not self.connected and not self.connect():
            return None
            
        try:
            self.socket.sendall((message + "\n").encode())
            return True
        except Exception as e:
            print(f"Error sending request: {e}")
            self.connected = False
            return None
    
    def process_photo(self, photo_path, xmp_path, output_dir=None):
        """处理照片"""
        # 确保文件存在
        if not os.path.exists(photo_path):
            raise FileNotFoundError(f"Photo file not found: {photo_path}")
        if not os.path.exists(xmp_path):
            raise FileNotFoundError(f"XMP preset file not found: {xmp_path}")
        
        # Lightroom 实际导出位置：原片所在目录下的 processed 子目录
        photo_path_obj = Path(photo_path).absolute()
        expected_output_dir = photo_path_obj.parent / "processed"
        
        # 清空之前的输出路径
        self.last_output_path = None
        
        # 发送给插件的消息（插件当前忽略自定义输出目录参数，但保留以兼容）
        message = f"process|{str(photo_path_obj)}|{str(Path(xmp_path).absolute())}|{str(expected_output_dir)}"
        
        result = self.send_request(message)
        if result:
            # 等待一小段时间让Lightroom处理完成
            # import time
            # time.sleep(5)  # 增加等待时间
            
            # 如果Lightroom插件没有返回路径，尝试在实际导出目录中定位文件
            if not self.last_output_path:
                import glob
                stem = photo_path_obj.stem
                # 优先尝试与原文件同名的导出文件
                preferred_candidates = [
                    expected_output_dir / f"{stem}.jpg",
                    expected_output_dir / f"{stem}.jpeg",
                    expected_output_dir / f"{stem}.tif",
                    expected_output_dir / f"{stem}.tiff",
                ]
                for candidate in preferred_candidates:
                    if candidate.exists():
                        self.last_output_path = str(candidate)
                        print(f"Found expected output file: {self.last_output_path}")
                        break
                # 如果未直接命中，回退到该目录下最新的图片文件
                if not self.last_output_path:
                    possible_patterns = [
                        str(expected_output_dir / "*.jpg"),
                        str(expected_output_dir / "*.jpeg"),
                        str(expected_output_dir / "*.tif"),
                        str(expected_output_dir / "*.tiff"),
                    ]
                    matches = []
                    for pattern in possible_patterns:
                        matches.extend(glob.glob(pattern))
                    if matches:
                        self.last_output_path = max(matches, key=lambda x: Path(x).stat().st_mtime)
                        print(f"Found output file by latest mtime: {self.last_output_path}")
                # 如果还是没找到，作为最后兜底，返回预期命名路径（不一定存在）
                if not self.last_output_path:
                    self.last_output_path = str(expected_output_dir / f"{stem}.jpg")
                    print(f"Using expected (fallback) output path: {self.last_output_path}")
        
        return result, self.last_output_path
    
    def close(self):
        """关闭连接"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None

def test_server_connection(api):
    """测试服务器连接"""
    if not api.connect():
        return False
        
    # 发送 ping 请求
    result = api.send_request("ping")
    if result:
        print("Server is running")
        return True
    
    print("Could not connect to server")
    return False

class PhotoProcessHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        photo_path = data.get('photo_path')
        xmp_path = data.get('xmp_path')
        
        if not photo_path or not xmp_path:
            self.send_error(400, "Missing photo_path or xmp_path")
            return
            
        try:
            # 为每个任务创建专用的输出目录 - Mac本地路径
            task_id = data.get('task_id', f"task_{int(time.time())}")
            import platform
            if platform.system() == "Darwin":  # macOS
                output_dir = f"/tmp/lightroom_mac_processed/{task_id}"
            else:
                output_dir = f"/tmp/lightroom_processed/{task_id}"
            
            result, output_path = self.server.lightroom_api.process_photo(photo_path, xmp_path, output_dir)
            if result:
                response_data = {
                    "status": "success",
                    "output_path": output_path,
                    "task_id": task_id
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())
            else:
                self.send_error(500, "Failed to process photo")
        except Exception as e:
            self.send_error(500, str(e))

def run_http_server(api, port=7777):
    """运行HTTP服务器"""
    server = HTTPServer(('localhost', port), PhotoProcessHandler)
    server.lightroom_api = api  # 将API实例附加到服务器
    print(f"Starting HTTP server on port {port}")
    server.serve_forever()

def main():
    # 创建API客户端
    api = LightroomAPI()
    
    try:
        # 测试连接
        if not test_server_connection(api):
            print("Error: Could not connect to the Lightroom plugin server")
            return
            
        # 启动HTTP服务器
        http_thread = threading.Thread(target=run_http_server, args=(api,))
        http_thread.daemon = True
        http_thread.start()
        
        print("\nHTTP server is running. You can now send POST requests to process photos.")
        print("Example POST request to http://localhost:7777:")
        print('''
        {
            "photo_path": "path/to/photo.dng",
            "xmp_path": "path/to/preset.xmp"
        }
        ''')
        
        # 保持主程序运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        api.close()

if __name__ == "__main__":
    main() 