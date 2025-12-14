# Lightroom Reverse Connection System

A distributed task processing system that enables Linux training servers to distribute Lightroom image processing tasks to multiple client machines.

## 📖 System Overview

This system uses a **reverse connection architecture** where clients actively connect to Linux servers to fetch tasks, solving the traditional challenge of servers accessing client machines.

### Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Linux Server   │         │   Client 1       │
│                 │         │                  │
│  lrc_task_      │◄────────┤ lr_task_client   │
│  server.py      │  Polling│  + Lightroom     │
│                 │         │                  │
│  Task Queue     │         └──────────────────┘
│  State Machine  │
│                 │         ┌──────────────────┐
│                 │◄────────┤   Client 2       │
│                 │  Polling│                  │
│                 │         │ lr_task_client   │
│                 │         │  + Lightroom     │
└─────────────────┘         └──────────────────┘
```

### Core Features

- ✅ **Reverse Connection** - Clients actively connect to servers, no port forwarding needed
- ✅ **Multi-Server Support** - Clients can connect to multiple Linux servers simultaneously
- ✅ **State Machine** - Complete state management for atomic task processing
- ✅ **Auto-Reconnect** - Automatic reconnection during network interruptions
- ✅ **Load Balancing** - Fair polling of multiple servers for load distribution
- ✅ **Smart Timeout** - Dynamic timeout adjustment based on task complexity
- ✅ **File Transfer** - Support for task file download and result upload

## 📁 Directory Structure

```
lrc_scripts/
├── README.md                    # This document
├── servers/                     # Server components
│   ├── README.md                # Server documentation
│   ├── lrc_task_server.py       # FastAPI task server
│   └── start_reverse_server.sh  # Server startup script
└── clients/                     # Client components
    ├── README.md                # Client documentation
    ├── lr_task_client.py        # Reverse connection client
    ├── start_mac_client.sh      # Client startup script
    └── agent_to_lightroom/      # Lightroom API integration
        └── ...
```

## 🚀 Quick Start

### Prerequisites

#### Server Side (Linux)
- Python 3.7+
- FastAPI, uvicorn, aiofiles
- Sufficient disk space for task files and results

#### Client Side (Mac)
- Python 3.7+
- Adobe Lightroom (must be running)
- aiohttp
- Network connection to Linux servers

### Installation

#### Server
```bash
cd servers
pip install fastapi uvicorn aiofiles pydantic
```

#### Client
```bash
cd clients
pip install aiohttp requests
```

### Startup Process

#### 1. Start Linux Server
```bash
cd lrc_scripts/servers
chmod +x start_reverse_server.sh
./start_reverse_server.sh
```

Server options:
- `--host` - Listen address (default: `0.0.0.0`)
- `--port` - Listen port (default: `8081`)
- `--upload-dir` - Upload directory
- `--results-dir` - Results directory
- See [Server Documentation](servers/README.md) for more options

#### 2. Start Mac Client
```bash
cd lrc_scripts/clients
chmod +x start_mac_client.sh
./start_mac_client.sh
```

Client options:
- `--servers` - Server addresses (format: `IP1:PORT1,IP2:PORT2`)
- `--api-port` - Local Lightroom API port (default: `7777`)
- `--api-path` - API_Lightroom project path
- See [Client Documentation](clients/README.md) for more options

## 📊 State Machine Flow

The system uses a state machine to manage task processing:

```
PENDING → READING → PROCESSING → COMPLETED/FAILED
```

- **PENDING**: Task waiting in queue
- **READING**: Task read by client, awaiting processing confirmation
- **PROCESSING**: Task being processed by client
- **COMPLETED/FAILED**: Task processing finished

## 📋 Task Processing Flow

```
1. Submit task (Linux server)
   ↓
2. Task enters queue (state: PENDING)
   ↓
3. Client polls and fetches task (state: PENDING → READING)
   ↓
4. Client confirms processing start (state: READING → PROCESSING)
   ↓
5. Client downloads task files (if needed)
   ↓
6. Client processes image with Lightroom
   ↓
7. Client uploads processed result
   ↓
8. Client reports result (state: PROCESSING → COMPLETED/FAILED)
   ↓
9. Task completed, result stored on server
```

## 🔌 API Endpoints

### Server Endpoints

#### Core Endpoints
- `POST /api/submit_task` - Submit task (files already accessible)
- `POST /api/submit_task_with_files` - Submit task (with file transfer)
- `GET /api/task_status/{task_id}` - Query task status
- `GET /api/download_task_result/{task_id}` - Download task result

#### Client Communication
- `POST /api/register_client` - Register client
- `GET /api/get_task/{client_id}` - Client fetches task
- `POST /api/start_processing/{task_id}` - Client confirms processing
- `POST /api/report_result` - Client reports result

### Client Features

- Multi-server polling with fair distribution
- Automatic reconnection on network failure
- Dynamic timeout adjustment based on task complexity
- Health checks and statistics reporting
- File download and upload support

## 📚 Documentation

- [Server Documentation](servers/README.md) - Detailed server configuration and API
- [Client Documentation](clients/README.md) - Client configuration and usage

## 🛠️ Troubleshooting

Common issues:
- **Client can't connect**: Check server status, network, firewall settings
- **Lightroom API not starting**: Ensure Lightroom is running, check port 7777
- **Task processing timeout**: Increase processing timeout, check task complexity
- **File upload failure**: Check network connection, server disk space

## 📝 Notes

- The system is designed for JarvisEvo's distributed Lightroom image processing
- Requires Lightroom and corresponding API plugin
- Server handles task queuing, state management, and file storage
- Client handles connection, task processing, and result reporting