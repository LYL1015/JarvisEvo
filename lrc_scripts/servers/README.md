# Lightroom Task Distribution Server

FastAPI server for distributing Lightroom editing tasks to client machines.

## 🚀 Quick Start

```bash
./start_reverse_server.sh                    # Default settings
./start_reverse_server.sh --port 9000        # Custom port
./start_reverse_server.sh --help             # View all options
```

## ⚙️ Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host` | `-h` | Listen address | `0.0.0.0` |
| `--port` | `-p` | Listen port | `8081` |
| `--script` | `-s` | Python script path | `./lrc_task_server.py` |
| `--upload-dir` | `-u` | Upload directory | See script |
| `--results-dir` | `-r` | Results directory | See script |
| `--max-retries` | `-m` | Maximum retries | `5` |
| `--wait-timeout` | `-w` | File wait timeout (s) | `180.0` |

## 📊 State Machine

```
PENDING → READING → PROCESSING → COMPLETED/FAILED
```

## 🔌 API Endpoints

| Category | Endpoint | Description |
|----------|----------|-------------|
| Core | `POST /api/register_client` | Register client |
| Core | `GET /api/get_task/{client_id}` | Fetch task |
| Core | `POST /api/start_processing/{task_id}` | Confirm processing |
| Core | `POST /api/report_result` | Report result |
| Core | `POST /api/submit_task` | Submit task |
| File | `GET /api/download_file/{task_id}/{file_type}` | Download file |
| File | `POST /api/upload_result` | Upload result |
| Monitor | `GET /api/health` | Health check |
| Monitor | `GET /api/stats` | Statistics |

## 🔍 Verification

- API docs: `http://localhost:PORT/docs`
- Health check: `http://localhost:PORT/api/health`

## 📚 Related

- [Client README](../clients/README.md)
