# JarvisArt Lightroom Client

Client that connects to Linux training servers and automatically processes Lightroom image editing tasks.

## Quick Start

### 1. Install Dependencies
```bash
pip install aiohttp fastapi uvicorn requests pillow pyyaml
```

### 2. Install Lightroom Plugin
1. Open Lightroom Classic
2. `File` → `Plug-in Manager`
3. Click `Add`, select `agent_to_lightroom/XMPlayer.lrplugin/` directory

### 3. Start Client
```bash
chmod +x start_mac_client.sh
./start_mac_client.sh
```

Or specify server:
```bash
./start_mac_client.sh --servers "SERVER_IP:PORT"
```

## Main Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--servers` | Server addresses (comma-separated) | `29.142.19.114:8081` |
| `--api-port` | Local API port | `7777` |
| `--poll-interval` | Polling interval (seconds) | `1.0` |

## Troubleshooting

**Connection Failed**:
- Ensure Lightroom is running
- Check server IP and port
- Confirm plugin is installed

**Port Occupied**:
```bash
netstat -an | grep 7878  # Check port
```

**Test Connection**:
```bash
python lr_task_client.py --servers "IP:PORT" --test
```