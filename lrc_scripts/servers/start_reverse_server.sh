#!/bin/bash
# Lightroom Reverse Connection Server Startup Script
# Used to control all customizable parameters

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default parameter values
HOST="0.0.0.0"
PORT=8081
UPLOAD_DIR="${SCRIPT_DIR}/lr_caches/uploads"
RESULTS_DIR="${SCRIPT_DIR}/lr_caches/results"
PY_SCRIPT="${SCRIPT_DIR}/lrc_task_server.py"
MAX_RETRIES=5
FILE_WAIT_TIMEOUT=180.0
RETRY_DELAY=2.0
BACKOFF_FACTOR=1.5

# Display help information
show_help() {
    echo "Lightroom Reverse Connection Server Startup Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST              Specify listen address (default: $HOST)"
    echo "  -p, --port PORT              Specify listen port (default: $PORT)"
    echo "  -s, --script PATH            Specify Python script path (default: $PY_SCRIPT)"
    echo "  -u, --upload-dir DIR         Set upload file storage directory (default: $UPLOAD_DIR)"
    echo "  -r, --results-dir DIR        Set result file storage directory (default: $RESULTS_DIR)"
    echo "  -m, --max-retries NUM        Set maximum retry count (default: $MAX_RETRIES)"
    echo "  -w, --wait-timeout SEC       Set file wait timeout in seconds (default: $FILE_WAIT_TIMEOUT)"
    echo "  -d, --retry-delay SEC        Set retry delay in seconds (default: $RETRY_DELAY)"
    echo "  -b, --backoff-factor NUM     Set backoff factor (default: $BACKOFF_FACTOR)"
    echo "  --help                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --port 8082 --max-retries 10 --wait-timeout 300"
    echo "  $0 --host 127.0.0.1 --port 9000 --upload-dir /custom/upload/path"
    echo "  $0 --script /path/to/custom_server.py --port 8081"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--host)
            if [[ -z "$2" ]]; then
                echo "Error: --host requires a value"
                show_help
                exit 1
            fi
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            if [[ -z "$2" ]]; then
                echo "Error: --port requires a value"
                show_help
                exit 1
            fi
            PORT="$2"
            shift 2
            ;;
        -s|--script)
            if [[ -z "$2" ]]; then
                echo "Error: --script requires a value"
                show_help
                exit 1
            fi
            PY_SCRIPT="$2"
            shift 2
            ;;
        -u|--upload-dir)
            if [[ -z "$2" ]]; then
                echo "Error: --upload-dir requires a value"
                show_help
                exit 1
            fi
            UPLOAD_DIR="$2"
            shift 2
            ;;
        -r|--results-dir)
            if [[ -z "$2" ]]; then
                echo "Error: --results-dir requires a value"
                show_help
                exit 1
            fi
            RESULTS_DIR="$2"
            shift 2
            ;;
        -m|--max-retries)
            if [[ -z "$2" ]]; then
                echo "Error: --max-retries requires a value"
                show_help
                exit 1
            fi
            MAX_RETRIES="$2"
            shift 2
            ;;
        -w|--wait-timeout)
            if [[ -z "$2" ]]; then
                echo "Error: --wait-timeout requires a value"
                show_help
                exit 1
            fi
            FILE_WAIT_TIMEOUT="$2"
            shift 2
            ;;
        -d|--retry-delay)
            if [[ -z "$2" ]]; then
                echo "Error: --retry-delay requires a value"
                show_help
                exit 1
            fi
            RETRY_DELAY="$2"
            shift 2
            ;;
        -b|--backoff-factor)
            if [[ -z "$2" ]]; then
                echo "Error: --backoff-factor requires a value"
                show_help
                exit 1
            fi
            BACKOFF_FACTOR="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown parameter $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate Python script exists
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Error: Python script not found: $PY_SCRIPT"
    exit 1
fi

# Create necessary directories
mkdir -p "$UPLOAD_DIR"
mkdir -p "$RESULTS_DIR"

# Set environment variables
export LIGHTROOM_UPLOAD_DIR="$UPLOAD_DIR"
export LIGHTROOM_RESULTS_DIR="$RESULTS_DIR"
export LIGHTROOM_MAX_RETRIES="$MAX_RETRIES"
export LIGHTROOM_FILE_WAIT_TIMEOUT="$FILE_WAIT_TIMEOUT"
export LIGHTROOM_RETRY_DELAY="$RETRY_DELAY"
export LIGHTROOM_BACKOFF_FACTOR="$BACKOFF_FACTOR"

# Display configuration information
echo "🚀 Starting Lightroom Reverse Connection Server"
echo "========================================"
echo "Listen Address: $HOST"
echo "Listen Port: $PORT"
echo "Python Script: $PY_SCRIPT"
echo "Upload Directory: $UPLOAD_DIR"
echo "Results Directory: $RESULTS_DIR"
echo "Max Retries: $MAX_RETRIES"
echo "Wait Timeout: $FILE_WAIT_TIMEOUT seconds"
echo "Retry Delay: $RETRY_DELAY seconds"
echo "Backoff Factor: $BACKOFF_FACTOR"
echo "========================================"

# Start the server
python "$PY_SCRIPT" --host "$HOST" --port "$PORT"
