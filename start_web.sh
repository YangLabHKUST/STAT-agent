#!/bin/bash

# Startup script for STAT web interface + Jupyter Lab

WEB_PORT=${1:-8889}
JUPYTER_PORT=${2:-8890}

# Ensure HTTPS proxy is set (needed for POE/external LLM APIs)
export https_proxy="${https_proxy:-${http_proxy:-}}"

echo "========================================"
echo "STAT - Spatial Transcriptomics Analysis"
echo "========================================"
echo ""
echo "Starting services:"
echo "  Web UI:       http://0.0.0.0:${WEB_PORT}"
echo "  Jupyter Lab:  http://0.0.0.0:${JUPYTER_PORT}"
echo ""
echo "Press Ctrl+C to stop all services"
echo "========================================"
echo ""

# cd /import/home3/yhchenmath/Code/STAT-agent

# Session logs and notebooks live here; Jupyter is rooted at the same directory so notebook links resolve
export STAT_LOGS_DIR="${STAT_LOGS_DIR:-$PWD/logs}"
mkdir -p "${STAT_LOGS_DIR}"

# Start Jupyter Lab in background (no token, no browser, allow iframe embedding)
jupyter lab \
    --port=${JUPYTER_PORT} \
    --ip=0.0.0.0 \
    --no-browser \
    --notebook-dir="${STAT_LOGS_DIR}" \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.tornado_settings='{"headers":{"Content-Security-Policy":"frame-ancestors *","X-Frame-Options":""}}' \
    --ServerApp.allow_origin='*' \
    &
JUPYTER_PID=$!

# Cleanup on exit
trap "kill $JUPYTER_PID 2>/dev/null; exit" INT TERM EXIT

# Start the STAT web server (Starlette + uvicorn, foreground)
python3 -m stat_agent.web --host 0.0.0.0 --port ${WEB_PORT} --jupyter-port ${JUPYTER_PORT}
