#!/bin/bash
set -e

# Setup PATH for Python venv
export PATH="/opt/venv/bin:${PATH}"

# Ensure workspace directories exist (may be on a bind-mounted volume)
mkdir -p /workspace/training/configs /workspace/training/outputs /workspace/training/logs

# Start supervisord in foreground (supervisord runs as PID 1)
exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
