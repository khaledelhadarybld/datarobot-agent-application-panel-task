#!/bin/sh
# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configure UV package manager
export UV_PROJECT=${CODE_DIR}

VENV2_PATH="/opt/venv"
if [ ! -f "${VENV2_PATH}/bin/activate" ]; then
    uv venv "${VENV2_PATH}"
fi
. "${VENV2_PATH}/bin/activate"

# Sync dependencies using UV
# --frozen: Skip dependency resolution, use exact versions from lock file
# --extra: Install the 'agentic_playground' optional dependency group
uv sync --frozen --active --no-progress --color never --extra agentic_playground || true

# Optional: Dump environment variables for debugging
if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = "1" ]; then
    echo "Environment variables:"
    env
fi

echo "Starting Custom Model environment with DRUM prediction server"

# Start DRUM server
echo
echo "Executing command: drum server $*"
echo
exec drum server "$@"
