#!/bin/bash
# Start training in screen with logging
#
# Usage (from repo root):
#   ./scripts/start_training.sh
#
# This will:
# - Create a screen session named "training"
# - Run training with output to timestamped log file
# - Detach automatically (you can logout safely)
#
# Monitor:
#   screen -r training          # Reattach to see live output
#   tail -f logs/baseline_1m_fixed/training_*.log   # Follow log
#
# Stop:
#   screen -r training          # Reattach
#   Ctrl+C                      # Stop training
#   exit                        # Exit screen

set -e  # Exit on error

# Make sure we're running from repo root
cd "$(dirname "$0")/.."

# Configuration
SESSION_NAME="training"
CONFIG_FILE="experiments/baseline_1m/config_fixed.yaml"
LOG_DIR="logs/baseline_1m_fixed"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Create log directory
mkdir -p ${LOG_DIR}

# Check if screen session already exists
if screen -list | grep -q "${SESSION_NAME}"; then
    echo "❌ Screen session '${SESSION_NAME}' already exists!"
    echo ""
    echo "Options:"
    echo "  screen -r ${SESSION_NAME}              # Reattach to existing session"
    echo "  screen -X -S ${SESSION_NAME} quit      # Kill existing session"
    echo ""
    exit 1
fi

# Training command with output redirection
TRAIN_CMD="uv run python scripts/train_from_config.py ${CONFIG_FILE} 2>&1 | tee ${LOG_FILE}"

echo "======================================================================"
echo "Starting Training"
echo "======================================================================"
echo "Config:  ${CONFIG_FILE}"
echo "Log:     ${LOG_FILE}"
echo "Session: ${SESSION_NAME}"
echo "======================================================================"
echo ""
echo "Training will run in background screen session."
echo "You can safely logout and it will continue running."
echo ""
echo "Monitor progress:"
echo "  screen -r ${SESSION_NAME}       # Reattach (Ctrl+A D to detach)"
echo "  tail -f ${LOG_FILE}             # Follow log file"
echo ""
echo "Stop training:"
echo "  screen -r ${SESSION_NAME}       # Reattach"
echo "  Ctrl+C                          # Stop training"
echo ""
echo "======================================================================"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Start screen session with training command
screen -dmS ${SESSION_NAME} bash -c "${TRAIN_CMD}"

# Wait a moment and verify
sleep 2

if screen -list | grep -q "${SESSION_NAME}"; then
    echo "✅ Training started successfully!"
    echo ""
    echo "Check logs at: ${LOG_FILE}"
    echo ""
else
    echo "❌ Failed to start screen session"
    exit 1
fi
