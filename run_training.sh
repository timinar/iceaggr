#!/bin/bash
# Script to run training in screen with logging
#
# Usage:
#   ./run_training.sh
#
# The script will:
# 1. Start a screen session named "training"
# 2. Run training with output logged to timestamped file
# 3. Detach automatically so you can logout
#
# To reattach: screen -r training
# To detach: Ctrl+A then D
# To kill: screen -X -S training quit

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/baseline_1m_fixed"
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Create log directory if needed
mkdir -p ${LOG_DIR}

# Training command
TRAIN_CMD="uv run python scripts/train_from_config.py experiments/baseline_1m/config_fixed.yaml"

echo "Starting training in screen session 'training'"
echo "Log file: ${LOG_FILE}"
echo ""
echo "To monitor:"
echo "  screen -r training     # Reattach to session"
echo "  tail -f ${LOG_FILE}    # Follow log file"
echo ""
echo "To stop:"
echo "  screen -X -S training quit"
echo ""

# Start screen session
# -dmS: detached, named session
# -L: enable logging
# -Logfile: specify log file
screen -dmS training -L -Logfile "${LOG_FILE}" bash -c "${TRAIN_CMD}"

# Give it a moment to start
sleep 2

# Check if screen session exists
if screen -list | grep -q "training"; then
    echo "✅ Training started successfully!"
    echo ""
    echo "Screen session 'training' is running."
    echo "Reattach with: screen -r training"
    echo "Or monitor log: tail -f ${LOG_FILE}"
else
    echo "❌ Failed to start training session"
    exit 1
fi
