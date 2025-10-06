#!/bin/bash
# Start DeepSets training in a screen session with logging

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG=${1:-experiments/deepsets_baseline/config.yaml}
LOG_DIR="logs/deepsets_baseline"
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Create log directory
mkdir -p ${LOG_DIR}

echo "Starting DeepSets training..."
echo "Config: ${CONFIG}"
echo "Log file: ${LOG_FILE}"
echo ""

# Start training in screen session
screen -dmS deepsets_training bash -c "
    cd /lustre/hpc/pheno/inar/iceaggr && \
    uv run python scripts/train_deepsets.py ${CONFIG} --device cuda 2>&1 | tee ${LOG_FILE}
"

echo "Training started in screen session 'deepsets_training'"
echo ""
echo "To attach: screen -r deepsets_training"
echo "To detach: Ctrl+A then D"
echo "To view log: tail -f ${LOG_FILE}"
echo ""
