#!/bin/bash
# Example inference script
#
# This script demonstrates how to run inference on an audio file.
# Update DATA_ROOT and AUDIO_FILE with your paths.

DATA_ROOT="${DATA_ROOT:-/path/to/RAVDESS}"
MODE="${MODE:-sentiment}"  # sentiment or emotion
AUDIO_FILE="${1:-/path/to/your_audio.wav}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root not found: $DATA_ROOT"
    echo "Please set DATA_ROOT environment variable or update the path in this script."
    exit 1
fi

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    echo "Usage: $0 <path_to_audio.wav>"
    exit 1
fi

echo "Running inference on: $AUDIO_FILE (mode: $MODE)"
python predict.py \
    --wav "$AUDIO_FILE" \
    --data_root "$DATA_ROOT" \
    --mode "$MODE" \
    --model smallcnn \
    --checkpoint artifacts/best_model.pt \
    --topk 3 \
    --out_dir pred_artifacts \
    --sr 22050 \
    --duration 3.0

echo ""
echo "Inference complete! Spectrogram saved to pred_artifacts/"

