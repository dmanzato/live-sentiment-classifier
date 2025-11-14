#!/bin/bash
# Example: Record audio and immediately run inference
#
# This script records 3 seconds of audio from your microphone (RAVDESS uses ~3s clips),
# then runs inference on the recorded file.

DATA_ROOT="${DATA_ROOT:-/path/to/RAVDESS}"
MODE="${MODE:-sentiment}"  # sentiment or emotion
OUTPUT_WAV="recorded_sample.wav"

echo "Recording 3 seconds of audio..."
python scripts/record_wav.py --out "$OUTPUT_WAV" --seconds 3 --sr 22050

if [ $? -ne 0 ]; then
    echo "Error: Recording failed"
    exit 1
fi

echo ""
echo "Running inference on recorded audio..."
python predict.py \
    --wav "$OUTPUT_WAV" \
    --data_root "$DATA_ROOT" \
    --mode "$MODE" \
    --model smallcnn \
    --checkpoint artifacts/best_model.pt \
    --topk 3 \
    --sr 22050 \
    --duration 3.0

echo ""
echo "Done! Check the predictions above."

