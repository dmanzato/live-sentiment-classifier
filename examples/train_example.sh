#!/bin/bash
# Example training script
# 
# This script demonstrates how to train a model on RAVDESS.
# Update DATA_ROOT with your actual dataset path.

DATA_ROOT="${DATA_ROOT:-/path/to/RAVDESS}"
MODE="${MODE:-sentiment}"  # sentiment or emotion

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root not found: $DATA_ROOT"
    echo "Please set DATA_ROOT environment variable or update the path in this script."
    exit 1
fi

echo "Training SmallCNN model in $MODE mode..."
python train.py \
    --data_root "$DATA_ROOT" \
    --mode "$MODE" \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --batch_size 16 \
    --epochs 5 \
    --model smallcnn \
    --use_specaug \
    --lr 1e-3 \
    --sr 22050 \
    --duration 3.0

echo ""
echo "Training complete! Best model saved to artifacts/best_model.pt"
echo ""
echo "To train ResNet18 instead, use:"
echo "  python train.py --data_root $DATA_ROOT --model resnet18 --mode $MODE --epochs 5"
echo ""
echo "To train in emotion mode (8 classes), use:"
echo "  MODE=emotion bash examples/train_example.sh"

