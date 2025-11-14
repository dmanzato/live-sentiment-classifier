# Examples

This directory contains example scripts demonstrating how to use live-sentiment-classifier.

## Quick Start

### 1. Training a Model

```bash
# Set your data root
export DATA_ROOT=/path/to/RAVDESS

# Run training example (sentiment mode, 3 classes)
bash examples/train_example.sh
```

Or run directly:
```bash
python train.py \
    --data_root /path/to/RAVDESS \
    --mode sentiment \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --batch_size 16 \
    --epochs 5 \
    --model smallcnn \
    --use_specaug \
    --sr 22050 \
    --duration 3.0
```

To train in emotion mode (8 classes):
```bash
MODE=emotion bash examples/train_example.sh
```

### 2. Running Inference

```bash
# On a single audio file
bash examples/inference_example.sh /path/to/your_audio.wav
```

Or run directly:
```bash
python predict.py \
    --wav /path/to/your_audio.wav \
    --data_root /path/to/RAVDESS \
    --mode sentiment \
    --model smallcnn \
    --topk 3 \
    --sr 22050 \
    --duration 3.0
```

### 3. Record and Predict

Record audio from your microphone and immediately classify it:

```bash
bash examples/record_and_predict.sh
```

### 4. Live Streaming Inference

For real-time microphone input:

```bash
python scripts/stream_infer.py \
    --data_root /path/to/RAVDESS \
    --mode sentiment \
    --checkpoint artifacts/best_model.pt \
    --model resnet18 \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 \
    --win_sec 15 --hop_sec 0.5 --topk 3 \
    --spec_auto_gain --spec_pmin 5 --spec_pmax 95 \
    --auto_gain_norm
```

### 5. Dataset Visualization

Browse the dataset interactively:

```bash
python scripts/vis_dataset.py \
    --data_root /path/to/RAVDESS \
    --mode sentiment \
    --split test \
    --checkpoint artifacts/best_model.pt \
    --model smallcnn \
    --play_audio \
    --sr 22050 \
    --duration 3.0
```

## Classification Modes

The project supports two classification modes:

- **Sentiment** (3 classes): `positive`, `neutral`, `negative`
- **Emotion** (8 classes): `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

Switch between them using the `--mode` flag:
- `--mode sentiment` (default)
- `--mode emotion`

## Environment Variables

Set these to avoid specifying paths repeatedly:

```bash
export DATA_ROOT=/path/to/RAVDESS
export MODE=sentiment  # or emotion
```

## Model Architectures

The project supports two model architectures:

- **SmallCNN**: Lightweight custom CNN (faster training, smaller model)
- **ResNet18**: Pre-trained ResNet adapted for audio (better accuracy, larger model)

Switch between them using the `--model` flag:
- `--model smallcnn` (default)
- `--model resnet18`

## Tips

1. **First time setup**: Make sure you have the RAVDESS dataset downloaded and extracted
2. **Training**: Start with SmallCNN for faster iteration, then try ResNet18 for better accuracy
3. **Inference**: The model expects audio clips at 22050Hz sample rate. Default duration is 3.0 seconds (RAVDESS clips are typically 3-4 seconds).
4. **Live streaming**: Use `--device` flag to select a specific microphone if you have multiple
5. **Mode selection**: Use `--mode sentiment` for 3-class sentiment classification, or `--mode emotion` for 8-class emotion classification

## Troubleshooting

- **Import errors**: Make sure you're running from the project root with `PYTHONPATH=.`
- **Audio device errors**: On macOS, grant microphone permissions in System Settings
- **CUDA errors**: The code will fall back to CPU automatically if CUDA is unavailable
- **Dataset errors**: Ensure RAVDESS is organized with Actor_XX folders containing properly named WAV files

