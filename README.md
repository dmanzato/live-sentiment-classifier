# live-sentiment-classifier

Real-time voice sentiment classification with **PyTorch**: microphone or dataset playback → **log-mel spectrogram** → **CNN** → **live sentiment predictions** (positive/negative/neutral).

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-%F0%9F%AA%80-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**Repository**: [https://github.com/dmanzato/live-sentiment-classifier](https://github.com/dmanzato/live-sentiment-classifier)

**Clone**: `git clone https://github.com/dmanzato/live-sentiment-classifier`

---

## Project Overview

This project provides a complete pipeline for **voice sentiment and emotion classification**:

- **Training**: Train CNN models (SmallCNN or ResNet18) on **RAVDESS** dataset with optional data augmentation
- **Classification Modes**: 
  - **Sentiment** (3 classes): positive, negative, neutral
  - **Emotion** (8 classes): neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **Live Streaming**: Real-time microphone input with live predictions
- **Inference**: Classify audio files with Top-K predictions

### Key Features

- **RAVDESS Dataset**: Support for the Ryerson Audio-Visual Database of Emotional Speech and Song
- **Dual Classification Modes**: Switch between sentiment (3-class) and emotion (8-class) classification
- **Model Architectures**: SmallCNN (custom lightweight CNN) and ResNet18 (adapted for 1-channel input)
- **Data Augmentation**: Optional SpecAugment (frequency & time masking)
- **Multi-device**: CPU, CUDA, and Apple Silicon MPS support
- **Robust Audio I/O**: Uses `soundfile`/`sounddevice` for reliable audio processing

### Project Structure

```
live-sentiment-classifier/
├── train.py              # Main training script
├── setup.py              # Package setup and CLI entry points
├── requirements.txt      # Python dependencies
├── Makefile              # Convenient shortcuts for common tasks
├── models/
│   └── small_cnn.py      # CNN architecture definitions
├── datasets/
│   └── ravdess.py        # RAVDESS dataset loader
├── transforms/
│   └── audio.py          # Audio preprocessing & augmentation
├── utils/
│   ├── models.py         # Shared model building utilities
│   ├── class_map.py      # Class map loading/saving utilities
│   ├── device.py         # Device selection utilities (CPU/CUDA/MPS)
│   ├── logging.py        # Logging configuration
│   └── normalization.py # Per-sample spectrogram normalization utilities
├── scripts/              # Additional scripts (streaming, visualization, etc.)
├── tests/                # Test suite
└── artifacts/            # Training outputs (models, confusion matrices)
```

---

## Installation

### Option 1: Install as a package (recommended)

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install package and dependencies
pip install -e .

# Now you can use CLI commands:
live-sentiment-train --help
live-sentiment-predict --help
live-sentiment-stream --help
```

### Option 2: Install dependencies only

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies only
pip install -r requirements.txt

# Run scripts with Python (requires PYTHONPATH=. or running from repo root)
python train.py --help
```

---

## Quickstart

### Download RAVDESS Dataset

1. Download the RAVDESS dataset from: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
2. Extract it to a directory (e.g., `../data/RAVDESS`)
3. The structure should be:
   ```
   RAVDESS/
     Actor_01/
       03-01-01-01-01-01-01.wav
       03-01-01-01-01-02-01.wav
       ...
     Actor_02/
       ...
     ...
     Actor_24/
       ...
   ```

### Training

Train a sentiment classification model:

```bash
# Using Makefile (recommended)
make setup
make train DATA_ROOT=../data/RAVDESS MODE=sentiment

# Or using Python directly
python train.py \
  --data_root ../data/RAVDESS \
  --mode sentiment \
  --epochs 25 \
  --model resnet18 \
  --use_specaug
```

Train an emotion classification model (8 classes):

```bash
make train DATA_ROOT=../data/RAVDESS MODE=emotion
```

### Makefile Shortcuts

```bash
# Setup virtual environment and install dependencies
make setup

# Train a model (default: sentiment mode)
make train DATA_ROOT=../data/RAVDESS

# Train emotion classification
make train DATA_ROOT=../data/RAVDESS MODE=emotion

# Run inference on a WAV file
make predict FILE=/path/to/audio.wav

# Start live streaming inference
make stream
```

---

## Dataset: RAVDESS

The **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset contains:

- **24 actors** (12 male, 12 female)
- **8 emotions**: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **2 emotional intensities**: normal, strong
- **2 statements**: "Kids are talking by the door", "Dogs are sitting by the door"
- **2 repetitions** per statement

### Sentiment Mapping

For sentiment classification, emotions are mapped as:
- **Positive**: happy, calm, surprised
- **Negative**: sad, angry, fearful, disgust
- **Neutral**: neutral

### File Naming Convention

RAVDESS files follow the pattern: `Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor.wav`

- **Modality**: 01=full, 02=video, 03=audio
- **Vocal**: 01=speech, 02=song
- **Emotion**: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
- **Intensity**: 01=normal, 02=strong
- **Statement**: 01=Kids, 02=Dogs
- **Repetition**: 01=first, 02=second
- **Actor**: 01-24

---

## Model Architectures

### SmallCNN

A lightweight 3-layer CNN:
- Input: `[B, 1, n_mels, time]` log-mel spectrograms
- Output: Class logits
- Parameters: ~100K

### ResNet18

Adapted ResNet18 for 1-channel input:
- First conv layer adapted for single-channel spectrograms
- Final classifier adapted for number of classes
- Parameters: ~11M

---

## Audio Processing

The pipeline uses **log-mel spectrograms** for feature extraction:

1. **Load audio**: Mono, resampled to target sample rate (default: 22050 Hz)
2. **STFT**: Short-Time Fourier Transform with configurable window/hop
3. **Mel filterbank**: Convert to mel scale (default: 128 mel bins)
4. **Log transform**: Convert to log scale (dB)
5. **Normalization**: Per-sample normalization (mean=0, std=1)

---

## Configuration

### Default Parameters

- **Sample rate**: 22050 Hz
- **Mel bins**: 128
- **FFT size**: 1024
- **Hop length**: 512
- **Duration**: 3.0 seconds (RAVDESS files are ~3-4 seconds)
- **Train/Val/Test split**: 80/10/10 (stratified by actor to avoid data leakage)

### Training Parameters

- **Batch size**: 32
- **Learning rate**: 3e-4
- **Weight decay**: 1e-4
- **Epochs**: 25
- **Optimizer**: Adam
- **Scheduler**: Cosine annealing with warmup

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Author

**Daniel A. G. Manzato**  
Email: dmanzato@gmail.com

---

## Acknowledgments

- **RAVDESS Dataset**: Ryerson Audio-Visual Database of Emotional Speech and Song
- Built with PyTorch, torchaudio, and other open-source libraries

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Status

**Version**: 0.1.0 (Initial release - November 14, 2025)

This is an early version. More features (streaming inference, visualization tools, etc.) will be added in future releases.

