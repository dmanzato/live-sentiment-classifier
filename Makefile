# Developer UX for live-sentiment-classifier
# Usage:
#   make setup                      # create .venv + install deps
#   make train                      # train on RAVDESS
#   make predict FILE=...           # run predict.py on a WAV
#   make stream [DEVICE=...]        # live mic inference
#   make vis [SPLIT=test]           # dataset visualizer
#   make lint test typecheck        # quality checks

SHELL := /usr/bin/env bash
.ONESHELL:

# System Python used to create venv
SYS_PY ?= python
# Virtualenv directory
VENV ?= .venv
# Venv Python interpreter
PY := $(VENV)/bin/python
# Venv pip
PIP := $(VENV)/bin/pip

# Root folder of RAVDESS dataset
DATA_ROOT ?= ../data/RAVDESS
# Model checkpoint path (mode-specific: artifacts/best_model_{MODE}.pt)
# Falls back to artifacts/best_model.pt for backward compatibility
CHECKPOINT ?=
# Output dir for predict artifacts
OUT_DIR ?= pred_artifacts

# Model architecture (resnet18|smallcnn)
MODEL ?= resnet18
# Audio sample rate (Hz)
SR ?= 22050
# Number of mel bins
N_MELS ?= 128
# FFT window size (samples)
N_FFT ?= 1024
# Hop length between FFT frames (samples)
HOPLEN ?= 512

# Default number of epochs (best performance at epoch 18, but train to 30 for safety)
EPOCHS ?= 30
# Training batch size
BATCH ?= 16
# Train split ratio
TRAIN_RATIO ?= 0.8
# Validation split ratio (test implied)
VAL_RATIO ?= 0.1
# Mode-specific training parameters
# Sentiment mode defaults (best config from v0.2.0 experiments - F1: 0.661)
SENTIMENT_EPOCHS ?= 30
SENTIMENT_USE_CLASS_WEIGHT ?= 1
SENTIMENT_USE_SPECAUG ?= 1
SENTIMENT_LABEL_SMOOTHING ?= 0.0

# Emotion mode defaults (optimized independently - current best F1: 0.567)
# Emotion has 8 classes vs 3, so tested independently
# NOTE: Tested label smoothing, balanced sampler, aggressive augmentation, longer training - all hurt performance
# Final optimal config is identical to sentiment mode, but verified independently
EMOTION_EPOCHS ?= 30
EMOTION_USE_CLASS_WEIGHT ?= 1
EMOTION_USE_SPECAUG ?= 1
EMOTION_LABEL_SMOOTHING ?= 0.0

# Legacy defaults (for backward compatibility)
USE_CLASS_WEIGHT ?= 1
USE_SPECAUG ?= 1

# Audio duration analyzed per sample (sec) - RAVDESS files are ~3-4 seconds
DUR ?= 3.0
# How long to keep each item before auto-advance
HOLD ?= $(DUR)
# Top-K classes shown in bars (default: empty = show all classes)
# Set TOPK=N to limit display to top N classes (e.g., TOPK=3 for top 3 only)
TOPK ?=
# Rolling analysis window length (sec) - increased for better accuracy
# Automatically capped to actual audio duration (RAVDESS files are ~3-4s)
WIN ?= 4.0
# Rolling analysis hop between updates (sec) - increased for smoother UI
HOP ?= 1.0
# Dataset split to visualize (train|val|test)
SPLIT ?= test
# Classification mode (sentiment|emotion)
MODE ?= sentiment

# Visualization shuffle control (default ON; disable with NO_SHUFFLE=1)
VIS_SHUFFLE_FLAG := $(if $(NO_SHUFFLE),--no_shuffle,)

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make setup"
	@echo "  make train [MODE=sentiment|emotion] [MODEL=resnet18|smallcnn] [EPOCHS=N]"
	@echo "  make predict FILE=path/to.wav [MODE=sentiment|emotion]"
	@echo "  make stream [MODE=sentiment|emotion] [DEVICE=substring or index]"
	@echo "  make vis [MODE=sentiment|emotion] [SPLIT=test] [NO_SHUFFLE=1] [DUR=3 HOLD=3 WIN=4 HOP=1.0]"
	@echo "  make lint | test | typecheck"
	@echo ""
	@echo "Training Examples:"
	@echo "  make train                    # Train sentiment model (best v0.2.0 config, F1: 0.661)"
	@echo "  make train MODE=emotion       # Train emotion model (optimized config, current F1: 0.558)"
	@echo "  make train EPOCHS=15          # Override epochs (sentiment: 30, emotion: 30)"
	@echo ""
	@echo "Mode-Specific Configurations:"
	@echo "  Sentiment (3 classes):"
	@echo "    - Epochs: 30 | Class-weight: enabled | SpecAugment: enabled | Label smoothing: 0.0"
	@echo "    - Expected F1-macro: ~0.661"
	@echo "  Emotion (8 classes):"
	@echo "    - Epochs: 30 | Class-weight: enabled | SpecAugment: enabled | Label smoothing: 0.0"
	@echo "    - Optimal F1-macro: 0.567 (30 epochs, class-weight + SpecAugment)"
	@echo ""
	@echo "Override mode-specific params:"
	@echo "  make train MODE=emotion EMOTION_LABEL_SMOOTHING=0.1  # Try different label smoothing"
	@echo "  make train MODE=sentiment SENTIMENT_EPOCHS=25       # Override sentiment epochs"

# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
$(VENV)/bin/python:
	$(SYS_PY) -m venv $(VENV)

.PHONY: setup
setup: $(VENV)/bin/python
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	# dev tools
	$(PIP) install ruff mypy pytest soundfile scipy
	@echo "✅ Setup complete. Use venv python: $(PY)"

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
.PHONY: train
train:
	@$(eval EPOCHS_VAL := $(if $(filter emotion,$(MODE)),$(EMOTION_EPOCHS),$(SENTIMENT_EPOCHS)))
	@$(eval USE_CW := $(if $(filter emotion,$(MODE)),$(EMOTION_USE_CLASS_WEIGHT),$(SENTIMENT_USE_CLASS_WEIGHT)))
	@$(eval USE_SA := $(if $(filter emotion,$(MODE)),$(EMOTION_USE_SPECAUG),$(SENTIMENT_USE_SPECAUG)))
	@$(eval LABEL_SMOOTH := $(if $(filter emotion,$(MODE)),$(EMOTION_LABEL_SMOOTHING),$(SENTIMENT_LABEL_SMOOTHING)))
	@echo "Training $(MODE) mode with optimized configuration"
	@echo "  Model: $(MODEL) | Epochs: $(EPOCHS_VAL)"
	@echo "  Class-weighted loss: $(USE_CW) | SpecAugment: $(USE_SA) | Label smoothing: $(LABEL_SMOOTH)"
	PYTHONPATH=. $(PY) train.py \
	  --data_root "$(DATA_ROOT)" \
	  --mode $(MODE) \
	  --train_ratio $(TRAIN_RATIO) \
	  --val_ratio $(VAL_RATIO) \
	  --batch_size $(BATCH) \
	  --epochs $(EPOCHS_VAL) \
	  --model $(MODEL) \
	  $(if $(filter 1,$(USE_CW)),--class_weight_loss,) \
	  $(if $(filter 1,$(USE_SA)),--use_specaug,) \
	  $(if $(filter-out 0.0 0,$(LABEL_SMOOTH)),--label_smoothing $(LABEL_SMOOTH),) \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN)

# ------------------------------------------------------------
# PREDICT SINGLE FILE
# ------------------------------------------------------------
.PHONY: predict
predict:
	@if [ -z "$(FILE)" ]; then echo "Usage: make predict FILE=path/to.wav [MODE=sentiment|emotion]"; exit 1; fi
	PYTHONPATH=. $(PY) predict.py \
	  --wav "$(FILE)" \
	  --data_root "$(DATA_ROOT)" \
	  --mode $(MODE) \
	  $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)",) \
	  --model $(MODEL) \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN) \
	  $(if $(TOPK),--topk $(TOPK),) \
	  --out_dir "$(OUT_DIR)"

# ------------------------------------------------------------
# LIVE STREAM INFERENCE
# ------------------------------------------------------------
.PHONY: stream
stream:
	PYTHONPATH=. $(PY) scripts/stream_infer.py \
	  --data_root "$(DATA_ROOT)" \
	  --mode $(MODE) \
	  $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)",) \
	  --model $(MODEL) \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN) \
	  --win_sec $(WIN) --inf_win_sec 3.0 --hop_sec $(HOP) \
	  $(if $(TOPK),--topk $(TOPK),) \
	  --temporal_avg --apply_class_weights \
	  --spec_auto_gain --spec_pmin 5 --spec_pmax 95 \
	  --auto_gain_norm \
	  $(if $(DEVICE),--device "$(DEVICE)",)

# ------------------------------------------------------------
# DATASET VISUALIZER
# ------------------------------------------------------------
.PHONY: vis
vis:
	PYTHONPATH=. $(PY) scripts/vis_dataset.py \
	  --data_root "$(DATA_ROOT)" \
	  --mode $(MODE) \
	  --split $(SPLIT) \
	  $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)",) \
	  --model $(MODEL) \
	  $(if $(TOPK),--topk $(TOPK),) \
	  --spec_auto_gain --spec_pmin 5 --spec_pmax 95 \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN) \
	  --duration $(DUR) --hold_sec $(HOLD) \
	  --ana_win_sec $(WIN) --ana_hop_sec $(HOP) \
	  --play_audio --out_latency high \
	  $(VIS_SHUFFLE_FLAG)

# ------------------------------------------------------------
# LINT / TEST / TYPECHECK
# ------------------------------------------------------------
.PHONY: lint
lint:
	$(PY) -m ruff check .

.PHONY: typecheck
typecheck:
	# Adjust paths if your package name differs
	$(PY) -m mypy scripts || true

.PHONY: test
test:
	PYTHONPATH=. $(PY) -m pytest -q

