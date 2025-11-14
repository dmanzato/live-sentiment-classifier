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
# Model checkpoint path
CHECKPOINT ?= artifacts/best_model.pt
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

# Default number of epochs
EPOCHS ?= 5
# Training batch size
BATCH ?= 16
# Train split ratio
TRAIN_RATIO ?= 0.8
# Validation split ratio (test implied)
VAL_RATIO ?= 0.1

# Audio duration analyzed per sample (sec) - RAVDESS files are ~3-4 seconds
DUR ?= 3.0
# How long to keep each item before auto-advance
HOLD ?= $(DUR)
# Top-K classes shown in bars
TOPK ?= 3
# Rolling analysis window length (sec)
WIN ?= 15.0
# Rolling analysis hop between updates (sec)
HOP ?= 0.5
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
	@echo "  make train [MODE=sentiment]"
	@echo "  make predict FILE=path/to.wav"
	@echo "  make stream [DEVICE=substring or index]"
	@echo "  make vis [SPLIT=test] [NO_SHUFFLE=1] [DUR=3 HOLD=3 WIN=3 HOP=0.5]"
	@echo "  make lint | test | typecheck"

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
	PYTHONPATH=. $(PY) train.py \
	  --data_root "$(DATA_ROOT)" \
	  --mode $(MODE) \
	  --train_ratio $(TRAIN_RATIO) \
	  --val_ratio $(VAL_RATIO) \
	  --batch_size $(BATCH) \
	  --epochs $(EPOCHS) \
	  --model $(MODEL) \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN)

# ------------------------------------------------------------
# PREDICT SINGLE FILE
# ------------------------------------------------------------
.PHONY: predict
predict:
	@if [ -z "$(FILE)" ]; then echo "Usage: make predict FILE=path/to.wav"; exit 1; fi
	PYTHONPATH=. $(PY) predict.py \
	  --wav "$(FILE)" \
	  --data_root "$(DATA_ROOT)" \
	  --checkpoint "$(CHECKPOINT)" \
	  --model $(MODEL) \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN) \
	  --topk $(TOPK) \
	  --out_dir "$(OUT_DIR)"

# ------------------------------------------------------------
# LIVE STREAM INFERENCE
# ------------------------------------------------------------
.PHONY: stream
stream:
	PYTHONPATH=. $(PY) scripts/stream_infer.py \
	  --data_root "$(DATA_ROOT)" \
	  --checkpoint "$(CHECKPOINT)" \
	  --model $(MODEL) \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN) \
	  --win_sec $(WIN) --hop_sec $(HOP) \
	  --topk $(TOPK) \
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
	  --checkpoint "$(CHECKPOINT)" \
	  --model $(MODEL) \
	  --topk $(TOPK) \
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

