# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Stream inference improvements**:
  - 3-second inference window (`--inf_win_sec`) matching training data duration for better accuracy
  - Temporal averaging (`--temporal_avg`) for smoother, more stable predictions
  - Class weight post-processing (`--apply_class_weights`) to handle class imbalance at inference time
  - All classes displayed in bars graph by default (was top-3)
  - Emoji subtitle showing detected class below main title
  - 60-second trend graph (reduced from 120s) for better visibility
- **Visualization improvements**:
  - All classes displayed in bars graph by default (was top-3)
  - Emoji display in subtitle for both prediction and ground truth
  - 4-second display duration per sample for better final status visibility
  - Improved final update logic to ensure last second is always shown
- **STREAM_INFER_IMPROVEMENTS.md**: Comprehensive guide for streaming inference enhancements

### Changed
- `stream_infer.py`: Default inference window changed from 15s to 3s (matches training data)
- `vis_dataset.py`: Default analysis window increased from 3.0s to 4.0s, hop from 0.5s to 1.0s
- `Makefile`: Stream target now includes `--temporal_avg` and `--apply_class_weights` by default
- `Makefile`: `TOPK` default changed to empty (show all classes) instead of 3

## [0.2.0] - 2025-11-15

### Added
- **Class-weighted loss function**: Automatically computes normalized class weights based on training data distribution to handle class imbalance (`--class_weight_loss`)
- **Focal Loss**: Advanced loss function for hard example mining (`--focal_loss --focal_gamma 2.0`)
- **Label smoothing**: Prevents overconfidence and improves generalization (`--label_smoothing 0.05-0.1`)
- **Enhanced SmallCNN architecture**: Optional Batch Normalization (`--use_bn`) and Dropout (`--dropout`) for regularization
- **Configurable SpecAugment**: Tune augmentation parameters (`--specaug_freq`, `--specaug_time`) for better results
- **IMPROVEMENTS.md**: Comprehensive documentation with experimental results and best practices

### Changed
- Class weights are now normalized to prevent extreme values (clips max weight to 3x minimum weight)
- SmallCNN model structure matches original exactly when using defaults (backward compatible)

### Improved
- Best configuration achieved: **0.661 F1-macro** (up from 0.532 baseline) with class-weighted loss + SpecAugment
- All improvements are opt-in and backward compatible
- Training stability improved with normalized class weights

### Testing
- All 28 tests passing
- Verified compatibility with ResNet18 and SmallCNN models

## [0.1.0] - 2024-11-14

### Added
- Initial release
- Training script for RAVDESS dataset
- Support for sentiment (3-class) and emotion (8-class) classification
- SmallCNN and ResNet18 model architectures
- Live streaming inference
- Dataset visualization tools
- Comprehensive test suite

[0.2.0]: https://github.com/dmanzato/live-sentiment-classifier/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dmanzato/live-sentiment-classifier/releases/tag/v0.1.0

