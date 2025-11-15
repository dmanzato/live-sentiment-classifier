# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

