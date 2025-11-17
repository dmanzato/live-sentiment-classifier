# Stream Inference Improvements

This document describes the improvements made to `stream_infer.py` for better real-time detection accuracy and stability.

## Overview

The streaming inference script (`stream_infer.py`) has been enhanced with several techniques to improve detection accuracy, especially when transitioning between different emotions/sentiments in live audio.

## Key Improvements

### 1. **3-Second Inference Window** ✅ (Default)

**Problem**: The original implementation used a 15-second buffer for both visualization and inference. This long window could dilute the signal when multiple emotions/sentiments are present, making it hard to detect transitions.

**Solution**: Separated inference window from visualization buffer:
- **`--inf_win_sec`** (default: 3.0s): Window size used for predictions
- **`--win_sec`** (default: 15.0s): Buffer size for visualization/spectrogram

**Why it helps**: The model was trained on ~3-second samples, so using a 3-second sliding window for inference matches training conditions and improves accuracy.

**Usage**:
```bash
# Default (3s inference window)
make stream MODE=emotion

# Custom inference window
make stream MODE=emotion -- --inf_win_sec 4.0
```

### 2. **Temporal Averaging** ✅ (Default)

**Problem**: Single-window predictions can be noisy and jittery, especially during transitions.

**Solution**: Average predictions over multiple overlapping windows:
- **`--temporal_avg`**: Enable temporal averaging
- **`--temporal_avg_windows`** (default: 3): Number of windows to average

**Why it helps**: Smooths predictions over time, reducing false positives and making transitions more stable.

**Usage**:
```bash
# Enabled by default
make stream MODE=emotion

# Disable if needed
make stream MODE=emotion -- --no-temporal_avg  # (if you add this flag)

# Custom number of windows
make stream MODE=emotion -- --temporal_avg_windows 5
```

### 3. **Class Weight Post-Processing** ✅ (Default)

**Problem**: Class imbalance can cause the model to favor common classes (e.g., "calm" in emotion mode), making it hard to detect rare classes.

**Solution**: Apply class weights to predictions as post-processing:
- **`--apply_class_weights`**: Enable class weight post-processing
- Automatically computes weights from training data distribution (same formula as training)

**Why it helps**: Boosts underrepresented classes by adjusting probabilities based on training distribution, helping detect transitions to less common classes.

**Usage**:
```bash
# Enabled by default
make stream MODE=emotion

# Disable if needed (not recommended)
# You would need to modify Makefile or call script directly
```

### 4. **Display All Classes in Bars Graph** ✅ (Default)

**Problem**: Only showing top-3 classes made it hard to see all available options.

**Solution**: Display all classes by default, sorted by probability:
- **`--topk`** (default: None = all classes): Number of classes to display
- Automatically shows all classes (3 for sentiment, 8 for emotion)

**Why it helps**: Provides complete visibility into model confidence across all classes.

### 5. **Emoji Display** ✅

**Problem**: Hard to quickly identify detected class at a glance.

**Solution**: Added emoji subtitle below main title showing the detected class emoji:
- Centered display below title
- Updates in real-time with detection
- Keeps original yellow emoji colors

**Emojis**:
- **Sentiment**: 😊 (positive), 😞 (negative), 😐 (neutral)
- **Emotion**: 😐 (neutral), 😌 (calm), 😊 (happy), 😢 (sad), 😠 (angry), 😨 (fearful), 😖 (disgust), 😲 (surprised)

### 6. **60-Second Trend Graph** ✅

**Problem**: Trend graph showed too much history (120s), making recent changes hard to see.

**Solution**: Reduced trend graph to 60 seconds:
- **`--trend_len`** (default: None = 60s worth): Number of hops to keep
- X-axis shows -60 to 0 seconds
- Better visibility of recent predictions

## Default Configuration

The Makefile `stream` target now uses optimal defaults:

```makefile
make stream MODE=emotion
```

This automatically includes:
- `--inf_win_sec 3.0` (3-second inference window)
- `--temporal_avg` (temporal averaging enabled)
- `--apply_class_weights` (class weight post-processing enabled)
- All classes displayed in bars graph
- 60-second trend graph

## Performance Impact

- **Accuracy**: Improved detection accuracy, especially for class transitions
- **Stability**: Smoother predictions with less jitter
- **Latency**: Minimal impact (~3 windows × hop_sec for temporal averaging)
- **Computational**: Slight increase due to temporal averaging, but negligible for real-time use

## Troubleshooting

### Detection Still Stuck on One Class

1. **Check class weights**: Ensure `--apply_class_weights` is enabled (default)
2. **Try longer temporal averaging**: Increase `--temporal_avg_windows` to 5 or 7
3. **Check audio quality**: Ensure microphone input is clear and not too quiet
4. **Verify model**: Make sure you're using a model trained with class-weighted loss

### Predictions Too Slow to Update

1. **Reduce temporal averaging**: Lower `--temporal_avg_windows` to 2
2. **Increase hop interval**: Use `--hop_sec 1.0` for less frequent updates
3. **Disable temporal averaging**: Remove `--temporal_avg` (not recommended)

### Too Much Jitter/Noise

1. **Increase temporal averaging**: Use `--temporal_avg_windows 5` or higher
2. **Check inference window**: Ensure `--inf_win_sec 3.0` matches training data
3. **Verify audio gain**: Use `--auto_gain_norm` for automatic normalization

## Comparison with vis_dataset.py

Both `stream_infer.py` and `vis_dataset.py` now use similar techniques:

| Feature | stream_infer.py | vis_dataset.py |
|---------|----------------|----------------|
| Inference window | 3.0s (configurable) | 4.0s (configurable) |
| Temporal averaging | ✅ Default | Optional (`--ana_temporal_avg`) |
| Class weights | ✅ Post-processing | N/A (uses ground truth) |
| Display all classes | ✅ Default | ✅ Default |
| Emoji display | ✅ Subtitle | ✅ Subtitle |

## Future Improvements

Potential areas for further enhancement:

1. **Adaptive window sizing**: Automatically adjust inference window based on audio characteristics
2. **Confidence thresholding**: Only update predictions when confidence exceeds threshold
3. **Ensemble methods**: Combine predictions from multiple models
4. **Voice activity detection**: Only predict when speech is detected
5. **Calibration**: Temperature scaling for better probability calibration

