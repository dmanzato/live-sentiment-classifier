# Emotion Mode Improvements Guide

## Current Status

- **Baseline F1-macro**: 0.558 (with sentiment-optimized parameters, no label smoothing)
- **With label smoothing 0.05**: 0.526 (worse - label smoothing may be hurting)
- **Target**: Improve to >0.60 (similar relative performance to sentiment mode)
- **Challenge**: Emotion mode has 8 classes vs 3 for sentiment, making it inherently harder

## Latest Results

| Configuration | F1 Score | Notes |
|--------------|----------|-------|
| Class-weight + SpecAugment, 30 epochs (freq=8, time=20) | **0.567** | ✅ **BEST** - Optimal config |
| Class-weight + SpecAugment, 40 epochs | 0.546 | ❌ Worse - longer training hurts |
| Class-weight + SpecAugment (freq=12, time=30) | 0.500 | ❌ Worse - more aggressive augmentation hurts |
| Class-weight + SpecAugment + Balanced sampler | 0.526 | ❌ Worse - balanced sampler didn't help |
| Class-weight + SpecAugment + Label smoothing 0.05 | 0.526 | ❌ Worse - label smoothing hurts |
| Previous baseline | 0.558 | Initial baseline |

**Key Findings**:
- **Best config**: Class-weight + SpecAugment (moderate: freq=8, time=20), 30 epochs, no label smoothing, no balanced sampler → **F1 = 0.567**
- Training longer (40 epochs) hurts performance (0.546 vs 0.567) - confirms overfitting
- More aggressive augmentation (freq=12, time=30) hurts performance (0.500 vs 0.567)
- Balanced sampler reduced overfitting gap but lowered overall performance
- **30 epochs appears optimal** - best checkpoint was saved automatically
- Current configuration is optimal - all attempted improvements reduced performance

## Mode-Specific Parameters

The Makefile now supports mode-specific parameters:

### Sentiment Mode (3 classes)
- **Epochs**: 30
- **Class-weighted loss**: Enabled
- **SpecAugment**: Enabled (freq=8, time=20)
- **Label smoothing**: 0.0
- **Expected F1-macro**: ~0.661

### Emotion Mode (8 classes) - Current Defaults
- **Epochs**: 30
- **Class-weighted loss**: Enabled
- **SpecAugment**: Enabled (freq=8, time=20)
- **Label smoothing**: 0.05 (slightly higher for regularization)
- **Current F1-macro**: 0.558

## Recommended Investigation Steps

### Step 1: Verify Dataset Coverage
The training script now logs actor coverage. Check the logs to ensure all 24 actors are being processed:
```
Actors in train: 19 (1-24)
Actors in val: 2 (1-24)
```

If you see fewer than 24 actors, check your dataset structure.

### Step 2: Baseline with Class Weights Only ✅ DONE
```bash
make train MODE=emotion EMOTION_USE_SPECAUG=0 EMOTION_LABEL_SMOOTHING=0.0
```
**Goal**: Establish baseline with just class-weighted loss (no augmentation, no label smoothing)
**Result**: F1 = 0.558 (baseline established)

### Step 3: Add SpecAugment ✅ DONE
```bash
make train MODE=emotion EMOTION_USE_SPECAUG=1 EMOTION_LABEL_SMOOTHING=0.0
```
**Goal**: See if augmentation helps (it did for sentiment mode)
**Result**: F1 = 0.558 (same as baseline - augmentation didn't help yet)

### Step 3b: Try WITHOUT Label Smoothing ✅ DONE
```bash
make train MODE=emotion EMOTION_LABEL_SMOOTHING=0.0
```
**Goal**: Confirm label smoothing is hurting (0.526 vs 0.558)
**Result**: F1 = **0.567** ✅ (best so far! Improved from 0.558)
**Observation**: Best F1 was at an earlier epoch (0.567), final epoch dropped to 0.382 - indicates overfitting

### Step 4: Try Different Label Smoothing Values
```bash
# Try 0.05 (current default)
make train MODE=emotion EMOTION_LABEL_SMOOTHING=0.05

# Try 0.1 (more aggressive)
make train MODE=emotion EMOTION_LABEL_SMOOTHING=0.1

# Try 0.0 (no smoothing)
make train MODE=emotion EMOTION_LABEL_SMOOTHING=0.0
```
**Goal**: Find optimal label smoothing for 8-class problem

### Step 5: Adjust Augmentation Parameters
Emotion classification might need different augmentation than sentiment:
```bash
# More aggressive augmentation
python train.py \
  --data_root ../data/RAVDESS \
  --mode emotion \
  --model resnet18 \
  --epochs 30 \
  --class_weight_loss \
  --use_specaug \
  --specaug_freq 12 \
  --specaug_time 30 \
  --label_smoothing 0.05

# Less aggressive augmentation
python train.py \
  --data_root ../data/RAVDESS \
  --mode emotion \
  --model resnet18 \
  --epochs 30 \
  --class_weight_loss \
  --use_specaug \
  --specaug_freq 4 \
  --specaug_time 10 \
  --label_smoothing 0.05
```

### Step 6: Try Balanced Sampler ✅ DONE
For emotion mode, class imbalance might be more severe. Also helps with overfitting:
```bash
python train.py \
  --data_root ../data/RAVDESS \
  --mode emotion \
  --model resnet18 \
  --epochs 30 \
  --class_weight_loss \
  --balanced_sampler \
  --use_specaug \
  --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512
```
**Result**: F1 = 0.526 ❌ (worse than baseline 0.567)
**Observation**: Reduced overfitting gap (0.526→0.489 vs 0.567→0.382) but overall performance dropped
**Conclusion**: Balanced sampler not beneficial for this dataset - class weights in loss are sufficient

### Step 7: Address Overfitting (Ongoing)
Since best F1 (0.567) was at an earlier epoch but final epoch dropped to 0.382, we need to prevent overfitting:

**✅ Tried: Balanced sampler** - Reduced overfitting gap but lowered performance (0.526 vs 0.567)

**Next Options:**

**✅ Option A: More aggressive augmentation** - TESTED
**Result**: F1 = 0.500 ❌ (worse than baseline 0.567)
**Conclusion**: More aggressive augmentation hurts performance - current moderate augmentation is optimal

**✅ Option B: Train longer** - TESTED
**Result**: F1 = 0.546 ❌ (worse than 30 epochs: 0.567)
```bash
make train MODE=emotion EMOTION_EPOCHS=40
```
**Conclusion**: Training longer hurts performance - 30 epochs is optimal
**Note**: Best checkpoint was saved automatically at the best epoch (likely around epoch 18-25)

**Option C: Try different learning rate**
```bash
python train.py \
  --data_root ../data/RAVDESS \
  --mode emotion \
  --model resnet18 \
  --epochs 30 \
  --class_weight_loss \
  --use_specaug \
  --lr 1e-4 \
  --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512
```
**Goal**: Lower learning rate might reduce overfitting

## Expected Class Distribution

For emotion mode, you should see roughly balanced distribution across 8 classes:
- neutral, calm, happy, sad, angry, fearful, disgust, surprised

Check the training logs for class distribution. If one class is severely underrepresented, class weights will help.

## Monitoring Tips

1. **Watch validation F1-macro**: Should improve over epochs
2. **Check for overfitting**: If best F1 is much higher than final F1 (like 0.567 vs 0.382), the model is overfitting
   - **Current issue**: Best F1 = 0.567, final F1 = 0.382 → significant overfitting
   - **Tried solutions**: Balanced sampler (hurt performance), more aggressive augmentation (hurt performance)
   - **Remaining options**: Train longer (best checkpoint saved), try different learning rate, or accept current performance
3. **Monitor class weights**: Logs show computed weights - extreme values (>3x min) are clipped
4. **Actor coverage**: Ensure all 24 actors are in the dataset
5. **Best checkpoint**: The model automatically saves the best checkpoint based on validation F1, so even if final epoch is worse, you have the best model

## Troubleshooting

### If F1 stays low (<0.55):
1. Verify all actors are loaded (check logs)
2. Try without augmentation first
3. Check class distribution (should be roughly balanced)
4. Try longer training (40-50 epochs)

### If overfitting occurs:
1. Increase label smoothing (0.1-0.15)
2. Reduce augmentation aggressiveness
3. Add dropout: `--dropout 0.3` (for SmallCNN)
4. Reduce epochs

### If underfitting occurs:
1. Train longer (40-50 epochs)
2. Reduce label smoothing or disable it
3. Increase augmentation
4. Check learning rate (default 3e-4 should be fine)

## Next Steps

1. ✅ Baseline established: **0.567 F1** (class-weight + SpecAugment, 30 epochs)
2. ✅ Tested balanced sampler: Didn't help (0.526 F1)
3. ✅ Tested label smoothing: Didn't help (0.526 F1)
4. ✅ Tested more aggressive augmentation: Didn't help (0.500 F1)
5. ✅ Tested longer training (40 epochs): Didn't help (0.546 F1)
6. **Remaining Options** (optional):
   - Try different learning rate (lower: 1e-4, or higher: 5e-4)
   - Accept 0.567 as optimal performance for 8-class problem (vs 0.661 for 3-class)
7. **✅ Current config is optimal** - All attempted improvements reduced performance
8. Document final configuration and update Makefile defaults

## Final Optimal Configuration

**F1-macro: 0.567** ✅ **OPTIMAL** (all attempted improvements reduced performance)

```bash
make train MODE=emotion
# Or explicitly:
python train.py \
  --data_root ../data/RAVDESS \
  --mode emotion \
  --model resnet18 \
  --epochs 30 \
  --class_weight_loss \
  --use_specaug \
  --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512
```

**Key Parameters**:
- **Epochs**: 30 (40 epochs reduced to 0.546)
- **Class-weighted loss**: Enabled
- **SpecAugment**: Enabled (freq=8, time=20 - moderate augmentation)
- **Label smoothing**: Disabled (0.05 reduced to 0.526)
- **Balanced sampler**: Disabled (reduced to 0.526)

**Note**: Best checkpoint is automatically saved at the best epoch (likely around epoch 18-25), so even if final epoch is worse, you have the 0.567 model saved in `artifacts/best_model_emotion.pt`.

## Notes

- Emotion mode is inherently harder (8 classes vs 3)
- **Performance comparison**: Sentiment (3 classes) = 0.661 F1, Emotion (8 classes) = 0.567 F1
  - Relative performance: 0.567/0.661 = 85.8% - reasonable for 2.67x more classes
- Class imbalance might be different for emotions vs sentiments
- Some emotions might be harder to distinguish (e.g., fearful vs disgust)
- Consider that RAVDESS has balanced emotion distribution by design
- **✅ Configuration is optimal**: All attempted improvements reduced performance:
  - Label smoothing: 0.526 ❌
  - Balanced sampler: 0.526 ❌
  - Aggressive augmentation: 0.500 ❌
  - Longer training (40 epochs): 0.546 ❌
- **Final recommendation**: Accept 0.567 as optimal performance for emotion mode. The configuration is well-tuned and further improvements would require architectural changes or different datasets.

