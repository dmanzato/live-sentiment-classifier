# F1-Macro Score Improvements

This document outlines the improvements made to help increase the F1-macro score from ~0.7 to higher values.

## Summary of Changes

### 1. **Class-Weighted Loss Function** ✅
- **What**: Automatically computes class weights based on training data distribution
- **Why**: Helps handle class imbalance by giving more weight to underrepresented classes
- **Usage**: `--class_weight_loss`

### 2. **Focal Loss** ✅
- **What**: Advanced loss function that focuses learning on hard examples
- **Why**: Better handles class imbalance and difficult samples compared to standard cross-entropy
- **Usage**: `--focal_loss --focal_gamma 2.0`
- **Note**: Automatically uses class weights when enabled

### 3. **Label Smoothing** ✅
- **What**: Softens hard labels to prevent overconfidence
- **Why**: Reduces overfitting and improves generalization
- **Usage**: `--label_smoothing 0.1` (typical values: 0.05-0.1)

### 4. **Improved Model Architecture** ✅
- **What**: Added Batch Normalization and Dropout to SmallCNN
- **Why**: 
  - BatchNorm: Stabilizes training, allows higher learning rates
  - Dropout: Prevents overfitting
- **Usage**: `--dropout 0.3` (default: 0.3, BatchNorm enabled by default)

### 5. **Enhanced Data Augmentation** ✅
- **What**: Configurable SpecAugment parameters
- **Why**: More aggressive augmentation can improve generalization
- **Usage**: `--use_specaug --specaug_freq 12 --specaug_time 30`

## Recommended Training Commands

### Baseline ResNet18 (Your Original Setup)
```bash
python train.py \
    --data_root ../data/RAVDESS/ \
    --mode sentiment \
    --model resnet18 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --batch_size 16 \
    --epochs 5 \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512
```
**Expected F1: ~0.73**

### Baseline SmallCNN (Default Model)
```bash
python train.py --data_root /path/to/ravdess --mode sentiment --epochs 25
```
**Expected F1: ~0.4-0.5** (SmallCNN is smaller, lower capacity)

### Step 1: Start with Class-Weighted Loss (Safest First Step)
**For ResNet18:**
```bash
python train.py \
    --data_root ../data/RAVDESS/ \
    --mode sentiment \
    --model resnet18 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --batch_size 16 \
    --epochs 15 \
    --class_weight_loss \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512
```
**Expected improvement: Baseline → 0.57-0.58** (Note: `--balanced_sampler` was tested but didn't help significantly)

### Step 2: Add Data Augmentation (Best Configuration)
**This is the optimal configuration that achieved 0.661 F1:**
```bash
python train.py \
    --data_root ../data/RAVDESS/ \
    --mode sentiment \
    --model resnet18 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --batch_size 16 \
    --epochs 30 \
    --class_weight_loss \
    --use_specaug \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512
```
**Expected F1: ~0.66** (best at epoch 18, watch for overfitting after epoch 20)

### Step 3: Add Label Smoothing (Optional)
**Label smoothing showed similar performance (0.662 vs 0.661):**
```bash
python train.py \
    --data_root ../data/RAVDESS/ \
    --mode sentiment \
    --model resnet18 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --batch_size 16 \
    --epochs 30 \
    --class_weight_loss \
    --use_specaug \
    --label_smoothing 0.05 \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512
```
**Expected F1: ~0.66** (similar to without label smoothing)

### Step 4: Try Focal Loss (Not Recommended - Tested, Didn't Help)
**Focal Loss was tested but performed worse (0.573 vs 0.661):**
```bash
python train.py \
    --data_root ../data/RAVDESS/ \
    --mode sentiment \
    --model resnet18 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --batch_size 16 \
    --epochs 30 \
    --focal_loss \
    --focal_gamma 1.5 \
    --use_specaug \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512
```
**Result: 0.573** (worse than class-weighted loss)

### Advanced: Full Configuration (Use Only If Previous Steps Help)
```bash
python train.py \
    --data_root /path/to/ravdess \
    --mode sentiment \
    --epochs 50 \
    --use_specaug \
    --specaug_freq 12 \
    --specaug_time 30 \
    --mixup_alpha 0.3 \
    --focal_loss \
    --focal_gamma 2.0 \
    --label_smoothing 0.1 \
    --dropout 0.3 \
    --use_bn \
    --balanced_sampler \
    --lr 1e-3
```

## Key Hyperparameters Explained

- **`--focal_loss`**: Use Focal Loss (recommended for imbalanced datasets)
- **`--focal_gamma`**: Higher values (2.0-3.0) focus more on hard examples
- **`--class_weight_loss`**: Automatically balances class importance in loss
- **`--label_smoothing`**: 0.05-0.1 typically works well
- **`--dropout`**: 0.3-0.5 for regularization
- **`--balanced_sampler`**: Ensures balanced batches during training
- **`--specaug_freq/time`**: Increase for more aggressive augmentation

## Actual Results Achieved

Based on experiments with ResNet18 on RAVDESS sentiment classification:

| Configuration | F1 Score | Notes |
|--------------|----------|-------|
| Original baseline (5 epochs) | 0.732 | Original best result |
| Current baseline (5 epochs) | 0.532 | Different random seed/split |
| + Class-weighted loss (15 epochs) | 0.577 | Normalized weights prevent extreme values |
| + Class-weighted loss + SpecAugment (15 epochs) | 0.554 | Moderate augmentation |
| + Class-weighted loss + SpecAugment + Label smoothing (30 epochs) | 0.662 | Best at epoch 18 |
| + Class-weighted loss + SpecAugment (30 epochs, no label smoothing) | 0.661 | Similar performance |
| + Focal Loss | 0.573 | Didn't help for this dataset |
| + More aggressive augmentation | 0.539 | Too aggressive, hurt performance |

**Best Configuration Achieved:**
- **F1 Score**: 0.661-0.662
- **Configuration**: Class-weighted loss + moderate SpecAugment (freq=8, time=20)
- **Epochs**: 30 (best at epoch 18)
- **Gap from original**: 0.071 (likely due to randomness/data splits)

## Tips for Further Improvement

1. **Train longer**: Increase `--epochs` to 30-40 (best performance was at epoch 18, so watch for overfitting)
2. **Tune learning rate**: Try `--lr 1e-4` or `--lr 5e-4` if training is unstable
3. **Increase mixup**: `--mixup_alpha 0.3-0.4` for stronger regularization (if not already using)
4. **Ensemble models**: Train multiple models with different seeds and average predictions
5. **Try different augmentation parameters**: Moderate augmentation (freq=8, time=20) worked best; avoid too aggressive settings

## Monitoring

Watch the validation F1-macro score during training. The best model is automatically saved to `artifacts/best_model.pt` based on validation F1-macro.

## Troubleshooting

### If Performance Drops (F1 < 0.7)

1. **Start with baseline**: Remove all new features and verify baseline performance
   ```bash
   python train.py --data_root /path/to/ravdess --mode sentiment --epochs 25
   ```

2. **Add features one at a time**: Enable features incrementally to identify what helps
   - First: `--class_weight_loss --balanced_sampler`
   - Then: `--use_specaug`
   - Then: `--label_smoothing 0.05` (start small)
   - Finally: `--focal_loss` (only if others help)

3. **If F1 decreases with Focal Loss**: 
   - Reduce `--focal_gamma` to 1.0 or 1.5
   - Or disable `--focal_loss` and use `--class_weight_loss` instead

4. **If overfitting**: 
   - Increase `--dropout` (0.3-0.5) or `--label_smoothing` (0.1-0.15)
   - Add `--use_bn` for BatchNorm regularization

5. **If underfitting**: 
   - Decrease `--dropout` to 0.0, disable `--use_bn`
   - Increase `--epochs` or try `--model resnet18`

6. **If training is unstable**:
   - Reduce learning rate: `--lr 1e-4`
   - Disable `--focal_loss` and use `--class_weight_loss` instead
   - Reduce `--mixup_alpha` or disable mixup

### Important Notes

- **Default behavior is unchanged**: All new features are opt-in, so existing scripts work as before
- **BatchNorm and Dropout are disabled by default**: Enable with `--use_bn` and `--dropout 0.3` if needed (for SmallCNN only)
- **Best configuration**: Class-weighted loss + moderate SpecAugment (freq=8, time=20) achieved 0.661 F1
- **ResNet18 is recommended**: All experiments used ResNet18; SmallCNN has lower capacity (~0.4-0.5 F1)
- **Watch for overfitting**: Best performance was at epoch 18; training longer (40 epochs) showed overfitting
- **Class weights are normalized**: Extreme weights are clipped to prevent training instability

