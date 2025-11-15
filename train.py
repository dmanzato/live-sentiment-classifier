"""Training script for voice sentiment/emotion classification models (RAVDESS dataset)."""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt

from datasets.ravdess import RAVDESS
from transforms.audio import SpecAugment
from utils.logging import setup_logging, get_logger
from utils.device import get_device, get_device_name
from utils.models import build_model
from utils.class_map import save_class_map
from utils.normalization import normalize_per_sample

logger = get_logger("train")


# -----------------------------
# Helpers
# -----------------------------
def _unpack_batch(batch):
    """Allow datasets that return (x,y) or (x,y,meta)."""
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            x, y = batch
            return x, y, None
        if len(batch) == 3:
            x, y, meta = batch
            return x, y, meta
    return batch, None, None


def mixup_batch(x, y, alpha=0.0):
    """Standard mixup on features (spectrograms)."""
    if alpha <= 0.0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1.0 - lam) * x[idx]
    return x_m, y, (idx, lam)


def compute_class_weights(subset, num_classes: int) -> torch.DoubleTensor:
    """
    Compute per-sample weights for a Subset for use with WeightedRandomSampler.
    Works whether subset is a torch.utils.data.Subset or a Dataset.
    """
    if hasattr(subset, "indices") and hasattr(subset, "dataset"):
        # Subset case
        indices = subset.indices
        base_ds = subset.dataset
        labels = []
        for i in indices:
            item = base_ds[i]
            # item may be (x,y) or (x,y,meta)
            _, y, _ = _unpack_batch(item)
            labels.append(int(y))
    else:
        # Dataset case
        labels = []
        for i in range(len(subset)):
            _, y, _ = _unpack_batch(subset[i])
            labels.append(int(y))

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    inv = 1.0 / np.maximum(counts, 1)
    sample_weights = inv[labels]
    return torch.DoubleTensor(sample_weights)


def compute_class_weights_for_loss(subset, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for loss function (inverse frequency weighting).
    Returns tensor of shape [num_classes] suitable for CrossEntropyLoss(weight=...).
    Uses sklearn-style balanced weights: n_samples / (n_classes * np.bincount(y))
    """
    if hasattr(subset, "indices") and hasattr(subset, "dataset"):
        indices = subset.indices
        base_ds = subset.dataset
        labels = []
        for i in indices:
            item = base_ds[i]
            _, y, _ = _unpack_batch(item)
            labels.append(int(y))
    else:
        labels = []
        for i in range(len(subset)):
            _, y, _ = _unpack_batch(subset[i])
            labels.append(int(y))

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    # sklearn balanced weights: n_samples / (n_classes * np.bincount(y))
    total = len(labels)
    weights = total / (num_classes * np.maximum(counts, 1))
    # Normalize weights to prevent extreme values (clamp max weight to 3x min weight)
    # This prevents one class from dominating the loss
    min_weight = weights.min()
    max_weight_allowed = min_weight * 3.0
    weights = np.clip(weights, min_weight, max_weight_allowed)
    return torch.FloatTensor(weights)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for rare class (can be a tensor for per-class weights).
        gamma: Focusing parameter (gamma=0 is equivalent to CE loss).
        reduction: Specifies the reduction to apply to the output.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute cross entropy loss without reduction
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Compute p_t (probability of true class)
        # ce_loss = -log(p_t), so p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # Clamp pt to avoid numerical instability
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        
        # Compute focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    """
    Evaluate model on a dataset. Returns (macro F1, confusion matrix).
    """
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            x, y, _ = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)
            x = normalize_per_sample(x)
            logits = model(x)
            pred = logits.argmax(dim=1)
            ys.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())

    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    cm = confusion_matrix(ys, preds, labels=list(range(num_classes)))
    f1 = f1_score(ys, preds, average="macro")
    return f1, cm


def plot_confusion_matrix(cm: np.ndarray, out_path: str, class_names: list) -> None:
    """Plot and save confusion matrix."""
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.debug(f"Saved confusion matrix to {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    setup_logging(level=logging.INFO)

    ap = argparse.ArgumentParser(description="Train voice sentiment/emotion classification model (RAVDESS dataset)")
    ap.add_argument("--data_root", type=str, required=True, help="Path to RAVDESS dataset root (with Actor_XX folders)")
    ap.add_argument("--mode", type=str, default="sentiment", choices=["sentiment", "emotion"], 
                    help="Classification mode: 'sentiment' (3 classes) or 'emotion' (8 classes)")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
    ap.add_argument("--epochs", type=int, default=25, help="Epochs")
    ap.add_argument("--lr", type=float, default=3e-4, help="Base learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    ap.add_argument("--model", type=str, default="smallcnn", choices=["smallcnn", "resnet18"], help="Model architecture")
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout probability for SmallCNN (default: 0.0, use 0.3-0.5 for regularization)")
    ap.add_argument("--use_bn", action="store_true", help="Use Batch Normalization in SmallCNN (disabled by default)")
    ap.add_argument("--use_specaug", action="store_true", help="Enable SpecAugment")
    ap.add_argument("--specaug_freq", type=int, default=8, help="SpecAugment frequency mask parameter")
    ap.add_argument("--specaug_time", type=int, default=20, help="SpecAugment time mask parameter")
    ap.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha (0=off)")
    ap.add_argument("--balanced_sampler", action="store_true", help="Balanced sampling on train split")
    ap.add_argument("--class_weight_loss", action="store_true", help="Use class-weighted loss function")
    ap.add_argument("--focal_loss", action="store_true", help="Use Focal Loss instead of CrossEntropy")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma parameter")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor (0.0-0.1)")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")

    # Speech-friendly defaults (RAVDESS files are ~3-4 seconds)
    ap.add_argument("--n_mels", type=int, default=128, help="Mel bins")
    ap.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    ap.add_argument("--hop_length", type=int, default=512, help="STFT hop length")
    ap.add_argument("--duration", type=float, default=3.0, help="Clip duration (s) - RAVDESS files are ~3-4s")
    ap.add_argument("--sr", type=int, default=22050, help="Sample rate")
    ap.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs (linear) before cosine schedule")

    ap.add_argument("--log_file", type=str, default=None, help="Optional log file path")
    args = ap.parse_args()

    if args.log_file:
        setup_logging(level=logging.INFO, log_file=args.log_file)

    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data root does not exist: {data_root}")
        sys.exit(1)

    device = get_device()
    logger.info(f"Using device: {get_device_name()} ({device})")
    logger.info(f"Classification mode: {args.mode}")

    if args.train_ratio + args.val_ratio >= 1.0:
        logger.error(f"train_ratio + val_ratio must be < 1.0 (got {args.train_ratio + args.val_ratio})")
        sys.exit(1)

    logger.info(
        f"Train ratio: {args.train_ratio}, Val ratio: {args.val_ratio}, "
        f"Test ratio (implied): {1.0 - args.train_ratio - args.val_ratio:.2f}"
    )

    augment = None
    if args.use_specaug:
        augment = SpecAugment(
            freq_mask_param=args.specaug_freq,
            time_mask_param=args.specaug_time,
            num_freq_masks=2,
            num_time_masks=2
        )
        logger.info(f"Using SpecAugment (freq={args.specaug_freq}, time={args.specaug_time})")

    # Load datasets with proper splits (RAVDESS handles splits internally)
    try:
        logger.info(f"Loading RAVDESS dataset (mode={args.mode})...")
        train_ds = RAVDESS(
            root=str(data_root),
            mode=args.mode,
            split="train",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            target_sr=args.sr,
            duration=args.duration,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            augment=augment,
        )
        val_ds = RAVDESS(
            root=str(data_root),
            mode=args.mode,
            split="val",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            target_sr=args.sr,
            duration=args.duration,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            augment=None,  # No augmentation on validation
        )
        logger.info(f"Train items: {len(train_ds)} | Val items: {len(val_ds)}")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}", exc_info=True)
        sys.exit(1)

    # Class names & count
    try:
        idx2name = train_ds.idx2name
        num_classes = len(idx2name)
        class_names = [idx2name[i] for i in range(num_classes)]
        logger.info(f"Classes ({num_classes}): {', '.join(class_names)}")
    except Exception as e:
        logger.error(f"Could not derive class names: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Classes: {num_classes} | Train items: {len(train_ds)} | Val items: {len(val_ds)}")

    # DataLoaders (optionally balanced sampler for train)
    try:
        pin_mem = (device.type == "cuda")
        persistent = args.num_workers > 0

        sampler = None
        shuffle = True
        if args.balanced_sampler:
            weights = compute_class_weights(train_ds, num_classes)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False
            logger.info("Using WeightedRandomSampler for class-balanced batches")

        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=pin_mem,
            persistent_workers=persistent,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_mem,
            persistent_workers=persistent,
        )
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}", exc_info=True)
        sys.exit(1)

    # Artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    logger.info(f"Artifacts directory: {artifacts_dir.absolute()}")

    # Model
    try:
        model = build_model(args.model, num_classes, dropout=args.dropout, use_bn=args.use_bn).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created: {args.model} with {n_params} parameters")
        if args.model == "smallcnn":
            logger.info(f"  Dropout: {args.dropout}, BatchNorm: {'enabled' if args.use_bn else 'disabled'}")
    except Exception as e:
        logger.error(f"Error building model: {e}", exc_info=True)
        sys.exit(1)

    # Save class map
    try:
        save_class_map(artifacts_dir, list(idx2name.values()))
        logger.info("Saved class map to artifacts/class_map.json")
    except Exception as e:
        logger.warning(f"Could not save class map: {e}")

    # Optimizer + schedule
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    warmup_epochs = max(0, int(args.warmup_epochs))
    T_max = max(1, args.epochs - warmup_epochs)
    scheduler = CosineAnnealingLR(opt, T_max=T_max)

    # Setup loss function
    loss_weight = None
    if args.class_weight_loss or args.focal_loss:
        loss_weight = compute_class_weights_for_loss(train_ds, num_classes).to(device)
        logger.info(f"Computed class weights for loss: {loss_weight.cpu().numpy()}")
    
    if args.focal_loss:
        loss_fn = FocalLoss(alpha=loss_weight, gamma=args.focal_gamma)
        logger.info(f"Using Focal Loss (gamma={args.focal_gamma})")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=args.label_smoothing)
        if args.label_smoothing > 0:
            logger.info(f"Using CrossEntropyLoss with label smoothing={args.label_smoothing}")
        elif loss_weight is not None:
            logger.info("Using class-weighted CrossEntropyLoss")
        else:
            logger.info("Using standard CrossEntropyLoss")
    
    best_f1 = -1.0
    logger.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")

        # Warmup: linear ramp lr during first warmup_epochs
        if epoch <= warmup_epochs:
            warm_lr = (epoch / float(warmup_epochs)) * args.lr
            for pg in opt.param_groups:
                pg["lr"] = warm_lr
            logger.info(f"Warmup LR set to {warm_lr:.6f}")

        # ----------------- Train -----------------
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0

        for batch_idx, batch in enumerate(train_dl):
            x, y, _ = _unpack_batch(batch)
            x, y = x.to(device), y.to(device)
            x = normalize_per_sample(x)

            # mixup
            x_in, y_in, mix = mixup_batch(x, y, alpha=args.mixup_alpha)

            logits = model(x_in)
            if mix is None:
                loss = loss_fn(logits, y_in)
                preds = logits.argmax(dim=1)
                total_correct += (preds == y_in).sum().item()
            else:
                index, lam = mix
                loss = lam * loss_fn(logits, y_in) + (1 - lam) * loss_fn(logits, y_in[index])
                preds = logits.argmax(dim=1)
                total_correct += (preds == y_in).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"  Batch {batch_idx + 1}/{len(train_dl)}, Loss: {loss.item():.4f}")

        train_loss = total_loss / total if total > 0 else 0.0
        train_acc = total_correct / total if total > 0 else 0.0

        # Step cosine schedule after warmup
        if epoch > warmup_epochs:
            scheduler.step()

        # ----------------- Validate -----------------
        f1, cm = evaluate(model, val_dl, device, num_classes)
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} val_f1_macro={f1:.3f}"
        )

        # Save confusion matrix
        try:
            cm_path = artifacts_dir / f"confusion_matrix_epoch{epoch}.png"
            plot_confusion_matrix(cm, str(cm_path), list(idx2name.values()))
        except Exception as e:
            logger.warning(f"Could not save confusion matrix: {e}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            try:
                checkpoint_path = artifacts_dir / "best_model.pt"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved best model (F1={f1:.3f}) to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error saving model checkpoint: {e}")

    logger.info("=" * 60)
    logger.info(f"Training completed. Best F1 score: {best_f1:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

