#!/usr/bin/env python
"""
Real-time streaming sentiment/emotion inference from microphone / audio device.

Keys:
  SPACE : pause/resume
  q     : quit

Example:
  PYTHONPATH=. python scripts/stream_infer.py \
    --data_root /path/to/RAVDESS \
    --checkpoint artifacts/best_model.pt \
    --model resnet18 \
    --mode sentiment \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 \
    --win_sec 15 --inf_win_sec 3.0 --hop_sec 0.5 \
    --spec_auto_gain --spec_pmin 5 --spec_pmax 95 \
    --auto_gain_norm
"""

import argparse
import json
from pathlib import Path
import queue
from collections import deque
import sys
import time

import numpy as np
import sounddevice as sd
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

from transforms.audio import get_mel_transform  # we'll compute mel here, log/pcen below
from utils.device import get_device, get_device_name
from utils.models import build_model
from utils.class_map import load_class_map
from datasets.ravdess import RAVDESS, SENTIMENTS, EMOTIONS

# ---------- emoji mappings ----------
# Emoji mappings for sentiment and emotion classes
SENTIMENT_EMOJIS = {
    "positive": "😊",
    "negative": "😞",
    "neutral": "😐",
}

EMOTION_EMOJIS = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "disgust": "😖",  # Changed from 🤢 to 😖 (confounded face) for better font compatibility
    "surprised": "😲",
}

def get_class_emoji(class_name: str, mode: str) -> str:
    """Get emoji for a class name based on mode."""
    if mode == "sentiment":
        return SENTIMENT_EMOJIS.get(class_name.lower(), "❓")
    else:  # emotion
        return EMOTION_EMOJIS.get(class_name.lower(), "❓")

# For matching training pipeline: AmplitudeToDB with stype="power" uses 10*log10
def power_to_db(power_spec: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    """Convert power spectrogram to decibel scale (matches AmplitudeToDB with stype='power')."""
    return 10.0 * np.log10(np.maximum(power_spec, amin) / ref)

# ----------------- helpers -----------------
def _get_default_classes(mode: str):
    """Get default class names based on mode."""
    if mode == "emotion":
        return EMOTIONS
    return SENTIMENTS

def _detect_num_classes_from_state_dict(state_dict: dict, mode: str = "sentiment") -> int:
    # Works for ResNet (fc.weight) and many small heads
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and k.endswith(".weight"):
            if any(t in k for t in ("fc.", "classifier", "head")):
                return v.shape[0]
    # fallback: first linear-like weight
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            return v.shape[0]
    # final fallback based on mode
    return len(SENTIMENTS) if mode == "sentiment" else len(EMOTIONS)

def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state

def _compute_spec_limits(mel_img: np.ndarray, auto_gain: bool, pmin: float, pmax: float, prev=None):
    """Compute (vmin, vmax) for the spectrogram, supporting per-frame auto-gain."""
    if auto_gain:
        lo = np.percentile(mel_img, pmin)
        hi = np.percentile(mel_img, pmax)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = (mel_img.min(), mel_img.max())
            if lo == hi:
                hi = lo + 1e-6
        return lo, hi
    else:
        if prev is None:
            lo, hi = (mel_img.min(), mel_img.max())
            if lo == hi:
                hi = lo + 1e-6
            return lo, hi
        return prev

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Live sentiment/emotion classification stream")
    ap.add_argument("--data_root", type=str, required=True, help="Path to RAVDESS root directory")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint. If not specified, uses artifacts/best_model_{mode}.pt (with backward compatibility)")
    ap.add_argument("--model", type=str, default="resnet18", choices=["smallcnn", "resnet18"])
    ap.add_argument("--mode", type=str, default="sentiment", choices=["sentiment", "emotion"],
                    help="Classification mode: 'sentiment' (3 classes) or 'emotion' (8 classes) [default: sentiment]")

    # audio / feature params
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)

    # streaming params
    ap.add_argument("--win_sec", type=float, default=15.0,
                    help="Buffer window size for visualization (seconds). Default: 15.0")
    ap.add_argument("--inf_win_sec", type=float, default=3.0,
                    help="Inference window size for predictions (seconds). Matches training data duration (~3s). Default: 3.0")
    ap.add_argument("--hop_sec", type=float, default=0.5)
    ap.add_argument("--device", type=str, default=None, help="Sounddevice input id or substring")
    ap.add_argument("--list-devices", action="store_true")

    # viz / inference
    ap.add_argument("--topk", type=int, default=None,
                    help="Top-K predictions to show (default: all classes). Set to limit display to top-k.")
    ap.add_argument("--spec_auto_gain", action="store_true", help="Auto color scale per frame")
    ap.add_argument("--spec_pmin", type=float, default=5.0, help="Lower percentile (auto-gain)")
    ap.add_argument("--spec_pmax", type=float, default=95.0, help="Upper percentile (auto-gain)")
    ap.add_argument("--spec_debug", action="store_true")
    ap.add_argument("--input_gain", type=float, default=1.0, help="Input gain multiplier (1.0=no change, >1.0=boost)")
    ap.add_argument("--auto_gain_norm", action="store_true", help="Auto-normalize input gain based on signal level")

    # trend line options
    ap.add_argument("--trend_len", type=int, default=None, help="Number of hops to keep in trend buffer (default: 60s worth)")
    ap.add_argument("--trend_ema", type=float, default=0.0, help="EMA smoothing (0.0 disables)")
    
    # inference improvement options
    ap.add_argument("--temporal_avg", action="store_true",
                    help="Average predictions over multiple windows for better accuracy (slower but more stable)")
    ap.add_argument("--temporal_avg_windows", type=int, default=3,
                    help="Number of windows to average when --temporal_avg is enabled (default: 3)")
    ap.add_argument("--apply_class_weights", action="store_true",
                    help="Apply class weights to predictions as post-processing (helps with class imbalance)")

    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    device = get_device()
    print(f"Using device: {get_device_name()} ({device})")
    print(f"Classification mode: {args.mode}")

    # classes
    artifacts_dir = Path("artifacts")
    data_root = Path(args.data_root)
    class_names = load_class_map(data_root, artifacts_dir, mode=args.mode)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")

    # Determine checkpoint path (mode-specific with backward compatibility)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Try mode-specific checkpoint first
        ckpt_path_mode = artifacts_dir / f"best_model_{args.mode}.pt"
        ckpt_path_old = artifacts_dir / "best_model.pt"
        
        if ckpt_path_mode.exists():
            ckpt_path = ckpt_path_mode
            print(f"Using mode-specific checkpoint: {ckpt_path}")
        elif ckpt_path_old.exists():
            ckpt_path = ckpt_path_old
            print(f"Using legacy checkpoint: {ckpt_path} (consider training with --mode {args.mode} for mode-specific checkpoint)")
        else:
            print(f"Error: No checkpoint found. Tried:")
            print(f"  - {ckpt_path_mode}")
            print(f"  - {ckpt_path_old}")
            print(f"Please train a model first or specify --checkpoint")
            return
    
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        return
    
    # model
    ckpt = _load_checkpoint(ckpt_path, device)
    ckpt_out = _detect_num_classes_from_state_dict(ckpt, mode=args.mode)
    if ckpt_out != num_classes:
        # Adapt to checkpoint
        if args.mode == "sentiment" and ckpt_out == len(SENTIMENTS):
            class_names = SENTIMENTS
        elif args.mode == "emotion" and ckpt_out == len(EMOTIONS):
            class_names = EMOTIONS
        else:
            class_names = [f"class_{i}" for i in range(ckpt_out)]
        num_classes = ckpt_out
        print(f"Adjusted class names to checkpoint: {num_classes} classes.")
    
    # Set topk to display all classes by default, or use user-specified value
    if args.topk is None:
        args.topk = num_classes  # Display all classes
    elif args.topk > num_classes:
        args.topk = num_classes
        print(f"Adjusted topk to {args.topk} (max available classes)")
    print(f"Displaying {args.topk} classes in bars graph")

    model = build_model(args.model, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    
    # Compute class weights for post-processing (if requested)
    class_weights = None
    if args.apply_class_weights:
        # Load a small subset of training data to compute class distribution
        try:
            temp_ds = RAVDESS(
                root=args.data_root,
                mode=args.mode,
                split="train",
                train_ratio=0.8,
                val_ratio=0.1,
                target_sr=args.sr,
                duration=3.0,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                augment=None,
            )
            # Compute class weights using same formula as training
            labels = [item.label for item in temp_ds.items]
            labels = np.asarray(labels, dtype=np.int64)
            counts = np.bincount(labels, minlength=num_classes)
            total = len(labels)
            weights = total / (num_classes * np.maximum(counts, 1))
            # Normalize weights (clamp max weight to 3x min weight, same as training)
            min_weight = weights.min()
            max_weight_allowed = min_weight * 3.0
            weights = np.clip(weights, min_weight, max_weight_allowed)
            class_weights = weights.astype(np.float32)
            print(f"Computed class weights for post-processing: {class_weights}")
        except Exception as e:
            print(f"Warning: Could not compute class weights: {e}")
            print("Continuing without class weight post-processing")

    # mel on CPU (avoid MPS STFT window mismatch)
    mel_t = get_mel_transform(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )

    # audio stream setup
    blocksize = int(args.sr * args.hop_sec)
    window_size = int(args.sr * args.win_sec)
    buf = np.zeros(window_size, dtype=np.float32)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        q.put(indata.copy())

    stream = sd.InputStream(
        device=args.device,
        channels=1,
        samplerate=args.sr,
        callback=callback,
        blocksize=blocksize,
    )

    # figure layout:
    #   left: spectrogram (rowspan=2)
    #   right-top: top-K bars (+ % text)
    #   right-bottom: rolling top-1 trend (past→now)
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    # Reserve space for emoji subtitle (top=0.88 instead of 0.90)
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 2], height_ratios=[3, 1], wspace=0.25, hspace=0.35,
                          top=0.88, bottom=0.12, left=0.06, right=0.98)
    ax_spec = fig.add_subplot(gs[:, 0])          # spans both rows
    ax_bar  = fig.add_subplot(gs[0, 1])          # top-right
    ax_trend = fig.add_subplot(gs[1, 1])         # bottom-right
    
    # Emoji subtitle (positioned below main title, above graphs)
    # This will be updated with the detected class emoji
    emoji_subtitle = fig.text(0.5, 0.92, "", ha="center", va="center", fontsize=18)

    # spectrogram image (negative time axis, past→now)
    init_img = np.random.randn(args.n_mels, 64) * 1e-6
    spec_img = ax_spec.imshow(
        init_img, origin="lower", aspect="auto",
        extent=[-args.win_sec, 0.0, 0.0, float(args.n_mels)],
        cmap='magma',
        interpolation='nearest'
    )
    ax_spec.set_title("Spectrogram")
    ax_spec.set_xlabel("Time (s, past → now)")
    ax_spec.set_ylabel("Mel bins")
    ax_spec.set_xlim(-args.win_sec, 0.0)

    # Bars - display all classes (or topk if specified)
    topk = max(1, min(args.topk, num_classes))
    bars = ax_bar.barh(range(topk), np.zeros(topk), align="center")
    ax_bar.set_xlim(0.0, 1.0)
    ax_bar.set_yticks(range(topk))
    ax_bar.set_yticklabels([""] * topk)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Probability")
    if topk == num_classes:
        ax_bar.set_title(f"All {num_classes} classes (sorted by probability)")
    else:
        ax_bar.set_title(f"Top-{topk} predictions")
    bar_texts = [ax_bar.text(0.0, i, "", va="center", ha="left", fontsize=9) for i in range(topk)]

    # Trend line with negative x (past→now at 0)
    # Default to 60 seconds worth of data if not specified
    if args.trend_len is None:
        args.trend_len = int(60.0 / args.hop_sec)  # 60 seconds worth of hops
    trend_len = max(2, args.trend_len)
    # Calculate actual duration and ensure x-axis matches the leftmost data point
    trend_duration = trend_len * args.hop_sec
    xs = -np.arange(trend_len - 1, -1, -1, dtype=float) * args.hop_sec  # e.g., [-59.5..0] for 60s
    trend_buf = deque([np.nan] * trend_len, maxlen=trend_len)
    ema_val = None if args.trend_ema <= 0.0 else 0.0
    (trend_line,) = ax_trend.plot(xs, [np.nan] * trend_len, lw=2)
    
    # Temporal averaging buffer (for --temporal_avg)
    pred_history = deque(maxlen=args.temporal_avg_windows) if args.temporal_avg else None
    ax_trend.set_ylim(0.0, 1.0)
    # Set x-axis to match the actual leftmost data point so curve touches y-axis
    ax_trend.set_xlim(xs[0], 0.0)  # Use xs[0] instead of -60.0 to align with data
    try:
        total_span = abs(xs[0])
        step = 10.0
        ticks = np.arange(-np.floor(total_span / step) * step, step, step)
        if 0.0 not in ticks:
            ticks = np.append(ticks, 0.0)
        ax_trend.set_xticks(ticks)
    except Exception:
        pass
    ax_trend.set_xlabel("Time (s, past → now)")
    ax_trend.set_ylabel("Top-1 p")
    ax_trend.grid(True, alpha=0.25)
    if ema_val is not None:
        (ema_line,) = ax_trend.plot(xs, [np.nan] * trend_len, lw=1, alpha=0.7)
    else:
        ema_line = None

    # spectrogram clim management
    last_clim = None
    first_spec_frame = True

    # ------------- keyboard controls -------------
    state = {"paused": False, "running": True}

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
            if state["paused"]:
                fig.suptitle("⏸ PAUSED", fontsize=12, color="gray", y=0.98)
                emoji_subtitle.set_text("")
            else:
                fig.suptitle("", fontsize=12, color="black", y=0.98)
                emoji_subtitle.set_text("")
            fig.canvas.draw_idle()
        elif event.key in ("q", "Q"):
            state["running"] = False

    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    # ---------------------------------------------

    print("🎙️  Streaming... (SPACE=pause/resume, q=quit)")
    if args.auto_gain_norm:
        print("📊 Auto gain normalization enabled (helps with external audio sources)")
    elif args.input_gain != 1.0:
        print(f"📊 Input gain set to {args.input_gain:.2f}x")
    try:
        with stream:
            while state["running"]:
                try:
                    data = q.get(timeout=0.1).squeeze()
                except queue.Empty:
                    plt.pause(0.001)
                    continue

                if data.ndim > 1:
                    data = data[:, 0]
                L = len(data)
                if L == 0:
                    continue

                if state["paused"]:
                    plt.pause(0.001)
                    continue

                # roll buffer and append
                if L >= window_size:
                    buf[:] = data[-window_size:]
                else:
                    buf = np.roll(buf, -L)
                    buf[-L:] = data

                # === feature extraction on CPU ===
                x = buf.astype(np.float32, copy=False)
                
                # Input gain normalization (helps with external audio sources)
                if args.auto_gain_norm:
                    # Normalize to a target RMS level (0.1 is a reasonable level for speech/music)
                    target_rms = 0.1
                    current_rms = np.sqrt(np.mean(x**2))
                    if current_rms > 1e-6:  # Avoid division by zero
                        gain = target_rms / current_rms
                        # Limit gain to reasonable range (0.1x to 10x)
                        gain = np.clip(gain, 0.1, 10.0)
                        x = x * gain
                else:
                    # Apply manual gain
                    x = x * args.input_gain
                
                # Clip to prevent overflow
                x = np.clip(x, -1.0, 1.0)
                
                # Extract the last inf_win_sec seconds for inference (matches training data duration)
                # This improves accuracy by using context similar to training samples (~3s)
                inf_window_size = int(args.sr * args.inf_win_sec)
                if len(x) >= inf_window_size:
                    x_inf = x[-inf_window_size:]  # Last inf_win_sec seconds
                else:
                    # Not enough data yet, pad with zeros on the left
                    pad_size = inf_window_size - len(x)
                    x_inf = np.pad(x, (pad_size, 0), mode='constant')
                
                wav_t_inf = torch.from_numpy(x_inf).unsqueeze(0)  # [1, T_inf]

                # Power mel for inference (using shorter window matching training)
                mel_inf = mel_t(wav_t_inf).squeeze(0).cpu().numpy()  # [n_mels, time], power

                # Log-mel in decibel scale (matches training: AmplitudeToDB with stype="power")
                # Training uses: 10 * log10(x), not just log10(x)
                mel_feat_inf = power_to_db(mel_inf, ref=1.0, amin=1e-10)

                # Per-clip standardization for inference (matches training)
                mel_feat_inf = (mel_feat_inf - mel_feat_inf.mean()) / (mel_feat_inf.std() + 1e-6)

                # inference on device (using 3s window matching training data)
                with torch.no_grad():
                    feats = torch.from_numpy(mel_feat_inf).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,n_mels,time]
                    logits = model(feats)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                
                # Temporal averaging: average predictions over multiple windows
                if args.temporal_avg and pred_history is not None:
                    pred_history.append(probs.copy())
                    if len(pred_history) >= args.temporal_avg_windows:
                        # Average probabilities over recent windows
                        probs = np.mean(list(pred_history), axis=0)
                    # If not enough history yet, use current prediction
                
                # Apply class weights as post-processing (if requested)
                if args.apply_class_weights and class_weights is not None:
                    # Multiply probabilities by class weights and renormalize
                    weighted_probs = probs * class_weights
                    probs = weighted_probs / weighted_probs.sum()
                
                # For visualization, use full buffer (win_sec) - keep existing visualization behavior
                wav_t = torch.from_numpy(x).unsqueeze(0)  # [1, T]
                mel = mel_t(wav_t).squeeze(0).cpu().numpy()  # [n_mels, time], power
                mel_feat_raw = power_to_db(mel, ref=1.0, amin=1e-10)  # Store raw for visualization

                # top-k (UNCHANGED bars)
                order = np.argsort(probs)[::-1][:topk]
                top_probs = probs[order]
                top_names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in order]

                # dynamic x-limit based on top-1
                xmax = max(1.0, float(top_probs[0]) * 1.1)
                ax_bar.set_xlim(0.0, xmax)

                # update bars and their percentage texts
                for i, (bar, p) in enumerate(zip(bars, top_probs)):
                    width = float(p)
                    bar.set_width(width)

                    label = f"{100.0 * width:.1f}%"
                    thresh_inside = 0.18 * xmax
                    if width >= thresh_inside:
                        x_text = max(0.0, width - 0.02 * xmax)
                        ha = "right"
                        color = "white"
                    else:
                        x_text = min(width + 0.02 * xmax, xmax * 0.98)
                        ha = "left"
                        color = "black"

                    bar_texts[i].set_text(label)
                    bar_texts[i].set_x(x_text)
                    bar_texts[i].set_y(i)
                    bar_texts[i].set_ha(ha)
                    bar_texts[i].set_color(color)

                ax_bar.set_yticklabels(top_names)

                # ---- spectrogram image: use RAW mel_feat for better visualization ----
                vmin, vmax = _compute_spec_limits(
                    mel_feat_raw, args.spec_auto_gain, args.spec_pmin, args.spec_pmax, last_clim
                )
                spec_img.set_data(mel_feat_raw)
                spec_img.set_extent([-args.win_sec, 0.0, 0.0, float(args.n_mels)])
                ax_spec.set_xlim(-args.win_sec, 0.0)

                if args.spec_auto_gain or first_spec_frame:
                    spec_img.set_clim(vmin=vmin, vmax=vmax)
                    last_clim = (vmin, vmax)
                    first_spec_frame = False

                # title w/ top-1
                pred_label = top_names[0]
                pred_prob = float(top_probs[0])
                fig.suptitle(f"{pred_label} ({pred_prob*100:4.1f}%)", fontsize=12, y=0.98)
                
                # Update emoji subtitle: centered, showing detected class emoji
                pred_emoji = get_class_emoji(pred_label, args.mode)
                emoji_subtitle.set_text(pred_emoji)

                # ---- update trend: newest prob goes at x=0 (right edge) ----
                trend_buf.append(pred_prob)
                y = np.array(trend_buf, dtype=float)
                trend_line.set_xdata(xs)
                trend_line.set_ydata(y)

                if ema_line is not None:
                    alpha = float(args.trend_ema)
                    ema_series = []
                    prev = None
                    for v in y:
                        if np.isnan(v):
                            ema_series.append(np.nan)
                        else:
                            prev = v if prev is None else (alpha * v + (1.0 - alpha) * prev)
                            ema_series.append(prev)
                    ema_line.set_xdata(xs)
                    ema_line.set_ydata(np.array(ema_series, dtype=float))

                if args.spec_debug:
                    print(f"spec range: vmin={vmin:.3f} vmax={vmax:.3f}; "
                          f"mel_raw mean={mel_feat_raw.mean():.3f} std={mel_feat_raw.std():.3f}; "
                          f"mel_feat_inf mean={mel_feat_inf.mean():.3f} std={mel_feat_inf.std():.3f}; "
                          f"top1={pred_label}:{pred_prob:.3f}")

                plt.pause(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass
        plt.ioff()
        plt.close(fig)
        # Ensure stream closes
        try:
            stream.abort()
            stream.close()
        except Exception:
            pass
        sys.exit(0)

if __name__ == "__main__":
    main()