#!/usr/bin/env python
"""
Visualize RAVDESS spectrograms + predictions with rolling updates and a top-1 trend line.

Keys:
  Left/Right : previous/next item
  Space      : pause/resume auto-advance
  p          : play/stop audio of current item
  q          : quit

Example:
  PYTHONPATH=. python scripts/vis_dataset.py \
    --data_root /path/to/RAVDESS \
    --mode sentiment \
    --split test \
    --checkpoint artifacts/best_model.pt \
    --model resnet18 \
    --topk 3 \
    --spec_auto_gain --spec_pmin 5 --spec_pmax 95 \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 \
    --duration 3.0 --hold_sec 3.0 \
    --ana_win_sec 3.0 --ana_hop_sec 0.5 \
    --play_audio --out_latency high
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from matplotlib import gridspec
from scipy.signal import resample_poly

from datasets.ravdess import RAVDESS, SENTIMENTS, EMOTIONS
from transforms.audio import get_mel_transform, wav_to_logmel
from utils.device import get_device, get_device_name
from utils.models import build_model
from utils.class_map import load_class_map

# ---------- util helpers ----------
def _compute_spec_limits(mel_img: np.ndarray, auto_gain: bool, pmin: float, pmax: float, prev=None):
    if auto_gain:
        lo = np.percentile(mel_img, pmin)
        hi = np.percentile(mel_img, pmax)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = (mel_img.min(), mel_img.max())
            if lo == hi: hi = lo + 1e-6
        return lo, hi
    else:
        if prev is None:
            lo, hi = (mel_img.min(), mel_img.max())
            if lo == hi: hi = lo + 1e-6
            return lo, hi
        return prev

def _load_wav_centered(path: Path, target_sr: int, duration: float) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != target_sr:
        x = resample_poly(x, target_sr, sr).astype(np.float32)
    N = int(duration * target_sr)
    if len(x) < N:
        pad = N - len(x)
        left, right = pad // 2, pad - pad // 2
        x = np.pad(x, (left, right))
    else:
        start = max(0, (len(x) - N) // 2)
        x = x[start:start + N]
    return x

# ---------- playback crackle-safe helpers ----------
def _device_default_samplerate(device=None) -> int | None:
    try:
        info = sd.query_devices(device, kind='output')
        sr = info.get('default_samplerate', None)
        if isinstance(sr, (int, float)) and sr > 0:
            return int(round(sr))
    except Exception:
        pass
    return None

def _apply_fade(x: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    n = max(1, int(sr * (ms / 1000.0)))
    if n*2 >= len(x):
        return x
    w = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = x.copy()
    y[:n] *= w
    y[-n:] *= w[::-1]
    return y

def _prepare_playback_audio(x_model_sr: np.ndarray, model_sr: int, out_sr: int) -> np.ndarray:
    if out_sr != model_sr:
        y = resample_poly(x_model_sr, out_sr, model_sr).astype(np.float32)
    else:
        y = x_model_sr.astype(np.float32, copy=False)
    y = _apply_fade(y, out_sr, ms=5.0)
    m = np.max(np.abs(y)) if y.size else 0.0
    if m > 1.0:
        y = y / m
    return y

# ---------- script ----------
def main():
    ap = argparse.ArgumentParser(
        description="Visualize RAVDESS spectrogram + labels with rolling updates and trend line"
    )
    ap.add_argument("--data_root", type=str, required=True, help="Path to RAVDESS root directory")
    ap.add_argument("--mode", type=str, default="sentiment", choices=["sentiment", "emotion"],
                    help="Classification mode: 'sentiment' (3 classes) or 'emotion' (8 classes) [default: sentiment]")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])  # Which split to visualize
    ap.add_argument("--train_ratio", type=float, default=0.8)  # Train split ratio
    ap.add_argument("--val_ratio", type=float, default=0.1)    # Val split ratio

    ap.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")  # Model weights to load
    ap.add_argument("--model", type=str, default="resnet18", choices=["smallcnn", "resnet18"])  # Backbone to use
    ap.add_argument("--topk", type=int, default=3, help="How many classes to display in bars (default: 3 for sentiment)")

    # feature/audio params (model side)
    ap.add_argument("--sr", type=int, default=22050)       # Model sample rate (Hz)
    ap.add_argument("--n_mels", type=int, default=128)     # Mel bins
    ap.add_argument("--n_fft", type=int, default=1024)     # FFT size
    ap.add_argument("--hop_length", type=int, default=512) # STFT hop (samples)
    ap.add_argument("--duration", type=float, default=3.0, help="Seconds analyzed/played per item (RAVDESS files are ~3-4s)")

    # pacing / viz options
    ap.add_argument("--hold_sec", type=float, default=None,  # Auto-advance dwell time; defaults to duration
                    help="Seconds to keep each sample before auto-advance (defaults to duration).")
    ap.add_argument("--spec_auto_gain", action="store_true")  # Enable per-frame percentile scaling
    ap.add_argument("--spec_pmin", type=float, default=5.0)   # Lower percentile for auto-gain
    ap.add_argument("--spec_pmax", type=float, default=95.0)  # Upper percentile for auto-gain
    ap.add_argument("--sleep", type=float, default=0.02,      # UI tick pause (smaller -> smoother updates)
                    help="UI tick pause (seconds).")
    ap.add_argument("--play_audio", action="store_true")      # Play audio through sounddevice
    ap.add_argument("--out_device", type=str, default=None)   # Output device name/index for playback
    ap.add_argument("--out_sr", type=int, default=0,          # Output playback rate; 0=auto (device default)
                    help="Output playback samplerate; 0 uses device default.")
    ap.add_argument("--out_latency", type=str, default="high", choices=["low","high"])  # Playback latency preference

    # rolling analysis window over the playing clip
    ap.add_argument("--ana_win_sec", type=float, default=3.0,  # Length of analysis window (seconds)
                    help="Rolling analysis window length (seconds).")
    ap.add_argument("--ana_hop_sec", type=float, default=0.5,  # Time between rolling updates (seconds) — matches your HOP change
                    help="Time between rolling updates (seconds).")

    # shuffle control
    ap.add_argument("--no_shuffle", action="store_true",      # Disable shuffling; keep sequential order
                    help="Disable random shuffling of sample order (sequential order).")

    args = ap.parse_args()
    hold_sec = args.hold_sec if args.hold_sec is not None else args.duration  # dwell time per item

    # Output playback configuration
    try:
        sd.default.device = (None, args.out_device) if args.out_device is not None else None  # set output device
    except Exception:
        pass
    if args.out_latency == "high":
        sd.default.latency = ('high', 'high')  # robust, crackle-free defaults
    if args.out_sr and args.out_sr > 0:
        out_sr = int(args.out_sr)  # explicit playback rate
    else:
        out_sr = _device_default_samplerate(args.out_device) or 44100  # device default or fallback

    device = get_device()
    print(f"Using device: {get_device_name()} ({device}) | playback_sr={out_sr}")
    print(f"Classification mode: {args.mode}")

    # classes
    artifacts_dir = Path("artifacts")
    data_root = Path(args.data_root)
    class_names = load_class_map(data_root, artifacts_dir, mode=args.mode)
    num_classes = len(class_names)  # total classes for model head and bars
    print(f"Loaded {num_classes} classes: {class_names}")

    # Adjust topk based on mode and num_classes
    if args.topk > num_classes:
        args.topk = num_classes
        print(f"Adjusted topk to {args.topk} (max available classes)")

    # dataset
    ds = RAVDESS(
        root=args.data_root,
        mode=args.mode,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_sr=args.sr,
        duration=args.duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        augment=None,
    )

    print(f"Dataset split='{args.split}' mode='{args.mode}' size={len(ds)}")

    # shuffled (or sequential) index order for this run
    order = np.arange(len(ds))  # permutation over dataset indices
    if args.no_shuffle:
        print("Playback order: sequential")
    else:
        rng = np.random.default_rng()  # different order each execution
        rng.shuffle(order)
        print("Playback order: shuffled")

    def _current_ds_index(pos: int) -> int:
        return int(order[pos])  # map position -> dataset index

    # model
    state = torch.load(args.checkpoint, map_location=device)
    state = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model = build_model(args.model, num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()

    # mel transform on CPU
    mel_t = get_mel_transform(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )

    # figure layout: left = spectrogram; right = [bars over trend]
    plt.ion()
    fig = plt.figure(figsize=(12, 7))  # main figure for spec + bars + trend
    outer = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.25, bottom=0.18, left=0.06, right=0.98, top=0.90)
    ax_spec = fig.add_subplot(outer[0, 0])  # spectrogram axis

    right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], height_ratios=[3, 1], hspace=0.35)
    ax_bar   = fig.add_subplot(right[0, 0])  # top-k bars axis
    ax_trend = fig.add_subplot(right[1, 0])  # top-1 probability over time

    # spectrogram
    init_img = np.random.randn(args.n_mels, 64) * 1e-6  # tiny noise prevents colormap solid block
    im = ax_spec.imshow(init_img, origin="lower", aspect="auto")
    ax_spec.set_xlabel("Frames")
    ax_spec.set_ylabel("Mel bins")
    ax_spec.set_title("Spectrogram (rolling window)")

    # bars
    topk = max(1, min(args.topk, num_classes))  # ensure 1..num_classes
    bars = ax_bar.barh(range(topk), np.zeros(topk), align="center")
    ax_bar.set_xlim(0.0, 1.0)
    ax_bar.set_yticks(range(topk))
    ax_bar.set_yticklabels([""] * topk)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Probability")
    ax_bar.set_title(f"Top-{topk} predictions")
    bar_texts = [ax_bar.text(0.0, i, "", va="center", ha="left", fontsize=9) for i in range(topk)]  # percentage labels

    # trend line (top-1 prob vs time in this clip)
    ax_trend.set_xlim(0.0, args.duration)  # span 0..duration (now 30s by default)
    ax_trend.set_ylim(0.0, 1.0)
    (trend_line,) = ax_trend.plot([], [], linewidth=2)  # top-1 p line
    trend_dot = ax_trend.plot([], [], marker="o")[0]    # latest point marker
    ax_trend.set_xlabel("Time (s)")
    ax_trend.set_ylabel("Top-1 p")
    ax_trend.grid(True, alpha=0.3)
    ax_trend.set_title("Top-1 trend")

    # state
    pos = 0                      # position inside 'order' permutation
    autoplay = True              # whether to auto-advance after hold_sec
    playing = False              # whether audio is currently playing
    last_clim = None             # last spectrogram clim (for auto-gain smoothing)
    last_show_ts = None          # wall-clock when current item was shown
    play_start_ts = None         # wall-clock when audio started
    last_ana_ts = None           # last time we ran rolling analysis
    wav_model_sr = None          # current audio (mono) at model SR
    ana_win = float(args.ana_win_sec)                 # analysis window length (sec)
    ana_hop = max(0.02, float(args.ana_hop_sec))      # analysis hop (sec), default 0.5s

    trend_t = []  # times (sec) for trend line
    trend_p = []  # top-1 probabilities for trend line

    # ---- per-item helpers ----
    def _predict_window_ending_at(end_sec: float, win_sec: float):
        """
        Slice a window that ENDS at end_sec (align with current playback time).
        If not enough audio yet, pad on the LEFT so that the RIGHT edge is 'now'.
        """
        assert wav_model_sr is not None
        sr = args.sr
        T = len(wav_model_sr)
        end_sec = max(0.0, min(end_sec, args.duration))
        end_n = int(round(end_sec * sr))
        end_n = max(0, min(end_n, T))
        win_n = int(round(min(win_sec, args.duration) * sr))
        start_n = max(0, end_n - win_n)

        seg = wav_model_sr[start_n:end_n]
        if len(seg) < win_n:
            pad_left = win_n - len(seg)
            seg = np.pad(seg, (pad_left, 0))
        seg_t = torch.from_numpy(seg).unsqueeze(0)  # [1, T]
        logmel = wav_to_logmel(seg_t, sr=sr, mel_transform=mel_t)
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
        with torch.no_grad():
            feats = logmel.unsqueeze(0).to(device)  # [1,1,n_mels,time]
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        mel_img = logmel.squeeze(0).cpu().numpy()
        return probs, mel_img

    def _update_viz(probs: np.ndarray, mel_img: np.ndarray, gt_idx: int):
        nonlocal last_clim
        order_idx = np.argsort(probs)[::-1][:topk]
        top_probs = probs[order_idx]
        top_names = [class_names[j] if j < len(class_names) else f"class_{j}" for j in order_idx]
        pred_idx = int(order_idx[0])
        pred_name = top_names[0]
        gt_name = class_names[int(gt_idx)] if 0 <= int(gt_idx) < len(class_names) else f"class_{int(gt_idx)}"
        correct = (pred_idx == int(gt_idx))

        xmax = max(1.0, float(top_probs[0]) * 1.1)
        ax_bar.set_xlim(0.0, xmax)
        ax_bar.set_yticklabels(top_names)
        for k, (bar, p) in enumerate(zip(bars, top_probs)):
            width = float(p)
            bar.set_width(width)
            label = f"{100.0 * width:.1f}%"
            thresh_inside = 0.18 * xmax
            if width >= thresh_inside:
                x_text = max(0.0, width - 0.02 * xmax)
                ha, color = "right", "white"
            else:
                x_text = min(width + 0.02 * xmax, xmax * 0.98)
                ha, color = "left", "black"
            bar_texts[k].set_text(label)
            bar_texts[k].set_x(x_text)
            bar_texts[k].set_y(k)
            bar_texts[k].set_ha(ha)
            bar_texts[k].set_color(color)

        vmin, vmax = _compute_spec_limits(mel_img, args.spec_auto_gain, args.spec_pmin, args.spec_pmax, last_clim)
        if args.spec_auto_gain:
            im.set_clim(vmin=vmin, vmax=vmax)
            last_clim = (vmin, vmax)
        im.set_data(mel_img)

        color = "green" if correct else "red"
        fig.suptitle(f"Pred: {pred_name}  |  GT: {gt_name}", color=color, fontsize=12)
        fig.canvas.draw_idle()
        return float(top_probs[0])

    def _reset_trend():
        trend_t.clear()
        trend_p.clear()
        trend_line.set_data([], [])
        trend_dot.set_data([], [])
        ax_trend.set_xlim(0.0, args.duration)  # keep trend span matched to duration (now 30s)
        ax_trend.set_ylim(0.0, 1.0)

    def _append_trend(t_sec: float, p_top1: float):
        t_sec = max(0.0, min(t_sec, args.duration))
        trend_t.append(t_sec)
        trend_p.append(float(p_top1))
        trend_line.set_data(trend_t, trend_p)
        trend_dot.set_data([t_sec], [p_top1])

    def show_item_at_pos():
        """Show the item at current 'pos' (mapped through shuffled 'order')."""
        nonlocal wav_model_sr, playing, last_ana_ts, play_start_ts
        _reset_trend()

        ds_idx = _current_ds_index(pos)
        x, y, meta = ds[ds_idx]
        wav_path = Path(meta.get("path", "")) if isinstance(meta, dict) else None
        wav_model_sr = None
        if wav_path and wav_path.exists():
            wav_model_sr = _load_wav_centered(wav_path, args.sr, args.duration)

        if wav_model_sr is not None:
            probs, mel_img = _predict_window_ending_at(0.0, ana_win)
            top1 = _update_viz(probs, mel_img, gt_idx=int(y))
            _append_trend(0.0, top1)
        else:
            logmel = x if isinstance(x, torch.Tensor) else torch.tensor(x)
            if logmel.dim() == 2:
                logmel = logmel.unsqueeze(0)
            with torch.no_grad():
                feats = logmel.unsqueeze(0).to(device)
                logits = model(feats)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top1 = _update_viz(probs, logmel.squeeze(0).cpu().numpy(), gt_idx=int(y))
            _append_trend(0.0, top1)

        if args.play_audio and wav_model_sr is not None:
            try:
                sd.stop()
                wav_out = _prepare_playback_audio(wav_model_sr, model_sr=args.sr, out_sr=out_sr)
                sd.play(wav_out, samplerate=out_sr, device=args.out_device)
                playing = True
                play_start_ts = time.time()
                last_ana_ts = None
            except Exception as e:
                print(f"[warn] audio playback failed: {e}")
                playing = False
                play_start_ts = time.time()
                last_ana_ts = None
        else:
            playing = False
            play_start_ts = time.time()
            last_ana_ts = None

    def _show_and_stamp():
        nonlocal last_show_ts
        show_item_at_pos()
        last_show_ts = time.time()

    def on_key(event):
        nonlocal pos, autoplay, playing
        if event.key in ("left", "left_arrow"):
            autoplay = False
            pos = (pos - 1) % len(order)
            _show_and_stamp()
        elif event.key in ("right", "right_arrow"):
            autoplay = False
            pos = (pos + 1) % len(order)
            _show_and_stamp()
        elif event.key == " ":
            autoplay = not autoplay
            tag = "▶️ autoplay" if autoplay else "⏸ autoplay"
            fig.suptitle(fig._suptitle.get_text() + f"  |  {tag}", fontsize=12)
            fig.canvas.draw_idle()
        elif event.key in ("p", "P"):
            if playing:
                sd.stop()
                playing = False
            else:
                if wav_model_sr is not None:
                    try:
                        sd.stop()
                        wav_out = _prepare_playback_audio(wav_model_sr, model_sr=args.sr, out_sr=out_sr)
                        sd.play(wav_out, samplerate=out_sr, device=args.out_device)
                        playing = True
                    except Exception as e:
                        print(f"[warn] audio playback failed: {e}")
        elif event.key in ("q","Q"):
            plt.close(fig)

    # init
    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    _show_and_stamp()

    # autoplay + rolling updates loop
    try:
        while plt.fignum_exists(fig.number):
            now = time.time()
            plt.pause(args.sleep)

            # Rolling analysis updates while current item is active
            if wav_model_sr is not None and play_start_ts is not None:
                elapsed = max(0.0, min(now - play_start_ts, args.duration))
                if (last_ana_ts is None) or ((now - last_ana_ts) >= ana_hop):
                    probs, mel_img = _predict_window_ending_at(end_sec=elapsed, win_sec=ana_win)
                    ds_idx = _current_ds_index(pos)
                    _, y, _ = ds[ds_idx]
                    top1 = _update_viz(probs, mel_img, gt_idx=int(y))
                    _append_trend(elapsed, top1)
                    last_ana_ts = now

            # Auto-advance by hold_sec
            if (autoplay and (last_show_ts is not None)
                and ((now - last_show_ts) >= hold_sec)):
                pos = (pos + 1) % len(order)
                _show_and_stamp()
    finally:
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass
        sd.stop()

if __name__ == "__main__":
    main()