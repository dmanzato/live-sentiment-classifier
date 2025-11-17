"""
Predict the sentiment/emotion of a single WAV file using a trained model.

Example:
  PYTHONPATH=. python predict.py \
    --data_root /path/to/RAVDESS \
    --checkpoint artifacts/best_model.pt \
    --model resnet18 \
    --mode sentiment \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 --duration 3.0 \
    --wav /path/to/file.wav --topk 3
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from transforms.audio import get_mel_transform, wav_to_logmel
from utils.device import get_device, get_device_name
from utils.models import build_model
from utils.class_map import load_class_map
from datasets.ravdess import SENTIMENTS, EMOTIONS

# Fallback class names
DEFAULT_SENTIMENTS = SENTIMENTS
DEFAULT_EMOTIONS = EMOTIONS


# ------------------------------
# Utilities
# ------------------------------
def _read_json_classmap(p: Path):
    """Return list of class names from a JSON file that is either:
       - {"idx2name": [...]} or
       - ["classA", "classB", ...]
    """
    with open(p, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "idx2name" in data:
        names = data["idx2name"]
    elif isinstance(data, list):
        names = data
    else:
        raise ValueError(f"Unsupported class_map format in {p}")
    if not isinstance(names, list) or not all(isinstance(s, str) for s in names):
        raise ValueError(f"Invalid class_map content in {p}")
    return names


def _detect_num_classes_from_state_dict(state_dict: dict) -> int:
    """Heuristic: read classifier head out_features from checkpoint weights."""
    candidates = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 2:  # linear weight
            out_features, in_features = v.shape
            if re.search(r"(fc|classifier|head).*\.weight$", k):
                candidates.append((k, out_features, True))
            else:
                candidates.append((k, out_features, False))
    for k, out_features, is_head in candidates:
        if is_head:
            return out_features
    if candidates:
        k, out_features, _ = max(candidates, key=lambda t: t[1])
        return out_features
    raise RuntimeError("Could not detect num_classes from checkpoint state_dict.")


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"], state
    return state, None


def load_audio(path: Path, target_sr: int, duration: float) -> torch.Tensor:
    """Load and resample mono audio to fixed length [1, T]. Center-crop/pad.
       NOTE: returns a **CPU** tensor to keep STFT/mel on CPU (avoids MPS window mismatch).
    """
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != target_sr:
        x = resample_poly(x, target_sr, sr).astype(np.float32)
    N = int(duration * target_sr)
    if len(x) < N:
        pad = N - len(x)
        left = pad // 2
        right = pad - left
        x = np.pad(x, (left, right))
    else:
        start = max(0, (len(x) - N) // 2)
        x = x[start:start + N]
    return torch.from_numpy(x).unsqueeze(0)  # CPU [1, T]


# ------------------------------
# Inference
# ------------------------------
def predict_one(
    wav_path: Path,
    model: torch.nn.Module,
    mel_t,                      # keep mel transform on CPU
    device,
    class_names,
    sr,
    topk=3,
    out_dir=None,
    duration: float = 3.0,
):
    """Compute log-mel spectrogram (on CPU), then move to device for inference."""
    # --- keep waveform on CPU to avoid STFT window device mismatch on MPS ---
    wav = load_audio(wav_path, target_sr=sr, duration=duration)            # CPU [1, T]
    logmel = wav_to_logmel(wav, sr=sr, mel_transform=mel_t)                # CPU [1, n_mels, time]
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)              # CPU normalize

    # Now move features to device and run the model
    model.eval()
    with torch.no_grad():
        feats = logmel.unsqueeze(0).to(device)  # [B=1, 1, n_mels, time]
        logits = model(feats)
        probs = torch.softmax(logits, dim=1)[0]
        topv, topi = probs.topk(min(topk, len(class_names)))
        topv, topi = topv.cpu().numpy(), topi.cpu().numpy()

    print(f"\n🎤 File: {wav_path.name}")
    print("Top predictions:")
    for rank, (p, i) in enumerate(zip(topv, topi), start=1):
        label = class_names[i] if 0 <= i < len(class_names) else f"class_{i}"
        print(f"  {rank}. {label:<12} {p*100:5.2f}%")

    # Visualization (still using the CPU tensor)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(logmel.squeeze(0).cpu(), origin="lower", aspect="auto", cmap="magma")
    ax.set_title(f"Predicted: {class_names[topi[0]] if topi[0] < len(class_names) else f'class_{topi[0]}'}")
    plt.xlabel("Time frames")
    plt.ylabel("Mel bins")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{wav_path.stem}_pred.png"
        plt.savefig(out_file)
        print(f"Saved spectrogram to {out_file}")
    else:
        plt.show()
    plt.close(fig)


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict sentiment/emotion of a single WAV file")
    ap.add_argument("--wav", type=str, required=True, help="Path to WAV file")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint. If not specified, uses artifacts/best_model_{mode}.pt (with backward compatibility)")
    ap.add_argument("--model", type=str, default="resnet18", choices=["smallcnn", "resnet18"])
    ap.add_argument("--mode", type=str, default="sentiment", choices=["sentiment", "emotion"],
                    help="Classification mode: 'sentiment' (3 classes) or 'emotion' (8 classes) [default: sentiment]")
    ap.add_argument("--data_root", type=str, required=True, help="Path to RAVDESS root directory")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--duration", type=float, default=3.0, help="Audio duration (s) - RAVDESS files are ~3-4s")
    ap.add_argument("--topk", type=int, default=3, help="Top-K predictions to show")
    ap.add_argument("--out_dir", type=str, default="pred_artifacts")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {get_device_name()} ({device})")
    print(f"Classification mode: {args.mode}")

    # 1) Determine checkpoint path (mode-specific with backward compatibility)
    artifacts_dir = Path("artifacts")
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
            exit(1)
    
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        exit(1)
    
    state_dict, raw_bundle = _load_checkpoint(ckpt_path, device)
    ckpt_num_classes = _detect_num_classes_from_state_dict(state_dict)

    # 2) Load class names (mode-specific with backward compatibility)
    data_root = Path(args.data_root)
    class_names = load_class_map(data_root, artifacts_dir, mode=args.mode)
    
    # 3) If class-name count mismatches checkpoint head, adapt to checkpoint
    if len(class_names) != ckpt_num_classes:
        print(
            f"Note: class count mismatch (class_map={len(class_names)} vs ckpt={ckpt_num_classes}). "
            f"Adapting to checkpoint."
        )
        if args.mode == "sentiment" and ckpt_num_classes == len(DEFAULT_SENTIMENTS):
            class_names = DEFAULT_SENTIMENTS
        elif args.mode == "emotion" and ckpt_num_classes == len(DEFAULT_EMOTIONS):
            class_names = DEFAULT_EMOTIONS
        else:
            class_names = [f"class_{i}" for i in range(ckpt_num_classes)]

    print(f"Using {len(class_names)} classes: {class_names}")

    # 4) Build model with the checkpoint's class count
    model = build_model(args.model, num_classes=len(class_names)).to(device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {ckpt_path} with {len(class_names)} classes.")

    # 5) Mel transform on **CPU** to avoid MPS STFT window mismatch
    mel_t = get_mel_transform(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
    # (Do NOT move mel_t to device; keep it on CPU.)

    # 6) Run prediction
    predict_one(
        Path(args.wav),
        model,
        mel_t,
        device,
        class_names,
        sr=args.sr,
        topk=args.topk,
        out_dir=args.out_dir,
        duration=args.duration,
    )

