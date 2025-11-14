# datasets/ravdess.py
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from transforms.audio import get_mel_transform, wav_to_logmel

# RAVDESS emotion codes (from file naming convention)
# File format: Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor.wav
# Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
EMOTIONS: List[str] = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
]

# Sentiment mapping (for sentiment classification mode)
# Positive: happy, calm, surprised
# Negative: sad, angry, fearful, disgust
# Neutral: neutral
SENTIMENTS: List[str] = ["positive", "negative", "neutral"]

EMOTION_TO_SENTIMENT: Dict[str, str] = {
    "neutral": "neutral",
    "calm": "positive",
    "happy": "positive",
    "surprised": "positive",
    "sad": "negative",
    "angry": "negative",
    "fearful": "negative",
    "disgust": "negative",
}


@dataclass(frozen=True)
class Item:
    path: Path
    label: int  # class index
    emotion: str  # emotion name
    sentiment: str  # sentiment label (positive/negative/neutral)
    actor: int  # actor ID (1-24)


class RAVDESS(Dataset):
    """
    RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset loader.
    
    Supports both emotion classification (8 classes) and sentiment classification (3 classes).
    
    Folder layout expected:
        root/
          Actor_01/
            03-01-01-01-01-01-01.wav
            03-01-01-01-01-02-01.wav
            ...
          Actor_02/
            ...
          ...
          Actor_24/
            ...
    
    File naming convention:
        Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor.wav
        - Modality: 01=full, 02=video, 03=audio
        - Vocal: 01=speech, 02=song
        - Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
        - Intensity: 01=normal, 02=strong
        - Statement: 01=Kids, 02=Dogs
        - Repetition: 01=first, 02=second
        - Actor: 01-24
    
    Args:
        mode: "emotion" for 8-class emotion classification, "sentiment" for 3-class sentiment
        split: "train", "val", or "test"
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        target_sr: Target sample rate
        duration: Audio duration in seconds
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Hop length for STFT
        augment: Optional augmentation function
        seed: Random seed for deterministic splits
    
    Returns:
        x: torch.FloatTensor [1, n_mels, time] (log-mel spectrogram)
        y: int (class index)
        meta: dict with keys {"path", "emotion", "sentiment", "actor"}
    """
    
    # Expose for external consumers
    EMOTIONS = EMOTIONS
    SENTIMENTS = SENTIMENTS
    
    def __init__(
        self,
        root: str,
        mode: str = "sentiment",  # "emotion" or "sentiment"
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        target_sr: int = 22050,
        duration: float = 3.0,  # RAVDESS files are typically 3-4 seconds
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        seed: int = 1337,
    ):
        assert mode in {"emotion", "sentiment"}, f"Invalid mode: {mode}"
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        assert 0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0, "Invalid ratios"
        assert (train_ratio + val_ratio) < 1.0, "train_ratio + val_ratio must be < 1.0"
        
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"RAVDESS root not found: {self.root}")
        
        self.mode = mode
        self.split = split
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.target_sr = int(target_sr)
        self.duration = float(duration)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.augment = augment
        self.seed = int(seed)
        
        # Set up class maps based on mode
        if mode == "emotion":
            self.classes_list = EMOTIONS
            self.num_classes = len(EMOTIONS)
        else:  # sentiment
            self.classes_list = SENTIMENTS
            self.num_classes = len(SENTIMENTS)
        
        self.name2idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes_list)}
        self.idx2name: Dict[int, str] = {i: c for c, i in self.name2idx.items()}
        
        # Build dataset items
        all_items: List[Item] = []
        
        # RAVDESS files are organized by actor folders (Actor_01, Actor_02, ..., Actor_24)
        actor_dirs = sorted([d for d in self.root.iterdir() if d.is_dir() and d.name.startswith("Actor_")])
        
        for actor_dir in actor_dirs:
            # Extract actor number
            actor_match = re.search(r'Actor_(\d+)', actor_dir.name)
            if not actor_match:
                continue
            actor_id = int(actor_match.group(1))
            
            # Find all audio files (only speech, not song)
            # Format: 03-01-XX-XX-XX-XX-XX.wav (03=audio, 01=speech)
            files = sorted([p for p in actor_dir.glob("*.wav") if p.is_file()])
            
            for filepath in files:
                # Parse filename: Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor
                parts = filepath.stem.split("-")
                if len(parts) != 7:
                    continue
                
                modality, vocal, emotion_code, intensity, statement, repetition, actor_file = parts
                
                # Only use audio speech files (modality=03, vocal=01)
                if modality != "03" or vocal != "01":
                    continue
                
                # Map emotion code to emotion name
                try:
                    emotion_idx = int(emotion_code) - 1  # Convert 01-08 to 0-7
                    if emotion_idx < 0 or emotion_idx >= len(EMOTIONS):
                        continue
                    emotion = EMOTIONS[emotion_idx]
                except (ValueError, IndexError):
                    continue
                
                # Get sentiment label
                sentiment = EMOTION_TO_SENTIMENT[emotion]
                
                # Determine label based on mode
                if mode == "emotion":
                    label = self.name2idx[emotion]
                else:  # sentiment
                    label = self.name2idx[sentiment]
                
                all_items.append(Item(
                    path=filepath,
                    label=label,
                    emotion=emotion,
                    sentiment=sentiment,
                    actor=actor_id
                ))
        
        # Deterministic split (stratified by actor to avoid data leakage)
        # Group by actor first
        items_by_actor: Dict[int, List[Item]] = {}
        for item in all_items:
            if item.actor not in items_by_actor:
                items_by_actor[item.actor] = []
            items_by_actor[item.actor].append(item)
        
        # Shuffle actors deterministically
        actor_ids = sorted(items_by_actor.keys())
        rng = random.Random(self.seed)
        rng.shuffle(actor_ids)
        
        # Split actors into train/val/test
        n_actors = len(actor_ids)
        n_train_actors = int(round(self.train_ratio * n_actors))
        n_val_actors = int(round(self.val_ratio * n_actors))
        n_train_actors = min(n_train_actors, n_actors)
        n_val_actors = min(n_val_actors, max(0, n_actors - n_train_actors))
        n_test_actors = max(0, n_actors - n_train_actors - n_val_actors)
        
        if self.split == "train":
            split_actors = actor_ids[:n_train_actors]
        elif self.split == "val":
            split_actors = actor_ids[n_train_actors:n_train_actors + n_val_actors]
        else:  # test
            split_actors = actor_ids[n_train_actors + n_val_actors:n_train_actors + n_val_actors + n_test_actors]
        
        # Collect items from split actors
        split_items: List[Item] = []
        for actor_id in split_actors:
            split_items.extend(items_by_actor[actor_id])
        
        # Shuffle items within split for better training
        rng.shuffle(split_items)
        self.items: List[Item] = split_items
        
        # Prepare mel transform ON CPU
        self._mel_t = get_mel_transform(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        
        # Precompute crop strategy
        self._random_crop = (self.split == "train")
        self._num_samples = int(round(self.duration * self.target_sr))
    
    # ------------ public utilities ------------
    
    def __len__(self) -> int:
        return len(self.items)
    
    def classes(self) -> Sequence[str]:
        return [self.idx2name[i] for i in range(len(self.idx2name))]
    
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {c: 0 for c in self.classes_list}
        for it in self.items:
            if self.mode == "emotion":
                counts[it.emotion] = counts.get(it.emotion, 0) + 1
            else:
                counts[it.sentiment] = counts.get(it.sentiment, 0) + 1
        return counts
    
    # ------------ core I/O & transforms ------------
    
    @staticmethod
    def _center_or_random_crop(x: np.ndarray, N: int, random_crop: bool, rng: Optional[random.Random] = None) -> np.ndarray:
        L = len(x)
        if L == N:
            return x
        if L < N:
            pad = N - L
            left = pad // 2
            right = pad - left
            return np.pad(x, (left, right), mode="constant")
        # L > N
        if random_crop:
            r = rng if rng is not None else random
            start = r.randint(0, L - N)
        else:
            start = max(0, (L - N) // 2)
        return x[start:start + N]
    
    def _load_mono(self, path: Path) -> Tuple[np.ndarray, int]:
        # Always_2d gives [T, C] -> we average channels
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if wav.shape[1] > 1:
            wav = wav.mean(axis=1)
        else:
            wav = wav[:, 0]
        return wav, int(sr)
    
    def _resample_if_needed(self, x: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.target_sr:
            return x
        # High-quality rational resampling
        return resample_poly(x, self.target_sr, sr).astype(np.float32)
    
    def _wav_to_logmel(self, x: np.ndarray) -> torch.Tensor:
        # x: mono float32 [T] at target_sr
        with torch.no_grad():
            wav_t = torch.from_numpy(x).unsqueeze(0)  # [1, T] CPU
            logmel = wav_to_logmel(wav_t, sr=self.target_sr, mel_transform=self._mel_t)  # [1, n_mels, time]
            # Per-example standardization helps optimization
            logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
        return logmel
    
    # ------------ dataset API ------------
    
    def __getitem__(self, index: int):
        it = self.items[index]
        
        # Load & preprocess audio
        wav, sr = self._load_mono(it.path)
        wav = self._resample_if_needed(wav, sr)
        
        # Crop/pad duration
        rng = random.Random(self.seed + index) if self._random_crop else None
        wav = self._center_or_random_crop(wav, self._num_samples, self._random_crop, rng)
        
        # To log-mel
        logmel = self._wav_to_logmel(wav)  # [1, n_mels, time]
        
        # Optional SpecAugment (or any augment callable expecting [1,n_mels,time])
        if self.augment is not None:
            logmel = self.augment(logmel)
        
        y = it.label
        meta = {
            "path": str(it.path),
            "emotion": it.emotion,
            "sentiment": it.sentiment,
            "actor": it.actor
        }
        return logmel, y, meta

