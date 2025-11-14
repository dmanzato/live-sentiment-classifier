"""Shared class map loading and saving utilities for live-sentiment-classifier."""
import json
from pathlib import Path
from typing import List
from utils.logging import get_logger

logger = get_logger("class_map")

# Standard RAVDESS sentiments (for sentiment mode)
RAVDESS_SENTIMENTS = ['positive', 'negative', 'neutral']
# Standard RAVDESS emotions (for emotion mode)
RAVDESS_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def load_class_map(data_root: Path, artifacts_dir: Path, mode: str = "sentiment") -> List[str]:
    """
    Load class names in index order.
    
    Priority:
    1. artifacts/class_map.json with key 'idx2name' (list or dict)
    2. RAVDESS dataset structure to derive class names
    
    Args:
        data_root: Path to RAVDESS root directory.
        artifacts_dir: Path to artifacts directory (where class_map.json would be).
        mode: "sentiment" for 3-class sentiment, "emotion" for 8-class emotion.
    
    Returns:
        List of class names in index order (idx 0, 1, 2, ...).
    
    Raises:
        FileNotFoundError: If neither class_map.json nor RAVDESS structure is found.
    
    Example:
        >>> from pathlib import Path
        >>> from utils.class_map import load_class_map
        >>> idx2name = load_class_map(Path("/path/to/RAVDESS"), Path("artifacts"), mode="sentiment")
        >>> print(idx2name[0])  # First class name
    """
    # Try to load from class_map.json first
    cm_path = artifacts_dir / "class_map.json"
    if cm_path.exists():
        try:
            with open(cm_path, "r") as f:
                data = json.load(f)
            if isinstance(data.get("idx2name"), list):
                logger.debug(f"Loaded class map from {cm_path} (list format)")
                return data["idx2name"]
            elif isinstance(data.get("idx2name"), dict):
                # Convert dict to list, sorted by key
                idx2name = [v for k, v in sorted(data["idx2name"].items(), key=lambda kv: int(kv[0]))]
                logger.debug(f"Loaded class map from {cm_path} (dict format)")
                return idx2name
        except Exception as e:
            logger.warning(f"Failed to load class_map.json: {e}, falling back to RAVDESS defaults")
    
    # Fallback: use standard RAVDESS classes
    if mode == "emotion":
        logger.debug(f"Using standard RAVDESS emotions: {len(RAVDESS_EMOTIONS)} classes")
        return RAVDESS_EMOTIONS.copy()
    else:  # sentiment
        logger.debug(f"Using standard RAVDESS sentiments: {len(RAVDESS_SENTIMENTS)} classes")
        return RAVDESS_SENTIMENTS.copy()


def save_class_map(artifacts_dir: Path, idx2name: List[str]) -> None:
    """
    Save class map to artifacts/class_map.json.
    
    Args:
        artifacts_dir: Path to artifacts directory.
        idx2name: List of class names in index order.
    
    Example:
        >>> from pathlib import Path
        >>> from utils.class_map import save_class_map
        >>> save_class_map(Path("artifacts"), ["positive", "negative", "neutral"])
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cm_path = artifacts_dir / "class_map.json"
    
    data = {
        "idx2name": idx2name,
        "num_classes": len(idx2name)
    }
    
    with open(cm_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved class map to {cm_path} ({len(idx2name)} classes)")

