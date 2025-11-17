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
    1. artifacts/class_map_{mode}.json (mode-specific, new format)
    2. artifacts/class_map.json (backward compatibility, old format)
    3. RAVDESS dataset structure to derive class names
    
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
    def _load_from_json(cm_path: Path) -> List[str]:
        """Helper to load class map from JSON file."""
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
            logger.warning(f"Failed to load {cm_path}: {e}")
        return None
    
    # Try mode-specific class map first (new format)
    cm_path_mode = artifacts_dir / f"class_map_{mode}.json"
    if cm_path_mode.exists():
        result = _load_from_json(cm_path_mode)
        if result is not None:
            return result
    
    # Try old format for backward compatibility
    cm_path_old = artifacts_dir / "class_map.json"
    if cm_path_old.exists():
        result = _load_from_json(cm_path_old)
        if result is not None:
            logger.info(f"Loaded class map from old format ({cm_path_old}), consider using mode-specific format")
            return result
    
    # Fallback: use standard RAVDESS classes
    if mode == "emotion":
        logger.debug(f"Using standard RAVDESS emotions: {len(RAVDESS_EMOTIONS)} classes")
        return RAVDESS_EMOTIONS.copy()
    else:  # sentiment
        logger.debug(f"Using standard RAVDESS sentiments: {len(RAVDESS_SENTIMENTS)} classes")
        return RAVDESS_SENTIMENTS.copy()


def save_class_map(artifacts_dir: Path, idx2name: List[str], class_map_path: Path = None) -> None:
    """
    Save class map to artifacts directory.
    
    Args:
        artifacts_dir: Path to artifacts directory.
        idx2name: List of class names in index order.
        class_map_path: Optional specific path to save class map. If None, uses artifacts_dir/class_map.json.
    
    Example:
        >>> from pathlib import Path
        >>> from utils.class_map import save_class_map
        >>> save_class_map(Path("artifacts"), ["positive", "negative", "neutral"])
        >>> # Or with specific path:
        >>> save_class_map(Path("artifacts"), ["positive", "negative", "neutral"], 
        ...                Path("artifacts/class_map_sentiment.json"))
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if class_map_path is None:
        cm_path = artifacts_dir / "class_map.json"
    else:
        cm_path = class_map_path
    
    data = {
        "idx2name": idx2name,
        "num_classes": len(idx2name)
    }
    
    with open(cm_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved class map to {cm_path} ({len(idx2name)} classes)")

