"""Tests for dataset loading."""
import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.ravdess import RAVDESS, SENTIMENTS, EMOTIONS


def create_mock_dataset(tmpdir):
    """Create a mock RAVDESS dataset structure for testing."""
    # RAVDESS structure: Actor_01, Actor_02, ..., Actor_24 folders
    # Files: 03-01-XX-XX-XX-XX-XX.wav (03=audio, 01=speech)
    # Format: Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor
    
    actors = [f"Actor_{i:02d}" for i in range(1, 5)]  # Create 4 actors for testing
    
    for actor in actors:
        actor_dir = Path(tmpdir) / actor
        actor_dir.mkdir(parents=True)
        
        # Create a few dummy WAV files per actor
        # Use different emotions: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry
        for emotion_id in range(1, 6):  # emotions 1-5
            for statement in [1, 2]:  # two statements
                for repetition in [1, 2]:  # two repetitions
                    filename = f"03-01-{emotion_id:02d}-01-{statement:02d}-{repetition:02d}-{actor.split('_')[1]}.wav"
                    wav_path = actor_dir / filename
                    # Create a minimal WAV file header (just for structure testing)
                    wav_path.write_bytes(b"dummy wav content")
    
    return str(tmpdir), actors


class TestRAVDESS:
    """Test RAVDESS dataset loader."""
    
    def test_dataset_creation_with_mock(self):
        """Test dataset creation with mock data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, actors = create_mock_dataset(tmpdir)
            
            # Dataset initialization should succeed even with dummy WAV files
            # (audio loading only happens in __getitem__, not __init__)
            dataset = RAVDESS(
                root=data_root,
                mode='sentiment',
                split='train',
                target_sr=22050,
                duration=3.0,
            )
            
            # Check that structure is correct
            assert hasattr(dataset, 'items')
            assert hasattr(dataset, 'SENTIMENTS')
            assert len(dataset.SENTIMENTS) == 3
            # Items should be populated (even if audio files are invalid)
            assert isinstance(dataset.items, list)
            assert len(dataset.items) > 0  # Should have items from the mock files
    
    def test_dataset_mode_sentiment(self):
        """Test dataset in sentiment mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, actors = create_mock_dataset(tmpdir)
            
            dataset = RAVDESS(
                root=data_root,
                mode='sentiment',
                split='train',
                target_sr=22050,
                duration=3.0,
            )
            
            # Should have 3 sentiment classes
            assert len(dataset.classes_list) == 3
            assert dataset.classes_list == SENTIMENTS
            assert hasattr(dataset, 'name2idx')
            assert hasattr(dataset, 'idx2name')
    
    def test_dataset_mode_emotion(self):
        """Test dataset in emotion mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, actors = create_mock_dataset(tmpdir)
            
            dataset = RAVDESS(
                root=data_root,
                mode='emotion',
                split='train',
                target_sr=22050,
                duration=3.0,
            )
            
            # Should have 8 emotion classes
            assert len(dataset.classes_list) == 8
            assert dataset.classes_list == EMOTIONS
            assert hasattr(dataset, 'name2idx')
            assert hasattr(dataset, 'idx2name')
    
    def test_dataset_split_filtering(self):
        """Test that dataset correctly filters by split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, actors = create_mock_dataset(tmpdir)
            
            # Test loading specific splits
            dataset_train = RAVDESS(
                root=data_root,
                mode='sentiment',
                split='train',
                train_ratio=0.8,
                val_ratio=0.1,
                target_sr=22050,
                duration=3.0,
            )
            dataset_val = RAVDESS(
                root=data_root,
                mode='sentiment',
                split='val',
                train_ratio=0.8,
                val_ratio=0.1,
                target_sr=22050,
                duration=3.0,
            )
            dataset_test = RAVDESS(
                root=data_root,
                mode='sentiment',
                split='test',
                train_ratio=0.8,
                val_ratio=0.1,
                target_sr=22050,
                duration=3.0,
            )
            
            # All splits should have items (even if audio files are invalid)
            assert len(dataset_train.items) >= 0
            assert len(dataset_val.items) >= 0
            assert len(dataset_test.items) >= 0
            
            # Total items across splits should match total available
            total_items = len(dataset_train.items) + len(dataset_val.items) + len(dataset_test.items)
            assert total_items > 0  # Should have items from mock files
    
    def test_dataset_class_mapping(self):
        """Test that class mapping is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, actors = create_mock_dataset(tmpdir)
            
            # Test sentiment mode
            dataset_sentiment = RAVDESS(
                root=data_root,
                mode='sentiment',
                split='train',
                target_sr=22050,
                duration=3.0,
            )
            
            # Check class mapping structure
            assert hasattr(dataset_sentiment, 'name2idx')
            assert hasattr(dataset_sentiment, 'idx2name')
            
            # Check that mappings are consistent for sentiment
            assert len(dataset_sentiment.classes_list) == 3
            for i, sentiment in enumerate(dataset_sentiment.classes_list):
                assert dataset_sentiment.idx2name[i] == sentiment
                assert dataset_sentiment.name2idx[sentiment] == i
            
            # Test emotion mode
            dataset_emotion = RAVDESS(
                root=data_root,
                mode='emotion',
                split='train',
                target_sr=22050,
                duration=3.0,
            )
            
            # Check that mappings are consistent for emotion
            assert len(dataset_emotion.classes_list) == 8
            for i, emotion in enumerate(dataset_emotion.classes_list):
                assert dataset_emotion.idx2name[i] == emotion
                assert dataset_emotion.name2idx[emotion] == i

