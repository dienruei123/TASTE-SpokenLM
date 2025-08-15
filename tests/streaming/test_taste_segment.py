"""
Unit tests for TASTESegment data structure.
"""

import torch
import pytest
from test_streaming_utils import DEVICE
from taste_speech.streaming import TASTESegment


class TestTASTESegment:
    """Test TASTESegment data structure validation and functionality."""
    
    def test_text_segment_creation(self):
        """Test pure text segment creation."""
        text_ids = torch.randint(0, 1000, (1, 10), device=DEVICE)
        segment = TASTESegment(
            role='user',
            modality='text', 
            text_ids=text_ids
        )
        assert segment.taste_ids is None
        assert segment.role == 'user'
        assert segment.modality == 'text'
        assert torch.equal(segment.text_ids, text_ids)
        
    def test_audio_segment_creation(self):
        """Test audio segment creation with taste_ids."""
        text_ids = torch.randint(0, 1000, (1, 10), device=DEVICE)
        taste_ids = torch.randint(0, 1024, (1, 10, 4), device=DEVICE)
        segment = TASTESegment(
            role='assistant',
            modality='audio',
            text_ids=text_ids,
            taste_ids=taste_ids
        )
        assert segment.role == 'assistant'
        assert segment.modality == 'audio'
        assert torch.equal(segment.text_ids, text_ids)
        assert torch.equal(segment.taste_ids, taste_ids)
        
    def test_audio_segment_validation_missing_taste_ids(self):
        """Test that audio segments require taste_ids."""
        text_ids = torch.randint(0, 1000, (1, 10), device=DEVICE)
        with pytest.raises(ValueError, match="Audio segments must have taste_ids"):
            TASTESegment(
                role='user',
                modality='audio',
                text_ids=text_ids,
                taste_ids=None
            )
    
    def test_invalid_role_validation(self):
        """Test validation of invalid roles."""
        text_ids = torch.randint(0, 1000, (1, 10), device=DEVICE)
        with pytest.raises(ValueError, match="Invalid role"):
            TASTESegment(
                role='invalid_role',
                modality='text',
                text_ids=text_ids
            )
    
    def test_invalid_modality_validation(self):
        """Test validation of invalid modalities."""
        text_ids = torch.randint(0, 1000, (1, 10), device=DEVICE)
        with pytest.raises(ValueError, match="Invalid modality"):
            TASTESegment(
                role='user',
                modality='invalid_modality',
                text_ids=text_ids
            )
    
    def test_batch_size_validation(self):
        """Test that batch size must be 1."""
        text_ids = torch.randint(0, 1000, (2, 10), device=DEVICE)  # Wrong batch size
        with pytest.raises(ValueError, match="batch size must be 1"):
            TASTESegment(
                role='user',
                modality='text',
                text_ids=text_ids
            )
    
    def test_sequence_length_alignment(self):
        """Test that taste_ids and text_ids must have matching sequence lengths."""
        text_ids = torch.randint(0, 1000, (1, 10), device=DEVICE)
        taste_ids = torch.randint(0, 1024, (1, 5, 4), device=DEVICE)  # Wrong sequence length
        with pytest.raises(ValueError, match="sequence lengths must match"):
            TASTESegment(
                role='user',
                modality='audio',
                text_ids=text_ids,
                taste_ids=taste_ids
            )


# Test runner
if __name__ == '__main__':
    pytest.main([__file__, '-v'])