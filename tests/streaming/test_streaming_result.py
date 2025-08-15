"""
Unit tests for StreamingResult data structure.
"""

import torch
import pytest
from test_streaming_utils import DEVICE
from taste_speech.streaming import TASTESegment, StreamingResult


class TestStreamingResult:
    """Test StreamingResult data structure."""
    
    def test_streaming_result_creation(self):
        """Test StreamingResult creation."""
        text_ids = torch.randint(0, 1000, (1, 10), device=DEVICE)
        segment = TASTESegment(
            role='assistant',
            modality='text',
            text_ids=text_ids
        )
        result = StreamingResult(
            is_complete=False,
            completion_reason='continuing',
            segment=segment
        )
        assert result.is_complete is False
        assert result.completion_reason == 'continuing'
        assert result.segment == segment


# Test runner
if __name__ == '__main__':
    pytest.main([__file__, '-v'])