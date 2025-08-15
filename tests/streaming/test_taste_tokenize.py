"""
Unit tests for taste_tokenize function.
"""

import torch
import pytest
from unittest.mock import Mock
from test_streaming_utils import DEVICE
from taste_speech.streaming import taste_tokenize


class TestTasteTokenize:
    """Test taste_tokenize function."""
    
    def create_mock_model_and_processor(self):
        """Create mock model and processor for testing."""
        model = Mock()
        model.device = DEVICE
        
        # Mock parameters() method to return an iterator with mock parameter having dtype
        mock_param = Mock()
        mock_param.dtype = torch.float32
        model.parameters = Mock(return_value=iter([mock_param]))
        
        def mock_audio_tower(**kwargs):
            # Get sequence length from asr_token_ids to ensure alignment
            asr_token_ids = kwargs.get('asr_token_ids')
            seq_len = asr_token_ids.shape[1]
            return {
                'quantized_indices': torch.randint(0, 1024, (1, seq_len, 4), device=DEVICE)
            }
        
        model.audio_tower = Mock()
        model.audio_tower.side_effect = mock_audio_tower
        
        processor = Mock()
        processor.whisper_feature_extractor = Mock()
        processor.whisper_feature_extractor.return_value = (
            torch.randn(1, 80, 100, device=DEVICE).cpu().numpy(),  # audio_features (convert to numpy)
            [100]  # audio_feature_lengths
        )
        
        return model, processor
    
    def test_basic_tokenization(self):
        """Test basic audio tokenization functionality."""
        model, processor = self.create_mock_model_and_processor()
        
        # Generate test data
        audio = torch.randn(1, 16000, device=DEVICE)  # 1 second audio
        text_ids = torch.randint(0, 1000, (1, 50), device=DEVICE)
        
        result = taste_tokenize(model, processor, audio, text_ids)
        
        # Verify output shape
        assert result.shape[0] == 1  # batch size
        assert result.shape[1] == text_ids.shape[1]  # sequence length alignment
        assert result.shape[2] > 0  # VQ dimension
        
    def test_batch_size_alignment_error(self):
        """Test batch size alignment requirement."""
        model, processor = self.create_mock_model_and_processor()
        
        audio = torch.randn(2, 16000, device=DEVICE)  # Mismatched batch size
        text_ids = torch.randint(0, 1000, (1, 50), device=DEVICE)
        
        with pytest.raises(AssertionError, match="Batch size must be 1"):
            taste_tokenize(model, processor, audio, text_ids)
    
    def test_invalid_input_types(self):
        """Test validation of input types."""
        model, processor = self.create_mock_model_and_processor()
        
        with pytest.raises(TypeError, match="audio must be a torch.Tensor"):
            taste_tokenize(model, processor, [1, 2, 3], torch.tensor([[1, 2, 3]]))
        
        with pytest.raises(TypeError, match="token_ids must be a torch.Tensor"):
            taste_tokenize(model, processor, torch.randn(1, 1000, device=DEVICE), [1, 2, 3])


# Test runner
if __name__ == '__main__':
    pytest.main([__file__, '-v'])