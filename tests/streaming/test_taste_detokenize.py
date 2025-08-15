"""
Unit tests for taste_detokenize function.
"""

import torch
import pytest
from unittest.mock import Mock
from test_streaming_utils import DEVICE
from taste_speech.streaming import taste_detokenize


class TestTasteDetokenize:
    """Test taste_detokenize function."""
    
    def create_mock_model_and_processor(self):
        """Create mock model and processor for detokenization testing."""
        model = Mock()
        model.device = DEVICE
        
        # Mock audio_tower
        model.audio_tower = Mock()
        model.audio_tower.vq = Mock()
        model.audio_tower.vq.rvq = Mock()
        
        # Mock spoken_lm
        model.spoken_lm = Mock()
        model.spoken_lm.get_audio_embeds_from_taste = Mock()
        model.spoken_lm.get_audio_embeds_from_taste.return_value = (
            torch.randn(1, 20, 256, device=DEVICE),  # audio_unit_embeds
            torch.tensor([20], dtype=torch.long, device=DEVICE)  # audio_unit_lengths
        )
        
        # Mock speech_decoder
        model.speech_decoder = Mock()
        model.speech_decoder.prepare_conditional_embeds = Mock()
        model.speech_decoder.prepare_conditional_embeds.return_value = (
            torch.randn(1, 1, 256, device=DEVICE),  # sos_eos_emb
            torch.randn(1, 256, device=DEVICE),     # speaker_embeds
            torch.randn(1, 20, 256, device=DEVICE), # audio_text_token_encoded
            torch.tensor([20], device=DEVICE),      # audio_text_token_len
            torch.randn(1, 1, 256, device=DEVICE)   # task_id_emb
        )
        
        model.speech_decoder.pad_unpad_sequence = Mock()
        model.speech_decoder.pad_unpad_sequence.return_value = (
            torch.randn(1, 25, 256, device=DEVICE),  # speech_lm_input
            torch.tensor([25], device=DEVICE)         # speech_lm_input_len
        )
        
        model.speech_decoder.llm = Mock()
        model.speech_decoder.llm.forward_chunk = Mock()
        model.speech_decoder.llm.forward_chunk.return_value = (
            torch.randn(1, 25, 256, device=DEVICE),  # y_pred
            torch.zeros((0, 0, 0, 0), device=DEVICE), # att_cache
            torch.zeros((0, 0, 0, 0), device=DEVICE)  # cnn_cache
        )
        
        model.speech_decoder.llm_decoder = Mock()
        model.speech_decoder.llm_decoder.return_value = torch.randn(1, 1000, device=DEVICE)
        
        model.speech_decoder.sampling_ids = Mock()
        model.speech_decoder.sampling_ids.side_effect = [
            torch.tensor(100, device=DEVICE), 
            torch.tensor(101, device=DEVICE), 
            torch.tensor(102, device=DEVICE), 
            torch.tensor(1000, device=DEVICE)  # EOS token
        ]
        
        model.speech_decoder.speech_token_size = 1000
        model.speech_decoder.speech_embedding = Mock()
        model.speech_decoder.speech_embedding.weight = torch.randn(1001, 256, device=DEVICE)
        
        # Mock voice_decoder_generate method
        model.voice_decoder_generate = Mock()
        model.voice_decoder_generate.return_value = {
            'speech_token_ids': torch.tensor([[100, 101, 102]], device=DEVICE),
            'speech_token_lengths': torch.tensor([3], device=DEVICE)
        }
        
        processor = Mock()
        processor.get_generator = Mock()
        
        # Mock generator
        generator = Mock()
        generator.inference = Mock()
        generator.inference.return_value = (
            torch.randn(1, 22050, device=DEVICE),  # 1 second at 22050Hz
            22050  # sampling rate
        )
        processor.get_generator.return_value = generator
        
        return model, processor
    
    def test_audio_generation(self):
        """Test audio generation functionality."""
        model, processor = self.create_mock_model_and_processor()
        
        # Prepare test data
        speaker_embeds = torch.randn(1, 256, device=DEVICE)
        text_ids = torch.randint(0, 1000, (1, 20), device=DEVICE)
        taste_ids = torch.randint(0, 1024, (1, 20, 4), device=DEVICE)
        text_word_ids = torch.arange(text_ids.shape[1], device=DEVICE).unsqueeze(0)
        
        result = taste_detokenize(
            model, processor, speaker_embeds,
            prev_text_ids=torch.empty(1, 0, dtype=torch.long, device=DEVICE),
            prev_taste_ids=torch.empty(1, 0, 4, dtype=torch.long, device=DEVICE),
            prev_speech_ids=torch.empty(1, 0, dtype=torch.long, device=DEVICE),
            prev_audio_ms=0,
            text_ids=text_ids,
            taste_ids=taste_ids,
            text_word_ids=text_word_ids
        )
        
        assert 'audio_waveform' in result
        assert 'sampling_rate' in result
        assert 'chunk_duration_ms' in result
        assert 'speech_ids' in result
        assert result['audio_waveform'].ndim == 2  # (1, T)
        assert result['sampling_rate'] == 16000
        assert isinstance(result['chunk_duration_ms'], int)
    
    def test_invalid_input_types(self):
        """Test validation of input types."""
        model, processor = self.create_mock_model_and_processor()
        
        with pytest.raises(TypeError, match="speaker_embeds must be a torch.Tensor"):
            taste_detokenize(
                model, processor, [1, 2, 3],  # Invalid speaker_embeds
                torch.empty(1, 0, dtype=torch.long, device=DEVICE), torch.empty(1, 0, 4, dtype=torch.long, device=DEVICE),
                torch.empty(1, 0, dtype=torch.long, device=DEVICE), 0,
                torch.tensor([[1, 2, 3]], device=DEVICE), torch.tensor([[[1, 2, 3, 4]]], device=DEVICE),
                torch.tensor([[0, 1, 2]], device=DEVICE)
            )


# Test runner
if __name__ == '__main__':
    pytest.main([__file__, '-v'])