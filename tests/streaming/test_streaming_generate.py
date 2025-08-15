"""
Unit tests for streaming_generate function.
"""

import torch
import pytest
from unittest.mock import Mock
from test_streaming_utils import DEVICE
from taste_speech.streaming import TASTESegment, streaming_generate


class TestStreamingGenerate:
    """Test streaming_generate function."""
    
    def create_mock_model_and_processor(self):
        """Create comprehensive mock model and processor."""
        model = Mock()
        model.device = DEVICE
        
        # Mock spoken_lm with register_taste_sampler and taste_sampler
        model.spoken_lm = Mock()
        
        # Mock taste_sampler
        taste_sampler = Mock()
        taste_sampler.reset = Mock()
        taste_sampler.update = Mock()
        # Configure update to return completion after a few steps
        taste_sampler.update.side_effect = [
            (100, torch.tensor([[[1, 2, 3, 4]]], dtype=torch.long, device=DEVICE), 'continue_at_word_start', 'sample'),
            (101, torch.tensor([[[2, 3, 4, 5]]], dtype=torch.long, device=DEVICE), 'continue_not_at_word_start', 'sample'),
            (102, torch.tensor([[[3, 4, 5, 6]]], dtype=torch.long, device=DEVICE), 'terminate', 'sample'),
        ]
        
        model.spoken_lm.register_taste_sampler = Mock()
        model.spoken_lm.taste_sampler = taste_sampler
        model.spoken_lm.prepare_inputs_embeds = Mock()
        model.spoken_lm.prepare_inputs_embeds.return_value = torch.randn(1, 10, 768, device=DEVICE)
        model.spoken_lm.model = Mock()
        
        # Mock model outputs
        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(1, 10, 32000, device=DEVICE)  # LLM vocab size
        model.spoken_lm.model.return_value = mock_outputs
        
        processor = Mock()
        processor.llm_tokenizer = Mock()
        processor.llm_tokenizer.__len__ = Mock(return_value=32000)
        processor.llm_tokenizer.encode = Mock()
        processor.llm_tokenizer.encode.side_effect = lambda x, **kwargs: [1] if x else [0]
        processor.llm_tokenizer.unk_token_id = 0
        
        return model, processor
    
    def test_streaming_output_format(self):
        """Test streaming output format and basic functionality."""
        model, processor = self.create_mock_model_and_processor()
        
        input_segments = [
            TASTESegment(
                role='system',
                modality='text',
                text_ids=torch.randint(0, 1000, (1, 20), device=DEVICE)
            )
        ]
        
        # Test Iterator behavior
        results = []
        for result in streaming_generate(model, processor, input_segments):
            assert 'is_complete' in result
            assert 'completion_reason' in result
            assert 'segment' in result
            assert isinstance(result['segment'], TASTESegment)
            
            results.append(result)
            if result['is_complete']:
                break
        
        assert len(results) > 0
        assert results[-1]['is_complete'] is True


# Test runner
if __name__ == '__main__':
    pytest.main([__file__, '-v'])