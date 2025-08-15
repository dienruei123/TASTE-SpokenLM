"""
Unit tests for TASTE-SpokenLM streaming components.
Supports CPU-only testing via NO_CUDA environment variable.
"""

import torch
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add the parent directory to path to import taste_speech
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taste_speech.streaming import TASTESegment, StreamingResult, taste_tokenize, streaming_generate, taste_detokenize

# NO_CUDA toggle for CPU-only testing (e.g., on Mac without GPU)
NO_CUDA = os.environ.get('NO_CUDA', 'True').lower() in ('true', '1', 'yes', 'on')

# Device configuration based on NO_CUDA flag
if NO_CUDA:
    DEVICE = torch.device('cpu')
    print("Running tests in CPU-only mode (NO_CUDA=True)")
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running tests with device: {DEVICE}")

# Force CPU-only mode if CUDA is not available regardless of NO_CUDA setting
if not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
    NO_CUDA = True


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