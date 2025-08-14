"""
TASTE tokenization: Convert audio waveform to TASTE tokens aligned with text.
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..modeling_taste import TasteForCausalLM
    from ..processing_taste import TasteProcessor


def taste_tokenize(
    model: "TasteForCausalLM",
    processor: "TasteProcessor", 
    audio: torch.Tensor,
    text_ids: torch.Tensor,
    sampling_rate: int = 16000
) -> torch.Tensor:
    """
    Convert audio waveform to TASTE token indices aligned with text tokens.
    
    This function extracts vector quantized (VQ) representations from audio
    that are temporally aligned with the provided text token sequence.
    
    Args:
        model: TasteForCausalLM model with audio_tower for VQ extraction
        processor: TasteProcessor with feature extraction capabilities
        audio: Input audio waveform tensor of shape (1, num_samples) 
        text_ids: Text token IDs tensor of shape (1, seq_len)
        sampling_rate: Audio sampling rate in Hz (default: 16000)
    
    Returns:
        torch.Tensor: TASTE token indices of shape (1, seq_len, vq_dim)
        
    Raises:
        AssertionError: If batch sizes of audio and text_ids don't match
        ValueError: If audio or text_ids have incorrect shapes
    """
    
    # Validate inputs
    if not isinstance(audio, torch.Tensor):
        raise TypeError("audio must be a torch.Tensor")
    if not isinstance(text_ids, torch.Tensor):
        raise TypeError("text_ids must be a torch.Tensor")
    
    if audio.ndim != 2:
        raise ValueError("audio must have shape (1, num_samples)")
    if text_ids.ndim != 2:  
        raise ValueError("text_ids must have shape (1, seq_len)")
    
    # Ensure batch size alignment (critical requirement from PRP)
    assert audio.size(0) == text_ids.size(0) == 1, "Batch size must be 1 for both audio and text_ids"
    
    device = model.device
    dtype = next(model.parameters()).dtype
    
    # Move inputs to model device
    audio = audio.to(device)
    text_ids = text_ids.to(device)
    
    with torch.no_grad():
        # Step 1: Extract audio features using the processor's feature extractor
        # Convert audio to numpy for WhisperFrontend compatibility
        audio_np = audio.cpu().numpy()[0]  # Remove batch dimension for processor
        
        # Use processor's whisper feature extractor
        audio_features, audio_feature_lengths = processor.whisper_feature_extractor(
            torch.tensor([audio_np], dtype=torch.float32), [audio_np.shape[0]]
        )
        
        # Convert to tensors and move to device
        audio_features = torch.tensor(audio_features, dtype=dtype, device=device)
        audio_feature_lengths = torch.tensor(audio_feature_lengths, dtype=torch.long, device=device)
        
        # Step 2: Prepare ASR token inputs (use text_ids as ASR tokens for alignment)
        asr_token_ids = text_ids
        asr_token_lengths = torch.tensor([text_ids.shape[1]], dtype=torch.long, device=device)
        
        # Step 3: Use audio_tower to encode and quantize audio features
        # Following the pattern from extract_vq method
        audio_encoded = model.audio_tower(
            asr_token_ids=asr_token_ids,
            asr_token_lengths=asr_token_lengths,
            audio_features=audio_features,
            audio_feature_lengths=audio_feature_lengths,
            asr_word_ids=None,  # Simplified version, not using word alignment
        )
        
        # Step 4: Extract quantized indices (TASTE tokens)
        taste_indices = audio_encoded['quantized_indices']
        
        # Ensure output shape is (1, seq_len, vq_dim)
        if taste_indices.size(1) != text_ids.size(1):
            # Handle potential sequence length mismatch by interpolation or padding
            if taste_indices.size(1) < text_ids.size(1):
                # Pad with special pad_audio_unit_embed values [-1,-1,-1,-1]
                pad_size = text_ids.size(1) - taste_indices.size(1) 
                pad_values = torch.full((1, pad_size, taste_indices.size(2)), -1, 
                                      dtype=taste_indices.dtype, device=device)
                taste_indices = torch.cat([taste_indices, pad_values], dim=1)
            else:
                # Truncate if too long
                taste_indices = taste_indices[:, :text_ids.size(1), :]
    
    return taste_indices