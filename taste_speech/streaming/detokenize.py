"""
TASTE detokenization: Convert TASTE tokens to audio waveforms with context continuity.
"""

import torch
import torchaudio
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..modeling_taste import TasteForCausalLM
    from ..processing_taste import TasteProcessor


def _validate_detokenize_inputs(
    speaker_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    text_word_ids: torch.Tensor,
    taste_ids: torch.Tensor,
    prev_text_ids: Optional[torch.Tensor] = None,
    prev_text_word_ids: Optional[torch.Tensor] = None,
    prev_taste_ids: Optional[torch.Tensor] = None,
    prev_speech_ids: Optional[torch.Tensor] = None,
) -> None:
    """Validate inputs for taste_detokenize function."""
    # Validate inputs
    if not isinstance(speaker_embeds, torch.Tensor):
        raise TypeError("speaker_embeds must be a torch.Tensor")
    if not isinstance(text_ids, torch.Tensor):
        raise TypeError("text_ids must be a torch.Tensor")
    if not isinstance(taste_ids, torch.Tensor):
        raise TypeError("taste_ids must be a torch.Tensor")
    if not isinstance(text_word_ids, torch.Tensor):
        raise TypeError("text_word_ids must be a torch.Tensor")
    
    # Validate optional parameters
    if prev_text_ids is not None and not isinstance(prev_text_ids, torch.Tensor):
        raise TypeError("prev_text_ids must be a torch.Tensor or None")
    if prev_taste_ids is not None and not isinstance(prev_taste_ids, torch.Tensor):
        raise TypeError("prev_taste_ids must be a torch.Tensor or None")
    if prev_speech_ids is not None and not isinstance(prev_speech_ids, torch.Tensor):
        raise TypeError("prev_speech_ids must be a torch.Tensor or None")
    if prev_text_word_ids is not None and not isinstance(prev_text_word_ids, torch.Tensor):
        raise TypeError("prev_text_word_ids must be a torch.Tensor or None")
    
    # Validate tensor shapes
    if text_ids.dim() != 2 or text_ids.shape[0] != 1:
        raise ValueError(f"text_ids must have shape (1, seq_len), got {text_ids.shape}")
    if taste_ids.dim() != 3 or taste_ids.shape[0] != 1:
        raise ValueError(f"taste_ids must have shape (1, seq_len, vq_dim), got {taste_ids.shape}")
    if text_word_ids.dim() != 2 or text_word_ids.shape[0] != 1:
        raise ValueError(f"text_word_ids must have shape (1, seq_len), got {text_word_ids.shape}")
    if text_ids.shape[1] != taste_ids.shape[1]:
        raise ValueError(f"text_ids and taste_ids must have same sequence length, got {text_ids.shape[1]} vs {taste_ids.shape[1]}")
    if text_ids.shape[1] != text_word_ids.shape[1]:
        raise ValueError(f"text_ids and text_word_ids must have same sequence length, got {text_ids.shape[1]} vs {text_word_ids.shape[1]}")


def taste_detokenize(
    model: "TasteForCausalLM",
    processor: "TasteProcessor",
    speaker_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    text_word_ids: torch.Tensor,
    taste_ids: torch.Tensor,
    prev_text_ids: Optional[torch.Tensor] = None,
    prev_text_word_ids: Optional[torch.Tensor] = None,
    prev_taste_ids: Optional[torch.Tensor] = None,
    prev_speech_ids: Optional[torch.Tensor] = None,
    prev_audio_ms: int = 0,
    out_sampling_rate: int = 16000,
) -> Dict:
    """
    Convert TASTE tokens to audio waveforms with context-aware synthesis.
    
    Synthesizes audio from TASTE tokens while maintaining continuity with previous
    audio segments. Supports incremental audio generation by properly handling
    context from previous segments.
    
    Args:
        model: TasteForCausalLM model with speech decoding capabilities
        processor: TasteProcessor containing VoiceGenerator
        speaker_embeds: Speaker embedding tensor of shape (1, embed_dim)
        text_ids: Current text token sequence of shape (1, seq_len)
        text_word_ids: Word IDs for current text of shape (1, seq_len)
        taste_ids: Current TASTE tokens of shape (1, seq_len, vq_dim)
        prev_text_ids: Previous text token sequence of shape (1, prev_seq_len) (optional)
        prev_text_word_ids: Word IDs for previous text (optional)
        prev_taste_ids: Previous TASTE tokens of shape (1, prev_seq_len, vq_dim) (optional)
        prev_speech_ids: Previous speech token IDs of shape (1, prev_speech_len) (optional)
        prev_audio_ms: Duration of previous audio in milliseconds (default: 0)
        out_sampling_rate: Target output sampling rate (default: 16000Hz)
        
    Returns:
        Dict containing:
            - audio_waveform: Generated audio tensor of shape (1, T)
            - sampling_rate: Output sampling rate
            - chunk_duration_ms: Duration of current chunk in milliseconds
            - speech_ids: Generated speech token IDs
    """
    
    # Validate all inputs
    _validate_detokenize_inputs(
        speaker_embeds=speaker_embeds,
        text_ids=text_ids,
        text_word_ids=text_word_ids,
        taste_ids=taste_ids,
        prev_text_ids=prev_text_ids,
        prev_text_word_ids=prev_text_word_ids,
        prev_taste_ids=prev_taste_ids,
        prev_speech_ids=prev_speech_ids,
    )
    
    device = model.device
    
    with torch.no_grad():
        # Step 1: Build complete text and TASTE token sequences
        if prev_text_ids is not None and prev_taste_ids is not None and prev_text_ids.numel() > 0:
            full_text_ids = torch.cat([prev_text_ids, text_ids], dim=1)
            full_taste_ids = torch.cat([prev_taste_ids, taste_ids], dim=1)
            
            # Handle word_ids concatenation with proper indexing
            if prev_text_word_ids is not None:
                max_prev_word_id = prev_text_word_ids.max().item()
                min_current_word_id = text_word_ids.min().item()
                # Adjust current word IDs to continue from previous max + 1
                adjusted_text_word_ids = text_word_ids - min_current_word_id + max_prev_word_id + 1
                full_text_word_ids = torch.cat([prev_text_word_ids, adjusted_text_word_ids], dim=1)
            else:
                full_text_word_ids = text_word_ids
        else:
            full_text_ids = text_ids
            full_taste_ids = taste_ids
            full_text_word_ids = text_word_ids
        
        # Move to device
        full_text_ids = full_text_ids.to(device)
        full_taste_ids = full_taste_ids.to(device)
        full_text_word_ids = full_text_word_ids.to(device)
        speaker_embeds = speaker_embeds.to(device)
        
        # Ensure previous tensors are on correct device
        if prev_text_ids is not None and prev_text_ids.numel() > 0:
            prev_text_ids = prev_text_ids.to(device)
        if prev_taste_ids is not None and prev_taste_ids.numel() > 0:
            prev_taste_ids = prev_taste_ids.to(device)
        if prev_text_word_ids is not None:
            prev_text_word_ids = prev_text_word_ids.to(device)
        
        # Step 2: Prepare ASR tokens (use text tokens as ASR tokens for alignment)
        asr_token_ids = full_text_ids
        asr_token_lengths = torch.tensor([full_text_ids.shape[1]], device=device, dtype=torch.long)
        asr_word_ids = full_text_word_ids
        
        # Step 3: Get audio unit embeddings from TASTE tokens
        vq_module = model.audio_tower.vq.rvq
        audio_unit_embeds, audio_unit_lengths = model.spoken_lm.get_audio_embeds_from_taste(
            vq_module=vq_module,
            taste_preds=full_taste_ids,
            asr_token_lengths=asr_token_lengths,
            asr_word_ids=asr_word_ids
        )
        
        # Step 4: Generate speech tokens using extended voice decoder
        speech_decoder_results = model.voice_decoder_generate(
            speaker_embeds=speaker_embeds,
            audio_unit_embeds=audio_unit_embeds,
            audio_unit_lengths=audio_unit_lengths,
            asr_token_ids=asr_token_ids,
            asr_token_lengths=asr_token_lengths,
            prev_speech_ids=prev_speech_ids,
        )

        generated_speech_tokens = speech_decoder_results['speech_token_ids']
        generated_speech_lengths = speech_decoder_results['speech_token_lengths']
        
        # Step 5: Generate audio using VoiceGenerator
        generator = processor.get_generator(device=device)
        flow_embedding = speaker_embeds.to(device)

        # Concatenate previous speech tokens with newly generated ones
        if prev_speech_ids is not None and prev_speech_ids.numel() > 0:
            prev_speech_ids = prev_speech_ids.to(device)
            speech_token_ids = torch.cat([prev_speech_ids, generated_speech_tokens], dim=1)
            prev_speech_length = prev_speech_ids.shape[1]
            speech_lengths = torch.tensor([prev_speech_length + generated_speech_lengths.item()], 
                                        device=device, dtype=torch.long)
        else:
            speech_token_ids = generated_speech_tokens
            speech_lengths = generated_speech_lengths
        
        tts_speech, original_sr = generator.inference(
            speech_token_ids=speech_token_ids,
            speech_token_lengths=speech_lengths,
            flow_embedding=flow_embedding
        )
        
        # Step 6: Resample to target sampling rate if needed
        if original_sr != out_sampling_rate:
            # Ensure tts_speech is on the same device as the resampler
            tts_speech = tts_speech.to(device)
            resampler = torchaudio.transforms.Resample(original_sr, out_sampling_rate).to(device)
            tts_speech = resampler(tts_speech)
        
        # Step 7: Extract current chunk by removing previous audio duration
        if prev_audio_ms > 0:
            # Calculate samples corresponding to previous audio duration
            prev_samples = int(prev_audio_ms * out_sampling_rate / 1000)
            # Crop to keep only current chunk
            if tts_speech.shape[1] > prev_samples:
                tts_speech = tts_speech[:, prev_samples:]
            else:
                # If generated audio is shorter than expected, return silence
                tts_speech = torch.zeros(1, 0, device=device, dtype=tts_speech.dtype)
        
        # Step 8: Calculate duration of current chunk
        chunk_duration_ms = int(tts_speech.shape[1] * 1000 / out_sampling_rate)
    
    return {
        'audio_waveform': tts_speech,
        'sampling_rate': out_sampling_rate,
        'chunk_duration_ms': chunk_duration_ms,
        'speech_ids': generated_speech_tokens,
    }
