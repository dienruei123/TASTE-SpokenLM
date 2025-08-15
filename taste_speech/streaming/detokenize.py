"""
TASTE detokenization: Convert TASTE tokens to audio waveforms with context continuity.
"""

import torch
import torchaudio
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..modeling_taste import TasteForCausalLM
    from ..processing_taste import TasteProcessor


def taste_detokenize(
    model: "TasteForCausalLM",
    processor: "TasteProcessor",
    speaker_embeds: torch.Tensor,
    prev_text_ids: torch.Tensor,
    prev_taste_ids: torch.Tensor,
    prev_speech_ids: torch.Tensor,
    prev_audio_ms: int,
    text_ids: torch.Tensor,
    taste_ids: torch.Tensor,
    text_word_ids: torch.Tensor,
    prev_text_word_ids: Optional[torch.Tensor] = None,
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
        prev_text_ids: Previous text token sequence of shape (1, prev_seq_len)
        prev_taste_ids: Previous TASTE tokens of shape (1, prev_seq_len, vq_dim)
        prev_speech_ids: Previous speech token IDs of shape (1, prev_speech_len)
        prev_audio_ms: Duration of previous audio in milliseconds
        text_ids: Current text token sequence of shape (1, seq_len)
        taste_ids: Current TASTE tokens of shape (1, seq_len, vq_dim)
        text_word_ids: Word IDs for current text of shape (1, seq_len)
        prev_text_word_ids: Word IDs for previous text (optional)
        out_sampling_rate: Target output sampling rate (default: 16000Hz)
        
    Returns:
        Dict containing:
            - audio_waveform: Generated audio tensor of shape (1, T)
            - sampling_rate: Output sampling rate
            - chunk_duration_ms: Duration of current chunk in milliseconds
            - speech_ids: Generated speech token IDs
    """
    
    # Validate inputs
    if not isinstance(speaker_embeds, torch.Tensor):
        raise TypeError("speaker_embeds must be a torch.Tensor")
    if not isinstance(text_ids, torch.Tensor):
        raise TypeError("text_ids must be a torch.Tensor")
    if not isinstance(taste_ids, torch.Tensor):
        raise TypeError("taste_ids must be a torch.Tensor")
    
    device = model.device
    
    with torch.no_grad():
        # Step 1: Build complete text and TASTE token sequences
        if prev_text_ids.numel() > 0:
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
        speech_decoder_results = _voice_decoder_generate_extended(
            model,
            speaker_embeds=speaker_embeds,
            audio_unit_embeds=audio_unit_embeds,
            audio_unit_lengths=audio_unit_lengths,
            asr_token_ids=asr_token_ids,
            asr_token_lengths=asr_token_lengths,
            prev_speech_ids=prev_speech_ids if prev_speech_ids.numel() > 0 else None,
        )
        
        current_speech_tokens = speech_decoder_results['speech_token_ids']
        current_speech_lengths = speech_decoder_results['speech_token_lengths']
        
        # Step 5: Generate audio using VoiceGenerator
        generator = processor.get_generator(device=device)
        flow_embedding = speaker_embeds.to(device)
        
        tts_speech, original_sr = generator.inference(
            speech_token_ids=current_speech_tokens,
            speech_token_lengths=current_speech_lengths,
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
        'speech_ids': current_speech_tokens,
    }


def _voice_decoder_generate_extended(
    model,
    speaker_embeds,
    audio_unit_embeds,
    audio_unit_lengths,
    asr_token_ids,
    asr_token_lengths,
    prev_speech_ids=None,  # New parameter for context continuity
):
    """
    Extended version of _voice_decoder_generate that supports context continuity.
    
    This function extends the original voice decoder generation to support
    previous speech context, enabling smooth audio transitions in streaming.
    """
    
    # Prepare conditional embeds
    (
        sos_eos_emb,
        speaker_embeds_processed, 
        audio_text_token_encoded,
        audio_text_token_len, 
        task_id_emb
    ) = model.speech_decoder.prepare_conditional_embeds(
        speaker_embeds,
        audio_unit_embeds,
        audio_unit_lengths,
        asr_token_ids,
        asr_token_lengths
    )

    # Handle previous speech context
    if prev_speech_ids is not None and prev_speech_ids.numel() > 0:
        # Convert previous speech IDs to embeddings
        prev_speech_embeds = model.speech_decoder.speech_embedding(prev_speech_ids)
        
        # Prepare input with previous context
        speech_lm_input, speech_lm_input_len = model.speech_decoder.pad_unpad_sequence(
            sos_eos_emb,
            speaker_embeds_processed, 
            audio_text_token_encoded,
            audio_text_token_len, 
            task_id_emb,
            prev_speech_embeds,  # Add previous speech context
            torch.tensor([prev_speech_ids.shape[1]], device=prev_speech_ids.device),
            padding_side='right'
        )
        # Start generation from the end of previous context
        initial_offset = speech_lm_input.size(1) - 1
    else:
        # Original logic without previous context
        speech_lm_input, speech_lm_input_len = model.speech_decoder.pad_unpad_sequence(
            sos_eos_emb,
            speaker_embeds_processed, 
            audio_text_token_encoded,
            audio_text_token_len, 
            task_id_emb,
            padding_side='right'
        )
        initial_offset = 0

    # Generation parameters
    beam_size = 1
    sampling = 25
    max_token_text_ratio = 20
    min_token_text_ratio = 2

    min_len = int(speech_lm_input_len[0] * min_token_text_ratio)
    max_len = int(speech_lm_input_len[0] * max_token_text_ratio)

    device = speech_lm_input.device

    out_tokens = []
    offset = initial_offset  # Use adjusted offset for context
    att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=device), torch.zeros((0, 0, 0, 0), device=device)
    
    # Generation loop
    for i in range(max_len):
        y_pred, att_cache, cnn_cache = model.speech_decoder.llm.forward_chunk(
            speech_lm_input, offset=offset, required_cache_size=-1, 
            att_cache=att_cache, cnn_cache=cnn_cache,
            att_mask=torch.tril(torch.ones((1, speech_lm_input.shape[1], speech_lm_input.shape[1]), device=device)).to(torch.bool)
        )
        logp = model.speech_decoder.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
        top_ids = model.speech_decoder.sampling_ids(logp.squeeze(dim=0), sampling, beam_size, ignore_eos=True if i < min_len else False).item()
        if top_ids == model.speech_decoder.speech_token_size:
            break
        out_tokens.append(top_ids)
        offset += speech_lm_input.size(1)
        speech_lm_input = model.speech_decoder.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    # Combine with previous speech tokens if they exist
    if prev_speech_ids is not None and prev_speech_ids.numel() > 0:
        final_speech_tokens = torch.cat([prev_speech_ids, torch.tensor([out_tokens], dtype=torch.int32, device=device)], dim=1)
    else:
        final_speech_tokens = torch.tensor([out_tokens], dtype=torch.int32, device=device)
    
    final_speech_lengths = torch.tensor([final_speech_tokens.shape[1]], dtype=torch.int32, device=device)

    return {
        'speech_token_ids': final_speech_tokens,
        'speech_token_lengths': final_speech_lengths,
    }