#!/usr/bin/env python3
"""
Simple demo script to test taste_tokenize and taste_detokenize functions.

This script demonstrates the basic usage of the streaming tokenization and 
detokenization functions using an example audio file.
"""

import os
import time
import torch
import torchaudio
import argparse
from pathlib import Path

# Import the streaming functions and models
from taste_speech import TasteForCausalLM, TasteProcessor
from taste_speech.streaming import taste_tokenize, taste_detokenize
import re
from datetime import timedelta


def parse_srt_file(srt_path):
    """
    Parse SRT subtitle file to extract word timing information.
    
    Args:
        srt_path: Path to the SRT file
        
    Returns:
        List of tuples: [(start_time_ms, end_time_ms, word), ...]
    """
    words_timing = []
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Split by double newlines to get subtitle blocks
    blocks = re.split(r'\n\s*\n', content)
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # Parse timing line (format: HH:MM:SS,mmm --> HH:MM:SS,mmm)
            timing_line = lines[1]
            if '-->' in timing_line:
                start_str, end_str = timing_line.split(' --> ')
                start_ms = srt_time_to_ms(start_str.strip())
                end_ms = srt_time_to_ms(end_str.strip())
                
                # Get word text (join all remaining lines)
                word = ' '.join(lines[2:]).strip()
                words_timing.append((start_ms, end_ms, word))
    
    return words_timing


def srt_time_to_ms(time_str):
    """Convert SRT time format (HH:MM:SS,mmm) to milliseconds."""
    time_str = time_str.replace(',', '.')  # Replace comma with dot for parsing
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    
    total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000
    return int(total_ms)


def extract_audio_segment(audio_waveform, start_ms, end_ms, sampling_rate=16000):
    """Extract audio segment from waveform based on time in milliseconds."""
    start_sample = int(start_ms * sampling_rate / 1000)
    end_sample = int(end_ms * sampling_rate / 1000)
    
    # Ensure we don't exceed audio boundaries
    start_sample = max(0, start_sample)
    end_sample = min(audio_waveform.shape[1], end_sample)
    
    if start_sample >= end_sample:
        # Return empty tensor if invalid range
        return audio_waveform[:, 0:0]
    
    return audio_waveform[:, start_sample:end_sample]


def generate_asr_tokens_from_srt(words_timing, processor, device):
    """
    Generate ASR token IDs and word IDs from SRT file content using ASR tokenizer.
    
    Args:
        words_timing: List of (start_ms, end_ms, word) from SRT file
        processor: TasteProcessor containing audio_tokenizer
        device: Device to put tensors on
        
    Returns:
        Tuple of (asr_token_ids, asr_word_ids)
    """
    # Extract full text from SRT
    full_text = " ".join([word for _, _, word in words_timing])
    
    # Tokenize using ASR tokenizer
    tokenized = processor.audio_tokenizer(
        full_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    asr_token_ids = tokenized['input_ids'].to(device)
    
    # Generate word IDs based on token-to-word mapping
    word_ids = []
    current_word_idx = 0
    
    # For each token, determine which word it belongs to
    decoded_tokens = processor.audio_tokenizer.convert_ids_to_tokens(asr_token_ids[0])
    
    for token_idx, token in enumerate(decoded_tokens):
        if token_idx == 0:  # First token
            word_ids.append(current_word_idx)
        elif token.startswith('▁') or token.startswith(' ') or token.startswith('Ġ'):  # Word boundary markers
            current_word_idx += 1
            if current_word_idx >= len(words_timing):
                current_word_idx = len(words_timing) - 1
            word_ids.append(current_word_idx)
        else:
            word_ids.append(current_word_idx)
    
    # Ensure word_ids has same length as token_ids
    while len(word_ids) < asr_token_ids.shape[1]:
        word_ids.append(current_word_idx)
    
    asr_word_ids = torch.tensor([word_ids], dtype=torch.long, device=device)
    
    return asr_token_ids, asr_word_ids


def create_streaming_chunks(asr_token_ids, asr_word_ids, words_timing, window_size_ms=8000, trigger_words=2):
    """
    Create streaming chunks based on sliding time window and word triggers.
    
    Args:
        asr_token_ids: ASR token IDs tensor of shape (1, seq_len)
        asr_word_ids: Word IDs tensor of shape (1, seq_len)
        words_timing: List of (start_ms, end_ms, word) from SRT file
        window_size_ms: Sliding window size in milliseconds
        trigger_words: Number of words that trigger processing
        
    Returns:
        List of processing chunks with timing and context information
    """
    chunks = []
    processed_words = 0
    
    # Group words by trigger_words
    for i in range(0, len(words_timing), trigger_words):
        # Get current trigger words
        current_words = words_timing[i:i + trigger_words]
        if not current_words:
            continue
            
        # Calculate time range for current words
        current_start_ms = current_words[0][0]
        current_end_ms = current_words[-1][1]
        
        # Calculate sliding window bounds
        window_start_ms = max(0, current_end_ms - window_size_ms)
        window_end_ms = current_end_ms
        
        # Find all words within the sliding window
        context_words = []
        for word_info in words_timing:
            word_start, word_end, word_text = word_info
            if word_end >= window_start_ms and word_start <= window_end_ms:
                context_words.append(word_info)
        
        # Find word IDs that correspond to current processing chunk
        current_word_indices = []
        context_word_indices = []
        
        word_ids_flat = asr_word_ids.squeeze(0)
        unique_word_ids = torch.unique(word_ids_flat, sorted=True)
        
        # Map current trigger words to word indices
        for word_idx in range(i, min(i + trigger_words, len(words_timing))):
            if word_idx < len(unique_word_ids):
                current_word_indices.extend([unique_word_ids[word_idx].item()])
        
        # Map context words to word indices (all words in window)
        window_word_start_idx = 0
        window_word_end_idx = 0
        for word_idx, (word_start, word_end, _) in enumerate(words_timing):
            if word_end >= window_start_ms and word_start <= window_end_ms:
                if window_word_start_idx == 0 or word_idx < window_word_start_idx:
                    window_word_start_idx = word_idx
                if word_idx >= window_word_end_idx:
                    window_word_end_idx = word_idx + 1
        
        # Extract context word IDs
        context_word_ids = unique_word_ids[window_word_start_idx:window_word_end_idx].tolist()
        
        # Find token positions for current processing words
        current_token_mask = torch.isin(word_ids_flat, torch.tensor(current_word_indices, device=word_ids_flat.device))
        current_token_indices = torch.where(current_token_mask)[0]
        
        # Find token positions for context (sliding window)
        context_token_mask = torch.isin(word_ids_flat, torch.tensor(context_word_ids, device=word_ids_flat.device))
        context_token_indices = torch.where(context_token_mask)[0]
        
        if len(current_token_indices) > 0 and len(context_token_indices) > 0:
            # Current processing tokens
            current_start_idx = current_token_indices[0].item()
            current_end_idx = current_token_indices[-1].item() + 1
            
            # Context tokens (sliding window)
            context_start_idx = context_token_indices[0].item()
            context_end_idx = context_token_indices[-1].item() + 1
            
            chunks.append({
                'current_token_range': (current_start_idx, current_end_idx),
                'context_token_range': (context_start_idx, context_end_idx),
                'current_time_range': (current_start_ms, current_end_ms),
                'context_time_range': (window_start_ms, window_end_ms),
                'words_info': current_words,
                'context_words_info': context_words,
                'processed_words_count': processed_words
            })
            
        processed_words += len(current_words)
    
    return chunks


def create_sentence_based_chunks(asr_token_ids, asr_word_ids, taste_tokens, processor, max_sentences_per_chunk=1):
    """
    Create chunks based on sentence boundaries by detecting punctuation.
    
    Args:
        asr_token_ids: ASR token IDs tensor of shape (1, seq_len)
        asr_word_ids: Word IDs tensor of shape (1, seq_len) 
        taste_tokens: TASTE tokens tensor of shape (1, seq_len, vq_dim)
        processor: TasteProcessor to decode tokens
        max_sentences_per_chunk: Maximum number of sentences to include in each chunk
        
    Returns:
        List of tuples: [(chunk_asr_tokens, chunk_asr_word_ids, chunk_taste_tokens), ...]
    """
    chunks = []
    seq_len = asr_token_ids.shape[1]
    
    # Decode tokens to text to find sentence boundaries
    try:
        full_text = processor.audio_tokenizer.decode(asr_token_ids.squeeze(0), skip_special_tokens=True)
    except Exception as e:
        print(e)
        # Fallback to word-based chunking if decoding fails
        print("Warning: Failed to decode tokens for sentence detection, falling back to word-based chunking")
        return create_word_based_chunks(asr_token_ids, asr_word_ids, taste_tokens, words_per_chunk=10)
    
    # Find sentence boundaries by looking for sentence-ending punctuation
    import re
    sentence_endings = ['.', '!', '?', '。', '！', '？', ',']  # Include Chinese punctuation
    
    # Find positions of sentence endings in the text
    sentence_end_positions = []
    for i, char in enumerate(full_text):
        if char in sentence_endings:
            sentence_end_positions.append(i)
    
    if not sentence_end_positions:
        # No sentence endings found, treat as one chunk
        return [(asr_token_ids, asr_word_ids, taste_tokens)]
    
    # Map character positions to token positions
    word_ids_flat = asr_word_ids.squeeze(0)  # Remove batch dimension
    unique_word_ids = torch.unique(word_ids_flat, sorted=True)
    
    # Group sentences into chunks
    current_chunk_start = 0
    sentence_count = 0
    
    for end_pos in sentence_end_positions:
        sentence_count += 1
        
        if sentence_count >= max_sentences_per_chunk:
            # Find the approximate token position for this sentence end
            # Use a heuristic: character position / total characters * total tokens
            approx_token_pos = min(int((end_pos + 1) / len(full_text) * seq_len), seq_len - 1)
            
            # Find the nearest word boundary
            end_token_idx = approx_token_pos
            while end_token_idx < seq_len - 1:
                current_word_id = word_ids_flat[end_token_idx].item()
                next_word_id = word_ids_flat[end_token_idx + 1].item()
                if current_word_id != next_word_id:  # Word boundary found
                    break
                end_token_idx += 1
            
            end_token_idx = min(end_token_idx + 1, seq_len)  # Include the current token
            
            # Create chunk
            if end_token_idx > current_chunk_start:
                chunk_asr_tokens = asr_token_ids[:, current_chunk_start:end_token_idx]
                chunk_asr_word_ids = asr_word_ids[:, current_chunk_start:end_token_idx]
                chunk_taste_tokens = taste_tokens[:, current_chunk_start:end_token_idx, :]
                
                chunks.append((chunk_asr_tokens, chunk_asr_word_ids, chunk_taste_tokens))
                
                current_chunk_start = end_token_idx
                sentence_count = 0
    
    # Add remaining tokens as the last chunk
    if current_chunk_start < seq_len:
        chunk_asr_tokens = asr_token_ids[:, current_chunk_start:]
        chunk_asr_word_ids = asr_word_ids[:, current_chunk_start:]
        chunk_taste_tokens = taste_tokens[:, current_chunk_start:, :]
        
        chunks.append((chunk_asr_tokens, chunk_asr_word_ids, chunk_taste_tokens))
    
    return chunks


def create_word_based_chunks(asr_token_ids, asr_word_ids, taste_tokens, words_per_chunk=2):
    """
    Create chunks based on word boundaries using word_ids.
    
    Args:
        asr_token_ids: ASR token IDs tensor of shape (1, seq_len)
        asr_word_ids: Word IDs tensor of shape (1, seq_len) 
        taste_tokens: TASTE tokens tensor of shape (1, seq_len, vq_dim)
        words_per_chunk: Number of words to include in each chunk
        
    Returns:
        List of tuples: [(chunk_asr_tokens, chunk_asr_word_ids, chunk_taste_tokens), ...]
    """
    chunks = []
    seq_len = asr_token_ids.shape[1]
    
    # Find unique word IDs and their positions
    word_ids_flat = asr_word_ids.squeeze(0)  # Remove batch dimension
    unique_word_ids = torch.unique(word_ids_flat, sorted=True)
    
    # Group tokens by words_per_chunk
    for i in range(0, len(unique_word_ids), words_per_chunk):
        # Get word IDs for this chunk
        chunk_word_ids = unique_word_ids[i:i + words_per_chunk]
        
        # Find token positions that belong to these words
        token_mask = torch.isin(word_ids_flat, chunk_word_ids)
        token_indices = torch.where(token_mask)[0]
        
        if len(token_indices) == 0:
            continue
            
        # Extract chunk data
        start_idx = token_indices[0].item()
        end_idx = token_indices[-1].item() + 1
        
        chunk_asr_tokens = asr_token_ids[:, start_idx:end_idx]
        chunk_asr_word_ids = asr_word_ids[:, start_idx:end_idx]
        chunk_taste_tokens = taste_tokens[:, start_idx:end_idx, :]
        
        chunks.append((chunk_asr_tokens, chunk_asr_word_ids, chunk_taste_tokens))
        
    return chunks


def main(args=None):
    
    print("=== TASTE Streaming Functions Demo ===")
    if args.no_chunk:
        print("Chunking mode: DISABLED")
    elif args.chunk_mode == 'sentence':
        print(f"Chunking mode: ENABLED (sentence-based, {args.sentences_per_chunk} sentences per chunk)")
    else:
        print(f"Chunking mode: ENABLED (word-based, {args.words_per_chunk} words per chunk)")
    
    if not args.no_chunk:
        if args.max_prev_chunks is None:
            print("Previous context: UNLIMITED (all previous chunks)")
        else:
            print(f"Previous context: LIMITED to {args.max_prev_chunks} previous chunks")
    
    # Configuration
    model_id = 'MediaTek-Research/Llama-1B-TASTE-Speech-V0'
    audio_path = args.input
    srt_path = args.srt
    output_path = 'demo_output.wav'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampling_rate = 16000
    
    print(f"Using device: {device}")
    print(f"Input audio: {audio_path}")
    print(f"Input SRT: {srt_path}")
    print(f"Output audio: {output_path}")
    
    # Check if input files exist
    if not os.path.exists(audio_path):
        print(f"Error: Input audio file {audio_path} not found!")
        return
    
    if not os.path.exists(srt_path):
        print(f"Error: Input SRT file {srt_path} not found!")
        return
    
    # Load model and processor
    print("\n1. Loading model and processor...")
    try:
        model = TasteForCausalLM.from_pretrained(model_id, attn_implementation='eager')
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
        
        processor = TasteProcessor.from_pretrained(model_id)
        print("✓ Processor loaded successfully")
    except Exception as e:
        print(f"Error loading model/processor: {e}")
        return
    
    # Process the input audio to extract speaker embeddings and load audio waveform
    print("\n2. Processing input audio for speaker embeddings...")
    try:
        # Process audio to get speaker embeddings only
        processed_data = processor(
            audio_path, 
            sampling_rate,
            ref_audio_list=[audio_path],
            output_text_info=False  # We don't need ASR tokens here since we'll generate them from SRT
        )
        
        # Extract speaker embeddings
        speaker_embeds = processed_data['speaker_embeds']
        if isinstance(speaker_embeds, torch.Tensor):
            speaker_embeds = speaker_embeds.clone().detach().to(device)
        else:
            speaker_embeds = torch.from_numpy(speaker_embeds).to(device)
        
        # Load the actual audio waveform separately
        audio_waveform, orig_sr = torchaudio.load(audio_path)
        # Average stereo to mono if needed
        if audio_waveform.shape[0] > 1:
            audio_waveform = audio_waveform.mean(dim=0, keepdim=True)
        if orig_sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sampling_rate)
            audio_waveform = resampler(audio_waveform)
        audio_waveform = audio_waveform.to(device)
        
        print(f"✓ Audio processed successfully")
        print(f"  - Speaker embeds shape: {speaker_embeds.shape}")
        print(f"  - Audio waveform shape: {audio_waveform.shape}")
        print(f"  - Audio duration: {audio_waveform.shape[1] / sampling_rate:.2f} seconds")
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return
    
    # Step 3: Streaming tokenization with sliding window
    print("\n3. Streaming TASTE tokenization (8s sliding window, 2-word triggers)...")
    try:
        # Parse SRT file to get word timing information
        print("  Parsing SRT file for word timing...")
        words_timing = parse_srt_file(srt_path)
        print(f"  - Found {len(words_timing)} word timings in SRT file")
        
        # Generate ASR tokens from SRT content using ASR tokenizer
        print("  Generating ASR tokens from SRT content...")
        asr_token_ids, asr_word_ids = generate_asr_tokens_from_srt(words_timing, processor, device)
        print(f"  - Generated ASR token IDs shape: {asr_token_ids.shape}")
        print(f"  - Generated ASR word IDs shape: {asr_word_ids.shape}")
        print(f"  - Text content: {processor.audio_tokenizer.decode(asr_token_ids[0], skip_special_tokens=True)}")
        
        # Create streaming chunks (8s window, 2-word triggers)
        streaming_chunks = create_streaming_chunks(
            asr_token_ids, asr_word_ids, words_timing,
            window_size_ms=8000, trigger_words=2
        )
        print(f"  - Created {len(streaming_chunks)} streaming processing chunks")
        
        # Initialize aggregated results
        all_taste_tokens = []
        processed_ranges = []
        
        # Process each chunk with streaming approach
        for chunk_idx, chunk_info in enumerate(streaming_chunks):
            current_start, current_end = chunk_info['current_token_range']
            context_start, context_end = chunk_info['context_token_range']
            current_time_start, current_time_end = chunk_info['current_time_range']
            
            print(f"  Processing chunk {chunk_idx + 1}/{len(streaming_chunks)}")
            print(f"    - Current words: {[w[2] for w in chunk_info['words_info']]}")
            print(f"    - Time range: {current_time_start}ms - {current_time_end}ms")
            print(f"    - Token range: {current_start} - {current_end}")
            print(f"    - Context token range: {context_start} - {context_end}")
            
            # Extract context (sliding window) data for processing
            context_asr_token_ids = asr_token_ids[:, context_start:context_end]
            context_asr_word_ids = asr_word_ids[:, context_start:context_end]
            context_audio = extract_audio_segment(
                audio_waveform, 
                chunk_info['context_time_range'][0], 
                chunk_info['context_time_range'][1], 
                sampling_rate
            )
            
            print(f"    - Context audio shape: {context_audio.shape}")
            print(f"    - Context tokens shape: {context_asr_token_ids.shape}")
            
            # Only process if we have valid audio and tokens
            if context_audio.shape[1] > 0 and context_asr_token_ids.shape[1] > 0:
                # Run taste_tokenize on the context window
                context_taste_tokens = taste_tokenize(
                    model=model,
                    processor=processor,
                    audio=context_audio,
                    token_ids=context_asr_token_ids,
                    word_ids=context_asr_word_ids,
                    sampling_rate=sampling_rate
                )
                
                # Extract only the newly processed portion (current words)
                current_offset_in_context = current_start - context_start
                current_length = current_end - current_start
                
                if current_offset_in_context >= 0 and current_offset_in_context + current_length <= context_taste_tokens.shape[1]:
                    new_taste_tokens = context_taste_tokens[:, current_offset_in_context:current_offset_in_context + current_length, :]
                    all_taste_tokens.append(new_taste_tokens)
                    processed_ranges.append((current_start, current_end))
                    
                    print(f"    - Generated TASTE tokens shape: {new_taste_tokens.shape}")
                    print(f"    - Tokens for range {current_start}-{current_end} processed and stored")
                else:
                    print(f"    - Warning: Invalid offset calculation, skipping this chunk")
            else:
                print(f"    - Warning: Empty audio or tokens, skipping this chunk")
        
        # Aggregate all processed TASTE tokens back into the original sequence
        if all_taste_tokens:
            # Initialize full taste_tokens tensor
            full_seq_len = asr_token_ids.shape[1]
            vq_dim = all_taste_tokens[0].shape[2]
            taste_tokens = torch.zeros(1, full_seq_len, vq_dim, dtype=all_taste_tokens[0].dtype, device=device)
            
            # Fill in the processed portions
            for i, (taste_chunk, (start_idx, end_idx)) in enumerate(zip(all_taste_tokens, processed_ranges)):
                taste_tokens[:, start_idx:end_idx, :] = taste_chunk
                print(f"  - Filled range {start_idx}-{end_idx} with chunk {i+1} results")
            
            print(f"✓ Streaming TASTE tokenization completed successfully")
            print(f"  - Final aggregated TASTE tokens shape: {taste_tokens.shape}")
            print(f"  - Processed {len(all_taste_tokens)} chunks with sliding window approach")
        else:
            print("Error: No TASTE tokens were generated from streaming processing")
            return
        
    except Exception as e:
        print(f"Error during streaming tokenization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Convert TASTE tokens back to audio (with or without chunking)
    if args.no_chunk:
        print("\n4. Converting TASTE tokens back to audio (NO CHUNKING)...")
        try:
            print(f"  - Total text tokens: {asr_token_ids.shape[1]}")
            print(f"  - Total taste tokens: {taste_tokens.shape[1]}")
            print("  - Processing all tokens at once...")
            
            # Process all tokens at once without chunking
            result = taste_detokenize(
                model=model,
                processor=processor,
                speaker_embeds=speaker_embeds,
                asr_token_ids=asr_token_ids,
                asr_taste_ids=taste_tokens,
                asr_word_ids=asr_word_ids,
                out_sampling_rate=sampling_rate
            )
            
            output_audio = result['audio_waveform']
            output_sr = result['sampling_rate']
            total_duration_ms = result['chunk_duration_ms']
            
            print(f"✓ Audio detokenized successfully (no chunking)")
            print(f"  - Output audio shape: {output_audio.shape}")
            print(f"  - Output sampling rate: {output_sr}")
            print(f"  - Total duration: {total_duration_ms} ms")
            
        except Exception as e:
            print(f"Error during non-chunked detokenization: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("\n4. Converting TASTE tokens back to audio with chunking...")
        try:
            if args.chunk_mode == 'sentence':
                print("  Using sentence-based chunking...")
                # Create sentence-based chunks
                chunks = create_sentence_based_chunks(
                    asr_token_ids, asr_word_ids, taste_tokens, processor,
                    max_sentences_per_chunk=args.sentences_per_chunk
                )
                
                print(f"  - Total text tokens: {asr_token_ids.shape[1]}")
                print(f"  - Total taste tokens: {taste_tokens.shape[1]}")
                print(f"  - Total sentence-based chunks: {len(chunks)}")
                print(f"  - Sentences per chunk: {args.sentences_per_chunk}")
                chunk_type = "sentence"
            else:
                print("  Using word-based chunking...")
                # Create word-based chunks
                chunks = create_word_based_chunks(
                    asr_token_ids, asr_word_ids, taste_tokens, 
                    words_per_chunk=args.words_per_chunk
                )
                
                print(f"  - Total text tokens: {asr_token_ids.shape[1]}")
                print(f"  - Total taste tokens: {taste_tokens.shape[1]}")
                print(f"  - Total word-based chunks: {len(chunks)}")
                print(f"  - Words per chunk: {args.words_per_chunk}")
                chunk_type = "word"
            
            # Initialize previous context (empty at start)
            prev_asr_token_ids = torch.empty(1, 0, dtype=torch.long, device=device)
            prev_asr_taste_ids = torch.empty(1, 0, 4, dtype=torch.long, device=device)  
            prev_speech_ids = torch.empty(1, 0, dtype=torch.long, device=device)
            prev_asr_word_ids = None
            prev_audio_ms = 0
            
            # Storage for merged outputs and chunk tracking
            all_audio_chunks = []
            total_duration_ms = 0
            processed_chunks = []  # Store all chunk info for context limiting
            chunk_timing_stats = []  # Store timing stats for each chunk
            
            # Process chunks
            for chunk_num, (current_asr_token_ids, current_asr_word_ids, current_asr_taste_ids) in enumerate(chunks, 1):
                chunk_start_time = time.time()
                print(f"  Processing {chunk_type} chunk {chunk_num}/{len(chunks)}")
                print(f"    - Chunk tokens shape: {current_asr_token_ids.shape}")
                print(f"    - Chunk word_ids: {torch.unique(current_asr_word_ids).tolist()}")
                
                # Call taste_detokenize with previous context
                detokenize_start_time = time.time()
                result = taste_detokenize(
                    model=model,
                    processor=processor,
                    speaker_embeds=speaker_embeds,
                    prev_asr_token_ids=prev_asr_token_ids,
                    prev_asr_taste_ids=prev_asr_taste_ids,
                    prev_speech_ids=prev_speech_ids,
                    prev_audio_ms=prev_audio_ms,
                    asr_token_ids=current_asr_token_ids,
                    asr_taste_ids=current_asr_taste_ids,
                    asr_word_ids=current_asr_word_ids,
                    prev_asr_word_ids=prev_asr_word_ids,
                    out_sampling_rate=sampling_rate
                )
                detokenize_end_time = time.time()
                detokenize_time_ms = (detokenize_end_time - detokenize_start_time) * 1000
                
                # Store this chunk's audio
                chunk_audio = result['audio_waveform']
                chunk_duration_ms = result['chunk_duration_ms']
                all_audio_chunks.append(chunk_audio)
                total_duration_ms += chunk_duration_ms
                
                print(f"    - Chunk {chunk_num} audio shape: {chunk_audio.shape}")
                print(f"    - Chunk {chunk_num} duration: {chunk_duration_ms} ms")
                print(f"    - Chunk {chunk_num} detokenize time: {detokenize_time_ms:.2f} ms")
                
                # Store current chunk info
                chunk_info = {
                    'asr_token_ids': current_asr_token_ids,
                    'asr_taste_ids': current_asr_taste_ids,
                    'asr_word_ids': current_asr_word_ids,
                    'speech_ids': result.get('speech_ids'),
                    'duration_ms': chunk_duration_ms
                }
                processed_chunks.append(chunk_info)
                
                # Apply max_prev_chunks limit if specified
                if args.max_prev_chunks is not None and len(processed_chunks) > args.max_prev_chunks:
                    if args.max_prev_chunks == 0:
                        # Keep only current chunk (no previous chunks)
                        processed_chunks = [processed_chunks[-1]]
                        print(f"    - Limited context to 0 previous chunks")
                    else:
                        # Keep only the last max_prev_chunks chunks
                        processed_chunks = processed_chunks[-args.max_prev_chunks:]
                        print(f"    - Limited context to last {args.max_prev_chunks} chunks")
                
                # Rebuild previous context from stored chunks
                context_start_time = time.time()
                if args.max_prev_chunks == 0:
                    # No previous chunks allowed
                    chunks_to_use = []
                elif args.max_prev_chunks is None or len(processed_chunks) <= args.max_prev_chunks:
                    # Use all processed chunks for unlimited or within limit
                    chunks_to_use = processed_chunks[:-1]  # Exclude current chunk
                else:
                    # Use only the allowed number of previous chunks
                    chunks_to_use = processed_chunks[-(args.max_prev_chunks+1):-1]  # Exclude current chunk
                
                # Rebuild previous context
                if chunks_to_use:
                    prev_asr_token_ids = torch.cat([chunk['asr_token_ids'] for chunk in chunks_to_use], dim=1)
                    prev_asr_taste_ids = torch.cat([chunk['asr_taste_ids'] for chunk in chunks_to_use], dim=1)
                    
                    # Handle word IDs with proper adjustment
                    prev_asr_word_ids = None
                    for chunk in chunks_to_use:
                        if prev_asr_word_ids is None:
                            prev_asr_word_ids = chunk['asr_word_ids']
                        else:
                            max_prev_word_id = prev_asr_word_ids.max().item()
                            min_current_word_id = chunk['asr_word_ids'].min().item()
                            adjusted_asr_word_ids = chunk['asr_word_ids'] - min_current_word_id + max_prev_word_id + 1
                            prev_asr_word_ids = torch.cat([prev_asr_word_ids, adjusted_asr_word_ids], dim=1)
                    
                    # Handle speech IDs
                    prev_speech_ids = torch.empty(1, 0, dtype=torch.long, device=device)
                    for chunk in chunks_to_use:
                        if chunk['speech_ids'] is not None:
                            if prev_speech_ids.shape[1] == 0:
                                prev_speech_ids = chunk['speech_ids']
                            else:
                                prev_speech_ids = torch.cat([prev_speech_ids, chunk['speech_ids']], dim=1)
                    
                    # Calculate cumulative audio duration
                    prev_audio_ms = sum(chunk['duration_ms'] for chunk in chunks_to_use)
                else:
                    # No previous chunks to use (first chunk or all cleared)
                    prev_asr_token_ids = torch.empty(1, 0, dtype=torch.long, device=device)
                    prev_asr_taste_ids = torch.empty(1, 0, 4, dtype=torch.long, device=device)
                    prev_speech_ids = torch.empty(1, 0, dtype=torch.long, device=device)
                    prev_asr_word_ids = None
                    prev_audio_ms = 0
                
                context_end_time = time.time()
                context_time_ms = (context_end_time - context_start_time) * 1000
                
                print(f"    - Updated prev_asr_token_ids shape: {prev_asr_token_ids.shape}")
                print(f"    - Updated prev_asr_taste_ids shape: {prev_asr_taste_ids.shape}")
                print(f"    - Cumulative audio_ms: {prev_audio_ms}")
                print(f"    - Context processing time: {context_time_ms:.2f} ms")

                chunk_end_time = time.time()
                total_chunk_time_ms = (chunk_end_time - chunk_start_time) * 1000
                
                # Store timing stats for this chunk
                chunk_timing = {
                    'chunk_num': chunk_num,
                    'context_time_ms': context_time_ms,
                    'detokenize_time_ms': detokenize_time_ms,
                    'total_time_ms': total_chunk_time_ms,
                    'audio_duration_ms': chunk_duration_ms
                }
                chunk_timing_stats.append(chunk_timing)
                
                print(f"    - Total chunk processing time: {total_chunk_time_ms:.2f} ms")
                print(f"  Save...")
                output_audio = torch.cat(all_audio_chunks, dim=-1)  # Concatenate along time dimension
                output_sr = result['sampling_rate']  # Use the last result's sampling rate
                output_audio_cpu = output_audio.cpu()
                torchaudio.save(f'tmp_chunk_{chunk_num}.wav', output_audio_cpu, output_sr)
                print(f"✓ Output audio saved to tmp_chunk_{chunk_num}.wav")
            
            # Merge all audio chunks
            print(f"  Merging {len(all_audio_chunks)} audio chunks...")
            output_audio = torch.cat(all_audio_chunks, dim=-1)  # Concatenate along time dimension
            output_sr = result['sampling_rate']  # Use the last result's sampling rate
            
            print(f"✓ {chunk_type.capitalize()}-based chunked audio detokenized successfully")
            print(f"  - Final output audio shape: {output_audio.shape}")
            print(f"  - Output sampling rate: {output_sr}")
            print(f"  - Total duration: {total_duration_ms} ms")
            
            # Print detailed timing statistics for each chunk
            print(f"\n=== Chunk Processing Time Statistics ===")
            print(f"{'Chunk':<6} {'Context(ms)':<12} {'Detokenize(ms)':<15} {'Total(ms)':<10} {'Audio(ms)':<10} {'Efficiency':<10}")
            print("-" * 75)
            
            total_context_time = 0
            total_detokenize_time = 0
            total_processing_time = 0
            
            for stats in chunk_timing_stats:
                chunk_num = stats['chunk_num']
                context_time = stats['context_time_ms']
                detokenize_time = stats['detokenize_time_ms']
                total_time = stats['total_time_ms']
                audio_duration = stats['audio_duration_ms']
                
                # Calculate efficiency ratio (audio duration / processing time)
                efficiency = audio_duration / total_time if total_time > 0 else 0
                
                print(f"{chunk_num:<6} {context_time:<12.2f} {detokenize_time:<15.2f} {total_time:<10.2f} {audio_duration:<10.2f} {efficiency:<10.2f}x")
                
                total_context_time += context_time
                total_detokenize_time += detokenize_time
                total_processing_time += total_time
            
            print("-" * 75)
            print(f"{'Total':<6} {total_context_time:<12.2f} {total_detokenize_time:<15.2f} {total_processing_time:<10.2f} {total_duration_ms:<10.2f} {total_duration_ms/total_processing_time:<10.2f}x")
            
            # Calculate average times
            num_chunks = len(chunk_timing_stats)
            if num_chunks > 0:
                avg_context_time = total_context_time / num_chunks
                avg_detokenize_time = total_detokenize_time / num_chunks
                avg_total_time = total_processing_time / num_chunks
                avg_audio_duration = total_duration_ms / num_chunks
                
                print(f"\n=== Average Times Per Chunk ===")
                print(f"Context processing: {avg_context_time:.2f} ms")
                print(f"Detokenization: {avg_detokenize_time:.2f} ms")
                print(f"Total processing: {avg_total_time:.2f} ms")
                print(f"Audio generated: {avg_audio_duration:.2f} ms")
                print(f"Overall efficiency: {total_duration_ms/total_processing_time:.2f}x (audio_duration/processing_time)")
            
        except Exception as e:
            print(f"Error during chunked detokenization: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Step 5: Save the output audio
    print("\n5. Saving output audio...")
    try:
        # Move to CPU for saving
        output_audio_cpu = output_audio.cpu()
        torchaudio.save(output_path, output_audio_cpu, output_sr)
        print(f"✓ Output audio saved to {output_path}")
        
        # Print comparison info
        original_duration = audio_waveform.shape[1] / sampling_rate
        output_duration = output_audio.shape[1] / output_sr
        print(f"\nComparison:")
        print(f"  - Original audio duration: {original_duration:.2f} seconds")
        print(f"  - Output audio duration: {output_duration:.2f} seconds")
        
    except Exception as e:
        print(f"Error saving output audio: {e}")
        return
    
    print("\n=== Demo completed successfully! ===")
    if args.no_chunk:
        print(f"Successfully tested non-chunked processing!")
        print(f"All tokens were processed at once without chunking.")
    else:
        if args.chunk_mode == 'sentence':
            print(f"Successfully tested sentence-based chunked streaming with prev_asr_token_ids and prev_asr_taste_ids!")
            print(f"The sentence-based chunked approach simulates real streaming conditions where")
            print(f"previous context is maintained across multiple detokenization calls, with chunks")
            print(f"aligned to sentence boundaries for more natural speech synthesis.")
        else:
            print(f"Successfully tested word-based chunked streaming with prev_asr_token_ids and prev_asr_taste_ids!")
            print(f"The word-based chunked approach simulates real streaming conditions where")
            print(f"previous context is maintained across multiple detokenization calls, with chunks")
            print(f"aligned to word boundaries for more natural speech synthesis.")
    print(f"You can now listen to the original audio ({audio_path}) and")
    print(f"the reconstructed audio ({output_path}) to compare the quality.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TASTE Streaming Functions Demo')
    parser.add_argument('--input', type=str, default='tests/streaming/samples/sample_01.wav',
                        help='Input audio file path (default: tests/streaming/samples/sample_01.wav)')
    parser.add_argument('--srt', type=str, default='tests/streaming/samples/sample_01.srt',
                        help='SRT subtitle file path (default: tests/streaming/samples/sample_01.srt)')
    parser.add_argument('--no-chunk', action='store_true', 
                        help='Disable chunking in step 4 (process all tokens at once)')
    parser.add_argument('--chunk-mode', choices=['sentence', 'word'], default='sentence',
                        help='Chunking mode: sentence-based or word-based (default: sentence)')
    parser.add_argument('--sentences-per-chunk', type=int, default=1,
                        help='Number of sentences per chunk when using sentence-based chunking (default: 1)')
    parser.add_argument('--words-per-chunk', type=int, default=2,
                        help='Number of words per chunk when using word-based chunking (default: 2)')
    parser.add_argument('--max-prev-chunks', type=int, default=None,
                        help='Maximum number of previous chunks to keep in context (default: None for unlimited)')
    args = parser.parse_args()
    
    main(args)