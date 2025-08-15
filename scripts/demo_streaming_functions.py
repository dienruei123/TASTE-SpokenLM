#!/usr/bin/env python3
"""
Simple demo script to test taste_tokenize and taste_detokenize functions.

This script demonstrates the basic usage of the streaming tokenization and 
detokenization functions using an example audio file.
"""

import os
import torch
import torchaudio
from pathlib import Path

# Import the streaming functions and models
from taste_speech import TasteForCausalLM, TasteProcessor
from taste_speech.streaming import taste_tokenize, taste_detokenize


def main():
    print("=== TASTE Streaming Functions Demo ===")
    
    # Configuration
    model_id = 'MediaTek-Research/Llama-1B-TASTE-Speech-V0'
    audio_path = 'examples/orig/ex01_happy_00209.wav'
    output_path = 'demo_output.wav'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampling_rate = 16000
    
    print(f"Using device: {device}")
    print(f"Input audio: {audio_path}")
    print(f"Output audio: {output_path}")
    
    # Check if input audio exists
    if not os.path.exists(audio_path):
        print(f"Error: Input audio file {audio_path} not found!")
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
    
    # Process the input audio to get text tokens and other information
    print("\n2. Processing input audio...")
    try:
        # Process audio to get all necessary information
        processed_data = processor(
            audio_path, 
            sampling_rate,
            ref_audio_list=[audio_path],
            output_text_info=True
        )
        
        # Extract necessary components
        speaker_embeds = processed_data['speaker_embeds']
        if isinstance(speaker_embeds, torch.Tensor):
            speaker_embeds = speaker_embeds.clone().detach().to(device)
        else:
            speaker_embeds = torch.from_numpy(speaker_embeds).to(device)
            
        asr_token_ids = processed_data['asr_token_ids']
        if isinstance(asr_token_ids, torch.Tensor):
            asr_token_ids = asr_token_ids.clone().detach().to(device)
        else:
            asr_token_ids = torch.from_numpy(asr_token_ids).to(device)
            
        asr_word_ids = processed_data['asr_word_ids']
        if isinstance(asr_word_ids, torch.Tensor):
            asr_word_ids = asr_word_ids.clone().detach().to(device)
        else:
            asr_word_ids = torch.from_numpy(asr_word_ids).to(device)
        
        # Load the actual audio waveform
        audio_waveform, orig_sr = torchaudio.load(audio_path)
        if orig_sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sampling_rate)
            audio_waveform = resampler(audio_waveform)
        audio_waveform = audio_waveform.to(device)
        
        print(f"✓ Audio processed successfully")
        print(f"  - Speaker embeds shape: {speaker_embeds.shape}")
        print(f"  - ASR token IDs shape: {asr_token_ids.shape}")
        print(f"  - Audio waveform shape: {audio_waveform.shape}")
        print(f"  - Text: {processed_data.get('text', [''])[0][:100]}...")
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return
    
    # Step 3: Tokenize audio to TASTE tokens
    print("\n3. Converting audio to TASTE tokens...")
    try:
        taste_tokens = taste_tokenize(
            model=model,
            processor=processor,
            audio=audio_waveform,
            token_ids=asr_token_ids,
            word_ids=asr_word_ids,
            sampling_rate=sampling_rate
        )
        print(f"✓ Audio tokenized successfully")
        print(f"  - TASTE tokens shape: {taste_tokens.shape}")
        
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return
    
    # Step 4: Split tokens into chunks and progressively detokenize
    print("\n4. Converting TASTE tokens back to audio with chunking...")
    try:
        # Configuration for chunking
        chunk_size = max(1, asr_token_ids.shape[1] // 4)  # Split into ~4 chunks
        print(f"  - Total text tokens: {asr_token_ids.shape[1]}")
        print(f"  - Total taste tokens: {taste_tokens.shape[1]}")
        print(f"  - Chunk size: {chunk_size}")
        
        # Initialize previous context (empty at start)
        prev_text_ids = torch.empty(1, 0, dtype=torch.long, device=device)
        prev_taste_ids = torch.empty(1, 0, 4, dtype=torch.long, device=device)  
        prev_speech_ids = torch.empty(1, 0, dtype=torch.long, device=device)
        prev_text_word_ids = None
        prev_audio_ms = 0
        
        # Storage for merged outputs
        all_audio_chunks = []
        total_duration_ms = 0
        
        # Process in chunks
        for i in range(0, asr_token_ids.shape[1], chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, asr_token_ids.shape[1])
            chunk_num = i // chunk_size + 1
            total_chunks = (asr_token_ids.shape[1] + chunk_size - 1) // chunk_size
            
            print(f"  Processing chunk {chunk_num}/{total_chunks} (tokens {chunk_start}:{chunk_end})")
            
            # Extract current chunk
            current_text_ids = asr_token_ids[:, chunk_start:chunk_end]
            current_taste_ids = taste_tokens[:, chunk_start:chunk_end, :]
            current_word_ids = asr_word_ids[:, chunk_start:chunk_end] 
            
            # Call taste_detokenize with previous context
            result = taste_detokenize(
                model=model,
                processor=processor,
                speaker_embeds=speaker_embeds,
                prev_text_ids=prev_text_ids,
                prev_taste_ids=prev_taste_ids,
                prev_speech_ids=prev_speech_ids,
                prev_audio_ms=prev_audio_ms,
                text_ids=current_text_ids,
                taste_ids=current_taste_ids,
                text_word_ids=current_word_ids,
                prev_text_word_ids=prev_text_word_ids,
                out_sampling_rate=sampling_rate
            )
            
            # Store this chunk's audio
            chunk_audio = result['audio_waveform']
            chunk_duration_ms = result['chunk_duration_ms']
            all_audio_chunks.append(chunk_audio)
            total_duration_ms += chunk_duration_ms
            
            print(f"    - Chunk {chunk_num} audio shape: {chunk_audio.shape}")
            print(f"    - Chunk {chunk_num} duration: {chunk_duration_ms} ms")
            
            # Update previous context for next iteration
            prev_text_ids = torch.cat([prev_text_ids, current_text_ids], dim=1)
            prev_taste_ids = torch.cat([prev_taste_ids, current_taste_ids], dim=1)
            if prev_text_word_ids is None:
                prev_text_word_ids = current_word_ids
            else:
                prev_text_word_ids = torch.cat([prev_text_word_ids, current_word_ids], dim=1)
            if 'speech_ids' in result:
                if prev_speech_ids.shape[1] == 0:
                    prev_speech_ids = result['speech_ids']
                else:
                    prev_speech_ids = torch.cat([prev_speech_ids, result['speech_ids']], dim=1)
            prev_audio_ms += chunk_duration_ms
            
            print(f"    - Updated prev_text_ids shape: {prev_text_ids.shape}")
            print(f"    - Updated prev_taste_ids shape: {prev_taste_ids.shape}")
            print(f"    - Cumulative audio_ms: {prev_audio_ms}")

            print(f"  Save...")
            output_audio = torch.cat(all_audio_chunks, dim=-1)  # Concatenate along time dimension
            output_sr = result['sampling_rate']  # Use the last result's sampling rate
            output_audio_cpu = output_audio.cpu()
            torchaudio.save(f'tmp_{i}.wav', output_audio_cpu, output_sr)
            print(f"✓ Output audio saved to tmp_{i}.wav")
        
        # Merge all audio chunks
        print(f"  Merging {len(all_audio_chunks)} audio chunks...")
        output_audio = torch.cat(all_audio_chunks, dim=-1)  # Concatenate along time dimension
        output_sr = result['sampling_rate']  # Use the last result's sampling rate
        
        print(f"✓ Chunked audio detokenized successfully")
        print(f"  - Final output audio shape: {output_audio.shape}")
        print(f"  - Output sampling rate: {output_sr}")
        print(f"  - Total duration: {total_duration_ms} ms")
        
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
    print(f"Successfully tested chunked streaming with prev_text_ids and prev_taste_ids!")
    print(f"You can now listen to the original audio ({audio_path}) and")
    print(f"the reconstructed audio ({output_path}) to compare the quality.")
    print(f"The chunked approach simulates real streaming conditions where")
    print(f"previous context is maintained across multiple detokenization calls.")


if __name__ == "__main__":
    main()