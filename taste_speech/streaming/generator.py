"""
Streaming generation for TASTE-SpokenLM with Iterator pattern and threading support.
"""

import torch
from typing import List, Dict, Iterator, Optional, Callable, TYPE_CHECKING
from threading import Thread
from queue import Queue, Empty
import time

from .segment import TASTESegment, StreamingResult
from .conversation import build_conditional_prompt, check_completion_status

if TYPE_CHECKING:
    from ..modeling_taste import TasteForCausalLM
    from ..processing_taste import TasteProcessor


def streaming_generate(
    model: "TasteForCausalLM",
    processor: "TasteProcessor", 
    input_segments: List[TASTESegment],
    text_top_p: float = 0.3,
    taste_top_p: float = 0.0,
    text_temperature: float = 0.5,
    repetition_penalty: float = 1.1,
    max_length: int = 512,
    eos_id: Optional[int] = None,
    should_emit_segment: Optional[Callable] = None,
    extra_words: int = 32
) -> Iterator[Dict]:
    """
    Streaming generation of conversation responses using Iterator pattern with threading.
    
    Generates multimodal conversation responses incrementally, yielding results as they
    become available. Uses background threading to avoid blocking the main thread.
    
    Args:
        model: TasteForCausalLM model for generation
        processor: TasteProcessor for tokenization 
        input_segments: List of input conversation segments
        text_top_p: Top-p sampling for text generation (default: 0.3)
        taste_top_p: Top-p sampling for TASTE tokens (default: 0.0, greedy)
        text_temperature: Temperature for text sampling (default: 0.5) 
        repetition_penalty: Penalty for repeated tokens (default: 1.1)
        max_length: Maximum generation length (default: 512)
        eos_id: End-of-sequence token ID (optional)
        should_emit_segment: Function to determine when to emit segments (optional)
        extra_words: Extra words to generate beyond minimum (default: 32)
        
    Yields:
        Dict: StreamingResult with keys:
            - is_complete: bool indicating if generation is finished
            - completion_reason: str reason for completion
            - segment: TASTESegment containing generated content
    """
    
    # Setup result queue for thread communication
    result_queue = Queue()
    exception_queue = Queue()
    
    def _generation_worker():
        """Background worker thread for generation."""
        try:
            device = model.device
            
            # Step 1: Build conditional prompt from input segments
            prompt_text_ids, prompt_taste_ids = build_conditional_prompt(
                input_segments, processor.llm_tokenizer
            )
            
            # Move to device
            prompt_text_ids = prompt_text_ids.to(device)
            prompt_taste_ids = prompt_taste_ids.to(device)
            
            # Step 2: Register TasteSampler with generation parameters
            model.spoken_lm.register_taste_sampler(
                llm_tokenizer=processor.llm_tokenizer,
                text_top_p=text_top_p,
                taste_top_p=taste_top_p,
                text_temperature=text_temperature,
                repetition_penalty=repetition_penalty,
            )
            
            # Step 3: Reset sampler state
            model.spoken_lm.taste_sampler.reset(
                extra_words=extra_words,
                has_prefix=True,
                stop_id=eos_id
            )
            
            # Step 4: Initialize generation state
            current_text_ids = prompt_text_ids.clone()
            current_taste_ids = prompt_taste_ids.clone()
            generation_step = 0
            max_steps = max_length
            
            # Track accumulated generation for segment emission
            generated_text_tokens = []
            generated_taste_tokens = []
            
            with torch.no_grad():
                # Step 5: Autoregressive generation loop
                while generation_step < max_steps:
                    
                    # Prepare embeddings for forward pass
                    # This follows the pattern from inference_completion
                    inputs_embeds = model.spoken_lm.prepare_inputs_embeds(
                        current_text_ids, current_taste_ids
                    )
                    
                    # Forward pass through the model
                    outputs = model.spoken_lm.model(inputs_embeds=inputs_embeds)
                    logits = outputs.logits
                    
                    # Extract text and taste logits 
                    text_logits = logits[:, :, :len(processor.llm_tokenizer)]
                    
                    # Get taste logits (this depends on model architecture)
                    # For now, use a placeholder - actual implementation would extract from model outputs
                    taste_logits = torch.randn(1, current_text_ids.size(1), 4, 1024, device=device)
                    
                    # Step 6: Sample next tokens using TasteSampler
                    text_id, taste_ids, action, taste_action = model.spoken_lm.taste_sampler.update(
                        text_logits, taste_logits, current_text_ids
                    )
                    
                    # Convert text_id to tensor format
                    next_text_id = torch.tensor([[text_id]], device=device, dtype=torch.long)
                    
                    # Update current sequences
                    current_text_ids = torch.cat([current_text_ids, next_text_id], dim=1)
                    current_taste_ids = torch.cat([current_taste_ids, taste_ids], dim=1)
                    
                    # Track generated tokens (excluding prompt)
                    generated_text_tokens.append(text_id)
                    generated_taste_tokens.append(taste_ids.squeeze(0).squeeze(0))
                    
                    # Step 7: Check completion status
                    completion_status = check_completion_status(text_id, action)
                    
                    # Step 8: Determine if we should emit a segment
                    should_emit = False
                    if completion_status.is_complete:
                        should_emit = True
                    elif should_emit_segment is not None:
                        should_emit = should_emit_segment(text_id, action, completion_status)
                    elif completion_status.reason == 'segment_end':
                        should_emit = True
                    else:
                        # Emit periodically for streaming experience (every ~10 tokens)
                        should_emit = (len(generated_text_tokens) % 10 == 0 and len(generated_text_tokens) > 0)
                    
                    # Step 9: Emit segment if needed
                    if should_emit and generated_text_tokens:
                        # Create generated text_ids tensor
                        gen_text_ids = torch.tensor([generated_text_tokens], device=device, dtype=torch.long)
                        
                        # Create generated taste_ids tensor
                        if generated_taste_tokens:
                            gen_taste_ids = torch.stack(generated_taste_tokens).unsqueeze(0)
                        else:
                            gen_taste_ids = torch.empty(1, len(generated_text_tokens), 4, device=device, dtype=torch.long)
                        
                        # Create TASTESegment for generated content
                        generated_segment = TASTESegment(
                            role='assistant',
                            modality='audio',  # Default to audio for assistant responses
                            text_ids=gen_text_ids,
                            taste_ids=gen_taste_ids
                        )
                        
                        # Create streaming result
                        result = {
                            'is_complete': completion_status.is_complete,
                            'completion_reason': completion_status.reason,
                            'segment': generated_segment
                        }
                        
                        result_queue.put(result)
                        
                        # If complete, break the generation loop
                        if completion_status.is_complete:
                            break
                        
                        # Reset accumulated tokens if we emitted a partial segment
                        if not completion_status.is_complete:
                            # Keep accumulating for next emission
                            pass
                    
                    generation_step += 1
                
                # Ensure we always emit a final result if generation ends
                if not completion_status.is_complete and generated_text_tokens:
                    # Final emission
                    gen_text_ids = torch.tensor([generated_text_tokens], device=device, dtype=torch.long)
                    gen_taste_ids = torch.stack(generated_taste_tokens).unsqueeze(0) if generated_taste_tokens else torch.empty(1, len(generated_text_tokens), 4, device=device, dtype=torch.long)
                    
                    final_segment = TASTESegment(
                        role='assistant',
                        modality='audio',
                        text_ids=gen_text_ids,
                        taste_ids=gen_taste_ids
                    )
                    
                    final_result = {
                        'is_complete': True,
                        'completion_reason': 'max_length',
                        'segment': final_segment
                    }
                    
                    result_queue.put(final_result)
        
        except Exception as e:
            exception_queue.put(e)
    
    # Step 10: Start background generation thread
    worker_thread = Thread(target=_generation_worker)
    worker_thread.daemon = True
    worker_thread.start()
    
    # Step 11: Yield results as they become available
    while True:
        try:
            # Check for exceptions first
            if not exception_queue.empty():
                raise exception_queue.get_nowait()
            
            # Try to get result with timeout
            result = result_queue.get(timeout=1.0)
            yield result
            
            # If this result is complete, we're done
            if result['is_complete']:
                break
                
        except Empty:
            # Check if thread is still alive
            if not worker_thread.is_alive():
                # Thread died without putting a result - check for exception
                if not exception_queue.empty():
                    raise exception_queue.get_nowait()
                else:
                    # Thread finished without error but no final result
                    break
            # Continue waiting for results
            continue