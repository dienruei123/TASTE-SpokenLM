#!/usr/bin/env python3
"""
Script to collect MP3 and TXT files and convert them to Apache Arrow format
compatible with TASTE-SpokenLM dataset processing pipeline.

This script pairs MP3 audio files with corresponding TXT transcription files
and creates an Arrow dataset that can be used with taste_speech/data/dataset.py.
"""

import argparse
import os
import logging
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torchaudio
import librosa
from datasets import Dataset
from tqdm import tqdm
import json


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def find_audio_text_pairs(input_dir: str, audio_extensions: List[str] = None, 
                          text_extensions: List[str] = None) -> List[Tuple[str, str]]:
    """
    Find matching audio and text file pairs in the input directory.
    
    Args:
        input_dir: Directory containing audio and text files
        audio_extensions: List of audio file extensions to search for
        text_extensions: List of text file extensions to search for
    
    Returns:
        List of tuples (audio_file_path, text_file_path)
    """
    if audio_extensions is None:
        audio_extensions = ['.mp3', '.wav', '.flac']
    if text_extensions is None:
        text_extensions = ['.txt']
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all audio files
    audio_files = {}
    for ext in audio_extensions:
        pattern = f"**/*{ext}"
        for audio_file in input_path.glob(pattern):
            basename = audio_file.stem
            audio_files[basename] = str(audio_file)
    
    # Find all text files and match with audio files
    pairs = []
    for ext in text_extensions:
        pattern = f"**/*{ext}"
        for text_file in input_path.glob(pattern):
            basename = text_file.stem
            if basename in audio_files:
                pairs.append((audio_files[basename], str(text_file)))
                logging.info(f"Found pair: {basename}")
            else:
                logging.warning(f"No matching audio file for text: {text_file}")
    
    if not pairs:
        raise ValueError(f"No matching audio-text pairs found in {input_dir}")
    
    logging.info(f"Found {len(pairs)} audio-text pairs")
    return pairs


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        # Try torchaudio first
        waveform, orig_sr = torchaudio.load(audio_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            waveform = resampler(waveform)
        
        # Convert to numpy and squeeze
        audio_array = waveform.squeeze().numpy().astype(np.float32)
        return audio_array, target_sr
        
    except Exception as e:
        logging.warning(f"Torchaudio failed for {audio_path}, trying librosa: {e}")
        try:
            # Fallback to librosa
            audio_array, orig_sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return audio_array.astype(np.float32), target_sr
        except Exception as e:
            logging.error(f"Failed to load audio {audio_path}: {e}")
            raise


def load_text(text_path: str, encoding: str = 'utf-8') -> str:
    """
    Load text from file.
    
    Args:
        text_path: Path to text file
        encoding: Text file encoding
    
    Returns:
        Text content as string
    """
    try:
        with open(text_path, 'r', encoding=encoding) as f:
            text = f.read().strip()
        return text
    except UnicodeDecodeError:
        # Try different encodings
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(text_path, 'r', encoding=enc) as f:
                    text = f.read().strip()
                logging.warning(f"Used encoding {enc} for {text_path}")
                return text
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode text file {text_path}")


def create_sample(audio_path: str, text_path: str, sample_id: str, target_sr: int = 16000, 
                  text_encoding: str = 'utf-8') -> Dict:
    """
    Create a single sample for the dataset.
    
    Args:
        audio_path: Path to audio file
        text_path: Path to text file
        sample_id: Unique identifier for the sample
        target_sr: Target sample rate
        text_encoding: Text file encoding
    
    Returns:
        Dictionary representing a single sample
    """
    # Load audio
    audio_array, sample_rate = load_audio(audio_path, target_sr)
    
    # Load text
    text = load_text(text_path, text_encoding)
    
    if not text:
        raise ValueError(f"Empty text file: {text_path}")
    
    # Create sample in the expected format
    sample = {
        '__key__': sample_id,
        'mp3': {
            'array': audio_array,
            'sampling_rate': sample_rate
        },
        'json': {
            'text': text
        },
        # Initialize empty fields that will be filled by other processing scripts
        's3_token': [],
        'spk_emb': []
    }
    
    return sample


def create_arrow_dataset(pairs: List[Tuple[str, str]], target_sr: int = 16000, 
                        text_encoding: str = 'utf-8', key_prefix: str = "sample") -> Dataset:
    """
    Create HuggingFace Dataset from audio-text pairs.
    
    Args:
        pairs: List of (audio_path, text_path) tuples
        target_sr: Target sample rate for audio
        text_encoding: Text file encoding
        key_prefix: Prefix for generating sample keys
    
    Returns:
        HuggingFace Dataset
    """
    samples = []
    
    for i, (audio_path, text_path) in enumerate(tqdm(pairs, desc="Processing samples")):
        try:
            sample_id = f"{key_prefix}_{i:06d}"
            sample = create_sample(audio_path, text_path, sample_id, target_sr, text_encoding)
            samples.append(sample)
            
        except Exception as e:
            logging.error(f"Failed to process {audio_path}, {text_path}: {e}")
            continue
    
    if not samples:
        raise ValueError("No samples were successfully created")
    
    logging.info(f"Successfully created {len(samples)} samples")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(samples)
    return dataset


def validate_output_compatibility(dataset: Dataset) -> bool:
    """
    Validate that the created dataset is compatible with the TASTE processing pipeline.
    
    Args:
        dataset: HuggingFace Dataset to validate
    
    Returns:
        True if compatible, raises ValueError if not
    """
    required_columns = ['__key__', 'mp3', 'json', 's3_token', 'spk_emb']
    
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Missing required column: {col}")
    
    # Check first sample structure
    sample = dataset[0]
    
    # Validate mp3 structure
    if not isinstance(sample['mp3'], dict):
        raise ValueError("mp3 field should be a dictionary")
    if 'array' not in sample['mp3'] or 'sampling_rate' not in sample['mp3']:
        raise ValueError("mp3 field should contain 'array' and 'sampling_rate'")
    
    # Validate json structure
    if not isinstance(sample['json'], dict):
        raise ValueError("json field should be a dictionary")
    if 'text' not in sample['json']:
        raise ValueError("json field should contain 'text'")
    
    # Validate audio array
    audio_array = sample['mp3']['array']
    if not isinstance(audio_array, np.ndarray):
        raise ValueError("Audio array should be numpy array")
    
    # Validate text
    text = sample['json']['text']
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text should be non-empty string")
    
    logging.info("Dataset validation passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert MP3 and TXT files to Arrow format")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing MP3 and TXT files')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output Arrow file path')
    parser.add_argument('--target_sr', type=int, default=16000,
                       help='Target sample rate for audio (default: 16000)')
    parser.add_argument('--text_encoding', type=str, default='utf-8',
                       help='Text file encoding (default: utf-8)')
    parser.add_argument('--key_prefix', type=str, default='sample',
                       help='Prefix for sample keys (default: sample)')
    parser.add_argument('--audio_extensions', nargs='+', default=['.mp3', '.wav', '.flac'],
                       help='Audio file extensions to search for')
    parser.add_argument('--text_extensions', nargs='+', default=['.txt'],
                       help='Text file extensions to search for')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate output compatibility with TASTE pipeline')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Find audio-text pairs
        logging.info(f"Searching for audio-text pairs in {args.input_dir}")
        pairs = find_audio_text_pairs(
            args.input_dir, 
            args.audio_extensions, 
            args.text_extensions
        )
        
        # Create Arrow dataset
        logging.info("Creating Arrow dataset...")
        dataset = create_arrow_dataset(
            pairs, 
            args.target_sr, 
            args.text_encoding, 
            args.key_prefix
        )
        
        # Validate if requested
        if args.validate:
            logging.info("Validating dataset compatibility...")
            validate_output_compatibility(dataset)
        
        # Save to file
        logging.info(f"Saving dataset to {args.output_file}")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        dataset.save_to_disk(args.output_file)
        
        logging.info(f"Successfully created Arrow dataset with {len(dataset)} samples")
        logging.info(f"Output saved to: {args.output_file}")
        
    except Exception as e:
        logging.error(f"Script failed: {e}")
        raise


if __name__ == "__main__":
    main()