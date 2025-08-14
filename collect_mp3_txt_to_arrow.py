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
import warnings


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def validate_audio_file(audio_path: str, min_file_size: int = 1024) -> bool:
    """
    Validate audio file before processing to detect corruption.
    
    Args:
        audio_path: Path to audio file
        min_file_size: Minimum file size in bytes to consider valid
    
    Returns:
        True if file appears valid, False otherwise
    """
    try:
        file_path = Path(audio_path)
        
        # Check if file exists
        if not file_path.exists():
            logging.error(f"Audio file does not exist: {audio_path}")
            return False
        
        # Check file size (skip very small files that are likely corrupted)
        file_size = file_path.stat().st_size
        if file_size < min_file_size:
            logging.warning(f"Audio file too small ({file_size} bytes), likely corrupted: {audio_path}")
            return False
        
        # Try to get basic audio info without loading the full file
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = torchaudio.info(audio_path)
                
                # Check if audio info is reasonable
                if info.num_frames == 0:
                    logging.warning(f"Audio file has zero frames: {audio_path}")
                    return False
                
                if info.sample_rate <= 0:
                    logging.warning(f"Audio file has invalid sample rate: {audio_path}")
                    return False
                    
        except Exception as e:
            logging.warning(f"Cannot get audio info for {audio_path}: {e}")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating audio file {audio_path}: {e}")
        return False


def list_corrupted_files(pairs: List[Tuple[str, str]], min_file_size: int = 1024) -> List[str]:
    """
    List files that fail validation for debugging purposes.
    
    Args:
        pairs: List of (audio_path, text_path) tuples
        min_file_size: Minimum file size in bytes to consider valid
    
    Returns:
        List of corrupted file paths
    """
    corrupted_files = []
    
    logging.info("Checking for corrupted files...")
    for audio_path, text_path in tqdm(pairs, desc="Validating files"):
        if not validate_audio_file(audio_path, min_file_size):
            corrupted_files.append(audio_path)
    
    return corrupted_files


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
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
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
            
            # Ensure it's a proper numpy array
            if not isinstance(audio_array, np.ndarray):
                raise ValueError(f"Failed to convert to numpy array for {audio_path}")
            
            # Check for empty or invalid audio
            if audio_array.size == 0:
                raise ValueError(f"Empty audio array for {audio_path}")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
                raise ValueError(f"Audio contains NaN or infinite values: {audio_path}")
                
            return audio_array, target_sr
            
        except Exception as e:
            logging.debug(f"Torchaudio failed for {audio_path}: {e}")
            try:
                # Fallback to librosa with more robust error handling
                audio_array, loaded_sr = librosa.load(audio_path, sr=target_sr, mono=True)
                
                # Ensure it's a proper numpy array
                if not isinstance(audio_array, np.ndarray):
                    raise ValueError(f"Librosa failed to return numpy array for {audio_path}")
                
                # Check for empty or invalid audio
                if audio_array.size == 0:
                    raise ValueError(f"Empty audio array from librosa for {audio_path}")
                
                # Check for NaN or infinite values
                if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
                    raise ValueError(f"Audio contains NaN or infinite values: {audio_path}")
                    
                return audio_array.astype(np.float32), target_sr
            except Exception as e2:
                logging.error(f"Both torchaudio and librosa failed for {audio_path}. Torchaudio: {e}, Librosa: {e2}")
                raise ValueError(f"Could not load audio file {audio_path}: {e2}")


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
                  text_encoding: str = 'utf-8', min_file_size: int = 1024) -> Dict:
    """
    Create a single sample for the dataset.
    
    Args:
        audio_path: Path to audio file
        text_path: Path to text file
        sample_id: Unique identifier for the sample
        target_sr: Target sample rate
        text_encoding: Text file encoding
        min_file_size: Minimum file size in bytes to consider valid
    
    Returns:
        Dictionary representing a single sample
    """
    # Validate audio file first
    if not validate_audio_file(audio_path, min_file_size):
        raise ValueError(f"Audio file validation failed: {audio_path}")
    
    # Load audio
    audio_array, sample_rate = load_audio(audio_path, target_sr)
    
    # Load text
    text = load_text(text_path, text_encoding)
    
    if not text:
        raise ValueError(f"Empty text file: {text_path}")
    
    # Additional validation
    if not isinstance(audio_array, np.ndarray):
        raise ValueError(f"Audio array is not numpy array: {audio_path}")
    
    if audio_array.size == 0:
        raise ValueError(f"Audio array is empty: {audio_path}")
    
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
                        text_encoding: str = 'utf-8', min_file_size: int = 1024) -> Dataset:
    """
    Create HuggingFace Dataset from audio-text pairs.
    
    Args:
        pairs: List of (audio_path, text_path) tuples
        target_sr: Target sample rate for audio
        text_encoding: Text file encoding
        min_file_size: Minimum file size in bytes to consider valid
    
    Returns:
        HuggingFace Dataset
    """
    samples = []
    failed_count = 0
    
    logging.info(f"Processing {len(pairs)} audio-text pairs...")
    
    for audio_path, text_path in tqdm(pairs, desc="Processing samples"):
        try:
            # Use filename prefix as sample_id
            audio_filename = Path(audio_path).stem
            sample_id = audio_filename
            
            # Create sample with validation
            sample = create_sample(audio_path, text_path, sample_id, target_sr, text_encoding, min_file_size)
            samples.append(sample)
            
        except Exception as e:
            failed_count += 1
            logging.warning(f"Skipping corrupted file {audio_filename}: {e}")
            continue
    
    if not samples:
        raise ValueError("No samples were successfully created - all files may be corrupted")
    
    success_rate = len(samples) / len(pairs) * 100
    logging.info(f"Successfully created {len(samples)} samples out of {len(pairs)} pairs")
    logging.info(f"Success rate: {success_rate:.1f}% ({failed_count} files skipped due to corruption)")
    
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
    parser.add_argument('--list_corrupted', action='store_true',
                       help='List files that fail validation (for debugging)')
    parser.add_argument('--min_file_size', type=int, default=1024,
                       help='Minimum file size in bytes to consider valid (default: 1024)')
    
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
        
        # List corrupted files if requested
        if args.list_corrupted:
            corrupted_files = list_corrupted_files(pairs, args.min_file_size)
            if corrupted_files:
                logging.warning(f"Found {len(corrupted_files)} corrupted files:")
                for corrupted_file in corrupted_files:
                    logging.warning(f"  - {corrupted_file}")
            else:
                logging.info("No corrupted files found!")
            return
        
        # Create Arrow dataset
        logging.info("Creating Arrow dataset...")
        dataset = create_arrow_dataset(
            pairs, 
            args.target_sr, 
            args.text_encoding,
            args.min_file_size
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