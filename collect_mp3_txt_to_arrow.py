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
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import json
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import psutil
from collections import defaultdict
import time


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
            # Try alternative metadata extraction methods
            logging.debug(f"torchaudio.info failed for {audio_path}: {e}")
            
            # Try librosa for metadata (fallback)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Just get duration without loading audio
                    duration = librosa.get_duration(path=audio_path)
                    if duration <= 0:
                        logging.warning(f"Audio file has zero duration: {audio_path}")
                        return False
            except Exception as e2:
                # Final fallback - try to actually load a small portion
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Try to load just first 0.1 second to verify file is readable
                        y, sr = librosa.load(audio_path, duration=0.1, sr=None)
                        if len(y) == 0 or sr <= 0:
                            logging.warning(f"Audio file failed verification load: {audio_path}")
                            return False
                except Exception as e3:
                    logging.warning(f"All metadata extraction methods failed for {audio_path}")
                    logging.debug(f"  torchaudio: {e}")
                    logging.debug(f"  librosa duration: {e2}")
                    logging.debug(f"  librosa load: {e3}")
                    return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating audio file {audio_path}: {e}")
        return False


def diagnose_mp3_files(pairs: List[Tuple[str, str]], max_samples: int = 10) -> None:
    """
    Run detailed diagnostics on MP3 files to understand format issues.
    
    Args:
        pairs: List of (audio_path, text_path) tuples
        max_samples: Maximum number of files to diagnose in detail
    """
    import subprocess
    
    logging.info(f"Running detailed MP3 diagnostics on up to {max_samples} files...")
    
    for i, (audio_path, text_path) in enumerate(pairs[:max_samples]):
        print(f"\n{'='*60}")
        print(f"DIAGNOSING FILE {i+1}/{min(max_samples, len(pairs))}: {Path(audio_path).name}")
        print(f"{'='*60}")
        
        # File info
        file_path = Path(audio_path)
        print(f"File size: {file_path.stat().st_size:,} bytes")
        print(f"File extension: {file_path.suffix}")
        
        # Try different methods
        methods_results = {}
        
        # 1. torchaudio.info
        try:
            info = torchaudio.info(audio_path)
            methods_results['torchaudio'] = f"✓ Success: {info.num_frames} frames, {info.sample_rate}Hz"
        except Exception as e:
            methods_results['torchaudio'] = f"✗ Failed: {e}"
        
        # 2. librosa duration
        try:
            duration = librosa.get_duration(path=audio_path)
            methods_results['librosa_duration'] = f"✓ Success: {duration:.2f}s"
        except Exception as e:
            methods_results['librosa_duration'] = f"✗ Failed: {e}"
        
        # 3. librosa load (small sample)
        try:
            y, sr = librosa.load(audio_path, duration=0.1, sr=None)
            methods_results['librosa_load'] = f"✓ Success: loaded {len(y)} samples at {sr}Hz"
        except Exception as e:
            methods_results['librosa_load'] = f"✗ Failed: {e}"
        
        # 4. ffprobe (if available)
        try:
            result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                                   '-show_format', audio_path], 
                                   capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                duration = float(info['format']['duration'])
                bit_rate = info['format'].get('bit_rate', 'unknown')
                methods_results['ffprobe'] = f"✓ Success: {duration:.2f}s, {bit_rate} bit_rate"
            else:
                methods_results['ffprobe'] = f"✗ Failed: ffprobe error"
        except Exception as e:
            methods_results['ffprobe'] = f"✗ Failed: {e}"
        
        # Print results
        for method, result in methods_results.items():
            print(f"{method:20}: {result}")
        
        # Check if file starts with valid MP3 header
        try:
            with open(audio_path, 'rb') as f:
                header = f.read(10)
                if header.startswith(b'\xff\xfb') or header.startswith(b'ID3'):
                    print(f"{'MP3 header':20}: ✓ Valid MP3 header detected")
                else:
                    print(f"{'MP3 header':20}: ✗ No valid MP3 header (first 10 bytes: {header.hex()})")
        except Exception as e:
            print(f"{'MP3 header':20}: ✗ Cannot read file: {e}")


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


def discover_random_string_folders(input_dir: str, min_depth: int = 2) -> List[str]:
    """
    Discover all random-string folders for chunked processing.
    
    Args:
        input_dir: Root directory to search in
        min_depth: Minimum depth to look for random-string folders
        
    Returns:
        List of random-string folder paths
    """
    random_folders = []
    input_path = Path(input_dir)
    
    logging.info("Discovering random-string folders for chunked processing...")
    
    # Walk through directory structure and identify random-string folders
    for root, dirs, files in os.walk(input_dir):
        current_depth = len(Path(root).relative_to(input_path).parts)
        
        # Look for folders at appropriate depth that contain audio/text files
        if current_depth >= min_depth:
            # Check if this directory contains audio/text files (leaf folder)
            has_audio = any(f.lower().endswith(('.mp3', '.wav', '.flac')) for f in files)
            has_text = any(f.lower().endswith('.txt') for f in files)
            
            if has_audio and has_text:
                random_folders.append(root)
                # Don't traverse deeper into this folder since it's a leaf
                dirs.clear()
    
    logging.info(f"Found {len(random_folders)} random-string folders to process")
    return sorted(random_folders)


def find_audio_text_pairs_optimized(input_dir: str, audio_extensions: List[str] = None, 
                                   text_extensions: List[str] = None, 
                                   max_workers: int = None) -> List[Tuple[str, str]]:
    """
    Efficiently find matching audio and text file pairs using parallel processing.
    Optimized for millions of files.
    
    Args:
        input_dir: Directory containing audio and text files
        audio_extensions: List of audio file extensions to search for
        text_extensions: List of text file extensions to search for
        max_workers: Number of parallel workers for file discovery
    
    Returns:
        List of tuples (audio_file_path, text_file_path)
    """
    if audio_extensions is None:
        audio_extensions = ['.mp3', '.wav', '.flac']
    if text_extensions is None:
        text_extensions = ['.txt']
    
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    logging.info(f"Starting optimized file discovery with {max_workers} workers...")
    start_time = time.time()
    
    # Use os.walk for better performance on very large directories
    audio_files = {}
    text_files = {}
    
    logging.info("Scanning directory structure...")
    total_files = 0
    processed_dirs = 0
    
    # First pass - count directories for progress estimation
    total_dirs = sum(1 for _, dirs, _ in os.walk(input_dir))
    
    for root, dirs, files in os.walk(input_dir):
        processed_dirs += 1
        total_files += len(files)
        
        # Progress indicator every directory (for large data per dir)
        if processed_dirs % 1 == 0:
            progress = (processed_dirs / total_dirs) * 100
            current_dir = Path(root).name
            print(f"\r📁 Scanning: {current_dir}... ({processed_dirs:,}/{total_dirs:,} - {progress:.1f}%) - {total_files:,} files", end='', flush=True)
        
        # Process files in batches to avoid memory issues
        for filename in files:
            filepath = os.path.join(root, filename)
            name, ext = os.path.splitext(filename)
            
            if ext.lower() in audio_extensions:
                audio_files[name] = filepath
            elif ext.lower() in text_extensions:
                text_files[name] = filepath
    
    print()  # New line after progress
    
    logging.info(f"Scanned {total_files} total files in {time.time() - start_time:.2f}s")
    logging.info(f"Found {len(audio_files)} audio files and {len(text_files)} text files")
    
    # Find matching pairs efficiently
    pairs = []
    matched_count = 0
    
    logging.info("Matching audio-text pairs...")
    total_audio = len(audio_files)
    
    for i, name in enumerate(audio_files):
        if name in text_files:
            pairs.append((audio_files[name], text_files[name]))
            matched_count += 1
        
        # Simple progress indicator for large datasets
        if (i + 1) % 50000 == 0:
            progress = ((i + 1) / total_audio) * 100
            match_rate = (matched_count / (i + 1)) * 100
            print(f"\r🔗 Matching: {i+1:,}/{total_audio:,} ({progress:.1f}%) - {matched_count:,} pairs ({match_rate:.1f}% match rate)", end='', flush=True)
    
    if total_audio >= 50000:  # Only print newline if we showed progress
        print()
    
    unmatched_audio = len(audio_files) - matched_count
    unmatched_text = len(text_files) - matched_count
    
    if unmatched_audio > 0:
        logging.warning(f"{unmatched_audio} audio files have no matching text")
    if unmatched_text > 0:
        logging.warning(f"{unmatched_text} text files have no matching audio")
    
    if not pairs:
        raise ValueError(f"No matching audio-text pairs found in {input_dir}")
    
    logging.info(f"Found {len(pairs)} audio-text pairs in {time.time() - start_time:.2f}s")
    return pairs


def find_audio_text_pairs(input_dir: str, audio_extensions: List[str] = None, 
                          text_extensions: List[str] = None) -> List[Tuple[str, str]]:
    """
    Legacy function - redirects to optimized version.
    """
    return find_audio_text_pairs_optimized(input_dir, audio_extensions, text_extensions)


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


def process_single_folder(folder_path: str, intermediate_dir: str, target_sr: int = 16000,
                         text_encoding: str = 'utf-8', min_file_size: int = 1024) -> Optional[str]:
    """
    Process a single random-string folder and save to intermediate .arrow file.
    
    Args:
        folder_path: Path to the random-string folder
        intermediate_dir: Directory to save intermediate files
        target_sr: Target sample rate
        text_encoding: Text file encoding
        min_file_size: Minimum file size validation
        
    Returns:
        Path to created intermediate file, or None if no valid samples
    """
    folder_name = Path(folder_path).name
    intermediate_path = Path(intermediate_dir)
    intermediate_path.mkdir(parents=True, exist_ok=True)
    
    # Generate intermediate file name
    intermediate_file = intermediate_path / f"intermediate_{folder_name}.arrow"
    
    # Skip if already processed
    if intermediate_file.exists():
        logging.debug(f"Skipping already processed folder: {folder_name}")
        return str(intermediate_file)
    
    try:
        # Find audio-text pairs in this folder only
        audio_files = {}
        text_files = {}
        
        for file in os.listdir(folder_path):
            filepath = os.path.join(folder_path, file)
            if os.path.isfile(filepath):
                name, ext = os.path.splitext(file)
                
                if ext.lower() in ['.mp3', '.wav', '.flac']:
                    audio_files[name] = filepath
                elif ext.lower() == '.txt':
                    text_files[name] = filepath
        
        # Create pairs
        pairs = []
        for name in audio_files:
            if name in text_files:
                pairs.append((audio_files[name], text_files[name]))
        
        if not pairs:
            logging.warning(f"No audio-text pairs found in folder: {folder_name}")
            return None
        
        # Process samples from this folder
        samples = []
        for audio_path, text_path in pairs:
            try:
                # Use filename prefix as sample_id
                audio_filename = Path(audio_path).stem
                sample_id = audio_filename
                
                # Create sample with validation
                sample = create_sample(audio_path, text_path, sample_id, target_sr, text_encoding, min_file_size)
                samples.append(sample)
                
            except Exception as e:
                logging.debug(f"Skipping corrupted file {Path(audio_path).stem} in {folder_name}: {e}")
                continue
        
        if not samples:
            logging.warning(f"No valid samples created from folder: {folder_name}")
            return None
        
        # Create and save intermediate dataset
        intermediate_dataset = Dataset.from_list(samples)
        intermediate_dataset.save_to_disk(str(intermediate_file))
        
        # Clear memory
        del samples, intermediate_dataset
        gc.collect()
        
        return str(intermediate_file)
        
    except Exception as e:
        logging.error(f"Failed to process folder {folder_name}: {e}")
        return None


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


def process_sample_batch(batch_data: Tuple[List[Tuple[str, str]], int, int, int, str, int]) -> List[Dict]:
    """
    Process a batch of audio-text pairs in a separate process.
    
    Args:
        batch_data: Tuple containing (pairs, target_sr, text_encoding, min_file_size, batch_id, total_batches)
    
    Returns:
        List of processed samples
    """
    pairs, target_sr, text_encoding, min_file_size, batch_id, total_batches = batch_data
    
    samples = []
    failed_count = 0
    
    for i, (audio_path, text_path) in enumerate(pairs):
        try:
            # Use filename prefix as sample_id
            audio_filename = Path(audio_path).stem
            sample_id = audio_filename
            
            # Create sample with validation
            sample = create_sample(audio_path, text_path, sample_id, target_sr, text_encoding, min_file_size)
            samples.append(sample)
            
        except Exception as e:
            failed_count += 1
            # Use a different approach for logging in multiprocessing
            print(f"Batch {batch_id}: Skipping corrupted file {Path(audio_path).stem}: {e}")
            continue
    
    # Simple progress indicator with emoji
    success_rate = len(samples) / len(pairs) * 100 if pairs else 0
    print(f"✅ Batch {batch_id}/{total_batches}: {len(samples)}/{len(pairs)} samples ({success_rate:.1f}%)")
    return samples


def merge_intermediate_files(intermediate_files: List[str], final_output: str) -> Dataset:
    """
    Merge intermediate .arrow files into final dataset using streaming approach.
    
    Args:
        intermediate_files: List of paths to intermediate .arrow files
        final_output: Path for final output file
        
    Returns:
        Final merged dataset
    """
    logging.info(f"Merging {len(intermediate_files)} intermediate files...")
    
    if not intermediate_files:
        raise ValueError("No intermediate files to merge")
    
    # Load datasets one by one to avoid memory overflow
    datasets = []
    total_samples = 0
    
    for i, file_path in enumerate(tqdm(intermediate_files, desc="Loading intermediate files")):
        try:
            if Path(file_path).exists():
                dataset = Dataset.load_from_disk(file_path)
                datasets.append(dataset)
                total_samples += len(dataset)
                
                # Memory management - merge in batches to avoid overflow
                if len(datasets) >= 100:  # Merge every 100 datasets
                    logging.info(f"Merging batch of {len(datasets)} datasets...")
                    merged_batch = concatenate_datasets(datasets)
                    datasets = [merged_batch]
                    gc.collect()
                    
            else:
                logging.warning(f"Intermediate file not found: {file_path}")
        except Exception as e:
            logging.error(f"Failed to load intermediate file {file_path}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid intermediate datasets loaded")
    
    # Final merge
    logging.info(f"Performing final merge of {len(datasets)} dataset batches...")
    final_dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    
    logging.info(f"Successfully merged {total_samples:,} samples into final dataset")
    
    # Clear memory
    del datasets
    gc.collect()
    
    return final_dataset


def create_arrow_dataset_chunked(input_dir: str, intermediate_dir: str, target_sr: int = 16000,
                                text_encoding: str = 'utf-8', min_file_size: int = 1024,
                                folders_per_batch: int = 1000, max_workers: int = None) -> Dataset:
    """
    Create dataset using folder-by-folder chunked processing for large datasets.
    
    Args:
        input_dir: Input directory containing random-string folders
        intermediate_dir: Directory for temporary intermediate files  
        target_sr: Target sample rate
        text_encoding: Text encoding
        min_file_size: Minimum file size validation
        folders_per_batch: Process this many folders before creating intermediate file
        max_workers: Number of parallel workers
        
    Returns:
        Final dataset
    """
    # Discover all random-string folders
    random_folders = discover_random_string_folders(input_dir)
    
    if not random_folders:
        raise ValueError(f"No random-string folders found in {input_dir}")
    
    logging.info(f"Processing {len(random_folders)} folders with chunked approach")
    
    # Create intermediate directory
    intermediate_path = Path(intermediate_dir)
    intermediate_path.mkdir(parents=True, exist_ok=True)
    
    # Process folders and collect intermediate files
    intermediate_files = []
    processed_count = 0
    
    for i, folder_path in enumerate(random_folders):
        folder_name = Path(folder_path).name
        
        # Progress indicator (every directory)
        progress = ((i + 1) / len(random_folders)) * 100
        print(f"\r📁 Processing: {folder_name}... ({i+1:,}/{len(random_folders):,} - {progress:.1f}%)", end='', flush=True)
        
        # Process single folder
        intermediate_file = process_single_folder(
            folder_path, intermediate_dir, target_sr, text_encoding, min_file_size
        )
        
        if intermediate_file:
            intermediate_files.append(intermediate_file)
            processed_count += 1
        
        # Memory management
        if (i + 1) % 100 == 0:
            gc.collect()
    
    print()  # New line after progress
    
    logging.info(f"Successfully processed {processed_count}/{len(random_folders)} folders")
    
    if not intermediate_files:
        raise ValueError("No valid intermediate files created")
    
    # Merge all intermediate files
    final_dataset = merge_intermediate_files(intermediate_files, "final_output")
    
    return final_dataset


def create_arrow_dataset_batch(pairs: List[Tuple[str, str]], target_sr: int = 16000, 
                              text_encoding: str = 'utf-8', min_file_size: int = 1024,
                              batch_size: int = 1000, max_workers: int = None) -> Dataset:
    """
    Create HuggingFace Dataset from audio-text pairs using batch processing.
    Optimized for millions of files.
    
    Args:
        pairs: List of (audio_path, text_path) tuples
        target_sr: Target sample rate for audio
        text_encoding: Text file encoding
        min_file_size: Minimum file size in bytes to consider valid
        batch_size: Number of samples to process in each batch
        max_workers: Number of parallel workers
    
    Returns:
        HuggingFace Dataset
    """
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1))  # More aggressive with 1TB RAM
    
    logging.info(f"Processing {len(pairs)} pairs with batch size {batch_size} and {max_workers} workers")
    
    # Split pairs into batches
    batches = []
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        batch_id = i // batch_size + 1
        batches.append((batch_pairs, target_sr, text_encoding, min_file_size, batch_id, total_batches))
    
    logging.info(f"Created {len(batches)} batches")
    
    # Process batches in parallel
    all_samples = []
    total_failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_sample_batch, batch_data): batch_data[4] 
                          for batch_data in batches}
        
        # Collect results with progress tracking
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_samples = future.result()
                    all_samples.extend(batch_samples)
                    pbar.update(1)
                    
                    # Memory management - force garbage collection  
                    if batch_id % 10 == 0:
                        gc.collect()
                        
                        # Simple memory usage indicator
                        memory_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                        completed_batches = len([f for f in future_to_batch if f.done()])
                        total_samples = len(all_samples)
                        
                        print(f"💾 Progress: {completed_batches}/{len(batches)} batches, {total_samples:,} samples, {memory_gb:.1f}GB RAM")
                        
                except Exception as e:
                    logging.error(f"Batch {batch_id} failed: {e}")
                    total_failed += 1
    
    if not all_samples:
        raise ValueError("No samples were successfully created - all batches may have failed")
    
    success_rate = len(all_samples) / len(pairs) * 100
    logging.info(f"Successfully created {len(all_samples)} samples out of {len(pairs)} pairs")
    logging.info(f"Success rate: {success_rate:.1f}% ({total_failed} batches failed)")
    
    # Create HuggingFace Dataset
    logging.info("Creating HuggingFace Dataset...")
    dataset = Dataset.from_list(all_samples)
    
    # Clear memory
    del all_samples
    gc.collect()
    
    return dataset


def create_arrow_dataset(pairs: List[Tuple[str, str]], target_sr: int = 16000, 
                        text_encoding: str = 'utf-8', min_file_size: int = 1024,
                        use_batch_processing: bool = None, batch_size: int = 1000,
                        max_workers: int = None) -> Dataset:
    """
    Create HuggingFace Dataset from audio-text pairs.
    Automatically chooses between single-threaded and batch processing based on dataset size.
    
    Args:
        pairs: List of (audio_path, text_path) tuples
        target_sr: Target sample rate for audio
        text_encoding: Text file encoding
        min_file_size: Minimum file size in bytes to consider valid
        use_batch_processing: Force batch processing (None = auto-detect)
        batch_size: Number of samples per batch for parallel processing
        max_workers: Number of parallel workers
    
    Returns:
        HuggingFace Dataset
    """
    # Auto-detect whether to use batch processing
    # With 1TB RAM, we can handle much larger datasets in memory
    if use_batch_processing is None:
        use_batch_processing = len(pairs) > 1000000  # Use batch processing for >1M samples
    
    if use_batch_processing:
        logging.info(f"Using batch processing for {len(pairs)} samples (threshold: >1M)")
        return create_arrow_dataset_batch(pairs, target_sr, text_encoding, min_file_size, 
                                        batch_size, max_workers)
    else:
        logging.info(f"Using single-threaded processing for {len(pairs)} samples")
        
        samples = []
        failed_count = 0
        
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
    # HuggingFace Dataset may convert numpy arrays to lists during serialization
    if not isinstance(audio_array, (np.ndarray, list)):
        raise ValueError(f"Audio array should be numpy array or list, got {type(audio_array)}")
    
    # Convert back to numpy array if it's a list
    if isinstance(audio_array, list):
        try:
            audio_array = np.array(audio_array, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Could not convert audio array to numpy: {e}")
    
    # Validate array properties
    if len(audio_array) == 0:
        raise ValueError("Audio array is empty")
    
    if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
        raise ValueError("Audio array contains NaN or infinite values")
    
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
    parser.add_argument('--batch_size', type=int, default=5000,
                       help='Batch size for parallel processing (default: 5000, optimized for 1TB RAM)')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: auto)')
    parser.add_argument('--force_batch_processing', action='store_true',
                       help='Force batch processing even for small datasets')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume processing from existing partial dataset')
    parser.add_argument('--diagnose_mp3', action='store_true',
                       help='Run detailed diagnostics on MP3 files')
    parser.add_argument('--chunk_by_folders', action='store_true',
                       help='Enable folder-by-folder processing for large datasets')
    parser.add_argument('--intermediate_dir', type=str, default='temp_intermediates',
                       help='Directory for intermediate .arrow files (default: temp_intermediates)')
    parser.add_argument('--folders_per_batch', type=int, default=1000,
                       help='Process N folders before intermediate save (default: 1000)')
    parser.add_argument('--keep_intermediates', action='store_true',
                       help='Keep intermediate files after merging (for debugging)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Check if using chunked processing
        if args.chunk_by_folders:
            logging.info(f"Using chunked folder-by-folder processing for {args.input_dir}")
            
            # Create dataset using chunked approach
            dataset = create_arrow_dataset_chunked(
                args.input_dir,
                args.intermediate_dir,
                args.target_sr,
                args.text_encoding,
                args.min_file_size,
                args.folders_per_batch,
                args.max_workers
            )
            
            # Save final dataset
            logging.info(f"Saving final dataset to {args.output_file}")
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            dataset.save_to_disk(args.output_file)
            
            # Cleanup intermediate files unless requested to keep
            if not args.keep_intermediates:
                logging.info("Cleaning up intermediate files...")
                import shutil
                if Path(args.intermediate_dir).exists():
                    shutil.rmtree(args.intermediate_dir)
            
            logging.info(f"Successfully created chunked dataset with {len(dataset)} samples")
            logging.info(f"Final output saved to: {args.output_file}")
            return
        
        # Standard processing approach
        logging.info(f"Searching for audio-text pairs in {args.input_dir}")
        pairs = find_audio_text_pairs(
            args.input_dir, 
            args.audio_extensions, 
            args.text_extensions
        )
        
        # Run diagnostics if requested
        if args.diagnose_mp3:
            diagnose_mp3_files(pairs, max_samples=10)
            return
        
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
        
        # Handle resume functionality
        existing_dataset = None
        if args.resume_from and Path(args.resume_from).exists():
            try:
                existing_dataset = Dataset.load_from_disk(args.resume_from)
                existing_keys = set(existing_dataset['__key__'])
                
                # Filter out already processed pairs
                original_count = len(pairs)
                pairs = [(a, t) for a, t in pairs if Path(a).stem not in existing_keys]
                
                logging.info(f"Resuming from {args.resume_from}")
                logging.info(f"Found {len(existing_dataset)} existing samples")
                logging.info(f"Filtering {original_count - len(pairs)} already processed pairs")
                logging.info(f"Remaining {len(pairs)} pairs to process")
                
                if not pairs:
                    logging.info("All pairs already processed!")
                    return
                    
            except Exception as e:
                logging.warning(f"Could not load existing dataset for resume: {e}")
                existing_dataset = None

        # Create Arrow dataset
        logging.info("Creating Arrow dataset...")
        dataset = create_arrow_dataset(
            pairs, 
            args.target_sr, 
            args.text_encoding,
            args.min_file_size,
            use_batch_processing=args.force_batch_processing or None,
            batch_size=args.batch_size,
            max_workers=args.max_workers
        )
        
        # Merge with existing dataset if resuming
        if existing_dataset is not None:
            from datasets import concatenate_datasets
            logging.info("Merging with existing dataset...")
            dataset = concatenate_datasets([existing_dataset, dataset])
            logging.info(f"Final dataset size: {len(dataset)} samples")
        
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