#!/usr/bin/env python3
"""
Extended test script for Arrow datasets with TASTE-SpokenLM processing support.

This script extends simple_test_arrow.py to include TASTE pipeline testing
while maintaining compatibility and avoiding import issues.
"""

import sys
import argparse
from pathlib import Path


def test_arrow_basic(arrow_path, max_samples=None):
    """Test basic arrow dataset structure without heavy dependencies."""
    print(f"Testing arrow dataset: {arrow_path}")
    
    # Apply sampling if max_samples is specified
    if max_samples is not None:
        print(f"Note: Will sample max {max_samples} samples for testing")
    
    try:
        # Try to import datasets with error handling
        from datasets import Dataset
        from pathlib import Path
        import os
        
        # Determine if input is a directory or a single .arrow file
        arrow_path_obj = Path(arrow_path)
        
        if arrow_path_obj.is_file() and arrow_path_obj.suffix == '.arrow':
            # Single .arrow file - use Dataset.from_file()
            print(f"✓ Loading single .arrow file: {arrow_path}")
            dataset = Dataset.from_file(arrow_path)
        elif arrow_path_obj.is_dir():
            # Directory - check if it's a dataset directory or contains .arrow files
            try:
                # Try loading as dataset directory first
                dataset = Dataset.load_from_disk(arrow_path)
                print(f"✓ Loaded as dataset directory: {arrow_path}")
            except:
                # Try finding .arrow files in directory
                arrow_files = list(arrow_path_obj.glob("*.arrow"))
                if arrow_files:
                    print(f"✓ Found {len(arrow_files)} .arrow files in directory")
                    # Load first .arrow file for testing
                    dataset = Dataset.from_file(str(arrow_files[0]))
                    print(f"✓ Testing with first file: {arrow_files[0].name}")
                else:
                    raise ValueError(f"No .arrow files found in directory: {arrow_path}")
        else:
            raise ValueError(f"Path must be either a .arrow file or a directory: {arrow_path}")
        
        # Apply sampling if requested
        original_length = len(dataset)
        if max_samples is not None and len(dataset) > max_samples:
            import random
            print(f"Sampling {max_samples} samples from {original_length} total samples for testing")
            # Use random seed for reproducible sampling
            random.seed(42)
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)
            print(f"Selected {len(dataset)} samples for testing")
        
        print(f"✓ Dataset loaded successfully")
        print(f"✓ Dataset length: {len(dataset)} (original: {original_length})")
        print(f"✓ Dataset columns: {dataset.column_names}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample keys: {list(sample.keys())}")
            
            # Check mp3 structure
            if 'mp3' in sample:
                mp3_data = sample['mp3']
                print(f"✓ MP3 keys: {list(mp3_data.keys())}")
                audio_array = mp3_data['array']
                print(f"✓ Audio array type: {type(audio_array)}")
                print(f"✓ Audio array length: {len(audio_array)}")
                print(f"✓ Sampling rate: {mp3_data['sampling_rate']}")
            
            # Check json structure
            if 'json' in sample:
                json_data = sample['json']
                print(f"✓ JSON keys: {list(json_data.keys())}")
                text = json_data['text']
                print(f"✓ Text preview: {text[:50]}...")
            
            # Check other fields
            print(f"✓ S3 token length: {len(sample['s3_token'])}")
            print(f"✓ Speaker embedding length: {len(sample['spk_emb'])}")
            
            if len(sample['s3_token']) == 0:
                print("⚠ WARNING: s3_token is empty")
            if len(sample['spk_emb']) == 0:
                print("⚠ WARNING: spk_emb is empty")
        
        print("✓ Basic test PASSED")
        return True, dataset
        
    except Exception as e:
        print(f"✗ Basic test FAILED: {e}")
        return False, None


def test_taste_imports():
    """Test if TASTE modules can be imported safely."""
    print("\n" + "=" * 50)
    print("TESTING TASTE MODULE IMPORTS")
    print("=" * 50)
    
    modules_status = {}
    
    # Test basic imports
    try:
        import torch
        modules_status['torch'] = True
        print("✓ torch imported successfully")
    except ImportError as e:
        modules_status['torch'] = False
        print(f"✗ torch import failed: {e}")
    
    try:
        from transformers import AutoTokenizer
        modules_status['transformers_basic'] = True
        print("✓ transformers (AutoTokenizer) imported successfully")
    except ImportError as e:
        modules_status['transformers_basic'] = False
        print(f"✗ transformers (AutoTokenizer) import failed: {e}")
    
    try:
        from transformers import WhisperProcessor
        modules_status['whisper_processor'] = True
        print("✓ WhisperProcessor imported successfully")
    except ImportError as e:
        modules_status['whisper_processor'] = False
        print(f"✗ WhisperProcessor import failed: {e}")
        print("  Try: pip install transformers[torch] or upgrade transformers")
    
    # Test TASTE specific imports
    try:
        sys.path.append(str(Path(__file__).parent))
        from taste_speech.data.dataset import load_from_arrows, process_one_sample, REQUIRED_COLUMNS
        modules_status['taste_dataset'] = True
        print("✓ taste_speech.data.dataset imported successfully")
    except ImportError as e:
        modules_status['taste_dataset'] = False
        print(f"✗ taste_speech.data.dataset import failed: {e}")
    
    try:
        from taste_speech.modules_taste.cosyvoice.whisper_frontend import WhisperFrontend
        modules_status['whisper_frontend'] = True
        print("✓ WhisperFrontend imported successfully")
    except ImportError as e:
        modules_status['whisper_frontend'] = False
        print(f"✗ WhisperFrontend import failed: {e}")
    
    all_available = all(modules_status.values())
    print(f"\nTASTE modules availability: {'ALL AVAILABLE' if all_available else 'SOME MISSING'}")
    
    return modules_status


def test_single_sample_processing(dataset, whisper_processor_path, llm_tokenizer_path, sample_idx=0):
    """Test processing a single sample with TASTE pipeline - STANDARD VERSION."""
    print(f"\n" + "=" * 50)
    print(f"TESTING SINGLE SAMPLE PROCESSING (sample {sample_idx})")
    print("=" * 50)
    
    try:
        import torch
        from transformers import AutoTokenizer, WhisperProcessor
        from taste_speech.data.dataset import process_one_sample
        from taste_speech.modules_taste.cosyvoice.whisper_frontend import WhisperFrontend
        
        # Check sample bounds
        if sample_idx >= len(dataset):
            print(f"✗ Sample index {sample_idx} out of range (dataset has {len(dataset)} samples)")
            return False
        
        sample = dataset[sample_idx]
        print(f"✓ Testing sample {sample_idx}")
        print(f"  - Audio length: {len(sample['mp3']['array'])} samples")
        print(f"  - Text length: {len(sample['json']['text'])} characters")
        print(f"  - Text: {sample['json']['text'][:100]}...")
        
        # Load processors
        print("Loading processors...")
        whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_path)
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path)
        whisper_feature_extractor = WhisperFrontend(
            whisper_model="large-v3",
            do_pad_trim=True,
            permute=True,
        )
        print("✓ All processors loaded")
        
        # Process the sample
        print("Processing sample...")
        resampler_dict = {}
        processed_sample = process_one_sample(
            sample,
            resampler_dict=resampler_dict,
            whisper_processor=whisper_processor,
            llm_tokenizer=llm_tokenizer,
            whisper_feature_extractor=whisper_feature_extractor
        )
        
        # Validate processed sample
        print(f"\n✓ Sample processed successfully!")
        print(f"  - Generated keys: {list(processed_sample.keys())}")
        
        for key, value in processed_sample.items():
            if hasattr(value, 'shape'):
                print(f"  - {key}: {value.shape} ({value.dtype})")
            elif hasattr(value, '__len__'):
                print(f"  - {key}: length {len(value)} ({type(value)})")
            else:
                print(f"  - {key}: {type(value)}")
        
        print("\n✓ Single sample processing PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Single sample processing FAILED: {e}")
        import traceback
        print("Detailed error:")
        traceback.print_exc()
        return False


def test_single_sample_processing_fast(dataset, whisper_processor_path, llm_tokenizer_path, sample_idx=0):
    """Test processing a single sample with TASTE pipeline - OPTIMIZED VERSION."""
    print(f"\n" + "=" * 50)
    print(f"TESTING SINGLE SAMPLE PROCESSING (FAST - sample {sample_idx})")
    print("=" * 50)
    
    try:
        import time
        start_time = time.time()
        
        # SPEEDUP 1: Fast imports with minimal error handling  
        import torch
        from transformers import AutoTokenizer, WhisperProcessor
        from taste_speech.data.dataset import process_one_sample
        from taste_speech.modules_taste.cosyvoice.whisper_frontend import WhisperFrontend
        
        # SPEEDUP 2: Quick sample validation
        if sample_idx >= len(dataset):
            print(f"✗ Sample index {sample_idx} out of range")
            return False
        
        sample = dataset[sample_idx]
        print(f"⚡ Sample {sample_idx} loaded - Audio: {len(sample['mp3']['array'])}, Text: {len(sample['json']['text'])} chars")
        
        # SPEEDUP 3: Fast processor loading with minimal output
        load_start = time.time()
        whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_path)
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path)
        whisper_feature_extractor = WhisperFrontend(
            whisper_model="large-v3",
            do_pad_trim=True,
            permute=True,
        )
        load_time = time.time() - load_start
        print(f"⚡ Processors loaded in {load_time:.2f}s")
        
        # SPEEDUP 4: Fast processing
        process_start = time.time()
        resampler_dict = {}
        processed_sample = process_one_sample(
            sample,
            resampler_dict=resampler_dict,
            whisper_processor=whisper_processor,
            llm_tokenizer=llm_tokenizer,
            whisper_feature_extractor=whisper_feature_extractor
        )
        process_time = time.time() - process_start
        
        # SPEEDUP 5: Minimal output - just tensor counts and shapes
        tensor_count = sum(1 for v in processed_sample.values() if hasattr(v, 'shape'))
        total_time = time.time() - start_time
        
        print(f"⚡ FAST processing completed!")
        print(f"  - Load time: {load_time:.2f}s, Process time: {process_time:.2f}s, Total: {total_time:.2f}s")
        print(f"  - Generated {tensor_count} tensors successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Single sample processing FAILED: {e}")
        import traceback
        print("Detailed error:")
        traceback.print_exc()
        return False


def test_full_dataset_processing(arrow_path, whisper_processor_path, llm_tokenizer_path, max_samples=None, test_samples=None):
    """Test full dataset processing with TASTE pipeline - STANDARD VERSION."""
    print(f"\n" + "=" * 50)
    print("TESTING FULL DATASET PROCESSING")
    if test_samples:
        print(f"(Limited to {test_samples} samples for faster testing)")
    print("=" * 50)
    
    try:
        from taste_speech.data.dataset import load_from_arrows, REQUIRED_COLUMNS
        from pathlib import Path
        import os
        
        # Determine arrow files to process and apply sampling if needed
        arrow_path_obj = Path(arrow_path)
        
        if arrow_path_obj.is_file() and arrow_path_obj.suffix == '.arrow':
            arrow_files = [arrow_path]
            print(f"✓ Processing single .arrow file: {arrow_path}")
        elif arrow_path_obj.is_dir():
            arrow_files = [str(f) for f in arrow_path_obj.glob("*.arrow")]
            if not arrow_files:
                try:
                    from datasets import Dataset
                    dataset = Dataset.load_from_disk(arrow_path)
                    
                    # Apply sampling before creating temp file
                    if test_samples is not None and len(dataset) > test_samples:
                        import random
                        print(f"Sampling {test_samples} samples from {len(dataset)} for faster testing")
                        random.seed(42)
                        indices = random.sample(range(len(dataset)), test_samples)
                        dataset = dataset.select(indices)
                    
                    temp_arrow = arrow_path_obj / "temp_test.arrow" 
                    dataset.save_to_disk(str(temp_arrow))
                    arrow_files = [str(temp_arrow)]
                    print(f"✓ Converted dataset directory to temporary file (samples: {len(dataset)})")
                except:
                    raise ValueError(f"No .arrow files found in directory: {arrow_path}")
            else:
                print(f"✓ Found {len(arrow_files)} .arrow files")
                # For multiple arrow files, we'll sample after loading
        else:
            raise ValueError(f"Path must be either a .arrow file or a directory: {arrow_path}")
        
        # Use moderate parallelization for standard processing
        max_workers = min(16, os.cpu_count())
        print(f"✓ Using {max_workers} parallel workers")
        
        # Process dataset
        print("Loading and processing dataset...")
        processed_dataset = load_from_arrows(
            arrow_fpath_list=arrow_files,
            whisper_processor_fpath=whisper_processor_path,
            llm_tokenizer_fpath=llm_tokenizer_path,
            streaming=False,
            num_proc=max_workers
        )
        
        # Apply sampling after loading if we have multiple arrow files
        if test_samples is not None and len(processed_dataset) > test_samples and len(arrow_files) > 1:
            import random
            print(f"Sampling {test_samples} samples from {len(processed_dataset)} for faster testing")
            random.seed(42)
            indices = random.sample(range(len(processed_dataset)), test_samples)
            processed_dataset = processed_dataset.select(indices)
        
        print(f"✓ Dataset processed successfully")
        print(f"  - Original files: {len(arrow_files)}")
        print(f"  - Processed samples: {len(processed_dataset):,}")
        print(f"  - Columns: {processed_dataset.column_names}")
        
        # Validate required columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in processed_dataset.column_names]
        if missing_columns:
            print(f"✗ Missing required columns: {missing_columns}")
            return False
        print("✓ All required columns present")
        
        # Test some samples
        dataset_size = len(processed_dataset)
        if max_samples is None:
            max_samples = min(5, dataset_size)  # Test up to 5 samples by default
        else:
            max_samples = min(max_samples, dataset_size)
        
        print(f"\n✓ Testing {max_samples} samples from dataset...")
        
        for i in range(max_samples):
            sample_idx = i if i < dataset_size else 0  # Fallback to first sample if out of range
            sample = processed_dataset[sample_idx]
            
            print(f"\n  Sample {sample_idx}:")
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    print(f"    - {key}: {value.shape} ({value.dtype})")
                elif hasattr(value, '__len__'):
                    print(f"    - {key}: length {len(value)} ({type(value)})")
                else:
                    print(f"    - {key}: {type(value)}")
        
        print(f"\n✓ Full dataset processing PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Full dataset processing FAILED: {e}")
        import traceback
        print("Detailed error:")
        traceback.print_exc()
        return False


def test_full_dataset_processing_fast(arrow_path, whisper_processor_path, llm_tokenizer_path, max_samples=None, test_samples=None):
    """Test full dataset processing with TASTE pipeline - OPTIMIZED VERSION."""
    print(f"\n" + "=" * 50)
    print("TESTING FULL DATASET PROCESSING (FAST)")
    if test_samples:
        print(f"(Limited to {test_samples} samples for maximum speed)")
    print("=" * 50)
    
    try:
        import os
        import time
        from taste_speech.data.dataset import load_from_arrows, REQUIRED_COLUMNS
        from pathlib import Path
        
        start_time = time.time()
        
        # Determine arrow files to process with fast sampling
        arrow_path_obj = Path(arrow_path)
        
        if arrow_path_obj.is_file() and arrow_path_obj.suffix == '.arrow':
            arrow_files = [arrow_path]
            print(f"⚡ Processing single .arrow file")
        elif arrow_path_obj.is_dir():
            arrow_files = [str(f) for f in arrow_path_obj.glob("*.arrow")]
            if not arrow_files:
                try:
                    from datasets import Dataset
                    dataset = Dataset.load_from_disk(arrow_path)
                    
                    # Fast sampling before processing
                    if test_samples is not None and len(dataset) > test_samples:
                        import random
                        print(f"⚡ Fast sampling {test_samples} from {len(dataset)} samples")
                        random.seed(42)
                        indices = random.sample(range(len(dataset)), test_samples)
                        dataset = dataset.select(indices)
                    
                    temp_arrow = arrow_path_obj / "temp_test.arrow" 
                    dataset.save_to_disk(str(temp_arrow))
                    arrow_files = [str(temp_arrow)]
                    print(f"⚡ Created optimized temp file ({len(dataset)} samples)")
                except:
                    raise ValueError(f"No .arrow files found in directory: {arrow_path}")
            else:
                print(f"⚡ Found {len(arrow_files)} .arrow files")
        else:
            raise ValueError(f"Path must be either a .arrow file or a directory: {arrow_path}")
        
        # SPEEDUP 1: Use maximum parallel processing
        max_workers = min(64, os.cpu_count() * 2)  # Aggressive parallelization
        print(f"⚡ Using {max_workers} parallel workers for maximum speed")
        
        # SPEEDUP 2: Process dataset with optimized settings
        processed_dataset = load_from_arrows(
            arrow_fpath_list=arrow_files,
            whisper_processor_fpath=whisper_processor_path,
            llm_tokenizer_fpath=llm_tokenizer_path,
            streaming=False,
            num_proc=max_workers  # Maximum parallelization instead of 1
        )
        
        # Apply fast sampling after loading if needed (for multiple arrow files)
        if test_samples is not None and len(processed_dataset) > test_samples and len(arrow_files) > 1:
            import random
            print(f"⚡ Post-processing sampling: {test_samples} from {len(processed_dataset)}")
            random.seed(42)
            indices = random.sample(range(len(processed_dataset)), test_samples)
            processed_dataset = processed_dataset.select(indices)
        
        processing_time = time.time() - start_time
        print(f"⚡ Dataset processed in {processing_time:.2f}s!")
        print(f"  - Processed length: {len(processed_dataset):,}")
        print(f"  - Processing rate: {len(processed_dataset)/processing_time:.1f} samples/sec")
        
        # SPEEDUP 3: Quick column validation (no detailed checking)
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in processed_dataset.column_names]
        if missing_columns:
            print(f"✗ Missing required columns: {missing_columns}")
            return False
        print("⚡ All required columns present")
        
        # SPEEDUP 4: Smart sampling - test only representative samples
        dataset_size = len(processed_dataset)
        if max_samples is None:
            # Smart sampling: test fewer samples for large datasets
            if dataset_size > 1000:
                test_samples = 3
            elif dataset_size > 100:
                test_samples = 5
            else:
                test_samples = min(3, dataset_size)
        else:
            test_samples = min(max_samples, dataset_size)
        
        print(f"⚡ Testing {test_samples} representative samples (out of {dataset_size:,})")
        
        # SPEEDUP 5: Minimal output - only essential info
        sample_indices = [0, dataset_size//2, dataset_size-1] if dataset_size >= 3 else [0]
        sample_indices = sample_indices[:test_samples]
        
        for idx in sample_indices:
            sample = processed_dataset[idx]
            # Only check tensor shapes, not detailed content
            tensor_info = []
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    tensor_info.append(f"{key}: {value.shape}")
            print(f"  Sample {idx}: {len(tensor_info)} tensors OK")
        
        print(f"⚡ FAST processing completed in {processing_time:.2f}s - {len(processed_dataset)/processing_time:.1f} samples/sec")
        return True
        
    except Exception as e:
        print(f"✗ Full dataset processing FAILED: {e}")
        import traceback
        print("Detailed error:")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Arrow dataset with optional TASTE processing")
    parser.add_argument('arrow_path', type=str, help='Path to Arrow dataset (.arrow file or directory containing .arrow files)')
    parser.add_argument('--whisper_processor', type=str, default='openai/whisper-large-v3',
                       help='Whisper processor model path (default: openai/whisper-large-v3)')
    parser.add_argument('--llm_tokenizer', type=str, 
                       help='LLM tokenizer path (e.g., path to Llama model)')
    parser.add_argument('--test_taste', action='store_true',
                       help='Run TASTE processing tests (requires whisper_processor and llm_tokenizer)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to test for single sample processing (default: 0)')
    parser.add_argument('--max_samples', type=int, default=3,
                       help='Maximum number of samples to test in full processing (default: 3)')
    parser.add_argument('--fast', action='store_true',
                       help='Use fast/optimized testing mode for better performance')
    parser.add_argument('--test_samples', type=int, default=None,
                       help='Maximum number of samples to use for testing (default: use all samples)')
    
    args = parser.parse_args()
    
    # Check if arrow path exists
    if not Path(args.arrow_path).exists():
        print(f"Error: Arrow dataset path does not exist: {args.arrow_path}")
        sys.exit(1)
    
    # Run basic test
    print("=" * 50)
    print("TESTING BASIC ARROW STRUCTURE")
    print("=" * 50)
    basic_success, dataset = test_arrow_basic(args.arrow_path, args.test_samples)
    
    if not basic_success:
        print("✗ Basic test failed. Cannot proceed with TASTE testing.")
        sys.exit(1)
    
    # Test TASTE functionality if requested
    if args.test_taste:
        if not args.llm_tokenizer:
            print("Error: --llm_tokenizer is required for TASTE testing")
            sys.exit(1)
        
        # Test module imports
        modules_status = test_taste_imports()
        
        if not all(modules_status.values()):
            print("⚠ Some TASTE modules are missing. TASTE tests may fail.")
            print("Make sure you have all required dependencies installed.")
        
        # Choose testing functions based on fast flag
        if args.fast:
            print("\n⚡ Using FAST/OPTIMIZED testing mode")
            single_test_fn = test_single_sample_processing_fast
            full_test_fn = test_full_dataset_processing_fast
        else:
            print("\n📋 Using STANDARD testing mode")
            single_test_fn = test_single_sample_processing
            full_test_fn = test_full_dataset_processing
        
        # Test single sample processing
        single_success = single_test_fn(
            dataset, args.whisper_processor, args.llm_tokenizer, args.sample_idx
        )
        
        # Test full dataset processing
        full_success = full_test_fn(
            args.arrow_path, args.whisper_processor, args.llm_tokenizer, args.max_samples, args.test_samples
        )
        
        # Final results
        print(f"\n" + "=" * 50)
        print("FINAL RESULTS")
        print("=" * 50)
        print(f"Basic test: {'PASSED' if basic_success else 'FAILED'}")
        print(f"Single sample processing: {'PASSED' if single_success else 'FAILED'}")
        print(f"Full dataset processing: {'PASSED' if full_success else 'FAILED'}")
        
        if basic_success and single_success and full_success:
            print("\n🎉 ALL TESTS PASSED! Your Arrow dataset is compatible with TASTE pipeline.")
        else:
            print("\n⚠ Some tests failed. Check the output above for details.")
            sys.exit(1)
    
    else:
        print(f"\n✓ Basic test completed successfully!")
        print("To test TASTE processing, run with --test_taste --llm_tokenizer /path/to/tokenizer")


if __name__ == "__main__":
    main()