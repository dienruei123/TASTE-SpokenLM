#!/usr/bin/env python3
"""
Extended test script for Arrow datasets with TASTE-SpokenLM processing support.

This script extends simple_test_arrow.py to include TASTE pipeline testing
while maintaining compatibility and avoiding import issues.
"""

import sys
import argparse
from pathlib import Path


def test_arrow_basic(arrow_path):
    """Test basic arrow dataset structure without heavy dependencies."""
    print(f"Testing arrow dataset: {arrow_path}")
    
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
        
        # Load the dataset
        
        print(f"✓ Dataset loaded successfully")
        print(f"✓ Dataset length: {len(dataset)}")
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
    """Test processing a single sample with TASTE pipeline."""
    print(f"\n" + "=" * 50)
    print(f"TESTING SINGLE SAMPLE PROCESSING (sample {sample_idx})")
    print("=" * 50)
    
    try:
        # Import required modules
        import torch
        from transformers import AutoTokenizer
        
        # Try to import WhisperProcessor with fallback
        try:
            from transformers import WhisperProcessor
        except ImportError:
            print("✗ WhisperProcessor not available. Trying alternative import...")
            try:
                from transformers import WhisperFeatureExtractor, WhisperTokenizer
                # Create a simple processor-like object
                class SimpleWhisperProcessor:
                    def __init__(self, model_name):
                        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
                        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)
                
                def WhisperProcessor_from_pretrained(model_name):
                    return SimpleWhisperProcessor(model_name)
                WhisperProcessor = type('WhisperProcessor', (), {'from_pretrained': staticmethod(WhisperProcessor_from_pretrained)})
            except ImportError as e:
                print(f"✗ Cannot create WhisperProcessor alternative: {e}")
                return False
        from taste_speech.data.dataset import process_one_sample
        from taste_speech.modules_taste.cosyvoice.whisper_frontend import WhisperFrontend
        
        # Get sample
        if sample_idx >= len(dataset):
            print(f"✗ Sample index {sample_idx} out of range (dataset size: {len(dataset)})")
            return False
        
        sample = dataset[sample_idx]
        print(f"✓ Sample {sample_idx} loaded")
        print(f"  - Audio length: {len(sample['mp3']['array'])}")
        print(f"  - Text: {sample['json']['text'][:50]}...")
        print(f"  - S3 token length: {len(sample['s3_token'])}")
        print(f"  - Speaker embedding length: {len(sample['spk_emb'])}")
        
        # Check for empty required fields
        if len(sample['s3_token']) == 0 or len(sample['spk_emb']) == 0:
            print("⚠ WARNING: s3_token or spk_emb is empty. This may cause processing to fail.")
            print("  You may need to populate these fields for full TASTE compatibility.")
        
        # Initialize processors
        print("✓ Loading processors...")
        whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_path)
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path)
        whisper_feature_extractor = WhisperFrontend(
            whisper_model="large-v3",
            do_pad_trim=True,
            permute=True,
        )
        print("✓ Processors loaded successfully")
        
        # Process sample
        print("✓ Processing sample...")
        resampler_dict = {}
        processed_sample = process_one_sample(
            sample,
            resampler_dict=resampler_dict,
            whisper_processor=whisper_processor,
            llm_tokenizer=llm_tokenizer,
            whisper_feature_extractor=whisper_feature_extractor
        )
        
        print("✓ Sample processing successful!")
        print("\nProcessed sample structure:")
        for key, value in processed_sample.items():
            if hasattr(value, 'shape'):  # torch.Tensor or numpy array
                print(f"  {key}: {type(value).__name__} shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__} - {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Single sample processing FAILED: {e}")
        import traceback
        print("Detailed error:")
        traceback.print_exc()
        return False


def test_full_dataset_processing(arrow_path, whisper_processor_path, llm_tokenizer_path, max_samples=None):
    """Test full dataset processing with TASTE pipeline."""
    print(f"\n" + "=" * 50)
    print("TESTING FULL DATASET PROCESSING")
    print("=" * 50)
    
    try:
        # Import required modules
        from taste_speech.data.dataset import load_from_arrows, REQUIRED_COLUMNS
        from pathlib import Path
        
        # Check if WhisperProcessor is available
        try:
            from transformers import WhisperProcessor
        except ImportError:
            print("⚠ WARNING: WhisperProcessor not available, using fallback approach")
            # This will be handled by the load_from_arrows function
        
        # Determine arrow files to process
        arrow_path_obj = Path(arrow_path)
        
        if arrow_path_obj.is_file() and arrow_path_obj.suffix == '.arrow':
            # Single .arrow file
            arrow_files = [arrow_path]
            print(f"✓ Processing single .arrow file: {arrow_path}")
        elif arrow_path_obj.is_dir():
            # Directory - find all .arrow files
            arrow_files = [str(f) for f in arrow_path_obj.glob("*.arrow")]
            if not arrow_files:
                # Check if it's a dataset directory - convert it to arrow file list
                try:
                    from datasets import Dataset
                    dataset = Dataset.load_from_disk(arrow_path)
                    # Create a temporary arrow file for testing
                    temp_arrow = arrow_path_obj / "temp_test.arrow" 
                    dataset.save_to_disk(str(temp_arrow))
                    arrow_files = [str(temp_arrow)]
                    print(f"✓ Converted dataset directory to temporary .arrow file")
                except:
                    raise ValueError(f"No .arrow files found in directory: {arrow_path}")
            else:
                print(f"✓ Found {len(arrow_files)} .arrow files in directory")
        else:
            raise ValueError(f"Path must be either a .arrow file or a directory: {arrow_path}")
        
        # Process dataset
        print("✓ Loading dataset with TASTE processing...")
        
        processed_dataset = load_from_arrows(
            arrow_fpath_list=arrow_files,
            whisper_processor_fpath=whisper_processor_path,
            llm_tokenizer_fpath=llm_tokenizer_path,
            streaming=False,
            num_proc=1  # Single process for testing stability
        )
        
        print(f"✓ Dataset processed successfully!")
        print(f"  - Original dataset: {arrow_path}")
        print(f"  - Processed length: {len(processed_dataset)}")
        print(f"  - Processed columns: {processed_dataset.column_names}")
        
        # Validate required columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in processed_dataset.column_names]
        if missing_columns:
            print(f"✗ Missing required columns: {missing_columns}")
            return False
        
        print("✓ All required columns present")
        
        # Test a few samples
        test_samples = min(max_samples or 3, len(processed_dataset))
        print(f"\nTesting first {test_samples} processed samples:")
        
        for i in range(test_samples):
            sample = processed_dataset[i]
            print(f"\nSample {i}:")
            for key, value in sample.items():
                if hasattr(value, 'shape'):  # torch.Tensor or numpy array
                    print(f"  {key}: {type(value).__name__} shape {value.shape}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        
        print("\n✓ Full dataset processing PASSED")
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
    
    args = parser.parse_args()
    
    # Check if arrow path exists
    if not Path(args.arrow_path).exists():
        print(f"Error: Arrow dataset path does not exist: {args.arrow_path}")
        sys.exit(1)
    
    # Run basic test
    print("=" * 50)
    print("TESTING BASIC ARROW STRUCTURE")
    print("=" * 50)
    basic_success, dataset = test_arrow_basic(args.arrow_path)
    
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
        
        # Test single sample processing
        single_success = test_single_sample_processing(
            dataset, args.whisper_processor, args.llm_tokenizer, args.sample_idx
        )
        
        # Test full dataset processing
        full_success = test_full_dataset_processing(
            args.arrow_path, args.whisper_processor, args.llm_tokenizer, args.max_samples
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