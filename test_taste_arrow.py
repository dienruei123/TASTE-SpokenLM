#!/usr/bin/env python3
"""
Simple, robust test script for TASTE Arrow datasets.

Tests the complete pipeline from Arrow dataset input to processed TASTE format output.
"""

import sys
import argparse
from pathlib import Path


def load_and_sample_dataset(arrow_path, test_samples=None):
    """Load arrow dataset and apply sampling if needed."""
    print(f"Loading dataset from: {arrow_path}")
    
    try:
        from datasets import Dataset
        
        arrow_path_obj = Path(arrow_path)
        
        # Load dataset
        if arrow_path_obj.is_file() and arrow_path_obj.suffix == '.arrow':
            dataset = Dataset.from_file(arrow_path)
            print(f"✓ Loaded single .arrow file: {len(dataset)} samples")
        elif arrow_path_obj.is_dir():
            dataset = Dataset.load_from_disk(arrow_path)
            print(f"✓ Loaded dataset directory: {len(dataset)} samples")
        else:
            raise ValueError(f"Invalid path: {arrow_path}")
        
        # Apply sampling if requested
        if test_samples and len(dataset) > test_samples:
            import random
            random.seed(42)
            indices = random.sample(range(len(dataset)), test_samples)
            dataset = dataset.select(indices)
            print(f"✓ Sampled {len(dataset)} samples for testing")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None


def validate_input_format(dataset):
    """Validate dataset has expected input format from collect_mp3_txt_to_arrow.py."""
    print("\nValidating input dataset format...")
    
    try:
        # Check required columns
        required_columns = ['__key__', 'mp3', 'json', 's3_token', 'spk_emb']
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        
        if missing_columns:
            print(f"✗ Missing required columns: {missing_columns}")
            return False
        
        print(f"✓ All required columns present: {dataset.column_names}")
        
        # Check sample structure
        sample = dataset[0]
        
        # Validate mp3 structure
        if not isinstance(sample['mp3'], dict) or 'array' not in sample['mp3'] or 'sampling_rate' not in sample['mp3']:
            print("✗ Invalid mp3 structure")
            return False
        
        # Validate json structure
        if not isinstance(sample['json'], dict) or 'text' not in sample['json']:
            print("✗ Invalid json structure")
            return False
        
        audio_len = len(sample['mp3']['array'])
        text_len = len(sample['json']['text'])
        print(f"✓ Sample format valid - Audio: {audio_len} samples, Text: {text_len} chars")
        
        return True
        
    except Exception as e:
        print(f"✗ Input validation failed: {e}")
        return False


def test_single_sample_processing(dataset, whisper_processor_path, llm_tokenizer_path):
    """Test processing a single sample using process_one_sample."""
    print("\nTesting single sample processing...")
    
    try:
        from transformers import AutoTokenizer, WhisperProcessor
        from taste_speech.data.dataset import process_one_sample
        from taste_speech.modules_taste.cosyvoice.whisper_frontend import WhisperFrontend
        
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
        
        # Process single sample
        sample = dataset[0]
        resampler_dict = {}
        
        processed_sample = process_one_sample(
            sample,
            resampler_dict=resampler_dict,
            whisper_processor=whisper_processor,
            llm_tokenizer=llm_tokenizer,
            whisper_feature_extractor=whisper_feature_extractor
        )
        
        # Validate output
        from taste_speech.data.dataset import REQUIRED_COLUMNS
        missing_output_cols = [col for col in REQUIRED_COLUMNS if col not in processed_sample.keys()]
        if missing_output_cols:
            print(f"✗ Missing output columns: {missing_output_cols}")
            return False
        
        tensor_count = sum(1 for v in processed_sample.values() if hasattr(v, 'shape'))
        print(f"✓ Single sample processed successfully: {tensor_count} tensors generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Single sample processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_from_arrows_no_cache(arrow_fpath_list, whisper_processor_fpath="", llm_tokenizer_fpath="", streaming=False, num_proc=1):
    """Custom load_from_arrows that disables caching."""
    from taste_speech.data.dataset import process_one_sample, REQUIRED_COLUMNS
    from datasets import Dataset, concatenate_datasets
    from transformers import AutoTokenizer, WhisperProcessor
    from taste_speech.modules_taste.cosyvoice.whisper_frontend import WhisperFrontend
    from functools import partial
    import tqdm
    
    # Concatenate datasets
    ds_of_arrows = concatenate_datasets([
        Dataset.from_file(_arrow_fpath) for _arrow_fpath in tqdm.tqdm(arrow_fpath_list, desc="concatenating...")
    ])
    
    if streaming:
        ds_of_arrows = ds_of_arrows.to_iterable_dataset()
    
    # Load processors
    resampler_dict = {}
    whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_fpath)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_fpath)
    whisper_feature_extractor = WhisperFrontend(
        whisper_model="large-v3",
        do_pad_trim=True,
        permute=True,
    )
    
    # Create processing function
    _process_one_sample = partial(
        process_one_sample, 
        resampler_dict=resampler_dict, 
        whisper_processor=whisper_processor,
        llm_tokenizer=llm_tokenizer, 
        whisper_feature_extractor=whisper_feature_extractor
    )
    
    # Map with caching disabled
    kwargs = {"load_from_cache_file": False}
    if not streaming:
        kwargs['num_proc'] = num_proc
    
    ds_processed_arrows = ds_of_arrows.map(
        _process_one_sample,
        **kwargs,
    ).select_columns(REQUIRED_COLUMNS)
    
    return ds_processed_arrows


def test_full_pipeline(arrow_path, whisper_processor_path, llm_tokenizer_path, test_samples=None, num_proc=1):
    """Test full pipeline using load_from_arrows."""
    print(f"\nTesting full pipeline with {num_proc} processes...")
    
    try:
        from taste_speech.data.dataset import REQUIRED_COLUMNS
        
        # Prepare arrow files list
        arrow_path_obj = Path(arrow_path)
        
        if arrow_path_obj.is_file() and arrow_path_obj.suffix == '.arrow':
            arrow_files = [arrow_path]
        elif arrow_path_obj.is_dir():
            # Load full dataset first
            from datasets import Dataset
            dataset = Dataset.load_from_disk(arrow_path)
            print(f"✓ Loaded dataset with {len(dataset)} samples")
            
            # Apply sampling if requested
            if test_samples and len(dataset) > test_samples:
                import random
                random.seed(42)
                indices = random.sample(range(len(dataset)), test_samples)
                dataset = dataset.select(indices)
                print(f"✓ Sampled to {len(dataset)} samples for testing")
            
            # Create temporary arrow file with sampled data
            temp_arrow_dir = arrow_path_obj / "temp_test"
            dataset.save_to_disk(str(temp_arrow_dir))
            
            # Find the actual .arrow file in the saved directory
            saved_arrow_files = list(temp_arrow_dir.glob("*.arrow"))
            if saved_arrow_files:
                arrow_files = [str(f) for f in saved_arrow_files]
                print(f"✓ Created temporary dataset with {len(dataset)} samples")
            else:
                raise ValueError("Failed to create temporary arrow file")
        else:
            raise ValueError(f"Invalid arrow path: {arrow_path}")
        
        # Process with load_from_arrows
        print("Processing with load_from_arrows...")
        
        # Force num_proc=1 to avoid multiprocessing issues with pre-loaded processors
        # The issue is that whisper_processor, llm_tokenizer, and whisper_feature_extractor
        # cannot be properly pickled and shared between processes
        safe_num_proc = 1 if num_proc > 1 else num_proc
        if num_proc > 1:
            print(f"⚠ Forcing num_proc=1 (was {num_proc}) to avoid multiprocessing issues")
        
        processed_dataset = load_from_arrows_no_cache(
            arrow_fpath_list=arrow_files,
            whisper_processor_fpath=whisper_processor_path,
            llm_tokenizer_fpath=llm_tokenizer_path,
            streaming=False,
            num_proc=safe_num_proc
        )
        
        print(f"✓ Pipeline completed: {len(processed_dataset)} samples processed")
        
        # Validate output columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in processed_dataset.column_names]
        if missing_columns:
            print(f"✗ Missing required output columns: {missing_columns}")
            return False
        
        print(f"✓ All required output columns present: {processed_dataset.column_names}")
        
        # Test a sample from processed dataset
        if len(processed_dataset) > 0:
            sample = processed_dataset[0]
            tensor_count = sum(1 for v in sample.values() if hasattr(v, 'shape'))
            print(f"✓ Output sample validation: {tensor_count} tensors")
        
        # Cleanup temp directory
        if arrow_path_obj.is_dir():
            temp_arrow_dir = arrow_path_obj / "temp_test"
            if temp_arrow_dir.exists():
                import shutil
                shutil.rmtree(temp_arrow_dir)
                print("✓ Cleaned up temporary files")
        
        return True
        
    except Exception as e:
        print(f"✗ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simple test for TASTE Arrow datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_taste_arrow.py dataset.arrow --llm_tokenizer /path/to/llm
  python test_taste_arrow.py dataset/ --llm_tokenizer /path/to/llm --test_samples 100
  python test_taste_arrow.py dataset.arrow --llm_tokenizer /path/to/llm --num_proc 4
        """
    )
    
    parser.add_argument('arrow_path', help='Path to Arrow dataset (.arrow file or directory)')
    parser.add_argument('--whisper_processor', default='openai/whisper-large-v3',
                       help='Whisper processor model path (default: openai/whisper-large-v3)')
    parser.add_argument('--llm_tokenizer', required=True,
                       help='LLM tokenizer path (required)')
    parser.add_argument('--test_samples', type=int,
                       help='Limit testing to N samples for faster testing')
    parser.add_argument('--num_proc', type=int, default=1,
                       help='Number of processes for load_from_arrows (default: 1)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.arrow_path).exists():
        print(f"Error: Arrow dataset path does not exist: {args.arrow_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("TASTE ARROW DATASET TEST")
    print("=" * 60)
    
    # Step 1: Load dataset
    dataset = load_and_sample_dataset(args.arrow_path, args.test_samples)
    if not dataset:
        sys.exit(1)
    
    # Step 2: Validate input format
    if not validate_input_format(dataset):
        sys.exit(1)
    
    # Step 3: Test single sample processing
    if not test_single_sample_processing(dataset, args.whisper_processor, args.llm_tokenizer):
        sys.exit(1)
    
    # Step 4: Test full pipeline
    if not test_full_pipeline(args.arrow_path, args.whisper_processor, args.llm_tokenizer, 
                             args.test_samples, args.num_proc):
        sys.exit(1)
    
    # Success!
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("Your Arrow dataset is fully compatible with the TASTE pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()