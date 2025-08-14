#!/usr/bin/env python3
"""
Simple test script for Arrow datasets without complex imports.
"""

import sys
import os
from pathlib import Path

def test_arrow_basic(arrow_path):
    """Test basic arrow dataset structure without heavy dependencies."""
    print(f"Testing arrow dataset: {arrow_path}")
    
    try:
        # Try to import datasets with error handling
        from datasets import Dataset
        
        # Load the dataset
        dataset = Dataset.load_from_disk(arrow_path)
        
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
        return True
        
    except Exception as e:
        print(f"✗ Test FAILED: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_test_arrow.py <arrow_dataset_path>")
        sys.exit(1)
    
    arrow_path = sys.argv[1]
    
    if not Path(arrow_path).exists():
        print(f"Error: Path does not exist: {arrow_path}")
        sys.exit(1)
    
    success = test_arrow_basic(arrow_path)
    sys.exit(0 if success else 1)