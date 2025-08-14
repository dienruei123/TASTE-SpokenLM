# MP3 + TXT to Arrow Collection Script

This script (`collect_mp3_txt_to_arrow.py`) converts paired MP3 audio files and TXT transcription files into Apache Arrow format compatible with the TASTE-SpokenLM dataset processing pipeline.

## Overview

The script is designed to work with the existing `taste_speech/data/dataset.py` processing pipeline by creating Arrow files with the exact format expected by the TASTE dataset loader.

## Requirements

### File Structure
Your input directory should contain:
- MP3 audio files (or WAV/FLAC)
- Corresponding TXT transcription files
- Files should have matching basenames (e.g., `audio001.mp3` and `audio001.txt`)

### Dependencies
The script requires the following Python packages:
```bash
pip install torch torchaudio librosa datasets numpy tqdm
```

## Usage

### Basic Usage
```bash
python collect_mp3_txt_to_arrow.py \
  --input_dir /path/to/your/mp3_txt_files \
  --output_file /path/to/output.arrow
```

### Advanced Usage with All Options
```bash
python collect_mp3_txt_to_arrow.py \
  --input_dir /path/to/your/mp3_txt_files \
  --output_file /path/to/output.arrow \
  --target_sr 16000 \
  --text_encoding utf-8 \
  --key_prefix "my_dataset" \
  --audio_extensions .mp3 .wav .flac \
  --text_extensions .txt \
  --log_level INFO \
  --validate
```

### Parameters

- `--input_dir`: **Required.** Directory containing your MP3 and TXT files
- `--output_file`: **Required.** Path where the Arrow file will be saved
- `--target_sr`: Target sample rate for audio (default: 16000 Hz)
- `--text_encoding`: Text file encoding (default: utf-8)
- `--key_prefix`: Prefix for generating unique sample IDs (default: "sample")
- `--audio_extensions`: Audio file extensions to search for (default: .mp3 .wav .flac)
- `--text_extensions`: Text file extensions to search for (default: .txt)
- `--log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `--validate`: Run compatibility validation with TASTE pipeline

## Input File Examples

### Directory Structure
```
your_data/
├── audio001.mp3
├── audio001.txt
├── audio002.mp3
├── audio002.txt
├── subfolder/
│   ├── recording123.wav
│   └── recording123.txt
└── ...
```

### Text File Format
Each `.txt` file should contain the transcription for the corresponding audio file:

**audio001.txt:**
```
Hello, this is a sample transcription of the audio file.
```

**audio002.txt:**
```
This script will convert your MP3 and text files to Arrow format.
```

## Output Format

The script creates an Arrow dataset with the following structure for each sample:

```python
{
    '__key__': 'sample_000001',  # Unique identifier
    'mp3': {
        'array': numpy_array,     # Audio waveform (16kHz, float32)
        'sampling_rate': 16000    # Sample rate
    },
    'json': {
        'text': 'transcription...'  # Text transcription
    },
    's3_token': [],              # Empty, filled by other scripts
    'spk_emb': []               # Empty, filled by other scripts
}
```

## Integration with TASTE Pipeline

After creating the Arrow file, you can use it with the existing TASTE dataset processing:

```python
from taste_speech.data.dataset import TasteStage1Dataset

# Use your Arrow file in the dataset
dataset = TasteStage1Dataset(
    data_list_dir="/path/containing/your/arrow/file",
    whisper_processor_fpath="openai/whisper-large-v3",
    llm_tokenizer_fpath="/path/to/llm/tokenizer"
)

# Or test with the dataset.py functions
from taste_speech.data.dataset import load_from_arrows

dataset = load_from_arrows(
    arrow_fpath_list=["/path/to/your/output.arrow"],
    whisper_processor_fpath="openai/whisper-large-v3",
    llm_tokenizer_fpath="/path/to/llm/tokenizer"
)
```

## Validation

The script includes validation to ensure compatibility with the TASTE pipeline:

1. **File Pairing**: Ensures each audio file has a corresponding text file
2. **Audio Processing**: Validates audio loading and resampling
3. **Text Processing**: Checks text encoding and content
4. **Format Compatibility**: Verifies the output format matches TASTE requirements

Run with `--validate` flag to enable full compatibility checking.

## Error Handling

The script handles common issues:
- Missing file pairs (logs warnings)
- Audio loading failures (tries multiple backends)
- Text encoding issues (tries multiple encodings)
- Empty or corrupted files (skips with error logs)

Check the logs for any processing issues.

## Example Complete Workflow

1. **Prepare your data:**
   ```
   my_data/
   ├── recording_001.mp3
   ├── recording_001.txt
   ├── recording_002.mp3
   └── recording_002.txt
   ```

2. **Run the collection script:**
   ```bash
   python collect_mp3_txt_to_arrow.py \
     --input_dir ./my_data \
     --output_file ./my_dataset.arrow \
     --validate
   ```

3. **Use with TASTE dataset:**
   ```python
   from taste_speech.data.dataset import load_from_arrows
   
   dataset = load_from_arrows(
       arrow_fpath_list=["./my_dataset.arrow"],
       whisper_processor_fpath="openai/whisper-large-v3",
       llm_tokenizer_fpath="/path/to/your/llm/tokenizer"
   )
   ```

## Troubleshooting

### Common Issues

1. **"No matching audio-text pairs found"**
   - Check that audio and text files have matching basenames
   - Verify file extensions are correct

2. **Audio loading failures**
   - Ensure audio files are not corrupted
   - Check that torchaudio/librosa can handle your audio format

3. **Text encoding errors**
   - Try different `--text_encoding` values (utf-8, latin-1, cp1252)
   - Check that text files are not binary or corrupted

4. **Memory issues with large datasets**
   - Process files in smaller batches
   - Consider the target sample rate (lower = less memory)

### Getting Help

Check the logs with `--log_level DEBUG` for detailed processing information.