# TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling

[[Demo](https://mtkresearch.github.io/LLaMA-TASTE-Speech.github.io/)] [[Paper]()] [[Model](https://huggingface.co/MediaTek-Research/Llama-1B-TASTE-Speech-V0)]


## Quick Start

Install the `taste_speech` package
```
git clone https://github.com/mtkresearch/TASTE-SpokenLM.git
cd TASTE-SpokenLM
pip install .
```

Install some dependencies,
```
pip install -q torch transformers 
pip install -q einx==0.3.0 HyperPyYAML==1.2.2 openai-whisper==20231117 onnxruntime-gpu==1.16.0 conformer==0.3.2 lightning==2.2.4 
```

### Inference Completion

```python
device = 0
model_id = 'MediaTek-Research/Llama-1B-TASTE-Speech-V0'
attn_implementation = 'eager'

model = TasteForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation)

model = model.to(device)
model.eval()

processor = TasteProcessor.from_pretrained(model_id)
generator = processor.get_generator(model_id, device=device)

generate_kwargs = dict(
    llm_tokenizer=processor.llm_tokenizer,
    asr_tokenizer=processor.audio_tokenizer,
    extra_words=8,
    text_top_p=0.3,
    taste_top_p=0.0,
    text_temperature=0.5,
    repetition_penalty=1.1,
)

conditional_audio_paths = ['/path/to/audio']
output_audio_paths = ['/path/to/output_audio']
sampling_rate = 16000

data = [
    processor(
        audio_path,
        sampling_rate,
        ref_audio_list=[audio_path]
    )
    for audio_path in conditional_audio_paths
]
dataset = Dataset.from_list(data)

for inputs, output_fpath in zip(data, output_audio_paths):
    inputs = {k: inputs[k].to(device) for k in cols}
    output = model.inference_completion(
        **inputs,
        conditional_mode='audio',
    )
    tts_speech, tts_sr = generator.inference(
        speech_token_ids=output['speech_token_ids'], 
        speech_token_lengths=output['speech_token_lengths'],
        flow_embedding=inputs['speaker_embeds']
    )
    torchaudio.save(output_fpath, tts_speech, tts_sr)
```

### Run Inference

```
python scripts/generate_audio.py --conditional_compl
```
