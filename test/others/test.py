from datasets import Dataset
import torchaudio

from taste_speech import TasteConfig, TasteForCausalLM, TasteProcessor

device = 0
model_id = 'MediaTek-Research/Llama-1B-TASTE-V0'
attn_implementation = 'eager'

model = TasteForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation)

model = model.to(device)
model.eval()

processor = TasteProcessor.from_pretrained(model_id)
generator = processor.get_generator(device=device)

generate_kwargs = dict(
    llm_tokenizer=processor.llm_tokenizer,
    asr_tokenizer=processor.audio_tokenizer,
    extra_words=8,
    text_top_p=0.3,
    taste_top_p=0.0,
    text_temperature=0.5,
    repetition_penalty=1.1,
)

conditional_audio_paths = ['examples/orig/ex01_happy_00209.wav']
output_audio_paths = ['test/latency_test/generated/ex01_happy_00209.wav']
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
    inputs = {k: inputs[k].to(device) for k in inputs.keys()}
    output = model.inference_completion(
        **inputs,
        conditional_mode='audio',
        **generate_kwargs,
    )
    tts_speech, tts_sr = generator.inference(
        speech_token_ids=output['speech_token_ids'], 
        speech_token_lengths=output['speech_token_lengths'],
        flow_embedding=inputs['speaker_embeds']
    )
    torchaudio.save(output_fpath, tts_speech, tts_sr)