
import glob
import json
from pathlib import Path
import argparse

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from datasets import Dataset
from transformers import WhisperProcessor, AutoTokenizer
from taste_speech import TasteConfig, TasteForCausalLM, TasteProcessor


def pad_seq_collate_fn(batch, device=None):
    padded = {}
    for key in batch[0].keys():
        packed_list = [
            x[key][0].clone().detach() if isinstance(x[key][0], torch.Tensor) else torch.tensor(x[key][0]) 
            for x in batch
        ]
        if 'length' in key:
            padded_tensor = torch.tensor(packed_list)
        else:
            padded_tensor = pad_sequence(packed_list, batch_first=True, padding_value=0)

        padded[key] = padded_tensor.to(device) if device is not None else padded_tensor
    return padded


def generate(model_id, out_dir, 
        model_mode='SpokenLLM', conditional_compl=False, conditional_text_compl=False,
        extra_words=16, text_top_p=0.0, taste_top_p=0.0, text_temperature=1.0, repetition_penalty=1.0):

    flow_matching_use_ref = True

    sampling_rate = 16000
    
    task = model_mode
    if model_mode == 'SpokenLLM':
        if conditional_compl:
            assert conditional_text_compl is False
            task += '_Conditional_Compl'
        elif conditional_text_compl:
            assert conditional_compl is False
            task += '_Conditional_Text_Compl'

    audio_paths = None
    forced_texts = None
    out_generated_part_only_list = None
    if task == 'SpeechAutoEncoder' or task == 'SpokenLLM':
        audio_names = ['ex01_happy_00209.wav', 'ex04_sad_00311.wav', 'ex04_00350_sad_happy.wav']
        audio_paths = [f'examples/orig/{name}' for name in audio_names]

    elif task == 'SpokenLLM_Conditional_Compl':
        audio_paths_1 = [
            'examples/orig/conditional/control/voice_control_pace_07_slow.mp3',
            'examples/orig/conditional/control/voice_control_pace_08_fast.mp3',
        ]
        forced_texts_1 = [
            "[Q] Repeat the following sentence at a fast pace: Hey, how's everything going with you lately? Hope you're doing well and staying healthy. \n[A] Hey, how's everything going with you lately? Hope you're doing well and staying healthy. \n[Q] Please read the following sentence at a slow pace: Wishing you a wonderful weekend ahead. \n[A] Wishing you a wonderful weekend ahead. \n[Q] Please read the following sentence at a slow pace: It feels like forever since we last caught up. How have you been these days? \n[A] ",
            "[Q] Repeat the following sentence at a fast pace: Hey, how's everything going with you lately? Hope you're doing well and staying healthy. \n[A] Hey, how's everything going with you lately? Hope you're doing well and staying healthy. \n[Q] Please read the following sentence at a slow pace: Wishing you a wonderful weekend ahead. \n[A] Wishing you a wonderful weekend ahead. \n[Q] Please read the following sentence at a fast pace: It feels like forever since we last caught up. How have you been these days? \n[A] "
        ]
        out_generated_part_only_list_1 = [True for _ in range(len(audio_paths_1))]
        audio_paths_2 = [path for path in glob.glob('examples/orig/conditional/*wav')]
        forced_texts_2 = [None for _ in range(len(audio_paths_2))]
        out_generated_part_only_list_2 = [False for _ in range(len(audio_paths_2))]

        audio_paths = audio_paths_1 + audio_paths_2
        forced_texts = forced_texts_1 + forced_texts_2
        audio_names = [path.split('/')[-1].split('.')[0] for path in audio_paths]
        out_generated_part_only_list = out_generated_part_only_list_1 + out_generated_part_only_list_2

    elif task == 'SpokenLLM_Conditional_Text_Compl':
        texts = [
            'This is no longer', 
            'Tom: Hello\nAmy: Hello! How can I assist you today?\nTom: Tell me your name\nAmy: I\'m an AI assistant created by MR.\nTom: What can you do?\nAmy:',
            'Hurry up. Hurry up. Hurry up.',
            'I never thought I\'d find myself in this place, feeling so lost and empty.',
            'I am so happy. This is my best day in my life',
        ]

    # model
    device = 0

    if model_mode == 'SpokenLLM':
        model = TasteForCausalLM.from_pretrained(model_id)
    else:
        model = TasteForCausalLM.from_pretrained_stage1(model_id)

    model = model.to(device)
    model.eval()

    processor = TasteProcessor.from_pretrained(model_id)
    generator = processor.get_generator(model_id, device=device)

    if audio_paths:
        data = [
            processor(
                audio_path, sampling_rate,
                ref_audio_list=[audio_path],
                output_text_info=True
            )
            for audio_path in audio_paths
        ]
        if forced_texts:
            for i, forced_text in enumerate(forced_texts):
                if forced_text:
                    data[i].update(
                        {
                            'text': [forced_text],
                            **processor.process_text(text=forced_text)[1]
                        }
                    )
        out_names = [path.split('/')[-1].split('.')[0] for path in audio_paths]

    else:
        data = [
            {
                'text': [text],
                **processor.process_text(text=text)[1]
            }
            for text in texts
        ]
        out_names = [f'case_{i}' for i, text in enumerate(texts)]
    dataset = Dataset.from_list(data)

    if flow_matching_use_ref or task == 'SpokenLLM_Conditional_Text_Compl':
        ref_paths = glob.glob('examples/orig/hifi-tts-dev-clean-speaker6097/*wav')
        ref_speaker_embeds = torch.tensor(
            processor(ref_paths[0], sampling_rate, ref_audio_list=ref_paths)['speaker_embeds']
        ).to(device)

    output_folder = f'{out_dir}/{task}/'
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if task == 'SpokenLLM_Conditional_Text_Compl':
        cols = [
            'asr_token_ids',
            'asr_token_lengths',
            'asr_word_ids',
            'llm_token_ids',
            'llm_token_lengths',
            'llm_word_ids',
        ]
    else:
        cols = [
            'speaker_embeds', 
            'audio_features',
            'audio_feature_lengths',
            'asr_token_ids',
            'asr_token_lengths',
            'asr_word_ids'
        ]
        if model_mode == 'SpokenLLM':
            cols += [
                'llm_token_ids',
                'llm_token_lengths',
                'llm_word_ids',
            ]

    for i, (name, batch) in enumerate(zip(out_names, dataset)):
        orig_text = batch['text'][0]
        inputs = pad_seq_collate_fn([{k: batch[k] for k in cols}], device=device)
        if flow_matching_use_ref or task == 'SpokenLLM_Conditional_Text_Compl':
            inputs['speaker_embeds'] = ref_speaker_embeds

        generate_kwargs = dict(
            llm_tokenizer=processor.llm_tokenizer,
            asr_tokenizer=processor.audio_tokenizer,
            extra_words=extra_words,
            text_top_p=text_top_p,
            taste_top_p=taste_top_p,
            text_temperature=text_temperature,
            repetition_penalty=repetition_penalty,
        )
        if task == 'SpeechAutoEncoder' or task == 'SpokenLLM':
            output = model.inference_reconstruction(**inputs)
        elif task == 'SpokenLLM_Conditional_Compl':
            if out_generated_part_only_list:
                generate_kwargs['out_generated_part_only'] = out_generated_part_only_list[i]
            output = model.inference_completion(
                **inputs,
                conditional_mode='audio',
                **generate_kwargs,
            )
        elif task == 'SpokenLLM_Conditional_Text_Compl':
            output = model.inference_completion(
                **inputs,
                conditional_mode='text',
                **generate_kwargs,
            )

        tts_speech, tts_sr = generator.inference(
            speech_token_ids=output['speech_token_ids'], 
            speech_token_lengths=output['speech_token_lengths'],
            flow_embedding=inputs['speaker_embeds']
        )

        output_fpath = f'{output_folder}/{name}.wav'
        torchaudio.save(output_fpath, tts_speech, tts_sr)

        output_trans_fpath = f'{output_folder}/{name}.json'
        meta_json = {
            'orig_text': orig_text,
        }
        if 'generated_text' in output:
            meta_json['generated_text'] = output['generated_text']
        if task.endswith('_Compl'):
            meta_json.update(
                {
                    'extra_words': extra_words,
                    'text_top_p': text_top_p,
                    'taste_top_p': taste_top_p,
                    'text_temperature': text_temperature,
                    'repetition_penalty': repetition_penalty,
                }
            )
        json.dump(meta_json,
            open(output_trans_fpath, 'w'),
            ensure_ascii=False, indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./Llama-1B-TASTE-Speech-V0', type=str)
    parser.add_argument('--model_mode', default='stage2', type=str)
    parser.add_argument('--out_dir', default='./examples/generated_cases', type=str)

    parser.add_argument('--text_top_p', default=0.5, type=float)
    parser.add_argument('--text_temperature', default=1.0, type=float)
    parser.add_argument('--repetition_penalty', default=1.2, type=float)
    parser.add_argument('--conditional_compl', action='store_true')
    parser.add_argument('--conditional_text_compl', action='store_true')
    parser.add_argument('--extra_words', default=16, type=int)

    args = parser.parse_args()

    generate(
        args.model,
        model_mode= 'SpokenLLM' if args.model_mode == 'stage2' else 'SpeechAutoEncoder',
        out_dir=args.out_dir,
        conditional_compl=args.conditional_compl,
        conditional_text_compl=args.conditional_text_compl,
        extra_words=args.extra_words,
        text_top_p=args.text_top_p,
        taste_top_p=0.0,
        text_temperature=args.text_temperature,
        repetition_penalty=args.repetition_penalty,
    )
