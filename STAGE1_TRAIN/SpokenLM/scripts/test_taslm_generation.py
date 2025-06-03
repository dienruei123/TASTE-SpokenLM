import os
import glob
import yaml
import torch
import torchaudio
import argparse
import torch.nn as nn
import torch.nn.functional as F
from cosyvoice.utils.file_utils import load_wav
from pprint import pp
from taslm.modeling_taslm import TaslmForCausalLM
from taslm.configuration_taslm import TaslmConfig
from taslm.utils_taslm import get_lora_config, load_training_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_dir', type=str, default=None)
    parser.add_argument('--ckpt_name', type=str, default="checkpoint-best")
    parser.add_argument('--speech_tokenizer_pretrained_dir', type=str, default=None)
    parser.add_argument('--speech_decoder_pretrained_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--tts_text_fpath', type=str, default=None)
    parser.add_argument('--conditional_generation_fpath', type=str, default=None)
    parser.add_argument('--no_latent_sampling', action='store_true')
    parser.add_argument('--text_topp', type=float, default=0.9)
    # parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args


def main(args):
    pretrained_dir = args.pretrained_dir
    speech_tokenizer_pretrained_dir = args.speech_tokenizer_pretrained_dir
    speech_decoder_pretrained_dir = args.speech_decoder_pretrained_dir
    output_dir = args.output_dir
    num_samples = args.num_samples
    # settings
    torch_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
    device = f"cuda" if torch.cuda.is_available() else 'cpu'
    # prepare model
    ## load config
    taslm_config = TaslmConfig.from_pretrained(pretrained_dir)
    ## load model from config
    taslm_model = TaslmForCausalLM._from_config(taslm_config)
    ## load training config -> get lora config -> apply lora
    training_config = load_training_config(pretrained_dir)
    _lora_config = get_lora_config(taslm_model.language_model, training_config)
    taslm_model.apply_lora(_lora_config, training_config)
    ## load from state dict
    state_dict_fpath = os.path.join(args.pretrained_dir, f'{args.ckpt_name}/model.safetensors')
    from safetensors.torch import load_file
    _state_dict = load_file(state_dict_fpath)
    taslm_model.load_state_dict(_state_dict)
    ## register speech tokenizer and asr model
    taslm_model = taslm_model.to(device)
    taslm_model.register_speech_tokenizer_decoder_and_asr(
        pretrained_dir,
        training_config,
        speech_tokenizer_pretrained_dir,
        speech_decoder_pretrained_dir,
        # asr_pretrained_dir (for asr pipeline and tokenizer)
        # llm_pretrained_dir (for llm tokenizer)
        device=device,
    )
    # prepare inputs
    # text input
    _speech_bos_idx = taslm_model.collate_fn_kwargs.get("speech_bos_idx", taslm_model.config.speech_bos_token_id)
    paired_input_ids_list = []
    cond_cutoff_speech_pt_list = []
    task_is_tts = False
    task_is_cond_gen = False
    if 's3' in taslm_model.speech_token_type:
        gen_kwargs = {
            'max_length': 500
        }
    else:
        gen_kwargs = {}
    if args.tts_text_fpath is not None:
        print(f"tts_text_fpath is assigned. will conduct tts using the given text.")
        with open(args.tts_text_fpath, 'r') as fr:
            for l in fr:
                l = f" {l.strip()}"
                tts_text_input_ids = taslm_model.prepare_tts_text_input_ids(l, prefix_token_to_wrap=[128000], suffix_token_to_wrap=[128002, 128001]).to(device)
                speech_input_ids = torch.tensor([_speech_bos_idx] * taslm_model.speech_num_channels, device=device, dtype=torch.long)
                speech_input_ids = speech_input_ids.view(1, 1, -1)
                paired_input_ids_list.append((tts_text_input_ids, speech_input_ids, f'tts_example-{i}', None))
        task_is_tts = True
    elif args.conditional_generation_fpath is not None:
        task_is_cond_gen = True
        print(f"conditional_generation_fpath is assigned. will use the audio fpaths in the list for conditional generation.")
        if 's3' in taslm_model.speech_token_type:
            drop_special_idx = -1 # eos
        else:
            drop_special_idx = -2 # special#1 and eos
        cutoff_word_idx = 5
        with open(args.conditional_generation_fpath, 'r') as fr:
            for l in fr:
                l = l.strip()
                audio_fpath = l.split('\t')[0]
                audio_name = audio_fpath.split('/')[-1].split('.')[0]
                speech_pt_16k = load_wav(audio_fpath, 16_000)
                # get pre asr result
                pre_asr_result = taslm_model._pre_asr([speech_pt_16k.squeeze(0).numpy()], cutoff_word_idx=cutoff_word_idx)
                # get the cutoff speech for conditional generation
                speech_pt_16k_for_cond = torch.from_numpy(pre_asr_result['cutoff_speech_npy_16k']).to(device).unsqueeze(0)
                spk_emb_for_cond = taslm_model._extract_spk_emb(speech_pt_16k_for_cond).to(device)
                # prepare batched input for conditional generation, including the spk embeddings (based on the cutoff speech)
                batched_input = taslm_model._prepare_input(speech_pt_16k_for_cond, pre_asr_result=pre_asr_result, device=device)
                print(batched_input['speech_labels'])
                if 's3' in taslm_model.speech_token_type:
                    speech_input_ids = batched_input['speech_input_ids'][:, :drop_special_idx]
                    text_input_ids_with_padding = batched_input['text_input_ids']
                    valid_text_input_ids_len = batched_input['text_input_ids_lens'][0].item() # currently only support single element in a batch
                    text_input_ids = text_input_ids_with_padding[:, :valid_text_input_ids_len+drop_special_idx]
                else:
                    speech_input_ids = batched_input['speech_input_ids'][:, :drop_special_idx, :]
                    text_input_ids = batched_input['text_input_ids'][:, :drop_special_idx]
                paired_input_ids_list.append((text_input_ids, speech_input_ids, f"cond-gen_{audio_name}", spk_emb_for_cond))
                cond_cutoff_speech_pt_list.append(speech_pt_16k_for_cond)
    else:
        text_input_ids = torch.tensor([taslm_model.config.llama_config.bos_token_id], device=device, dtype=torch.long)
        text_input_ids = text_input_ids.view(1, -1)
        if 's3' in taslm_model.speech_token_type:
            speech_input_ids = torch.tensor([_speech_bos_idx], device=device, dtype=torch.long)
            speech_input_ids = speech_input_ids.view(1, 1)
        else:
            speech_input_ids = torch.tensor([_speech_bos_idx] * taslm_model.speech_num_channels, device=device, dtype=torch.long)
            speech_input_ids = speech_input_ids.view(1, 1, -1)
        paired_input_ids_list.append((text_input_ids, speech_input_ids), f"uncond-gen_example-{i}", None)


    if output_dir is not None:
        if task_is_tts:
            if not "tts" in output_dir:
                output_dir = os.path.join(output_dir, "tts") 
        elif task_is_cond_gen:
            if not 'cond-gen' in output_dir:
                output_dir = os.path.join(output_dir, "cond-gen") 
        elif not 'uncond-gen' in output_dir:
            output_dir = os.path.join(output_dir, "uncond-gen")
        os.makedirs(output_dir, exist_ok=True)
    if args.no_latent_sampling:
        print("will disable latent sampling and directly use the predicted mean.")
        taslm_model.speech_latent_sampler.conduct_reparameterization = False
    
    with torch.amp.autocast(device_type=device, dtype=torch_dtype):
        taslm_model.eval()
        for gen_id, (_text_input_ids, _speech_input_ids, audio_fname, spk_emb) in enumerate(paired_input_ids_list):
            print(_text_input_ids, _speech_input_ids)
            generation_result = taslm_model.generate(
                _text_input_ids,
                _speech_input_ids,
                text_top_p=args.text_topp,
                speech_top_p=0.9,
                is_tts=task_is_tts,
                **gen_kwargs,
            )
        
            generation_output_fpath = os.path.join(output_dir, f"{audio_fname}.txt")
            with open(generation_output_fpath, 'w') as fw:
                for _text_id, _speech_ids in zip(generation_result['generated_text_ids'][0], generation_result['generated_speech_ids'][0]):
                    _text = taslm_model.llm_tokenizer.decode(_text_id)
                    # print(_text_id, _text, _speech_ids, sep='\t')
                    print(_text_id, _text, _speech_ids, sep='\t', file=fw)
            # assert False, "stop for debug"

            pp(generation_result)
            speech_result = taslm_model.generate_speech(generation_result, spk_emb=spk_emb)
            speech_output_fpath = os.path.join(output_dir, f"{audio_fname}.wav")
            if task_is_cond_gen:
                cond_speech_output_fpath = os.path.join(output_dir, f"{audio_fname}_cond.wav")
                torchaudio.save(cond_speech_output_fpath, cond_cutoff_speech_pt_list[gen_id].cpu(), 16000)
            torchaudio.save(speech_output_fpath, speech_result, 22050)


if __name__ == "__main__":
    args = parse_args()
    main(args)