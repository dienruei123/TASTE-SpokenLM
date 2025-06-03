import os
import glob
import yaml
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from pprint import pp
from taslm.modeling_taslm import TaslmForCausalLM
from taslm.configuration_taslm import TaslmConfig
from taslm.utils_taslm import get_lora_config, load_training_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_dir', type=str, default=None)
    parser.add_argument('--speech_tokenizer_pretrained_dir', type=str, default=None)
    parser.add_argument('--speech_decoder_pretrained_dir', type=str, default=None)
    # parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args


def main(args):
    pretrained_dir = args.pretrained_dir
    speech_tokenizer_pretrained_dir = args.speech_tokenizer_pretrained_dir
    speech_decoder_pretrained_dir = args.speech_decoder_pretrained_dir
    # 
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
    state_dict_fpath = os.path.join(args.pretrained_dir, 'checkpoint-best/model.safetensors')
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
    # speech_fpath = "/proj/mtklmadm/data/speech/LibriTTS/LibriTTS/test-clean/7127/75946/7127_75946_000033_000000.wav"
    speech_fpath = "/proj/mtklmadm/data/speech/LibriTTS/LibriTTS/test-clean/2300/131720/2300_131720_000013_000007.wav"
    log_likelihood = taslm_model.calculate_log_likelihood(speech_fpath, device=device)
    pp(log_likelihood)


if __name__ == "__main__":
    args = parse_args()
    main(args)