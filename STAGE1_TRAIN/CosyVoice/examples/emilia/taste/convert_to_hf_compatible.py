# under TASTE-SpokenLM
import os
import yaml
import torch
from pprint import pp
from hyperpyyaml import load_hyperpyyaml, resolve_references

# convert the original hyperpyyaml config to normal reference by substituting the !new tag into nothing
exp_dir = "./exp/llm/torch_ddp/taste_pretrained" # TODO: change for your own need
ckpt_name = "checkpoint_best" # TODO: change for your own need
# start converting config
ckpt_fpath = os.path.join(exp_dir, f"{ckpt_name}.pt")
config_fpath = os.path.join(exp_dir, "config.yaml")
tmp_config_fpath = os.path.join(exp_dir, "tmp_config.yaml") # config for TASTE-SpokenLM
parsed_config_fpath = os.path.join(exp_dir, "parsed_config.yaml") # config for TASTE-SpokenLM
if not os.path.exists(parsed_config_fpath):
    with open(config_fpath, 'r') as fr, open(tmp_config_fpath, 'w') as fw:
        for l in fr:
            if "!new" in l:
                # keep only the keys before !new
                fw.write(l.split('!new')[0] + "\n")
            elif "!name" in l:
                # keep only the keys before !new
                fw.write(l.split('!name')[0] + "\n")
            elif "!apply" in l:
                continue
            else:
                fw.write(l)

    with open(tmp_config_fpath, 'r') as fr, open(parsed_config_fpath, 'w') as fw:
        resolved_config = resolve_references(fr).getvalue()
        fw.write(resolved_config)

# load the converted config
with open(parsed_config_fpath, 'r') as fr:
    parsed_config = yaml.safe_load(fr)
    # print(parsed_config)

from taste_speech import TasteConfig, TasteForCausalLM, TasteAudioTowerConfig, TasteSpeechDecoderConfig

taste_config = TasteConfig()
## overwrite the tokenizer config properly
audio_tower_config = TasteAudioTowerConfig()
audio_tower_config.kwargs_for_joint_encoder_segmenter = parsed_config['llm']['audio_tokenizer']['audio_joint_encoder_segmenter']
audio_tower_config.kwargs_for_quantizer = parsed_config['llm']['audio_tokenizer']['audio_quantizer']
taste_config.audio_tower_config = audio_tower_config
## overwrite the speech decoder config properly
speech_decoder_config = TasteSpeechDecoderConfig()
for key, val in parsed_config['llm'].items():
    if key == "text_encoder":
        for _sub_key, _sub_val in val.items():
            _new_sub_key = f"encoder__{_sub_key}"
            if hasattr(speech_decoder_config, _new_sub_key):
                setattr(speech_decoder_config, _new_sub_key, _sub_val)
            else:
                if _sub_key != "input_size" and _sub_key != "output_size":
                    print(_new_sub_key + " is not in speech_decoder_config")
    elif key == "llm":
        for _sub_key, _sub_val in val.items():
            _new_sub_key = f"llm__{_sub_key}"
            if hasattr(speech_decoder_config, _new_sub_key):
                setattr(speech_decoder_config, _new_sub_key, _sub_val)
            else:
                if _sub_key != "input_size" and _sub_key != "output_size":
                    print(_new_sub_key + " is not in speech_decoder_config")
    elif key == "audio_embed_dim":
        speech_decoder_config.audio_encoder_input_size = val
    elif key == "text_encoder_input_size":
        speech_decoder_config.encoder_input_size = val
    else:
        if hasattr(speech_decoder_config, key):
            setattr(speech_decoder_config, key, val)
taste_config.speech_decoder_config = speech_decoder_config
del taste_config.spoken_lm_config # avoid loading slm
taste_model_wo_slm = TasteForCausalLM(taste_config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
taste_model_wo_slm.eval()
taste_model_wo_slm.to(device)
# load the pretrained ckpt from cosyvoice
## load the speech tokenizer
taste_model_wo_slm.audio_tower.load_from_cosyvoice_ckpt(ckpt_fpath)
## load the speech decoder
taste_model_wo_slm.speech_decoder.load_from_cosyvoice_ckpt(ckpt_fpath)
pp(taste_model_wo_slm)
# save the model for future usage
hf_pretrained_dir = os.path.join(exp_dir, ckpt_name)
taste_model_wo_slm.save_pretrained(hf_pretrained_dir)
# now you can load the hf model directly using hf
del taste_model_wo_slm
del taste_config
# test loading using `from_pretrained`
taste_config = TasteConfig.from_pretrained(hf_pretrained_dir)
del taste_config.spoken_lm_config # delete slm config part to avoid loading it
taste_model_wo_slm = TasteForCausalLM(taste_config)
pp(taste_model_wo_slm)